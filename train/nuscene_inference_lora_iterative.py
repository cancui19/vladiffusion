"""
Iterative Inference: Planning-Explanation Refinement Loop

This script implements an iterative refinement process:
- Iteration 1:
  - Step 1 (planning baseline): call `model.generate(...)` once to produce a continuation sequence,
    then take the first 10 generated tokens as action tokens (planning). Any following tokens (if generated)
    are ignored in this step.
  - Step 2 (explanation): fix the 10 action tokens and use `generate_fix_mask(...)` to in-fill the
    remaining positions as the explanation.
- Iteration 2-5: Use previous explanation to regenerate planning, then regenerate explanation

Each iteration's results are saved separately for analysis.

Output structure:
  results/iterative_refinement/
    ├── iter_1.json
    ├── iter_2.json
    ├── iter_3.json
    ├── iter_4.json
    └── iter_5.json
"""

import argparse
import copy
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from PIL import Image

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX

from train.config.nuscene_inference_vla import config_explanation as config
from tokenizer.example_usage import load_point_tokenizer


def _extract_segments(text: str) -> Dict[str, Optional[str]]:
    """
    Simple parsing of Narration/Reasoning/Description segments, returning a dictionary.
    Returns None if missing.
    """
    import re

    if not isinstance(text, str):
        return {}
    label_pattern = re.compile(r"(Action|Narration|Reasoning|Description)\s*[:：]+", re.IGNORECASE)
    matches = list(label_pattern.finditer(text))
    if not matches:
        return {}
    segments: Dict[str, Optional[str]] = {
        "action": None,
        "narration": None,
        "reasoning": None,
        "description": None,
    }
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        key = match.group(1).lower()
        value = text[start:end].strip()
        segments[key] = value
    return segments


def _format_points(points: List[List[float]]) -> str:
    if not points:
        return ""
    return "[" + "], [".join(f"{p[0]:.2f}, {p[1]:.2f}" for p in points) + "]"


def generate_planning_baseline(
    model, tokenizer, input_ids, image_tensor, image_size, 
    point_tokenizer, ids_tensor, mask_id
):
    """
    Baseline planning generation.
    Returns: (points_ids, recovered_points)
    """
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        steps=config.generation_steps,
        gen_length=config.generation_length,
        block_length=config.generation_block_length,
        tokenizer=tokenizer,
        stopping_criteria=list(config.stopping_criteria),
    )
    
    generated_ids = cont[0]
    # Take the first 10 generated tokens as planning.
    points_ids = generated_ids[:10]
    
    # Recover points from Action Token IDs
    pos_in_sorted = torch.searchsorted(-ids_tensor, -points_ids)
    recovered_points = point_tokenizer.indices_to_points(pos_in_sorted)
    
    return points_ids, recovered_points


def generate_explanation_from_planning(
    model, tokenizer, input_ids, image_tensor, image_size,
    points_ids, ids_tensor, mask_id
):
    """
    Generate explanation conditioned on given planning (action tokens).
    Uses generate_fix_mask to fix action tokens and generate explanation.
    Returns: explanation_text
    """
    # Estimate explanation length
    num_explanation_masks = config.generation_length - 10
    mask_tokens = torch.full((num_explanation_masks,), mask_id, dtype=torch.long, device=config.device)
    
    # Construct template: [Action Tokens (10)] + [MASK for Explanation]
    template_ids = torch.cat([points_ids, mask_tokens]).unsqueeze(0)
    
    # Use generate_fix_mask for in-filling
    cont = model.generate_fix_mask(
        template_ids=template_ids,
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        steps=config.generation_steps,
        tokenizer=tokenizer,
    )
    
    generated_ids = cont[0]
    explanation_ids = generated_ids[10:]
    explanation_text = tokenizer.decode(explanation_ids, skip_special_tokens=True)
    
    return explanation_text


def generate_planning_from_explanation(
    model, tokenizer, input_ids, image_tensor, image_size,
    explanation_text, point_tokenizer, ids_tensor, mask_id
):
    """
    Generate planning (action tokens) conditioned on given explanation.
    Uses generate_fix_mask to fix explanation and generate action tokens.
    Returns: (points_ids, recovered_points)
    """
    # Format explanation as suffix
    if explanation_text.startswith("\n"):
        explanation_suffix = explanation_text
    else:
        explanation_suffix = "\n" + explanation_text
    
    # Tokenize the explanation suffix
    explanation_token_ids = tokenizer.encode(explanation_suffix, add_special_tokens=False)
    explanation_token_ids = torch.tensor(explanation_token_ids, dtype=torch.long, device=config.device)
    
    # Construct template: [MASK]*10 (for Action Tokens) + [Explanation Token IDs]
    num_action_tokens = 10
    mask_tokens = torch.full((num_action_tokens,), mask_id, dtype=torch.long, device=config.device)
    template_ids = torch.cat([mask_tokens, explanation_token_ids]).unsqueeze(0)
    
    # Use generate_fix_mask for in-filling
    cont = model.generate_fix_mask(
        template_ids=template_ids,
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        steps=config.generation_steps,
        tokenizer=tokenizer,
    )
    
    generated_ids = cont[0]
    points_ids = generated_ids[:num_action_tokens]
    
    # Recover points from Action Token IDs
    pos_in_sorted = torch.searchsorted(-ids_tensor, -points_ids)
    recovered_points = point_tokenizer.indices_to_points(pos_in_sorted)
    
    return points_ids, recovered_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of refinement iterations")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional, limit number of inference samples for quick testing")
    args = parser.parse_args()

    print(f"Iterative Refinement: {args.num_iterations} iterations")

    warnings.filterwarnings("ignore")

    # Load model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        config.pretrained,
        None,
        config.model_name,
        attn_implementation="sdpa",
        device_map=config.device_map,
    )

    model = PeftModel.from_pretrained(model, config.lora_path, adapter_name="default")
    model = model.merge_and_unload()

    # --- Tokenizer Embedding Transfer ---
    weights_file = Path(config.tokenizer_weights)
    if not weights_file.exists():
        raise FileNotFoundError(f"Tokenizer weights not found at {weights_file}")

    tokenizer_state = torch.load(weights_file, map_location="cpu")
    point_embeddings = tokenizer_state["embedding.E"]

    ids_array = np.load(config.unused_token_ids_path)
    if isinstance(ids_array, np.ndarray):
        action_token_id_list = ids_array.tolist()
    else:
        action_token_id_list = list(ids_array)

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        point_embeddings_tensor = point_embeddings.to(
            input_embeddings.weight.device,
            dtype=input_embeddings.weight.dtype,
        )
        input_embeddings.weight[action_token_id_list] = point_embeddings_tensor
        if hasattr(model, "lm_head") and model.lm_head.weight.shape[0] >= len(action_token_id_list):
            model.lm_head.weight[action_token_id_list] = point_embeddings_tensor
        if hasattr(model, "config"):
            model.config.action_token_ids = action_token_id_list

    point_tokenizer = load_point_tokenizer(weights_file)

    # Pre-load ids_tensor once
    ids_tensor = torch.from_numpy(np.load(config.unused_token_ids_path)).to(config.device)

    model.eval()

    # Load data
    with open(config.data_path, "r") as f:
        data_val = json.load(f)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.results_path).parent / "iterative_refinement"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Determine the mask_id used by LLaDA
    mask_id = 126336
    conv_template = config.conv_template

    # Store intermediate states for each sample (persists across iterations)
    # sample_states[sample_idx] = {"explanation": str} - only need explanation for next iteration
    sample_states = {}

    total_time = 0.0
    num_samples = len(data_val) if args.max_samples is None else min(args.max_samples, len(data_val))

    print(f"\n{'='*60}")
    print(f"Starting iterative inference on {num_samples} samples")
    print(f"{'='*60}")

    # Outer loop: iterations
    for iter_idx in range(1, args.num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iter_idx}/{args.num_iterations}")
        print(f"{'='*60}")
        
        iter_results = []
        iter_start_time = time.time()

        # Inner loop: samples
        for sample_idx in range(num_samples):
            data_sample = data_val[sample_idx]
            
            image = Image.open(data_sample["image"])
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=config.device) for _image in image_tensor]

            base_question = data_sample["conversations"][0]["value"]
            gt_text = data_sample["conversations"][1]["value"]

            # Prepare input_ids
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], base_question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(config.device)

            # Setup cache if enabled
            if config.use_cache:
                dLLMCache.new_instance(
                    **asdict(
                        dLLMCacheConfig(
                            prompt_interval_steps=config.prompt_interval_steps,
                            gen_interval_steps=config.gen_interval_steps,
                            transfer_ratio=config.transfer_ratio,
                        )
                    )
                )
                register_cache_LLaDA_V(model, "model.layers")

            start_time = time.time()

            if iter_idx == 1:
                # Iteration 1: Baseline generation
                # Step 1: Generate a continuation, then take the first 10 tokens as planning (action tokens)
                current_points_ids, current_recovered_points = generate_planning_baseline(
                    model, tokenizer, input_ids, image_tensor, image.size,
                    point_tokenizer, ids_tensor, mask_id
                )
                
                # Step 2: Fix those 10 planning tokens and in-fill the explanation part via generate_fix_mask
                current_explanation = generate_explanation_from_planning(
                    model, tokenizer, input_ids, image_tensor, image.size,
                    current_points_ids, ids_tensor, mask_id
                )
            else:
                # Iteration 2+: Conditioned generation
                # Get previous explanation from sample_states
                prev_explanation = sample_states[sample_idx]["explanation"]
                
                # Step 1: Generate planning conditioned on previous explanation
                current_points_ids, current_recovered_points = generate_planning_from_explanation(
                    model, tokenizer, input_ids, image_tensor, image.size,
                    prev_explanation, point_tokenizer, ids_tensor, mask_id
                )
                
                # Step 2: Generate explanation based on new planning
                current_explanation = generate_explanation_from_planning(
                    model, tokenizer, input_ids, image_tensor, image.size,
                    current_points_ids, ids_tensor, mask_id
                )

            end_time = time.time()
            sample_time = end_time - start_time
            total_time += sample_time

            # Update sample state for next iteration
            sample_states[sample_idx] = {
                "explanation": current_explanation,
            }

            # Store result
            iter_results.append([
                str(current_recovered_points.tolist())[1:-1],
                current_explanation,
                gt_text,
            ])

            # Progress logging (every 10 samples or last sample)
            if (sample_idx + 1) % 10 == 0 or sample_idx == num_samples - 1:
                expl_display = current_explanation[:80] + "..." if current_explanation and len(current_explanation) > 80 else (current_explanation or "(empty)")
                print(f"  [{sample_idx + 1}/{num_samples}] Time: {sample_time:.2f}s | Points: {current_recovered_points.tolist()[:2]}... | Expl: {expl_display[:50]}...")

        iter_end_time = time.time()
        iter_total_time = iter_end_time - iter_start_time

        # Save results for this iteration immediately
        iter_output_path = output_dir / f"iter_{iter_idx}.json"
        with open(iter_output_path, "w") as f:
            json.dump(iter_results, f, indent=2)
        print(f"\n  ✓ Iteration {iter_idx} completed in {iter_total_time:.2f}s")
        print(f"  ✓ Saved to {iter_output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Total iterations: {args.num_iterations}")
    print(f"Total inference calls: {num_samples * args.num_iterations}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample per iteration: {total_time / (num_samples * args.num_iterations):.2f}s")
    print(f"Results saved to: {output_dir}")
    for i in range(1, args.num_iterations + 1):
        print(f"  - iter_{i}.json")


if __name__ == "__main__":
    main()
