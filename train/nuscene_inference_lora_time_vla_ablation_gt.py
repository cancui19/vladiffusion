"""
Ablation study: Compares VLA inference results under different GT conditioning settings.

All modes use the diffusion model's in-filling capability (generate_fix_mask) to avoid
introducing out-of-distribution prompt formats that the model never saw during training.

Modes:
- baseline: No GT conditioning. Standard generation of [Action Tokens] + [Explanation].
- gt_plan_hint: Fix Action Tokens (from GT), let model generate Explanation.
  Template: [GT Action Tokens (10)] + [MASK for Explanation]
- gt_expl_hint: Fix Explanation (from GT), let model generate Action Tokens.
  Template: [MASK (10)] + [GT Explanation Tokens]

Output format is compatible with existing evaluation scripts: [pred_points_str, pred_expl_text, gt_text]
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


def build_question(base_question: str, gt_points: Optional[List[List[float]]], gt_text: str, mode: str) -> str:
    """
    Return the base question as-is. 
    All GT conditioning is handled via in-filling (generate_fix_mask), not prompt modification.
    This ensures the prompt format matches training distribution.
    """
    # Do NOT add any hint text to the prompt - model has never seen such format during training
    return base_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "gt_plan_hint", "gt_expl_hint"],
        help="baseline: no hint; gt_plan_hint: fix GT trajectory, generate explanation; gt_expl_hint: fix GT explanation, generate trajectory",
    )
    parser.add_argument("--output", type=str, default=None, help="Result file path, defaults to config.results_path with suffix")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional, limit number of inference samples for quick testing")
    args = parser.parse_args()

    print(f"Mode: {args.mode}")

    warnings.filterwarnings("ignore")

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

    # Pre-load ids_tensor once (avoid repeated disk I/O in loop)
    ids_tensor = torch.from_numpy(np.load(config.unused_token_ids_path)).to(config.device)

    model.eval()

    with open(config.data_path, "r") as f:
        data_val = json.load(f)

    output_path = Path(args.output) if args.output else Path(config.results_path).with_stem(
        Path(config.results_path).stem + f"_{args.mode}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    conv_template = config.conv_template
    inference_results = []

    for i, data_sample in enumerate(data_val):
        if args.max_samples is not None and i >= args.max_samples:
            break

        image = Image.open(data_sample["image"])
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=config.device) for _image in image_tensor]

        base_question = data_sample["conversations"][0]["value"]
        gt_text = data_sample["conversations"][1]["value"]
        gt_points = data_sample.get("action_targets", [])

        question = build_question(base_question, gt_points, gt_text, args.mode)
        print("\n==== Sample", i, "====")
        print(question)

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

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
            print("Testing with cache enabled")
        else:
            print("Testing without cache")

        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(config.device)

        # Determine the mask_id used by LLaDA
        mask_id = 126336  # Default LLaDA mask token ID
        
        start_time = time.time()
        
        if args.mode == "baseline":
            # Baseline: standard generation, no GT conditioning
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                steps=config.generation_steps,
                gen_length=config.generation_length,
                block_length=config.generation_block_length,
                tokenizer=tokenizer,
                stopping_criteria=list(config.stopping_criteria),
            )
            
            generated_ids = cont[0]
            # Parse output: first 10 tokens -> action tokens, subsequent -> explanation
            points_ids = generated_ids[:10]
            explanation_ids = generated_ids[10:]
            explanation_text = tokenizer.decode(explanation_ids, skip_special_tokens=True)
            
        elif args.mode == "gt_plan_hint":
            # Fix Action Tokens (from GT), let model generate Explanation
            # Template: [GT Action Tokens (10)] + [MASK for Explanation]
            
            # Convert GT points to Action Token IDs
            gt_points_tensor = torch.tensor(gt_points, dtype=torch.float32, device=config.device)
            gt_action_indices = point_tokenizer.points_to_indices(gt_points_tensor)
            
            # Map indices back to actual token IDs (ids_tensor pre-loaded outside loop)
            gt_action_token_ids = ids_tensor[gt_action_indices]
            
            # Estimate explanation length (use a reasonable max)
            num_explanation_masks = config.generation_length - 10
            mask_tokens = torch.full((num_explanation_masks,), mask_id, dtype=torch.long, device=config.device)
            
            # Construct template: [GT Action Tokens (10)] + [MASK for Explanation]
            template_ids = torch.cat([gt_action_token_ids, mask_tokens]).unsqueeze(0)
            
            print(f"Template length: {template_ids.shape[1]} (10 GT action tokens + {num_explanation_masks} masks)")
            
            # Use generate_fix_mask for in-filling
            cont = model.generate_fix_mask(
                template_ids=template_ids,
                inputs=input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                steps=config.generation_steps,
                tokenizer=tokenizer,
            )
            
            # cont is the filled template: [GT Action Tokens (10)] + [Generated Explanation]
            generated_ids = cont[0]
            points_ids = generated_ids[:10]  # These are the GT action tokens (unchanged)
            explanation_ids = generated_ids[10:]
            explanation_text = tokenizer.decode(explanation_ids, skip_special_tokens=True)
            
        elif args.mode == "gt_expl_hint":
            # Fix Explanation (from GT), let model generate Action Tokens
            # Template: [MASK (10)] + [GT Explanation Tokens]
            
            # Extract and tokenize GT explanation
            segments = _extract_segments(gt_text)
            hint_segments = []
            if segments.get("narration"):
                hint_segments.append(f"Narration: {segments['narration']}")
            if segments.get("reasoning"):
                hint_segments.append(f"Reasoning: {segments['reasoning']}")
            
            if hint_segments:
                explanation_suffix = "\n" + "\n".join(hint_segments)
            else:
                explanation_suffix = "\n" + gt_text
            
            # Tokenize the explanation suffix
            explanation_token_ids = tokenizer.encode(explanation_suffix, add_special_tokens=False)
            explanation_token_ids = torch.tensor(explanation_token_ids, dtype=torch.long, device=config.device)
            
            # Construct template: [MASK]*10 (for Action Tokens) + [Explanation Token IDs]
            num_action_tokens = 10
            mask_tokens = torch.full((num_action_tokens,), mask_id, dtype=torch.long, device=config.device)
            template_ids = torch.cat([mask_tokens, explanation_token_ids]).unsqueeze(0)
            
            print(f"Template length: {template_ids.shape[1]} (10 masks + {len(explanation_token_ids)} GT explanation tokens)")
            
            # Use generate_fix_mask for in-filling
            cont = model.generate_fix_mask(
                template_ids=template_ids,
                inputs=input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                steps=config.generation_steps,
                tokenizer=tokenizer,
            )
            
            # cont is the filled template: [Generated Action Tokens (10)] + [GT Explanation (unchanged)]
            generated_ids = cont[0]
            points_ids = generated_ids[:num_action_tokens]
            # For explanation, we use the GT directly since it was fixed in the template
            explanation_text = explanation_suffix.strip()

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.4f} seconds")
        total_time += generation_time

        # Recover points from Action Token IDs (ids_tensor pre-loaded outside loop)
        pos_in_sorted = torch.searchsorted(-ids_tensor, -points_ids)
        recovered_points = point_tokenizer.indices_to_points(pos_in_sorted)

        print("Points:", recovered_points)
        print("Explanation:", explanation_text)

        inference_results.append(
            [
                str(recovered_points.tolist())[1:-1],
                explanation_text,
                gt_text,
            ]
        )

    with open(output_path, "w") as f:
        json.dump(inference_results, f)
    print(f"Saved inference results to {output_path}")
    if inference_results:
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Average time: {total_time / len(inference_results):.4f} seconds")


if __name__ == "__main__":
    main()

