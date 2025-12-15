from dataclasses import asdict
import copy
import json
import time
import warnings
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from peft import PeftModel

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX

from train.config.nuscene_inference_vla import config_explanation as config
from tokenizer.example_usage import load_point_tokenizer


print("start")

warnings.filterwarnings("ignore")

tokenizer, model, image_processor, max_length = load_pretrained_model(
    config.pretrained,
    None,
    config.model_name,
    attn_implementation="sdpa",
    device_map=config.device_map,
)

model = PeftModel.from_pretrained(model, config.lora_path, adapter_name="default")
model = model.merge_and_unload()

# --- Begin Tokenizer Embedding Transfer (from _from_curr.py) ---
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
id_to_index = {token: idx for idx, token in enumerate(action_token_id_list)}
# --- End Tokenizer Embedding Transfer ---

model.eval()

with open(config.data_path, 'r') as f:
    data_val = json.load(f)

total_time = 0

conv_template = config.conv_template 

inference_results = []

for i, data_sample in enumerate(data_val):
    image = Image.open(data_sample['image'])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=config.device) for _image in image_tensor]
    image_sizes = [image.size]
    
    # Use raw question from data, assuming prompt alignment isn't strictly needed or is already correct in val data
    # If you need prompt alignment like in _from_curr.py, uncomment the logic below
    question = data_sample['conversations'][0]['value']
    print(question)

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    model.eval()

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

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(config.device)
    image_sizes = [image.size]

    start_time = time.time()
    # Note: Removed flatten() logic inside generate if it was implicit, standard generate returns [batch, seq_len]
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        steps=config.generation_steps,
        gen_length=config.generation_length,
        block_length=config.generation_block_length,
        tokenizer=tokenizer,
        stopping_criteria=list(config.stopping_criteria),
    )
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")
    total_time += generation_time

    # Output parsing
    # The model output `cont` contains [action_tokens (10)] + [explanation tokens]
    # We need to slice carefully.
    
    # Decode full output first to debug/check
    full_text = tokenizer.batch_decode(cont, skip_special_tokens=False)[0]
    # print("Full decoded:", full_text)
    
    # Parse points from the first 10 tokens
    # Note: cont is [1, seq_len], so we take cont[0, :10]
    generated_ids = cont[0]
    
    # Assuming first 10 tokens are action tokens as per training task
    points_ids = generated_ids[:10]
    explanation_ids = generated_ids[10:]
    
    # Recover points
    ids_tensor = torch.from_numpy(np.load(config.unused_token_ids_path)).to(config.device)
    # searchsorted expects 1D input
    pos_in_sorted = torch.searchsorted(-ids_tensor, -points_ids)      
    recovered_points = point_tokenizer.indices_to_points(pos_in_sorted)
    
    # Decode explanation
    explanation_text = tokenizer.decode(explanation_ids, skip_special_tokens=True)
    
    print("Explanation:", explanation_text)
    print("Points:", recovered_points)

    inference_results.append([
        str(recovered_points.tolist())[1:-1], 
        explanation_text, 
        data_sample['conversations'][1]['value']
    ])

with open(config.results_path, 'w') as f:
    json.dump(inference_results, f)
print(f"Saved inference results for {i}th data sample")

print(f"Total time: {total_time:.4f} seconds")
print(f"Average time: {total_time/len(inference_results):.4f} seconds")
