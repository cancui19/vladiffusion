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
from tqdm import tqdm
from random import sample, Random

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX

from train.config.nuscene_inference_vla import config
from tokenizer.example_usage import load_point_tokenizer

LEGACY_WAYPOINT_PROMPT = (
    "Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., "
    "[x_10, y_10]. Write the raw text, not markdown or LaTeX. Future waypoints:"
)

ACTION_TOKEN_PROMPT = "Generate the predicted ten future waypoints in the action token format:"


def _to_action_token_prompt(question: str) -> str:
    """Align legacy text prompts with the action-token instruction used in training."""
    if ACTION_TOKEN_PROMPT in question:
        return question
    if LEGACY_WAYPOINT_PROMPT in question:
        print(1)
        return question.replace(LEGACY_WAYPOINT_PROMPT, ACTION_TOKEN_PROMPT)
    return question


print("start")

warnings.filterwarnings("ignore")

tokenizer, model, image_processor, max_length = load_pretrained_model(
    config.pretrained,
    config.model_base,
    config.model_name,
    attn_implementation="sdpa",
    device_map=config.device_map,
)

model = PeftModel.from_pretrained(model, config.lora_path, adapter_name="default")
model = model.merge_and_unload()
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


model.eval()
# image = Image.open("test.jpg")
# image = Image.open("/scratch/gilbreth/cancui/data/nuscenes/full/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg")
# with open('../LLaDA-AV/data/nuscenes_drive_data_single_image_val_v2.json', 'r') as f:
with open(config.data_path, 'r') as f:
    data_val = json.load(f)

total_time = 0

conv_template = config.conv_template 

inference_results = []

N_points = 1000
rnd = Random(42)
val_sample = rnd.sample(range(len(data_val)), N_points)
data_val_sample = [data_val[i] for i in val_sample]

# inference_results_len = len(inference_results)
# inference_results = []
# print(f"Inference results length: {inference_results_len}")
for i, data_sample in tqdm(enumerate(data_val_sample), total=N_points):
    # if i < inference_results_len:
    #     continue
    image = Image.open(data_sample['image'])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=config.device) for _image in image_tensor]
    image_sizes = [image.size]
    question = _to_action_token_prompt(data_sample['conversations'][0]['value'])
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

    best_res = None
    best_confidence = -float("inf")

    for gen_iter in range(4):

        start_time = time.time()
        torch.manual_seed(42 + gen_iter) # randomize each generation

        cont, x0_p = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=config.generation_steps,
            gen_length=config.generation_length,
            block_length=config.generation_block_length,
            tokenizer=tokenizer,
            stopping_criteria=list(config.stopping_criteria),
            temperature=config.temperature,
            remasking=config.remasking,
            return_confidence = True
        )

        # mean confidence
        # TODO: investigate whether choosing the individual most 
        # confident token at each index would be better??
        mean_confidence = x0_p.mean().item()
        print(f"Mean confidence {gen_iter}: {mean_confidence:.4f}")

        if mean_confidence > best_confidence:
            best_confidence = mean_confidence
            best_res = cont

        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time {gen_iter}: {generation_time:.4f} seconds")
        total_time += generation_time

    cont = best_res

    # print(cont)
    weights_file = Path(config.tokenizer_weights)
    if not weights_file.exists():
        raise FileNotFoundError()

    ids_tensor = torch.from_numpy(np.load(config.unused_token_ids_path)).to(config.device)
    pos_in_sorted = torch.searchsorted(-ids_tensor, -cont.flatten())      
    recovered_point = point_tokenizer.indices_to_points(pos_in_sorted)
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    print(text_outputs)
    print(recovered_point)

    inference_results.append([str(recovered_point.tolist())[1:-1], data_sample['conversations'][1]['value']])

# with open(f'/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_inference_{config.job_name}.json', 'w') as f:
with open(config.results_path, 'w') as f:
    json.dump(inference_results, f)
print(f"Saved inference results for {i}th data sample")

print(f"Total time: {total_time:.4f} seconds")
print(f"Average time: {total_time/len(inference_results):.4f} seconds")
# with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_lora_{job_name}_time.txt', 'w') as f:
#     f.write(f"Total Steps: {128}\n")
#     f.write(f"Total time: {total_time:.4f} seconds\n")
#     f.write(f"Average time: {total_time/len(inference_results):.4f} seconds")
