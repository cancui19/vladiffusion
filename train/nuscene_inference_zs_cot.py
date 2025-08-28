from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict

from PIL import Image
import requests
import copy
import torch
import time

import sys
import warnings

job_name = 'zero_shot_cot'
prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_cache = True  # In this demo, we consider using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it.
print('start')

warnings.filterwarnings("ignore")
# pretrained = "GSAI-ML/LLaDA-V"
pretrained = "/scratch/gilbreth/cancui/models/LLaDA-V"
# save_dir = "/scratch/gilbreth/cancui/models/LLaDA-V"

model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()
# image = Image.open("test.jpg")
# image = Image.open("/scratch/gilbreth/cancui/data/nuscenes/full/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg")
import json
with open('/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_v2.json', 'r') as f:
    data_val = json.load(f)

with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_zs_cot_zero_shot_cot.json', 'r') as f:
    inference_results = json.load(f)

inference_results_len = len(inference_results)
print(f"Inference results length: {inference_results_len}")
conv_template = "llava_llada" 

# text_prompt = """You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.
# The 5 second historical velocities and curvatures of the ego car are [2.09, 0.09], [2.09, 0.09], [1.97, 0.10], [2.00, 0.10], [2.17, 0.08], [2.71, 0.05], [2.54, 0.02], [2.20, 0.00], [2.80, -0.00], [2.94, -0.00]. 
# Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:
# """
# question = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt

# inference_results = []
chain_of_thought = """
Generate  the following information: \nScene description:\n[r1]\nCritial objects:\n[r2]\nBehavior description:\n[r3]\nMeta driving decision:\n[r4]
        In r1, Provide a short description of driving scenarios.
        In r2, Plesase list 2-3 key objects the ego car should focus on, specifying only the object's name and its location within the image of the driving scene. Only output the name of the object and its related location in the image of the driving scence without any other information.
        In r3, Provide a short description of the current status and intended actions to the r2 identified critical object for the ego car.
        In r4, Provide a short description of the meta driving decision.
"""

for i, data_sample in enumerate(data_val):
    if i < inference_results_len:
        continue
    image = Image.open(data_sample['image'])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size]
    sys_msg, user_msg = data_sample['conversations'][0]['value'].split(DEFAULT_IMAGE_TOKEN)

    question_cot = DEFAULT_IMAGE_TOKEN + "\n" + chain_of_thought
    # question = DEFAULT_IMAGE_TOKEN + "\n" + chain_of_thought
    print(question_cot)

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_cot)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    model.eval()

    if use_cache:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        print("Testing with cache enabled")
    else:
        print("Testing without cache")

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    start_time = time.time()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>']
    )
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")

    # print(cont)
    cot_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    print('cot_outputs', cot_outputs)


    user_question = sys_msg + "\n" + cot_outputs[0] + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + user_msg
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], user_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    print('user_question', prompt_question)

    model.eval()

    if use_cache:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        print("Testing with cache enabled")
    else:
        print("Testing without cache")

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    start_time = time.time()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>']
    )
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")

    # print(cont)
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    print('text_outputs', text_outputs)

    inference_results.append([text_outputs, data_sample['conversations'][1]['value']])
    # break

    # with open(f'/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_inference_{job_name}.json', 'w') as f:
    with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_{job_name}.json', 'w') as f:
        json.dump(inference_results, f)
    print(f"Saved inference results for {i}th data sample")

# print(inference_results)
