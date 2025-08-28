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
from peft import PeftModel

job_name = 'single_image_train_lora_sc_time_step_32_threshold_0.7'
prompt_interval_steps = 32
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
lora_path = '/home/cancui/Research/LLaDA-V/train/exp/062325_llada_v_finetune_nuscenes_single_image_train_shortcut'

model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")
model = model.merge_and_unload()
total_time = 0
model.eval()
# image = Image.open("test.jpg")
# image = Image.open("/scratch/gilbreth/cancui/data/nuscenes/full/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg")
import json
with open('/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_v2.json', 'r') as f:
    data_val = json.load(f)
    # image = Image.open(data_val[0]['image'])
    # text_prompt = data_val[0]['conversations'][0]['value']
    # print(text_prompt)
#     f.close()

# image_tensor = process_images([image], image_processor, model.config)
# image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 

# text_prompt = """You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.
# The 5 second historical velocities and curvatures of the ego car are [2.09, 0.09], [2.09, 0.09], [1.97, 0.10], [2.00, 0.10], [2.17, 0.08], [2.71, 0.05], [2.54, 0.02], [2.20, 0.00], [2.80, -0.00], [2.94, -0.00]. 
# Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:
# """
# question = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt

inference_results = []

for i, data_sample in enumerate(data_val):
    image = Image.open(data_sample['image'])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size]
    # question = DEFAULT_IMAGE_TOKEN + "\n" + data_sample['conversations'][0]['value']
    question = data_sample['conversations'][0]['value']
    print(question)

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
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
        steps=32, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'],
        remasking='confidence_threshold',
        confidence_threshold=0.7,
        verbose=False
    )
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")
    total_time += generation_time
    # print(cont)
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    print(text_outputs)
    inference_results.append([text_outputs, data_sample['conversations'][1]['value']])
    print('current average time: ', total_time/(int(i)+1))

    # with open(f'/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_inference_{job_name}.json', 'w') as f:
#     with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_sc_{job_name}.json', 'w') as f:
#         json.dump(inference_results, f)
#     print(f"Saved inference results for {i}th data sample")

# print(f"Total time: {total_time:.4f} seconds")
# print(f"Average time: {total_time/len(inference_results):.4f} seconds")
# with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_sc_{job_name}_time.txt', 'w') as f:
#     f.write(f"Total Steps: {32}\n")
#     f.write(f"Total time: {total_time:.4f} seconds\n")
#     f.write(f"Average time: {total_time/len(inference_results):.4f} seconds")

# print(inference_results)