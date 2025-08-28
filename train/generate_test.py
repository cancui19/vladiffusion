from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from peft import PeftModel
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
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, 
#                                                     None, 
#                                                     model_name, 
#                                                     attn_implementation="sdpa", 
#                                                     device_map=device_map)  # Add any other thing you want to pass in llava_model_args

lora_path = '/home/cancui/Research/LLaDA-V/train/exp/llada_v_finetune_nuscenes_single_image'

# from llava.model.language_model.llava_llada import LlavaLLaDAConfig
# from transformers import AutoTokenizer


model_path = pretrained
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# llada_cfg = LlavaLLaDAConfig.from_pretrained(model_path)


# if overwrite_config is not None:
#     rank0_print(f"Overwriting config with {overwrite_config}")
#     for k, v in overwrite_config.items():
#         setattr(llada_cfg, k, v)

######################
# from transformers import AutoConfig
# lora_cfg_pretrained = LlavaLLaDAConfig.from_pretrained(lora_path)

# from llava.model.language_model.llava_llada import LlavaLLaDAModelLM

# model = LlavaLLaDAModelLM.from_pretrained(model_path, 
#                                         low_cpu_mem_usage=True, 
#                                         attn_implementation="sdpa", 
#                                         config=lora_cfg_pretrained,
#                                         torch_dtype=torch.float16,
#                                         # config=llada_cfg,
#                                         trust_remote_code=True,
#                                         device_map=device_map)
##########################
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
print('model architecture: ', model)
# model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")

# mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
# mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
# if mm_use_im_patch_token:
#     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
# if mm_use_im_start_end:
#     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
# model.resize_token_embeddings(len(tokenizer))

# vision_tower = model.get_vision_tower()
# if not vision_tower.is_loaded:
#     vision_tower.load_model(device_map=device_map)
# if device_map != "auto":
#     vision_tower.to(device="cuda", dtype=torch.float16)
# image_processor = vision_tower.image_processor

# if hasattr(model.config, "max_sequence_length"):
#     context_len = model.config.max_sequence_length
# elif hasattr(model.config, "max_position_embeddings"):
#     context_len = model.config.max_position_embeddings
# elif hasattr(model.config, "tokenizer_model_max_length"):
#     context_len = model.config.tokenizer_model_max_length
# else:
#     context_len = 2048

# model = LlavaLLaDAModelLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llada_cfg, **kwargs)


# tokenizer, model, image_processor, max_length = load_pretrained_model(lora_path, 
#                                                     pretrained, 
#                                                     model_name, 
#                                                     attn_implementation="sdpa", 
#                                                     device_map=device_map)

# model = PeftModel.from_pretrained(base_model, "/scratch/gilbreth/cancui/models/LLaDA-V-lora")
# print('model architecture: ', model)
# model.eval()
# image = Image.open("test.jpg")
image = Image.open("/scratch/gilbreth/cancui/data/nuscenes/full/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg")
# import json
# with open('/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image.json', 'r') as f:
#     data_example = json.load(f)
# image = Image.open(data_example[0]['image'])

image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 
# question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
# with open("../prompt/prompt_v2.txt", "r") as f:
#     text_prompt = f.read()
# text_prompt = """
# You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.
# These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval.

# The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 

# Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:
# """
text_prompt = """You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.
The 5 second historical velocities and curvatures of the ego car are [2.09, 0.09], [2.09, 0.09], [1.97, 0.10], [2.00, 0.10], [2.17, 0.08], [2.71, 0.05], [2.54, 0.02], [2.20, 0.00], [2.80, -0.00], [2.94, -0.00]. 
Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:
"""

question = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

model.eval()
print(model)
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
    # register_cache_LLaDA_V(model, "base_model.model.model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

# print(len(image_tensor))
# print(type(image_tensor[0]))
# for i in range(len(image_tensor)):
#     image_tensor[i] = image_tensor[i].to(dtype=torch.float32, device=device)

# print(type(image_sizes))

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

print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)
