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





job_name = 'qwen2vl_zero_shot'
prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_cache = True  # In this demo, we consider using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it.
print('start')

import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
sum_time = 0

with open('/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_v2.json', 'r') as f:
    data_val = json.load(f)
inference_results = []
print("Loaded annotations")

processor = AutoProcessor.from_pretrained("/scratch/gilbreth/cancui/models/Qwen2-VL-7B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/scratch/gilbreth/cancui/models/Qwen2-VL-7B-Instruct", torch_dtype="float16", device_map="auto"
)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/scratch/gilbreth/cancui/output/qwen2_vl_lora_sft", torch_dtype="auto", device_map="auto"
# )
model.to("cuda:0")

answer_dic = {}
modal = 'image'
for i , data_sample in enumerate(data_val):

    modal_path = data_sample['image']
    i = str(i)
    print("Sample: ", i, "\n")
    instruct = data_sample['conversations'][0]['value']
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": modal_path,
                # "max_pixels": 1024 * 1024,
            },
            {"type": "text", "text": instruct},
        ],
    }
]


    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    time_start = time.time()
    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128,use_cache=True,do_sample=True,temperature=1.5,top_p=0.95)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    inference_results.append([output_text, data_sample['conversations'][1]['value']])
    print(output_text)
    time_end = time.time()
    generation_time = time_end - time_start
    print(f"Generation time: {generation_time:.4f} seconds")
    sum_time += generation_time
    print('Average time: ', sum_time/(int(i)+1))
    with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_lora_{job_name}.json', 'w') as f:
        json.dump(inference_results, f)
    print(output_text)
    # with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/{job_name}_withcache_time_nocache.txt', 'w') as f:
    #     f.write(f"Total time: {sum_time:.4f} seconds\n")
    #     f.write(f"Average time: {sum_time/len(inference_results):.4f} seconds")
    # print(f"Saved inference results for {i}th data sample")

