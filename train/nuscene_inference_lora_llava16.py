from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import json
from transformers import pipeline


# model_id = "/scratch/gilbreth/cancui/output/llava_lora_sft"
model_id = '/scratch/gilbreth/cancui/models/llava-v1.6-mistral-7b-hf'
# pip = pipeline("image-text-to-text", model=model_id)
job_name = 'llava16_zero_shot'
with open('/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_v2.json', 'r') as f:
    data_val = json.load(f)
inference_results = []
print("Loaded annotations")
sum_time = 0
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False) 
model.to("cuda:0")
for i , data_sample in enumerate(data_val):

    modal_path = data_sample['image']
    image = Image.open(modal_path).convert("RGB") 
    # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)
    assert image is not None, "Image is None"
    i = str(i)
    print("Sample: ", i, "\n")
    instruct = data_sample['conversations'][0]['value'].replace('<image>', '')

    # print(instruct)

    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruct}
    ]}
]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # print(prompt)
    inputs = processor(images=image, text=prompt,add_special_tokens=False, return_tensors="pt").to(model.device)
    time_start = time.time()
# autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=128,use_cache=True,do_sample=True,temperature=1.5,top_p=0.95)
    # output = model.generate(**inputs, max_new_tokens=128,use_cache=True,do_sample=True,temperature=1.5,top_p=0.95)
    output_text = processor.decode(output[0],skip_special_tokens=False)
    print(output_text)
    time_end = time.time()
    generation_time = time_end - time_start
    print(f"Generation time: {generation_time:.4f} seconds")
    sum_time += generation_time
    print('Average time: ', sum_time/(int(i)+1))
    inference_results.append([output_text, data_sample['conversations'][1]['value']])
    print(output_text)
    with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_lora_{job_name}.json', 'w') as f:
        json.dump(inference_results, f)
    # print(output_text)
    # with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/{job_name}_withcache_time_nocache.txt', 'w') as f:
    #     f.write(f"Total time: {sum_time:.4f} seconds\n")
    #     f.write(f"Average time: {sum_time/len(inference_results):.4f} seconds")
    # print(f"Saved inference results for {i}th data sample")


