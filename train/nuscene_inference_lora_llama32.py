
import json
import time
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
# model_id = "/scratch/gilbreth/cancui/output/llama3_lora_sft"
model_id = '/scratch/gilbreth/cancui/models/Llama-3.2-11B-Vision-Instruct'
job_name = 'llama32_zero_shot'
with open('/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_single_image_val_v2.json', 'r') as f:
    data_val = json.load(f)
inference_results = []
print("Loaded annotations")
sum_time = 0
processor = AutoProcessor.from_pretrained(model_id)
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


for i , data_sample in enumerate(data_val):

    modal_path = data_sample['image']
    image = Image.open(modal_path)
    i = str(i)
    print("Sample: ", i, "\n")
    instruct = data_sample['conversations'][0]['value']

    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruct}
    ]}
]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    time_start = time.time()
    output = model.generate(**inputs, max_new_tokens=128,use_cache=True,do_sample=True,temperature=1.5,top_p=0.95)
    output_text = processor.decode(output[0])
    time_end = time.time()
    generation_time = time_end - time_start
    print(f"Generation time: {generation_time:.4f} seconds")
    sum_time += generation_time
    print('Average time: ', sum_time/(int(i)+1))
    inference_results.append([output_text, data_sample['conversations'][1]['value']])
    print(output_text)
    with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/nuscenes_drive_data_single_image_val_inference_{job_name}.json', 'w') as f:
        json.dump(inference_results, f)
    print(output_text)
    # with open(f'/scratch/gilbreth/cancui/LLaDA-V/results/{job_name}_withcache_time_nocache.txt', 'w') as f:
    #     f.write(f"Total time: {sum_time:.4f} seconds\n")
    #     f.write(f"Average time: {sum_time/len(inference_results):.4f} seconds")
    # print(f"Saved inference results for {i}th data sample")

