#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1

# echo "sleep 36000"

# sleep 36000

echo "start differnt steps tasks"

# python nuscene_inference_lora_time_step32.py
# python nuscene_inference_lora_time_step64.py
# python nuscene_inference_lora_time_step96.py
python nuscene_inference_lora_time_step16.py
python nuscene_inference_lora_time_step128.py

echo "end differnt steps task"