#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# echo "sleep 36000"

# sleep 36000

echo "start differnt steps tasks"


python nuscene_inference_lora_time_step32_threshold0.7.py
python nuscene_inference_lora_time_step32_threshold0.3.py
python nuscene_inference_lora_time_step32_threshold0.5.py
python nuscene_inference_lora_time_step32_threshold0.9.py

echo "end differnt steps task"