#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cd exp_scripts_nosc

echo "Running on GPU 0: nuscene_inference_lora_time_16.py..."
python nuscene_inference_lora_time_16.py

echo "Running on GPU 0: nuscene_inference_lora_time_96.py..."
python nuscene_inference_lora_time_96.py

echo "GPU 0 experiments completed!" 