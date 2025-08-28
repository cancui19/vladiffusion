#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

cd exp_scripts_nosc

echo "Running on GPU 1: nuscene_inference_lora_time_32.py..."
python nuscene_inference_lora_time_32.py

echo "Running on GPU 1: nuscene_inference_lora_time_64.py..."
python nuscene_inference_lora_time_64.py

echo "GPU 1 experiments completed!" 