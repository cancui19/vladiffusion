export CUDA_VISIBLE_DEVICES=1

echo "Running nuscene inference lora sc time 32 v1"
python nuscene_inference_lora_sc_time_32_v1.py

echo "Running nuscene inference lora sc time 32 v2"
python nuscene_inference_lora_sc_time_32_v2.py

echo "Running nuscene inference lora sc time 32 v3"
python nuscene_inference_lora_sc_time_32_v3.py

echo "Running nuscene inference lora sc time 32 v4"
python nuscene_inference_lora_sc_time_32_v4.py

echo "Running nuscene inference lora sc time 32 v5"
python nuscene_inference_lora_sc_time_32_v5.py