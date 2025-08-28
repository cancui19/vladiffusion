export CUDA_VISIBLE_DEVICES=1

echo "start"

echo "nosc_time_thres_0.3"
python nuscene_inference_lora_nosc_time_thres_0.3.py

echo "nosc_time_thres_0.5"
python nuscene_inference_lora_nosc_time_thres_0.5.py

echo "nosc_time_thres_0.7"
python nuscene_inference_lora_nosc_time_thres_0.7.py

echo "nosc_time_thres_0.9"
python nuscene_inference_lora_nosc_time_thres_0.9.py

echo "end"