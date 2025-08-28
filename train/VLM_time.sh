echo "VLM test begin"

export CUDA_VISIBLE_DEVICES=0
python nuscene_inference_lora_llava16.py
python nuscene_inference_lora_qwenvl.py
python nuscene_inference_lora_llama32.py

echo "VLM test end"