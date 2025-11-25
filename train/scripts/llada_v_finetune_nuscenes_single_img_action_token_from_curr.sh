set -x
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ibp161s0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train:${PYTHONPATH}"

num_node=$1
gpu_num=$2

# num_node=1
# gpu_num=1


# need to change num_node and gpu_num! 
# Configuration note: This script is typically run with 4 nodes and 8 GPUs per node.
# The gradient_accumulation_steps should be adjusted based on your GPU count to maintain effective batch size.
# For example, with 8 GPUs, set gradient_accumulation_steps=8.

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29198"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="/scratch/gilbreth/cancui/models/LLaDA-V"
# LLM_VERSION="GSAI-ML/LLaDA-V"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION="google/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
INIT_LORA_DIR="/depot/ziran/apps/jiaru/projects/vladiffusion/exp/VLA_finetune_train_curr_longer_lora"
INIT_CKPT="${INIT_LORA_DIR}"   # 看目录里实际存的是哪个 checkpoint


############### Finetune ################

PROMPT_VERSION="llava_llada"

DATE=$(date +%Y%m%d_%H%M)
BASE_RUN_NAME="VLA_finetune_nuscenes_single_image_train_${DATE}_from_curr"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"


# torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --node_rank=${RANK} \
    # train/llava/train/train_mem.py \
# --ddp_find_unused_parameters True \

python -m llava.train.train_mem \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "/depot/ziran/apps/jiaru/projects/vladiffusion/data/nuscenes_waypoint_text_long_prompt_train.json" \
    --lora_init_path ${INIT_CKPT} \
    --image_folder "/" \
    --video_folder "/" \
    --lora_enable True \
    --lora_r 256 \
    --mm_tunable_parts="" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bits 16 \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "/depot/ziran/apps/jiaru/projects/vladiffusion/exp/$BASE_RUN_NAME" \
    --num_train_epochs 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    # --deepspeed train/scripts/zero2.json 

# torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
#     /home/cancui/Research/LLaDA-V/train/llava/train/train_mem.py \
#     --model_name_or_path ${LLM_VERSION} \
#     --version ${PROMPT_VERSION} \
#     --data_path "/home/cancui/Research/LLaDA-V/data/nuscenes_drive_data_simple.json" \
#     --image_folder "/" \
#     --video_folder "/" \
#     --lora_enable True \
#     --lora_r 64 \
#     --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
#     --mm_vision_tower_lr=2e-6 \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --image_aspect_ratio anyres_max_4 \
#     --image_grid_pinpoints "(1x1),...,(6x6)" \
#     --mm_patch_merge_type spatial_unpad \
#     --bf16 True \
#     --run_name $BASE_RUN_NAME \
#     --output_dir "exp/$BASE_RUN_NAME" \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 5000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 0 \
#     --lazy_preprocess True \
#     --report_to tensorboard \
#     --torch_compile True \
#     --torch_compile_backend "inductor" \
#     --dataloader_drop_last True \
#     --attn_implementation sdpa \
#     --use_conversation_mask False