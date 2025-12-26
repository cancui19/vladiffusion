"""Configuration for NuScenes inference with VLA LoRA adapter."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NuScenesVLAInferenceConfig:
    job_name: str = "nuscenes_lora_vla"
    prompt_interval_steps: int = 25
    gen_interval_steps: int = 7
    transfer_ratio: float = 0.25
    use_cache: bool = True
    pretrained: str = "/anvil/scratch/x-mgagvani/vladiffusion_scratch/VLA_finetune_nuscenes_single_image_train_20251224_1124_from_curr"
    model_base: str = "GSAI-ML/LLaDA-V"
    model_name: str = "llava_llada_lora"
    device: str = "cuda:0"
    device_map: str = "cuda:0"
    lora_path: str = "/anvil/scratch/x-mgagvani/vladiffusion_scratch/VLA_finetune_nuscenes_single_image_train_20251224_1124_from_curr"
    data_path: str = "data/nuscenes_delta_text_long_prompt_train.json"
    results_path: str = "/home/x-mgagvani/vladiffusion/results/vla_20251224_1124.json"
    tokenizer_weights: str = "/home/x-mgagvani/vladiffusion/tokenizer/tokenizer_model.pth"
    unused_token_ids_path: str = "/home/x-mgagvani/vladiffusion/tokenizer/unused_token_ids.npy"
    conv_template: str = "llava_llada"
    generation_steps: int = 10
    generation_length: int = 10
    generation_block_length: int = 10
    stopping_criteria: tuple[str, ...] = ("<|eot_id|>",)


config = NuScenesVLAInferenceConfig()

config_from_curr = NuScenesVLAInferenceConfig(
    job_name="nuscenes_lora_vla_from_curr",
    pretrained="/scratch/gilbreth/cancui/models/LLaDA-V",
    model_name="llava_llada_lora",
    lora_path="/depot/ziran/apps/jiaru/projects/vladiffusion/exp/VLA_finetune_nuscenes_single_image_train_20251119_1334_from_curr",
    data_path="/depot/ziran/apps/jiaru/projects/vladiffusion/data/nuscenes_waypoint_text_long_prompt_val.json",
    results_path="/depot/ziran/apps/jiaru/projects/vladiffusion/results/vla_20251119_1334_from_curr.json",
    tokenizer_weights="/depot/ziran/apps/jiaru/projects/vladiffusion/tokenizer/tokenizer_model.pth",
    unused_token_ids_path="/depot/ziran/apps/jiaru/projects/vladiffusion/tokenizer/unused_token_ids.npy"
)

config_explanation = NuScenesVLAInferenceConfig(
    job_name="nuscenes_lora_vla_explanation",
    pretrained="/scratch/gilbreth/cancui/models/LLaDA-V",
    model_name="llava_llada",
    lora_path="/depot/ziran/apps/jiaru/projects/vladiffusion/exp/VLA_finetune_nuscenes_single_image_train_explanation",
    data_path="data/nuscenes_waypoint_text_long_prompt_val_Nu_X.json",
    results_path="/depot/ziran/apps/jiaru/projects/vladiffusion/results/vla_explanation_init.json",
    generation_steps=128,
    generation_length=128,
    generation_block_length=128
)


@dataclass(frozen=True)
class NuScenesVLAInferenceGilbrethConfig:
    job_name: str = "nuscenes_lora_vla"
    prompt_interval_steps: int = 25
    gen_interval_steps: int = 7
    transfer_ratio: float = 0.25
    use_cache: bool = True
    pretrained: str = "/scratch/gilbreth/cancui/models/LLaDA-V"
    model_name: str = "llava_llada_lora"
    model_base: str = "GSAI-ML/LLaDA-V"
    device: str = "cuda:0"
    device_map: str = "cuda:0"
    lora_path: str = "/depot/ziran/apps/jiaru/projects/vladiffusion/exp/VLA_finetune_nuscenes_single_image_train"
    data_path: str = "data/nuscenes_waypoint_text_long_prompt_val.json"
    results_path: str = "/depot/ziran/apps/jiaru/projects/vladiffusion/results/vla_init.json"
    tokenizer_weights: str = "/depot/ziran/apps/jiaru/projects/vladiffusion/tokenizer/tokenizer_model.pth"
    unused_token_ids_path: str = "tokenizer/unused_token_ids.npy"
    conv_template: str = "llava_llada"
    generation_steps: int = 10
    generation_length: int = 10
    generation_block_length: int = 10
    stopping_criteria: tuple[str, ...] = ("<|eot_id|>",)


config_curr = NuScenesVLAInferenceGilbrethConfig(
    job_name="nuscenes_lora_vla_explanation",
    lora_path="/depot/ziran/apps/jiaru/projects/vladiffusion/exp/VLA_finetune_train_curr_longer_lora",
    data_path="data/nuscenes_waypoint_text_long_prompt_val.json",
    results_path="/depot/ziran/apps/jiaru/projects/vladiffusion/results/vla_curr_longer_lora.json",
    generation_steps=20,
    generation_length=20,
    generation_block_length=20
)