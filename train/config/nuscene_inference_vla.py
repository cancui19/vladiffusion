"""Configuration for NuScenes inference with VLA LoRA adapter."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NuScenesVLAInferenceConfig:
    job_name: str = "nuscenes_lora_vla"
    prompt_interval_steps: int = 25
    gen_interval_steps: int = 7
    transfer_ratio: float = 0.25
    use_cache: bool = True
    pretrained: str = "/scratch/gautschi/mgagvani/vladiffusion_scratch/VLA_finetune_nuscenes_single_image_train_20251029_2026"
    model_base: str = "GSAI-ML/LLaDA-V"
    model_name: str = "llava_llada_lora"
    device: str = "cuda:0"
    device_map: str = "cuda:0"
    lora_path: str = "/scratch/gautschi/mgagvani/vladiffusion_scratch/VLA_finetune_nuscenes_single_image_train_20251029_2026"
    data_path: str = "/scratch/gautschi/mgagvani/new_nuscenes/downloads/nuscenes_waypoint_long_prompt_train.json"
    results_path: str = "/home/mgagvani/vladiffusion/results/vla_init.json"
    tokenizer_weights: str = "/home/mgagvani/vladiffusion/tokenizer/tokenizer_model.pth"
    unused_token_ids_path: str = "tokenizer/unused_token_ids.npy"
    conv_template: str = "llava_llada"
    generation_steps: int = 10
    generation_length: int = 10
    generation_block_length: int = 10
    stopping_criteria: tuple[str, ...] = ("<|eot_id|>",)


config = NuScenesVLAInferenceConfig()

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