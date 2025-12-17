#!/bin/bash
#SBATCH --job-name=nuscene_infer_from_curr
#SBATCH --account=cis251316-gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --chdir=/home/x-mgagvani/vladiffusion

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output=exp/slurm-%x-%j.out
#SBATCH --error=exp/slurm-%x-%j.err

module load conda
conda activate vladiffusion

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/x-mgagvani/vladiffusion}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train:${PYTHONPATH}"
export HF_HOME=$(findscratch)/hfcache
export TRANSFORMERS_TRUST_REMOTE_CODE=1

set -x
cd "${REPO_ROOT}"
python "${REPO_ROOT}/train/nuscene_inference_lora_time_vla_from_curr.py"
