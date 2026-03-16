#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=test-hf-token
#SBATCH --output=logs/test_hf_token_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:l40s:1

set -e

PROJECT_DIR="/scratch/ah7660/assistant-fictionalism"
CONTAINER="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
VENV="/scratch/ah7660/venvs/assistant-fictionalism-env"

export HF_HOME="/scratch/ah7660/hf_cache"

# Load HF_TOKEN from .env
set -a
source "${PROJECT_DIR}/.env"
set +a

cd "${PROJECT_DIR}"

echo "Testing HF token authentication..."
echo "HF_TOKEN is set: $([ -n \"$HF_TOKEN\" ] && echo yes || echo no)"
echo "HF_TOKEN length: ${#HF_TOKEN}"

singularity exec --nv \
    --bind /scratch:/scratch \
    --env HF_TOKEN="${HF_TOKEN}" \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python -c \"
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('google/gemma-3-1b-it')
print(f'Success! Model: {info.modelId}')
print(f'Pipeline tag: {info.pipeline_tag}')
print('HF token is working for gated Gemma models.')
\"
    "
