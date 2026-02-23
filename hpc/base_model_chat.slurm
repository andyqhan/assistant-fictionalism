#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=base-model-chat
#SBATCH --output=logs/base_model_chat_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00

# Exit on error
set -e

# Print job info
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Paths
PROJECT_DIR="/scratch/ah7660/assistant-fictionalism"
CONTAINER="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
VENV="/scratch/ah7660/venvs/assistant-fictionalism-env"

# Set Hugging Face cache
export HF_HOME="/scratch/ah7660/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ah7660/hf_cache"

# Disable Python output buffering for immediate log output
export PYTHONUNBUFFERED=1

# Change to project directory
cd "${PROJECT_DIR}"

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

BASE_MODEL="Qwen/Qwen3-8B-Base"
INSTRUCT_TOKENIZER="Qwen/Qwen3-8B"
OUTPUT_DIR="logs/base-model-chat-${SLURM_JOB_ID}"

# ---- Experiment 1: Assistant-turn generation ----
echo ""
echo "=========================================="
echo "Experiment 1: Assistant-turn generation (base model with instruct chat template)"
echo "=========================================="

singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python -m src.batch.inference \
            --prompts-json datasets/prompts_small.jsonl \
            --personae-json datasets/personae_small.json \
            --model ${BASE_MODEL} \
            --tokenizer ${INSTRUCT_TOKENIZER} \
            --no-thinking-mode \
            --temperature 0.6 \
            --max-tokens 256 \
            --batch-size 32 \
            --n-reps 1 \
            --subset-prompt 0.05 \
            --subset-category-persona 0.15 \
            --output-dir ${OUTPUT_DIR}/assistant-turns \
            --gpu-monitor-interval 60
    "

echo ""
echo "Assistant-turn generation complete."

# ---- Experiment 2: User-turn generation ----
echo ""
echo "=========================================="
echo "Experiment 2: User-turn generation (base model with instruct chat template)"
echo "=========================================="

singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python -m src.batch.user_turn_prediction \
            --prompts-jsonl datasets/consistency_prompts.jsonl \
            --personae-json datasets/consistency_personae.json \
            --model ${BASE_MODEL} \
            --tokenizer ${INSTRUCT_TOKENIZER} \
            --temperature 0.6 \
            --max-tokens 256 \
            --batch-size 32 \
            --n-samples 3 \
            --output-dir ${OUTPUT_DIR}/user-turns \
            --gpu-monitor-interval 60
    "

echo ""
echo "User-turn generation complete."

# Print completion info
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Job completed successfully"
echo "=========================================="
