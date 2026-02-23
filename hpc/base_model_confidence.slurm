#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=base-vs-instruct
#SBATCH --output=logs/base_vs_instruct_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --constraint=h200

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

# Shared config
OUTPUT_ROOT="logs/base-vs-instruct-${SLURM_JOB_ID}"
COMMON_ARGS=(
    --prompts-json=datasets/prompts.jsonl
    --personae-json=datasets/personae.json
    --temperature=0.7
    --max-tokens=1024
    --batch-size=512
    --n-reps=1
    --top-k-mass-k=5
    --no-thinking-mode
    --system-prompt-style=you-are-a
    --backend=transformers
)

# ──────────────────────────────────────────────────────────────────────────────
# Run 1: Base model (Qwen3-8B-Base with instruct tokenizer)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "RUN 1: Base model (Qwen/Qwen3-8B-Base)"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python -m src.batch.inference \
            --model=Qwen/Qwen3-8B-Base \
            --tokenizer=Qwen/Qwen3-8B \
            --output-dir=${OUTPUT_ROOT}/base \
            ${COMMON_ARGS[*]}
    "

echo ""
echo "=========================================="
echo "RUN 1 complete: $(date)"
echo "=========================================="

# ──────────────────────────────────────────────────────────────────────────────
# Run 2: Instruct model (Qwen3-8B)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "RUN 2: Instruct model (Qwen/Qwen3-8B)"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python -m src.batch.inference \
            --model=Qwen/Qwen3-8B \
            --output-dir=${OUTPUT_ROOT}/instruct \
            ${COMMON_ARGS[*]}
    "

echo ""
echo "=========================================="
echo "RUN 2 complete: $(date)"
echo "=========================================="

# Print completion info
echo ""
echo "=========================================="
echo "Both runs complete."
echo "Base results:     ${OUTPUT_ROOT}/base/results.jsonl"
echo "Instruct results: ${OUTPUT_ROOT}/instruct/results.jsonl"
echo "End time: $(date)"
echo "Job completed successfully"
echo "=========================================="
