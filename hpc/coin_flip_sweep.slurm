#!/bin/bash
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --job-name=coin-flip-sweep
#SBATCH --output=logs/coin_flip_sweep_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00

# Runs coin flip experiment across multiple Qwen3 models sequentially.
# Each model gets its own output directory: logs/coin-flip-{SLURM_JOB_ID}-{model_slug}

set -e

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Paths
PROJECT_DIR="/scratch/ah7660/assistant-fictionalism"
CONTAINER="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
VENV="/scratch/ah7660/venvs/assistant-fictionalism-env"

export HF_HOME="/scratch/ah7660/hf_cache"
export TRANSFORMERS_CACHE="/scratch/ah7660/hf_cache"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

COMMON_ARGS="--personae-json=datasets/coin-flip-personae.json --use-tasks-from=assistant --system-prompt-style=you-are-a"

for MODEL in "${MODELS[@]}"; do
    # Create a slug from model name: Qwen/Qwen3-4B -> qwen3-4b
    SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR="logs/coin-flip-${SLURM_JOB_ID}-${SLUG}"

    echo ""
    echo "=========================================="
    echo "Running: ${MODEL}"
    echo "Output:  ${OUTPUT_DIR}"
    echo "Time:    $(date)"
    echo "=========================================="

    singularity exec --nv \
        --bind /scratch:/scratch \
        "${CONTAINER}" \
        bash -c "
            source ${VENV}/bin/activate && \
            python -m src.batch.coin_flip \
                --model=${MODEL} \
                --output-dir=${OUTPUT_DIR} \
                ${COMMON_ARGS}
        "

    echo "Finished ${MODEL} at $(date)"
done

echo ""
echo "=========================================="
echo "All models complete"
echo "End time: $(date)"
echo "=========================================="
