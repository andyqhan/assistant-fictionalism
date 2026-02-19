#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=user-turn-embeddings
#SBATCH --output=logs/user_turn_embeddings_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00

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

# Parse --input argument
INPUT=""
for arg in "$@"; do
    case "$arg" in
        --input=*) INPUT="${arg#--input=}" ;;
    esac
done

if [ -z "$INPUT" ]; then
    echo "Usage: sbatch hpc/compute_user_turn_embeddings.slurm --input=<path/to/results.jsonl>"
    exit 1
fi

INPUT_DIR=$(dirname "$INPUT")

# Define embedding models: model_id|pooling|output_key
MODELS=(
    "Qwen/Qwen3-Embedding-0.6B|last_token|qwen3-embedding-0.6b"
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5|mean|kalm-embedding"
    "codefuse-ai/F2LLM-1.7B|last_token|f2llm-1.7b"
)

# Response columns to embed
RESPONSE_COLS=("response1" "response2")

# Run all models x response columns
for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL POOLING KEY <<< "$entry"

    for RESP_COL in "${RESPONSE_COLS[@]}"; do
        OUTPUT="${INPUT_DIR}/embeddings_${KEY}_${RESP_COL}.parquet"

        echo ""
        echo "=========================================="
        echo "Running: ${KEY} / ${RESP_COL}"
        echo "Model:   ${MODEL} (pooling=${POOLING})"
        echo "Output:  ${OUTPUT}"
        echo "=========================================="

        singularity exec --nv \
            --bind /scratch:/scratch \
            "${CONTAINER}" \
            bash -c "
                source ${VENV}/bin/activate && \
                python -m src.batch.embeddings \
                    --input='${INPUT}' \
                    --model='${MODEL}' \
                    --pooling='${POOLING}' \
                    --output='${OUTPUT}' \
                    --response-column='${RESP_COL}'
            "

        echo "Finished: ${KEY} / ${RESP_COL}"
    done
done

# Print completion info
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "=========================================="
