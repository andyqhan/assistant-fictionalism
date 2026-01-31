#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=embedding-variance
#SBATCH --output=logs/embedding_variance_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00

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

# Disable Python output buffering for immediate log output
export PYTHONUNBUFFERED=1

# Change to project directory
cd "${PROJECT_DIR}"

# Default paths
INPUT="${1:-logs/consistency-inference-1225210/embeddings.parquet}"
OUTPUT="${2:-logs/consistency-inference-1225210/embedding_variance.parquet}"

echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo ""

# Run variance computation inside container
singularity exec \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python scripts/compute_embedding_variance.py \
            --input '${INPUT}' \
            --output '${OUTPUT}'
    "

# Print completion info
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "=========================================="
