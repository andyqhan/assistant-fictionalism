#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=personae-inference
#SBATCH --output=logs/personae_inference_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
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

# Run inference inside container
# Pass all script arguments to the Python script
singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        source ${VENV}/bin/activate && \
        python -m src.batch.inference \"\$@\"
    " -- "$@"

# Print completion info
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Job completed successfully"
echo "=========================================="
