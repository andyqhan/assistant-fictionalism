#!/bin/bash
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=install-package
#SBATCH --output=logs/install_package_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --constraint=h200

# Install Python packages that require CUDA at compile time
# Usage: sbatch hpc/install_package.slurm <package> [pip-options...]
# Example: sbatch hpc/install_package.slurm flash-attn --no-build-isolation

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: sbatch hpc/install_package.slurm <package> [pip-options...]"
    echo "Example: sbatch hpc/install_package.slurm flash-attn --no-build-isolation"
    exit 1
fi

PACKAGE="$1"
shift
PIP_OPTIONS="$@"

# Print job info
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Package: ${PACKAGE}"
echo "Pip options: ${PIP_OPTIONS}"
echo "=========================================="

# Paths
PROJECT_DIR="/scratch/ah7660/assistant-fictionalism"
CONTAINER="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
VENV="/scratch/ah7660/venvs/assistant-fictionalism-env"

# Set Hugging Face cache (some packages check for CUDA via torch)
export HF_HOME="/scratch/ah7660/hf_cache"

# Set pip cache and temp dirs to /scratch to avoid cross-device link errors
export PIP_CACHE_DIR="/scratch/ah7660/pip_cache"
export TMPDIR="/scratch/ah7660/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

# Change to project directory
cd "${PROJECT_DIR}"

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Print CUDA info
echo "CUDA Info:"
singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "nvcc --version || echo 'nvcc not found'"
echo ""

# Install package inside container
# Use python -m pip since uv venvs don't include pip executable by default
# Pass env vars for cache dirs to avoid cross-device link errors
echo "Installing ${PACKAGE}..."
singularity exec --nv \
    --bind /scratch:/scratch \
    --env PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
    --env TMPDIR="${TMPDIR}" \
    "${CONTAINER}" \
    bash -c "${VENV}/bin/python -m pip install ${PACKAGE} ${PIP_OPTIONS}"

# Verify installation
echo ""
echo "Verifying installation..."
singularity exec --nv \
    --bind /scratch:/scratch \
    "${CONTAINER}" \
    bash -c "
        ${VENV}/bin/python -c \"import ${PACKAGE//-/_}; print('${PACKAGE} imported successfully')\" 2>/dev/null || \
        ${VENV}/bin/python -m pip show ${PACKAGE}
    "

# Print completion info
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Installation completed successfully"
echo "=========================================="
