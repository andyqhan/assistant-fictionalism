#!/bin/bash
#
# Consistency Judge Pipeline
#
# Orchestrates parallel embedding + TC-LLM analyses on prefill inference output.
# Run with: bash hpc/consistency_judge_pipeline.slurm <input_dir>
#
# Example:
#   bash hpc/consistency_judge_pipeline.slurm logs/prefill-inference-1472066
#
# Submits 7 SLURM jobs:
#   - 3 GPU jobs: embeddings (3 models)
#   - 3 CPU jobs: embedding variance (dependent on corresponding embedding job)
#   - 1 GPU job: TC-LLM text clustering (independent)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: bash hpc/consistency_judge_pipeline.slurm <input_dir>"
    echo "  <input_dir> must contain results.jsonl"
    exit 1
fi

INPUT_DIR="$1"
INPUT_FILE="${INPUT_DIR}/results.jsonl"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: ${INPUT_FILE} not found"
    exit 1
fi

echo "=========================================="
echo "Consistency Judge Pipeline"
echo "Input: ${INPUT_FILE}"
echo "Start time: $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="logs/consistency-judge-pipeline-${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Symlink input results
ln -sf "$(realpath "${INPUT_FILE}")" "${OUTPUT_DIR}/results.jsonl"

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Embedding models configuration
# ---------------------------------------------------------------------------
declare -a EMB_MODELS=(
    "Qwen/Qwen3-Embedding-0.6B"
    "codefuse-ai/F2LLM-1.7B"
    "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
)
declare -a EMB_POOLING=(
    "last_token"
    "last_token"
    "mean"
)
declare -a EMB_SLUGS=(
    "qwen3-embedding-0.6b"
    "f2llm-1.7b"
    "kalm-embedding"
)

# ---------------------------------------------------------------------------
# Submit embedding jobs (3 parallel GPU jobs)
# ---------------------------------------------------------------------------
declare -a EMB_JOB_IDS=()

for i in 0 1 2; do
    MODEL="${EMB_MODELS[$i]}"
    POOLING="${EMB_POOLING[$i]}"
    SLUG="${EMB_SLUGS[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/embeddings_${SLUG}.parquet"

    EMB_JOB_ID=$(sbatch --parsable \
        hpc/compute_embeddings.slurm \
        --input "${OUTPUT_DIR}/results.jsonl" \
        --model "${MODEL}" \
        --pooling "${POOLING}" \
        --output "${OUTPUT_FILE}")

    EMB_JOB_IDS+=("${EMB_JOB_ID}")
    echo "Submitted embedding job [${SLUG}]: ${EMB_JOB_ID}"
    echo "  Model: ${MODEL}"
    echo "  Pooling: ${POOLING}"
    echo "  Output: ${OUTPUT_FILE}"
    echo ""
done

# ---------------------------------------------------------------------------
# Submit variance jobs (3 CPU jobs, each depends on its embedding job)
# ---------------------------------------------------------------------------
declare -a VAR_JOB_IDS=()

for i in 0 1 2; do
    SLUG="${EMB_SLUGS[$i]}"
    EMB_FILE="${OUTPUT_DIR}/embeddings_${SLUG}.parquet"
    VAR_FILE="${OUTPUT_DIR}/embedding_variance_${SLUG}.parquet"
    DEP_JOB="${EMB_JOB_IDS[$i]}"

    VAR_JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${DEP_JOB} \
        hpc/compute_embedding_variance.slurm \
        "${EMB_FILE}" "${VAR_FILE}")

    VAR_JOB_IDS+=("${VAR_JOB_ID}")
    echo "Submitted variance job [${SLUG}]: ${VAR_JOB_ID} (depends on ${DEP_JOB})"
    echo "  Input: ${EMB_FILE}"
    echo "  Output: ${VAR_FILE}"
    echo ""
done

# ---------------------------------------------------------------------------
# Submit TC-LLM job (1 GPU job, independent)
# ---------------------------------------------------------------------------
TC_LLM_JOB_ID=$(sbatch --parsable \
    hpc/tc_llm.slurm \
    --input-path "${OUTPUT_DIR}/results.jsonl" \
    --output-dir "${OUTPUT_DIR}")

echo "Submitted TC-LLM job: ${TC_LLM_JOB_ID}"
echo "  Input: ${OUTPUT_DIR}/results.jsonl"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Save pipeline config
# ---------------------------------------------------------------------------
cat > "${OUTPUT_DIR}/pipeline_config.json" <<EOF
{
  "input_dir": "$(realpath "${INPUT_DIR}")",
  "output_dir": "${OUTPUT_DIR}",
  "timestamp": "${TIMESTAMP}",
  "embeddings": [
    {"model": "${EMB_MODELS[0]}", "pooling": "${EMB_POOLING[0]}", "slug": "${EMB_SLUGS[0]}", "job_id": "${EMB_JOB_IDS[0]}"},
    {"model": "${EMB_MODELS[1]}", "pooling": "${EMB_POOLING[1]}", "slug": "${EMB_SLUGS[1]}", "job_id": "${EMB_JOB_IDS[1]}"},
    {"model": "${EMB_MODELS[2]}", "pooling": "${EMB_POOLING[2]}", "slug": "${EMB_SLUGS[2]}", "job_id": "${EMB_JOB_IDS[2]}"}
  ],
  "variance": [
    {"slug": "${EMB_SLUGS[0]}", "job_id": "${VAR_JOB_IDS[0]}", "depends_on": "${EMB_JOB_IDS[0]}"},
    {"slug": "${EMB_SLUGS[1]}", "job_id": "${VAR_JOB_IDS[1]}", "depends_on": "${EMB_JOB_IDS[1]}"},
    {"slug": "${EMB_SLUGS[2]}", "job_id": "${VAR_JOB_IDS[2]}", "depends_on": "${EMB_JOB_IDS[2]}"}
  ],
  "tc_llm": {
    "model": "Qwen/Qwen3-8B",
    "temperature": 0.6,
    "max_tokens_label_gen": 1024,
    "max_tokens_label_merge": 2048,
    "max_tokens_classify": 512,
    "batch_size": 64,
    "texts_per_label_batch": 15,
    "n_ranked_labels": 5,
    "thinking_mode": false,
    "job_id": "${TC_LLM_JOB_ID}"
  }
}
EOF
echo "Saved pipeline config to ${OUTPUT_DIR}/pipeline_config.json"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Pipeline submitted successfully!"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Job IDs:"
echo "  Embeddings:"
for i in 0 1 2; do
    echo "    ${EMB_SLUGS[$i]}: ${EMB_JOB_IDS[$i]}"
done
echo "  Variance:"
for i in 0 1 2; do
    echo "    ${EMB_SLUGS[$i]}: ${VAR_JOB_IDS[$i]} (after ${EMB_JOB_IDS[$i]})"
done
echo "  TC-LLM: ${TC_LLM_JOB_ID}"
echo ""
echo "Total: 7 jobs (3 embedding GPU, 3 variance CPU, 1 TC-LLM GPU)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
