"""
Precompute embedding variance for the visualization webapp.

Run on HPC with:
    sbatch hpc/compute_embedding_variance.slurm

Or locally with sufficient memory:
    uv run python scripts/compute_embedding_variance.py \
        --input logs/consistency-inference-1225210/embeddings.parquet \
        --output logs/consistency-inference-1225210/embedding_variance.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_total_variance(embeddings: list[list[float]]) -> float:
    """Compute total variance (trace of covariance matrix) for a set of embeddings.

    Args:
        embeddings: List of N embedding vectors, each of dimension D

    Returns:
        Trace of the covariance matrix (sum of variances along each dimension)
    """
    matrix = np.array(embeddings)  # Shape: (N, D)
    assert matrix.ndim == 2, f"Expected 2D matrix, got shape {matrix.shape}"
    # Use np.var with axis=0 to get per-dimension variance, then sum
    # ddof=1 for sample variance (unbiased estimator)
    return np.var(matrix, axis=0, ddof=1).sum()


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embedding variance for visualization"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to embeddings parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output variance parquet file",
    )
    args = parser.parse_args()

    print(f"Loading embeddings from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows")

    # Find embedding columns
    embedding_columns = [col for col in df.columns if col.startswith("embedding_")]
    print(f"Found {len(embedding_columns)} embedding columns: {embedding_columns}")

    # Get unique prompts with their questions (if available)
    if "question" in df.columns:
        prompts_df = df[["prompt_id", "question"]].drop_duplicates()
        prompt_to_question = dict(zip(prompts_df["prompt_id"], prompts_df["question"]))
    else:
        prompt_to_question = {}

    # Get persona to category mapping (if available)
    if "category" in df.columns:
        persona_to_category = dict(
            df[["persona", "category"]].drop_duplicates().values
        )
    else:
        persona_to_category = {}

    # Compute variance for each (prompt_id, persona, embedding_column) tuple
    results = []
    groups = list(df.groupby(["prompt_id", "persona"]))
    total_groups = len(groups)

    for i, ((prompt_id, persona), group) in enumerate(groups):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing group {i + 1}/{total_groups}: prompt={prompt_id}, persona={persona}")

        for embedding_col in embedding_columns:
            embeddings = group[embedding_col].tolist()
            variance = compute_total_variance(embeddings)
            result = {
                "prompt_id": prompt_id,
                "persona": persona,
                "embedding_column": embedding_col,
                "total_variance": variance,
                "n_samples": len(embeddings),
            }
            if persona_to_category:
                result["category"] = persona_to_category.get(persona, "unknown")
            results.append(result)

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Add question text if available
    if prompt_to_question:
        result_df["question"] = result_df["prompt_id"].map(prompt_to_question)

    # Save to parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(args.output, index=False)
    print(f"Saved {len(result_df):,} variance records to {args.output}")

    # Print summary
    print("\nSummary:")
    print(f"  Unique prompts: {result_df['prompt_id'].nunique()}")
    print(f"  Unique personas: {result_df['persona'].nunique()}")
    print(f"  Embedding columns: {len(embedding_columns)}")
    print(f"  Total records: {len(result_df)}")


if __name__ == "__main__":
    main()
