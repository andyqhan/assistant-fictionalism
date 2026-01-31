#!/usr/bin/env python3
"""
Compare metrics between top-20 logprobs (vLLM) and full logprobs (Transformers).

This script analyzes how much accuracy is lost by using only top-20 logprobs
for entropy computation versus using full vocabulary logprobs.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_results(path: Path) -> list[dict]:
    """Load results from either .json or .jsonl format."""
    if path.suffix == ".jsonl":
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    else:
        with open(path) as f:
            return json.load(f)


def create_key(result: dict) -> tuple:
    """Create a unique key for matching results across runs."""
    return (result["persona"], result["prompt_id"], result["rep_idx"])


def main():
    # Paths to results
    vllm_path = Path("logs/personae-inference-1050719/results.json")
    transformers_path = Path("logs/personae-inference-1139848/results.jsonl")

    print("Loading results...")
    vllm_results = load_results(vllm_path)
    transformers_results = load_results(transformers_path)

    print(f"  vLLM (top-20): {len(vllm_results)} results")
    print(f"  Transformers (full): {len(transformers_results)} results")

    # Index by key
    vllm_by_key = {create_key(r): r for r in vllm_results}
    transformers_by_key = {create_key(r): r for r in transformers_results}

    # Find matching results
    common_keys = set(vllm_by_key.keys()) & set(transformers_by_key.keys())
    print(f"  Matched results: {len(common_keys)}")

    if len(common_keys) == 0:
        print("ERROR: No matching results found!")
        return

    # Collect paired metrics
    metrics = [
        "avg_entropy",
        "avg_entropy_thinking",
        "avg_entropy_output",
        "avg_top_k_mass",
        "avg_top_k_mass_thinking",
        "avg_top_k_mass_output",
    ]

    paired_data = {m: {"vllm": [], "transformers": []} for m in metrics}

    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]

        for metric in metrics:
            v_val = v.get(metric)
            t_val = t.get(metric)
            if v_val is not None and t_val is not None:
                paired_data[metric]["vllm"].append(v_val)
                paired_data[metric]["transformers"].append(t_val)

    # Analyze each metric
    print("\n" + "=" * 80)
    print("COMPARISON: Top-20 Logprobs (vLLM) vs Full Logprobs (Transformers)")
    print("=" * 80)

    for metric in metrics:
        vllm_vals = np.array(paired_data[metric]["vllm"])
        trans_vals = np.array(paired_data[metric]["transformers"])

        if len(vllm_vals) == 0:
            print(f"\n{metric}: No valid pairs")
            continue

        # Compute differences
        diff = trans_vals - vllm_vals  # positive = transformers higher
        rel_diff = diff / trans_vals * 100  # relative difference as percentage

        # Correlation
        corr = np.corrcoef(vllm_vals, trans_vals)[0, 1]

        print(f"\n{metric} ({len(vllm_vals)} pairs)")
        print("-" * 60)
        print(f"  vLLM (top-20):     mean={vllm_vals.mean():.6f}, std={vllm_vals.std():.6f}")
        print(f"  Transformers:      mean={trans_vals.mean():.6f}, std={trans_vals.std():.6f}")
        print(f"  Difference (T-V):  mean={diff.mean():.6f}, std={diff.std():.6f}")
        print(f"  Relative diff:     mean={rel_diff.mean():.2f}%, std={rel_diff.std():.2f}%")
        print(f"  Correlation:       {corr:.6f}")
        print(f"  Max underestimate: {diff.max():.6f} ({rel_diff.max():.2f}%)")
        print(f"  Min underestimate: {diff.min():.6f} ({rel_diff.min():.2f}%)")

    # Summary interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    entropy_diff = np.array(paired_data["avg_entropy"]["transformers"]) - np.array(
        paired_data["avg_entropy"]["vllm"]
    )
    topk_diff = np.array(paired_data["avg_top_k_mass"]["transformers"]) - np.array(
        paired_data["avg_top_k_mass"]["vllm"]
    )

    print(f"""
Entropy:
  - Top-20 logprobs underestimates entropy by {entropy_diff.mean():.4f} nats on average
  - This is because entropy from top-20 = -sum(p_i * log(p_i)) for i in top-20 only
  - Missing tail probability contributes: sum(p_tail * log(p_tail))
  - Relative underestimate: {(entropy_diff / np.array(paired_data['avg_entropy']['transformers'])).mean() * 100:.2f}%

Top-5 Mass:
  - Difference in top-5 mass: {topk_diff.mean():.6f} on average
  - This should be ~0 since top-5 is contained within top-20
  - Any difference is likely due to different responses or numerical precision
""")


if __name__ == "__main__":
    main()
