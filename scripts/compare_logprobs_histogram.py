#!/usr/bin/env python3
"""
Generate histogram plots comparing top-20 vs full logprobs entropy.
Saves plots as PNG files.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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
    return (result["persona"], result["prompt_id"], result["rep_idx"])


def main():
    # Load data
    vllm_path = Path("logs/personae-inference-1050719/results.json")
    transformers_path = Path("logs/personae-inference-1139848/results.jsonl")
    output_dir = Path("scripts/output")
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    vllm_results = load_results(vllm_path)
    transformers_results = load_results(transformers_path)

    vllm_by_key = {create_key(r): r for r in vllm_results}
    transformers_by_key = {create_key(r): r for r in transformers_results}
    common_keys = list(set(vllm_by_key.keys()) & set(transformers_by_key.keys()))

    # Separate identical vs different responses
    identical_diffs = []
    different_diffs = []
    identical_vllm = []
    identical_trans = []

    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]

        v_ent = v.get("avg_entropy")
        t_ent = t.get("avg_entropy")

        if v_ent is None or t_ent is None:
            continue

        diff = t_ent - v_ent

        if v["response"] == t["response"]:
            identical_diffs.append(diff)
            identical_vllm.append(v_ent)
            identical_trans.append(t_ent)
        else:
            different_diffs.append(diff)

    identical_diffs = np.array(identical_diffs)
    different_diffs = np.array(different_diffs)
    identical_vllm = np.array(identical_vllm)
    identical_trans = np.array(identical_trans)

    # Plot 1: Histogram of entropy differences (identical responses only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(identical_diffs, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", label="No difference")
    ax1.axvline(x=identical_diffs.mean(), color="green", linestyle="-",
                label=f"Mean: {identical_diffs.mean():.4f}")
    ax1.set_xlabel("Entropy Difference (Transformers - vLLM)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Entropy Underestimate (Identical Responses, n={len(identical_diffs)})")
    ax1.legend()

    # Plot 2: Scatter plot vLLM vs Transformers entropy
    ax2 = axes[1]
    ax2.scatter(identical_vllm, identical_trans, alpha=0.5, s=10)
    ax2.plot([0, 1.5], [0, 1.5], "r--", label="y = x (perfect agreement)")
    ax2.set_xlabel("vLLM Entropy (top-20 logprobs)")
    ax2.set_ylabel("Transformers Entropy (full logprobs)")
    ax2.set_title("Entropy Comparison (Identical Responses)")
    ax2.legend()
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_comparison_identical.png", dpi=150)
    print(f"Saved: {output_dir / 'entropy_comparison_identical.png'}")

    # Plot 3: Histogram comparing identical vs different responses
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(-0.3, 0.3, 61)
    ax.hist(identical_diffs, bins=bins, alpha=0.7, label=f"Identical responses (n={len(identical_diffs)})")
    ax.hist(different_diffs, bins=bins, alpha=0.5, label=f"Different responses (n={len(different_diffs)})")
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Entropy Difference (Transformers - vLLM)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Entropy Differences")
    ax.legend()
    ax.set_xlim(-0.3, 0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_diff_distribution.png", dpi=150)
    print(f"Saved: {output_dir / 'entropy_diff_distribution.png'}")

    # Plot 4: Relative error distribution
    rel_diff = identical_diffs / identical_trans * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rel_diff, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.axvline(x=rel_diff.mean(), color="green", linestyle="-",
               label=f"Mean: {rel_diff.mean():.2f}%")
    ax.set_xlabel("Relative Entropy Error (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Relative Entropy Underestimate (Identical Responses)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_relative_error.png", dpi=150)
    print(f"Saved: {output_dir / 'entropy_relative_error.png'}")

    # Summary stats table
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (IDENTICAL RESPONSES)")
    print("=" * 60)
    print(f"N pairs: {len(identical_diffs)}")
    print(f"\nAbsolute Difference (nats):")
    print(f"  Mean:   {identical_diffs.mean():.6f}")
    print(f"  Median: {np.median(identical_diffs):.6f}")
    print(f"  Std:    {identical_diffs.std():.6f}")
    print(f"  IQR:    [{np.percentile(identical_diffs, 25):.6f}, {np.percentile(identical_diffs, 75):.6f}]")
    print(f"\nRelative Error (%):")
    print(f"  Mean:   {rel_diff.mean():.2f}%")
    print(f"  Median: {np.median(rel_diff):.2f}%")
    print(f"  Std:    {rel_diff.std():.2f}%")
    print(f"  IQR:    [{np.percentile(rel_diff, 25):.2f}%, {np.percentile(rel_diff, 75):.2f}%]")

    plt.close("all")


if __name__ == "__main__":
    main()
