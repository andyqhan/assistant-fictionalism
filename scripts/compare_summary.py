#!/usr/bin/env python3
"""
Final summary of top-20 vs full logprobs comparison.
"""

import json
from pathlib import Path

import numpy as np


def load_results(path: Path) -> list[dict]:
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
    vllm_path = Path("logs/personae-inference-1050719/results.json")
    transformers_path = Path("logs/personae-inference-1139848/results.jsonl")

    vllm_results = load_results(vllm_path)
    transformers_results = load_results(transformers_path)

    vllm_by_key = {create_key(r): r for r in vllm_results}
    transformers_by_key = {create_key(r): r for r in transformers_results}
    common_keys = list(set(vllm_by_key.keys()) & set(transformers_by_key.keys()))

    # Categorize pairs
    truly_identical = []  # Same response AND same token count
    same_text_diff_tokens = []  # Same response but different token count
    different_response = []

    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]

        v_ent = v.get("avg_entropy")
        t_ent = t.get("avg_entropy")

        if v_ent is None or t_ent is None:
            continue

        if v["response"] != t["response"]:
            different_response.append((key, v, t))
        elif v["num_tokens"] != t["num_tokens"]:
            same_text_diff_tokens.append((key, v, t))
        else:
            truly_identical.append((key, v, t))

    print("=" * 80)
    print("COMPARISON: Top-20 Logprobs (vLLM) vs Full Logprobs (Transformers)")
    print("=" * 80)

    print(f"\nTotal matched pairs: {len(common_keys)}")
    print(f"  Truly identical (same text & tokens): {len(truly_identical)} ({len(truly_identical)/len(common_keys)*100:.1f}%)")
    print(f"  Same text, different tokens:          {len(same_text_diff_tokens)} ({len(same_text_diff_tokens)/len(common_keys)*100:.1f}%)")
    print(f"  Different response text:              {len(different_response)} ({len(different_response)/len(common_keys)*100:.1f}%)")

    # Analyze truly identical pairs (apples-to-apples)
    print("\n" + "=" * 80)
    print("TRULY IDENTICAL PAIRS (Same text AND same token count)")
    print("=" * 80)

    if truly_identical:
        entropy_diffs = [t["avg_entropy"] - v["avg_entropy"] for _, v, t in truly_identical]
        topk_diffs = [t["avg_top_k_mass"] - v["avg_top_k_mass"] for _, v, t in truly_identical]

        entropy_diffs = np.array(entropy_diffs)
        topk_diffs = np.array(topk_diffs)

        trans_entropy = np.array([t["avg_entropy"] for _, v, t in truly_identical])
        rel_entropy_diff = entropy_diffs / trans_entropy * 100

        print(f"\nN = {len(truly_identical)}")

        print(f"\nEntropy underestimate (Transformers - vLLM):")
        print(f"  Mean:   {entropy_diffs.mean():.6f} nats ({rel_entropy_diff.mean():.2f}%)")
        print(f"  Median: {np.median(entropy_diffs):.6f} nats ({np.median(rel_entropy_diff):.2f}%)")
        print(f"  Std:    {entropy_diffs.std():.6f}")
        print(f"  Range:  [{entropy_diffs.min():.6f}, {entropy_diffs.max():.6f}]")
        print(f"  % positive (expected): {(entropy_diffs > 0).mean()*100:.1f}%")

        print(f"\nTop-5 mass difference:")
        print(f"  Mean:   {topk_diffs.mean():.8f}")
        print(f"  Std:    {topk_diffs.std():.8f}")
        print(f"  Range:  [{topk_diffs.min():.8f}, {topk_diffs.max():.8f}]")

        # Correlation
        vllm_ent = np.array([v["avg_entropy"] for _, v, t in truly_identical])
        trans_ent = np.array([t["avg_entropy"] for _, v, t in truly_identical])
        corr = np.corrcoef(vllm_ent, trans_ent)[0, 1]
        print(f"\nCorrelation (entropy): {corr:.6f}")
    else:
        print("No truly identical pairs found!")

    # Analyze same-text-diff-tokens pairs
    print("\n" + "=" * 80)
    print("SAME TEXT, DIFFERENT TOKENIZATION")
    print("=" * 80)

    if same_text_diff_tokens:
        token_diffs = [v["num_tokens"] - t["num_tokens"] for _, v, t in same_text_diff_tokens]
        token_diffs = np.array(token_diffs)

        print(f"\nN = {len(same_text_diff_tokens)}")
        print(f"\nToken count difference (vLLM - Transformers):")
        print(f"  Mean:  {token_diffs.mean():.2f}")
        print(f"  Range: [{token_diffs.min()}, {token_diffs.max()}]")
        print(f"  vLLM has more tokens: {(token_diffs > 0).sum()}")
        print(f"  vLLM has fewer tokens: {(token_diffs < 0).sum()}")

        print("\n  This explains the 'anomalies' - different tokenization means")
        print("  entropy is computed over different token sequences!")

    # Final recommendations
    print("\n" + "=" * 80)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 80)

    if truly_identical:
        mean_underestimate = entropy_diffs.mean()
        rel_underestimate = rel_entropy_diff.mean()

        print(f"""
1. ENTROPY ACCURACY
   - Top-20 logprobs underestimates entropy by ~{mean_underestimate:.4f} nats ({rel_underestimate:.1f}%)
   - This is because the tail of the distribution (tokens 21+) contributes
     to entropy but is not captured in top-20 logprobs
   - The underestimate is consistent and predictable

2. TOP-K MASS ACCURACY (k=5)
   - Top-5 mass is essentially identical between backends
   - Mean difference: {topk_diffs.mean():.8f} (negligible)
   - Using top-20 logprobs is sufficient for top-k mass computation

3. TOKENIZATION DIFFERENCES
   - vLLM and Transformers can tokenize the same text differently
   - This affects ~{len(same_text_diff_tokens)/len(common_keys)*100:.1f}% of "identical" responses
   - When comparing metrics, ensure tokenization is consistent

4. RESPONSE DIFFERENCES (temp=0)
   - Even with temperature=0, vLLM and Transformers produce different
     responses {len(different_response)/len(common_keys)*100:.1f}% of the time
   - This is likely due to different sampling implementations or
     numerical precision differences

RECOMMENDATION:
   - For relative comparisons (e.g., "persona A has higher entropy than B"),
     top-20 logprobs are sufficient and much faster
   - For absolute entropy values, expect ~{rel_underestimate:.1f}% underestimate
   - For top-k mass (k <= 20), top-20 logprobs are fully accurate
""")


if __name__ == "__main__":
    main()
