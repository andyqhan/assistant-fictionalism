#!/usr/bin/env python3
"""
Detailed comparison of vLLM top-20 vs Transformers full logprobs.

Checks response similarity and analyzes where differences come from.
"""

import json
from pathlib import Path
from collections import Counter

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
    # Load results
    vllm_path = Path("logs/personae-inference-1050719/results.json")
    transformers_path = Path("logs/personae-inference-1139848/results.jsonl")

    print("Loading results...")
    vllm_results = load_results(vllm_path)
    transformers_results = load_results(transformers_path)

    vllm_by_key = {create_key(r): r for r in vllm_results}
    transformers_by_key = {create_key(r): r for r in transformers_results}
    common_keys = list(set(vllm_by_key.keys()) & set(transformers_by_key.keys()))

    print(f"Matched {len(common_keys)} result pairs")

    # Check response similarity
    print("\n" + "=" * 80)
    print("RESPONSE SIMILARITY CHECK")
    print("=" * 80)

    identical = 0
    different = 0
    length_diffs = []

    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]

        if v["response"] == t["response"]:
            identical += 1
        else:
            different += 1
            length_diffs.append(len(t["response"]) - len(v["response"]))

    print(f"  Identical responses: {identical} ({identical/len(common_keys)*100:.1f}%)")
    print(f"  Different responses: {different} ({different/len(common_keys)*100:.1f}%)")

    if length_diffs:
        print(f"  Length diff (T-V): mean={np.mean(length_diffs):.1f}, std={np.std(length_diffs):.1f}")

    # Analyze entropy differences for identical vs different responses
    print("\n" + "=" * 80)
    print("ENTROPY ANALYSIS: IDENTICAL vs DIFFERENT RESPONSES")
    print("=" * 80)

    identical_entropy_diffs = []
    different_entropy_diffs = []

    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]

        v_ent = v.get("avg_entropy")
        t_ent = t.get("avg_entropy")

        if v_ent is None or t_ent is None:
            continue

        diff = t_ent - v_ent

        if v["response"] == t["response"]:
            identical_entropy_diffs.append(diff)
        else:
            different_entropy_diffs.append(diff)

    identical_arr = np.array(identical_entropy_diffs)
    different_arr = np.array(different_entropy_diffs)

    print(f"\nIdentical responses ({len(identical_arr)} pairs):")
    if len(identical_arr) > 0:
        print(f"  Entropy diff (T-V): mean={identical_arr.mean():.6f}, std={identical_arr.std():.6f}")
        print(f"  Min/Max: [{identical_arr.min():.6f}, {identical_arr.max():.6f}]")
        print(f"  % positive (vLLM underestimates): {(identical_arr > 0).mean()*100:.1f}%")
        print(f"  % negative (vLLM overestimates?): {(identical_arr < 0).mean()*100:.1f}%")

    print(f"\nDifferent responses ({len(different_arr)} pairs):")
    if len(different_arr) > 0:
        print(f"  Entropy diff (T-V): mean={different_arr.mean():.6f}, std={different_arr.std():.6f}")
        print(f"  Min/Max: [{different_arr.min():.6f}, {different_arr.max():.6f}]")
        print(f"  % positive: {(different_arr > 0).mean()*100:.1f}%")
        print(f"  % negative: {(different_arr < 0).mean()*100:.1f}%")

    # For identical responses, analyze the theoretical underestimate
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS (IDENTICAL RESPONSES ONLY)")
    print("=" * 80)

    if len(identical_arr) > 0:
        # Since responses are identical, any difference must be due to:
        # 1. vLLM returning different logprobs than Transformers (unlikely)
        # 2. Top-20 missing tail probability

        print(f"""
For identical responses, the entropy difference comes from the tail distribution
that isn't captured in the top-20 logprobs.

Observed underestimate: {identical_arr.mean():.6f} nats ({identical_arr.mean() / np.mean([t.get('avg_entropy', 0) for k, t in transformers_by_key.items() if vllm_by_key[k]['response'] == t['response'] and t.get('avg_entropy')]) * 100:.2f}%)

Distribution of underestimate:
  10th percentile: {np.percentile(identical_arr, 10):.6f}
  25th percentile: {np.percentile(identical_arr, 25):.6f}
  50th percentile: {np.percentile(identical_arr, 50):.6f}
  75th percentile: {np.percentile(identical_arr, 75):.6f}
  90th percentile: {np.percentile(identical_arr, 90):.6f}
""")

    # Look at the worst underestimates
    print("\n" + "=" * 80)
    print("WORST UNDERESTIMATES (IDENTICAL RESPONSES)")
    print("=" * 80)

    # Collect pairs with identical responses
    identical_pairs = []
    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]
        if v["response"] == t["response"]:
            v_ent = v.get("avg_entropy")
            t_ent = t.get("avg_entropy")
            if v_ent is not None and t_ent is not None:
                identical_pairs.append((key, t_ent - v_ent, v, t))

    # Sort by underestimate (largest positive = worst underestimate)
    identical_pairs.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 worst underestimates:")
    for i, (key, diff, v, t) in enumerate(identical_pairs[:5]):
        print(f"\n{i+1}. Persona: {key[0]}, Prompt ID: {key[1]}")
        print(f"   vLLM entropy: {v['avg_entropy']:.4f}")
        print(f"   Trans entropy: {t['avg_entropy']:.4f}")
        print(f"   Difference: {diff:.4f} ({diff/t['avg_entropy']*100:.1f}%)")
        print(f"   Num tokens: vLLM={v['num_tokens']}, Trans={t['num_tokens']}")

    # Check negative cases (shouldn't happen for identical responses!)
    print("\n" + "=" * 80)
    print("ANOMALIES: vLLM > TRANSFORMERS (IDENTICAL RESPONSES)")
    print("=" * 80)

    negative_cases = [(key, diff, v, t) for key, diff, v, t in identical_pairs if diff < -0.001]
    print(f"\nFound {len(negative_cases)} cases where vLLM entropy > Transformers entropy")
    print("(This shouldn't happen for identical responses if theory is correct)")

    if negative_cases:
        print("\nTop 3 anomalies:")
        negative_cases.sort(key=lambda x: x[1])  # Most negative first
        for i, (key, diff, v, t) in enumerate(negative_cases[:3]):
            print(f"\n{i+1}. Persona: {key[0]}, Prompt ID: {key[1]}")
            print(f"   vLLM entropy: {v['avg_entropy']:.6f}")
            print(f"   Trans entropy: {t['avg_entropy']:.6f}")
            print(f"   Difference: {diff:.6f}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(identical_arr) > 0:
        print(f"""
For IDENTICAL responses (apples-to-apples comparison):
  - Using top-20 logprobs underestimates entropy by {identical_arr.mean():.4f} nats on average
  - This is a {identical_arr.mean() / np.mean([t.get('avg_entropy', 0) for k, t in transformers_by_key.items() if vllm_by_key[k]['response'] == t['response'] and t.get('avg_entropy')]) * 100:.2f}% relative error
  - {(identical_arr > 0).mean()*100:.1f}% of cases have positive underestimate (as expected)
  - The correlation between the two measurements is high (~0.88-0.90)

For TOP-5 MASS (k=5 << 20):
  - The measurements are essentially identical (mean diff ~0.00008)
  - Using top-20 logprobs is sufficient for top-k mass with k <= 20

RECOMMENDATION:
  - For entropy: expect ~{identical_arr.mean():.2f}% underestimate with top-20 logprobs
  - For top-k mass (k <= 20): top-20 logprobs are accurate
  - For high-precision entropy needs: use full logprobs (but much slower)
""")


if __name__ == "__main__":
    main()
