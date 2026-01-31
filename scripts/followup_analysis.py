#!/usr/bin/env python3
"""
Follow-up analysis answering specific questions.
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
    truly_identical = []
    same_text_diff_tokens = []
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

    # =========================================================================
    # Q1: Is the underestimate constant or is there wide variation?
    # =========================================================================
    print("=" * 80)
    print("Q1: IS THE ENTROPY UNDERESTIMATE CONSTANT?")
    print("=" * 80)

    if truly_identical:
        entropy_diffs = np.array([t["avg_entropy"] - v["avg_entropy"] for _, v, t in truly_identical])
        trans_entropy = np.array([t["avg_entropy"] for _, v, t in truly_identical])
        rel_diffs = entropy_diffs / trans_entropy * 100

        print(f"\nTruly identical pairs (n={len(truly_identical)}):")
        print(f"\nAbsolute underestimate (nats):")
        print(f"  Mean:   {entropy_diffs.mean():.6f}")
        print(f"  Std:    {entropy_diffs.std():.6f}")
        print(f"  CV:     {entropy_diffs.std() / entropy_diffs.mean() * 100:.1f}% (coefficient of variation)")
        print(f"  Min:    {entropy_diffs.min():.6f}")
        print(f"  Max:    {entropy_diffs.max():.6f}")
        print(f"  Range:  {entropy_diffs.max() - entropy_diffs.min():.6f}")

        print(f"\nRelative underestimate (%):")
        print(f"  Mean:   {rel_diffs.mean():.2f}%")
        print(f"  Std:    {rel_diffs.std():.2f}%")
        print(f"  Min:    {rel_diffs.min():.2f}%")
        print(f"  Max:    {rel_diffs.max():.2f}%")

        print(f"\nPercentiles of absolute underestimate:")
        for p in [10, 25, 50, 75, 90]:
            print(f"  {p}th: {np.percentile(entropy_diffs, p):.6f}")

        print(f"\nIndividual values (all {len(truly_identical)} pairs):")
        for i, (key, v, t) in enumerate(truly_identical):
            diff = t["avg_entropy"] - v["avg_entropy"]
            rel = diff / t["avg_entropy"] * 100
            print(f"  {i+1}. {key[0][:20]:20s} prompt={key[1]:3d}: "
                  f"diff={diff:.6f} ({rel:.1f}%), trans_ent={t['avg_entropy']:.4f}")

    # =========================================================================
    # Q2: Show a pair of responses that differ entirely
    # =========================================================================
    print("\n" + "=" * 80)
    print("Q2: EXAMPLE OF DIFFERING RESPONSES")
    print("=" * 80)

    # Pick a few interesting examples
    examples = different_response[:3]

    for i, (key, v, t) in enumerate(examples):
        print(f"\n{'─' * 80}")
        print(f"Example {i+1}: Persona='{key[0]}', Prompt ID={key[1]}")
        print(f"{'─' * 80}")

        print(f"\nPrompt: {v['prompt'][:100]}...")

        print(f"\n>>> vLLM Response ({v['num_tokens']} tokens):")
        print(v["response"][:500])
        if len(v["response"]) > 500:
            print(f"... [{len(v['response']) - 500} more chars]")

        print(f"\n>>> Transformers Response ({t['num_tokens']} tokens):")
        print(t["response"][:500])
        if len(t["response"]) > 500:
            print(f"... [{len(t['response']) - 500} more chars]")

        print(f"\nMetrics comparison:")
        print(f"  Entropy:     vLLM={v['avg_entropy']:.4f}, Trans={t['avg_entropy']:.4f}, diff={t['avg_entropy']-v['avg_entropy']:.4f}")
        print(f"  Top-5 mass:  vLLM={v['avg_top_k_mass']:.4f}, Trans={t['avg_top_k_mass']:.4f}, diff={t['avg_top_k_mass']-v['avg_top_k_mass']:.4f}")

    # =========================================================================
    # Q3: Metrics for non-perfect matches
    # =========================================================================
    print("\n" + "=" * 80)
    print("Q3: METRICS FOR NON-PERFECT MATCHES")
    print("=" * 80)

    # Same text, different tokens
    print(f"\n### Same text, different tokenization (n={len(same_text_diff_tokens)})")
    if same_text_diff_tokens:
        entropy_diffs = np.array([t["avg_entropy"] - v["avg_entropy"] for _, v, t in same_text_diff_tokens])
        topk_diffs = np.array([t["avg_top_k_mass"] - v["avg_top_k_mass"] for _, v, t in same_text_diff_tokens])
        vllm_ent = np.array([v["avg_entropy"] for _, v, t in same_text_diff_tokens])
        trans_ent = np.array([t["avg_entropy"] for _, v, t in same_text_diff_tokens])

        print(f"\nEntropy difference (Trans - vLLM):")
        print(f"  Mean:   {entropy_diffs.mean():.6f}")
        print(f"  Std:    {entropy_diffs.std():.6f}")
        print(f"  Min:    {entropy_diffs.min():.6f}")
        print(f"  Max:    {entropy_diffs.max():.6f}")
        print(f"  % positive: {(entropy_diffs > 0).mean()*100:.1f}%")
        print(f"  % negative: {(entropy_diffs < 0).mean()*100:.1f}%")

        print(f"\nTop-5 mass difference:")
        print(f"  Mean:   {topk_diffs.mean():.6f}")
        print(f"  Std:    {topk_diffs.std():.6f}")

        corr = np.corrcoef(vllm_ent, trans_ent)[0, 1]
        print(f"\nCorrelation (entropy): {corr:.6f}")

    # Different responses
    print(f"\n### Different responses (n={len(different_response)})")
    if different_response:
        entropy_diffs = np.array([t["avg_entropy"] - v["avg_entropy"] for _, v, t in different_response])
        topk_diffs = np.array([t["avg_top_k_mass"] - v["avg_top_k_mass"] for _, v, t in different_response])
        vllm_ent = np.array([v["avg_entropy"] for _, v, t in different_response])
        trans_ent = np.array([t["avg_entropy"] for _, v, t in different_response])

        print(f"\nEntropy difference (Trans - vLLM):")
        print(f"  Mean:   {entropy_diffs.mean():.6f}")
        print(f"  Std:    {entropy_diffs.std():.6f}")
        print(f"  Min:    {entropy_diffs.min():.6f}")
        print(f"  Max:    {entropy_diffs.max():.6f}")
        print(f"  % positive: {(entropy_diffs > 0).mean()*100:.1f}%")
        print(f"  % negative: {(entropy_diffs < 0).mean()*100:.1f}%")

        print(f"\nTop-5 mass difference:")
        print(f"  Mean:   {topk_diffs.mean():.6f}")
        print(f"  Std:    {topk_diffs.std():.6f}")

        corr = np.corrcoef(vllm_ent, trans_ent)[0, 1]
        print(f"\nCorrelation (entropy): {corr:.6f}")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"""
| Category                    | N      | Entropy Diff | Entropy Corr | Top-5 Diff |
|-----------------------------|--------|--------------|--------------|------------|""")

    if truly_identical:
        ed = np.array([t["avg_entropy"] - v["avg_entropy"] for _, v, t in truly_identical])
        td = np.array([t["avg_top_k_mass"] - v["avg_top_k_mass"] for _, v, t in truly_identical])
        ve = np.array([v["avg_entropy"] for _, v, t in truly_identical])
        te = np.array([t["avg_entropy"] for _, v, t in truly_identical])
        c = np.corrcoef(ve, te)[0, 1]
        print(f"| Truly identical             | {len(truly_identical):6d} | {ed.mean():+.6f}   | {c:.6f}     | {td.mean():+.8f} |")

    if same_text_diff_tokens:
        ed = np.array([t["avg_entropy"] - v["avg_entropy"] for _, v, t in same_text_diff_tokens])
        td = np.array([t["avg_top_k_mass"] - v["avg_top_k_mass"] for _, v, t in same_text_diff_tokens])
        ve = np.array([v["avg_entropy"] for _, v, t in same_text_diff_tokens])
        te = np.array([t["avg_entropy"] for _, v, t in same_text_diff_tokens])
        c = np.corrcoef(ve, te)[0, 1]
        print(f"| Same text, diff tokens      | {len(same_text_diff_tokens):6d} | {ed.mean():+.6f}   | {c:.6f}     | {td.mean():+.8f} |")

    if different_response:
        ed = np.array([t["avg_entropy"] - v["avg_entropy"] for _, v, t in different_response])
        td = np.array([t["avg_top_k_mass"] - v["avg_top_k_mass"] for _, v, t in different_response])
        ve = np.array([v["avg_entropy"] for _, v, t in different_response])
        te = np.array([t["avg_entropy"] for _, v, t in different_response])
        c = np.corrcoef(ve, te)[0, 1]
        print(f"| Different responses         | {len(different_response):6d} | {ed.mean():+.6f}   | {c:.6f}     | {td.mean():+.8f} |")


if __name__ == "__main__":
    main()
