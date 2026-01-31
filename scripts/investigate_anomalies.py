#!/usr/bin/env python3
"""
Investigate anomalies where vLLM entropy > Transformers entropy for identical responses.
"""

import json
from pathlib import Path


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

    print("Loading results...")
    vllm_results = load_results(vllm_path)
    transformers_results = load_results(transformers_path)

    vllm_by_key = {create_key(r): r for r in vllm_results}
    transformers_by_key = {create_key(r): r for r in transformers_results}
    common_keys = list(set(vllm_by_key.keys()) & set(transformers_by_key.keys()))

    # Find anomalies: identical responses where vLLM entropy > Transformers entropy
    anomalies = []
    for key in common_keys:
        v = vllm_by_key[key]
        t = transformers_by_key[key]

        if v["response"] != t["response"]:
            continue

        v_ent = v.get("avg_entropy")
        t_ent = t.get("avg_entropy")

        if v_ent is None or t_ent is None:
            continue

        diff = t_ent - v_ent
        if diff < -0.001:  # vLLM > Transformers by at least 0.001
            anomalies.append((key, diff, v, t))

    print(f"Found {len(anomalies)} anomalies (identical response, vLLM entropy > Transformers)")

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF ANOMALIES")
    print("=" * 80)

    # Check if token counts differ
    token_count_differs = 0
    for key, diff, v, t in anomalies:
        if v["num_tokens"] != t["num_tokens"]:
            token_count_differs += 1

    print(f"\nToken count differs: {token_count_differs}/{len(anomalies)}")

    # Check think_end_position differences
    think_pos_differs = 0
    for key, diff, v, t in anomalies:
        if v.get("think_end_position") != t.get("think_end_position"):
            think_pos_differs += 1

    print(f"Think position differs: {think_pos_differs}/{len(anomalies)}")

    # Show a few examples in detail
    print("\n" + "=" * 80)
    print("EXAMPLE ANOMALIES")
    print("=" * 80)

    # Sort by severity
    anomalies.sort(key=lambda x: x[1])  # Most negative first

    for i, (key, diff, v, t) in enumerate(anomalies[:5]):
        print(f"\n{'='*60}")
        print(f"Anomaly #{i+1}: {key[0]} / Prompt {key[1]}")
        print(f"{'='*60}")
        print(f"vLLM entropy:        {v['avg_entropy']:.6f}")
        print(f"Transformers entropy: {t['avg_entropy']:.6f}")
        print(f"Difference (T-V):    {diff:.6f}")
        print(f"\nvLLM num_tokens:        {v['num_tokens']}")
        print(f"Transformers num_tokens: {t['num_tokens']}")
        print(f"\nvLLM think_end_pos:        {v.get('think_end_position')}")
        print(f"Transformers think_end_pos: {t.get('think_end_position')}")
        print(f"\nvLLM avg_entropy_thinking:        {v.get('avg_entropy_thinking')}")
        print(f"Transformers avg_entropy_thinking: {t.get('avg_entropy_thinking')}")
        print(f"\nvLLM avg_entropy_output:        {v.get('avg_entropy_output')}")
        print(f"Transformers avg_entropy_output: {t.get('avg_entropy_output')}")

        # Check if response text is truly identical
        v_resp = v["response"]
        t_resp = t["response"]
        print(f"\nResponse length - vLLM: {len(v_resp)}, Trans: {len(t_resp)}")
        print(f"Responses equal: {v_resp == t_resp}")

        if v_resp == t_resp:
            # They're truly identical, so why different entropy?
            print("\nResponses are byte-identical, but entropies differ!")
            print("This suggests differences in tokenization or logprob computation.")
        else:
            print(f"\nFirst diff at char: {next((i for i, (a, b) in enumerate(zip(v_resp, t_resp)) if a != b), None)}")

    # Hypothesis: token count difference
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)

    # For anomalies with same response text, is token count always different?
    same_text_diff_tokens = 0
    same_text_same_tokens = 0

    for key, diff, v, t in anomalies:
        if v["num_tokens"] != t["num_tokens"]:
            same_text_diff_tokens += 1
        else:
            same_text_same_tokens += 1

    print(f"\nAmong anomalies (identical response text):")
    print(f"  Different token count: {same_text_diff_tokens}")
    print(f"  Same token count:      {same_text_same_tokens}")

    if same_text_same_tokens > 0:
        print("\n  Cases with same token count (unexpected):")
        for key, diff, v, t in anomalies:
            if v["num_tokens"] == t["num_tokens"]:
                print(f"    {key[0]} / Prompt {key[1]}: tokens={v['num_tokens']}, diff={diff:.4f}")


if __name__ == "__main__":
    main()
