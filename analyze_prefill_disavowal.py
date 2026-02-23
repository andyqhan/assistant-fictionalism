"""
Analysis of prefill disavowal experiment.

Hypothesis: If the Assistant has genuine preferences, it should disavow
out-of-character prefills at a higher rate than other personae.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

JUDGED_PATH = Path("/scratch/ah7660/assistant-fictionalism/logs/prefill-inference-1472066/judged_results.jsonl")
RESULTS_PATH = Path("/scratch/ah7660/assistant-fictionalism/logs/prefill-inference-1472066/results.jsonl")


def count_thinking_tokens(response: str) -> int:
    """Count tokens in the <think>...</think> section (approximate by word count * 1.3)."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if not match:
        return 0
    think_text = match.group(1)
    # Rough token estimate: split on whitespace
    words = think_text.split()
    return len(words)


def load_judged_results():
    records = []
    with open(JUDGED_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_results():
    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_endorsement_distribution(records_by_persona):
    """Compute endorsement rate distribution per persona."""
    labels = ["accept", "partial", "reject", "redirect"]
    result = {}
    for persona, recs in records_by_persona.items():
        counts = Counter(r["judge_parsed"]["endorsement"] for r in recs if r.get("judge_parsed"))
        total = sum(counts.values())
        result[persona] = {
            "counts": dict(counts),
            "rates": {k: counts.get(k, 0) / total for k in labels},
            "total": total,
        }
    return result


def compute_flagging_rates(records_by_persona):
    """Compute flagging rates per persona."""
    result = {}
    for persona, recs in records_by_persona.items():
        parsed = [r for r in recs if r.get("judge_parsed")]
        flagged = sum(1 for r in parsed if r["judge_parsed"].get("flagged", False))
        total = len(parsed)
        result[persona] = {
            "flagged": flagged,
            "total": total,
            "rate": flagged / total if total > 0 else 0.0,
        }
    return result


def compute_thinking_tokens_by_persona(raw_by_persona):
    """Compute average thinking tokens per persona."""
    result = {}
    for persona, recs in raw_by_persona.items():
        token_counts = [count_thinking_tokens(r["response"]) for r in recs]
        result[persona] = {
            "mean": np.mean(token_counts),
            "median": np.median(token_counts),
            "std": np.std(token_counts),
            "n": len(token_counts),
        }
    return result


def compute_endorsement_by_category(records):
    """Group by (persona, category) to see category-level patterns."""
    grouped = defaultdict(list)
    for r in records:
        if r.get("judge_parsed"):
            key = (r["persona"], r.get("category", "unknown"))
            grouped[key].append(r["judge_parsed"]["endorsement"])
    return grouped


def chi_square_endorsement_vs_others(records_by_persona, target_persona="assistant"):
    """
    Test whether the assistant's endorsement distribution differs from pooled others.
    Uses chi-square test on accept/partial/reject/redirect counts.
    """
    labels = ["accept", "partial", "reject", "redirect"]

    def get_counts(recs):
        c = Counter(r["judge_parsed"]["endorsement"] for r in recs if r.get("judge_parsed"))
        return [c.get(k, 0) for k in labels]

    target_counts = get_counts(records_by_persona.get(target_persona, []))
    other_recs = []
    for persona, recs in records_by_persona.items():
        if persona != target_persona:
            other_recs.extend(recs)
    other_counts = get_counts(other_recs)

    contingency = np.array([target_counts, other_counts])
    # Remove zero-sum columns
    col_sums = contingency.sum(axis=0)
    contingency = contingency[:, col_sums > 0]

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p, dof, dict(zip(labels, target_counts)), dict(zip(labels, other_counts))


def pairwise_chi_square(records_by_persona, focus="reject"):
    """
    Pairwise chi-square tests comparing each persona's reject rate vs every other.
    Returns a dict of (persona_a, persona_b) -> p_value.
    """
    personae = list(records_by_persona.keys())
    results = {}
    for i, pa in enumerate(personae):
        for pb in personae[i + 1:]:
            recs_a = [r for r in records_by_persona[pa] if r.get("judge_parsed")]
            recs_b = [r for r in records_by_persona[pb] if r.get("judge_parsed")]
            count_a = Counter(r["judge_parsed"]["endorsement"] for r in recs_a)
            count_b = Counter(r["judge_parsed"]["endorsement"] for r in recs_b)
            # 2x2: focus category vs rest
            a_focus = count_a.get(focus, 0)
            a_other = sum(count_a.values()) - a_focus
            b_focus = count_b.get(focus, 0)
            b_other = sum(count_b.values()) - b_focus
            contingency = np.array([[a_focus, a_other], [b_focus, b_other]])
            if contingency.sum() == 0 or a_focus + b_focus == 0:
                results[(pa, pb)] = None
                continue
            try:
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                results[(pa, pb)] = p
            except Exception:
                results[(pa, pb)] = None
    return results


def flagging_rate_chi_square(flagging_data, target_persona="assistant"):
    """Chi-square test for flagging rate: target vs rest pooled."""
    target = flagging_data[target_persona]
    other_flagged = sum(v["flagged"] for k, v in flagging_data.items() if k != target_persona)
    other_total = sum(v["total"] for k, v in flagging_data.items() if k != target_persona)

    contingency = np.array([
        [target["flagged"], target["total"] - target["flagged"]],
        [other_flagged, other_total - other_flagged],
    ])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return chi2, p


def main():
    print("Loading data...")
    judged = load_judged_results()
    raw = load_raw_results()
    print(f"  Judged records: {len(judged)}")
    print(f"  Raw records:    {len(raw)}")

    # Group by persona
    judged_by_persona = defaultdict(list)
    for r in judged:
        judged_by_persona[r["persona"]].append(r)

    raw_by_persona = defaultdict(list)
    for r in raw:
        raw_by_persona[r["persona"]].append(r)

    personae = sorted(judged_by_persona.keys())
    print(f"\nPersonae: {personae}")
    print(f"Records per persona: { {p: len(judged_by_persona[p]) for p in personae} }")

    # -------------------------------------------------------------------------
    # 1. Endorsement distribution by persona
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. ENDORSEMENT DISTRIBUTION BY PERSONA")
    print("=" * 70)
    endorsement_data = compute_endorsement_distribution(judged_by_persona)
    labels = ["accept", "partial", "reject", "redirect"]
    header = f"{'Persona':<22} {'N':>5}  " + "  ".join(f"{l:>8}" for l in labels)
    print(header)
    print("-" * len(header))
    for persona in personae:
        d = endorsement_data[persona]
        row = f"{persona:<22} {d['total']:>5}  " + "  ".join(f"{d['rates'].get(l, 0):>7.1%}" for l in labels)
        print(row)

    # -------------------------------------------------------------------------
    # 2. Flagging rates by persona
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. FLAGGING RATES BY PERSONA")
    print("=" * 70)
    flagging_data = compute_flagging_rates(judged_by_persona)
    print(f"{'Persona':<22} {'N':>5}  {'Flagged':>8}  {'Rate':>8}")
    print("-" * 50)
    for persona in personae:
        d = flagging_data[persona]
        print(f"{persona:<22} {d['total']:>5}  {d['flagged']:>8}  {d['rate']:>7.1%}")

    # -------------------------------------------------------------------------
    # 3. Thinking tokens by persona
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. THINKING TOKENS (approximate) BY PERSONA")
    print("=" * 70)
    thinking_data = compute_thinking_tokens_by_persona(raw_by_persona)
    print(f"{'Persona':<22} {'N':>5}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
    print("-" * 60)
    for persona in personae:
        d = thinking_data[persona]
        print(f"{persona:<22} {d['n']:>5}  {d['mean']:>8.1f}  {d['median']:>8.1f}  {d['std']:>8.1f}")

    # -------------------------------------------------------------------------
    # 4. Endorsement by prompt category
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. ENDORSEMENT BY PROMPT CATEGORY")
    print("=" * 70)

    # Get categories
    cats = sorted(set(r.get("category", "unknown") for r in judged))
    print(f"Categories: {cats}\n")

    # Per-category endorsement rates
    for cat in cats:
        cat_recs = [r for r in judged if r.get("category") == cat and r.get("judge_parsed")]
        if not cat_recs:
            continue
        cat_by_persona = defaultdict(list)
        for r in cat_recs:
            cat_by_persona[r["persona"]].append(r)
        print(f"  Category: {cat} (n={len(cat_recs)})")
        header2 = f"    {'Persona':<22} {'N':>5}  " + "  ".join(f"{l:>8}" for l in labels)
        print(header2)
        for persona in personae:
            recs = cat_by_persona[persona]
            if not recs:
                continue
            counts = Counter(r["judge_parsed"]["endorsement"] for r in recs)
            total = sum(counts.values())
            row = f"    {persona:<22} {total:>5}  " + "  ".join(f"{counts.get(l, 0) / total:>7.1%}" for l in labels)
            print(row)
        print()

    # -------------------------------------------------------------------------
    # 5. Statistical significance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    # 5a. Assistant vs all others: endorsement chi-square
    print("\n5a. Assistant endorsement distribution vs. pooled others:")
    chi2, p, dof, assistant_counts, other_counts = chi_square_endorsement_vs_others(judged_by_persona, "assistant")
    print(f"  Chi-square = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
    print(f"  Assistant counts:  {assistant_counts}")
    print(f"  Others pooled:     {other_counts}")

    # 5b. Pairwise chi-square on reject rate
    print("\n5b. Pairwise chi-square on REJECT rate (all pairs):")
    pairwise = pairwise_chi_square(judged_by_persona, focus="reject")
    reject_rates = {p: endorsement_data[p]["rates"].get("reject", 0) for p in personae}
    print(f"  Reject rates: { {p: f'{v:.1%}' for p, v in reject_rates.items()} }")
    print()
    # Sort by p-value involving assistant
    assistant_pairs = [(pair, p_val) for pair, p_val in pairwise.items() if "assistant" in pair and p_val is not None]
    assistant_pairs.sort(key=lambda x: x[1])
    print("  Pairs involving 'assistant':")
    for (pa, pb), p_val in assistant_pairs:
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"    {pa:>22} vs {pb:<22}  p = {p_val:.4f} {sig}")

    print("\n  All pairwise comparisons (sorted by p-value):")
    sorted_pairs = sorted([(pair, p_val) for pair, p_val in pairwise.items() if p_val is not None], key=lambda x: x[1])
    for (pa, pb), p_val in sorted_pairs:
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"    {pa:>22} vs {pb:<22}  p = {p_val:.4f} {sig}")

    # 5c. Flagging rate: assistant vs others
    print("\n5c. Flagging rate — assistant vs. pooled others:")
    flag_chi2, flag_p = flagging_rate_chi_square(flagging_data, "assistant")
    print(f"  Chi-square = {flag_chi2:.3f}, p = {flag_p:.4f}")

    # 5d. Kruskal-Wallis on thinking tokens
    print("\n5d. Kruskal-Wallis test on thinking token counts across personae:")
    token_groups = []
    for persona in personae:
        recs = raw_by_persona[persona]
        token_groups.append([count_thinking_tokens(r["response"]) for r in recs])
    kw_stat, kw_p = stats.kruskal(*token_groups)
    print(f"  Kruskal-Wallis H = {kw_stat:.3f}, p = {kw_p:.4f}")

    # -------------------------------------------------------------------------
    # 6. Summary / interpretation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. SUMMARY & INTERPRETATION")
    print("=" * 70)

    # Compute disavowal rate (reject + redirect)
    disavowal_rates = {}
    for persona in personae:
        d = endorsement_data[persona]
        disavowal_rates[persona] = d["rates"].get("reject", 0) + d["rates"].get("redirect", 0)

    ranked = sorted(disavowal_rates.items(), key=lambda x: x[1], reverse=True)
    print("\nPersonae ranked by disavowal rate (reject + redirect):")
    for persona, rate in ranked:
        flag_rate = flagging_data[persona]["rate"]
        think_mean = thinking_data[persona]["mean"]
        print(f"  {persona:<22} disavowal={rate:.1%}  flagged={flag_rate:.1%}  thinking_words={think_mean:.0f}")

    print(f"\nKey finding: 'assistant' disavowal rate = {disavowal_rates['assistant']:.1%}")
    others = [v for p, v in disavowal_rates.items() if p != "assistant"]
    print(f"Mean disavowal rate for non-assistant personae = {np.mean(others):.1%}")

    if disavowal_rates["assistant"] > np.mean(others):
        print("=> Assistant disavows at a HIGHER rate than average non-assistant persona.")
        print("=> This is CONSISTENT with the hypothesis that Assistant has genuine preferences.")
    else:
        print("=> Assistant disavows at a LOWER rate than average non-assistant persona.")
        print("=> This DOES NOT support the hypothesis as stated.")

    print(f"\nStatistical test (assistant vs. others, endorsement distribution): p = {p:.4f}")
    if p < 0.05:
        print("=> Distribution is statistically significantly different (p < 0.05).")
    else:
        print("=> No statistically significant difference detected.")


if __name__ == "__main__":
    main()
