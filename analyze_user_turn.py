"""
Analysis script for user-turn prediction experiment.

Hypothesis: If the Assistant persona is genuinely special to the model,
it should be MORE CONFIDENT when predicting USER tokens in the user turn
when the user persona is "assistant" vs. other personae.

Key metrics: avg_entropy (lower = more confident), avg_top_k_mass (higher = more confident)
"""
import json
import numpy as np
from collections import defaultdict

RESULTS_PATH = "/scratch/ah7660/assistant-fictionalism/logs/user-turn-prediction-1930352/results.jsonl"
HEURISTIC_TAGS_PATH = "/scratch/ah7660/assistant-fictionalism/logs/user-turn-prediction-1930352/heuristic_tags.jsonl"
CLASSIFIER_PATH = "/scratch/ah7660/assistant-fictionalism/logs/response-classifier-2342906/classified_responses.jsonl"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load all data
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("USER-TURN PREDICTION EXPERIMENT — ANALYSIS")
print("=" * 70)

results = {}
with open(RESULTS_PATH) as f:
    for line in f:
        r = json.loads(line)
        key = (r["persona"], r["prompt_id"], r["sample_idx"])
        results[key] = r

tags = {}
with open(HEURISTIC_TAGS_PATH) as f:
    for line in f:
        t = json.loads(line)
        key = (t["persona"], t["prompt_id"], t["sample_idx"])
        tags[key] = t

classifier_rows = []
with open(CLASSIFIER_PATH) as f:
    for line in f:
        c = json.loads(line)
        classifier_rows.append(c)

print(f"\nLoaded {len(results)} result rows")
print(f"Loaded {len(tags)} heuristic-tag rows")
print(f"Loaded {len(classifier_rows)} classifier rows")

# Get all personas and their categories
persona_to_category = {}
for r in results.values():
    persona_to_category[r["persona"]] = r["category"]

print("\nPersonae in experiment:")
for persona, cat in sorted(persona_to_category.items()):
    print(f"  {persona!r:30s} (category={cat!r})")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Compute degenerate-response fractions by persona
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("DEGENERATE RESPONSE RATES")
print("=" * 70)
print("(A response is degenerate if it has repetitive_loop OR intro_echo OR max_tokens_hit)")

def is_degenerate_r1(tag_row):
    return (
        tag_row["response1_repetitive_loop"]
        or tag_row["response1_intro_echo"]
        or tag_row["response1_max_tokens_hit"]
    )

def is_degenerate_r2(tag_row):
    return (
        tag_row["response2_repetitive_loop"]
        or tag_row["response2_intro_echo"]
        or tag_row["response2_max_tokens_hit"]
    )

# Count by persona
degen_counts = defaultdict(lambda: {"r1_degen": 0, "r2_degen": 0, "r1_loop": 0, "r2_loop": 0,
                                     "r1_echo": 0, "r2_echo": 0, "r1_maxtok": 0, "r2_maxtok": 0, "total": 0})
for key, t in tags.items():
    persona = t["persona"]
    degen_counts[persona]["total"] += 1
    if t["response1_repetitive_loop"]: degen_counts[persona]["r1_loop"] += 1
    if t["response2_repetitive_loop"]: degen_counts[persona]["r2_loop"] += 1
    if t["response1_intro_echo"]:      degen_counts[persona]["r1_echo"] += 1
    if t["response2_intro_echo"]:      degen_counts[persona]["r2_echo"] += 1
    if t["response1_max_tokens_hit"]:  degen_counts[persona]["r1_maxtok"] += 1
    if t["response2_max_tokens_hit"]:  degen_counts[persona]["r2_maxtok"] += 1
    if is_degenerate_r1(t):            degen_counts[persona]["r1_degen"] += 1
    if is_degenerate_r2(t):            degen_counts[persona]["r2_degen"] += 1

print(f"\n{'Persona':<25} {'Cat':<22} {'N':>4}  {'R1 degen%':>9}  {'R2 degen%':>9}  {'R1 loop%':>8}  {'R2 loop%':>8}  {'R1 maxtok%':>10}  {'R2 maxtok%':>10}")
print("-" * 120)
for persona in sorted(degen_counts.keys()):
    d = degen_counts[persona]
    N = d["total"]
    cat = persona_to_category[persona]
    r1d = 100 * d["r1_degen"] / N
    r2d = 100 * d["r2_degen"] / N
    r1l = 100 * d["r1_loop"] / N
    r2l = 100 * d["r2_loop"] / N
    r1m = 100 * d["r1_maxtok"] / N
    r2m = 100 * d["r2_maxtok"] / N
    print(f"  {persona:<23} {cat:<22} {N:>4}  {r1d:>9.1f}  {r2d:>9.1f}  {r1l:>8.1f}  {r2l:>8.1f}  {r1m:>10.1f}  {r2m:>10.1f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Compute entropy/top-k-mass, excluding degenerate responses
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ENTROPY & TOP-K-MASS BY PERSONA (excluding degenerate responses)")
print("=" * 70)
print("Metrics are for the USER-TURN tokens predicted by the model.")
print("Lower entropy = higher confidence; Higher top_k_mass = higher confidence.")

# Collect stats for turn 1 (response1) and turn 2 (response2)
persona_metrics = defaultdict(lambda: {
    "r1_entropy": [], "r2_entropy": [],
    "r1_topk": [], "r2_topk": [],
    "combined_entropy": [], "combined_topk": []
})

for key, r in results.items():
    tag = tags.get(key)
    if tag is None:
        continue
    persona = r["persona"]

    # Turn 1 (response1)
    if not is_degenerate_r1(tag):
        e1 = r.get("response1_avg_entropy")
        t1 = r.get("response1_avg_top_k_mass")
        if e1 is not None:
            persona_metrics[persona]["r1_entropy"].append(e1)
            persona_metrics[persona]["combined_entropy"].append(e1)
        if t1 is not None:
            persona_metrics[persona]["r1_topk"].append(t1)
            persona_metrics[persona]["combined_topk"].append(t1)

    # Turn 2 (response2)
    if not is_degenerate_r2(tag):
        e2 = r.get("response2_avg_entropy")
        t2 = r.get("response2_avg_top_k_mass")
        if e2 is not None:
            persona_metrics[persona]["r2_entropy"].append(e2)
            persona_metrics[persona]["combined_entropy"].append(e2)
        if t2 is not None:
            persona_metrics[persona]["r2_topk"].append(t2)
            persona_metrics[persona]["combined_topk"].append(t2)

def stats(vals):
    if not vals:
        return "N/A"
    a = np.array(vals)
    return f"{a.mean():.4f} ± {a.std():.4f} (N={len(vals)})"

print(f"\n{'Persona':<25} {'Category':<22}  {'Turn1 Entropy':>22}  {'Turn2 Entropy':>22}  {'Combined Entropy':>25}")
print("-" * 120)
for persona in sorted(persona_metrics.keys()):
    d = persona_metrics[persona]
    cat = persona_to_category[persona]
    print(f"  {persona:<23} {cat:<22}  {stats(d['r1_entropy']):>22}  {stats(d['r2_entropy']):>22}  {stats(d['combined_entropy']):>25}")

print(f"\n{'Persona':<25} {'Category':<22}  {'Turn1 TopK':>22}  {'Turn2 TopK':>22}  {'Combined TopK':>25}")
print("-" * 120)
for persona in sorted(persona_metrics.keys()):
    d = persona_metrics[persona]
    cat = persona_to_category[persona]
    print(f"  {persona:<23} {cat:<22}  {stats(d['r1_topk']):>22}  {stats(d['r2_topk']):>22}  {stats(d['combined_topk']):>25}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Rank personas by confidence (combined entropy)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("CONFIDENCE RANKING BY PERSONA (combined entropy, lower = more confident)")
print("=" * 70)

ranked = []
for persona, d in persona_metrics.items():
    if d["combined_entropy"]:
        cat = persona_to_category[persona]
        mean_ent = np.mean(d["combined_entropy"])
        mean_topk = np.mean(d["combined_topk"]) if d["combined_topk"] else float("nan")
        ranked.append((persona, cat, mean_ent, mean_topk, len(d["combined_entropy"])))

ranked.sort(key=lambda x: x[2])  # sort by entropy ascending (most confident first)
print(f"\n  Rank  {'Persona':<25} {'Category':<22}  {'Avg Entropy':>12}  {'Avg TopK Mass':>13}  {'N':>6}")
print("  " + "-" * 90)
for i, (persona, cat, ent, topk, n) in enumerate(ranked, 1):
    marker = " <-- ASSISTANT" if cat == "assistant" else ""
    print(f"  {i:>4}  {persona:<25} {cat:<22}  {ent:>12.4f}  {topk:>13.4f}  {n:>6}{marker}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Hypothesis test: is "assistant" persona more confident than others?
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("HYPOTHESIS TEST: Is 'assistant' user-turn more confident than others?")
print("=" * 70)

from scipy import stats as scipy_stats

assistant_entropy = np.array(persona_metrics["assistant"]["combined_entropy"])
helper_entropy = np.array(persona_metrics["helper"]["combined_entropy"])
other_entropies = {}
for persona, d in persona_metrics.items():
    if persona not in ("assistant", "helper"):
        other_entropies[persona] = np.array(d["combined_entropy"])

all_other = np.concatenate([v for v in other_entropies.values()])
non_assistant_synonyms = np.concatenate([v for p, v in other_entropies.items()])

print(f"\n  assistant:    mean entropy = {assistant_entropy.mean():.4f}, N = {len(assistant_entropy)}")
print(f"  helper:       mean entropy = {helper_entropy.mean():.4f}, N = {len(helper_entropy)}")

for persona, vals in sorted(other_entropies.items()):
    print(f"  {persona:<20}: mean entropy = {vals.mean():.4f}, N = {len(vals)}")

print(f"\n  All non-assistant: mean entropy = {all_other.mean():.4f}, N = {len(all_other)}")

# Mann-Whitney U test: is assistant entropy distribution lower than all others?
u_stat, p_val = scipy_stats.mannwhitneyu(
    assistant_entropy, all_other, alternative="less"
)
print(f"\n  Mann-Whitney U test (assistant < all-other):")
print(f"    U = {u_stat:.1f}, p = {p_val:.6f}")
if p_val < 0.05:
    print("    => SIGNIFICANT: assistant is more confident (lower entropy) than others (p < 0.05)")
else:
    print("    => NOT significant")

# Also test against non-assistant-synonym group only
u_stat2, p_val2 = scipy_stats.mannwhitneyu(
    assistant_entropy, non_assistant_synonyms, alternative="less"
)
print(f"\n  Mann-Whitney U test (assistant < non-assistant-synonyms):")
print(f"    U = {u_stat2:.1f}, p = {p_val2:.6f}")

# Test assistant vs helper (both assistant-like)
u_stat3, p_val3 = scipy_stats.mannwhitneyu(
    assistant_entropy, helper_entropy, alternative="less"
)
print(f"\n  Mann-Whitney U test (assistant < helper):")
print(f"    U = {u_stat3:.1f}, p = {p_val3:.6f}")

# Effect size: Cohen's d
def cohens_d(a, b):
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

d_effect = cohens_d(assistant_entropy, all_other)
print(f"\n  Cohen's d (assistant vs all-other entropy): {d_effect:.4f}")
print(f"  (negative d means assistant has lower entropy = more confident)")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Classifier results: in-character vs meta-commentary rates
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RESPONSE CLASSIFIER RESULTS: In-character vs Meta-commentary rates")
print("=" * 70)
print("(Based on LLM classifier on response text)")

classifier_by_persona = defaultdict(lambda: defaultdict(int))
classifier_total = defaultdict(lambda: {"parsed": 0, "null": 0, "total": 0})

for row in classifier_rows:
    persona = row["persona"]
    parsed = row.get("classifier_parsed")
    classifier_total[persona]["total"] += 1
    if parsed is None:
        classifier_total[persona]["null"] += 1
    else:
        classifier_total[persona]["parsed"] += 1
        primary = parsed.get("primary", "unknown")
        classifier_by_persona[persona][primary] += 1

categories_seen = set()
for d in classifier_by_persona.values():
    categories_seen.update(d.keys())
categories_seen = sorted(categories_seen)

print(f"\n{'Persona':<25} {'Category':<22}  {'Total':>6}  {'%Parsed':>8}", end="")
for cat in categories_seen:
    print(f"  {cat[:15]:>15}", end="")
print()
print("-" * (70 + 17 * len(categories_seen)))

for persona in sorted(classifier_by_persona.keys()):
    d = classifier_by_persona[persona]
    tot = classifier_total[persona]
    cat = persona_to_category[persona]
    n_parsed = tot["parsed"]
    pct_parsed = 100 * n_parsed / tot["total"] if tot["total"] > 0 else 0
    print(f"  {persona:<23} {cat:<22}  {tot['total']:>6}  {pct_parsed:>8.1f}", end="")
    for c in categories_seen:
        count = d.get(c, 0)
        pct = 100 * count / n_parsed if n_parsed > 0 else 0
        print(f"  {pct:>15.1f}", end="")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Cross-analysis: classifier category vs entropy
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("CROSS-ANALYSIS: Entropy by classifier category (across all personae)")
print("=" * 70)

# Build a lookup from (persona, prompt_id, sample_idx, turn) -> result metrics
result_by_key = {}
for key, r in results.items():
    result_by_key[(r["persona"], r["prompt_id"], r["sample_idx"])] = r

entropy_by_classification = defaultdict(list)
topk_by_classification = defaultdict(list)

for row in classifier_rows:
    parsed = row.get("classifier_parsed")
    if parsed is None:
        continue
    rkey = (row["persona"], row["prompt_id"], row["sample_idx"])
    turn = row["turn"]
    tag = tags.get(rkey)
    rdata = result_by_key.get(rkey)
    if rdata is None or tag is None:
        continue
    primary = parsed.get("primary", "unknown")
    if turn == 1:
        if not is_degenerate_r1(tag):
            e = rdata.get("response1_avg_entropy")
            t = rdata.get("response1_avg_top_k_mass")
    else:
        if not is_degenerate_r2(tag):
            e = rdata.get("response2_avg_entropy")
            t = rdata.get("response2_avg_top_k_mass")
    if e is not None:
        entropy_by_classification[primary].append(e)
    if t is not None:
        topk_by_classification[primary].append(t)

print(f"\n{'Classification':<20}  {'N':>6}  {'Mean Entropy':>13}  {'Std Entropy':>12}  {'Mean TopK':>10}  {'Std TopK':>10}")
print("-" * 80)
for cls in sorted(entropy_by_classification.keys()):
    e_vals = np.array(entropy_by_classification[cls])
    t_vals = np.array(topk_by_classification[cls])
    print(f"  {cls:<18}  {len(e_vals):>6}  {e_vals.mean():>13.4f}  {e_vals.std():>12.4f}  {t_vals.mean():>10.4f}  {t_vals.std():>10.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Classifier in-character rates by persona (normalized)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("IN-CHARACTER RATE BY PERSONA (% of parsed responses that are in-character)")
print("=" * 70)

ic_ranked = []
for persona in sorted(classifier_by_persona.keys()):
    d = classifier_by_persona[persona]
    n_parsed = classifier_total[persona]["parsed"]
    n_ic = d.get("in-character", 0)
    n_meta = d.get("meta-commentary", 0)
    n_safety = d.get("safety-override", 0)
    n_harmful = d.get("harmful-generation", 0)
    pct_ic = 100 * n_ic / n_parsed if n_parsed > 0 else 0
    cat = persona_to_category[persona]
    ic_ranked.append((persona, cat, pct_ic, n_ic, n_meta, n_safety, n_parsed))

ic_ranked.sort(key=lambda x: -x[2])
print(f"\n  Rank  {'Persona':<25} {'Category':<22}  {'%In-Char':>9}  {'%Meta':>9}  {'%Safety':>9}  {'N parsed':>9}")
print("  " + "-" * 90)
for i, (persona, cat, pct_ic, n_ic, n_meta, n_safety, n_parsed) in enumerate(ic_ranked, 1):
    pct_meta = 100 * n_meta / n_parsed if n_parsed > 0 else 0
    pct_safety = 100 * n_safety / n_parsed if n_parsed > 0 else 0
    marker = " <-- ASSISTANT" if cat == "assistant" else ""
    print(f"  {i:>4}  {persona:<25} {cat:<22}  {pct_ic:>9.1f}  {pct_meta:>9.1f}  {pct_safety:>9.1f}  {n_parsed:>9}{marker}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. Summary and interpretation
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY & INTERPRETATION")
print("=" * 70)

# Collect the key numbers
best_persona, best_cat, best_ent, best_topk, _ = ranked[0]
worst_persona, worst_cat, worst_ent, worst_topk, _ = ranked[-1]
assistant_ent = np.mean(persona_metrics["assistant"]["combined_entropy"])
assistant_topk = np.mean(persona_metrics["assistant"]["combined_topk"])
assistant_rank = next(i for i, (p,*_) in enumerate(ranked, 1) if p == "assistant")

print(f"""
Hypothesis: If "assistant" persona is special, the model should be MORE CONFIDENT
(lower entropy, higher top-k mass) when predicting USER-turn tokens from a user
presenting as an "assistant" vs. other personae.

Results (user-turn entropy, excluding degenerate responses):
  - Most confident persona: {best_persona!r} ({best_cat}), entropy = {best_ent:.4f}
  - Least confident persona: {worst_persona!r} ({worst_cat}), entropy = {worst_ent:.4f}
  - 'assistant' ranks {assistant_rank}/{len(ranked)} (1 = most confident), entropy = {assistant_ent:.4f}

Mann-Whitney U test (assistant lower entropy than all others): p = {p_val:.6f}
Cohen's d: {d_effect:.4f}

In-character response rates (% classified as in-character by LLM):
  Personas presented as assistant-like ({', '.join([p for p,c,*_ in ic_ranked if c == 'assistant'])}):
    {ic_ranked[[p for p,c,*_ in ic_ranked].index('assistant')][2]:.1f}% in-character for 'assistant'
    {next(r[2] for r in ic_ranked if r[0] == 'helper'):.1f}% in-character for 'helper'
""")

if p_val < 0.05:
    print("  => SUPPORTS hypothesis: 'assistant' user persona elicits meaningfully more")
    print("     confident model predictions in the user turn.")
elif assistant_rank <= 2:
    print("  => WEAK SUPPORT: 'assistant' is among most confident, but not statistically significant.")
else:
    print("  => DOES NOT SUPPORT hypothesis: 'assistant' user persona does NOT elicit")
    print("     more confident predictions than other personae.")

print("\n  Note on degeneracy: High max_tokens_hit rates suggest this task is hard for the model")
print("  (long, rambling responses). Degenerate responses were excluded from entropy analysis.")
