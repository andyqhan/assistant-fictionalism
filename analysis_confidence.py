"""
Analysis of confidence metrics (entropy, top-k mass, thinking tokens)
across personae from the Assistant Fictionalism experiment.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return dict(groups)


def compute_stats(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}


def think_vals_of(records):
    """Extract think_end_position, dropping None values."""
    return [r["think_end_position"] for r in records if r.get("think_end_position") is not None]


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Load primary 8B data
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY = Path("/scratch/ah7660/assistant-fictionalism/logs/personae-inference-1164036/results.jsonl")
data8b = load_jsonl(PRIMARY)
print(f"Loaded {len(data8b)} records from 8B run")

# Basic schema check
keys_needed = ["persona", "category", "avg_entropy_output", "avg_top_k_mass_output", "think_end_position"]
for k in keys_needed:
    assert k in data8b[0], f"Missing key: {k}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean / std by PERSONA (sorted by avg_entropy_output ascending = most confident first)
# ─────────────────────────────────────────────────────────────────────────────

print_header("1. Stats by PERSONA (ranked by avg_entropy_output, ascending = most confident)")

persona_groups = group_by(data8b, "persona")

persona_stats = []
for persona, records in persona_groups.items():
    category = records[0]["category"]
    entropy_vals = [r["avg_entropy_output"] for r in records]
    topk_vals = [r["avg_top_k_mass_output"] for r in records]
    think_vals = think_vals_of(records)
    persona_stats.append({
        "persona": persona,
        "category": category,
        "n": len(records),
        "entropy_mean": np.mean(entropy_vals),
        "entropy_std": np.std(entropy_vals),
        "topk_mean": np.mean(topk_vals),
        "topk_std": np.std(topk_vals),
        "think_mean": np.mean(think_vals) if think_vals else float("nan"),
        "think_std": np.std(think_vals) if think_vals else float("nan"),
    })

persona_stats.sort(key=lambda x: x["entropy_mean"])

print(f"{'Rank':<5} {'Persona':<30} {'Category':<25} {'N':<6} "
      f"{'Entropy(out)':<14} {'TopK(out)':<12} {'ThinkTok':<10}")
print("-" * 102)
for rank, ps in enumerate(persona_stats, 1):
    print(f"{rank:<5} {ps['persona']:<30} {ps['category']:<25} {ps['n']:<6} "
          f"{ps['entropy_mean']:.4f}±{ps['entropy_std']:.4f}  "
          f"{ps['topk_mean']:.4f}±{ps['topk_std']:.4f}  "
          f"{ps['think_mean']:.1f}±{ps['think_std']:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Mean / std by CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

print_header("2. Stats by CATEGORY (ranked by avg_entropy_output, ascending = most confident)")

category_groups = group_by(data8b, "category")

category_stats = []
for category, records in category_groups.items():
    entropy_vals = [r["avg_entropy_output"] for r in records]
    topk_vals = [r["avg_top_k_mass_output"] for r in records]
    think_vals = think_vals_of(records)
    category_stats.append({
        "category": category,
        "n": len(records),
        "entropy_mean": np.mean(entropy_vals),
        "entropy_std": np.std(entropy_vals),
        "topk_mean": np.mean(topk_vals),
        "topk_std": np.std(topk_vals),
        "think_mean": np.mean(think_vals) if think_vals else float("nan"),
        "think_std": np.std(think_vals) if think_vals else float("nan"),
    })

category_stats.sort(key=lambda x: x["entropy_mean"])

print(f"{'Rank':<5} {'Category':<30} {'N':<6} "
      f"{'Entropy(out)':<14} {'TopK(out)':<12} {'ThinkTok':<10}")
print("-" * 77)
for rank, cs in enumerate(category_stats, 1):
    print(f"{rank:<5} {cs['category']:<30} {cs['n']:<6} "
          f"{cs['entropy_mean']:.4f}±{cs['entropy_std']:.4f}  "
          f"{cs['topk_mean']:.4f}±{cs['topk_std']:.4f}  "
          f"{cs['think_mean']:.1f}±{cs['think_std']:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Significance test: assistant-related vs. other categories
# ─────────────────────────────────────────────────────────────────────────────

print_header("3. Significance Tests: assistant/assistant-synonyms vs. others")

ASSISTANT_CATS = {"assistant", "assistant-synonym"}

assistant_records = [r for r in data8b if r["category"] in ASSISTANT_CATS]
other_records = [r for r in data8b if r["category"] not in ASSISTANT_CATS]

print(f"  Assistant-related records : {len(assistant_records)}")
print(f"  Other records             : {len(other_records)}")
print()

for metric, label in [
    ("avg_entropy_output", "avg_entropy_output"),
    ("avg_top_k_mass_output", "avg_top_k_mass_output"),
    ("think_end_position", "think_end_position"),
]:
    asst_vals = np.array([r[metric] for r in assistant_records if r.get(metric) is not None], dtype=float)
    other_vals = np.array([r[metric] for r in other_records if r.get(metric) is not None], dtype=float)

    if len(asst_vals) == 0 or len(other_vals) == 0:
        print(f"  Metric: {label} -- insufficient data (asst={len(asst_vals)}, other={len(other_vals)})")
        print()
        continue

    # Mann-Whitney U (non-parametric, more robust)
    u_stat, p_mw = stats.mannwhitneyu(asst_vals, other_vals, alternative="two-sided")
    # T-test as well
    t_stat, p_t = stats.ttest_ind(asst_vals, other_vals)

    print(f"  Metric: {label} (n_asst={len(asst_vals)}, n_other={len(other_vals)})")
    print(f"    Assistant: mean={asst_vals.mean():.4f}  std={asst_vals.std():.4f}")
    print(f"    Other    : mean={other_vals.mean():.4f}  std={other_vals.std():.4f}")
    print(f"    Mann-Whitney U={u_stat:.0f}, p={p_mw:.4e}")
    print(f"    t-test        t={t_stat:.4f},  p={p_t:.4e}")
    print()

# One-sided: assistant lower entropy than others
asst_entropy = np.array([r["avg_entropy_output"] for r in assistant_records], dtype=float)
other_entropy = np.array([r["avg_entropy_output"] for r in other_records], dtype=float)
_, p_one = stats.mannwhitneyu(asst_entropy, other_entropy, alternative="less")
print(f"  One-sided test (assistant entropy < other entropy): p={p_one:.4e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation: think_end_position vs avg_entropy_output
# ─────────────────────────────────────────────────────────────────────────────

print_header("4. Correlation: think_end_position vs avg_entropy_output")

data8b_with_think = [r for r in data8b if r.get("think_end_position") is not None]
think_pos = np.array([r["think_end_position"] for r in data8b_with_think], dtype=float)
entropy_out = np.array([r["avg_entropy_output"] for r in data8b_with_think], dtype=float)
topk_out = np.array([r["avg_top_k_mass_output"] for r in data8b_with_think], dtype=float)
print(f"  Records with think_end_position: {len(data8b_with_think)} / {len(data8b)}")

pearson_r, pearson_p = stats.pearsonr(think_pos, entropy_out)
spearman_r, spearman_p = stats.spearmanr(think_pos, entropy_out)

print(f"  think_end_position vs avg_entropy_output:")
print(f"    Pearson  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print(f"    Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")

pearson_r2, pearson_p2 = stats.pearsonr(think_pos, topk_out)
spearman_r2, spearman_p2 = stats.spearmanr(think_pos, topk_out)

print(f"  think_end_position vs avg_top_k_mass_output:")
print(f"    Pearson  r = {pearson_r2:.4f}, p = {pearson_p2:.4e}")
print(f"    Spearman r = {spearman_r2:.4f}, p = {spearman_p2:.4e}")

# Also per-category correlation
print()
print("  Per-category Spearman correlation (think_tokens vs entropy_out):")
for cat, records in sorted(category_groups.items()):
    tp = np.array([r["think_end_position"] for r in records], dtype=float)
    en = np.array([r["avg_entropy_output"] for r in records], dtype=float)
    tp = np.array([r["think_end_position"] for r in records if r.get("think_end_position") is not None], dtype=float)
    en = np.array([r["avg_entropy_output"] for r in records if r.get("think_end_position") is not None], dtype=float)
    if len(tp) > 10:
        r_sp, p_sp = stats.spearmanr(tp, en)
        print(f"    {cat:<30} r={r_sp:.4f}, p={p_sp:.4e}, n={len(tp)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 32B scaling comparison
# ─────────────────────────────────────────────────────────────────────────────

SCALING_RUNS = [
    Path("/scratch/ah7660/assistant-fictionalism/logs/personae-inference-2410336/results.jsonl"),
    Path("/scratch/ah7660/assistant-fictionalism/logs/personae-inference-2410338/results.jsonl"),
    Path("/scratch/ah7660/assistant-fictionalism/logs/personae-inference-2410341/results.jsonl"),
]

print_header("5. Scaling Comparison (8B baseline @ temp=0 vs temp=0.7 runs)")

# Read configs first to know which model each run is
scaling_datasets = []
for run in SCALING_RUNS:
    config_path = run.parent / "config.json"
    cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_name = cfg.get("model", "?")
    print(f"  {run.parent.name}: model={model_name}, temp={cfg.get('temperature','?')}, "
          f"max_tokens={cfg.get('max_tokens','?')}, n_reps={cfg.get('n_reps','?')}")
    recs = load_jsonl(run)
    # Filter out None entropy values
    recs = [r for r in recs if r.get("avg_entropy_output") is not None]
    scaling_datasets.append({"model": model_name, "run": run.parent.name, "records": recs})
    print(f"    -> {len(recs)} valid records")

print()

# Category-level stats for each scaling run, then cross-model table
# Build per-model category stats
all_model_cat_stats = {}
for ds in scaling_datasets:
    model = ds["model"]
    records = ds["records"]
    cat_grp = group_by(records, "category")
    cat_stats = {}
    for category, recs in cat_grp.items():
        entropy_vals = [r["avg_entropy_output"] for r in recs if r.get("avg_entropy_output") is not None]
        cat_stats[category] = {
            "entropy_mean": np.mean(entropy_vals) if entropy_vals else float("nan"),
            "n": len(entropy_vals),
        }
    all_model_cat_stats[model] = cat_stats

    # Significance test per model
    asst_recs = [r for r in records if r["category"] in ASSISTANT_CATS]
    other_recs = [r for r in records if r["category"] not in ASSISTANT_CATS]
    asst_e = np.array([r["avg_entropy_output"] for r in asst_recs], dtype=float)
    other_e = np.array([r["avg_entropy_output"] for r in other_recs], dtype=float)
    _, p_val = stats.mannwhitneyu(asst_e, other_e, alternative="less")
    print(f"  {model}: asst mean={asst_e.mean():.4f} (n={len(asst_e)}), "
          f"other mean={other_e.mean():.4f} (n={len(other_e)}), one-sided p={p_val:.4e}")

    think_recs = [r for r in records if r.get("think_end_position") is not None]
    if think_recs:
        think_vals_arr = np.array([r["think_end_position"] for r in think_recs], dtype=float)
        entropy_arr = np.array([r["avg_entropy_output"] for r in think_recs], dtype=float)
        r_sp, p_sp = stats.spearmanr(think_vals_arr, entropy_arr)
        print(f"    Spearman(think_tokens, entropy_out): r={r_sp:.4f}, p={p_sp:.4e}, n={len(think_recs)}")

print()

# Cross-model entropy comparison table
print_header("5b. Cross-Model Category Entropy Comparison")
# Include the 8B baseline (temp=0) as a column
cat8_map = {cs["category"]: cs for cs in category_stats}

all_models = [ds["model"] for ds in scaling_datasets]
all_cats_scaling = sorted(set.union(*[set(m.keys()) for m in all_model_cat_stats.values()]) | set(cat8_map.keys()))

header = f"{'Category':<30} {'8B(t=0)':<12}"
for m in all_models:
    short = m.split("/")[-1]
    header += f" {short:<16}"
print(header)
print("-" * (30 + 12 + 17 * len(all_models)))
for cat in sorted(all_cats_scaling, key=lambda c: cat8_map.get(c, {}).get("entropy_mean", 9)):
    e8 = cat8_map[cat]["entropy_mean"] if cat in cat8_map else float("nan")
    row = f"{cat:<30} {e8:<12.4f}"
    for m in all_models:
        e = all_model_cat_stats[m].get(cat, {}).get("entropy_mean", float("nan"))
        row += f" {e:<16.4f}"
    print(row)

print()
print("=" * 70)
print("Analysis complete.")
print("=" * 70)
