"""
Analysis: Base model vs Instruct model confidence metrics.

Tests whether the persona confidence gradient (assistant > unspecified > famous > fictional > historical)
is a pre-training phenomenon or an instruction-tuning artifact.

Usage:
    uv run python analyze_base_vs_instruct.py <base_results_dir> <instruct_results_dir>

Example:
    uv run python analyze_base_vs_instruct.py \
        logs/base-vs-instruct-12345/base \
        logs/base-vs-instruct-12345/instruct
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return dict(groups)


def extract_metric(records: list[dict], metric: str) -> np.ndarray:
    """Extract a metric as a float array, dropping None values."""
    return np.array(
        [r[metric] for r in records if r.get(metric) is not None],
        dtype=float,
    )


def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size (non-parametric).

    Returns value in [-1, 1]. Positive means x tends to be larger than y.
    """
    n_x, n_y = len(x), len(y)
    assert n_x > 0 and n_y > 0
    # Vectorized comparison
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return (more - less) / (n_x * n_y)


# ─────────────────────────────────────────────────────────────────────────────
# Load & validate
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(base_dir: Path, instruct_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load both result files and validate matching structure."""
    base_path = base_dir / "results.jsonl"
    instruct_path = instruct_dir / "results.jsonl"

    assert base_path.exists(), f"Base results not found: {base_path}"
    assert instruct_path.exists(), f"Instruct results not found: {instruct_path}"

    base = load_jsonl(base_path)
    instruct = load_jsonl(instruct_path)

    print(f"Base model:     {len(base):,} records from {base_path}")
    print(f"Instruct model: {len(instruct):,} records from {instruct_path}")

    # Load configs
    for label, d in [("Base", base_dir), ("Instruct", instruct_dir)]:
        cfg_path = d / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            print(f"  {label} config: model={cfg.get('model')}, temp={cfg.get('temperature')}, "
                  f"thinking={cfg.get('thinking_mode')}, max_tokens={cfg.get('max_tokens')}")

    # Check matching persona/prompt pairs
    base_keys = {(r["persona"], r["prompt_id"], r["rep_idx"]) for r in base}
    instruct_keys = {(r["persona"], r["prompt_id"], r["rep_idx"]) for r in instruct}

    shared = base_keys & instruct_keys
    base_only = base_keys - instruct_keys
    instruct_only = instruct_keys - base_keys

    print(f"\nTask overlap: {len(shared):,} shared, "
          f"{len(base_only):,} base-only, {len(instruct_only):,} instruct-only")

    if base_only or instruct_only:
        print("  WARNING: Not all tasks match between runs. Analysis uses shared tasks only.")

    # Schema check
    required_keys = ["persona", "category", "avg_entropy_output", "avg_top_k_mass_output", "prompt_id"]
    for label, data in [("base", base), ("instruct", instruct)]:
        for k in required_keys:
            assert k in data[0], f"Missing key '{k}' in {label} results"

    return base, instruct


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Category-level comparison table
# ─────────────────────────────────────────────────────────────────────────────

def category_comparison(base: list[dict], instruct: list[dict]) -> None:
    print_header("1. Category-Level Comparison (entropy_output, top_k_mass_output)")

    base_by_cat = group_by(base, "category")
    instruct_by_cat = group_by(instruct, "category")

    all_cats = sorted(set(base_by_cat) | set(instruct_by_cat))

    header = (f"{'Category':<30} {'Base Entropy':<16} {'Inst Entropy':<16} {'Delta':<10} "
              f"{'Base TopK':<12} {'Inst TopK':<12}")
    print(header)
    print("-" * len(header))

    rows = []
    for cat in all_cats:
        b_ent = extract_metric(base_by_cat.get(cat, []), "avg_entropy_output")
        i_ent = extract_metric(instruct_by_cat.get(cat, []), "avg_entropy_output")
        b_topk = extract_metric(base_by_cat.get(cat, []), "avg_top_k_mass_output")
        i_topk = extract_metric(instruct_by_cat.get(cat, []), "avg_top_k_mass_output")

        b_ent_mean = np.mean(b_ent) if len(b_ent) > 0 else float("nan")
        i_ent_mean = np.mean(i_ent) if len(i_ent) > 0 else float("nan")
        delta = i_ent_mean - b_ent_mean
        b_topk_mean = np.mean(b_topk) if len(b_topk) > 0 else float("nan")
        i_topk_mean = np.mean(i_topk) if len(i_topk) > 0 else float("nan")

        rows.append((cat, b_ent_mean, i_ent_mean, delta, b_topk_mean, i_topk_mean))

    # Sort by base entropy (ascending = most confident first)
    rows.sort(key=lambda r: r[1])

    for cat, b_e, i_e, d, b_t, i_t in rows:
        print(f"{cat:<30} {b_e:<16.4f} {i_e:<16.4f} {d:<+10.4f} {b_t:<12.4f} {i_t:<12.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Persona-level rank correlation
# ─────────────────────────────────────────────────────────────────────────────

def persona_rank_correlation(base: list[dict], instruct: list[dict]) -> None:
    print_header("2. Persona-Level Rank Correlation (Base vs Instruct)")

    base_by_persona = group_by(base, "persona")
    instruct_by_persona = group_by(instruct, "persona")

    shared_personae = sorted(set(base_by_persona) & set(instruct_by_persona))
    print(f"Shared personae: {len(shared_personae)}")

    base_means = []
    instruct_means = []
    persona_labels = []

    for persona in shared_personae:
        b_vals = extract_metric(base_by_persona[persona], "avg_entropy_output")
        i_vals = extract_metric(instruct_by_persona[persona], "avg_entropy_output")
        if len(b_vals) > 0 and len(i_vals) > 0:
            base_means.append(np.mean(b_vals))
            instruct_means.append(np.mean(i_vals))
            persona_labels.append(persona)

    base_arr = np.array(base_means)
    instruct_arr = np.array(instruct_means)

    rho, p_rho = stats.spearmanr(base_arr, instruct_arr)
    tau, p_tau = stats.kendalltau(base_arr, instruct_arr)
    r, p_r = stats.pearsonr(base_arr, instruct_arr)

    print(f"\n  Spearman rho = {rho:.4f}, p = {p_rho:.4e}")
    print(f"  Kendall tau  = {tau:.4f}, p = {p_tau:.4e}")
    print(f"  Pearson r    = {r:.4f}, p = {p_r:.4e}")

    print(f"\n  Interpretation:")
    if rho > 0.7:
        print(f"    HIGH correlation ({rho:.2f}): Pre-training dominates persona confidence ordering.")
    elif rho > 0.3:
        print(f"    MODERATE correlation ({rho:.2f}): Both pre-training and instruction tuning shape confidence.")
    else:
        print(f"    LOW correlation ({rho:.2f}): Instruction tuning fundamentally reshapes persona confidence.")

    # Show top/bottom movers (biggest rank changes)
    base_ranks = stats.rankdata(base_arr)
    instruct_ranks = stats.rankdata(instruct_arr)
    rank_changes = instruct_ranks - base_ranks

    sorted_idx = np.argsort(rank_changes)
    print(f"\n  Biggest rank drops (became more confident after instruct tuning):")
    for i in sorted_idx[:5]:
        print(f"    {persona_labels[i]:<30} base_rank={int(base_ranks[i]):>3} -> "
              f"instruct_rank={int(instruct_ranks[i]):>3} (delta={int(rank_changes[i]):>+4})")

    print(f"\n  Biggest rank rises (became less confident after instruct tuning):")
    for i in sorted_idx[-5:][::-1]:
        print(f"    {persona_labels[i]:<30} base_rank={int(base_ranks[i]):>3} -> "
              f"instruct_rank={int(instruct_ranks[i]):>3} (delta={int(rank_changes[i]):>+4})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Jonckheere-Terpstra gradient test
# ─────────────────────────────────────────────────────────────────────────────

def jonckheere_terpstra_test(samples: list[np.ndarray]) -> tuple[float, float]:
    """Jonckheere-Terpstra trend test for ordered groups.

    Tests H1: samples[0] <= samples[1] <= ... <= samples[k] (stochastically).

    Args:
        samples: List of arrays, ordered by hypothesized increasing trend.

    Returns:
        (J statistic, two-sided p-value via normal approximation)
    """
    k = len(samples)
    n_total = sum(len(s) for s in samples)

    # Compute J = sum of U_ij for all i < j
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            # Mann-Whitney U count: how many (x_a, x_b) pairs with x_a < x_b
            for x_a in samples[i]:
                J += np.sum(samples[j] > x_a) + 0.5 * np.sum(samples[j] == x_a)

    # Expected value and variance under H0
    n_i = [len(s) for s in samples]

    E_J = (n_total**2 - sum(ni**2 for ni in n_i)) / 4

    numerator_var = (
        n_total**2 * (2 * n_total + 3)
        - sum(ni**2 * (2 * ni + 3) for ni in n_i)
    )
    Var_J = numerator_var / 72

    z = (J - E_J) / np.sqrt(Var_J)
    p = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided

    return J, z, p


def gradient_test(base: list[dict], instruct: list[dict]) -> None:
    print_header("3. Gradient Test (Jonckheere-Terpstra Trend Test)")

    # Hypothesized ordering (lowest entropy first)
    GRADIENT_ORDER = [
        "assistant",
        "assistant-synonym",
        "unspecified",
        "famous-person",
        "fictional-character",
        "historical-person",
    ]

    for label, data in [("BASE", base), ("INSTRUCT", instruct)]:
        print(f"\n  --- {label} MODEL ---")
        by_cat = group_by(data, "category")

        # Filter to categories present in data and in our gradient order
        available = [c for c in GRADIENT_ORDER if c in by_cat]
        missing = [c for c in GRADIENT_ORDER if c not in by_cat]
        if missing:
            print(f"  Missing categories: {missing}")

        samples = [extract_metric(by_cat[cat], "avg_entropy_output") for cat in available]

        print(f"  Category order: {' < '.join(available)}")
        print(f"  Group sizes: {[len(s) for s in samples]}")
        print(f"  Group means: {[f'{np.mean(s):.4f}' for s in samples]}")

        J, z, p = jonckheere_terpstra_test(samples)
        print(f"  J = {J:.1f}, z = {z:.4f}, p = {p:.4e}")

        if p < 0.001:
            direction = "increasing" if z > 0 else "decreasing"
            print(f"  -> Significant {direction} trend (p < 0.001)")
        elif p < 0.05:
            direction = "increasing" if z > 0 else "decreasing"
            print(f"  -> Significant {direction} trend (p < 0.05)")
        else:
            print(f"  -> No significant trend (p = {p:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Assistant separation (Mann-Whitney U with effect sizes)
# ─────────────────────────────────────────────────────────────────────────────

def assistant_separation(base: list[dict], instruct: list[dict]) -> None:
    print_header("4. Assistant Separation (Mann-Whitney U + Effect Size)")

    ASSISTANT_CATS = {"assistant", "assistant-synonym"}

    for label, data in [("BASE", base), ("INSTRUCT", instruct)]:
        print(f"\n  --- {label} MODEL ---")

        asst = [r for r in data if r["category"] in ASSISTANT_CATS]
        other = [r for r in data if r["category"] not in ASSISTANT_CATS]

        print(f"  Assistant records: {len(asst):,}")
        print(f"  Other records:     {len(other):,}")

        for metric_name, metric_key in [
            ("avg_entropy_output", "avg_entropy_output"),
            ("avg_top_k_mass_output", "avg_top_k_mass_output"),
        ]:
            a_vals = extract_metric(asst, metric_key)
            o_vals = extract_metric(other, metric_key)

            if len(a_vals) == 0 or len(o_vals) == 0:
                print(f"\n    {metric_name}: insufficient data")
                continue

            u_stat, p_two = stats.mannwhitneyu(a_vals, o_vals, alternative="two-sided")
            _, p_less = stats.mannwhitneyu(a_vals, o_vals, alternative="less")

            # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
            n1, n2 = len(a_vals), len(o_vals)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)

            # Cliff's delta (sampled for speed if large)
            if n1 * n2 > 10_000_000:
                # Subsample for Cliff's delta
                rng = np.random.default_rng(42)
                a_sub = rng.choice(a_vals, size=min(5000, n1), replace=False)
                o_sub = rng.choice(o_vals, size=min(5000, n2), replace=False)
                cd = cliff_delta(a_sub, o_sub)
                cd_note = " (subsampled)"
            else:
                cd = cliff_delta(a_vals, o_vals)
                cd_note = ""

            print(f"\n    {metric_name}:")
            print(f"      Assistant: mean={a_vals.mean():.4f}, std={a_vals.std():.4f}")
            print(f"      Other:     mean={o_vals.mean():.4f}, std={o_vals.std():.4f}")
            print(f"      Mann-Whitney U = {u_stat:.0f}")
            print(f"      Two-sided p = {p_two:.4e}")
            print(f"      One-sided p (assistant < other) = {p_less:.4e}")
            print(f"      Rank-biserial r = {r_rb:.4f}")
            print(f"      Cliff's delta = {cd:.4f}{cd_note}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Per-prompt paired analysis
# ─────────────────────────────────────────────────────────────────────────────

def per_prompt_paired_analysis(base: list[dict], instruct: list[dict]) -> None:
    print_header("5. Per-Prompt Paired Analysis: Does base-instruct gap vary by category?")

    # Build lookup for instruct results by (persona, prompt_id, rep_idx)
    instruct_lookup = {}
    for r in instruct:
        key = (r["persona"], r["prompt_id"], r["rep_idx"])
        instruct_lookup[key] = r

    # Compute per-task entropy delta = instruct - base
    deltas_by_cat: dict[str, list[float]] = defaultdict(list)
    matched = 0

    for r in base:
        key = (r["persona"], r["prompt_id"], r["rep_idx"])
        if key in instruct_lookup:
            i_r = instruct_lookup[key]
            b_ent = r.get("avg_entropy_output")
            i_ent = i_r.get("avg_entropy_output")
            if b_ent is not None and i_ent is not None:
                deltas_by_cat[r["category"]].append(i_ent - b_ent)
                matched += 1

    print(f"  Matched task pairs: {matched:,}")

    # Show per-category deltas
    print(f"\n  {'Category':<30} {'N':<8} {'Mean Delta':<14} {'Std Delta':<14} {'Median Delta':<14}")
    print(f"  {'-'*80}")

    cat_delta_arrays = {}
    for cat in sorted(deltas_by_cat):
        d = np.array(deltas_by_cat[cat])
        cat_delta_arrays[cat] = d
        print(f"  {cat:<30} {len(d):<8} {np.mean(d):<+14.4f} {np.std(d):<14.4f} {np.median(d):<+14.4f}")

    # Kruskal-Wallis: do deltas differ across categories?
    if len(cat_delta_arrays) >= 2:
        kw_stat, kw_p = stats.kruskal(*cat_delta_arrays.values())
        print(f"\n  Kruskal-Wallis test (do deltas differ across categories?):")
        print(f"    H = {kw_stat:.4f}, p = {kw_p:.4e}")

        if kw_p < 0.05:
            print(f"    -> Significant: instruction tuning affects categories differently.")
        else:
            print(f"    -> Not significant: instruction tuning has uniform effect across categories.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Degeneracy check for base model
# ─────────────────────────────────────────────────────────────────────────────

def degeneracy_check(base: list[dict]) -> None:
    print_header("6. Degeneracy Check (Base Model)")

    n_total = len(base)

    # Check for very short responses (< 10 tokens)
    short = [r for r in base if r.get("num_tokens", 0) < 10]
    print(f"  Very short responses (< 10 tokens): {len(short):,} / {n_total:,} ({100*len(short)/n_total:.1f}%)")

    # Check for null entropy (model produced no valid output)
    null_entropy = [r for r in base if r.get("avg_entropy_output") is None]
    print(f"  Null entropy_output: {len(null_entropy):,} / {n_total:,} ({100*len(null_entropy)/n_total:.1f}%)")

    # Entropy distribution
    ent = extract_metric(base, "avg_entropy_output")
    if len(ent) > 0:
        pcts = np.percentile(ent, [5, 25, 50, 75, 95])
        print(f"\n  Entropy distribution:")
        print(f"    p5={pcts[0]:.4f}  p25={pcts[1]:.4f}  median={pcts[2]:.4f}  "
              f"p75={pcts[3]:.4f}  p95={pcts[4]:.4f}")
        print(f"    mean={np.mean(ent):.4f}  std={np.std(ent):.4f}")

    # Check for repetitive outputs: high max entropy with low min entropy
    # (typical of degenerate repetition loops)
    max_ent = extract_metric(base, "max_entropy_output")
    min_ent = extract_metric(base, "min_entropy_output")
    if len(max_ent) > 0 and len(min_ent) > 0:
        ent_range = max_ent - min_ent
        high_range = np.sum(ent_range > 5.0)
        print(f"\n  Entropy range (max - min) > 5.0: {high_range:,} / {len(ent_range):,} "
              f"({100*high_range/len(ent_range):.1f}%)")

    # Per-category degeneracy
    by_cat = group_by(base, "category")
    print(f"\n  Per-category null entropy counts:")
    for cat in sorted(by_cat):
        recs = by_cat[cat]
        n_null = sum(1 for r in recs if r.get("avg_entropy_output") is None)
        if n_null > 0:
            print(f"    {cat:<30} {n_null:>5} / {len(recs):>5} ({100*n_null/len(recs):.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Overall summary
# ─────────────────────────────────────────────────────────────────────────────

def overall_summary(base: list[dict], instruct: list[dict]) -> None:
    print_header("7. Overall Summary")

    b_ent = extract_metric(base, "avg_entropy_output")
    i_ent = extract_metric(instruct, "avg_entropy_output")

    print(f"  Base model:     mean entropy = {np.mean(b_ent):.4f} (std={np.std(b_ent):.4f}, n={len(b_ent):,})")
    print(f"  Instruct model: mean entropy = {np.mean(i_ent):.4f} (std={np.std(i_ent):.4f}, n={len(i_ent):,})")
    print(f"  Difference:     {np.mean(i_ent) - np.mean(b_ent):+.4f} (instruct - base)")

    b_topk = extract_metric(base, "avg_top_k_mass_output")
    i_topk = extract_metric(instruct, "avg_top_k_mass_output")
    print(f"\n  Base model:     mean top-k mass = {np.mean(b_topk):.4f} (std={np.std(b_topk):.4f})")
    print(f"  Instruct model: mean top-k mass = {np.mean(i_topk):.4f} (std={np.std(i_topk):.4f})")

    # Overall Mann-Whitney between models
    u, p = stats.mannwhitneyu(b_ent, i_ent, alternative="two-sided")
    print(f"\n  Overall Mann-Whitney (base vs instruct entropy): U={u:.0f}, p={p:.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: uv run python analyze_base_vs_instruct.py <base_dir> <instruct_dir>")
        print("Example: uv run python analyze_base_vs_instruct.py "
              "logs/base-vs-instruct-12345/base logs/base-vs-instruct-12345/instruct")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    instruct_dir = Path(sys.argv[2])

    assert base_dir.is_dir(), f"Not a directory: {base_dir}"
    assert instruct_dir.is_dir(), f"Not a directory: {instruct_dir}"

    # Load & validate
    base, instruct = load_and_validate(base_dir, instruct_dir)

    # Run all analyses
    category_comparison(base, instruct)
    persona_rank_correlation(base, instruct)
    gradient_test(base, instruct)
    assistant_separation(base, instruct)
    per_prompt_paired_analysis(base, instruct)
    degeneracy_check(base)
    overall_summary(base, instruct)

    print()
    print("=" * 78)
    print("Analysis complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
