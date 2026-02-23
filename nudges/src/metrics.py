#!/usr/bin/env python3
"""Compute nudge experiment metrics from judged results.

Outputs:
- choice_rates.csv: P(chose A) per (persona, prompt_id, nudge_type)
- flip_rates.csv: P(A|nudge) - P(A|baseline) per (persona, prompt_id, nudge_type)
- confidence_shifts.csv: entropy_output(nudge) - entropy_output(baseline)
- reference_rates.csv: Proportions of IGNORES/ACKNOWLEDGES/USES/DRIVEN
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_judged_results(path: str) -> pd.DataFrame:
    """Load judged results from JSONL into a DataFrame."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    assert len(results) > 0, f"No results found in {path}"
    return pd.DataFrame(results)


def compute_choice_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P(chose A) per (persona, prompt_id, nudge_type).

    Filters out AMBIGUOUS and PARSE_ERROR choices.
    """
    valid = df[df["judge_choice"].isin(["A", "B"])].copy()
    valid["chose_a"] = (valid["judge_choice"] == "A").astype(int)

    rates = (
        valid.groupby(["persona", "prompt_id", "nudge_type"])["chose_a"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    rates.columns = ["persona", "prompt_id", "nudge_type", "choice_rate_a", "n", "choice_rate_std"]
    return rates


def compute_flip_rates(choice_rates: pd.DataFrame) -> pd.DataFrame:
    """Compute flip rate = P(A|nudge) - P(A|baseline) per (persona, prompt_id, nudge_type)."""
    baselines = choice_rates[choice_rates["nudge_type"] == "baseline"][
        ["persona", "prompt_id", "choice_rate_a"]
    ].rename(columns={"choice_rate_a": "baseline_rate"})

    nudged = choice_rates[choice_rates["nudge_type"] != "baseline"].copy()
    merged = nudged.merge(baselines, on=["persona", "prompt_id"], how="left")
    merged["flip_rate"] = merged["choice_rate_a"] - merged["baseline_rate"]
    return merged


def compute_confidence_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute entropy shift between nudge and baseline conditions.

    confidence_shift = mean(avg_entropy_output|nudge) - mean(avg_entropy_output|baseline)
    """
    # Filter to rows with non-null output entropy
    valid = df[df["avg_entropy_output"].notna()].copy()

    grouped = (
        valid.groupby(["persona", "prompt_id", "nudge_type"])["avg_entropy_output"]
        .mean()
        .reset_index()
    )

    baselines = grouped[grouped["nudge_type"] == "baseline"][
        ["persona", "prompt_id", "avg_entropy_output"]
    ].rename(columns={"avg_entropy_output": "baseline_entropy"})

    nudged = grouped[grouped["nudge_type"] != "baseline"].copy()
    merged = nudged.merge(baselines, on=["persona", "prompt_id"], how="left")
    merged["confidence_shift"] = merged["avg_entropy_output"] - merged["baseline_entropy"]
    return merged


def compute_reference_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute proportions of reference levels per (persona, nudge_type).

    Only includes non-baseline results with valid reference classifications.
    """
    non_baseline = df[df["nudge_type"] != "baseline"].copy()
    valid = non_baseline[
        non_baseline["judge_reference"].isin(["IGNORES", "ACKNOWLEDGES", "USES", "DRIVEN"])
    ]

    counts = (
        valid.groupby(["persona", "nudge_type", "judge_reference"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        valid.groupby(["persona", "nudge_type"])
        .size()
        .reset_index(name="total")
    )

    merged = counts.merge(totals, on=["persona", "nudge_type"])
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute nudge experiment metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Path to judged_results.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_judged_results(args.input)
    print(f"Loaded {len(df)} judged results")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choice rates
    choice_rates = compute_choice_rates(df)
    choice_rates.to_csv(output_dir / "choice_rates.csv", index=False)
    print(f"Saved choice_rates.csv ({len(choice_rates)} rows)")

    # Flip rates
    flip_rates = compute_flip_rates(choice_rates)
    flip_rates.to_csv(output_dir / "flip_rates.csv", index=False)
    print(f"Saved flip_rates.csv ({len(flip_rates)} rows)")

    # Confidence shifts
    confidence_shifts = compute_confidence_shifts(df)
    confidence_shifts.to_csv(output_dir / "confidence_shifts.csv", index=False)
    print(f"Saved confidence_shifts.csv ({len(confidence_shifts)} rows)")

    # Reference rates
    reference_rates = compute_reference_rates(df)
    reference_rates.to_csv(output_dir / "reference_rates.csv", index=False)
    print(f"Saved reference_rates.csv ({len(reference_rates)} rows)")

    # Print summary
    print("\n--- Summary ---")
    n_total = len(df)
    n_valid_choice = len(df[df["judge_choice"].isin(["A", "B"])])
    n_ambiguous = len(df[df["judge_choice"] == "AMBIGUOUS"])
    n_parse_error = len(df[df["judge_choice"] == "PARSE_ERROR"])
    print(f"Total results: {n_total}")
    print(f"Valid choices: {n_valid_choice} ({100*n_valid_choice/n_total:.1f}%)")
    print(f"Ambiguous: {n_ambiguous} ({100*n_ambiguous/n_total:.1f}%)")
    print(f"Parse errors: {n_parse_error} ({100*n_parse_error/n_total:.1f}%)")

    non_baseline = df[df["nudge_type"] != "baseline"]
    valid_ref = non_baseline[non_baseline["judge_reference"].isin(["IGNORES", "ACKNOWLEDGES", "USES", "DRIVEN"])]
    print(f"\nNon-baseline results: {len(non_baseline)}")
    print(f"Valid reference classifications: {len(valid_ref)} ({100*len(valid_ref)/len(non_baseline):.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
