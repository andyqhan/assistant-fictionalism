#!/usr/bin/env python3
"""
Heuristic classifier for detecting degenerate responses in user-turn prediction results.

Tags each response with:
- repetitive_loop: Most common word 5-gram appears 4+ times
- intro_echo: difflib.SequenceMatcher ratio with intro_message > 0.7
- max_tokens_hit: num_tokens >= 508 (within 4 of 512 max)
"""

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path


def get_word_ngrams(text: str, n: int = 5) -> list[tuple[str, ...]]:
    """Extract word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def has_repetitive_loop(text: str) -> bool:
    """Check if the most common word 5-gram appears 4+ times."""
    ngrams = get_word_ngrams(text, 5)
    if not ngrams:
        return False
    most_common_count = Counter(ngrams).most_common(1)[0][1]
    return most_common_count >= 4


def has_intro_echo(response: str, intro_message: str) -> bool:
    """Check if response is too similar to the intro message."""
    response_norm = response.lower().strip()
    intro_norm = intro_message.lower().strip()
    ratio = difflib.SequenceMatcher(None, response_norm, intro_norm).ratio()
    return ratio > 0.7


def has_max_tokens_hit(num_tokens: int) -> bool:
    """Check if response hit the max token limit (within 4 of 512)."""
    return num_tokens >= 508


def classify_record(record: dict) -> dict:
    """Classify a single result record and return heuristic tags."""
    intro_message = record["intro_message"]

    tags = {
        "persona": record["persona"],
        "prompt_id": record["prompt_id"],
        "sample_idx": record["sample_idx"],
    }

    for turn in ("response1", "response2"):
        response_text = record[turn]
        num_tokens = record[f"{turn}_num_tokens"]

        tags[f"{turn}_repetitive_loop"] = has_repetitive_loop(response_text)
        tags[f"{turn}_intro_echo"] = has_intro_echo(response_text, intro_message)
        tags[f"{turn}_max_tokens_hit"] = has_max_tokens_hit(num_tokens)

    return tags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heuristic classifier for degenerate response detection",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to results.jsonl from user-turn prediction",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    assert input_path.exists(), f"Input file not found: {input_path}"

    output_path = input_path.parent / "heuristic_tags.jsonl"

    # Process all records
    n_records = 0
    tag_counts: dict[str, int] = {}

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tags = classify_record(record)
            fout.write(json.dumps(tags) + "\n")
            n_records += 1

            # Accumulate counts for summary
            for key, value in tags.items():
                if isinstance(value, bool) and value:
                    tag_counts[key] = tag_counts.get(key, 0) + 1

    print(f"Processed {n_records} records -> {output_path}")
    print(f"\nTag counts (across {n_records} records):")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count} ({count / n_records * 100:.1f}%)")


if __name__ == "__main__":
    main()
