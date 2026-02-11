#!/usr/bin/env python3
"""
TC-LLM text clustering for persona consistency analysis.

Implements a modified version of TC-LLM (Huang & He 2025) that assigns
up to N ranked labels per text. The 3-stage pipeline:
  1. Label generation: LLM generates category labels from batches of texts
  2. Label merging: LLM merges redundant labels into a clean set per group
  3. Classification: LLM assigns ranked labels to each text

Groups are defined by (prompt_id, persona) pairs from prefill inference results.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .embeddings import extract_output
from .judge_inference import GPUMonitor
from .tc_llm_config import TCLLMConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TCLLMGroup:
    """A group of texts sharing the same (prompt_id, persona)."""

    prompt_id: int
    persona: str
    texts: list[str]
    rep_indices: list[int]
    raw_labels: list[str] | None = None
    merged_labels: list[str] | None = None


@dataclass
class TCLLMRecord:
    """Classification result for a single text."""

    prompt_id: int
    persona: str
    rep_idx: int
    labels: list[str]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LABEL_GEN_TEMPLATE = """\
You are a text classification expert. Below are {n_texts} text samples from the same source. \
Generate descriptive category labels (2-5 words each) that capture the main themes, topics, \
or rhetorical strategies present across these samples. \
Return ONLY a JSON array of label strings: ["label1", "label2", ...]

Text samples:
{texts_block}"""

LABEL_MERGE_TEMPLATE = """\
You are a text classification expert. Analyze the following list of category labels. \
Merge similar or redundant entries into single representative labels. \
Produce a flat JSON array with no more than 20 merged labels: ["merged1", "merged2", ...]

Labels to merge:
{labels_json}"""

CLASSIFY_TEMPLATE = """\
You are a text classification expert. Given the label list and the text below, \
assign up to {n_labels} labels ranked by relevance.

IMPORTANT: You MUST copy labels EXACTLY as they appear in the list below — same spelling, \
same capitalization, same punctuation. Do NOT rephrase, abbreviate, or reword any label.

Return ONLY a JSON array ordered from most to least relevant: ["most_relevant", "second", ...]

Available labels:
{labels_json}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_as_groups(input_path: str) -> list[TCLLMGroup]:
    """Load results.jsonl and group by (prompt_id, persona).

    Args:
        input_path: Path to results.jsonl

    Returns:
        List of TCLLMGroup objects, one per (prompt_id, persona) pair.
    """
    records: dict[tuple[int, str], TCLLMGroup] = {}

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)

            prompt_id = result["prompt_id"]
            persona = result["persona"]
            rep_idx = result["rep_idx"]
            text = extract_output(result["response"])

            key = (prompt_id, persona)
            if key not in records:
                records[key] = TCLLMGroup(
                    prompt_id=prompt_id,
                    persona=persona,
                    texts=[],
                    rep_indices=[],
                )
            records[key].texts.append(text)
            records[key].rep_indices.append(rep_idx)

    groups = list(records.values())
    assert len(groups) > 0, "No groups found in results file"
    return groups


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------

def parse_json_array(text: str) -> list[str] | None:
    """Extract a JSON array of strings from LLM output.

    Tries multiple strategies:
    1. Extract from ```json ... ``` code block
    2. Extract from ``` ... ``` code block
    3. Find JSON array [...] in text
    4. Parse entire text as JSON
    5. Fallback: split by newlines, strip numbering/bullets

    Returns:
        List of strings, or None if parsing fails completely.
    """
    # Strategy 1: ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        parsed = _try_parse_array(match.group(1))
        if parsed is not None:
            return parsed

    # Strategy 2: ``` ... ```
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        parsed = _try_parse_array(match.group(1))
        if parsed is not None:
            return parsed

    # Strategy 3: find [...] in text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        parsed = _try_parse_array(match.group(0))
        if parsed is not None:
            return parsed

    # Strategy 4: entire text
    parsed = _try_parse_array(text)
    if parsed is not None:
        return parsed

    # Strategy 5: line-by-line fallback
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip common list prefixes: "1. ", "- ", "* ", "1) "
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        # Strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if line:
            lines.append(line)

    return lines if lines else None


def _try_parse_array(text: str) -> list[str] | None:
    """Try to parse text as a JSON array of strings."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Resume support — save/load intermediate results
# ---------------------------------------------------------------------------

def save_groups_jsonl(groups: list[TCLLMGroup], path: Path) -> None:
    """Save groups to a JSONL file."""
    with open(path, "w") as f:
        for g in groups:
            record = {
                "prompt_id": g.prompt_id,
                "persona": g.persona,
                "n_texts": len(g.texts),
                "raw_labels": g.raw_labels,
                "merged_labels": g.merged_labels,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(groups)} groups to {path}")


def load_groups_metadata(path: Path, groups: list[TCLLMGroup]) -> None:
    """Load raw_labels and merged_labels from a saved JSONL file back into groups.

    Matches by (prompt_id, persona) key.
    """
    saved: dict[tuple[int, str], dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record["prompt_id"], record["persona"])
            saved[key] = record

    for g in groups:
        key = (g.prompt_id, g.persona)
        if key in saved:
            g.raw_labels = saved[key].get("raw_labels")
            g.merged_labels = saved[key].get("merged_labels")


def save_records_jsonl(records: list[TCLLMRecord], path: Path) -> None:
    """Save classification records to a JSONL file."""
    with open(path, "w") as f:
        for r in records:
            record = {
                "prompt_id": r.prompt_id,
                "persona": r.persona,
                "rep_idx": r.rep_idx,
                "labels": r.labels,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} records to {path}")


def save_config(cfg: TCLLMConfig, output_dir: Path) -> None:
    """Save configuration to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "tc_llm_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")


def configs_match(saved_config: dict, current_config: TCLLMConfig) -> bool:
    """Check if saved config matches current config (ignoring output_dir)."""
    current_dict = current_config.to_comparable_dict()
    saved_comparable = {k: v for k, v in saved_config.items() if k != "output_dir"}
    return current_dict == saved_comparable


# ---------------------------------------------------------------------------
# TC-LLM Runner
# ---------------------------------------------------------------------------

class TCLLMRunner:
    """Runs the 3-stage TC-LLM clustering pipeline using vLLM."""

    def __init__(self, cfg: TCLLMConfig, output_dir: Path):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        self.output_dir = output_dir

        print(f"Loading model {cfg.model} with vLLM...")

        # Load tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)

        # Initialize vLLM
        self.llm = LLM(
            model=cfg.model,
            dtype="auto",
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

        # Sampling params per stage
        self.sp_label_gen = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_label_gen,
        )
        self.sp_label_merge = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_label_merge,
        )
        self.sp_classify = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_classify,
        )

        print("Model loaded successfully.")

    def _apply_chat_template(self, user_content: str) -> str:
        """Format user content with chat template."""
        messages = [{"role": "user", "content": user_content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )

    def _generate_in_chunks(
        self, prompts: list[str], sampling_params, desc: str
    ) -> list[str]:
        """Run vLLM generation on prompts, chunked by batch_size.

        Returns list of generated text strings in the same order as prompts.
        """
        results: list[str] = []
        n_chunks = math.ceil(len(prompts) / self.cfg.batch_size)

        for i in tqdm(
            range(0, len(prompts), self.cfg.batch_size),
            total=n_chunks,
            desc=desc,
        ):
            chunk = prompts[i : i + self.cfg.batch_size]
            outputs = self.llm.generate(chunk, sampling_params)
            for output in outputs:
                results.append(output.outputs[0].text)

        assert len(results) == len(prompts)
        return results

    # ----- Step 1: Label Generation -----

    def run_step1_label_generation(self, groups: list[TCLLMGroup]) -> None:
        """Generate raw labels for each group by batching texts."""
        B = self.cfg.texts_per_label_batch

        # Build all label-gen prompts across all groups
        prompt_map: list[tuple[int, str]] = []  # (group_idx, prompt)
        for gi, g in enumerate(groups):
            for start in range(0, len(g.texts), B):
                batch_texts = g.texts[start : start + B]
                texts_block = "\n\n".join(
                    f"[{j+1}] {t}" for j, t in enumerate(batch_texts)
                )
                user_content = LABEL_GEN_TEMPLATE.format(
                    n_texts=len(batch_texts), texts_block=texts_block
                )
                prompt_map.append((gi, self._apply_chat_template(user_content)))

        print(f"Step 1: Label generation — {len(prompt_map)} prompts across {len(groups)} groups")

        # Generate
        formatted_prompts = [p for _, p in prompt_map]
        raw_outputs = self._generate_in_chunks(
            formatted_prompts, self.sp_label_gen, "Step 1: Label gen"
        )

        # Parse and deduplicate labels per group
        group_labels: dict[int, set[str]] = {i: set() for i in range(len(groups))}
        parse_failures = 0

        for (gi, _), output_text in zip(prompt_map, raw_outputs):
            parsed = parse_json_array(output_text)
            if parsed:
                group_labels[gi].update(parsed)
            else:
                parse_failures += 1

        if parse_failures > 0:
            print(f"  Warning: {parse_failures}/{len(prompt_map)} label-gen responses failed to parse")

        for gi, g in enumerate(groups):
            g.raw_labels = sorted(group_labels[gi])

        # Save intermediate
        raw_labels_path = self.output_dir / "tc_llm_raw_labels.jsonl"
        save_groups_jsonl(groups, raw_labels_path)

    # ----- Step 2: Label Merging -----

    def run_step2_label_merging(self, groups: list[TCLLMGroup]) -> None:
        """Merge redundant labels for each group."""
        # Build one merge prompt per group
        prompts = []
        for g in groups:
            assert g.raw_labels is not None, (
                f"Group ({g.prompt_id}, {g.persona}) missing raw_labels"
            )
            user_content = LABEL_MERGE_TEMPLATE.format(
                labels_json=json.dumps(g.raw_labels)
            )
            prompts.append(self._apply_chat_template(user_content))

        print(f"Step 2: Label merging — {len(prompts)} prompts")

        raw_outputs = self._generate_in_chunks(
            prompts, self.sp_label_merge, "Step 2: Label merge"
        )

        parse_failures = 0
        for g, output_text in zip(groups, raw_outputs):
            parsed = parse_json_array(output_text)
            if parsed:
                g.merged_labels = parsed[:20]  # Enforce max 20
            else:
                parse_failures += 1
                # Fallback: use raw labels directly (truncated)
                g.merged_labels = (g.raw_labels or [])[:20]

        if parse_failures > 0:
            print(f"  Warning: {parse_failures}/{len(prompts)} merge responses failed to parse, using raw labels as fallback")

        # Save intermediate
        groups_path = self.output_dir / "tc_llm_groups.jsonl"
        save_groups_jsonl(groups, groups_path)

    # ----- Step 3: Classification -----

    def run_step3_classification(self, groups: list[TCLLMGroup]) -> list[TCLLMRecord]:
        """Classify each text with ranked labels."""
        n_labels = self.cfg.n_ranked_labels

        # Build one classify prompt per text
        prompt_map: list[tuple[int, str, int]] = []  # (prompt_id, persona, rep_idx)
        classify_prompts: list[str] = []

        for g in groups:
            assert g.merged_labels is not None, (
                f"Group ({g.prompt_id}, {g.persona}) missing merged_labels"
            )
            labels_json = json.dumps(g.merged_labels)

            for text, rep_idx in zip(g.texts, g.rep_indices):
                user_content = CLASSIFY_TEMPLATE.format(
                    n_labels=n_labels,
                    labels_json=labels_json,
                    text=text,
                )
                classify_prompts.append(self._apply_chat_template(user_content))
                prompt_map.append((g.prompt_id, g.persona, rep_idx))

        print(f"Step 3: Classification — {len(classify_prompts)} prompts")

        raw_outputs = self._generate_in_chunks(
            classify_prompts, self.sp_classify, "Step 3: Classify"
        )

        # Parse and validate
        records: list[TCLLMRecord] = []
        parse_failures = 0

        # Build a case-insensitive lookup for valid labels per group
        # Maps (prompt_id, persona) -> {lowercase_label: original_label}
        group_valid_labels: dict[tuple[int, str], dict[str, str]] = {}
        for g in groups:
            group_valid_labels[(g.prompt_id, g.persona)] = {
                l.lower(): l for l in (g.merged_labels or [])
            }

        for (prompt_id, persona, rep_idx), output_text in zip(prompt_map, raw_outputs):
            parsed = parse_json_array(output_text)
            valid_lookup = group_valid_labels[(prompt_id, persona)]

            if parsed:
                # Case-insensitive match, canonicalize to original label form
                filtered = []
                for l in parsed:
                    canonical = valid_lookup.get(l.lower())
                    if canonical is not None and canonical not in filtered:
                        filtered.append(canonical)
                labels = filtered[:n_labels] if filtered else ["UNCLASSIFIED"]
            else:
                parse_failures += 1
                labels = ["UNCLASSIFIED"]

            records.append(TCLLMRecord(
                prompt_id=prompt_id,
                persona=persona,
                rep_idx=rep_idx,
                labels=labels,
            ))

        if parse_failures > 0:
            print(f"  Warning: {parse_failures}/{len(classify_prompts)} classify responses failed to parse")

        # Save results
        records_path = self.output_dir / "tc_llm_records.jsonl"
        save_records_jsonl(records, records_path)

        return records

    def run_all(self, groups: list[TCLLMGroup]) -> list[TCLLMRecord]:
        """Run all 3 stages of TC-LLM pipeline."""
        self.run_step1_label_generation(groups)
        self.run_step2_label_merging(groups)
        return self.run_step3_classification(groups)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TC-LLM text clustering for persona consistency analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to results.jsonl from prefill inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="LLM model ID for clustering",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens-label-gen",
        type=int,
        default=1024,
        help="Max tokens for label generation",
    )
    parser.add_argument(
        "--max-tokens-label-merge",
        type=int,
        default=2048,
        help="Max tokens for label merging",
    )
    parser.add_argument(
        "--max-tokens-classify",
        type=int,
        default=512,
        help="Max tokens for classification",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="vLLM submission chunk size",
    )
    parser.add_argument(
        "--texts-per-label-batch",
        type=int,
        default=15,
        help="Number of texts per label generation batch (B in TC-LLM paper)",
    )
    parser.add_argument(
        "--n-ranked-labels",
        type=int,
        default=5,
        help="Max labels per text, ranked by relevance",
    )
    parser.add_argument(
        "--thinking-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable thinking mode for LLM",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between GPU utilization logs (0 to disable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (defaults to input file's parent directory)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for TC-LLM text clustering."""
    args = parse_args()

    # Create config (validates inputs)
    cfg = TCLLMConfig(
        input_path=args.input_path,
        model=args.model,
        temperature=args.temperature,
        max_tokens_label_gen=args.max_tokens_label_gen,
        max_tokens_label_merge=args.max_tokens_label_merge,
        max_tokens_classify=args.max_tokens_classify,
        batch_size=args.batch_size,
        texts_per_label_batch=args.texts_per_label_batch,
        n_ranked_labels=args.n_ranked_labels,
        thinking_mode=args.thinking_mode,
        gpu_monitor_interval=args.gpu_monitor_interval,
        output_dir=args.output_dir,
    )

    print(f"Configuration: {cfg.to_dict()}")

    output_dir = Path(cfg.output_dir)

    # Check for resume
    config_path = output_dir / "tc_llm_config.json"
    records_path = output_dir / "tc_llm_records.jsonl"
    groups_path = output_dir / "tc_llm_groups.jsonl"
    raw_labels_path = output_dir / "tc_llm_raw_labels.jsonl"

    # If final output exists, we're done
    if records_path.exists():
        print(f"All stages complete — {records_path} already exists.")
        return

    # Validate config on resume
    if config_path.exists():
        with open(config_path, "r") as f:
            saved_config = json.load(f)

        if not configs_match(saved_config, cfg):
            print("Error: Config mismatch with existing TC-LLM results.")
            print("Either delete the existing tc_llm_config.json or use a different --output-dir.")
            print(f"  Existing config: {config_path}")
            sys.exit(1)
        print("Resuming from previous run...")
    else:
        save_config(cfg, output_dir)

    # Load data
    groups = load_results_as_groups(cfg.input_path)
    total_texts = sum(len(g.texts) for g in groups)
    print(f"Loaded {len(groups)} groups with {total_texts} total texts")

    # Determine resume point
    skip_step1 = False
    skip_step2 = False

    if groups_path.exists():
        print(f"Found {groups_path} — skipping steps 1 and 2")
        load_groups_metadata(groups_path, groups)
        skip_step1 = True
        skip_step2 = True
    elif raw_labels_path.exists():
        print(f"Found {raw_labels_path} — skipping step 1")
        load_groups_metadata(raw_labels_path, groups)
        skip_step1 = True

    # Run pipeline
    runner = TCLLMRunner(cfg, output_dir)

    if cfg.gpu_monitor_interval > 0:
        gpu_monitor = GPUMonitor(interval=cfg.gpu_monitor_interval)
        gpu_monitor.start()

    if not skip_step1:
        runner.run_step1_label_generation(groups)

    if not skip_step2:
        runner.run_step2_label_merging(groups)

    records = runner.run_step3_classification(groups)

    if cfg.gpu_monitor_interval > 0:
        gpu_monitor.stop()

    # Summary
    n_unclassified = sum(1 for r in records if r.labels == ["UNCLASSIFIED"])
    print(f"\nDone! {len(records)} records classified.")
    if n_unclassified > 0:
        print(f"  {n_unclassified} records marked UNCLASSIFIED ({n_unclassified/len(records)*100:.1f}%)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
