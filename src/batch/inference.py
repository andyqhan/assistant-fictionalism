#!/usr/bin/env python3
"""
Batch persona inference script.

Runs inference across multiple personas and prompts, computing
entropy and top-k mass metrics for generated responses.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import get_device, get_dtype
from src.template import patch_chat_template

from .config import BatchInferenceConfig
from .metrics import compute_metrics_for_sequence
from .system_prompts import generate_system_prompt


# Qwen3 </think> token ID
THINK_END_TOKEN_ID = 151668


class GPUMonitor:
    """Background thread that periodically logs GPU utilization and memory."""

    def __init__(self, interval: float = 10.0):
        """
        Args:
            interval: Seconds between each nvidia-smi query.
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _log_gpu_stats(self) -> None:
        """Query nvidia-smi and print stats."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 7:
                        timestamp, idx, name, util, mem_used, mem_total, temp = parts[:7]
                        print(
                            f"[GPU {idx}] {timestamp} | "
                            f"Util: {util}% | "
                            f"Mem: {mem_used}/{mem_total} MB | "
                            f"Temp: {temp}°C",
                            flush=True,
                        )
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"[GPU Monitor] Error querying nvidia-smi: {e}", flush=True)

    def _monitor_loop(self) -> None:
        """Background loop that logs GPU stats at regular intervals."""
        while not self._stop_event.is_set():
            self._log_gpu_stats()
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[GPU Monitor] Started (interval: {self.interval}s)", flush=True)

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._thread = None
        print("[GPU Monitor] Stopped", flush=True)

    def __enter__(self) -> "GPUMonitor":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


@dataclass
class PersonaInfo:
    """Information about a persona."""

    persona: str
    category: str
    article: str


@dataclass
class PromptInfo:
    """Information about a prompt."""

    question: str
    prompt_id: int


@dataclass
class InferenceTask:
    """A single inference task to run."""

    persona_info: PersonaInfo
    system_prompt: str
    prompt_info: PromptInfo
    rep_idx: int


def load_prompts(path: str) -> list[PromptInfo]:
    """Load prompts from JSON or JSONL file.

    Supports two formats:
    - JSON: Array of strings ["prompt1", "prompt2", ...]
    - JSONL: One JSON object per line with "question" and "id" fields
    """
    prompts = []

    if path.endswith(".jsonl"):
        # JSONL format: one JSON object per line
        with open(path, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "question" in obj, f"Line {line_num + 1}: Missing 'question' key"
                prompt_id = obj.get("id", line_num)
                prompts.append(PromptInfo(question=obj["question"], prompt_id=prompt_id))
    else:
        # JSON format: array of strings
        with open(path, "r") as f:
            data = json.load(f)

        assert isinstance(data, list), f"prompts.json must be a list, got {type(data)}"
        for idx, item in enumerate(data):
            if isinstance(item, str):
                prompts.append(PromptInfo(question=item, prompt_id=idx))
            elif isinstance(item, dict) and "question" in item:
                prompt_id = item.get("id", idx)
                prompts.append(PromptInfo(question=item["question"], prompt_id=prompt_id))
            else:
                raise ValueError(f"Invalid prompt format at index {idx}: {item}")

    assert len(prompts) > 0, "Prompts file must not be empty"

    return prompts


def load_personae(path: str) -> list[PersonaInfo]:
    """Load personae from JSON file."""
    with open(path, "r") as f:
        personae_raw = json.load(f)

    assert isinstance(personae_raw, list), f"personae.json must be a list, got {type(personae_raw)}"
    assert len(personae_raw) > 0, "personae.json must not be empty"

    personae = []
    for p in personae_raw:
        assert "persona" in p, f"Missing 'persona' key: {p}"
        assert "category" in p, f"Missing 'category' key: {p}"
        assert "article" in p, f"Missing 'article' key: {p}"
        personae.append(PersonaInfo(
            persona=p["persona"],
            category=p["category"],
            article=p["article"],
        ))

    return personae


def build_tasks(
    prompts: list[PromptInfo],
    personae: list[PersonaInfo],
    n_reps: int,
    system_prompt_style: str,
) -> list[InferenceTask]:
    """Build list of all inference tasks to run."""
    tasks = []

    for persona_info in personae:
        system_prompt = generate_system_prompt(
            persona_info.persona,
            persona_info.article,
            system_prompt_style,
        )

        for prompt_info in prompts:
            for rep_idx in range(n_reps):
                tasks.append(InferenceTask(
                    persona_info=persona_info,
                    system_prompt=system_prompt,
                    prompt_info=prompt_info,
                    rep_idx=rep_idx,
                ))

    return tasks


class BatchInferenceRunner:
    """Runs batch inference with persona support."""

    def __init__(self, cfg: BatchInferenceConfig):
        self.cfg = cfg
        self.device = get_device()
        self.dtype = get_dtype(self.device)

        print(f"Loading model {cfg.model} on {self.device} with {self.dtype}...")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        # Set up left padding for batching
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Store original template for persona patching
        self._original_template = self.tokenizer.chat_template

        # Load model
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype=self.dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

        assert self.model is not None, "Failed to load model"
        self.model.eval()

        print("Model loaded successfully.")

    def set_persona(self, persona: str) -> None:
        """Set the current persona for template patching."""
        patch_chat_template(self.tokenizer, self._original_template, persona)

    def prepare_input(self, system_prompt: str, user_prompt: str) -> str:
        """Prepare a single input text with system prompt and user message."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )

        return text

    def run_batch(self, tasks: list[InferenceTask]) -> list[dict]:
        """
        Run inference on a batch of tasks with the same persona.

        All tasks must have the same persona (for consistent template).

        Returns:
            List of result dictionaries.
        """
        if len(tasks) == 0:
            return []

        # Verify all tasks have same persona
        persona = tasks[0].persona_info.persona
        assert all(t.persona_info.persona == persona for t in tasks), (
            "All tasks in batch must have the same persona"
        )

        # Set persona for template
        self.set_persona(persona)

        # Prepare input texts
        input_texts = [
            self.prepare_input(t.system_prompt, t.prompt_info.question)
            for t in tasks
        ]

        # Tokenize with padding
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Use the padded input length (same for all sequences in batch)
        # This is the correct starting position for generated tokens with left-padding
        padded_input_len = inputs.input_ids.shape[1]

        # Generate with logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Process outputs
        results = []
        generated_ids = outputs.sequences
        scores = outputs.scores  # Tuple of (batch_size, vocab_size) per step

        for batch_idx, task in enumerate(tasks):
            input_len = padded_input_len

            # Extract generated token IDs (excluding input)
            full_seq = generated_ids[batch_idx]
            gen_token_ids = full_seq[input_len:].tolist()

            # Extract logits for this sequence
            logits_list = []
            for step_idx, step_scores in enumerate(scores):
                if step_idx < len(gen_token_ids):
                    logits_list.append(step_scores[batch_idx])

            # Compute metrics
            metrics = compute_metrics_for_sequence(
                logits_list=logits_list,
                token_ids=gen_token_ids,
                think_end_token_id=THINK_END_TOKEN_ID,
                top_k=self.cfg.top_k_mass_k,
            )

            # Decode response
            response = self.tokenizer.decode(
                gen_token_ids,
                skip_special_tokens=True,
            )

            results.append({
                "persona": task.persona_info.persona,
                "category": task.persona_info.category,
                "article": task.persona_info.article,
                "system_prompt": task.system_prompt,
                "prompt_id": task.prompt_info.prompt_id,
                "prompt": task.prompt_info.question,
                "rep_idx": task.rep_idx,
                "response": response,
                **metrics,
            })

        return results

    def run_all(self, tasks: list[InferenceTask]) -> list[dict]:
        """Run inference on all tasks, batching by persona."""
        results = []

        # Group tasks by persona
        tasks_by_persona: dict[str, list[InferenceTask]] = {}
        for task in tasks:
            persona = task.persona_info.persona
            if persona not in tasks_by_persona:
                tasks_by_persona[persona] = []
            tasks_by_persona[persona].append(task)

        # Process each persona's tasks
        total_batches = sum(
            (len(persona_tasks) + self.cfg.batch_size - 1) // self.cfg.batch_size
            for persona_tasks in tasks_by_persona.values()
        )

        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for persona, persona_tasks in tasks_by_persona.items():
                # Process in batches
                for i in range(0, len(persona_tasks), self.cfg.batch_size):
                    batch = persona_tasks[i : i + self.cfg.batch_size]
                    batch_results = self.run_batch(batch)
                    results.extend(batch_results)
                    pbar.update(1)

        return results


def save_results(cfg: BatchInferenceConfig, results: list[dict]) -> None:
    """Save configuration and results to output directory."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {results_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch persona inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompts-json",
        type=str,
        required=True,
        help="Path to prompts JSON file",
    )
    parser.add_argument(
        "--personae-json",
        type=str,
        required=True,
        help="Path to personae JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model ID to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=1,
        help="Number of repetitions per prompt",
    )
    parser.add_argument(
        "--top-k-mass-k",
        type=int,
        default=5,
        help="k for top-k mass computation",
    )
    parser.add_argument(
        "--thinking-mode",
        action="store_true",
        default=True,
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--no-thinking-mode",
        action="store_false",
        dest="thinking_mode",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--system-prompt-style",
        type=str,
        default="you-are-a",
        choices=["you-are-a"],
        help="System prompt generation style",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (auto-generated if empty)",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between GPU utilization logs (0 to disable)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for batch inference."""
    args = parse_args()

    # Create config (validates inputs)
    cfg = BatchInferenceConfig(
        prompts_json=args.prompts_json,
        personae_json=args.personae_json,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_reps=args.n_reps,
        top_k_mass_k=args.top_k_mass_k,
        thinking_mode=args.thinking_mode,
        system_prompt_style=args.system_prompt_style,
        output_dir=args.output_dir,
    )

    print(f"Configuration: {cfg.to_dict()}")

    # Load inputs
    prompts = load_prompts(cfg.prompts_json)
    print(f"Loaded {len(prompts)} prompts")

    personae = load_personae(cfg.personae_json)
    print(f"Loaded {len(personae)} personae")

    # Build task list
    tasks = build_tasks(prompts, personae, cfg.n_reps, cfg.system_prompt_style)
    print(f"Created {len(tasks)} tasks")

    # Run inference with GPU monitoring
    runner = BatchInferenceRunner(cfg)

    if args.gpu_monitor_interval > 0:
        with GPUMonitor(interval=args.gpu_monitor_interval):
            results = runner.run_all(tasks)
    else:
        results = runner.run_all(tasks)

    # Save results
    save_results(cfg, results)

    print("Done!")


if __name__ == "__main__":
    main()
