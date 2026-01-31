#!/usr/bin/env python3
"""
Batch persona inference script.

Runs inference across multiple personas and prompts, computing
entropy and top-k mass metrics for generated responses.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.template import patch_chat_template

from .config import BatchInferenceConfig
from .metrics import (
    compute_entropy,
    compute_entropy_and_top_k_mass,
    compute_metrics_for_sequence,
    compute_metrics_for_vllm_output,
    compute_top_k_mass,
)
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


def subset_prompts(prompts: list[PromptInfo], fraction: float) -> list[PromptInfo]:
    """Subset prompts by taking a fraction (rounded up) from the list."""
    if fraction >= 1.0:
        return prompts
    count = math.ceil(len(prompts) * fraction)
    return prompts[:count]


def subset_personae_by_category(personae: list[PersonaInfo], fraction: float) -> list[PersonaInfo]:
    """
    Subset personae by taking a fraction (rounded up) from each category.

    Args:
        personae: List of PersonaInfo objects
        fraction: Fraction of personae to keep per category (rounded up)

    Returns:
        Subsetted list preserving order within categories
    """
    if fraction >= 1.0:
        return personae

    # Group personae by category while preserving order
    categories: dict[str, list[PersonaInfo]] = {}
    for p in personae:
        if p.category not in categories:
            categories[p.category] = []
        categories[p.category].append(p)

    # Take fraction (rounded up) from each category
    result = []
    for category, category_personae in categories.items():
        count = math.ceil(len(category_personae) * fraction)
        result.extend(category_personae[:count])

    return result


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


def get_task_key(task: InferenceTask) -> tuple[str, int, int]:
    """Get the unique key for a task: (persona, prompt_id, rep_idx)."""
    return (task.persona_info.persona, task.prompt_info.prompt_id, task.rep_idx)


def get_result_key(result: dict) -> tuple[str, int, int]:
    """Get the unique key for a result: (persona, prompt_id, rep_idx)."""
    return (result["persona"], result["prompt_id"], result["rep_idx"])


def load_completed_tasks(output_dir: Path) -> set[tuple[str, int, int]]:
    """Load set of completed task keys from existing results.jsonl.

    Returns:
        Set of (persona, prompt_id, rep_idx) tuples for completed tasks.
    """
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        return set()

    completed = set()
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            completed.add(get_result_key(result))

    return completed


def configs_match(saved_config: dict, current_config: "BatchInferenceConfig") -> bool:
    """Check if saved config matches current config (ignoring output_dir).

    Args:
        saved_config: Dictionary loaded from saved config.json
        current_config: Current BatchInferenceConfig instance

    Returns:
        True if configs match (excluding output_dir), False otherwise.
    """
    current_dict = current_config.to_comparable_dict()

    # Compare all keys except output_dir (which was already excluded)
    saved_comparable = {k: v for k, v in saved_config.items() if k != "output_dir"}

    return current_dict == saved_comparable


def filter_completed_tasks(
    tasks: list[InferenceTask], completed: set[tuple[str, int, int]]
) -> list[InferenceTask]:
    """Remove tasks that have already been completed.

    Args:
        tasks: List of all tasks to run
        completed: Set of (persona, prompt_id, rep_idx) keys for completed tasks

    Returns:
        Filtered list of tasks that still need to be run.
    """
    return [t for t in tasks if get_task_key(t) not in completed]


def append_results(output_dir: Path, results: list[dict]) -> None:
    """Append batch of results to results.jsonl (one JSON object per line).

    Args:
        output_dir: Output directory containing results.jsonl
        results: List of result dictionaries to append
    """
    results_path = output_dir / "results.jsonl"
    with open(results_path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def save_config(cfg: "BatchInferenceConfig") -> None:
    """Save configuration to output directory."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")


class VLLMInferenceRunner:
    """Runs batch inference with persona support using vLLM."""

    def __init__(self, cfg: BatchInferenceConfig, output_dir: Path):
        # Import vLLM lazily to avoid import errors when not using this backend
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        self.output_dir = output_dir

        print(f"Loading model {cfg.model} with vLLM...")

        # Load tokenizer (still from transformers for chat template)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        # Set up left padding for batching (used for chat template)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Store original template for persona patching
        self._original_template = self.tokenizer.chat_template

        # Resolve logprobs_k: -1 means full vocab
        if cfg.logprobs_k == -1:
            self.logprobs_k = self.tokenizer.vocab_size
            print(f"logprobs_k=-1 resolved to vocab_size={self.logprobs_k}")
        else:
            self.logprobs_k = cfg.logprobs_k

        # Initialize vLLM
        # max_logprobs=-1 removes the cap, allowing full vocab logprobs for accurate entropy
        self.llm = LLM(
            model=cfg.model,
            dtype="auto",
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_logprobs=-1,
        )

        # Sampling params with logprobs
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            logprobs=self.logprobs_k,
        )

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
        Uses vLLM for efficient batched inference with logprobs for metrics.

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

        # Prepare input texts (prompts for vLLM)
        prompts = [
            self.prepare_input(t.system_prompt, t.prompt_info.question)
            for t in tasks
        ]

        # vLLM generation
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Process outputs
        results = []
        for task, output in zip(tasks, outputs):
            generation = output.outputs[0]
            token_ids = list(generation.token_ids)
            response = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            # Compute metrics from logprobs
            metrics = compute_metrics_for_vllm_output(
                logprobs=generation.logprobs,
                token_ids=token_ids,
                think_end_token_id=THINK_END_TOKEN_ID,
                top_k_mass_k=self.cfg.top_k_mass_k,
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

    def run_all(self, tasks: list[InferenceTask]) -> None:
        """Run inference on all tasks, batching by persona.

        Results are written incrementally to results.jsonl after each batch.
        """
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
                    # Write results incrementally
                    append_results(self.output_dir, batch_results)
                    pbar.update(1)


class TransformersInferenceRunner:
    """Runs batch inference with persona support using Transformers + FlashAttention2."""

    def __init__(self, cfg: BatchInferenceConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir

        print(f"Loading model {cfg.model} with transformers + FlashAttention2...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        # Set up left padding for batching
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Store original template for persona patching
        self._original_template = self.tokenizer.chat_template

        # Load model with FlashAttention2 (fallback to default if unavailable)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            print("Loaded model with FlashAttention2")
        except Exception as e:
            print(f"FlashAttention2 unavailable ({e}), using default attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
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
        Uses incremental generation to compute metrics per-step without
        storing all logits in memory.

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

        # Tokenize with left-padding
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        batch_size = len(tasks)
        input_ids = inputs.input_ids
        input_len = input_ids.shape[1]

        # Pre-allocate attention mask to avoid repeated concatenation
        max_total_len = input_len + self.cfg.max_tokens
        attention_mask = torch.zeros(
            (batch_size, max_total_len),
            device=inputs.attention_mask.device,
            dtype=inputs.attention_mask.dtype,
        )
        attention_mask[:, :input_len] = inputs.attention_mask
        current_len = input_len

        # Per-sequence tracking (store only scalars, not tensors)
        generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        entropies: list[list[float]] = [[] for _ in range(batch_size)]
        top_k_masses: list[list[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        # Incremental generation with KV caching
        past_key_values = None

        with torch.inference_mode():
            for step in range(self.cfg.max_tokens):
                if all(finished):
                    break

                # Forward pass (use only the valid portion of attention mask)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask[:, :current_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Get logits for last position only
                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                past_key_values = outputs.past_key_values

                # Compute metrics on GPU with fused operation (single softmax)
                step_entropies, step_top_k = compute_entropy_and_top_k_mass(
                    logits, self.cfg.top_k_mass_k
                )

                # Sample next token
                if self.cfg.temperature > 0:
                    probs = F.softmax(logits / self.cfg.temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = logits.argmax(dim=-1)

                # Store results per sequence (scalars only)
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_tokens[i].item()
                        generated_tokens[i].append(token_id)
                        entropies[i].append(step_entropies[i].item())
                        top_k_masses[i].append(step_top_k[i].item())

                        if token_id == self.tokenizer.eos_token_id:
                            finished[i] = True

                # Prepare for next step: update attention mask in-place
                input_ids = next_tokens.unsqueeze(-1)
                attention_mask[:, current_len] = 1
                current_len += 1

        # Build results from accumulated scalars
        results = []
        for i, task in enumerate(tasks):
            tokens = generated_tokens[i]
            ents = entropies[i]
            top_ks = top_k_masses[i]

            # Trim EOS token if present
            if tokens and tokens[-1] == self.tokenizer.eos_token_id:
                tokens = tokens[:-1]
                ents = ents[:-1]
                top_ks = top_ks[:-1]

            # Find think_end position
            think_end_pos = None
            for j, tid in enumerate(tokens):
                if tid == THINK_END_TOKEN_ID:
                    think_end_pos = j
                    break

            # Compute averages for thinking/output sections
            if think_end_pos is not None:
                thinking_ents = ents[:think_end_pos + 1]
                thinking_top_k = top_ks[:think_end_pos + 1]
                output_ents = ents[think_end_pos + 1:]
                output_top_k = top_ks[think_end_pos + 1:]

                avg_entropy_thinking = sum(thinking_ents) / len(thinking_ents) if thinking_ents else None
                avg_top_k_mass_thinking = sum(thinking_top_k) / len(thinking_top_k) if thinking_top_k else None
                avg_entropy_output = sum(output_ents) / len(output_ents) if output_ents else None
                avg_top_k_mass_output = sum(output_top_k) / len(output_top_k) if output_top_k else None
            else:
                avg_entropy_thinking = None
                avg_top_k_mass_thinking = None
                avg_entropy_output = sum(ents) / len(ents) if ents else None
                avg_top_k_mass_output = sum(top_ks) / len(top_ks) if top_ks else None

            response = self.tokenizer.decode(tokens, skip_special_tokens=True)

            results.append({
                "persona": task.persona_info.persona,
                "category": task.persona_info.category,
                "article": task.persona_info.article,
                "system_prompt": task.system_prompt,
                "prompt_id": task.prompt_info.prompt_id,
                "prompt": task.prompt_info.question,
                "rep_idx": task.rep_idx,
                "response": response,
                "avg_entropy_thinking": avg_entropy_thinking,
                "avg_entropy_output": avg_entropy_output,
                "avg_entropy": sum(ents) / len(ents) if ents else None,
                "avg_top_k_mass_thinking": avg_top_k_mass_thinking,
                "avg_top_k_mass_output": avg_top_k_mass_output,
                "avg_top_k_mass": sum(top_ks) / len(top_ks) if top_ks else None,
                "think_end_position": think_end_pos,
                "num_tokens": len(tokens),
            })

        return results

    def run_all(self, tasks: list[InferenceTask]) -> None:
        """Run inference on all tasks, batching by persona.

        Results are written incrementally to results.jsonl after each batch.
        """
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
                    # Write results incrementally
                    append_results(self.output_dir, batch_results)
                    pbar.update(1)


def create_inference_runner(cfg: BatchInferenceConfig, output_dir: Path):
    """Factory function to create the appropriate inference runner."""
    if cfg.backend == "transformers":
        return TransformersInferenceRunner(cfg, output_dir)
    else:
        return VLLMInferenceRunner(cfg, output_dir)


def count_results(output_dir: Path) -> int:
    """Count the number of results in results.jsonl."""
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        return 0

    count = 0
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


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
        default=256,
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
        "--logprobs-k",
        type=int,
        default=10000,
        help="Number of top logprobs for entropy computation (-1 for full vocab)",
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
    parser.add_argument(
        "--subset-category-persona",
        type=float,
        default=1.0,
        help="Fraction of personae to use per category (rounded up)",
    )
    parser.add_argument(
        "--subset-prompt",
        type=float,
        default=1.0,
        help="Fraction of prompts to use",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend: 'transformers' (accurate full-vocab entropy) or 'vllm' (fast, approximate entropy)",
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
        logprobs_k=args.logprobs_k,
        thinking_mode=args.thinking_mode,
        system_prompt_style=args.system_prompt_style,
        output_dir=args.output_dir,
        subset_category_persona=args.subset_category_persona,
        subset_prompt=args.subset_prompt,
        backend=args.backend,
    )

    print(f"Configuration: {cfg.to_dict()}")

    # Check for resume
    output_dir = Path(cfg.output_dir)
    completed_keys: set[tuple[str, int, int]] = set()

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            saved_config = json.load(f)

        if configs_match(saved_config, cfg):
            completed_keys = load_completed_tasks(output_dir)
            print(f"Resuming: {len(completed_keys)} tasks already completed")
        else:
            print("Error: Config mismatch with existing results.")
            print("Either delete the existing results directory or use a different --output-dir.")
            print(f"  Existing results: {output_dir}")
            sys.exit(1)
    else:
        # Save config at start (new run)
        save_config(cfg)

    # Load inputs
    prompts = load_prompts(cfg.prompts_json)
    print(f"Loaded {len(prompts)} prompts")

    personae = load_personae(cfg.personae_json)
    print(f"Loaded {len(personae)} personae")

    # Apply subsetting
    if cfg.subset_prompt < 1.0:
        prompts = subset_prompts(prompts, cfg.subset_prompt)
        print(f"Subsetted to {len(prompts)} prompts (fraction={cfg.subset_prompt})")

    if cfg.subset_category_persona < 1.0:
        personae = subset_personae_by_category(personae, cfg.subset_category_persona)
        print(f"Subsetted to {len(personae)} personae (fraction={cfg.subset_category_persona} per category)")

    # Build task list
    tasks = build_tasks(prompts, personae, cfg.n_reps, cfg.system_prompt_style)
    total_tasks = len(tasks)
    print(f"Created {total_tasks} total tasks")

    # Filter out completed tasks
    if completed_keys:
        tasks = filter_completed_tasks(tasks, completed_keys)
        print(f"Remaining tasks after resume: {len(tasks)} (skipped {len(completed_keys)})")

    if not tasks:
        print("All tasks already completed!")
        final_count = count_results(output_dir)
        print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")
        return

    # Run inference with GPU monitoring
    runner = create_inference_runner(cfg, output_dir)

    if args.gpu_monitor_interval > 0:
        with GPUMonitor(interval=args.gpu_monitor_interval):
            runner.run_all(tasks)
    else:
        runner.run_all(tasks)

    # Report final results
    final_count = count_results(output_dir)
    print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")
    print("Done!")


if __name__ == "__main__":
    main()
