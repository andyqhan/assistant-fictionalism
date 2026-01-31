#!/usr/bin/env python3
"""
Consistency inference script for multi-turn persona experiments.

Tests how consistent different personae are when asked to explain their choices
through multi-turn conversations:
1. Ask a question and get a response (turn 1)
2. Ask a follow-up question in the same conversation (turn 2)
3. Compute entropy and top-k mass metrics from logprobs
4. Compute embeddings on all responses after generation completes
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

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.template import patch_chat_template

from .embeddings import extract_output, extract_thinking, last_token_pool
from .metrics import compute_metrics_for_vllm_output
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
                            f"Temp: {temp}C",
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
class ConsistencyConfig:
    """Configuration for consistency inference."""

    prompts_jsonl: str
    personae_json: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.7
    max_tokens: int = 1024
    batch_size: int = 256
    n_samples: int = 100
    top_k_mass_k: int = 20
    logprobs_k: int = 20
    thinking_mode: bool = True
    system_prompt_style: str = "you-are-a"
    output_dir: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_batch_size: int = 16
    embedding_max_length: int = 2048

    def __post_init__(self) -> None:
        # Validate paths exist
        assert os.path.exists(self.prompts_jsonl), f"Prompts file not found: {self.prompts_jsonl}"
        assert os.path.exists(self.personae_json), f"Personae file not found: {self.personae_json}"

        # Validate numeric parameters
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.n_samples > 0, f"n_samples must be positive, got {self.n_samples}"
        assert self.top_k_mass_k > 0, f"top_k_mass_k must be positive, got {self.top_k_mass_k}"
        assert self.logprobs_k > 0 or self.logprobs_k == -1, f"logprobs_k must be positive or -1, got {self.logprobs_k}"
        assert self.embedding_batch_size > 0, f"embedding_batch_size must be positive, got {self.embedding_batch_size}"

        # Validate system prompt style
        valid_styles = ["you-are-a"]
        assert self.system_prompt_style in valid_styles, f"Invalid system_prompt_style: {self.system_prompt_style}"

        # Handle n_samples with temperature=0
        if self.temperature == 0.0 and self.n_samples > 1:
            print(f"Warning: n_samples={self.n_samples} with temperature=0.0 is redundant. Forcing n_samples=1.")
            self.n_samples = 1

        # Warn about thinking mode with temperature=0
        if self.thinking_mode and self.temperature == 0.0:
            print("Warning: thinking_mode=True with temperature=0.0 may produce deterministic thinking.")

        # Auto-generate output directory if not specified
        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/consistency-inference-{slurm_job_id}"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "prompts_jsonl": self.prompts_jsonl,
            "personae_json": self.personae_json,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "top_k_mass_k": self.top_k_mass_k,
            "logprobs_k": self.logprobs_k,
            "thinking_mode": self.thinking_mode,
            "system_prompt_style": self.system_prompt_style,
            "output_dir": self.output_dir,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_length": self.embedding_max_length,
        }

    def to_comparable_dict(self) -> dict:
        """Convert config to dictionary for comparison, excluding output_dir.

        Used for resume detection - configs match if all parameters except
        output_dir are identical.
        """
        d = self.to_dict()
        del d["output_dir"]
        return d


@dataclass
class ConsistencyPrompt:
    """A consistency prompt with question and follow-up."""

    question: str
    follow_up: str
    prompt_id: int


@dataclass
class PersonaInfo:
    """Information about a persona."""

    persona: str
    category: str
    article: str


@dataclass
class ConsistencyTask:
    """A single consistency inference task to run."""

    persona_info: PersonaInfo
    system_prompt: str
    prompt: ConsistencyPrompt
    sample_idx: int


def load_consistency_prompts(path: str) -> list[ConsistencyPrompt]:
    """Load consistency prompts from JSONL file.

    Each line must have "question" and "follow_up" fields.
    Optional "id" field, defaults to line number.
    """
    prompts = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "question" in obj, f"Line {line_num + 1}: Missing 'question' key"
            assert "follow_up" in obj, f"Line {line_num + 1}: Missing 'follow_up' key"
            prompt_id = obj.get("id", line_num)
            prompts.append(ConsistencyPrompt(
                question=obj["question"],
                follow_up=obj["follow_up"],
                prompt_id=prompt_id,
            ))

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


def build_consistency_tasks(
    prompts: list[ConsistencyPrompt],
    personae: list[PersonaInfo],
    n_samples: int,
    system_prompt_style: str,
) -> list[ConsistencyTask]:
    """Build list of all consistency inference tasks to run."""
    tasks = []

    for persona_info in personae:
        system_prompt = generate_system_prompt(
            persona_info.persona,
            persona_info.article,
            system_prompt_style,
        )

        for prompt in prompts:
            for sample_idx in range(n_samples):
                tasks.append(ConsistencyTask(
                    persona_info=persona_info,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    sample_idx=sample_idx,
                ))

    return tasks


def get_task_key(task: ConsistencyTask) -> tuple[str, int, int]:
    """Get the unique key for a task: (persona, prompt_id, sample_idx)."""
    return (task.persona_info.persona, task.prompt.prompt_id, task.sample_idx)


def get_result_key(result: dict) -> tuple[str, int, int]:
    """Get the unique key for a result: (persona, prompt_id, sample_idx)."""
    return (result["persona"], result["prompt_id"], result["sample_idx"])


def load_completed_tasks(output_dir: Path) -> set[tuple[str, int, int]]:
    """Load set of completed task keys from existing results.jsonl.

    Returns:
        Set of (persona, prompt_id, sample_idx) tuples for completed tasks.
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


def configs_match(saved_config: dict, current_config: ConsistencyConfig) -> bool:
    """Check if saved config matches current config (ignoring output_dir).

    Args:
        saved_config: Dictionary loaded from saved config.json
        current_config: Current ConsistencyConfig instance

    Returns:
        True if configs match (excluding output_dir), False otherwise.
    """
    current_dict = current_config.to_comparable_dict()

    # Compare all keys except output_dir (which was already excluded)
    saved_comparable = {k: v for k, v in saved_config.items() if k != "output_dir"}

    return current_dict == saved_comparable


def filter_completed_tasks(
    tasks: list[ConsistencyTask], completed: set[tuple[str, int, int]]
) -> list[ConsistencyTask]:
    """Remove tasks that have already been completed.

    Args:
        tasks: List of all tasks to run
        completed: Set of (persona, prompt_id, sample_idx) keys for completed tasks

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


def save_config(cfg: ConsistencyConfig) -> None:
    """Save configuration to output directory."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")


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


class ConsistencyInferenceRunner:
    """Runs multi-turn consistency inference with persona support using vLLM."""

    def __init__(self, cfg: ConsistencyConfig, output_dir: Path):
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

    def prepare_turn1_input(self, system_prompt: str, question: str) -> str:
        """Prepare input text for turn 1 (system + question)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )

        return text

    def prepare_turn2_input(
        self, system_prompt: str, question: str, response1: str, follow_up: str
    ) -> str:
        """Prepare input text for turn 2 (system + question + response1 + follow_up)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": follow_up},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )

        return text

    def run_batch(self, tasks: list[ConsistencyTask]) -> list[dict]:
        """
        Run multi-turn inference on a batch of tasks with the same persona.

        All tasks must have the same persona (for consistent template).
        Uses vLLM for efficient batched inference with logprobs for metrics.

        Phase 1: Generate turn 1 responses for all tasks
        Phase 2: Generate turn 2 responses using turn 1 responses

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

        # Phase 1: Turn 1 generation
        turn1_prompts = [
            self.prepare_turn1_input(t.system_prompt, t.prompt.question)
            for t in tasks
        ]

        turn1_outputs = self.llm.generate(turn1_prompts, self.sampling_params)

        # Process turn 1 outputs and prepare turn 2 inputs
        turn1_responses = []
        turn1_metrics = []

        for task, output in zip(tasks, turn1_outputs):
            generation = output.outputs[0]
            token_ids = list(generation.token_ids)
            response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            turn1_responses.append(response)

            # Compute metrics from logprobs
            metrics = compute_metrics_for_vllm_output(
                logprobs=generation.logprobs,
                token_ids=token_ids,
                think_end_token_id=THINK_END_TOKEN_ID,
                top_k_mass_k=self.cfg.top_k_mass_k,
            )
            turn1_metrics.append(metrics)

        # Phase 2: Turn 2 generation
        turn2_prompts = [
            self.prepare_turn2_input(
                t.system_prompt,
                t.prompt.question,
                turn1_responses[i],
                t.prompt.follow_up,
            )
            for i, t in enumerate(tasks)
        ]

        turn2_outputs = self.llm.generate(turn2_prompts, self.sampling_params)

        # Process turn 2 outputs and build final results
        results = []

        for i, (task, output) in enumerate(zip(tasks, turn2_outputs)):
            generation = output.outputs[0]
            token_ids = list(generation.token_ids)
            response2 = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            # Compute metrics from logprobs
            metrics2 = compute_metrics_for_vllm_output(
                logprobs=generation.logprobs,
                token_ids=token_ids,
                think_end_token_id=THINK_END_TOKEN_ID,
                top_k_mass_k=self.cfg.top_k_mass_k,
            )

            results.append({
                # Metadata
                "persona": task.persona_info.persona,
                "category": task.persona_info.category,
                "article": task.persona_info.article,
                "system_prompt": task.system_prompt,
                "prompt_id": task.prompt.prompt_id,
                "question": task.prompt.question,
                "follow_up": task.prompt.follow_up,
                "sample_idx": task.sample_idx,

                # Turn 1 results
                "response1": turn1_responses[i],
                "response1_avg_entropy_thinking": turn1_metrics[i]["avg_entropy_thinking"],
                "response1_avg_entropy_output": turn1_metrics[i]["avg_entropy_output"],
                "response1_avg_entropy": turn1_metrics[i]["avg_entropy"],
                "response1_avg_top_k_mass_thinking": turn1_metrics[i]["avg_top_k_mass_thinking"],
                "response1_avg_top_k_mass_output": turn1_metrics[i]["avg_top_k_mass_output"],
                "response1_avg_top_k_mass": turn1_metrics[i]["avg_top_k_mass"],
                "response1_think_end_position": turn1_metrics[i]["think_end_position"],
                "response1_num_tokens": turn1_metrics[i]["num_tokens"],

                # Turn 2 results
                "response2": response2,
                "response2_avg_entropy_thinking": metrics2["avg_entropy_thinking"],
                "response2_avg_entropy_output": metrics2["avg_entropy_output"],
                "response2_avg_entropy": metrics2["avg_entropy"],
                "response2_avg_top_k_mass_thinking": metrics2["avg_top_k_mass_thinking"],
                "response2_avg_top_k_mass_output": metrics2["avg_top_k_mass_output"],
                "response2_avg_top_k_mass": metrics2["avg_top_k_mass"],
                "response2_think_end_position": metrics2["think_end_position"],
                "response2_num_tokens": metrics2["num_tokens"],
            })

        return results

    def run_all(self, tasks: list[ConsistencyTask]) -> None:
        """Run inference on all tasks, batching by persona.

        Results are written incrementally to results.jsonl after each batch.
        """
        # Group tasks by persona
        tasks_by_persona: dict[str, list[ConsistencyTask]] = {}
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


class EmbeddingRunner:
    """Computes embeddings for consistency inference results."""

    def __init__(self, cfg: ConsistencyConfig):
        self.cfg = cfg

        print(f"Loading embedding model {cfg.embedding_model}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        # Set left padding (required for last_token_pool)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with FlashAttention2 if available
        try:
            self.model = AutoModel.from_pretrained(
                cfg.embedding_model,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            print("Loaded embedding model with FlashAttention2")
        except Exception as e:
            print(f"FlashAttention2 unavailable ({e}), using default attention")
            self.model = AutoModel.from_pretrained(
                cfg.embedding_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

        print("Embedding model loaded successfully.")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Compute normalized embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (as Python lists of floats)
        """
        if not texts:
            return []

        # Handle empty strings by replacing with a space
        texts = [t if t.strip() else " " for t in texts]

        # Tokenize with left-padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.embedding_max_length,
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs.attention_mask)

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to Python lists
        return embeddings.cpu().float().tolist()


def compute_all_embeddings(cfg: ConsistencyConfig, output_dir: Path) -> None:
    """Compute embeddings for all results and save to parquet.

    Reads results.jsonl, computes embeddings for response1 and response2
    (full, thinking, and output sections), and saves to embeddings.parquet.
    """
    results_path = output_dir / "results.jsonl"
    output_path = output_dir / "embeddings.parquet"

    print(f"Loading results from {results_path}...")

    # Load all results
    rows = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows")

    if len(rows) == 0:
        print("No rows to process, skipping embedding computation")
        return

    # Initialize embedding runner
    runner = EmbeddingRunner(cfg)

    # Extract texts for 6 embedding types
    texts_response1_full = [r["response1"] for r in rows]
    texts_response1_thinking = [extract_thinking(r["response1"]) for r in rows]
    texts_response1_output = [extract_output(r["response1"]) for r in rows]
    texts_response2_full = [r["response2"] for r in rows]
    texts_response2_thinking = [extract_thinking(r["response2"]) for r in rows]
    texts_response2_output = [extract_output(r["response2"]) for r in rows]

    num_batches = (len(rows) + cfg.embedding_batch_size - 1) // cfg.embedding_batch_size

    # Compute embeddings for each text type
    def compute_embeddings_for_texts(texts: list[str], desc: str) -> list[list[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), cfg.embedding_batch_size), total=num_batches, desc=desc):
            batch_texts = texts[i : i + cfg.embedding_batch_size]
            batch_embeddings = runner.embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    print("Computing response1 full embeddings...")
    embeddings_response1_full = compute_embeddings_for_texts(
        texts_response1_full, "response1_full"
    )

    print("Computing response1 thinking embeddings...")
    embeddings_response1_thinking = compute_embeddings_for_texts(
        texts_response1_thinking, "response1_thinking"
    )

    print("Computing response1 output embeddings...")
    embeddings_response1_output = compute_embeddings_for_texts(
        texts_response1_output, "response1_output"
    )

    print("Computing response2 full embeddings...")
    embeddings_response2_full = compute_embeddings_for_texts(
        texts_response2_full, "response2_full"
    )

    print("Computing response2 thinking embeddings...")
    embeddings_response2_thinking = compute_embeddings_for_texts(
        texts_response2_thinking, "response2_thinking"
    )

    print("Computing response2 output embeddings...")
    embeddings_response2_output = compute_embeddings_for_texts(
        texts_response2_output, "response2_output"
    )

    # Build output dataframe
    output_rows = []
    for i, row in enumerate(rows):
        output_row = {
            # Text columns
            "text_response1": row["response1"],
            "text_response1_thinking": texts_response1_thinking[i],
            "text_response1_output": texts_response1_output[i],
            "text_response2": row["response2"],
            "text_response2_thinking": texts_response2_thinking[i],
            "text_response2_output": texts_response2_output[i],
            # Embedding columns
            "embedding_response1_full": embeddings_response1_full[i],
            "embedding_response1_thinking": embeddings_response1_thinking[i],
            "embedding_response1_output": embeddings_response1_output[i],
            "embedding_response2_full": embeddings_response2_full[i],
            "embedding_response2_thinking": embeddings_response2_thinking[i],
            "embedding_response2_output": embeddings_response2_output[i],
            # Preserve all original metadata columns (except response1/response2 which are in text_ columns)
            **{k: v for k, v in row.items() if k not in ("response1", "response2")},
        }
        output_rows.append(output_row)

    df = pd.DataFrame(output_rows)

    # Save to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved embeddings to {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Embedding dimension: {len(embeddings_response1_full[0])}")
    print(f"  Columns: {list(df.columns)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Consistency inference for multi-turn persona experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompts-jsonl",
        type=str,
        default="datasets/consistency_prompts.jsonl",
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--personae-json",
        type=str,
        default="datasets/personae_small.json",
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
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per turn",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per (prompt, persona) pair",
    )
    parser.add_argument(
        "--top-k-mass-k",
        type=int,
        default=20,
        help="k for top-k mass computation",
    )
    parser.add_argument(
        "--logprobs-k",
        type=int,
        default=20,
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
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model ID",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=2048,
        help="Max sequence length for embedding tokenization",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Skip generation phase and only compute embeddings (requires existing results.jsonl)",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between GPU utilization logs (0 to disable)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for consistency inference."""
    args = parse_args()

    # Create config (validates inputs)
    cfg = ConsistencyConfig(
        prompts_jsonl=args.prompts_jsonl,
        personae_json=args.personae_json,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        top_k_mass_k=args.top_k_mass_k,
        logprobs_k=args.logprobs_k,
        thinking_mode=args.thinking_mode,
        system_prompt_style=args.system_prompt_style,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
    )

    print(f"Configuration: {cfg.to_dict()}")

    output_dir = Path(cfg.output_dir)

    # Embeddings-only mode: skip generation, just compute embeddings
    if args.embeddings_only:
        results_path = output_dir / "results.jsonl"
        if not results_path.exists():
            print(f"Error: --embeddings-only requires existing results at {results_path}")
            sys.exit(1)
        print(f"Embeddings-only mode: skipping generation, using existing {results_path}")
    else:
        # Check for resume
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
        prompts = load_consistency_prompts(cfg.prompts_jsonl)
        print(f"Loaded {len(prompts)} prompts")

        personae = load_personae(cfg.personae_json)
        print(f"Loaded {len(personae)} personae")

        # Build task list
        tasks = build_consistency_tasks(prompts, personae, cfg.n_samples, cfg.system_prompt_style)
        total_tasks = len(tasks)
        print(f"Created {total_tasks} total tasks")

        # Filter out completed tasks
        if completed_keys:
            tasks = filter_completed_tasks(tasks, completed_keys)
            print(f"Remaining tasks after resume: {len(tasks)} (skipped {len(completed_keys)})")

        # Phase 1: Generation
        if tasks:
            print("\n=== Phase 1: Generation ===")
            runner = ConsistencyInferenceRunner(cfg, output_dir)

            if args.gpu_monitor_interval > 0:
                with GPUMonitor(interval=args.gpu_monitor_interval):
                    runner.run_all(tasks)
            else:
                runner.run_all(tasks)

            # Report generation results
            final_count = count_results(output_dir)
            print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")
        else:
            print("All generation tasks already completed!")
            final_count = count_results(output_dir)
            print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")

    # Phase 2: Embedding computation
    print("\n=== Phase 2: Embedding Computation ===")
    compute_all_embeddings(cfg, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
