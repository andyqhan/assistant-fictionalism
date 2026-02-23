#!/usr/bin/env python3
"""
User-turn prediction experiment for persona inference.

Flips the standard assistant-generation paradigm: the model generates **user** tokens
while the assistant role contains fixed questions. This lets us study what the model
predicts a persona would say when positioned as the user in a conversation.

Transcript structure:
    User: Hello. I'm an assistant. Did you have some questions for me?
    Assistant: Should moral principles be absolute, or should they depend on circumstances?
    User: [GENERATE]
    Assistant: Why?
    User: [GENERATE]

Uses transformers backend for full logit distribution access (entropy/top-k-mass metrics).
"""

import argparse
import json
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

from .consistency import (
    ConsistencyPrompt,
    PersonaInfo,
    load_consistency_prompts,
    load_personae,
)
from .metrics import compute_entropy_and_top_k_mass
from .system_prompts import generate_system_prompt


# Qwen3 special token IDs
THINK_END_TOKEN_ID = 151668   # </think>
IM_END_TOKEN_ID = 151645      # <|im_end|>
ENDOFTEXT_TOKEN_ID = 151643   # <|endoftext|>

# Stop tokens for user-turn generation
STOP_TOKEN_IDS = {IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID}


class GPUMonitor:
    """Background thread that periodically logs GPU utilization and memory."""

    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _log_gpu_stats(self) -> None:
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
        while not self._stop_event.is_set():
            self._log_gpu_stats()
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[GPU Monitor] Started (interval: {self.interval}s)", flush=True)

    def stop(self) -> None:
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
class UserTurnPredictionConfig:
    """Configuration for user-turn prediction experiment."""

    prompts_jsonl: str
    personae_json: str
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer: str = ""  # Tokenizer model ID (defaults to model if empty)
    temperature: float = 0.7
    max_tokens: int = 512
    batch_size: int = 256
    n_samples: int = 100
    top_k_mass_k: int = 20
    thinking_mode: bool = False
    think_in_user_turn: bool = False
    patch_user_role: bool = False
    system_prompt_style: str = ""
    output_dir: str = ""
    gpu_monitor_interval: float = 30.0

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

        # Validate system prompt style
        valid_styles = ["", "you-are-a"]
        assert self.system_prompt_style in valid_styles, (
            f"Invalid system_prompt_style: {self.system_prompt_style!r}, must be one of {valid_styles}"
        )

        # Handle n_samples with temperature=0
        if self.temperature == 0.0 and self.n_samples > 1:
            print(f"Warning: n_samples={self.n_samples} with temperature=0.0 is redundant. Forcing n_samples=1.")
            self.n_samples = 1

        # Auto-generate output directory if not specified
        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/user-turn-prediction-{slurm_job_id}"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "prompts_jsonl": self.prompts_jsonl,
            "personae_json": self.personae_json,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "top_k_mass_k": self.top_k_mass_k,
            "thinking_mode": self.thinking_mode,
            "think_in_user_turn": self.think_in_user_turn,
            "patch_user_role": self.patch_user_role,
            "system_prompt_style": self.system_prompt_style,
            "output_dir": self.output_dir,
            "gpu_monitor_interval": self.gpu_monitor_interval,
        }

    def to_comparable_dict(self) -> dict:
        """Convert config to dictionary for comparison, excluding output_dir."""
        d = self.to_dict()
        del d["output_dir"]
        return d


@dataclass
class UserTurnTask:
    """A single user-turn prediction task."""

    persona_info: PersonaInfo
    prompt: ConsistencyPrompt
    sample_idx: int


def generate_intro_message(persona: str, article: str) -> str:
    """Generate the intro message for the user turn.

    Examples:
        article="an"   -> "Hello. I'm an assistant. Did you have some questions for me?"
        article="a"    -> "Hello. I'm a helper. Did you have some questions for me?"
        article="none" -> "Hello. I'm Hamlet. Did you have some questions for me?"
        article=""     -> "Hello. I'm Hamlet. Did you have some questions for me?"
    """
    if article and article != "none":
        return f"Hello. I'm {article} {persona}. Did you have some questions for me?"
    else:
        return f"Hello. I'm {persona}. Did you have some questions for me?"


def build_turn1_prompt(
    intro_message: str,
    question: str,
    user_role: str,
    think_in_user_turn: bool,
    system_prompt: str = "",
) -> str:
    """Build the raw prompt string for turn 1 user-turn generation.

    Template (without system prompt):
        <|im_start|>{user_role}
        {intro_message}<|im_end|>
        <|im_start|>assistant
        {question}<|im_end|>
        <|im_start|>{user_role}
        {think_prefix}

    Template (with system prompt):
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>{user_role}
        {intro_message}<|im_end|>
        ...
    """
    think_prefix = "<think>\n" if think_in_user_turn else ""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    parts.extend([
        f"<|im_start|>{user_role}\n{intro_message}<|im_end|>",
        f"<|im_start|>assistant\n{question}<|im_end|>",
        f"<|im_start|>{user_role}\n{think_prefix}",
    ])
    return "\n".join(parts)


def build_turn2_prompt(
    intro_message: str,
    question: str,
    response1_text: str,
    follow_up: str,
    user_role: str,
    think_in_user_turn: bool,
    system_prompt: str = "",
) -> str:
    """Build the raw prompt string for turn 2 user-turn generation.

    Template (without system prompt):
        <|im_start|>{user_role}
        {intro_message}<|im_end|>
        <|im_start|>assistant
        {question}<|im_end|>
        <|im_start|>{user_role}
        {response1_text}<|im_end|>
        <|im_start|>assistant
        {follow_up}<|im_end|>
        <|im_start|>{user_role}
        {think_prefix}

    Template (with system prompt):
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>{user_role}
        {intro_message}<|im_end|>
        ...
    """
    think_prefix = "<think>\n" if think_in_user_turn else ""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    parts.extend([
        f"<|im_start|>{user_role}\n{intro_message}<|im_end|>",
        f"<|im_start|>assistant\n{question}<|im_end|>",
        f"<|im_start|>{user_role}\n{response1_text}<|im_end|>",
        f"<|im_start|>assistant\n{follow_up}<|im_end|>",
        f"<|im_start|>{user_role}\n{think_prefix}",
    ])
    return "\n".join(parts)


def build_tasks(
    prompts: list[ConsistencyPrompt],
    personae: list[PersonaInfo],
    n_samples: int,
) -> list[UserTurnTask]:
    """Build list of all user-turn prediction tasks to run."""
    tasks = []
    for persona_info in personae:
        for prompt in prompts:
            for sample_idx in range(n_samples):
                tasks.append(UserTurnTask(
                    persona_info=persona_info,
                    prompt=prompt,
                    sample_idx=sample_idx,
                ))
    return tasks


def get_task_key(task: UserTurnTask) -> tuple[str, int, int]:
    """Get the unique key for a task: (persona, prompt_id, sample_idx)."""
    return (task.persona_info.persona, task.prompt.prompt_id, task.sample_idx)


def get_result_key(result: dict) -> tuple[str, int, int]:
    """Get the unique key for a result: (persona, prompt_id, sample_idx)."""
    return (result["persona"], result["prompt_id"], result["sample_idx"])


def load_completed_tasks(output_dir: Path) -> set[tuple[str, int, int]]:
    """Load set of completed task keys from existing results.jsonl."""
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


def configs_match(saved_config: dict, current_config: UserTurnPredictionConfig) -> bool:
    """Check if saved config matches current config (ignoring output_dir)."""
    current_dict = current_config.to_comparable_dict()
    saved_comparable = {k: v for k, v in saved_config.items() if k != "output_dir"}
    return current_dict == saved_comparable


def filter_completed_tasks(
    tasks: list[UserTurnTask], completed: set[tuple[str, int, int]]
) -> list[UserTurnTask]:
    """Remove tasks that have already been completed."""
    return [t for t in tasks if get_task_key(t) not in completed]


def append_results(output_dir: Path, results: list[dict]) -> None:
    """Append batch of results to results.jsonl."""
    results_path = output_dir / "results.jsonl"
    with open(results_path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def save_config(cfg: UserTurnPredictionConfig) -> None:
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


class UserTurnPredictionRunner:
    """Runs user-turn prediction with persona support using Transformers + FlashAttention2."""

    def __init__(self, cfg: UserTurnPredictionConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir

        self.tokenizer_id = cfg.tokenizer or cfg.model
        print(f"Loading model {cfg.model} with transformers + FlashAttention2 (tokenizer: {self.tokenizer_id})...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        assert self.tokenizer is not None, "Failed to load tokenizer"
        assert hasattr(self.tokenizer, "chat_template"), "Tokenizer must have chat_template attribute"

        # Set up left padding for batching
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

    def _generate_turn(
        self,
        prompt_texts: list[str],
    ) -> list[dict]:
        """Run incremental generation for a list of prompt strings.

        Returns a list of dicts, one per prompt, with keys:
            - tokens: list[int] (generated token IDs, excluding stop token)
            - text: str (decoded text, skip_special_tokens=True)
            - entropies: list[float] (per-token entropy, excluding stop token)
            - top_k_masses: list[float] (per-token top-k mass, excluding stop token)
        """
        # Tokenize with left-padding
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        batch_size = len(prompt_texts)
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

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask[:, :current_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Get logits for last position only
                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                assert logits.shape[0] == batch_size, (
                    f"Expected batch dim {batch_size}, got {logits.shape[0]}"
                )
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

                        # Check stop tokens: <|im_end|> AND <|endoftext|>
                        if token_id in STOP_TOKEN_IDS:
                            finished[i] = True

                # Prepare for next step: update attention mask in-place
                input_ids = next_tokens.unsqueeze(-1)
                attention_mask[:, current_len] = 1
                current_len += 1

        # Build per-sequence results
        results = []
        for i in range(batch_size):
            tokens = generated_tokens[i]
            ents = entropies[i]
            top_ks = top_k_masses[i]

            assert len(tokens) == len(ents), (
                f"Token/entropy length mismatch: {len(tokens)} vs {len(ents)}"
            )
            assert len(tokens) == len(top_ks), (
                f"Token/top_k_mass length mismatch: {len(tokens)} vs {len(top_ks)}"
            )

            # Trim stop token if present
            if tokens and tokens[-1] in STOP_TOKEN_IDS:
                tokens = tokens[:-1]
                ents = ents[:-1]
                top_ks = top_ks[:-1]

            text = self.tokenizer.decode(tokens, skip_special_tokens=True)

            results.append({
                "tokens": tokens,
                "text": text,
                "entropies": ents,
                "top_k_masses": top_ks,
            })

        return results

    def _compute_turn_metrics(
        self,
        tokens: list[int],
        entropies_list: list[float],
        top_k_masses_list: list[float],
    ) -> dict:
        """Compute turn-level metrics with thinking/output splitting.

        Follows the same logic as inference.py lines 716-737.
        """
        if not tokens:
            return {
                "avg_entropy_thinking": None,
                "avg_entropy_output": None,
                "avg_entropy": None,
                "avg_top_k_mass_thinking": None,
                "avg_top_k_mass_output": None,
                "avg_top_k_mass": None,
                "think_end_position": None,
                "num_tokens": 0,
            }

        # Find </think> position
        think_end_pos = None
        for j, tid in enumerate(tokens):
            if tid == THINK_END_TOKEN_ID:
                think_end_pos = j
                break

        # Compute averages for thinking/output sections
        if think_end_pos is not None:
            thinking_ents = entropies_list[:think_end_pos + 1]
            thinking_top_k = top_k_masses_list[:think_end_pos + 1]
            output_ents = entropies_list[think_end_pos + 1:]
            output_top_k = top_k_masses_list[think_end_pos + 1:]

            avg_entropy_thinking = sum(thinking_ents) / len(thinking_ents) if thinking_ents else None
            avg_top_k_mass_thinking = sum(thinking_top_k) / len(thinking_top_k) if thinking_top_k else None
            avg_entropy_output = sum(output_ents) / len(output_ents) if output_ents else None
            avg_top_k_mass_output = sum(output_top_k) / len(output_top_k) if output_top_k else None
        else:
            avg_entropy_thinking = None
            avg_top_k_mass_thinking = None
            avg_entropy_output = sum(entropies_list) / len(entropies_list) if entropies_list else None
            avg_top_k_mass_output = sum(top_k_masses_list) / len(top_k_masses_list) if top_k_masses_list else None

        return {
            "avg_entropy_thinking": avg_entropy_thinking,
            "avg_entropy_output": avg_entropy_output,
            "avg_entropy": sum(entropies_list) / len(entropies_list) if entropies_list else None,
            "avg_top_k_mass_thinking": avg_top_k_mass_thinking,
            "avg_top_k_mass_output": avg_top_k_mass_output,
            "avg_top_k_mass": sum(top_k_masses_list) / len(top_k_masses_list) if top_k_masses_list else None,
            "think_end_position": think_end_pos,
            "num_tokens": len(tokens),
        }

    def run_batch(self, tasks: list[UserTurnTask]) -> list[dict]:
        """Run two-turn user-turn prediction on a batch of tasks.

        All tasks must have the same persona (for consistent user_role when
        --patch-user-role is on).

        Phase 1: Generate turn 1 responses (user replies to assistant question)
        Phase 2: Generate turn 2 responses (user replies to assistant follow-up)

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

        user_role = persona if self.cfg.patch_user_role else "user"

        # Generate system prompt if style is set
        article = tasks[0].persona_info.article
        if self.cfg.system_prompt_style:
            system_prompt = generate_system_prompt(persona, article, self.cfg.system_prompt_style)
        else:
            system_prompt = ""

        # Build intro messages
        intro_messages = [
            generate_intro_message(t.persona_info.persona, t.persona_info.article)
            for t in tasks
        ]

        # Phase 1: Turn 1 generation
        turn1_prompts = [
            build_turn1_prompt(
                intro_messages[i],
                t.prompt.question,
                user_role,
                self.cfg.think_in_user_turn,
                system_prompt,
            )
            for i, t in enumerate(tasks)
        ]

        turn1_results = self._generate_turn(turn1_prompts)

        # Phase 2: Turn 2 generation (uses turn 1 decoded text)
        turn2_prompts = [
            build_turn2_prompt(
                intro_messages[i],
                t.prompt.question,
                turn1_results[i]["text"],
                t.prompt.follow_up,
                user_role,
                self.cfg.think_in_user_turn,
                system_prompt,
            )
            for i, t in enumerate(tasks)
        ]

        turn2_results = self._generate_turn(turn2_prompts)

        # Build final result dicts
        results = []
        for i, task in enumerate(tasks):
            t1 = turn1_results[i]
            t2 = turn2_results[i]

            m1 = self._compute_turn_metrics(t1["tokens"], t1["entropies"], t1["top_k_masses"])
            m2 = self._compute_turn_metrics(t2["tokens"], t2["entropies"], t2["top_k_masses"])

            results.append({
                # Metadata
                "persona": task.persona_info.persona,
                "category": task.persona_info.category,
                "article": task.persona_info.article,
                "prompt_id": task.prompt.prompt_id,
                "question": task.prompt.question,
                "follow_up": task.prompt.follow_up,
                "sample_idx": task.sample_idx,
                "intro_message": intro_messages[i],
                # Turn 1
                "response1": t1["text"],
                "response1_avg_entropy_thinking": m1["avg_entropy_thinking"],
                "response1_avg_entropy_output": m1["avg_entropy_output"],
                "response1_avg_entropy": m1["avg_entropy"],
                "response1_avg_top_k_mass_thinking": m1["avg_top_k_mass_thinking"],
                "response1_avg_top_k_mass_output": m1["avg_top_k_mass_output"],
                "response1_avg_top_k_mass": m1["avg_top_k_mass"],
                "response1_think_end_position": m1["think_end_position"],
                "response1_num_tokens": m1["num_tokens"],
                # Turn 2
                "response2": t2["text"],
                "response2_avg_entropy_thinking": m2["avg_entropy_thinking"],
                "response2_avg_entropy_output": m2["avg_entropy_output"],
                "response2_avg_entropy": m2["avg_entropy"],
                "response2_avg_top_k_mass_thinking": m2["avg_top_k_mass_thinking"],
                "response2_avg_top_k_mass_output": m2["avg_top_k_mass_output"],
                "response2_avg_top_k_mass": m2["avg_top_k_mass"],
                "response2_think_end_position": m2["think_end_position"],
                "response2_num_tokens": m2["num_tokens"],
            })

        return results

    def run_all(self, tasks: list[UserTurnTask]) -> None:
        """Run inference on all tasks, batching by persona.

        Results are written incrementally to results.jsonl after each batch.
        """
        # Group tasks by persona
        tasks_by_persona: dict[str, list[UserTurnTask]] = {}
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
                print(f"\nProcessing persona: {persona} ({len(persona_tasks)} tasks)")
                for i in range(0, len(persona_tasks), self.cfg.batch_size):
                    batch = persona_tasks[i : i + self.cfg.batch_size]
                    batch_results = self.run_batch(batch)
                    # Write results incrementally
                    append_results(self.output_dir, batch_results)
                    pbar.update(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="User-turn prediction experiment for persona inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompts-jsonl",
        type=str,
        default="datasets/consistency_prompts.jsonl",
        help="Path to consistency prompts JSONL file",
    )
    parser.add_argument(
        "--personae-json",
        type=str,
        default="datasets/consistency_personae.json",
        help="Path to personae JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model ID to use",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Tokenizer model ID (defaults to --model if empty). Use to load a different tokenizer than the model, e.g. instruct tokenizer with base model weights.",
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
        default=512,
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
        "--thinking-mode",
        action="store_true",
        default=False,
        help="Enable thinking mode (controls metric splitting by think/output)",
    )
    parser.add_argument(
        "--think-in-user-turn",
        action="store_true",
        default=False,
        help="If set, adds <think> tag after user generation prompt",
    )
    parser.add_argument(
        "--patch-user-role",
        action="store_true",
        default=False,
        help="If set, replaces user role label with persona name for all user turns",
    )
    parser.add_argument(
        "--system-prompt-style",
        type=str,
        default="",
        choices=["", "you-are-a"],
        help="System prompt style to prepend (empty string = no system prompt)",
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
        "--debug-prompts",
        action="store_true",
        default=False,
        help="Print the first 2 constructed prompts (turn 1 and turn 2) and exit",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for user-turn prediction."""
    args = parse_args()

    # Create config (validates inputs)
    cfg = UserTurnPredictionConfig(
        prompts_jsonl=args.prompts_jsonl,
        personae_json=args.personae_json,
        model=args.model,
        tokenizer=args.tokenizer,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        top_k_mass_k=args.top_k_mass_k,
        thinking_mode=args.thinking_mode,
        think_in_user_turn=args.think_in_user_turn,
        patch_user_role=args.patch_user_role,
        system_prompt_style=args.system_prompt_style,
        output_dir=args.output_dir,
        gpu_monitor_interval=args.gpu_monitor_interval,
    )

    print(f"Configuration: {cfg.to_dict()}")

    # Load inputs early (needed for debug mode and task building)
    prompts = load_consistency_prompts(cfg.prompts_jsonl)
    print(f"Loaded {len(prompts)} prompts")

    personae = load_personae(cfg.personae_json)
    print(f"Loaded {len(personae)} personae")

    # Debug prompts mode: print constructed prompts and exit
    if args.debug_prompts:
        persona = personae[0]
        prompt = prompts[0]
        user_role = persona.persona if cfg.patch_user_role else "user"
        intro = generate_intro_message(persona.persona, persona.article)
        sys_prompt = generate_system_prompt(persona.persona, persona.article, cfg.system_prompt_style) if cfg.system_prompt_style else ""

        print(f"\n=== DEBUG: Persona={persona.persona}, Article={persona.article} ===")
        print(f"User role: {user_role}")
        print(f"Intro message: {intro}")
        print(f"System prompt: {sys_prompt!r}")

        print("\n=== Turn 1 Prompt ===")
        t1 = build_turn1_prompt(intro, prompt.question, user_role, cfg.think_in_user_turn, sys_prompt)
        print(t1)

        print("\n=== Turn 2 Prompt (with placeholder response1) ===")
        t2 = build_turn2_prompt(
            intro, prompt.question, "This is a placeholder response.",
            prompt.follow_up, user_role, cfg.think_in_user_turn, sys_prompt,
        )
        print(t2)
        print("\n=== END DEBUG ===")
        return

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

    # Build task list
    tasks = build_tasks(prompts, personae, cfg.n_samples)
    total_tasks = len(tasks)
    print(f"Created {total_tasks} total tasks "
          f"({len(prompts)} prompts x {len(personae)} personae x {cfg.n_samples} samples)")

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
    runner = UserTurnPredictionRunner(cfg, output_dir)

    if cfg.gpu_monitor_interval > 0:
        with GPUMonitor(interval=cfg.gpu_monitor_interval):
            runner.run_all(tasks)
    else:
        runner.run_all(tasks)

    # Report final results
    final_count = count_results(output_dir)
    print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")
    print("Done!")


if __name__ == "__main__":
    main()
