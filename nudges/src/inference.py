#!/usr/bin/env python3
"""Nudge experiment inference script.

Generates responses for forced-choice prompts across personae and nudge conditions,
computing per-token entropy, top-k mass, and surprisal metrics.
"""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.batch.metrics import compute_entropy_and_top_k_mass, compute_section_summaries
from src.template import patch_chat_template

from .config import NudgeInferenceConfig
from .data import (
    NudgeInferenceTask,
    build_nudge_tasks,
    get_result_key,
    get_task_key,
    load_nudge_personae,
    load_nudge_prompts,
)

# Qwen3 </think> token ID
THINK_END_TOKEN_ID = 151668


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
                            f"Temp: {temp}°C",
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


class NudgeInferenceRunner:
    """Runs nudge inference using Transformers + FlashAttention2."""

    def __init__(self, cfg: NudgeInferenceConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir

        self.tokenizer_id = cfg.tokenizer or cfg.model
        print(f"Loading model {cfg.model} (tokenizer: {self.tokenizer_id})...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._original_template = self.tokenizer.chat_template

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

    def prepare_input(self, task: NudgeInferenceTask) -> str:
        """Build the full chat-template input for a task."""
        if task.nudge_type == "baseline":
            user_content = task.prompt.prompt_text
        else:
            user_content = f"{task.nudge_sentence}\n\n{task.prompt.prompt_text}"

        messages = [
            {"role": "system", "content": task.persona.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )

    def run_batch(self, tasks: list[NudgeInferenceTask]) -> list[dict]:
        """Run inference on a batch of tasks with the same persona.

        Uses incremental KV-cached generation with per-step metric computation.
        """
        if not tasks:
            return []

        persona = tasks[0].persona.persona
        assert all(t.persona.persona == persona for t in tasks), (
            "All tasks in batch must have the same persona"
        )

        self.set_persona(persona)

        input_texts = [self.prepare_input(t) for t in tasks]
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        batch_size = len(tasks)
        input_ids = inputs.input_ids
        input_len = input_ids.shape[1]

        # Pre-allocate attention mask
        max_total_len = input_len + self.cfg.max_tokens
        attention_mask = torch.zeros(
            (batch_size, max_total_len),
            device=inputs.attention_mask.device,
            dtype=inputs.attention_mask.dtype,
        )
        attention_mask[:, :input_len] = inputs.attention_mask
        current_len = input_len

        # Per-sequence tracking (scalars only, not tensors)
        generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        entropies: list[list[float]] = [[] for _ in range(batch_size)]
        top_k_masses: list[list[float]] = [[] for _ in range(batch_size)]
        surprisals: list[list[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        past_key_values = None

        with torch.inference_mode():
            for step in range(self.cfg.max_tokens):
                if all(finished):
                    break

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask[:, :current_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                past_key_values = outputs.past_key_values

                step_entropies, step_top_k = compute_entropy_and_top_k_mass(
                    logits, self.cfg.top_k_mass_k
                )

                if self.cfg.temperature > 0:
                    probs = F.softmax(logits / self.cfg.temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = logits.argmax(dim=-1)

                step_log_probs = F.log_softmax(logits, dim=-1)
                step_chosen_log_probs = step_log_probs.gather(
                    1, next_tokens.unsqueeze(1)
                ).squeeze(1)

                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_tokens[i].item()
                        generated_tokens[i].append(token_id)
                        entropies[i].append(step_entropies[i].item())
                        top_k_masses[i].append(step_top_k[i].item())
                        surprisals[i].append(-step_chosen_log_probs[i].item())

                        if token_id == self.tokenizer.eos_token_id:
                            finished[i] = True

                input_ids = next_tokens.unsqueeze(-1)
                attention_mask[:, current_len] = 1
                current_len += 1

        # Build results
        results = []
        for i, task in enumerate(tasks):
            tokens = generated_tokens[i]
            ents = entropies[i]
            top_ks = top_k_masses[i]
            surps = surprisals[i]

            # Trim EOS token if present
            if tokens and tokens[-1] == self.tokenizer.eos_token_id:
                tokens = tokens[:-1]
                ents = ents[:-1]
                top_ks = top_ks[:-1]
                surps = surps[:-1]

            response = self.tokenizer.decode(tokens, skip_special_tokens=True)
            metrics = compute_section_summaries(ents, top_ks, surps, tokens, THINK_END_TOKEN_ID)

            results.append({
                "persona": task.persona.persona,
                "core_trait": task.persona.core_trait,
                "system_prompt": task.persona.system_prompt,
                "prompt_id": task.prompt.prompt_id,
                "prompt_text": task.prompt.prompt_text,
                "prompt_category": task.prompt.category,
                "option_a": task.prompt.option_a,
                "option_b": task.prompt.option_b,
                "option_a_full": task.prompt.option_a_full,
                "option_b_full": task.prompt.option_b_full,
                "nudge_type": task.nudge_type,
                "nudge_sentence": task.nudge_sentence,
                "rep_idx": task.rep_idx,
                "response": response,
                **metrics,
            })

        return results

    def run_all(self, tasks: list[NudgeInferenceTask]) -> None:
        """Run inference on all tasks, batching by persona."""
        tasks_by_persona: dict[str, list[NudgeInferenceTask]] = {}
        for task in tasks:
            persona = task.persona.persona
            if persona not in tasks_by_persona:
                tasks_by_persona[persona] = []
            tasks_by_persona[persona].append(task)

        total_batches = sum(
            (len(pt) + self.cfg.batch_size - 1) // self.cfg.batch_size
            for pt in tasks_by_persona.values()
        )

        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for persona, persona_tasks in tasks_by_persona.items():
                for i in range(0, len(persona_tasks), self.cfg.batch_size):
                    batch = persona_tasks[i : i + self.cfg.batch_size]
                    batch_results = self.run_batch(batch)
                    append_results(self.output_dir, batch_results)
                    pbar.update(1)


def load_completed_tasks(output_dir: Path) -> set[tuple[str, int, str, int]]:
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


def configs_match(saved_config: dict, current_config: NudgeInferenceConfig) -> bool:
    """Check if saved config matches current config (ignoring output_dir)."""
    current_dict = current_config.to_comparable_dict()
    saved_comparable = {k: v for k, v in saved_config.items() if k != "output_dir"}
    return current_dict == saved_comparable


def append_results(output_dir: Path, results: list[dict]) -> None:
    """Append batch of results to results.jsonl."""
    results_path = output_dir / "results.jsonl"
    with open(results_path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def save_config(cfg: NudgeInferenceConfig) -> None:
    """Save configuration to output directory."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")


def count_results(output_dir: Path) -> int:
    """Count results in results.jsonl."""
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
    parser = argparse.ArgumentParser(
        description="Nudge experiment inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts-jsonl", type=str, required=True, help="Path to nudge prompts JSONL")
    parser.add_argument("--personae-jsonl", type=str, required=True, help="Path to nudge personae JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model ID")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer ID (defaults to --model)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-reps", type=int, default=50, help="Repetitions per condition")
    parser.add_argument("--top-k-mass-k", type=int, default=5, help="k for top-k mass")
    parser.add_argument("--thinking-mode", action="store_true", default=True, help="Enable thinking mode")
    parser.add_argument("--no-thinking-mode", action="store_false", dest="thinking_mode", help="Disable thinking mode")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (auto if empty)")
    parser.add_argument("--gpu-monitor-interval", type=float, default=30.0, help="GPU monitor interval (0 to disable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = NudgeInferenceConfig(
        prompts_jsonl=args.prompts_jsonl,
        personae_jsonl=args.personae_jsonl,
        model=args.model,
        tokenizer=args.tokenizer,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_reps=args.n_reps,
        top_k_mass_k=args.top_k_mass_k,
        thinking_mode=args.thinking_mode,
        output_dir=args.output_dir,
    )

    print(f"Configuration: {cfg.to_dict()}")

    # Check for resume
    output_dir = Path(cfg.output_dir)
    completed_keys: set[tuple[str, int, str, int]] = set()

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
            sys.exit(1)
    else:
        save_config(cfg)

    # Load inputs
    prompts = load_nudge_prompts(cfg.prompts_jsonl)
    print(f"Loaded {len(prompts)} prompts")

    personae = load_nudge_personae(cfg.personae_jsonl)
    print(f"Loaded {len(personae)} personae")

    # Build task list
    tasks = build_nudge_tasks(prompts, personae, cfg.n_reps)
    total_tasks = len(tasks)
    print(f"Created {total_tasks} total tasks ({len(personae)} personae x {len(prompts)} prompts x 7 nudge types x {cfg.n_reps} reps)")

    # Filter out completed tasks
    if completed_keys:
        tasks = [t for t in tasks if get_task_key(t) not in completed_keys]
        print(f"Remaining tasks after resume: {len(tasks)} (skipped {len(completed_keys)})")

    if not tasks:
        print("All tasks already completed!")
        final_count = count_results(output_dir)
        print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")
        return

    # Run inference
    runner = NudgeInferenceRunner(cfg, output_dir)

    if args.gpu_monitor_interval > 0:
        with GPUMonitor(interval=args.gpu_monitor_interval):
            runner.run_all(tasks)
    else:
        runner.run_all(tasks)

    final_count = count_results(output_dir)
    print(f"Total results in {output_dir / 'results.jsonl'}: {final_count}")
    print("Done!")


if __name__ == "__main__":
    main()
