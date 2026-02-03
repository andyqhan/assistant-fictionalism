#!/usr/bin/env python3
"""
Prefill inference script for persona disavowal experiments.

Tests whether personas will disavow "out of character" prefilled responses.
The model is given a prompt, a prefilled response (labeled as the persona),
and then asked a follow-up question.
"""

import argparse
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prefill_config import PrefillInferenceConfig


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
class PrefillPersona:
    """Information about a persona with its system prompt."""

    persona: str
    system_prompt: str


@dataclass
class PrefillPrompt:
    """A prefill prompt with the initial prompt and prefilled response."""

    prompt: str
    prefill: str
    category: str
    persona: str | None  # None = expand to all personas
    prompt_id: int


@dataclass
class PrefillTask:
    """A single prefill inference task to run."""

    persona: PrefillPersona
    prompt: PrefillPrompt
    followup: str
    rep_idx: int


def load_prefill_personae(path: str) -> list[PrefillPersona]:
    """Load personae from JSONL file.

    Each line must have 'persona' and 'system_prompt' fields.
    """
    personae = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "persona" in obj, f"Line {line_num + 1}: Missing 'persona' key"
            assert "system_prompt" in obj, f"Line {line_num + 1}: Missing 'system_prompt' key"
            personae.append(PrefillPersona(
                persona=obj["persona"],
                system_prompt=obj["system_prompt"],
            ))

    assert len(personae) > 0, "Personae file must not be empty"

    return personae


def load_prefill_prompts(path: str) -> list[PrefillPrompt]:
    """Load prefill prompts from JSONL file.

    Each line must have 'prompt', 'prefill', and 'category' fields.
    Optional 'persona' field restricts the prompt to a specific persona.
    """
    prompts = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "prompt" in obj, f"Line {line_num + 1}: Missing 'prompt' key"
            assert "prefill" in obj, f"Line {line_num + 1}: Missing 'prefill' key"
            assert "category" in obj, f"Line {line_num + 1}: Missing 'category' key"

            prompts.append(PrefillPrompt(
                prompt=obj["prompt"],
                prefill=obj["prefill"],
                category=obj["category"],
                persona=obj.get("persona"),  # None if not specified
                prompt_id=line_num,
            ))

    assert len(prompts) > 0, "Prompts file must not be empty"

    return prompts


def build_prefill_tasks(
    prompts: list[PrefillPrompt],
    personae: list[PrefillPersona],
    followup: str,
    n_reps: int,
) -> list[PrefillTask]:
    """Build list of all prefill inference tasks to run.

    If a prompt has a 'persona' key, create tasks only for that persona.
    If no 'persona' key, expand to ALL personas.
    """
    tasks = []

    # Build persona lookup
    persona_lookup = {p.persona: p for p in personae}

    for prompt in prompts:
        if prompt.persona is not None:
            # Prompt is specific to one persona
            assert prompt.persona in persona_lookup, (
                f"Prompt {prompt.prompt_id} references unknown persona '{prompt.persona}'"
            )
            target_personae = [persona_lookup[prompt.persona]]
        else:
            # Expand to all personas
            target_personae = personae

        for persona in target_personae:
            for rep_idx in range(n_reps):
                tasks.append(PrefillTask(
                    persona=persona,
                    prompt=prompt,
                    followup=followup,
                    rep_idx=rep_idx,
                ))

    return tasks


def get_task_key(task: PrefillTask) -> tuple[str, int, int]:
    """Get the unique key for a task: (persona, prompt_id, rep_idx)."""
    return (task.persona.persona, task.prompt.prompt_id, task.rep_idx)


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


def configs_match(saved_config: dict, current_config: PrefillInferenceConfig) -> bool:
    """Check if saved config matches current config (ignoring output_dir).

    Args:
        saved_config: Dictionary loaded from saved config.json
        current_config: Current PrefillInferenceConfig instance

    Returns:
        True if configs match (excluding output_dir), False otherwise.
    """
    current_dict = current_config.to_comparable_dict()

    # Compare all keys except output_dir (which was already excluded)
    saved_comparable = {k: v for k, v in saved_config.items() if k != "output_dir"}

    return current_dict == saved_comparable


def filter_completed_tasks(
    tasks: list[PrefillTask], completed: set[tuple[str, int, int]]
) -> list[PrefillTask]:
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


def save_config(cfg: PrefillInferenceConfig) -> None:
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


class VLLMPrefillRunner:
    """Runs prefill inference using vLLM."""

    def __init__(self, cfg: PrefillInferenceConfig, output_dir: Path):
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        self.output_dir = output_dir

        print(f"Loading model {cfg.model} with vLLM...")

        # Load tokenizer for decoding
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        assert self.tokenizer is not None, "Failed to load tokenizer"

        # Initialize vLLM
        self.llm = LLM(
            model=cfg.model,
            dtype="auto",
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

        # Sampling params
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        print("Model loaded successfully.")

    def prepare_prefill_input_manual(
        self,
        system_prompt: str,
        user_prompt: str,
        prefill: str,
        followup: str,
        persona: str,
    ) -> str:
        """Manually construct prompt with persona labels on all assistant turns.

        This bypasses apply_chat_template() to ensure persona labeling on both
        the prefill message and the generation prompt.

        Includes <think></think> tags to match Qwen3 thinking format.
        """
        parts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>",
            f"<|im_start|>user\n{user_prompt}<|im_end|>",
            f"<|im_start|>{persona}\n<think>\n\n</think>\n\n{prefill}<|im_end|>",
            f"<|im_start|>user\n{followup}<|im_end|>",
            f"<|im_start|>{persona}\n",  # generation prompt
        ]
        return "\n".join(parts)

    def run_batch(self, tasks: list[PrefillTask]) -> list[dict]:
        """Run inference on a batch of tasks with the same persona.

        All tasks must have the same persona (for consistent template).
        """
        if len(tasks) == 0:
            return []

        # Verify all tasks have same persona
        persona = tasks[0].persona.persona
        assert all(t.persona.persona == persona for t in tasks), (
            "All tasks in batch must have the same persona"
        )

        # Prepare input texts
        prompts = [
            self.prepare_prefill_input_manual(
                t.persona.system_prompt,
                t.prompt.prompt,
                t.prompt.prefill,
                t.followup,
                persona,
            )
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

            results.append({
                "persona": task.persona.persona,
                "category": task.prompt.category,
                "prompt": task.prompt.prompt,
                "prefill": task.prompt.prefill,
                "followup": task.followup,
                "system_prompt": task.persona.system_prompt,
                "prompt_id": task.prompt.prompt_id,
                "rep_idx": task.rep_idx,
                "response": response,
                "num_tokens": len(token_ids),
                "model": self.cfg.model,
                "temperature": self.cfg.temperature,
            })

        return results

    def run_all(self, tasks: list[PrefillTask]) -> None:
        """Run inference on all tasks, batching by persona.

        Results are written incrementally to results.jsonl after each batch.
        """
        # Group tasks by persona
        tasks_by_persona: dict[str, list[PrefillTask]] = {}
        for task in tasks:
            persona = task.persona.persona
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


class TransformersPrefillRunner:
    """Runs prefill inference using Transformers + FlashAttention2."""

    def __init__(self, cfg: PrefillInferenceConfig, output_dir: Path):
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

    def prepare_prefill_input_manual(
        self,
        system_prompt: str,
        user_prompt: str,
        prefill: str,
        followup: str,
        persona: str,
    ) -> str:
        """Manually construct prompt with persona labels on all assistant turns.

        This bypasses apply_chat_template() to ensure persona labeling on both
        the prefill message and the generation prompt.

        Includes <think></think> tags to match Qwen3 thinking format.
        """
        parts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>",
            f"<|im_start|>user\n{user_prompt}<|im_end|>",
            f"<|im_start|>{persona}\n<think>\n\n</think>\n\n{prefill}<|im_end|>",
            f"<|im_start|>user\n{followup}<|im_end|>",
            f"<|im_start|>{persona}\n",  # generation prompt
        ]
        return "\n".join(parts)

    def run_batch(self, tasks: list[PrefillTask]) -> list[dict]:
        """Run inference on a batch of tasks with the same persona.

        All tasks must have the same persona (for consistent template).
        Uses incremental generation with KV caching.
        """
        if len(tasks) == 0:
            return []

        # Verify all tasks have same persona
        persona = tasks[0].persona.persona
        assert all(t.persona.persona == persona for t in tasks), (
            "All tasks in batch must have the same persona"
        )

        # Prepare input texts
        input_texts = [
            self.prepare_prefill_input_manual(
                t.persona.system_prompt,
                t.prompt.prompt,
                t.prompt.prefill,
                t.followup,
                persona,
            )
            for t in tasks
        ]

        # Tokenize with left-padding
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

        # Per-sequence tracking
        generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]
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
                past_key_values = outputs.past_key_values

                # Sample next token
                if self.cfg.temperature > 0:
                    probs = F.softmax(logits / self.cfg.temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = logits.argmax(dim=-1)

                # Store results per sequence
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_tokens[i].item()
                        generated_tokens[i].append(token_id)

                        if token_id == self.tokenizer.eos_token_id:
                            finished[i] = True

                # Prepare for next step
                input_ids = next_tokens.unsqueeze(-1)
                attention_mask[:, current_len] = 1
                current_len += 1

        # Build results
        results = []
        for i, task in enumerate(tasks):
            tokens = generated_tokens[i]

            # Trim EOS token if present
            if tokens and tokens[-1] == self.tokenizer.eos_token_id:
                tokens = tokens[:-1]

            response = self.tokenizer.decode(tokens, skip_special_tokens=True)

            results.append({
                "persona": task.persona.persona,
                "category": task.prompt.category,
                "prompt": task.prompt.prompt,
                "prefill": task.prompt.prefill,
                "followup": task.followup,
                "system_prompt": task.persona.system_prompt,
                "prompt_id": task.prompt.prompt_id,
                "rep_idx": task.rep_idx,
                "response": response,
                "num_tokens": len(tokens),
                "model": self.cfg.model,
                "temperature": self.cfg.temperature,
            })

        return results

    def run_all(self, tasks: list[PrefillTask]) -> None:
        """Run inference on all tasks, batching by persona.

        Results are written incrementally to results.jsonl after each batch.
        """
        # Group tasks by persona
        tasks_by_persona: dict[str, list[PrefillTask]] = {}
        for task in tasks:
            persona = task.persona.persona
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


def create_inference_runner(cfg: PrefillInferenceConfig, output_dir: Path):
    """Factory function to create the appropriate inference runner."""
    if cfg.backend == "transformers":
        return TransformersPrefillRunner(cfg, output_dir)
    else:
        return VLLMPrefillRunner(cfg, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prefill inference for persona disavowal experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompts-jsonl",
        type=str,
        required=True,
        help="Path to prefill prompts JSONL file",
    )
    parser.add_argument(
        "--personae-jsonl",
        type=str,
        required=True,
        help="Path to prefill personae JSONL file",
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
        default=100,
        help="Number of repetitions per prompt",
    )
    parser.add_argument(
        "--followup-prompt",
        type=str,
        default="Can you say more about that?",
        help="Follow-up prompt after prefill",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (auto-generated if empty)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend: 'transformers' or 'vllm'",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Seconds between GPU utilization logs (0 to disable)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for prefill inference."""
    args = parse_args()

    # Create config (validates inputs)
    cfg = PrefillInferenceConfig(
        prompts_jsonl=args.prompts_jsonl,
        personae_jsonl=args.personae_jsonl,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_reps=args.n_reps,
        followup_prompt=args.followup_prompt,
        output_dir=args.output_dir,
        backend=args.backend,
        gpu_monitor_interval=args.gpu_monitor_interval,
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
    personae = load_prefill_personae(cfg.personae_jsonl)
    print(f"Loaded {len(personae)} personae")

    prompts = load_prefill_prompts(cfg.prompts_jsonl)
    print(f"Loaded {len(prompts)} prompts")

    # Build task list
    tasks = build_prefill_tasks(prompts, personae, cfg.followup_prompt, cfg.n_reps)
    total_tasks = len(tasks)
    print(f"Created {total_tasks} total tasks")

    # Count task breakdown
    persona_specific = sum(1 for p in prompts if p.persona is not None)
    all_persona = len(prompts) - persona_specific
    print(f"  - {persona_specific} prompts with specific persona")
    print(f"  - {all_persona} prompts expanded to all {len(personae)} personae")

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
