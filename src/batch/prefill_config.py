"""Configuration for prefill inference experiments."""

import os
from dataclasses import dataclass


@dataclass
class PrefillInferenceConfig:
    """Configuration for prefill persona inference."""

    prompts_jsonl: str
    personae_jsonl: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.7
    max_tokens: int = 1024
    batch_size: int = 256
    n_reps: int = 100
    followup_prompt: str = "Can you say more about that?"
    output_dir: str = ""
    backend: str = "transformers"  # "transformers" or "vllm"
    gpu_monitor_interval: float = 30.0

    def __post_init__(self) -> None:
        # Validate paths exist
        assert os.path.exists(self.prompts_jsonl), f"Prompts file not found: {self.prompts_jsonl}"
        assert os.path.exists(self.personae_jsonl), f"Personae file not found: {self.personae_jsonl}"

        # Validate backend
        assert self.backend in ["transformers", "vllm"], f"backend must be 'transformers' or 'vllm', got {self.backend}"

        # Validate numeric parameters
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.n_reps > 0, f"n_reps must be positive, got {self.n_reps}"
        assert self.gpu_monitor_interval >= 0.0, f"gpu_monitor_interval must be non-negative, got {self.gpu_monitor_interval}"

        # Handle n_reps with temperature=0
        if self.temperature == 0.0 and self.n_reps > 1:
            print(f"Warning: n_reps={self.n_reps} with temperature=0.0 is redundant. Forcing n_reps=1.")
            self.n_reps = 1

        # Auto-generate output directory if not specified
        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/prefill-inference-{slurm_job_id}"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "prompts_jsonl": self.prompts_jsonl,
            "personae_jsonl": self.personae_jsonl,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "n_reps": self.n_reps,
            "followup_prompt": self.followup_prompt,
            "output_dir": self.output_dir,
            "backend": self.backend,
            "gpu_monitor_interval": self.gpu_monitor_interval,
        }

    def to_comparable_dict(self) -> dict:
        """Convert config to dictionary for comparison, excluding output_dir.

        Used for resume detection - configs match if all parameters except
        output_dir are identical.
        """
        d = self.to_dict()
        del d["output_dir"]
        return d
