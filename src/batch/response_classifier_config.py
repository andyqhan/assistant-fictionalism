"""Configuration for LLM response classifier."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResponseClassifierConfig:
    """Configuration for LLM response classification inference."""

    input_path: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.3
    max_tokens: int = 512
    batch_size: int = 256
    thinking_mode: bool = True
    gpu_monitor_interval: float = 30.0
    output_dir: str = ""

    def __post_init__(self) -> None:
        assert os.path.exists(self.input_path), f"Input file not found: {self.input_path}"
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.gpu_monitor_interval >= 0.0, f"gpu_monitor_interval must be non-negative, got {self.gpu_monitor_interval}"

        # Auto output_dir: use SLURM_JOB_ID if available, else input parent
        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id:
                self.output_dir = f"logs/response-classifier-{slurm_job_id}"
            else:
                self.output_dir = str(Path(self.input_path).parent)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "input_path": self.input_path,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "thinking_mode": self.thinking_mode,
            "gpu_monitor_interval": self.gpu_monitor_interval,
            "output_dir": self.output_dir,
        }

    def to_comparable_dict(self) -> dict:
        """Convert config to dictionary for comparison, excluding output_dir."""
        d = self.to_dict()
        del d["output_dir"]
        return d
