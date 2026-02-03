"""Configuration for LLM judge inference."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JudgeConfig:
    """Configuration for LLM judge inference."""

    input_path: str
    model: str = "Qwen/Qwen3-32B"
    temperature: float = 0.4
    max_tokens: int = 2048
    batch_size: int = 64
    exclude_thinking: bool = False
    gpu_monitor_interval: float = 30.0
    output_dir: str = ""

    def __post_init__(self) -> None:
        # Validate input path exists
        assert os.path.exists(self.input_path), f"Input file not found: {self.input_path}"

        # Validate numeric parameters
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.gpu_monitor_interval >= 0.0, f"gpu_monitor_interval must be non-negative, got {self.gpu_monitor_interval}"

        # Default output_dir to input file's parent directory
        if not self.output_dir:
            self.output_dir = str(Path(self.input_path).parent)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "input_path": self.input_path,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "exclude_thinking": self.exclude_thinking,
            "gpu_monitor_interval": self.gpu_monitor_interval,
            "output_dir": self.output_dir,
        }

    def to_comparable_dict(self) -> dict:
        """Convert config to dictionary for comparison, excluding output_dir.

        Used for resume detection - configs match if all parameters except
        output_dir are identical.
        """
        d = self.to_dict()
        del d["output_dir"]
        return d
