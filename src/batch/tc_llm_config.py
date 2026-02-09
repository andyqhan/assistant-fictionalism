"""Configuration for TC-LLM text clustering."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TCLLMConfig:
    """Configuration for TC-LLM text clustering inference."""

    input_path: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.6
    max_tokens_label_gen: int = 1024
    max_tokens_label_merge: int = 2048
    max_tokens_classify: int = 512
    batch_size: int = 64
    texts_per_label_batch: int = 15
    n_ranked_labels: int = 5
    thinking_mode: bool = False
    gpu_monitor_interval: float = 30.0
    output_dir: str = ""

    def __post_init__(self) -> None:
        # Validate input path exists
        assert os.path.exists(self.input_path), f"Input file not found: {self.input_path}"

        # Validate numeric parameters
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens_label_gen > 0, f"max_tokens_label_gen must be positive, got {self.max_tokens_label_gen}"
        assert self.max_tokens_label_merge > 0, f"max_tokens_label_merge must be positive, got {self.max_tokens_label_merge}"
        assert self.max_tokens_classify > 0, f"max_tokens_classify must be positive, got {self.max_tokens_classify}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.texts_per_label_batch > 0, f"texts_per_label_batch must be positive, got {self.texts_per_label_batch}"
        assert self.n_ranked_labels > 0, f"n_ranked_labels must be positive, got {self.n_ranked_labels}"
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
            "max_tokens_label_gen": self.max_tokens_label_gen,
            "max_tokens_label_merge": self.max_tokens_label_merge,
            "max_tokens_classify": self.max_tokens_classify,
            "batch_size": self.batch_size,
            "texts_per_label_batch": self.texts_per_label_batch,
            "n_ranked_labels": self.n_ranked_labels,
            "thinking_mode": self.thinking_mode,
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
