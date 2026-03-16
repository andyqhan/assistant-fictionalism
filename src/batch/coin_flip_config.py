import os
from dataclasses import dataclass


@dataclass
class CoinFlipConfig:
    """Configuration for coin flip experiment."""

    personae_json: str
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer: str = ""  # Tokenizer model ID (defaults to model if empty)
    batch_size: int = 512
    system_prompt_style: str = "you-are-a"
    use_tasks_from: str = ""  # Empty = per-persona tasks
    output_dir: str = ""
    gpu_monitor_interval: float = 30.0

    def __post_init__(self) -> None:
        assert os.path.exists(self.personae_json), f"Personae file not found: {self.personae_json}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"

        valid_styles = ["you-are-a"]
        assert self.system_prompt_style in valid_styles, f"Invalid system_prompt_style: {self.system_prompt_style}"

        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/coin-flip-{slurm_job_id}"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "personae_json": self.personae_json,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "batch_size": self.batch_size,
            "system_prompt_style": self.system_prompt_style,
            "use_tasks_from": self.use_tasks_from,
            "output_dir": self.output_dir,
            "gpu_monitor_interval": self.gpu_monitor_interval,
        }

    def to_comparable_dict(self) -> dict:
        """Convert config to dictionary for comparison, excluding output_dir."""
        d = self.to_dict()
        del d["output_dir"]
        return d
