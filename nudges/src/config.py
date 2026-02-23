import os
from dataclasses import dataclass


@dataclass
class NudgeInferenceConfig:
    """Configuration for nudge experiment inference."""

    prompts_jsonl: str
    personae_jsonl: str
    model: str = "Qwen/Qwen3-8B"
    tokenizer: str = ""  # Defaults to model if empty
    temperature: float = 0.7
    max_tokens: int = 1024
    batch_size: int = 64
    n_reps: int = 50
    top_k_mass_k: int = 5
    thinking_mode: bool = True
    output_dir: str = ""

    def __post_init__(self) -> None:
        assert os.path.exists(self.prompts_jsonl), f"Prompts file not found: {self.prompts_jsonl}"
        assert os.path.exists(self.personae_jsonl), f"Personae file not found: {self.personae_jsonl}"
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.n_reps > 0, f"n_reps must be positive, got {self.n_reps}"
        assert self.top_k_mass_k > 0, f"top_k_mass_k must be positive, got {self.top_k_mass_k}"

        if self.temperature == 0.0 and self.n_reps > 1:
            print(f"Warning: n_reps={self.n_reps} with temperature=0.0 is redundant. Forcing n_reps=1.")
            self.n_reps = 1

        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/nudge-inference-{slurm_job_id}"

    def to_dict(self) -> dict:
        return {
            "prompts_jsonl": self.prompts_jsonl,
            "personae_jsonl": self.personae_jsonl,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "n_reps": self.n_reps,
            "top_k_mass_k": self.top_k_mass_k,
            "thinking_mode": self.thinking_mode,
            "output_dir": self.output_dir,
        }

    def to_comparable_dict(self) -> dict:
        d = self.to_dict()
        del d["output_dir"]
        return d


@dataclass
class NudgeJudgeConfig:
    """Configuration for nudge judge inference."""

    input_path: str
    model: str = "Qwen/Qwen3-32B"
    temperature: float = 0.0
    max_tokens: int = 512
    batch_size: int = 64
    exclude_thinking: bool = True
    output_dir: str = ""

    def __post_init__(self) -> None:
        assert os.path.exists(self.input_path), f"Input file not found: {self.input_path}"
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"

        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/nudge-judge-{slurm_job_id}"

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "exclude_thinking": self.exclude_thinking,
            "output_dir": self.output_dir,
        }

    def to_comparable_dict(self) -> dict:
        d = self.to_dict()
        del d["output_dir"]
        return d
