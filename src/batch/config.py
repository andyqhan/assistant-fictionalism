import os
from dataclasses import dataclass, field


@dataclass
class BatchInferenceConfig:
    """Configuration for batch persona inference."""

    prompts_json: str
    personae_json: str
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.0
    max_tokens: int = 1024
    batch_size: int = 64
    n_reps: int = 1
    top_k_mass_k: int = 5
    thinking_mode: bool = True
    system_prompt_style: str = "you-are-a"
    output_dir: str = ""
    subset_category_persona: float = 1.0
    subset_prompt: float = 1.0

    def __post_init__(self) -> None:
        # Validate paths exist
        assert os.path.exists(self.prompts_json), f"Prompts file not found: {self.prompts_json}"
        assert os.path.exists(self.personae_json), f"Personae file not found: {self.personae_json}"

        # Validate numeric parameters
        assert self.temperature >= 0.0, f"Temperature must be non-negative, got {self.temperature}"
        assert self.max_tokens > 0, f"max_tokens must be positive, got {self.max_tokens}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.n_reps > 0, f"n_reps must be positive, got {self.n_reps}"
        assert self.top_k_mass_k > 0, f"top_k_mass_k must be positive, got {self.top_k_mass_k}"
        assert 0.0 < self.subset_category_persona <= 1.0, f"subset_category_persona must be in (0, 1], got {self.subset_category_persona}"
        assert 0.0 < self.subset_prompt <= 1.0, f"subset_prompt must be in (0, 1], got {self.subset_prompt}"

        # Validate system prompt style
        valid_styles = ["you-are-a"]
        assert self.system_prompt_style in valid_styles, f"Invalid system_prompt_style: {self.system_prompt_style}"

        # Handle n_reps with temperature=0
        if self.temperature == 0.0 and self.n_reps > 1:
            print(f"Warning: n_reps={self.n_reps} with temperature=0.0 is redundant. Forcing n_reps=1.")
            self.n_reps = 1

        # Warn about thinking mode with temperature=0
        if self.thinking_mode and self.temperature == 0.0:
            print("Warning: thinking_mode=True with temperature=0.0 may produce deterministic thinking.")

        # Auto-generate output directory if not specified
        if not self.output_dir:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            self.output_dir = f"logs/personae-inference-{slurm_job_id}"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "prompts_json": self.prompts_json,
            "personae_json": self.personae_json,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "n_reps": self.n_reps,
            "top_k_mass_k": self.top_k_mass_k,
            "thinking_mode": self.thinking_mode,
            "system_prompt_style": self.system_prompt_style,
            "output_dir": self.output_dir,
            "subset_category_persona": self.subset_category_persona,
            "subset_prompt": self.subset_prompt,
        }
