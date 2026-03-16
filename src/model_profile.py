from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelProfile:
    """Model family profile for chat template handling."""

    family: str  # "qwen3" or "gemma3"
    supports_thinking: bool
    think_end_token_id: int | None
    assistant_role_name: str  # "assistant" for Qwen3, "model" for Gemma3


def detect_model_profile(tokenizer: PreTrainedTokenizerBase) -> ModelProfile:
    """Detect model family from the tokenizer's chat template.

    Inspects the chat template string for signature delimiters:
    - '<|im_start|>' -> Qwen3
    - '<start_of_turn>' -> Gemma 3

    Args:
        tokenizer: A loaded tokenizer with a chat_template attribute.

    Returns:
        A ModelProfile describing the model family's capabilities.
    """
    template = tokenizer.chat_template
    assert template is not None, "Tokenizer has no chat_template"

    if "<|im_start|>" in template:
        supports_thinking = "enable_thinking" in template
        think_end_token_id = tokenizer.convert_tokens_to_ids("</think>") if supports_thinking else None
        return ModelProfile(
            family="qwen3",
            supports_thinking=supports_thinking,
            think_end_token_id=think_end_token_id,
            assistant_role_name="assistant",
        )

    if "<start_of_turn>" in template:
        return ModelProfile(
            family="gemma3",
            supports_thinking=False,
            think_end_token_id=None,
            assistant_role_name="model",
        )

    raise ValueError(
        "Unable to detect model family from chat template. "
        "Expected '<|im_start|>' (Qwen3) or '<start_of_turn>' (Gemma 3) in template."
    )
