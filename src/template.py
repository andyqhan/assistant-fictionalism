from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from .model_profile import ModelProfile


def _escape_for_jinja_string(s: str, quote_char: str) -> str:
    """
    Escape a string for safe insertion into a Jinja2 string literal.

    Args:
        s: The string to escape
        quote_char: The quote character used for the string literal (' or ")

    Returns:
        The escaped string safe for insertion into a Jinja2 template
    """
    # Escape backslashes first, then the quote character
    escaped = s.replace("\\", "\\\\")
    escaped = escaped.replace(quote_char, f"\\{quote_char}")
    return escaped


def _patch_qwen3(original_template: str, persona: str) -> str:
    """Patch a Qwen3 chat template to use a custom persona."""
    patched = original_template

    escaped_persona_single = _escape_for_jinja_string(persona, "'")
    patched = patched.replace(
        "'<|im_start|>assistant\\n'",
        f"'<|im_start|>{escaped_persona_single}\\n'" if persona else "'<|im_start|>'"
    )

    escaped_persona_double = _escape_for_jinja_string(persona, '"')
    patched = patched.replace(
        "\"<|im_start|>assistant\\n\"",
        f"\"<|im_start|>{escaped_persona_double}\\n\"" if persona else "\"<|im_start|>\""
    )

    return patched


def _patch_gemma3(original_template: str, persona: str) -> str:
    """Patch a Gemma 3 chat template to use a custom persona.

    Two replacements:
    1. Role mapping for past assistant turns:
       '{%- set role = "model" -%}' -> '{%- set role = "<persona>" -%}'
    2. Generation prompt for new generation:
       "{{'<start_of_turn>model\n'}}" -> "{{'<start_of_turn><persona>\n'}}"
       (the template contains actual newlines, not escaped \\n)
    """
    patched = original_template

    escaped_persona_double = _escape_for_jinja_string(persona, '"')
    patched = patched.replace(
        '{%- set role = "model" -%}',
        f'{{% set role = "{escaped_persona_double}" %}}' if persona else '{%- set role = "model" -%}',
    )

    escaped_persona_single = _escape_for_jinja_string(persona, "'")
    # The generation prompt in the Gemma 3 template: {{'<start_of_turn>model\n'}}
    # where \n is a real newline character
    old_gen = "{{'<start_of_turn>model\n'}}"
    new_gen = "{{'" + f"<start_of_turn>{escaped_persona_single}\n" + "'}}" if persona else old_gen
    patched = patched.replace(old_gen, new_gen)

    return patched


def patch_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    original_template: str,
    persona: str,
    profile: ModelProfile,
) -> None:
    """
    Patch the chat template to use a custom persona instead of the default assistant role.

    Dispatches to model-family-specific patching logic based on the profile.

    Args:
        tokenizer: The tokenizer to patch
        original_template: The original chat template to patch from
        persona: The new persona string (can be empty)
        profile: The model profile (from detect_model_profile)
    """
    assert hasattr(tokenizer, "chat_template"), "Tokenizer must have chat_template attribute"
    assert isinstance(persona, str), f"Persona must be a string, got {type(persona)}"

    if profile.family == "qwen3":
        tokenizer.chat_template = _patch_qwen3(original_template, persona)
    elif profile.family == "gemma3":
        tokenizer.chat_template = _patch_gemma3(original_template, persona)
    else:
        raise ValueError(f"Unknown model family: {profile.family}")


def get_generation_prefix(persona: str, profile: ModelProfile) -> str:
    """
    Get the generation prefix for the given persona.

    This is used when decoding to know what prefix was added before generation.
    """
    assert isinstance(persona, str), f"Persona must be a string, got {type(persona)}"

    if profile.family == "qwen3":
        if persona:
            return f"<|im_start|>{persona}\n"
        return "<|im_start|>"
    elif profile.family == "gemma3":
        if persona:
            return f"<start_of_turn>{persona}\n"
        return "<start_of_turn>model\n"
    else:
        raise ValueError(f"Unknown model family: {profile.family}")
