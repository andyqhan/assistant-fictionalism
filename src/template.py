from transformers import PreTrainedTokenizerBase


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


def patch_chat_template(tokenizer: PreTrainedTokenizerBase, original_template: str, persona: str) -> None:
    """
    Patch the chat template to use a custom persona instead of 'assistant'.

    The Qwen3 template uses '<|im_start|>assistant\\n' for the assistant role.
    This function replaces 'assistant' with the provided persona string.

    Args:
        tokenizer: The tokenizer to patch
        original_template: The original chat template to patch from
        persona: The new persona string (can be empty)
    """
    assert hasattr(tokenizer, "chat_template"), "Tokenizer must have chat_template attribute"
    assert isinstance(persona, str), f"Persona must be a string, got {type(persona)}"

    patched = original_template

    # Patch message rendering: where role == 'assistant' gets rendered
    # In Qwen3 template: '<|im_start|>' + message['role'] is used
    # We need to replace the assistant role check and rendering

    # The Qwen3 template has logic like:
    # {% if message['role'] == 'assistant' %}
    # We patch the generation prompt suffix specifically

    # Patch the generation prompt that adds '<|im_start|>assistant\n' at the end
    # Escape the persona for safe insertion into Jinja2 single-quoted string
    escaped_persona_single = _escape_for_jinja_string(persona, "'")
    patched = patched.replace(
        "'<|im_start|>assistant\\n'",
        f"'<|im_start|>{escaped_persona_single}\\n'" if persona else "'<|im_start|>'"
    )

    # Also handle the case where it might not have the escaped newline
    # Escape the persona for safe insertion into Jinja2 double-quoted string
    escaped_persona_double = _escape_for_jinja_string(persona, '"')
    patched = patched.replace(
        "\"<|im_start|>assistant\\n\"",
        f"\"<|im_start|>{escaped_persona_double}\\n\"" if persona else "\"<|im_start|>\""
    )

    tokenizer.chat_template = patched


def get_generation_prefix(persona: str) -> str:
    """
    Get the generation prefix for the given persona.

    This is used when decoding to know what prefix was added before generation.
    """
    assert isinstance(persona, str), f"Persona must be a string, got {type(persona)}"

    if persona:
        return f"<|im_start|>{persona}\n"
    return "<|im_start|>"
