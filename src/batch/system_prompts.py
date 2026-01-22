def generate_system_prompt_you_are_a(persona: str, article: str) -> str:
    """
    Generate a system prompt using the 'you-are-a' style.

    Args:
        persona: The persona name (e.g., "assistant", "Hamlet")
        article: The article to use (e.g., "an", "a", or "" for none)

    Returns:
        System prompt string like "You are an assistant." or "You are Hamlet."
    """
    assert isinstance(persona, str), f"Persona must be a string, got {type(persona)}"
    assert isinstance(article, str), f"Article must be a string, got {type(article)}"

    if article:
        return f"You are {article} {persona}."
    else:
        return f"You are {persona}."


def generate_system_prompt(persona: str, article: str, style: str) -> str:
    """
    Generate a system prompt using the specified style.

    Args:
        persona: The persona name
        article: The article to use (or "" for none)
        style: The style to use (e.g., "you-are-a")

    Returns:
        Generated system prompt string
    """
    assert isinstance(style, str), f"Style must be a string, got {type(style)}"

    if style == "you-are-a":
        return generate_system_prompt_you_are_a(persona, article)
    else:
        raise ValueError(f"Unknown system prompt style: {style}")
