from dataclasses import dataclass
from typing import TYPE_CHECKING

from .colors import system_color, error_color

if TYPE_CHECKING:
    from .model import ChatModel


@dataclass
class CommandResult:
    """Result of command processing."""

    handled: bool
    clear_history: bool = False
    new_persona: str | None = None


def process_command(user_input: str, model: "ChatModel") -> CommandResult:
    """
    Process user input for commands.

    Args:
        user_input: The raw user input string
        model: The chat model instance

    Returns:
        CommandResult indicating what action to take
    """
    assert isinstance(user_input, str), f"User input must be a string, got {type(user_input)}"

    stripped = user_input.strip()

    if not stripped.startswith("/"):
        return CommandResult(handled=False)

    parts = stripped.split(maxsplit=1)
    command = parts[0].lower()

    if command == "/clear":
        print(system_color("Context cleared."))
        return CommandResult(handled=True, clear_history=True)

    if command == "/persona":
        if len(parts) < 2:
            print(error_color("Usage: /persona <name> or /persona \"\" for empty"))
            return CommandResult(handled=True)

        new_persona = parts[1].strip()

        # Handle quoted empty string
        if new_persona in ('""', "''"):
            new_persona = ""

        # Remove surrounding quotes if present
        if (new_persona.startswith('"') and new_persona.endswith('"')) or \
           (new_persona.startswith("'") and new_persona.endswith("'")):
            new_persona = new_persona[1:-1]

        model.set_persona(new_persona)

        if new_persona:
            print(system_color(f"Persona changed to: {new_persona}"))
        else:
            print(system_color("Persona cleared (empty)."))

        print(system_color("Context cleared."))
        return CommandResult(handled=True, clear_history=True, new_persona=new_persona)

    if command == "/system":
        if len(parts) < 2:
            print(error_color("Usage: /system <prompt> or /system \"\" for empty"))
            return CommandResult(handled=True)

        new_prompt = parts[1].strip()

        # Handle quoted empty string
        if new_prompt in ('""', "''"):
            new_prompt = ""

        # Remove surrounding quotes if present
        if (new_prompt.startswith('"') and new_prompt.endswith('"')) or \
           (new_prompt.startswith("'") and new_prompt.endswith("'")):
            new_prompt = new_prompt[1:-1]

        model.set_system_prompt(new_prompt)

        if new_prompt:
            print(system_color(f"System prompt changed to: {new_prompt}"))
        else:
            print(system_color("System prompt cleared (empty)."))

        print(system_color("Context cleared."))
        return CommandResult(handled=True, clear_history=True)

    if command == "/thinking":
        model.set_thinking(True)
        print(system_color("Thinking mode enabled."))
        return CommandResult(handled=True)

    if command == "/nothinking":
        model.set_thinking(False)
        print(system_color("Thinking mode disabled."))
        return CommandResult(handled=True)

    if command == "/tokens":
        if len(parts) < 2:
            print(system_color(f"Current max tokens: {model.max_new_tokens}"))
            return CommandResult(handled=True)

        try:
            tokens = int(parts[1].strip())
            model.set_max_tokens(tokens)
            print(system_color(f"Max tokens set to: {tokens}"))
        except ValueError:
            print(error_color("Usage: /tokens <number>"))
        except AssertionError as e:
            print(error_color(str(e)))
        return CommandResult(handled=True)

    if command == "/help":
        print(system_color("Available commands:"))
        print(system_color("  /clear           - Clear conversation history"))
        print(system_color("  /persona <name>  - Change assistant persona"))
        print(system_color("  /system <prompt> - Change system prompt"))
        print(system_color("  /thinking        - Enable thinking mode"))
        print(system_color("  /nothinking      - Disable thinking mode"))
        print(system_color("  /tokens <n>      - Set max tokens (current: show)"))
        print(system_color("  /help            - Show this help message"))
        return CommandResult(handled=True)

    print(error_color(f"Unknown command: {command}"))
    print(system_color("Type /help for available commands."))
    return CommandResult(handled=True)
