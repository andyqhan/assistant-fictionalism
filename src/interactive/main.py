from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

from src.model import ChatModel

from .commands import process_command
from .colors import assistant_color, system_color, error_color
from .keyboard import cbreak_mode, check_escape


def main() -> None:
    """Main entry point for the chatbot."""
    print(system_color("Loading chatbot..."))

    try:
        model = ChatModel()
    except Exception as e:
        print(error_color(f"Failed to load model: {e}"))
        return

    print(system_color(f"Device: {model.device}"))
    print(system_color("Type /help for commands, Ctrl+C to exit."))
    print()

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = prompt(HTML("<ansigreen>You: </ansigreen>"))
        except KeyboardInterrupt:
            print()
            print(system_color("Goodbye!"))
            break
        except EOFError:
            print()
            print(system_color("Goodbye!"))
            break

        if not user_input.strip():
            continue

        result = process_command(user_input, model)

        if result.handled:
            if result.clear_history:
                messages = []
            continue

        messages.append({"role": "user", "content": user_input})

        # Display persona label before streaming
        label = model.persona if model.persona else ""
        if label:
            print(assistant_color(f"{label}: "), end="", flush=True)

        try:
            response_chunks: list[str] = []
            interrupted = False
            with cbreak_mode():
                for chunk in model.generate(messages):
                    if check_escape():
                        interrupted = True
                        break
                    print(assistant_color(chunk), end="", flush=True)
                    response_chunks.append(chunk)
            print()  # Newline after streaming completes
            if interrupted:
                print(system_color("(interrupted)"))
        except Exception as e:
            print()  # Newline after any partial output
            print(error_color(f"Generation error: {e}"))
            messages.pop()
            continue

        response = "".join(response_chunks)
        messages.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()
