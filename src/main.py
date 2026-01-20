from .model import ChatModel
from .commands import process_command
from .colors import user_color, assistant_color, system_color, error_color


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
            user_input = input(user_color("You: "))
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

        try:
            response = model.generate(messages)
        except Exception as e:
            print(error_color(f"Generation error: {e}"))
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})

        # Display with persona label
        label = model.persona if model.persona else ""
        if label:
            print(assistant_color(f"{label}: {response}"))
        else:
            print(assistant_color(response))
        print()


if __name__ == "__main__":
    main()
