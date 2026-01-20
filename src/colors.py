from colorama import Fore, Style, init

init()


def user_color(text: str) -> str:
    """Format text in green for user messages."""
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def assistant_color(text: str) -> str:
    """Format text in cyan for assistant/persona messages."""
    return f"{Fore.CYAN}{text}{Style.RESET_ALL}"


def system_color(text: str) -> str:
    """Format text in yellow for system messages."""
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


def error_color(text: str) -> str:
    """Format text in red for error messages."""
    return f"{Fore.RED}{text}{Style.RESET_ALL}"
