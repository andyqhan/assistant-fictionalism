import sys
import select
import termios
import tty
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def cbreak_mode() -> Iterator[None]:
    """Context manager to put terminal in cbreak mode for key detection."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def check_escape() -> bool:
    """Check if escape key was pressed (non-blocking). Must be in cbreak mode."""
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        return key == "\x1b"  # Escape character
    return False
