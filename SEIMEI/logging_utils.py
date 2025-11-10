from __future__ import annotations

import sys
from typing import Optional

ANSI_RESET = "\033[0m"


class LogColors:
    """Named ANSI color codes for consistent logging styles."""

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD_MAGENTA = "\033[1m\033[95m"
    RED = "\033[91m"
    YELLOW = "\033[93m"


def supports_color() -> bool:
    """Best-effort detection for whether stdout/stderr support ANSI colors."""
    return bool(
        getattr(sys.stdout, "isatty", lambda: False)()
        or getattr(sys.stderr, "isatty", lambda: False)()
    )


def colorize(text: str, color_code: Optional[str] = None, *, enable: Optional[bool] = None) -> str:
    """Wrap text with an ANSI color code when enabled."""
    if enable is None:
        enable = supports_color()
    if not (enable and color_code):
        return text
    return f"{color_code}{text}{ANSI_RESET}"


__all__ = ["LogColors", "colorize", "supports_color", "ANSI_RESET"]
