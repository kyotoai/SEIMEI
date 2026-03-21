from __future__ import annotations

import logging
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


class ColoredFormatter(logging.Formatter):
    """logging.Formatter that wraps messages in ANSI color codes by log level.

    If a message already contains ANSI codes (e.g. pre-colorized agent output
    blocks), it is passed through unchanged so colors are never doubled.
    """

    _LEVEL_COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED,
    }

    def __init__(self, *args: object, enable_color: Optional[bool] = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._enable_color: bool = enable_color if enable_color is not None else supports_color()

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        # If the message already contains ANSI codes, leave it untouched.
        if ANSI_RESET in message:
            return message
        color = self._LEVEL_COLORS.get(record.levelno)
        return colorize(message, color, enable=self._enable_color)


def setup_seimei_logging(level: int = logging.INFO) -> None:
    """Configure the ``seimei`` logger with a colored stream handler.

    Safe to call multiple times — subsequent calls only update the level on
    existing handlers without adding duplicates.
    """
    logger = logging.getLogger("seimei")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter("%(message)s"))
    logger.addHandler(handler)


__all__ = [
    "LogColors",
    "colorize",
    "supports_color",
    "ANSI_RESET",
    "ColoredFormatter",
    "setup_seimei_logging",
]
