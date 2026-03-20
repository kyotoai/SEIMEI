from __future__ import annotations

import codecs
import os
import sys
import termios
import tty
import unicodedata
from typing import List, Sequence

from ..logging_utils import LogColors, colorize


def _char_display_width(ch: str) -> int:
    if not ch:
        return 0
    if unicodedata.combining(ch):
        return 0
    if unicodedata.category(ch)[0] == "C":
        return 0
    return 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1


def _text_display_width(text: str) -> int:
    return sum(_char_display_width(ch) for ch in text)


class TerminalUI:
    """Interactive terminal input with slash-command suggestion support."""

    def __init__(self, *, color_enabled: bool) -> None:
        self.color_enabled = color_enabled

    def read_line(self, prompt: str, commands: Sequence[str]) -> str:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return input(prompt)

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        chars: List[str] = []
        cursor = 0
        selected = 0
        utf8_decoder = codecs.getincrementaldecoder("utf-8")()

        def current_text() -> str:
            return "".join(chars)

        def matching_commands() -> List[str]:
            text = current_text()
            if not text.startswith("/"):
                return []
            if " " in text:
                return []
            return [cmd for cmd in commands if cmd.startswith(text)]

        def render(suggestions: Sequence[str], selected_index: int) -> None:
            text = current_text()
            sys.stdout.write("\r\x1b[2K" + prompt + text)

            right_span = max(_text_display_width(text[cursor:]), 0)
            if right_span:
                sys.stdout.write(f"\x1b[{right_span}D")

            # Paint suggestions below input line while keeping cursor position on input.
            sys.stdout.write("\x1b[s")
            sys.stdout.write("\x1b[J")
            for idx, cmd in enumerate(suggestions):
                prefix = "> " if idx == selected_index else "  "
                line = f"{prefix}{cmd}"
                if idx == selected_index:
                    line = colorize(line, LogColors.BOLD_MAGENTA, enable=self.color_enabled)
                else:
                    line = colorize(line, LogColors.GRAY, enable=self.color_enabled)
                sys.stdout.write("\n" + line)
            sys.stdout.write("\x1b[u")
            sys.stdout.flush()

        try:
            tty.setraw(fd)
            render([], 0)

            while True:
                ch = os.read(fd, 1)
                if not ch:
                    raise EOFError

                if ch in {b"\r", b"\n"}:
                    utf8_decoder.reset()
                    matches = matching_commands()
                    entered = current_text().strip()
                    if matches and entered.startswith("/") and " " not in entered:
                        entered = matches[min(selected, len(matches) - 1)]

                    # Clear suggestion region before finalizing line.
                    chars = list(entered)
                    cursor = len(chars)
                    render([], 0)
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return entered

                if ch == b"\x03":
                    utf8_decoder.reset()
                    raise KeyboardInterrupt

                if ch == b"\x04":
                    utf8_decoder.reset()
                    if not chars:
                        raise EOFError
                    continue

                if ch in {b"\x7f", b"\x08"}:  # backspace
                    utf8_decoder.reset()
                    if cursor > 0:
                        del chars[cursor - 1]
                        cursor -= 1
                    selected = 0
                    matches = matching_commands()
                    render(matches, min(selected, max(len(matches) - 1, 0)) if matches else 0)
                    continue

                if ch == b"\x1b":
                    utf8_decoder.reset()
                    nxt = os.read(fd, 1)
                    if nxt != b"[":
                        continue
                    direction = os.read(fd, 1)
                    matches = matching_commands()
                    if direction == b"A" and matches:  # up
                        selected = (selected - 1) % len(matches)
                        render(matches, selected)
                        continue
                    if direction == b"B" and matches:  # down
                        selected = (selected + 1) % len(matches)
                        render(matches, selected)
                        continue
                    if direction == b"C":  # right
                        if cursor < len(chars):
                            cursor += 1
                        render(matches, min(selected, max(len(matches) - 1, 0)) if matches else 0)
                        continue
                    if direction == b"D":  # left
                        if cursor > 0:
                            cursor -= 1
                        render(matches, min(selected, max(len(matches) - 1, 0)) if matches else 0)
                        continue
                    continue

                if ch == b"\t":
                    utf8_decoder.reset()
                    matches = matching_commands()
                    if matches:
                        chosen = matches[min(selected, len(matches) - 1)]
                        chars = list(chosen)
                        cursor = len(chars)
                        selected = 0
                        render(matching_commands(), selected)
                    continue

                try:
                    decoded = utf8_decoder.decode(ch, final=False)
                except UnicodeDecodeError:
                    utf8_decoder.reset()
                    continue

                if not decoded:
                    continue

                for letter in decoded:
                    if letter.isprintable():
                        chars.insert(cursor, letter)
                        cursor += 1
                        selected = 0

                matches = matching_commands()
                render(matches, min(selected, max(len(matches) - 1, 0)) if matches else 0)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
