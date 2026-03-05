from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_RUN_NAME_RE = re.compile(r"^run-(\d{8})-(\d{6})-[a-zA-Z0-9]+$")


@dataclass
class ChatHistoryItem:
    run_name: str
    run_path: Path
    user_message: str
    assistant_message: str

    @property
    def preview(self) -> str:
        one_line = " ".join(self.user_message.split())
        if len(one_line) > 100:
            return one_line[:100].rstrip() + "..."
        return one_line


def _run_sort_key(run_dir: Path) -> Tuple[int, datetime, float]:
    name = run_dir.name
    match = _RUN_NAME_RE.match(name)
    dt = datetime.min
    if match:
        try:
            dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            dt = datetime.min
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (1 if match else 0, dt, mtime)


def _load_messages(messages_path: Path) -> List[Dict[str, object]]:
    try:
        raw = json.loads(messages_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [dict(entry) for entry in raw if isinstance(entry, dict)]


def _extract_last_user_assistant_pair(
    messages: Sequence[Dict[str, object]],
) -> Optional[Tuple[str, str]]:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        role = str(msg.get("role") or "").lower()
        if role != "user":
            continue

        question = str(msg.get("content") or "").strip()
        if not question:
            continue

        for nxt in range(idx + 1, len(messages)):
            reply = messages[nxt]
            if str(reply.get("role") or "").lower() != "assistant":
                continue
            answer = str(reply.get("content") or "").strip()
            if not answer:
                continue
            return question, answer
    return None


def load_recent_history(log_dir: str, run_limit: int = 20) -> List[ChatHistoryItem]:
    root = Path(log_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    run_dirs = [
        entry
        for entry in root.iterdir()
        if entry.is_dir() and entry.name.startswith("run-")
    ]
    run_dirs.sort(key=_run_sort_key, reverse=True)

    items: List[ChatHistoryItem] = []
    for run_dir in run_dirs[:run_limit]:
        messages_path = run_dir / "messages.json"
        if not messages_path.exists():
            continue
        messages = _load_messages(messages_path)
        pair = _extract_last_user_assistant_pair(messages)
        if not pair:
            continue
        user_message, assistant_message = pair
        items.append(
            ChatHistoryItem(
                run_name=run_dir.name,
                run_path=run_dir,
                user_message=user_message,
                assistant_message=assistant_message,
            )
        )
    return items
