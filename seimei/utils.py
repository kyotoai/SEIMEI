from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

DEFAULT_RUNS_DIR = Path("seimei_runs")


def load_run_messages(
    run_path: Union[str, Path],
    *,
    step: Optional[int] = None,
    runs_dir: Union[str, Path, None] = DEFAULT_RUNS_DIR,
) -> List[Dict[str, Any]]:
    """Return the recorded conversation for a completed SEIMEI run.

    Args:
        run_path: Either a run directory (e.g., "run-20250101-123000-abc") or an
            explicit path to a ``messages.json`` file.
        step: Optional 1-indexed number of agent turns to retain. When provided,
            messages are returned up to and including that agent turn, allowing a
            caller to resume execution before the final assistant response.
        runs_dir: Optional base directory that stores run artefacts. When
            ``run_path`` does not resolve to an existing file/directory, this
            directory is combined with ``run_path`` to locate ``messages.json``.

    Returns:
        The conversation history as a list of dictionaries ready to be passed
        back into :class:`seimei.seimei`.

    Raises:
        FileNotFoundError: If ``messages.json`` cannot be found.
        ValueError: If ``messages.json`` does not contain a list of messages or
            if ``step`` is present but < 1.
    """

    messages_path = _resolve_messages_path(run_path, runs_dir)

    with messages_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{messages_path} does not contain a list of messages")

    normalized: List[Dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            normalized.append(dict(entry))

    if not normalized:
        return []

    if step is None:
        return normalized

    try:
        limit = int(step)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("step must be an integer >= 1") from exc
    if limit <= 0:
        raise ValueError("step must be >= 1 when provided")

    truncated: List[Dict[str, Any]] = []
    agent_seen = 0
    for msg in normalized:
        truncated.append(msg)
        if _is_agent_message(msg):
            agent_seen += 1
            if agent_seen >= limit:
                break

    return truncated


def _resolve_messages_path(
    run_path: Union[str, Path],
    runs_dir: Union[str, Path, None],
) -> Path:
    candidate = Path(run_path).expanduser()
    possible: List[Path] = []

    if candidate.exists():
        possible.append(candidate)
    if runs_dir is not None:
        possible.append(Path(runs_dir).expanduser() / str(run_path))

    for path in possible:
        if path.is_file():
            if path.name.endswith(".json"):
                return path
            continue
        if path.is_dir():
            msg_file = path / "messages.json"
            if msg_file.exists():
                return msg_file

    raise FileNotFoundError(
        f"messages.json not found for run '{run_path}' (checked: {possible or [candidate]})"
    )


def _is_agent_message(message: Dict[str, Any]) -> bool:
    role = str(message.get("role") or "").lower()
    if role == "agent":
        return True
    if role == "system" and message.get("agent"):
        return True
    return False


def format_query_for_rmsearch(body: str) -> str:
    """Wrap a query body with the standardized <query>...</query> block."""
    text = (body or "").strip()
    if text.startswith("<query>") and "</query>" in text:
        return text
    if not text:
        text = "[missing query context]"
    return f"<query>\n{text}\n</query>"


def format_key_for_rmsearch(
    content: str,
    *,
    tags: Optional[Sequence[Any]] = None,
) -> str:
    """Format a candidate key with <key>...</key> and the score suffix."""
    tag_values: List[str] = []
    for tag in tags or []:
        tag_text = str(tag).strip()
        if tag_text:
            tag_values.append(tag_text)

    base_text = (content or "").strip()
    if not base_text:
        base_text = "[missing key text]"

    if base_text.startswith("<key>") and "</key>" in base_text:
        formatted = base_text
    else:
        lines = [base_text]
        if tag_values:
            lines.append(f"Tags: {', '.join(tag_values)}")
        join_lines = "\n".join(lines)
        formatted = f"<key>\n{join_lines}\n</key>"

    formatted = formatted.strip()
    if "Query-Key Relevance Score:" not in formatted:
        formatted = f"{formatted}\n\n\nQuery-Key Relevance Score:"
    return formatted


def _normalize_purpose(raw_purpose: Optional[str]) -> str:
    if not raw_purpose:
        return "knowledge_search"
    value = str(raw_purpose).strip().lower()
    if value == "agent_routing":
        return "agent_routing"
    return "knowledge_search"


def _coerce_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        tags: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                tags.append(text)
        return tags
    return []


def _coerce_messages(
    query: Union[str, Sequence[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    if isinstance(query, str):
        try:
            data = json.loads(query)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception:
            pass
        return [{"role": "user", "content": query}]
    if isinstance(query, Sequence):
        return [dict(m) for m in query if isinstance(m, dict)]
    return [{"role": "user", "content": str(query)}]


def _prepare_query_input(
    query: Union[str, Sequence[Dict[str, Any]]],
    *,
    max_messages: int = 8,
    max_chars_per_message: int = 400,
) -> Tuple[List[Dict[str, Any]], str, str]:
    messages = _coerce_messages(query)

    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    focus = ""
    for msg in reversed(messages):
        if str(msg.get("role") or "").lower() == "user":
            focus = _stringify(msg.get("content", ""))
            if focus:
                break

    label_map = {
        "user": "User",
        "assistant": "Assistant",
        "agent": "Agent",
        "system": "System",
        "function": "Function",
        "developer": "Developer",
    }
    lines: List[str] = []
    for msg in list(messages)[-max_messages:]:
        role_raw = (msg.get("role") or "").lower()
        if role_raw == "system":
            continue
        role = label_map.get(role_raw, "Message")
        snippet = _stringify(msg.get("content", "")).strip()
        if len(snippet) > max_chars_per_message:
            snippet = snippet[:max_chars_per_message].rstrip() + "..."
        lines.append(f"{role}: {snippet}")

    conversation = "\n".join(lines) if lines else ""
    if not focus:
        focus = conversation
    return messages, conversation, focus
