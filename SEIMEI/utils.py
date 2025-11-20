from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
