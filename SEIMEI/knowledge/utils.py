from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List


def get_agent_knowledge(shared_ctx: Dict[str, Any], agent_name: str) -> List[Dict[str, Any]]:
    """Fetch normalized knowledge entries for an agent (including wildcard entries)."""
    knowledge = shared_ctx.get("knowledge")
    if not isinstance(knowledge, dict):
        return []

    collected: List[Dict[str, Any]] = []
    for key in (agent_name, "*"):
        value = knowledge.get(key)
        if value is None:
            continue
        for raw_entry in _iter_entries(value):
            normalized = _normalize_entry(raw_entry)
            if not normalized:
                continue
            if not normalized.get("id"):
                normalized["id"] = f"{agent_name}_{len(collected)}"
            collected.append(normalized)
    return collected


def _iter_entries(value: Any) -> Iterator[Dict[str, Any]]:
    if value is None:
        return
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, str):
        yield {"knowledge": value}
        return
    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_entries(item)


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    text = str(
        entry.get("knowledge")
        or entry.get("text")
        or entry.get("content")
        or entry.get("value")
        or ""
    ).strip()
    if not text:
        return {}

    tags_raw = entry.get("tags")
    if isinstance(tags_raw, str):
        tags = [part.strip() for part in tags_raw.split(",") if part.strip()]
    elif isinstance(tags_raw, Iterable):
        tags = [str(part).strip() for part in tags_raw if str(part).strip()]
    else:
        tags = []

    normalized: Dict[str, Any] = {
        "id": entry.get("id") or entry.get("knowledge_id"),
        "text": text,
        "tags": tags,
    }
    return normalized


__all__ = ["get_agent_knowledge"]
