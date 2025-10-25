from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union


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


def load_knowledge(path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Load knowledge entries from CSV, JSON, or JSONL into a dict keyed by agent."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        entries = _load_csv(file_path)
    elif suffix == ".json":
        entries = _load_json(file_path)
    elif suffix == ".jsonl":
        entries = _load_jsonl(file_path)
    else:
        raise ValueError(f"Unsupported knowledge file format: {file_path.suffix}")

    knowledge_store: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        agent = str(entry.get("agent", "")).strip()
        if not agent:
            continue
        knowledge_store.setdefault(agent, []).append(entry)
    return knowledge_store


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            agent = (raw.get("agent") or "").strip()
            knowledge_text = (
                raw.get("knowledge")
                or raw.get("text")
                or raw.get("content")
                or raw.get("value")
                or ""
            ).strip()
            if not agent or not knowledge_text:
                continue
            entry: Dict[str, Any] = {"agent": agent, "knowledge": knowledge_text}
            if raw.get("id"):
                entry["id"] = raw["id"]
            tags = _parse_tags(raw.get("tags"))
            if tags:
                entry["tags"] = tags
            for extra_key in raw.keys():
                if extra_key in {"agent", "knowledge", "text", "content", "value", "tags", "id"}:
                    continue
                value = raw.get(extra_key)
                if value not in (None, ""):
                    entry.setdefault("meta", {})[extra_key] = value
            rows.append(entry)
    return rows


def _load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("knowledge") or data.get("entries") or data
    if not isinstance(data, list):
        raise ValueError("Knowledge JSON must contain a list of entries.")
    entries: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        coerced = _coerce_json_entry(item)
        if coerced:
            entries.append(coerced)
    return entries


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                coerced = _coerce_json_entry(obj)
                if coerced:
                    entries.append(coerced)
    return entries


def _coerce_json_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    agent = (item.get("agent") or item.get("name") or "").strip()
    knowledge_text = (
        item.get("knowledge")
        or item.get("text")
        or item.get("content")
        or item.get("value")
        or ""
    ).strip()
    if not agent or not knowledge_text:
        return {}
    entry: Dict[str, Any] = {"agent": agent, "knowledge": knowledge_text}
    if item.get("id"):
        entry["id"] = item["id"]
    tags = _parse_tags(item.get("tags"))
    if tags:
        entry["tags"] = tags
    for key, value in item.items():
        if key in {"agent", "name", "knowledge", "text", "content", "value", "tags", "id"}:
            continue
        entry.setdefault("meta", {})[key] = value
    return entry


def _parse_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, Iterable):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


__all__ = ["get_agent_knowledge", "load_knowledge"]
