from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Type, Sequence, Tuple, Union
import csv
import json
import random
import re
from pathlib import Path

from seimei.utils import format_query_for_rmsearch
from seimei.llm import TokenLimitExceeded


_AGENT_REGISTRY: Dict[str, Type['Agent']] = {}

def get_agent_subclasses() -> Dict[str, Type['Agent']]:
    return dict(_AGENT_REGISTRY)

def _register_agent(subcls: Type['Agent']) -> None:
    # Use explicit `name` if provided, otherwise class name in snake-ish
    name = getattr(subcls, "name", None) or subcls.__name__
    setattr(subcls, "name", name)
    _AGENT_REGISTRY[name] = subcls

@dataclass
class Agent:
    """Base Agent with friendly async call and step logging.

    Subclasses should override `inference` and optionally set:
      - `name`: short identifier (default = class name)
      - `description`: routing hint for rmsearch/heuristics
    """

    name: str = field(init=False)
    description: str = field(init=False, default="")

    def __post_init__(self) -> None:
        cls = type(self)
        cls_name = getattr(cls, "name", None) or cls.__name__
        setattr(cls, "name", cls_name)
        self.name = cls_name

        cls_desc = getattr(cls, "description", "")
        setattr(cls, "description", cls_desc)
        self.description = cls_desc
        self.__agent_routing_knowledge: Optional[List[Dict[str, Any]]] = None

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Wraps concrete inference with basic logging/error capture.

        self.__shared_ctx = shared_ctx

        try:
            res = await self.inference(messages=messages, shared_ctx=shared_ctx or {}, **kwargs)
            if not isinstance(res, dict):
                res = {"content": str(res)}
            # Normalize shape
            res.setdefault("content", "")
            return res
        except TokenLimitExceeded:
            # Surface token limit errors so the orchestrator can halt the run.
            raise
        except Exception as e:
            return {"content": f"[error] {type(e).__name__}: {e}"}

    # ------------------ to override ------------------
    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError
    

    async def get_agent_knowledge(
        self,
        max_items: Optional[int] = 3,
    ) -> List[Dict[str, Any]]:
        """Fetch normalized knowledge entries for an agent (including wildcard entries)."""

        shared_ctx = self.__shared_ctx or {}
        agent_name = self.name

        routing_knowledge = getattr(self, "_Agent__agent_routing_knowledge", None)
        if routing_knowledge:
            routed: List[Dict[str, Any]] = []
            for raw_entry in _iter_entries(routing_knowledge):
                normalized = _normalize_entry(raw_entry)
                if not normalized:
                    continue
                if not normalized.get("id"):
                    normalized["id"] = len(routed) + 1
                routed.append(normalized)
            if routed:
                limit = _sanitize_limit(max_items, default=3)
                if limit >= len(routed):
                    return list(routed)
                return list(routed[:limit])

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
                    normalized["id"] = len(collected) + 1
                collected.append(normalized)
        if not collected:
            return []

        limit = _sanitize_limit(max_items, default=3)
        search_fn = shared_ctx.get("search")
        if callable(search_fn):
            keys: List[Dict[str, Any]] = []
            for entry in collected:
                text = entry.get("text") or ""
                if not text:
                    continue
                key_entry = {
                    "key": text,
                    "knowledge": entry,
                    "knowledge_id": entry.get("id"),
                    "tags": entry.get("tags", []),
                }
                keys.append(key_entry)
            if keys:
                query = _build_knowledge_query(shared_ctx, agent_name)
                try:
                    ranked = await search_fn(
                        query=query,
                        keys=list(keys),
                        k=min(limit, len(keys)),
                        context={
                            "purpose": "knowledge_search",
                            "query_override": shared_ctx.get("knowledge_query"),
                        },
                    )
                except Exception:
                    ranked = []
                ranked_entries: List[Dict[str, Any]] = []
                if isinstance(ranked, list):
                    key_map = {item.get("key"): item.get("knowledge") for item in keys if item.get("key")}
                    for item in ranked:
                        if not isinstance(item, dict):
                            continue
                        payload = item.get("payload")
                        entry: Optional[Dict[str, Any]] = None
                        if isinstance(payload, dict) and payload.get("knowledge"):
                            entry = payload.get("knowledge")
                        elif isinstance(payload, dict):
                            entry = payload
                        else:
                            key = item.get("key")
                            if key in key_map:
                                entry = key_map[key]
                        if entry and entry not in ranked_entries:
                            ranked_entries.append(entry)
                        if len(ranked_entries) >= limit:
                            break
                if ranked_entries:
                    return ranked_entries

        knowledge_search_mode = str(shared_ctx.get("knowledge_search_mode") or "rm").strip().lower()
        rm_config = dict(shared_ctx.get("rm_config") or {})
        rmsearch_fn = shared_ctx.get("rmsearch_fn")
        if knowledge_search_mode == "rm" and rm_config and callable(rmsearch_fn):
            ranked = _rank_with_rmsearch(
                rmsearch_fn=rmsearch_fn,
                rm_config=rm_config,
                agent_name=agent_name,
                candidates=collected,
                limit=limit,
                shared_ctx=shared_ctx,
                purpose="knowledge_search",
            )
            if ranked:
                return ranked

        if limit >= len(collected):
            return list(collected)
        return random.sample(collected, k=limit)



# Class decorator to auto-register agents
def register(cls: Type[Agent]) -> Type[Agent]:
    _register_agent(cls)
    return cls


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

    raw_id = entry.get("id") or entry.get("knowledge_id")
    normalized_id = _coerce_numeric_id(raw_id)
    if normalized_id is None:
        meta = entry.get("meta")
        if isinstance(meta, dict):
            normalized_id = _coerce_numeric_id(meta.get("row_index"))

    normalized: Dict[str, Any] = {
        "id": normalized_id,
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
        for idx, raw in enumerate(reader, start=1):
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
            explicit_id = raw.get("id")
            if explicit_id not in (None, ""):
                entry["id"] = explicit_id
            else:
                entry["id"] = idx
            tags = _parse_tags(raw.get("tags"))
            if tags:
                entry["tags"] = tags
            for extra_key in raw.keys():
                if extra_key in {"agent", "knowledge", "text", "content", "value", "tags", "id"}:
                    continue
                value = raw.get(extra_key)
                if value not in (None, ""):
                    entry.setdefault("meta", {})[extra_key] = value
            entry.setdefault("meta", {})["row_index"] = idx
            rows.append(entry)
    return rows


def _coerce_numeric_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.isdigit() or (candidate.startswith("-") and candidate[1:].isdigit()):
            try:
                return int(candidate)
            except ValueError:
                return None
        match = re.search(r"(\d+)", candidate)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def prepare_knowledge_payload(
    entries: Optional[Sequence[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    normalized: List[Dict[str, Any]] = []
    log_texts: List[str] = []
    id_list: List[int] = []
    if not entries:
        return normalized, log_texts, id_list

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text_value = (
            entry.get("text")
            or entry.get("knowledge")
            or entry.get("content")
            or entry.get("value")
            or ""
        )
        text_str = str(text_value).strip()
        if not text_str:
            continue

        normalized_entry: Dict[str, Any] = {"text": text_str}
        tags_raw = entry.get("tags")
        tags_list: List[str] = []
        if isinstance(tags_raw, (list, tuple, set)):
            for tag in tags_raw:
                tag_str = str(tag).strip()
                if tag_str:
                    tags_list.append(tag_str)
        if tags_list:
            normalized_entry["tags"] = tags_list

        kid = entry.get("id") or entry.get("knowledge_id")
        kid_int = _coerce_numeric_id(kid)
        if kid_int is not None:
            normalized_entry["id"] = kid_int
            id_list.append(kid_int)

        normalized.append(normalized_entry)
        log_texts.append(text_str)

    return normalized, log_texts, id_list


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


def _sanitize_limit(max_items: Optional[int], default: int) -> int:
    if max_items is None:
        return max(int(default), 1)
    try:
        value = int(max_items)
    except (TypeError, ValueError):
        return max(int(default), 1)
    return max(value, 1)


def _rank_with_rmsearch(
    rmsearch_fn: Any,
    rm_config: Dict[str, Any],
    agent_name: str,
    candidates: List[Dict[str, Any]],
    limit: int,
    shared_ctx: Dict[str, Any],
    purpose: str,
) -> List[Dict[str, Any]]:
    if not candidates or not callable(rmsearch_fn):
        return []

    keys: List[Dict[str, Any]] = []
    for entry in candidates:
        text = entry.get("text") or ""
        if not text:
            continue
        key_entry = {
            "key": text,
            "knowledge": entry,
            "knowledge_id": entry.get("id"),
            "tags": entry.get("tags", []),
        }
        keys.append(key_entry)
    if not keys:
        return []

    query = _build_knowledge_query(shared_ctx, agent_name)
    key_map = {item["key"]: item["knowledge"] for item in keys if item.get("key")}
    try:
        result = rmsearch_fn(
            query=query,
            keys=list(keys),
            k_key=min(limit, len(keys)),
            purpose=purpose,
            **rm_config,
        )
    except Exception:
        return []

    ranked_entries: List[Dict[str, Any]] = []
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            payload = item.get("payload")
            entry: Optional[Dict[str, Any]] = None
            if isinstance(payload, dict) and payload.get("knowledge"):
                entry = payload["knowledge"]
            elif key in key_map:
                entry = key_map[key]
            if entry and entry not in ranked_entries:
                ranked_entries.append(entry)
            if len(ranked_entries) >= limit:
                break
    return ranked_entries[:limit]


def _build_knowledge_query(shared_ctx: Dict[str, Any], agent_name: str) -> str:
    override = shared_ctx.get("knowledge_query")
    if isinstance(override, dict):
        value = override.get(agent_name) or override.get("*")
        if isinstance(value, str) and value.strip():
            return format_query_for_rmsearch(value.strip())
    elif isinstance(override, str) and override.strip():
        return format_query_for_rmsearch(override.strip())
    return format_query_for_rmsearch(f"Relevant knowledge for agent '{agent_name}'")


__all__ = ["get_agent_knowledge", "load_knowledge"]
