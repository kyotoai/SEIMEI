from __future__ import annotations

import asyncio
import importlib.util
import inspect
import io
import json
import os
import sys
import time
import types
import uuid
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests

# Re-export convenience (allows: `from seimei import seimei, llm, agent`)
from . import llm as llm_module
from . import agent as agent_module
llm = llm_module
agent = agent_module

from .agent import Agent, get_agent_subclasses
from . import agents as builtin_agents  # noqa: F401  # ensure built-in agents register
from .llm import LLMClient, TokenLimiter, TokenLimitExceeded
from .knowledge import DEFAULT_RUN_PROMPT, generate_knowledge_from_runs, load_knowledge
from .logging_utils import LogColors, colorize, supports_color

STEP_TITLE_COLOR = LogColors.GREEN
LOG_BLOCK_COLOR = LogColors.CYAN
ANSWER_BLOCK_COLOR = LogColors.BOLD_MAGENTA
ERROR_COLOR = LogColors.RED
KNOWLEDGE_COLOR = LogColors.YELLOW
DEFAULT_RMSEARCH_URL = "https://hm465ys5n3.execute-api.ap-southeast-2.amazonaws.com/prod/v1/rmsearch"

class seimei:
    """Main orchestrator.

    Loads agents from user-specified paths, routes steps (via rmsearch if available),
    calls the LLM, and writes a dataset for each run.
    """

    AGENT_OUTPUT_LIMIT = 8000
    _SENSITIVE_METADATA_KEYS = {
        "api_key",
        "apikey",
        "api_token",
        "access_token",
        "refresh_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "secret",
        "password",
    }

    def __init__(
        self,
        agent_config: Optional[Sequence[Dict[str, Any]]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        rm_kwargs: Optional[Dict[str, Any]] = None,
        log_dir: str = "./seimei_runs",
        max_steps: int = 8,
        allow_code_exec: bool = False,
        allowed_commands: Optional[Sequence[str]] = None,
        approval_callback: Optional[Callable[[str], bool]] = None,
        agent_log_head_lines: int = 3,
        max_tokens_per_question: Optional[int] = None,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # LLM
        if llm_kwargs is None:
            raise ValueError("llm_kwargs must be provided")
        self.llm = LLMClient(**llm_kwargs)

        # Routing
        default_rm_settings = {
            "agent_routing": False,
            "knowledge_search": True,
            "url": DEFAULT_RMSEARCH_URL,
        }
        provided_rm_kwargs = rm_kwargs or {}
        self.rm_kwargs = {**default_rm_settings, **provided_rm_kwargs}
        self.max_steps = max_steps
        self._rm_warned_missing_url = False
        self.max_tokens_per_question = max_tokens_per_question

        # Safety for code_act
        self.allow_code_exec = allow_code_exec
        self.allowed_commands = list(allowed_commands) if allowed_commands else None
        self.approval_callback = approval_callback
        self.agent_log_head_lines = max(int(agent_log_head_lines), 0)
        self.load_knowledge_path: Optional[str] = None
        self.knowledge_store: Dict[str, List[Dict[str, Any]]] = {}

        # Load agents
        self.agents: Dict[str, Agent] = {}
        normalized_agent_config = list(agent_config or [])
        self.agent_config_spec = [
            dict(cfg) for cfg in normalized_agent_config if isinstance(cfg, dict)
        ]
        self._load_agents(normalized_agent_config)

        # Attach shared ctx visible to agents (e.g., llm, rmsearch, safety flags)
        self.shared_ctx = {
            "llm": self.llm,
            "rm_kwargs": self.rm_kwargs,
            "rmsearch_fn": self._rmsearch,
            "allow_code_exec": self.allow_code_exec,
            "allowed_commands": self.allowed_commands,
            "approval_callback": self.approval_callback,
            "search": None,
            "knowledge": self.knowledge_store,
        }

    # -------------------------- Agent loading --------------------------

    def _load_agents(self, configs: Sequence[Dict[str, Any]]) -> None:
        for cfg in configs:
            dir_path = cfg.get("dir_path")
            file_path = cfg.get("file_path")
            if dir_path:
                self._load_agents_from_dir(dir_path)
            if file_path:
                self._load_agents_from_file(file_path)

        # instantiate
        for cls in get_agent_subclasses().values():
            try:
                inst = cls()
                self.agents[inst.name] = inst
            except Exception as e:
                print(
                    colorize(f"[seimei] Failed to instantiate agent {cls}: {e}", ERROR_COLOR),
                    file=sys.stderr,
                )

    def _load_agents_from_dir(self, path: str) -> None:
        if not os.path.isdir(path):
            return
        for fname in os.listdir(path):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            self._load_agents_from_file(os.path.join(path, fname))

    def _load_agents_from_file(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        mod_name = f"seimei_user_agents_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)  # type: ignore

    def _load_knowledge_store(self, path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not path:
            return {}
        try:
            store = load_knowledge(path)
            print(colorize(f"[seimei] Knowledge loaded from {path} ({sum(len(v) for v in store.values())} entries)", STEP_TITLE_COLOR))
            return store
        except FileNotFoundError as exc:
            print(colorize(f"[seimei] Knowledge file not found: {exc}", ERROR_COLOR), file=sys.stderr)
        except Exception as exc:  # pragma: no cover
            print(colorize(f"[seimei] Failed to load knowledge from {path}: {exc}", ERROR_COLOR), file=sys.stderr)
        return {}

    def _refresh_knowledge_store(self, path: Path) -> None:
        try:
            store = load_knowledge(path)
        except Exception as exc:  # pragma: no cover - best-effort reload
            print(colorize(f"[seimei] Failed to reload knowledge from {path}: {exc}", ERROR_COLOR), file=sys.stderr)
            return
        self.load_knowledge_path = str(path)
        self.knowledge_store = store
        self.shared_ctx["knowledge"] = store

    def _resolve_knowledge_output_path(self, override: Optional[str]) -> Path:
        candidate = override or self.load_knowledge_path
        if candidate:
            return Path(candidate).expanduser()
        return Path("seimei_knowledge") / "knowledge.csv"

    def _normalize_knowledge_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(config or {})
        manual_entries, manual_stores, manual_store_sources = self._normalize_manual_knowledge(cfg.get("knowledge"))

        load_path_provided = "load_knowledge_path" in cfg
        load_value = cfg.get("load_knowledge_path")
        normalized_load_path = self._coerce_path_string(load_value)

        save_value = cfg.get("save_knowledge_path")
        save_path = Path(save_value).expanduser() if save_value not in (None, "") else None

        prompt_value = cfg.get("knowledge_prompt_path")
        prompt_path = Path(prompt_value).expanduser() if prompt_value not in (None, "") else None

        return {
            "generate_knowledge": bool(cfg.get("generate_knowledge", False)),
            "save_knowledge_path": save_path,
            "knowledge_prompt_path": prompt_path,
            "load_knowledge_path": normalized_load_path,
            "load_path_provided": load_path_provided,
            "manual_entries": manual_entries,
            "manual_stores": manual_stores,
            "manual_store_sources": manual_store_sources,
        }

    def _normalize_manual_knowledge(
        self, value: Any
    ) -> Tuple[
        Dict[Optional[int], List[Dict[str, Any]]],
        Dict[Optional[int], List[Dict[str, Any]]],
        Dict[Optional[int], List[Dict[str, Any]]],
    ]:
        entry_map: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        store_map: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        store_source_map: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        entries = self._coerce_manual_knowledge_list(value)
        if not entries:
            return {}, {}, {}

        for raw in entries:
            steps = self._coerce_step_targets(raw.get("step"))
            targets = steps or [None]
            tags = self._coerce_tags(raw.get("tags"))
            agent_name = str(raw.get("agent") or "*").strip() or "*"
            text_value = raw.get("text") or raw.get("knowledge") or raw.get("content") or raw.get("value")
            text = str(text_value).strip() if text_value is not None else ""
            entry_id = self._coerce_optional_int(raw.get("id"))
            load_path_value = raw.get("load_knowledge_path")
            load_path = self._coerce_path_string(load_path_value)

            entry_payload: Optional[Dict[str, Any]] = None
            if text:
                entry_payload = {"agent": agent_name, "knowledge": text}
                if tags:
                    entry_payload["tags"] = tags
                if entry_id is not None:
                    entry_payload["id"] = entry_id

            store_payload: Optional[Dict[str, List[Dict[str, Any]]]] = None
            store_meta: Optional[Dict[str, Any]] = None
            if load_path:
                store_meta = {"path": load_path, "loaded": False}
                store_payload = self._load_manual_knowledge_store(load_path)
                if store_payload:
                    store_meta["loaded"] = True
                    total_entries = 0
                    agent_names: List[str] = []
                    for agent, entries_list in store_payload.items():
                        agent_names.append(str(agent))
                        if isinstance(entries_list, list):
                            total_entries += len(entries_list)
                    store_meta["entries"] = total_entries
                    store_meta["agents"] = sorted(set(agent_names))
                else:
                    store_payload = None

            for step in targets:
                if entry_payload:
                    entry_map[step].append(dict(entry_payload))
                if store_payload:
                    store_map[step].append(store_payload)
                if store_meta:
                    store_source_map[step].append(dict(store_meta))
                elif load_path:
                    store_source_map[step].append({"path": load_path, "loaded": False})

        return dict(entry_map), dict(store_map), dict(store_source_map)

    def _coerce_manual_knowledge_list(self, raw: Any) -> List[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [{"text": raw}]
        if isinstance(raw, dict):
            return [dict(raw)]
        if isinstance(raw, Iterable):
            entries: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, str):
                    entries.append({"text": item})
                elif isinstance(item, dict):
                    entries.append(dict(item))
            return entries
        return []

    @staticmethod
    def _coerce_step_targets(value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            steps: List[int] = []
            for item in value:
                coerced = seimei._coerce_positive_int(item)
                if coerced is not None:
                    steps.append(coerced)
            return steps
        coerced = seimei._coerce_positive_int(value)
        return [coerced] if coerced is not None else []

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number

    @staticmethod
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

    def _compose_step_knowledge(
        self,
        *,
        base_store: Dict[str, List[Dict[str, Any]]],
        manual_entries: Dict[Optional[int], List[Dict[str, Any]]],
        manual_stores: Dict[Optional[int], List[Dict[str, Any]]],
        step: Optional[int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        merged: Dict[str, List[Dict[str, Any]]] = {}

        def _merge_store(store: Dict[str, List[Dict[str, Any]]]) -> None:
            for agent, entries in store.items():
                if not isinstance(entries, list):
                    continue
                agent_entries = merged.setdefault(agent, [])
                for entry in entries:
                    if isinstance(entry, dict):
                        agent_entries.append(dict(entry))

        if isinstance(base_store, dict):
            for agent, entries in base_store.items():
                if isinstance(entries, list):
                    merged[agent] = [dict(entry) for entry in entries if isinstance(entry, dict)]

        for store in manual_stores.get(None, []):
            if isinstance(store, dict):
                _merge_store(store)
        if step is not None:
            for store in manual_stores.get(step, []):
                if isinstance(store, dict):
                    _merge_store(store)

        def _merge_entries(entry_list: List[Dict[str, Any]]) -> None:
            for entry in entry_list:
                if not isinstance(entry, dict):
                    continue
                agent_name = str(entry.get("agent") or "*").strip() or "*"
                merged.setdefault(agent_name, []).append(dict(entry))

        _merge_entries(manual_entries.get(None, []))
        if step is not None:
            _merge_entries(manual_entries.get(step, []))

        return merged

    def _load_manual_knowledge_store(
        self, path: Union[str, Path]
    ) -> Dict[str, List[Dict[str, Any]]]:
        try:
            store = load_knowledge(path)
            print(
                colorize(
                    f"[seimei] Manual knowledge loaded from {path}",
                    KNOWLEDGE_COLOR,
                    enable=supports_color(),
                )
            )
            return store
        except FileNotFoundError as exc:
            print(
                colorize(f"[seimei] Manual knowledge file not found: {exc}", ERROR_COLOR, enable=supports_color()),
                file=sys.stderr,
            )
        except Exception as exc:  # pragma: no cover - auxiliary loads best effort
            print(
                colorize(f"[seimei] Failed to load manual knowledge from {path}: {exc}", ERROR_COLOR, enable=supports_color()),
                file=sys.stderr,
            )
        return {}

    @staticmethod
    def _coerce_path_string(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            return str(Path(value).expanduser())
        except Exception:
            return None

    # -------------------------- Shared search --------------------------

    def _make_search_fn(self, run_llm: LLMClient) -> Callable[..., Any]:
        async def _search(
            query: str,
            keys: Sequence[Dict[str, Any]],
            *,
            k: int = 1,
            context: Optional[Dict[str, Any]] = None,
        ) -> List[Dict[str, Any]]:
            return await self._search_with_backends(
                query=query,
                keys=keys,
                k=k,
                run_llm=run_llm,
                context=context or {},
            )

        return _search

    @staticmethod
    def _normalize_purpose(raw_purpose: Optional[str]) -> str:
        if not raw_purpose:
            return "knowledge_search"
        value = str(raw_purpose).strip().lower()
        if value == "agent_routing":
            return "agent_routing"
        return "knowledge_search"

    def _should_use_rmsearch(self, purpose: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
        config = cfg or self.rm_kwargs
        url = str(config.get("url") or "").strip()
        if not url:
            return False
        if purpose == "agent_routing":
            return bool(config.get("agent_routing", False))
        return bool(config.get("knowledge_search", True))

    def _rmsearch(
        self,
        *,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        k_key: Optional[int] = None,
        k: Optional[int] = None,
        purpose: Optional[str] = None,
        timeout: Optional[float] = None,
        **overrides: Any,
    ) -> List[Dict[str, Any]]:
        config = dict(self.rm_kwargs)
        config.update(overrides)
        effective_purpose = self._normalize_purpose(purpose or config.get("purpose"))

        if not self._should_use_rmsearch(effective_purpose, config):
            return []

        url = str(config.get("url") or "").strip()
        if not url:
            if not self._rm_warned_missing_url:
                print(colorize("[seimei] rmsearch skipped: rm_kwargs['url'] not set.", ERROR_COLOR), file=sys.stderr)
                self._rm_warned_missing_url = True
            return []

        limit_raw = k_key if k_key is not None else k if k is not None else config.get("k")
        try:
            limit = max(int(limit_raw), 1)
        except (TypeError, ValueError):
            limit = 1

        final_timeout = timeout if timeout is not None else config.get("timeout")

        #print()
        #print("--- rmsearch 1 ---")
        #print("query: ", query)

        try:
            return self._rmsearch_http(
                url=url,
                query=query,
                keys=keys,
                limit=limit,
                purpose=effective_purpose,
                timeout=final_timeout,
            )
        except Exception as exc:
            print(colorize(f"[seimei] rmsearch request failed: {exc}", ERROR_COLOR), file=sys.stderr)
            return []

    def _rmsearch_http(
        self,
        *,
        url: str,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        limit: int,
        purpose: str,
        timeout: Optional[float],
    ) -> List[Dict[str, Any]]:
        serialized_query = self._format_rmsearch_query(query)
        key_payload, index_map, text_map = self._format_rmsearch_keys(keys)
        if not key_payload:
            return []
        
        #print()
        #print("--- rmsearch 2 ---")
        #print("serialized_query: ", serialized_query)

        payload: Dict[str, Any] = {
            "queries": [serialized_query],
            "keys": key_payload,
            "k": limit,
        }
        if purpose:
            payload["purpose"] = purpose

        api_key = os.getenv("KYOTOAI_API_KEY")
        if not api_key:
            raise RuntimeError("KYOTOAI_API_KEY environment variable is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=timeout or 10)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from RMSearch: {exc}") from exc

        return self._parse_rmsearch_response(
            data=data,
            limit=limit,
            index_map=index_map,
            text_map=text_map,
        )

    @staticmethod
    def _format_rmsearch_query(query: Union[str, Sequence[Dict[str, Any]]]) -> Union[str, Dict[str, Any]]:
        if isinstance(query, str):
            stripped = query.strip()
            return stripped or query
        if isinstance(query, Sequence):
            messages = [dict(m) for m in query if isinstance(m, dict)]
            if messages:
                return {"message": messages}
        return str(query)

    @staticmethod
    def _format_rmsearch_keys(
        keys: Sequence[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        payload: List[str] = []
        index_map: Dict[int, Dict[str, Any]] = {}
        text_map: Dict[str, Dict[str, Any]] = {}
        for item in keys:
            if not isinstance(item, dict):
                continue
            key_text = str(item.get("key") or "").strip()
            if not key_text:
                continue
            index_map[len(payload)] = item
            payload.append(key_text)
            text_map.setdefault(key_text, item)
        return payload, index_map, text_map

    @staticmethod
    def _parse_rmsearch_response(
        *,
        data: Any,
        limit: int,
        index_map: Dict[int, Dict[str, Any]],
        text_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        entries: Sequence[Any]
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("results") or data.get("queries") or []
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                entries = []
        else:
            return []

        for entry in entries:
            keys = []
            if isinstance(entry, dict):
                keys = entry.get("keys") or []
            if not isinstance(keys, Sequence):
                continue
            for item in keys:
                if not isinstance(item, dict):
                    continue
                key_idx = item.get("key_id")
                key_text = item.get("key")
                payload = None
                if isinstance(key_idx, int) and key_idx in index_map:
                    payload = index_map[key_idx]
                elif isinstance(key_text, str) and key_text in text_map:
                    payload = text_map[key_text]
                if not payload:
                    continue
                result = {
                    "key": payload.get("key"),
                    "payload": payload,
                    "score": item.get("relevance"),
                    "source": "rmsearch",
                }
                if "reason" in item:
                    result["reason"] = item["reason"]
                records.append(result)
                if len(records) >= limit:
                    return records
        return records[:limit]

    async def _search_with_backends(
        self,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        *,
        k: int,
        run_llm: Optional[LLMClient],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not keys:
            return []

        limit = max(int(k), 1)
        key_map = {item.get("key"): item for item in keys if isinstance(item, dict) and item.get("key")}
        _, conversation_text, focus_text = self._prepare_query_input(query)
        rm_query = focus_text or conversation_text
        if isinstance(query, str):
            stripped_query = query.strip()
            if stripped_query:
                rm_query = stripped_query

        purpose = self._normalize_purpose((context or {}).get("purpose"))
        query_override = None
        if isinstance(context, dict):
            for key in ("query_override", "knowledge_query"):
                value = context.get(key)
                if isinstance(value, str) and value.strip():
                    query_override = value.strip()
                    break
        if query_override:
            rm_query = query_override
        try:
            rm_result = self._rmsearch(
                query=rm_query,
                keys=list(keys),
                k_key=limit,
                purpose=purpose,
            )
            if asyncio.iscoroutine(rm_result):
                rm_result = await rm_result
            results = self._attach_payloads(rm_result or [], key_map)
            if results:
                return results[:limit]
        except Exception as exc:
            print(colorize(f"[seimei] rmsearch selection failed: {exc}", ERROR_COLOR), file=sys.stderr)

        return await self._llm_route_search(
            query=query,
            keys=keys,
            k=limit,
            run_llm=run_llm or self.llm,
            context=context or {},
        )

    @staticmethod
    def _attach_payloads(results: Sequence[Dict[str, Any]], key_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            enriched.append({**item, "payload": key_map.get(key)})
        return enriched

    async def _llm_route_search(
        self,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        *,
        k: int,
        run_llm: LLMClient,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        candidates = [item for item in keys if isinstance(item, dict) and item.get("key")]
        if not candidates:
            return []

        context = context or {}
        reason_hint = context.get("reason_hint", "")
        numbered = "\n".join(f"{idx}. {item['key']}" for idx, item in enumerate(candidates, 1))
        history_messages, _, focus_text = self._prepare_query_input(query)
        conversation_messages = self._convert_history_to_llm(history_messages)
        system_prompt = (
            "You rank candidate keys for relevance according to the recent conversation among the user, assistants, and tools. "
            "Return a JSON array, each element containing: "
            '{"index": <1-based index of the candidate>, "score": optional float between 0 and 1, "reason": short string}. '
            "Only return up to the requested number of entries. Respond with JSON only.\n\n"
            f"Candidates:\n{numbered}\n"
            f"Select up to {k} candidates most relevant to the conversation."
        )
        user_prompt = focus_text or "There is no explicit user question. Choose the candidate that best progresses the conversation."
        if reason_hint:
            user_prompt += f"\nAdditional context: {reason_hint}"

        try:
            routing_messages = conversation_messages if conversation_messages else [
                {"role": "user", "content": user_prompt}
            ]
            reply, _usage = await run_llm.chat(
                messages=routing_messages,
                system=system_prompt,
            )
        except Exception as exc:
            print(colorize(f"[seimei] LLM routing failed: {exc}", ERROR_COLOR), file=sys.stderr)
            return [
                {"key": item["key"], "payload": item, "source": "llm-fallback", "score": None}
                for item in candidates[:k]
            ]

        data = self._parse_llm_ranking(reply)
        selected: List[Dict[str, Any]] = []
        for entry in data:
            try:
                idx = int(entry.get("index", 0))
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > len(candidates):
                continue
            candidate = candidates[idx - 1]
            result = {
                "key": candidate["key"],
                "payload": candidate,
                "score": entry.get("score"),
                "reason": entry.get("reason"),
                "source": "llm-routing",
            }
            selected.append(result)
            if len(selected) >= k:
                break

        if not selected:
            return [
                {"key": item["key"], "payload": item, "source": "llm-fallback", "score": None}
                for item in candidates[:k]
            ]
        return selected

    @staticmethod
    def _parse_llm_ranking(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except Exception:
            return []

    # -------------------------- Routing --------------------------

    async def _select_next_agent(
        self,
        messages: List[Dict[str, Any]],
        search_fn: Callable[..., Any],
    ) -> Optional[Agent]:
        if not self.agents:
            return None

        keys = [
            {"key": f"{agent.name}: {agent.description}", "agent_name": agent.name}
            for agent in self.agents.values()
        ]
        try:
            query = json.dumps(messages, ensure_ascii=False)
        except Exception:
            query = str(messages)

        if search_fn:
            try:
                ranked = await search_fn(
                    query=query,
                    keys=keys,
                    k=1,
                    context={"purpose": "agent_routing"},
                )
                if ranked:
                    agent_name = ranked[0].get("payload", {}).get("agent_name")
                    if agent_name and agent_name in self.agents:
                        return self.agents[agent_name]
                    key = ranked[0].get("key", "")
                    agent_name = key.split(":", 1)[0].strip()
                    if agent_name in self.agents:
                        return self.agents[agent_name]
            except Exception as exc:
                print(colorize(f"[seimei] search-based routing failed: {exc}", ERROR_COLOR), file=sys.stderr)

        # Fallback heuristic based on the most recent user turn
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break
        if last_user is None:
            return None

        lower = last_user.get("content", "").lower()
        if "search" in lower or "web" in lower:
            for a in self.agents.values():
                if a.name.endswith("web_search") or a.name == "web_search":
                    return a
        if any(tok in lower for tok in ["bash", "shell", "terminal", "run ", "execute ", "pip ", "python "]):
            for a in self.agents.values():
                if a.name.endswith("code_act") or a.name == "code_act":
                    return a
        for pref in ("think", "default"):
            if pref in self.agents:
                return self.agents[pref]
        return next(iter(self.agents.values()), None)

    # -------------------------- Logging --------------------------

    def _make_run_dirs(self, run_id: str) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.log_dir, f"run-{ts}-{run_id[:8]}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _append_dataset(self, record: Dict[str, Any]) -> None:
        path = os.path.join(self.log_dir, "dataset.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    # -------------------------- Metadata helpers --------------------------

    @classmethod
    def _is_sensitive_metadata_key(cls, key: Optional[str]) -> bool:
        if not key:
            return False
        normalized = str(key).strip().lower().replace("-", "_")
        return normalized in cls._SENSITIVE_METADATA_KEYS

    @classmethod
    def _prepare_metadata_value(cls, value: Any, *, key_hint: Optional[str] = None) -> Any:
        if key_hint and cls._is_sensitive_metadata_key(key_hint):
            return "<redacted>"
        if isinstance(value, dict):
            prepared: Dict[str, Any] = {}
            for raw_key, raw_val in value.items():
                normalized_key = "*" if raw_key is None else str(raw_key)
                prepared[normalized_key] = cls._prepare_metadata_value(raw_val, key_hint=normalized_key)
            return prepared
        if isinstance(value, (list, tuple)):
            return [cls._prepare_metadata_value(item) for item in value]
        if isinstance(value, set):
            prepared_items = [cls._prepare_metadata_value(item) for item in value]
            try:
                return sorted(
                    prepared_items,
                    key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True)
                    if isinstance(item, (dict, list))
                    else str(item),
                )
            except TypeError:
                return prepared_items
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _summarize_role_counts(messages: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").lower()
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _build_run_metadata(
        self,
        *,
        run_id: str,
        run_dir: str,
        run_name: Optional[str],
        started_at: float,
        completed_at: float,
        usage: Dict[str, int],
        input_messages: Sequence[Dict[str, Any]],
        message_history: Sequence[Dict[str, Any]],
        system_prompt: Optional[str],
        step_sequence: Sequence[str],
        stop_reason: Optional[str],
        final_response_source: str,
        final_agent_name: Optional[str],
        token_limit_hit: bool,
        token_limit_error: Optional[TokenLimitExceeded],
        token_limiter: Optional[TokenLimiter],
        normalized_knowledge_config: Dict[str, Any],
        manual_store_map: Dict[Optional[int], List[Dict[str, Any]]],
        raw_knowledge_config: Optional[Dict[str, Any]],
        knowledge_generation_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        started_ts = float(started_at)
        completed_ts = float(completed_at)
        total_sec = max(completed_ts - started_ts, 0.0)
        run_dir_path = Path(run_dir)
        try:
            artifact_files = sorted(os.listdir(run_dir))
        except FileNotFoundError:
            artifact_files = []

        resolved_stop_reason = stop_reason or ("token_limit_hit" if token_limit_hit else "completed")
        system_text = system_prompt or ""
        system_meta: Dict[str, Any] = {
            "provided": bool(system_text),
            "length": len(system_text),
        }
        if system_text:
            system_meta["preview"] = system_text[:200]

        agent_snapshot = [
            {
                "name": name,
                "class": agent_obj.__class__.__name__,
                "module": agent_obj.__class__.__module__,
                "description": getattr(agent_obj, "description", None),
            }
            for name, agent_obj in sorted(self.agents.items())
        ]

        allowed_commands = list(self.allowed_commands) if self.allowed_commands else []
        approval_callback_name = None
        if callable(self.approval_callback):
            approval_callback_name = getattr(self.approval_callback, "__name__", repr(self.approval_callback))

        seimei_config_meta = {
            "log_dir": self.log_dir,
            "max_steps": self.max_steps,
            "agent_log_head_lines": self.agent_log_head_lines,
            "allow_code_exec": self.allow_code_exec,
            "allowed_commands": allowed_commands,
            "has_approval_callback": callable(self.approval_callback),
            "approval_callback_name": approval_callback_name,
            "agent_config": self._prepare_metadata_value(getattr(self, "agent_config_spec", [])),
            "max_tokens_per_question": self.max_tokens_per_question,
        }

        llm_meta = {
            "model": self.llm.model,
            "base_url": getattr(self.llm, "base_url", None),
            "timeout": getattr(self.llm, "timeout", None),
            "using_openai": getattr(self.llm, "using_openai", False),
            "using_generate": getattr(self.llm, "using_generate", False),
            "has_api_key": bool(getattr(self.llm, "api_key", None)),
            "extra_headers": self._prepare_metadata_value(getattr(self.llm, "extra_headers", {})),
            "kwargs": self._prepare_metadata_value(getattr(self.llm, "kwargs", {})),
        }

        token_limit_info: Dict[str, Any] = {
            "limit": getattr(token_limiter, "limit", None) if token_limiter else None,
            "consumed": getattr(token_limiter, "consumed", None) if token_limiter else None,
            "hit": token_limit_hit,
        }
        if token_limit_error:
            token_limit_info["last_error"] = {
                "limit": getattr(token_limit_error, "limit", None),
                "consumed": getattr(token_limit_error, "consumed", None),
                "last_usage": getattr(token_limit_error, "last_usage", None),
            }

        manual_entry_map = normalized_knowledge_config.get("manual_entries", {}) or {}
        manual_store_sources = normalized_knowledge_config.get("manual_store_sources", {}) or {}
        manual_entry_counts: Dict[str, int] = {}
        for step_key, values in manual_entry_map.items():
            label = "*" if step_key is None else str(step_key)
            manual_entry_counts[label] = len(values)

        manual_store_counts: Dict[str, int] = {}
        for step_key, stores in manual_store_map.items():
            total_entries = 0
            for store in stores:
                if not isinstance(store, dict):
                    continue
                for entry_list in store.values():
                    if isinstance(entry_list, list):
                        total_entries += len(entry_list)
            label = "*" if step_key is None else str(step_key)
            manual_store_counts[label] = total_entries

        knowledge_entries = 0
        knowledge_agents: List[str] = []
        if isinstance(self.knowledge_store, dict):
            for agent_name, entries in self.knowledge_store.items():
                knowledge_agents.append(str(agent_name))
                if isinstance(entries, list):
                    knowledge_entries += len(entries)

        knowledge_meta = {
            "generate_knowledge": bool(normalized_knowledge_config.get("generate_knowledge")),
            "load_path": normalized_knowledge_config.get("load_knowledge_path"),
            "load_path_provided": bool(normalized_knowledge_config.get("load_path_provided")),
            "save_path": str(normalized_knowledge_config.get("save_knowledge_path"))
            if normalized_knowledge_config.get("save_knowledge_path")
            else None,
            "knowledge_prompt_path": str(normalized_knowledge_config.get("knowledge_prompt_path"))
            if normalized_knowledge_config.get("knowledge_prompt_path")
            else None,
            "manual_entry_counts": manual_entry_counts,
            "manual_store_counts": manual_store_counts,
            "manual_entries": self._prepare_metadata_value(manual_entry_map),
            "manual_store_sources": self._prepare_metadata_value(manual_store_sources),
            "raw_config": self._prepare_metadata_value(raw_knowledge_config or {}),
            "knowledge_store": {
                "path": self.load_knowledge_path,
                "entries": knowledge_entries,
                "agents": sorted(set(knowledge_agents)),
            },
            "knowledge_generation_result": self._prepare_metadata_value(knowledge_generation_result)
            if knowledge_generation_result
            else None,
        }

        agent_steps_meta = {
            "count": len(step_sequence),
            "sequence": list(step_sequence),
        }

        final_response_meta = {
            "source": final_response_source,
            "agent_name": final_agent_name,
        }

        artifacts_meta = {
            "directory": run_dir,
            "folder": run_dir_path.name,
            "files": artifact_files,
        }

        input_summary = {
            "count": len(input_messages or []),
            "role_counts": self._summarize_role_counts(input_messages),
        }
        history_summary = {
            "count": len(message_history or []),
            "role_counts": self._summarize_role_counts(message_history),
        }

        meta: Dict[str, Any] = {
            "run_id": run_id,
            "run_name": run_name,
            "model": self.llm.model,
            "run_directory": run_dir,
            "run_folder": run_dir_path.name,
            "timings": {
                "started_at": started_ts,
                "completed_at": completed_ts,
                "total_sec": total_sec,
            },
            "usage": {k: int(v) for k, v in usage.items()},
            "max_steps": self.max_steps,
            "token_limit_hit": token_limit_hit,
            "token_limit": token_limit_info,
            "stop_reason": resolved_stop_reason,
            "final_response": final_response_meta,
            "input_messages": input_summary,
            "message_history": history_summary,
            "system_prompt": system_meta,
            "artifacts": artifacts_meta,
            "agent_steps": agent_steps_meta,
            "agents": {
                "count": len(agent_snapshot),
                "loaded": agent_snapshot,
            },
            "seimei_config": self._prepare_metadata_value(seimei_config_meta),
            "llm_config": self._prepare_metadata_value(llm_meta),
            "rmsearch_config": self._prepare_metadata_value(self.rm_kwargs),
            "knowledge": knowledge_meta,
        }
        return meta

    def _write_run_metadata(self, run_dir: str, payload: Dict[str, Any]) -> None:
        meta_path = os.path.join(run_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


    # -------------------------- Inference --------------------------

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        stop_when: Optional[Callable[[List[Dict[str, Any]]], bool]] = None,
        return_usage: bool = True,
        run_name: Optional[str] = None,
        knowledge_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Make a deep-ish copy so we can append steps
        msg_history: List[Dict[str, Any]] = []
        for raw in messages:
            msg = dict(raw)
            k_texts, k_ids = self._extract_message_knowledge(msg)
            k_value = self._collapse_knowledge_field(k_texts)
            k_id_value = None
            if any(kid is not None for kid in k_ids):
                k_id_value = self._collapse_knowledge_field(k_ids, keep_none=True)
            if k_value is not None:
                msg["knowledge"] = k_value
            if k_id_value is not None:
                msg["knowledge_id"] = k_id_value
            msg_history.append(msg)
        run_label = (run_name or "").strip()
        log_prefix = f"[seimei {run_label}]" if run_label else "[seimei]"
        color_enabled = supports_color()

        normalized_config = self._normalize_knowledge_config(knowledge_config)
        manual_entry_map = normalized_config["manual_entries"]
        manual_store_map = normalized_config["manual_stores"]
        config_load_path = normalized_config["load_knowledge_path"]
        if normalized_config["load_path_provided"]:
            self.load_knowledge_path = config_load_path
            if config_load_path:
                self.knowledge_store = self._load_knowledge_store(config_load_path)
            else:
                self.knowledge_store = {}
            self.shared_ctx["knowledge"] = self.knowledge_store

        run_id = str(uuid.uuid4())
        run_dir = self._make_run_dirs(run_id)
        steps_path = os.path.join(run_dir, "steps.jsonl")
        t0 = time.time()

        def write_step(step: Dict[str, Any]) -> None:
            with open(steps_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")

        def log_step_blocks(
            blocks: Dict[str, Optional[str]],
            *,
            agent_name: Optional[str] = None,
        ) -> None:
            printed_any = False
            normalized_agent = (agent_name or "").lower()
            is_answer_agent = normalized_agent.endswith("answer")
            block_color = ANSWER_BLOCK_COLOR if is_answer_agent else LOG_BLOCK_COLOR
            for label, value in blocks.items():
                if value is None:
                    continue
                text = str(value).strip("\n")
                lines = text.splitlines() if text else [""]
                print(colorize(f"    <{label}>", block_color, enable=color_enabled))
                for line in lines:
                    print(colorize(f"        {line}", block_color, enable=color_enabled))
                printed_any = True
            if printed_any:
                print()

        usage_agg: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        token_limiter: Optional[TokenLimiter] = None
        run_generate_knowledge: bool = normalized_config["generate_knowledge"]
        run_save_path = normalized_config["save_knowledge_path"]
        run_prompt_path = normalized_config["knowledge_prompt_path"]
        run_shared_ctx = dict(self.shared_ctx)

        def _update_run_knowledge(step: Optional[int]) -> None:
            run_shared_ctx["knowledge"] = self._compose_step_knowledge(
                base_store=self.knowledge_store,
                manual_entries=manual_entry_map,
                manual_stores=manual_store_map,
                step=step,
            )

        _update_run_knowledge(None)
        if self.max_tokens_per_question and self.max_tokens_per_question > 0:
            token_limiter = TokenLimiter(self.max_tokens_per_question)
            run_shared_ctx["token_limiter"] = token_limiter
            run_llm = self.llm.bind_token_limiter(token_limiter)
        else:
            run_llm = self.llm
        run_shared_ctx["llm"] = run_llm
        search_fn = self._make_search_fn(run_llm)
        run_shared_ctx["search"] = search_fn
        self._update_shared_knowledge_query(run_shared_ctx, msg_history)

        token_limit_hit = False
        token_limit_error: Optional[TokenLimitExceeded] = None
        final_agent_output: Optional[str] = None
        final_agent_name: Optional[str] = None
        knowledge_generation_result: Optional[Dict[str, Any]] = None
        resolved_prompt_path: Optional[Path] = None
        if run_prompt_path is not None:
            resolved_prompt_path = run_prompt_path
        executed_steps = 0
        step_agent_sequence: List[str] = []
        stop_reason: Optional[str] = None
        final_response_source = "pending"

        # Agent loop (very simple – customize as needed)
        step_idx = 0
        while step_idx < self.max_steps:
            step_idx += 1

            run_shared_ctx["current_step"] = step_idx
            _update_run_knowledge(step_idx)
            self._update_shared_knowledge_query(run_shared_ctx, msg_history)

            # Decide which agent to run
            agent_obj = await self._select_next_agent(msg_history, search_fn)
            if agent_obj is None:
                if stop_reason is None:
                    stop_reason = "no_agent_available"
                break

            step_label = f"{log_prefix} Step {step_idx}"
            print(colorize(f"{step_label}: Running {agent_obj.name} agent", STEP_TITLE_COLOR, enable=color_enabled))

            try:
                step_res = await agent_obj(
                    messages=msg_history,
                    shared_ctx=run_shared_ctx,
                )
            except TokenLimitExceeded as err:
                token_limit_hit = True
                token_limit_error = err
                print(
                    colorize(
                        f"{log_prefix} !! Token limit exceeded by agent {agent_obj.name}: "
                        f"limit={err.limit}, consumed={err.consumed}",
                        ERROR_COLOR,
                        enable=color_enabled,
                    )
                )
                step_res = {
                    "content": f"[token_limit] Token limit {err.limit} exceeded with {err.consumed} tokens.",
                    "stop": True,
                    "log": {"limit": err.limit, "consumed": err.consumed, "last_usage": err.last_usage},
                }
            except Exception as e:
                step_res = {"content": f"[agent_error] {type(e).__name__}: {e}"}

            step_res = self._apply_agent_output_limit(step_res)
            executed_steps += 1
            step_agent_sequence.append(agent_obj.name)

            knowledge_texts, knowledge_ids = self._extract_message_knowledge(step_res)
            knowledge_value = self._collapse_knowledge_field(knowledge_texts)
            knowledge_id_value = None
            if any(kid is not None for kid in knowledge_ids):
                knowledge_id_value = self._collapse_knowledge_field(knowledge_ids, keep_none=True)
            if knowledge_value is not None and "knowledge" not in step_res:
                step_res["knowledge"] = knowledge_value
            if knowledge_id_value is not None and "knowledge_id" not in step_res:
                step_res["knowledge_id"] = knowledge_id_value

            if step_res.get("final_output"):
                final_agent_output = str(step_res.get("final_output"))
                final_agent_name = agent_obj.name

            # Append to history
            agent_msg = {
                "role": "agent",
                "name": agent_obj.name,
                "content": step_res.get("content", ""),
            }
            if knowledge_value is not None or step_res.get("knowledge") is not None:
                agent_msg["knowledge"] = step_res.get("knowledge", knowledge_value)
            if knowledge_id_value is not None or step_res.get("knowledge_id") is not None:
                agent_msg["knowledge_id"] = step_res.get("knowledge_id", knowledge_id_value)
            # Optional fields for richer dataset
            for k in ("code", "chosen_instructions", "log"):
                if k in step_res:
                    agent_msg[k] = step_res[k]
            msg_history.append(agent_msg)
            self._update_shared_knowledge_query(run_shared_ctx, msg_history)

            # Persist step
            step_payload = {
                "step": step_idx,
                "agent": agent_obj.name,
                "result": step_res,
                "time": time.time(),
            }
            if "knowledge" in step_res:
                step_payload["knowledge"] = step_res["knowledge"]
            if "knowledge_id" in step_res:
                step_payload["knowledge_id"] = step_res["knowledge_id"]
            write_step(step_payload)

            print(colorize(f"{step_label}: Done {agent_obj.name} agent", STEP_TITLE_COLOR, enable=color_enabled))

            content = step_res.get("content", "")
            log_data = step_res.get("log", None)
            blocks: Dict[str, Optional[str]] = {}

            if log_data:
                blocks.update(log_data)

            else:
                output_text: Optional[str] = None
                if self.agent_log_head_lines:
                    if content:
                        preview = content[:1000]
                        if len(content) > 1000:
                            preview = preview.rstrip() + "..."
                        output_text = preview
                    else:
                        output_text = "[no content]"

                if output_text:
                    blocks["output"] = output_text

            log_step_blocks(blocks, agent_name=agent_obj.name)

            # Optional termination predicate
            if stop_when and stop_when(msg_history):
                if stop_reason is None:
                    stop_reason = "stop_when"
                break

            # Stop early if agent says so
            if step_res.get("stop", False):
                if stop_reason is None:
                    stop_reason = "agent_requested_stop"
                break

            if token_limit_hit:
                if stop_reason is None:
                    stop_reason = "token_limit_hit"
                break

        if stop_reason is None:
            if executed_steps >= self.max_steps:
                stop_reason = "max_steps_reached"
            elif executed_steps == 0:
                stop_reason = "no_agent_executed"
            else:
                stop_reason = "completed"

        # Final assistant call
        if not token_limit_hit:
            if final_agent_output is not None:
                answer = final_agent_output
                usage = {}
                msg_history.append({"role": "assistant", "content": answer})
                final_response_source = "agent"
                if final_agent_name:
                    print(
                        colorize(
                            f"{log_prefix} Final output provided by {final_agent_name} agent",
                            STEP_TITLE_COLOR,
                            enable=color_enabled,
                        )
                    )
            else:
                try:
                    answer, usage = await run_llm.chat(messages=msg_history, system=system)
                    final_response_source = "llm"
                except TokenLimitExceeded as err:
                    token_limit_hit = True
                    token_limit_error = err
                    final_response_source = "token_limit"
                    if stop_reason is None:
                        stop_reason = "token_limit_hit"
                    print(
                        colorize(
                            f"{log_prefix} !! Token limit exceeded during final response: "
                            f"limit={err.limit}, consumed={err.consumed}",
                            ERROR_COLOR,
                            enable=color_enabled,
                        )
                    )
                    answer = f"[token_limit] Token limit {err.limit} exceeded with {err.consumed} tokens."
                    usage = {}
                msg_history.append({"role": "assistant", "content": answer})
                if usage:
                    for k, v in usage.items():
                        usage_agg[k] = usage_agg.get(k, 0) + int(v)
        else:
            answer = (
                msg_history[-1].get("content", "")
                if msg_history and msg_history[-1].get("role") == "agent"
                else "[token_limit] Token limit exceeded before final response."
            )
            msg_history.append({"role": "assistant", "content": answer})
            usage = {}
            final_response_source = "token_limit"
            if stop_reason is None:
                stop_reason = "token_limit_hit"

        # Log final assistant output without truncation
        # -> Now answer agent is doing this.
        '''
        final_text = answer if answer else "[no content]"
        if "\n" in final_text:
            print(f"{log_prefix} Final Output:")
            for line in final_text.splitlines():
                print(f"    {line}")
        else:
            print(f"{log_prefix} Final Output: {final_text}")
        print()
        '''

        run_completed_at = time.time()

        # Save run artifacts
        with open(os.path.join(run_dir, "messages.json"), "w", encoding="utf-8") as f:
            json.dump(msg_history, f, ensure_ascii=False, indent=2)

        with open(os.path.join(run_dir, "output.txt"), "w", encoding="utf-8") as f:
            f.write(answer)
        def persist_meta() -> None:
            payload = self._build_run_metadata(
                run_id=run_id,
                run_dir=run_dir,
                run_name=run_name,
                started_at=t0,
                completed_at=run_completed_at,
                usage=usage_agg,
                input_messages=messages,
                message_history=msg_history,
                system_prompt=system,
                step_sequence=step_agent_sequence,
                stop_reason=stop_reason,
                final_response_source=final_response_source,
                final_agent_name=final_agent_name,
                token_limit_hit=token_limit_hit,
                token_limit_error=token_limit_error,
                token_limiter=token_limiter,
                normalized_knowledge_config=normalized_config,
                manual_store_map=manual_store_map,
                raw_knowledge_config=knowledge_config,
                knowledge_generation_result=knowledge_generation_result,
            )
            self._write_run_metadata(run_dir, payload)

        persist_meta()

        # Append dataset record
        dataset_record = {
            "schema_version": 1,
            "run_id": run_id,
            "input_messages": messages,
            "final_output": answer,
            "steps": self._read_jsonl(steps_path),
            "model_info": {"name": self.llm.model, "kwargs": self.llm.kwargs},
            "timestamps": {"started": t0, "ended": time.time()},
            "token_limit": {
                "limit": getattr(token_limiter, "limit", None),
                "consumed": getattr(token_limiter, "consumed", None),
                "hit": token_limit_hit,
                "last_error": {
                    "limit": getattr(token_limit_error, "limit", None),
                    "consumed": getattr(token_limit_error, "consumed", None),
                    "last_usage": getattr(token_limit_error, "last_usage", None),
                } if token_limit_error else None,
            },
        }
        if run_name is not None:
            dataset_record["run_name"] = run_name
        self._append_dataset(dataset_record)

        if run_generate_knowledge:
            target_override = str(run_save_path) if run_save_path else None
            target_path = self._resolve_knowledge_output_path(target_override)
            print(
                colorize(
                    f"{log_prefix} Knowledge generation: starting (target={target_path})",
                    KNOWLEDGE_COLOR,
                    enable=color_enabled,
                )
            )
            knowledge_start = time.time()
            try:
                knowledge_generation_result = await generate_knowledge_from_runs(
                    run_ids=[Path(run_dir).name],
                    save_file_path=target_path,
                    runs_dir=Path(self.log_dir),
                    prompt_path=resolved_prompt_path or DEFAULT_RUN_PROMPT,
                    model=self.llm.model,
                    base_url=self.llm.base_url,
                    api_key=self.llm.api_key,
                )
                knowledge_generation_result["duration_sec"] = time.time() - knowledge_start
                knowledge_generation_result["target_path"] = str(target_path)
                knowledge_generation_result["status"] = "completed"
                self._refresh_knowledge_store(target_path)
                added = knowledge_generation_result.get("count")
                added_display = added if added is not None else 0
                print(
                    colorize(
                        f"{log_prefix} Knowledge generation: completed "
                        f"(entries={added_display}, saved_to={target_path})",
                        KNOWLEDGE_COLOR,
                        enable=color_enabled,
                    )
                )
            except Exception as exc:  # pragma: no cover - knowledge generation best effort
                knowledge_generation_result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "runs": [Path(run_dir).name],
                    "target_path": str(target_path),
                    "duration_sec": time.time() - knowledge_start,
                    "status": "failed",
                }
                print(
                    colorize(
                        f"[seimei] Failed to auto-generate knowledge for run {run_id}: {exc}",
                        ERROR_COLOR,
                        enable=color_enabled,
                    ),
                    file=sys.stderr,
                )

        persist_meta()

        out: Dict[str, Any] = {"run_id": run_id, "output": answer, "msg_history": msg_history}
        if knowledge_generation_result:
            out["knowledge_result"] = knowledge_generation_result
        if return_usage:
            out["usage"] = usage_agg
        if run_name:
            out["run_name"] = run_name
        return out

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows

    @staticmethod
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

    @staticmethod
    def _prepare_query_input(
        query: Union[str, Sequence[Dict[str, Any]]],
        *,
        max_messages: int = 8,
        max_chars_per_message: int = 400,
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        messages = seimei._coerce_messages(query)

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

    @staticmethod
    def _convert_history_to_llm(
        messages: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prepared, _ = llm_module.prepare_messages(messages, drop_normal_system=False)
        return prepared

    def _apply_agent_output_limit(self, step_res: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(step_res, dict):
            return step_res
        limit = self.AGENT_OUTPUT_LIMIT
        content = step_res.get("content")
        if isinstance(content, str):
            clipped, truncated = self._truncate_text(content, limit)
            if truncated:
                step_res["content"] = clipped
        log_data = step_res.get("log")
        if isinstance(log_data, dict):
            for key, value in list(log_data.items()):
                if isinstance(value, str):
                    clipped, truncated = self._truncate_text(value, limit)
                    if truncated:
                        log_data[key] = clipped
        return step_res

    @staticmethod
    def _truncate_text(text: str, limit: int) -> Tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        marker = "[CONTENT OMITTED]"
        if limit <= len(marker):
            return marker[:limit], True
        prefix = text[: limit - len(marker)].rstrip()
        return f"{prefix}{marker}", True

    def _update_shared_knowledge_query(self, shared_ctx: Dict[str, Any], messages: Sequence[Dict[str, Any]]) -> None:
        if not isinstance(shared_ctx, dict):
            return
        query_block = self._build_knowledge_query_block(messages)
        if query_block:
            shared_ctx["knowledge_query"] = query_block
        else:
            shared_ctx.pop("knowledge_query", None)

    def _build_knowledge_query_block(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        max_agent_steps: int = 4,
        max_agent_chars: int = 800,
        max_knowledge_per_step: int = 3,
        max_knowledge_chars: int = 200,
    ) -> str:
        normalized = seimei._coerce_messages(messages)
        _, conversation_text, focus_text = self._prepare_query_input(normalized)
        user_query = (focus_text or conversation_text or "").strip()

        agent_entries: List[Dict[str, Any]] = []
        agent_idx = 0
        for msg in normalized:
            if str(msg.get("role") or "").lower() != "agent":
                continue
            agent_idx += 1
            label = f"step{agent_idx}"
            content = self._shorten_for_query(self._stringify_for_query(msg.get("content", "")), max_agent_chars)
            knowledge_cues = self._extract_knowledge_cues(msg, max_knowledge_per_step, max_knowledge_chars)
            agent_entries.append({"label": label, "content": content, "knowledge": knowledge_cues})
        recent_entries = agent_entries[-max(int(max_agent_steps), 1):]

        sections: List[str] = []
        sections.append(f"User query:\n{user_query or '[missing user query]'}")

        findings_lines: List[str] = []
        for entry in recent_entries:
            content = entry.get("content") or ""
            if not content:
                continue
            findings_lines.append(entry["label"])
            findings_lines.append(content)
            findings_lines.append("")
        findings_block = "\n".join(findings_lines).strip()
        if findings_block:
            sections.append("Recent agent findings:\n" + findings_block)

        knowledge_lines: List[str] = []
        for entry in recent_entries:
            cues: List[str] = entry.get("knowledge") or []
            if not cues:
                continue
            knowledge_lines.append(entry["label"])
            for cue in cues:
                knowledge_lines.append(f"- {cue}")
            knowledge_lines.append("")
        knowledge_block = "\n".join(knowledge_lines).strip()
        if knowledge_block:
            sections.append("Selected knowledge cues:\n" + knowledge_block)

        return "\n\n".join(section for section in sections if section).strip()

    def _extract_knowledge_cues(
        self,
        message: Dict[str, Any],
        limit: int,
        max_chars: int,
    ) -> List[str]:
        cues: List[str] = []
        sources: List[Any] = []
        log_data = message.get("log")
        if isinstance(log_data, dict):
            for key in ("knowledge", "knowledge_used", "chosen_knowledge"):
                raw = log_data.get(key)
                if raw is not None:
                    sources.append(raw)
        for key in ("chosen_knowledge",):
            raw = message.get(key)
            if raw is not None:
                sources.append(raw)

        for raw in sources:
            for text in self._coerce_knowledge_texts(raw, max_chars):
                if text:
                    cues.append(text)
                if len(cues) >= limit:
                    return cues[:limit]
        return cues[:limit]

    @staticmethod
    def _coerce_knowledge_texts(value: Any, max_chars: int) -> List[str]:
        texts: List[str] = []
        if value is None:
            return texts
        if isinstance(value, str):
            snippet = seimei._shorten_for_query(value, max_chars)
            if snippet:
                texts.append(snippet)
            return texts
        if isinstance(value, dict):
            snippet = seimei._shorten_for_query(seimei._stringify_knowledge_dict(value), max_chars)
            if snippet:
                texts.append(snippet)
            return texts
        if isinstance(value, (list, tuple, set)):
            for item in value:
                texts.extend(seimei._coerce_knowledge_texts(item, max_chars))
            return texts
        snippet = seimei._shorten_for_query(str(value), max_chars)
        if snippet:
            texts.append(snippet)
        return texts

    @staticmethod
    def _stringify_for_query(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    @staticmethod
    def _shorten_for_query(text: str, limit: int) -> str:
        snippet = (text or "").strip()
        if not snippet:
            return ""
        if limit > 0 and len(snippet) > limit:
            return snippet[:limit].rstrip() + "..."
        return snippet

    @staticmethod
    def _stringify_knowledge_dict(data: Dict[str, Any]) -> str:
        for key in ("text", "knowledge", "key", "content", "summary"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return json.dumps(data, ensure_ascii=False)
        except TypeError:
            return str(data)

    @staticmethod
    def _coerce_knowledge_items_with_ids(value: Any) -> List[Tuple[str, Any]]:
        items: List[Tuple[str, Any]] = []
        if value is None:
            return items
        if isinstance(value, (int, float, bool)):
            return items
        if isinstance(value, str):
            text = value.strip()
            if text:
                items.append((text, None))
            return items
        if isinstance(value, dict):
            kid = value.get("knowledge_id") or value.get("id")
            text_candidate: Optional[str] = None
            for key in ("knowledge", "text", "content", "key", "value"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    text_candidate = candidate.strip()
                    break
                if isinstance(candidate, dict):
                    nested_text = seimei._stringify_knowledge_dict(candidate)
                    nested_id = candidate.get("id") or candidate.get("knowledge_id")
                    if nested_text and not text_candidate:
                        text_candidate = nested_text
                    if nested_id is not None and kid is None:
                        kid = nested_id
            if text_candidate is None:
                text_candidate = seimei._stringify_knowledge_dict(value)
            if text_candidate:
                items.append((text_candidate, kid))
            for alt_key in ("knowledge", "knowledge_used", "chosen_knowledge"):
                alt_val = value.get(alt_key)
                if alt_val is not None:
                    items.extend(seimei._coerce_knowledge_items_with_ids(alt_val))
            return items
        if isinstance(value, (list, tuple, set)):
            for entry in value:
                items.extend(seimei._coerce_knowledge_items_with_ids(entry))
            return items
        try:
            text = str(value).strip()
        except Exception:
            return items
        if text:
            items.append((text, None))
        return items

    @staticmethod
    def _pair_knowledge_and_ids(knowledge_value: Any, ids_value: Any) -> List[Tuple[str, Any]]:
        if knowledge_value is None or ids_value is None:
            return []
        knowledge_items: List[str] = []
        if isinstance(knowledge_value, (list, tuple, set)):
            for entry in knowledge_value:
                if isinstance(entry, str) and entry.strip():
                    knowledge_items.append(entry.strip())
        elif isinstance(knowledge_value, str) and knowledge_value.strip():
            knowledge_items.append(knowledge_value.strip())
        if not knowledge_items:
            return []

        if isinstance(ids_value, (list, tuple, set)):
            ids_list = list(ids_value)
        else:
            ids_list = [ids_value]
        if not ids_list:
            return []

        if len(ids_list) == len(knowledge_items):
            return list(zip(knowledge_items, ids_list))
        if len(ids_list) == 1:
            return [(text, ids_list[0]) for text in knowledge_items]
        return []

    @staticmethod
    def _dedupe_knowledge_items(items: Sequence[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        deduped: List[Tuple[str, Any]] = []
        seen: set = set()
        for text, kid in items:
            if not text:
                continue
            key = (text, kid)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((text, kid))
        return deduped

    def _extract_message_knowledge(self, payload: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        if not isinstance(payload, dict):
            return [], []
        sources: List[Any] = []
        for key in ("knowledge", "chosen_knowledge", "knowledge_used"):
            if key in payload:
                sources.append(payload.get(key))
        log_data = payload.get("log")
        if isinstance(log_data, dict):
            for key in ("knowledge", "knowledge_used", "chosen_knowledge"):
                if key in log_data:
                    sources.append(log_data.get(key))
        items: List[Tuple[str, Any]] = []
        for src in sources:
            items.extend(self._coerce_knowledge_items_with_ids(src))
        if "knowledge" in payload or "knowledge_id" in payload:
            items.extend(self._pair_knowledge_and_ids(payload.get("knowledge"), payload.get("knowledge_id")))
        if isinstance(log_data, dict) and ("knowledge" in log_data or "knowledge_id" in log_data):
            items.extend(self._pair_knowledge_and_ids(log_data.get("knowledge"), log_data.get("knowledge_id")))
        deduped = self._dedupe_knowledge_items(items)
        texts = [text for text, _ in deduped]
        ids = [kid for _, kid in deduped]
        return texts, ids

    @staticmethod
    def _collapse_knowledge_field(values: Sequence[Any], *, keep_none: bool = False) -> Optional[Union[Any, List[Any]]]:
        filtered = list(values or [])
        if not filtered:
            return None
        if not keep_none:
            filtered = [v for v in filtered if v is not None]
            if not filtered:
                return None
        if len(filtered) == 1:
            return filtered[0]
        return filtered
