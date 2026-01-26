from __future__ import annotations

import asyncio
import importlib.util
import inspect
import io
import json
import os
import random
import re
import sys
import time
import types
import uuid
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

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
from .utils import format_query_for_rmsearch, format_key_for_rmsearch

STEP_TITLE_COLOR = LogColors.GREEN
LOG_BLOCK_COLOR = LogColors.CYAN
ANSWER_BLOCK_COLOR = LogColors.BOLD_MAGENTA
ERROR_COLOR = LogColors.RED
WARNING_COLOR = LogColors.YELLOW
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
        llm_config: Optional[Dict[str, Any]] = None,
        rm_config: Optional[Dict[str, Any]] = None,
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
        if llm_config is None:
            raise ValueError("llm_config must be provided")
        self.llm = LLMClient(**llm_config)

        # Routing
        default_rm_settings = {
            "base_url": DEFAULT_RMSEARCH_URL,
        }
        provided_rm_config = dict(rm_config or {})
        if "base_url" not in provided_rm_config and "url" in provided_rm_config:
            provided_rm_config["base_url"] = provided_rm_config.pop("url")
        self.rm_config = {**default_rm_settings, **provided_rm_config}
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
        (
            self._designated_agent_files,
            self._designated_agent_names,
        ) = self._resolve_agent_config_targets(self.agent_config_spec)
        self._restrict_agents_to_config = bool(self.agent_config_spec)
        self._load_agents(self.agent_config_spec)

        # Attach shared ctx visible to agents (e.g., llm, rmsearch, safety flags)
        self.shared_ctx = {
            "llm": self.llm,
            "rm_config": self.rm_config,
            "rmsearch_fn": self._rmsearch,
            "allow_code_exec": self.allow_code_exec,
            "allowed_commands": self.allowed_commands,
            "approval_callback": self.approval_callback,
            "search": None,
            "knowledge": self.knowledge_store,
        }

    # -------------------------- Agent loading --------------------------

    def _load_agents(self, configs: Sequence[Dict[str, Any]]) -> None:
        requested_names: List[str] = []
        seen_names: Set[str] = set()
        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            name = self._normalize_agent_name(cfg.get("name"))
            if name and name not in seen_names:
                requested_names.append(name)
                seen_names.add(name)
            dir_path = cfg.get("dir_path")
            file_path = cfg.get("file_path")
            if dir_path:
                self._load_agents_from_dir(dir_path)
            if file_path:
                self._load_agents_from_file(file_path)
        if requested_names:
            self._load_agents_from_names(requested_names)

        # instantiate
        for cls in get_agent_subclasses().values():
            if not self._should_include_agent_class(cls):
                continue
            try:
                inst = cls()
                self.agents[inst.name] = inst
            except Exception as e:
                print(
                    colorize(f"[seimei] Failed to instantiate agent {cls}: {e}", ERROR_COLOR),
                    file=sys.stderr,
                )

    def _resolve_agent_config_targets(
        self, configs: Sequence[Dict[str, Any]]
    ) -> Tuple[Set[Path], Set[str]]:
        resolved_paths: Set[Path] = set()
        agent_names: Set[str] = set()

        def _safe_resolve(value: Any) -> Optional[Path]:
            if value in (None, ""):
                return None
            try:
                return Path(value).expanduser().resolve()
            except Exception:
                try:
                    expanded = os.path.expanduser(str(value))
                except Exception:
                    return None
                return Path(os.path.abspath(expanded))

        for cfg in configs:
            if not isinstance(cfg, dict):
                continue
            name = self._normalize_agent_name(cfg.get("name"))
            if name:
                agent_names.add(name)
            file_path = _safe_resolve(cfg.get("file_path"))
            if file_path and file_path.is_file():
                resolved_paths.add(file_path)

            dir_path = _safe_resolve(cfg.get("dir_path"))
            if dir_path and dir_path.is_dir():
                try:
                    for entry in dir_path.iterdir():
                        if not entry.is_file():
                            continue
                        name = entry.name
                        if not name.endswith(".py") or name.startswith("_"):
                            continue
                        try:
                            resolved_paths.add(entry.resolve())
                        except Exception:
                            resolved_paths.add(entry)
                except OSError:
                    continue

        return resolved_paths, agent_names

    @staticmethod
    def _normalize_agent_name(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        name = value.strip()
        if not name:
            return None
        for sep in (os.sep, os.altsep):
            if sep and sep in name:
                name = name.rsplit(sep, 1)[-1]
        if name.endswith(".py"):
            name = name[:-3]
        name = name.strip()
        return name or None

    def _should_include_agent_class(self, cls: Type[Agent]) -> bool:
        name = getattr(cls, "name", None) or cls.__name__
        if name == "answer":
            return True
        if not getattr(self, "_restrict_agents_to_config", False):
            return True
        module = sys.modules.get(cls.__module__)
        module_path: Optional[Path] = None
        if module is not None:
            module_file = getattr(module, "__file__", None)
            if module_file:
                try:
                    module_path = Path(module_file).resolve()
                except Exception:
                    module_path = Path(os.path.abspath(module_file))
        designated_names = getattr(self, "_designated_agent_names", set())
        designated_files = getattr(self, "_designated_agent_files", set())
        if module_path and module_path in designated_files:
            return True
        if designated_names and name in designated_names:
            return True
        return False

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

    def _load_agents_from_names(self, names: Sequence[str]) -> None:
        base_dir = Path(__file__).resolve().parent / "agents"
        for name in names:
            if name in get_agent_subclasses():
                continue
            candidate = base_dir / f"{name}.py"
            if candidate.is_file():
                self._load_agents_from_file(str(candidate))

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

    def _normalize_knowledge_config(
        self,
        load_config: Optional[Sequence[Dict[str, Any]]],
        generate_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        (
            manual_entries,
            manual_stores,
            manual_store_sources,
            manual_agent_routes,
            base_load_path,
            load_path_provided,
        ) = self._normalize_manual_knowledge(load_config)

        normalized_load_path = self._coerce_path_string(base_load_path)

        generate_cfg = dict(generate_config or {})
        save_value = generate_cfg.get("save_knowledge_path")
        save_path = Path(save_value).expanduser() if save_value not in (None, "") else None

        prompt_value = generate_cfg.get("knowledge_generation_prompt_path")
        prompt_path = Path(prompt_value).expanduser() if prompt_value not in (None, "") else None

        return {
            "generate_knowledge": bool(generate_config),
            "save_knowledge_path": save_path,
            "knowledge_generation_prompt_path": prompt_path,
            "load_knowledge_path": normalized_load_path,
            "load_path_provided": load_path_provided,
            "manual_entries": manual_entries,
            "manual_stores": manual_stores,
            "manual_store_sources": manual_store_sources,
            "manual_agent_routes": manual_agent_routes,
        }

    def _normalize_manual_knowledge(
        self, value: Any
    ) -> Tuple[
        Dict[Optional[int], List[Dict[str, Any]]],
        Dict[Optional[int], List[Dict[str, Any]]],
        Dict[Optional[int], List[Dict[str, Any]]],
        Dict[int, List[str]],
        Optional[str],
        bool,
    ]:
        entry_map: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        store_map: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        store_source_map: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        agent_route_map: Dict[int, List[str]] = defaultdict(list)
        base_load_path: Optional[str] = None
        load_path_provided = False
        entries = self._coerce_manual_knowledge_list(value)
        if not entries:
            return {}, {}, {}, {}, None, False

        def _tag_store_entries_with_config_step(store: Dict[str, List[Dict[str, Any]]], config_step: Any) -> None:
            if config_step in (None, ""):
                return
            for agent_entries in store.values():
                if not isinstance(agent_entries, list):
                    continue
                for entry in agent_entries:
                    if not isinstance(entry, dict):
                        continue
                    meta = entry.get("meta")
                    if not isinstance(meta, dict):
                        meta = {}
                        entry["meta"] = meta
                    meta.setdefault("config_step", config_step)

        for raw in entries:
            step_spec = raw.get("step")
            steps = self._coerce_step_targets(step_spec)
            step_spec_present = "step" in raw and step_spec not in (None, "")
            if step_spec_present and not steps:
                continue
            targets = steps or [None]
            tags = self._coerce_tags(raw.get("tags"))
            agent_values = self._coerce_agent_values(raw.get("agent"))
            agent_field_present = "agent" in raw
            knowledge_agents = agent_values if agent_values else ["*"]
            routing_candidates = [name for name in agent_values if name and name != "*"]
            text_value = raw.get("text") or raw.get("knowledge") or raw.get("content") or raw.get("value")
            text = str(text_value).strip() if text_value is not None else ""
            entry_id = self._coerce_optional_int(raw.get("id"))
            load_path_value = raw.get("load_knowledge_path")
            load_path = self._coerce_path_string(load_path_value)
            is_base_load = bool(load_path) and not step_spec_present
            if is_base_load:
                base_load_path = load_path
                load_path_provided = True

            entry_payloads: List[Dict[str, Any]] = []
            if text:
                for agent_name in knowledge_agents:
                    payload: Dict[str, Any] = {"agent": agent_name, "knowledge": text}
                    if tags:
                        payload["tags"] = list(tags)
                    if entry_id is not None:
                        payload["id"] = entry_id
                    entry_payloads.append(payload)

            store_payload: Optional[Dict[str, List[Dict[str, Any]]]] = None
            store_meta: Optional[Dict[str, Any]] = None
            if load_path and not is_base_load:
                store_meta = {"path": load_path, "loaded": False}
                store_payload = self._load_manual_knowledge_store(load_path)
                if store_payload:
                    if step_spec_present:
                        _tag_store_entries_with_config_step(store_payload, step_spec)
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
                for payload in entry_payloads:
                    entry_map[step].append(dict(payload))
                if (
                    agent_field_present
                    and routing_candidates
                    and step is not None
                ):
                    route = agent_route_map.setdefault(step, [])
                    for candidate in routing_candidates:
                        if candidate not in route:
                            route.append(candidate)

            if load_path and not is_base_load:
                store_targets = [None] if step_spec_present else targets
                for step in store_targets:
                    if store_payload:
                        store_map[step].append(store_payload)
                    if store_meta:
                        store_source_map[step].append(dict(store_meta))
                    else:
                        store_source_map[step].append({"path": load_path, "loaded": False})

        return (
            dict(entry_map),
            dict(store_map),
            dict(store_source_map),
            dict(agent_route_map),
            base_load_path,
            load_path_provided,
        )

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

    def _coerce_step_targets(self, value: Any) -> List[int]:
        matcher = self._build_step_matcher(value)
        if matcher is None:
            return []
        return [step for step in range(1, self.max_steps + 1) if matcher(step)]

    def _build_step_matcher(self, value: Any) -> Optional[Callable[[int], bool]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            matchers = [self._build_step_matcher(item) for item in value]
            matchers = [matcher for matcher in matchers if matcher]
            if not matchers:
                return None
            return lambda step: any(matcher(step) for matcher in matchers)

        if isinstance(value, (int, float, str)):
            if isinstance(value, (int, float)):
                number = self._coerce_positive_int(value)
                if number is None:
                    return None
                return lambda step, target=number: step == target
            text = str(value).strip()
            if not text:
                return None
            if not re.search(r"[<>]=?|==|=", text):
                numbers: List[int] = []
                for part in text.split(","):
                    num = self._coerce_positive_int(part.strip())
                    if num is not None:
                        numbers.append(num)
                if not numbers:
                    return None
                number_set = set(numbers)
                return lambda step: step in number_set
            conditions = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                match = re.match(r"^(<=|>=|<|>|==|=)?\s*(\d+)\s*$", part)
                if not match:
                    continue
                op = match.group(1) or "=="
                num = int(match.group(2))
                conditions.append((op, num))
            if not conditions:
                return None

            def _matches(step: int) -> bool:
                for op, num in conditions:
                    if op in ("=", "==") and step != num:
                        return False
                    if op == "<" and step >= num:
                        return False
                    if op == "<=" and step > num:
                        return False
                    if op == ">" and step <= num:
                        return False
                    if op == ">=" and step < num:
                        return False
                return True

            return _matches
        return None

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
    def _coerce_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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

    @staticmethod
    def _coerce_agent_values(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, Iterable):
            names: List[str] = []
            seen: Set[str] = set()
            for item in value:
                if item in (None, ""):
                    continue
                text = str(item).strip()
                if text and text not in seen:
                    names.append(text)
                    seen.add(text)
            return names
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _normalize_search_mode(
        value: Any,
        *,
        allowed: Set[str],
        default: str,
        label: str,
    ) -> str:
        if value in (None, ""):
            return default
        mode = str(value).strip().lower()
        if mode not in allowed:
            raise ValueError(f"{label} must be one of {sorted(allowed)}")
        return mode

    @staticmethod
    def _validate_search_mode(value: Any, *, allowed: Set[str], label: str) -> str:
        if value in (None, ""):
            raise ValueError(f"{label} must be one of {sorted(allowed)}")
        mode = str(value).strip().lower()
        if mode not in allowed:
            raise ValueError(f"{label} must be one of {sorted(allowed)}")
        return mode

    @staticmethod
    def _coerce_search_config_list(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            entries: List[Dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    entries.append(dict(item))
            return entries
        return []

    @staticmethod
    def _coerce_sampling_distribution(value: Any, *, label: str) -> List[float]:
        if isinstance(value, dict):
            raise ValueError(f"{label} must be a list of non-negative numbers")
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            items = list(value)
        else:
            raise ValueError(f"{label} must be a list of non-negative numbers")
        if not items:
            raise ValueError(f"{label} must not be empty")
        weights: List[float] = []
        for item in items:
            try:
                weight = float(item)
            except (TypeError, ValueError):
                raise ValueError(f"{label} must be a list of non-negative numbers")
            if weight < 0:
                raise ValueError(f"{label} must be a list of non-negative numbers")
            weights.append(weight)
        if all(weight == 0 for weight in weights):
            raise ValueError(f"{label} must include a positive weight")
        return weights

    def _normalize_search_config(
        self,
        config: Any,
        *,
        allowed: Set[str],
        label: str,
        enforce_topk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in self._coerce_search_config_list(config):
            mode = self._validate_search_mode(raw.get("mode"), allowed=allowed, label=f"{label}.mode")
            matcher = self._build_step_matcher(raw.get("step"))
            if matcher is None:
                raise ValueError(f"{label}.step must be set for mode '{mode}'")
            entry: Dict[str, Any] = {"mode": mode, "matcher": matcher}

            if "topk" in raw and raw.get("topk") not in (None, ""):
                topk_value = self._coerce_positive_int(raw.get("topk"))
                if topk_value is None:
                    raise ValueError(f"{label}.topk must be a positive integer")
                if enforce_topk is not None and topk_value != enforce_topk:
                    raise ValueError(f"{label}.topk must be {enforce_topk}")
                entry["topk"] = topk_value

            if "sampling_topk" in raw and raw.get("sampling_topk") not in (None, ""):
                sampling_topk = self._coerce_positive_int(raw.get("sampling_topk"))
                if sampling_topk is None:
                    raise ValueError(f"{label}.sampling_topk must be a positive integer")
                entry["sampling_topk"] = sampling_topk

            if "sampling_distribution" in raw and raw.get("sampling_distribution") not in (None, ""):
                entry["sampling_distribution"] = self._coerce_sampling_distribution(
                    raw.get("sampling_distribution"),
                    label=f"{label}.sampling_distribution",
                )

            if "sampling_distribution_decay_rate" in raw and raw.get("sampling_distribution_decay_rate") not in (None, ""):
                decay_rate = self._coerce_float(raw.get("sampling_distribution_decay_rate"))
                if decay_rate is None or decay_rate < 0:
                    raise ValueError(f"{label}.sampling_distribution_decay_rate must be a non-negative number")
                entry["sampling_distribution_decay_rate"] = decay_rate

            if "random_sampling_rate" in raw and raw.get("random_sampling_rate") not in (None, ""):
                random_rate = self._coerce_float(raw.get("random_sampling_rate"))
                if random_rate is None or random_rate < 0 or random_rate > 1:
                    raise ValueError(f"{label}.random_sampling_rate must be between 0 and 1")
                entry["random_sampling_rate"] = random_rate

            normalized.append(entry)
        return normalized

    @staticmethod
    def _resolve_search_mode(
        base_mode: str,
        config: Sequence[Dict[str, Any]],
        step: Optional[int],
    ) -> str:
        if step is None:
            return base_mode
        resolved = base_mode
        for entry in config:
            matcher = entry.get("matcher")
            if callable(matcher) and matcher(step):
                resolved = entry.get("mode", resolved)
        return resolved

    @staticmethod
    def _resolve_search_settings(
        base_mode: str,
        config: Sequence[Dict[str, Any]],
        step: Optional[int],
        *,
        base_topk: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        settings: Dict[str, Any] = {
            "mode": base_mode,
            "topk": base_topk,
            "sampling_topk": None,
            "sampling_distribution": None,
            "sampling_distribution_decay_rate": None,
            "random_sampling_rate": None,
        }
        matched_entry: Optional[Dict[str, Any]] = None
        if step is None:
            return settings, matched_entry
        for entry in config:
            matcher = entry.get("matcher")
            if callable(matcher) and matcher(step):
                matched_entry = entry
                settings["mode"] = entry.get("mode", settings["mode"])
                for key in (
                    "topk",
                    "sampling_topk",
                    "sampling_distribution",
                    "sampling_distribution_decay_rate",
                    "random_sampling_rate",
                ):
                    if key in entry:
                        settings[key] = entry.get(key)
        return settings, matched_entry

    def _compose_step_knowledge(
        self,
        *,
        base_store: Dict[str, List[Dict[str, Any]]],
        manual_entries: Dict[Optional[int], List[Dict[str, Any]]],
        manual_stores: Dict[Optional[int], List[Dict[str, Any]]],
        step: Optional[int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        merged: Dict[str, List[Dict[str, Any]]] = {}
        matcher_cache: Dict[str, Optional[Callable[[int], bool]]] = {}

        def _get_entry_step_spec(entry: Dict[str, Any]) -> Optional[Any]:
            step_value = entry.get("step")
            if step_value not in (None, ""):
                return step_value
            meta = entry.get("meta")
            if isinstance(meta, dict):
                meta_step = meta.get("step")
                if meta_step not in (None, ""):
                    return meta_step
            return None

        def _get_config_step_spec(entry: Dict[str, Any]) -> Optional[Any]:
            meta = entry.get("meta")
            if isinstance(meta, dict):
                config_step = meta.get("config_step")
                if config_step not in (None, ""):
                    return config_step
            return None

        def _matches_step(entry: Dict[str, Any]) -> bool:
            entry_step = _get_entry_step_spec(entry)
            config_step = _get_config_step_spec(entry) if entry_step is None else None
            step_spec = entry_step if entry_step is not None else config_step
            if step_spec is None:
                return True
            if step is None:
                return False
            cache_key = repr(step_spec)
            matcher = matcher_cache.get(cache_key)
            if cache_key not in matcher_cache:
                matcher = self._build_step_matcher(step_spec)
                matcher_cache[cache_key] = matcher
            if matcher is None:
                return False
            return bool(matcher(step))

        def _merge_store(store: Dict[str, List[Dict[str, Any]]]) -> None:
            for agent, entries in store.items():
                if not isinstance(entries, list):
                    continue
                agent_entries = merged.setdefault(agent, [])
                for entry in entries:
                    if isinstance(entry, dict) and _matches_step(entry):
                        agent_entries.append(dict(entry))

        if isinstance(base_store, dict):
            for agent, entries in base_store.items():
                if isinstance(entries, list):
                    merged[agent] = [
                        dict(entry) for entry in entries if isinstance(entry, dict) and _matches_step(entry)
                    ]

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
                if not _matches_step(entry):
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

    def _make_search_fn(
        self,
        run_llm: LLMClient,
        mode_getter: Optional[Callable[[], str]] = None,
        settings_getter: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ) -> Callable[..., Any]:
        async def _search(
            query: str,
            keys: Sequence[Dict[str, Any]],
            *,
            k: int = 1,
            context: Optional[Dict[str, Any]] = None,
        ) -> List[Dict[str, Any]]:
            mode = mode_getter() if mode_getter else "rm"
            settings = settings_getter() if settings_getter else None
            return await self._search_with_mode(
                query=query,
                keys=keys,
                k=k,
                run_llm=run_llm,
                mode=mode,
                context=context or {},
                search_settings=settings,
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

    def _should_use_rmsearch(self, cfg: Optional[Dict[str, Any]] = None) -> bool:
        config = cfg or self.rm_config
        url = str(config.get("base_url") or config.get("url") or "").strip()
        return bool(url)

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
        config = dict(self.rm_config)
        config.update(overrides)
        effective_purpose = self._normalize_purpose(purpose or config.get("purpose"))

        if not self._should_use_rmsearch(config):
            return []

        url = str(config.get("base_url") or config.get("url") or "").strip()
        if not url:
            if not self._rm_warned_missing_url:
                print(colorize("[seimei] rmsearch skipped: rm_config['base_url'] not set.", ERROR_COLOR), file=sys.stderr)
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
            print(colorize(f"[seimei] warning: KYOTOAI_API_KEY variable is not set", WARNING_COLOR), file=sys.stderr)
            #raise RuntimeError("KYOTOAI_API_KEY environment variable is not set")

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
    def _format_rmsearch_query(query: Union[str, Sequence[Dict[str, Any]]]) -> str:
        if isinstance(query, str):
            stripped = query.strip()
            return format_query_for_rmsearch(stripped or query)
        if isinstance(query, Sequence):
            messages = seimei._coerce_messages(query)
            _, conversation_text, focus_text = seimei._prepare_query_input(messages)
            body = focus_text or conversation_text or ""
            return format_query_for_rmsearch(body)
        return format_query_for_rmsearch(str(query))

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
            tags = seimei._coerce_tags(item.get("tags"))
            formatted = format_key_for_rmsearch(key_text, tags=tags)
            index_map[len(payload)] = item
            payload.append(formatted)
            text_map.setdefault(formatted, item)
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
            entries = data.get("output")
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

    @staticmethod
    def _build_decay_weights(count: int, decay_rate: float) -> List[float]:
        weights: List[float] = []
        current = 1.0
        for _ in range(max(count, 0)):
            weights.append(current)
            current *= decay_rate
        return weights

    @staticmethod
    def _sample_without_replacement(
        items: Sequence[Dict[str, Any]],
        weights: Sequence[float],
        k: int,
    ) -> List[Dict[str, Any]]:
        pool = list(items)
        weight_pool = list(weights)
        selected: List[Dict[str, Any]] = []
        k = min(k, len(pool))
        while pool and weight_pool and len(selected) < k:
            total = sum(weight for weight in weight_pool if weight > 0)
            if total <= 0:
                break
            threshold = random.random() * total
            cumulative = 0.0
            chosen_idx = None
            for idx, weight in enumerate(weight_pool):
                if weight <= 0:
                    continue
                cumulative += weight
                if threshold <= cumulative:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                break
            selected.append(pool.pop(chosen_idx))
            weight_pool.pop(chosen_idx)
        if len(selected) < k and pool:
            selected.extend(pool[: k - len(selected)])
        return selected

    @staticmethod
    def _build_random_results(
        candidates: Sequence[Dict[str, Any]],
        k: int,
        *,
        source: str,
    ) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        pool = list(candidates)
        if not pool:
            return []
        if k >= len(pool):
            chosen = pool
        else:
            chosen = random.sample(pool, k)
        return [
            {
                "key": item.get("key"),
                "payload": item,
                "score": None,
                "source": source,
            }
            for item in chosen
        ]

    def _apply_sampling_results(
        self,
        results: Sequence[Dict[str, Any]],
        *,
        effective_k: int,
        sampling_topk: int,
        distribution_weights: Sequence[float],
    ) -> List[Dict[str, Any]]:
        if not results or effective_k <= 0:
            return []
        pool = list(results[:sampling_topk])
        if len(pool) <= effective_k:
            return pool
        if not distribution_weights:
            return pool[:effective_k]
        weights = list(distribution_weights[: len(pool)])
        if not weights:
            return pool[:effective_k]
        return self._sample_without_replacement(pool, weights, effective_k)

    async def _search_with_mode(
        self,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        *,
        k: int,
        run_llm: Optional[LLMClient],
        mode: str,
        context: Optional[Dict[str, Any]] = None,
        search_settings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not keys:
            return []

        candidates = [item for item in keys if isinstance(item, dict) and item.get("key")]
        if not candidates:
            return []

        requested_k = self._coerce_positive_int(k) or 1
        settings = search_settings or {}
        topk_override = self._coerce_positive_int(settings.get("topk")) if isinstance(settings, dict) else None
        effective_k = topk_override or requested_k
        candidate_count = len(candidates)
        if effective_k > candidate_count:
            effective_k = candidate_count
        if effective_k <= 0:
            return []

        sampling_topk = None
        if isinstance(settings, dict):
            sampling_topk = self._coerce_positive_int(settings.get("sampling_topk"))
        distribution = settings.get("sampling_distribution") if isinstance(settings, dict) else None
        decay_rate = settings.get("sampling_distribution_decay_rate") if isinstance(settings, dict) else None

        if distribution:
            distribution_weights = list(distribution)
            if sampling_topk is None:
                sampling_topk = len(distribution_weights)
            else:
                sampling_topk = min(sampling_topk, len(distribution_weights))
        else:
            distribution_weights = []
            if decay_rate is not None and sampling_topk is None:
                sampling_topk = effective_k
        if sampling_topk is None:
            sampling_topk = effective_k
        sampling_topk = min(sampling_topk, candidate_count)
        if sampling_topk < effective_k:
            effective_k = sampling_topk
        if effective_k <= 0:
            return []

        if distribution_weights:
            distribution_weights = distribution_weights[:sampling_topk]
        elif decay_rate is not None:
            distribution_weights = self._build_decay_weights(sampling_topk, float(decay_rate))

        random_rate = None
        if isinstance(settings, dict):
            random_rate = settings.get("random_sampling_rate")
        if random_rate is not None:
            try:
                rate_value = float(random_rate)
            except (TypeError, ValueError):
                rate_value = 0.0
            if rate_value > 0 and random.random() < rate_value:
                return self._build_random_results(
                    candidates,
                    effective_k,
                    source="random_sampling",
                )

        search_limit = sampling_topk if (sampling_topk > effective_k or distribution_weights) else effective_k
        search_limit = min(search_limit, candidate_count)
        key_map = {item.get("key"): item for item in candidates}
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
        normalized_mode = str(mode or "rm").strip().lower()
        if normalized_mode == "rm":
            try:
                rm_result = self._rmsearch(
                    query=rm_query,
                    keys=list(candidates),
                    k_key=search_limit,
                    purpose=purpose,
                )
                if asyncio.iscoroutine(rm_result):
                    rm_result = await rm_result
                results = self._attach_payloads(rm_result or [], key_map)
                if results:
                    return self._apply_sampling_results(
                        results,
                        effective_k=effective_k,
                        sampling_topk=sampling_topk,
                        distribution_weights=distribution_weights,
                    )
            except Exception as exc:
                print(colorize(f"[seimei] rmsearch selection failed: {exc}", ERROR_COLOR), file=sys.stderr)
        results = await self._llm_route_search(
            query=query,
            keys=candidates,
            k=search_limit,
            run_llm=run_llm or self.llm,
            context=context or {},
        )
        return self._apply_sampling_results(
            results,
            effective_k=effective_k,
            sampling_topk=sampling_topk,
            distribution_weights=distribution_weights,
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
        #conversation_messages = self._convert_history_to_llm(history_messages)

        step_note = ""
        current_step = context.get("current_step")
        total_steps = context.get("max_steps") or self.max_steps
        try:
            step_number = int(current_step)
        except (TypeError, ValueError):
            step_number = None
        try:
            total_number = int(total_steps)
        except (TypeError, ValueError):
            total_number = None
        if step_number is not None:
            if total_number and total_number >= step_number:
                step_note = (
                    f"\nCurrent agent step: {step_number} / {total_number}. "
                    #"Choose the agent whose skills progress this specific step."
                )
            else:
                step_note = (
                    f"\nCurrent agent step: {step_number}. "
                    #"Choose the agent whose skills progress this specific step."
                )
        ''' performs bad with gpt-oss
        system_prompt = (
            "You rank candidate keys for relevance according to the recent conversation among the user, assistants, and tools. "
            "Return a JSON array, each element containing: "
            '{"index": <1-based index of the candidate>, "score": optional float between 0 and 1, "reason": short string}. '
            "Only return up to the requested number of entries. Respond with JSON only.\n\n"
            f"Candidates:\n{numbered}\n"
            f"Select up to {k} candidates most relevant to the conversation."
        )
        '''

        system_prompt = (
            "Select one of candidate agents to be acted according to user system and the recent conversation among the user, assistants, and agents. You should carefully read them and think what agent (next action) should be done in next step. "
            "Return a JSON array, each element containing: "
            '{"reason": short string, "index": <1-based index of the candidate>, "score": optional float between 0 and 1}. '
            "Only return up to the requested number of entries. Respond with JSON only.\n"
            f"Candidates:\n{numbered}\n"
            f"Select up to {k} candidates most relevant to the conversation."
        )
        if step_note:
            system_prompt = f"{system_prompt}{step_note}"
        user_prompt = focus_text or "There is no explicit user question. Choose the candidate that best progresses the conversation."
        if reason_hint:
            user_prompt += f"\nAdditional context: {reason_hint}"
        if step_number is not None and not focus_text:
            qualifier = f"This routing decision occurs at agent step {step_number}"
            if total_number and total_number >= step_number:
                qualifier += f" of {total_number}"
            user_prompt = f"{qualifier}.\n{user_prompt}"

        try:
            routing_messages = history_messages if history_messages else [
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

        if type(data)== dict:
            try:
                idx = int(data.get("index", 0))
            except (TypeError, ValueError):
                raise Exception(f"llm_routing output was not formattable. {reply}")
            
            if idx < 1 or idx > len(candidates):
                raise Exception(f"llm_routing output was not formattable. {reply}")
            
            candidate = candidates[idx - 1]
            result = {
                "key": candidate["key"],
                "payload": candidate,
                "score": data.get("score"),
                "reason": data.get("reason"),
                "source": "llm-routing",
            }
            selected.append(result)
                
            
        elif type(data)== list:
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
        else:
            raise Exception(f"llm_routing output was not formattable. {reply}")

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
        run_llm: LLMClient,
        shared_ctx: Dict[str, Any],
        step: Optional[int] = None,
        allowed_agent_names: Optional[Sequence[str]] = None,
        agent_search_mode: str = "llm",
        knowledge_search_mode: str = "rm",
        agent_search_settings: Optional[Dict[str, Any]] = None,
        knowledge_search_settings: Optional[Dict[str, Any]] = None,
        knowledge_settings_override: bool = False,
    ) -> Optional[Agent]:
        if not self.agents:
            return None

        for agent_obj in self.agents.values():
            self._clear_agent_routing_knowledge(agent_obj)

        candidate_agent_names: List[str] = []
        if allowed_agent_names:
            allowed_iterable: Sequence[Any]
            if isinstance(allowed_agent_names, str):
                allowed_iterable = [allowed_agent_names]
            else:
                allowed_iterable = allowed_agent_names
            seen: Set[str] = set()
            for raw_name in allowed_iterable:
                if raw_name in (None, ""):
                    continue
                name = str(raw_name).strip()
                if not name or name in seen:
                    continue
                if name in self.agents:
                    candidate_agent_names.append(name)
                    seen.add(name)
            if candidate_agent_names:
                if len(candidate_agent_names) == 1:
                    return self.agents[candidate_agent_names[0]]

        if not candidate_agent_names:
            candidate_agent_names = list(self.agents.keys())
        candidate_agents = [self.agents[name] for name in candidate_agent_names]
        if not candidate_agents:
            return None
        candidate_name_set = set(candidate_agent_names)

        sanitized_agent_settings = dict(agent_search_settings or {})
        sanitized_agent_settings.pop("random_sampling_rate", None)
        random_rate = None
        if isinstance(agent_search_settings, dict):
            random_rate = agent_search_settings.get("random_sampling_rate")
        if random_rate is not None:
            try:
                rate_value = float(random_rate)
            except (TypeError, ValueError):
                rate_value = 0.0
            if rate_value > 0 and random.random() < rate_value:
                return random.choice(candidate_agents)

        try:
            query = json.dumps(messages, ensure_ascii=False)
        except Exception:
            query = str(messages)

        if agent_search_mode == "klg":
            knowledge_settings = dict(knowledge_search_settings or {})
            if knowledge_settings_override:
                knowledge_settings.pop("random_sampling_rate", None)
            selected = await self._select_agent_by_knowledge(
                messages=messages,
                run_llm=run_llm,
                shared_ctx=shared_ctx,
                step=step,
                candidate_name_set=candidate_name_set,
                knowledge_search_mode=knowledge_search_mode,
                search_settings=knowledge_settings,
            )
            if selected is not None:
                return selected
            agent_search_mode = "llm"

        selected = await self._select_agent_from_candidates(
            candidates=candidate_agents,
            candidate_name_set=candidate_name_set,
            query=query,
            run_llm=run_llm,
            mode=agent_search_mode,
            step=step,
            search_settings=sanitized_agent_settings,
        )
        if selected is not None:
            return selected

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
            for a in candidate_agents:
                if a.name.endswith("web_search") or a.name == "web_search":
                    return a
        if any(tok in lower for tok in ["bash", "shell", "terminal", "run ", "execute ", "pip ", "python "]):
            for a in candidate_agents:
                if a.name.endswith("code_act") or a.name == "code_act":
                    return a
        for pref in ("think", "default"):
            if pref in candidate_name_set:
                return self.agents[pref]
        return self.agents[candidate_agent_names[0]] if candidate_agent_names else None

    async def _select_agent_from_candidates(
        self,
        *,
        candidates: Sequence[Agent],
        candidate_name_set: Set[str],
        query: str,
        run_llm: LLMClient,
        mode: str,
        step: Optional[int],
        search_settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        keys = [
            {"key": f"{agent.name}: {agent.description}", "agent_name": agent.name}
            for agent in candidates
        ]
        if not keys:
            return None
        context = {"purpose": "agent_routing"}
        if step is not None:
            context["current_step"] = int(step)
            context["max_steps"] = self.max_steps
        try:
            ranked = await self._search_with_mode(
                query=query,
                keys=keys,
                k=1,
                run_llm=run_llm,
                mode=mode,
                context=context,
                search_settings=search_settings,
            )
            if ranked:
                agent_name = ranked[0].get("payload", {}).get("agent_name")
                if agent_name and agent_name in candidate_name_set:
                    return self.agents[agent_name]
                key = ranked[0].get("key", "")
                agent_name = key.split(":", 1)[0].strip()
                if agent_name in candidate_name_set:
                    return self.agents[agent_name]
        except Exception as exc:
            print(colorize(f"[seimei] search-based routing failed: {exc}", ERROR_COLOR), file=sys.stderr)
        return None

    async def _select_agent_by_knowledge(
        self,
        *,
        messages: List[Dict[str, Any]],
        run_llm: LLMClient,
        shared_ctx: Dict[str, Any],
        step: Optional[int],
        candidate_name_set: Set[str],
        knowledge_search_mode: str,
        search_settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        knowledge_store = shared_ctx.get("knowledge")
        if not isinstance(knowledge_store, dict):
            return None

        knowledge_candidates: List[Dict[str, Any]] = []
        for agent_name, entries in knowledge_store.items():
            if entries is None:
                continue
            for entry in self._iter_knowledge_entries(entries):
                if not isinstance(entry, dict):
                    continue
                text = (
                    entry.get("knowledge")
                    or entry.get("text")
                    or entry.get("content")
                    or entry.get("value")
                )
                if not text:
                    continue
                agent_value = entry.get("agent") or agent_name
                agent_candidates = [name for name in self._coerce_agent_values(agent_value) if name != "*"]
                if not agent_candidates:
                    continue
                if candidate_name_set and agent_candidates:
                    if not any(name in candidate_name_set for name in agent_candidates):
                        continue
                candidate = dict(entry)
                candidate.setdefault("agent", agent_value)
                knowledge_candidates.append(candidate)

        if not knowledge_candidates:
            return None

        keys: List[Dict[str, Any]] = []
        for entry in knowledge_candidates:
            key_text = (
                entry.get("knowledge")
                or entry.get("text")
                or entry.get("content")
                or entry.get("value")
            )
            key_text = str(key_text).strip() if key_text is not None else ""
            if not key_text:
                continue
            keys.append(
                {
                    "key": key_text,
                    "knowledge": entry,
                    "tags": self._coerce_tags(entry.get("tags")),
                }
            )
        if not keys:
            return None

        context: Dict[str, Any] = {"purpose": "knowledge_search"}
        knowledge_query = shared_ctx.get("knowledge_query")
        if isinstance(knowledge_query, str) and knowledge_query.strip():
            context["knowledge_query"] = knowledge_query
        if step is not None:
            context["current_step"] = int(step)
            context["max_steps"] = self.max_steps

        try:
            ranked = await self._search_with_mode(
                query=messages,
                keys=keys,
                k=1,
                run_llm=run_llm,
                mode=knowledge_search_mode,
                context=context,
                search_settings=search_settings,
            )
        except Exception as exc:
            print(colorize(f"[seimei] knowledge routing failed: {exc}", ERROR_COLOR), file=sys.stderr)
            ranked = []

        selected_entries: List[Dict[str, Any]] = []
        if ranked:
            for entry in ranked:
                payload = entry.get("payload") or {}
                selected_entry: Optional[Dict[str, Any]] = None
                if isinstance(payload, dict) and payload.get("knowledge"):
                    selected_entry = payload.get("knowledge")
                elif isinstance(payload, dict):
                    selected_entry = payload
                if isinstance(selected_entry, dict):
                    selected_entries.append(selected_entry)
        if not selected_entries:
            if knowledge_candidates:
                selected_entries = [knowledge_candidates[0]]

        agent_values: List[str] = []
        seen_agents: Set[str] = set()
        for entry in selected_entries:
            for name in self._coerce_agent_values(entry.get("agent")):
                if name in (None, "", "*"):
                    continue
                if name not in seen_agents:
                    seen_agents.add(name)
                    agent_values.append(name)
        if candidate_name_set:
            agent_values = [name for name in agent_values if name in candidate_name_set]
        if not agent_values:
            return None

        if len(agent_values) == 1:
            agent_name = agent_values[0]
            agent_obj = self.agents.get(agent_name)
            if agent_obj:
                self._set_agent_routing_knowledge(agent_obj, selected_entries)
                return agent_obj
            return None

        agent_candidates = [self.agents[name] for name in agent_values if name in self.agents]
        chosen_agent = await self._select_agent_from_candidates(
            candidates=agent_candidates,
            candidate_name_set=set(agent_values),
            query=json.dumps(messages, ensure_ascii=False) if messages else "",
            run_llm=run_llm,
            mode=knowledge_search_mode,
            step=step,
        )
        if chosen_agent:
            self._set_agent_routing_knowledge(chosen_agent, selected_entries)
        return chosen_agent

    @staticmethod
    def _set_agent_routing_knowledge(agent_obj: Agent, entries: Sequence[Dict[str, Any]]) -> None:
        setattr(agent_obj, "_Agent__agent_routing_knowledge", list(entries))

    @staticmethod
    def _clear_agent_routing_knowledge(agent_obj: Agent) -> None:
        setattr(agent_obj, "_Agent__agent_routing_knowledge", None)

    @staticmethod
    def _iter_knowledge_entries(value: Any) -> Iterable[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            return [{"knowledge": value}]
        if isinstance(value, Iterable):
            entries: List[Dict[str, Any]] = []
            for item in value:
                entries.extend(seimei._iter_knowledge_entries(item))
            return entries
        return []

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
        raw_knowledge_load_config: Optional[Sequence[Dict[str, Any]]],
        raw_knowledge_generate_config: Optional[Dict[str, Any]],
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
        manual_agent_routes = normalized_knowledge_config.get("manual_agent_routes", {}) or {}
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
            "knowledge_generation_prompt_path": str(
                normalized_knowledge_config.get("knowledge_generation_prompt_path")
            )
            if normalized_knowledge_config.get("knowledge_generation_prompt_path")
            else None,
            "manual_entry_counts": manual_entry_counts,
            "manual_store_counts": manual_store_counts,
            "manual_entries": self._prepare_metadata_value(manual_entry_map),
            "manual_store_sources": self._prepare_metadata_value(manual_store_sources),
            "manual_agent_routes": self._prepare_metadata_value(manual_agent_routes),
            "raw_load_config": self._prepare_metadata_value(raw_knowledge_load_config or []),
            "raw_generate_config": self._prepare_metadata_value(raw_knowledge_generate_config or {}),
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
            "rmsearch_config": self._prepare_metadata_value(self.rm_config),
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
        workspace: Optional[Union[str, Path]] = None,
        agent_search_mode: str = "llm",
        agent_search_config: Optional[Sequence[Dict[str, Any]]] = None,
        knowledge_search_mode: str = "rm",
        knowledge_search_config: Optional[Sequence[Dict[str, Any]]] = None,
        knowledge_load_config: Optional[Sequence[Dict[str, Any]]] = None,
        knowledge_generate_config: Optional[Dict[str, Any]] = None,
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

        normalized_agent_mode = self._normalize_search_mode(
            agent_search_mode,
            allowed={"llm", "rm", "klg"},
            default="llm",
            label="agent_search_mode",
        )
        normalized_knowledge_mode = self._normalize_search_mode(
            knowledge_search_mode,
            allowed={"llm", "rm"},
            default="rm",
            label="knowledge_search_mode",
        )
        normalized_agent_search_config = self._normalize_search_config(
            agent_search_config,
            allowed={"llm", "rm", "klg"},
            label="agent_search_config",
            enforce_topk=1,
        )
        normalized_knowledge_search_config = self._normalize_search_config(
            knowledge_search_config,
            allowed={"llm", "rm"},
            label="knowledge_search_config",
        )

        normalized_config = self._normalize_knowledge_config(
            knowledge_load_config,
            knowledge_generate_config,
        )
        manual_entry_map = normalized_config["manual_entries"]
        manual_store_map = normalized_config["manual_stores"]
        manual_agent_routes = normalized_config["manual_agent_routes"]
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
        run_prompt_path = normalized_config["knowledge_generation_prompt_path"]
        run_shared_ctx = dict(self.shared_ctx)
        initial_agent_settings, _ = self._resolve_search_settings(
            normalized_agent_mode,
            normalized_agent_search_config,
            None,
            base_topk=1,
        )
        initial_knowledge_settings, _ = self._resolve_search_settings(
            normalized_knowledge_mode,
            normalized_knowledge_search_config,
            None,
        )
        run_shared_ctx["agent_search_settings"] = initial_agent_settings
        run_shared_ctx["knowledge_search_settings"] = initial_knowledge_settings
        run_shared_ctx["agent_search_mode"] = initial_agent_settings["mode"]
        run_shared_ctx["knowledge_search_mode"] = initial_knowledge_settings["mode"]
        if workspace not in (None, ""):
            try:
                workspace_path = Path(workspace).expanduser()
            except Exception:
                workspace_path = None
            if workspace_path:
                run_shared_ctx["workspace"] = str(workspace_path)
        else:
            run_shared_ctx.pop("workspace", None)

        def _update_run_knowledge(step: Optional[int]) -> None:
            base_store = self.knowledge_store
            run_shared_ctx["knowledge"] = self._compose_step_knowledge(
                base_store=base_store,
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
        search_fn = self._make_search_fn(
            run_llm,
            mode_getter=lambda: run_shared_ctx.get("knowledge_search_mode", normalized_knowledge_mode),
            settings_getter=lambda: run_shared_ctx.get("knowledge_search_settings"),
        )
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
        existing_agent_steps = 0
        for entry in msg_history:
            role = str(entry.get("role") or "").lower()
            if role == "agent":
                existing_agent_steps += 1
                continue
            if role == "system" and entry.get("agent"):
                existing_agent_steps += 1
        step_idx = existing_agent_steps
        while step_idx < self.max_steps:
            step_idx += 1

            run_shared_ctx["current_step"] = step_idx
            current_agent_settings, agent_match = self._resolve_search_settings(
                normalized_agent_mode,
                normalized_agent_search_config,
                step_idx,
                base_topk=1,
            )
            current_knowledge_settings, knowledge_match = self._resolve_search_settings(
                normalized_knowledge_mode,
                normalized_knowledge_search_config,
                step_idx,
            )
            knowledge_settings_override = False
            if agent_match and agent_match.get("mode") == "klg" and knowledge_match:
                current_knowledge_settings = {
                    "mode": normalized_knowledge_mode,
                    "topk": current_agent_settings.get("topk"),
                    "sampling_topk": current_agent_settings.get("sampling_topk"),
                    "sampling_distribution": current_agent_settings.get("sampling_distribution"),
                    "sampling_distribution_decay_rate": current_agent_settings.get("sampling_distribution_decay_rate"),
                    "random_sampling_rate": current_agent_settings.get("random_sampling_rate"),
                }
                knowledge_settings_override = True
            current_agent_mode = current_agent_settings["mode"]
            current_knowledge_mode = current_knowledge_settings["mode"]
            run_shared_ctx["agent_search_mode"] = current_agent_mode
            run_shared_ctx["knowledge_search_mode"] = current_knowledge_mode
            run_shared_ctx["agent_search_settings"] = current_agent_settings
            run_shared_ctx["knowledge_search_settings"] = current_knowledge_settings
            _update_run_knowledge(step_idx)
            self._update_shared_knowledge_query(run_shared_ctx, msg_history)

            # Decide which agent to run
            allowed_agents = manual_agent_routes.get(step_idx)
            agent_obj = await self._select_next_agent(
                msg_history,
                run_llm,
                run_shared_ctx,
                step_idx,
                allowed_agent_names=allowed_agents,
                agent_search_mode=current_agent_mode,
                knowledge_search_mode=current_knowledge_mode,
                agent_search_settings=current_agent_settings,
                knowledge_search_settings=current_knowledge_settings,
                knowledge_settings_override=knowledge_settings_override,
            )
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
                raw_knowledge_load_config=knowledge_load_config,
                raw_knowledge_generate_config=knowledge_generate_config,
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
            if "entries" in knowledge_generation_result:
                entries_snapshot = knowledge_generation_result.get("entries")
                if entries_snapshot is not None:
                    out["generated_knowledge"] = entries_snapshot
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

        body = "\n\n".join(section for section in sections if section).strip()
        return format_query_for_rmsearch(body)

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
        index_by_text: Dict[str, int] = {}
        for text, kid in items:
            if not text:
                continue
            existing_idx = index_by_text.get(text)
            if existing_idx is None:
                index_by_text[text] = len(deduped)
                deduped.append((text, kid))
                continue
            existing_text, existing_kid = deduped[existing_idx]
            if existing_kid is None and kid is not None:
                deduped[existing_idx] = (existing_text, kid)
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
