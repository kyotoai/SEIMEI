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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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

class seimei:
    """Main orchestrator.

    Loads agents from user-specified paths, routes steps (via rmsearch if available),
    calls the LLM, and writes a dataset for each run.
    """

    AGENT_OUTPUT_LIMIT = 5000

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
        load_knowledge_path: Optional[str] = None,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # LLM
        if llm_kwargs is None:
            raise ValueError("llm_kwargs must be provided")
        self.llm = LLMClient(**llm_kwargs)

        # Routing
        default_rm_settings = {"agent_routing": False, "knowledge_search": True}
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
        self.load_knowledge_path = load_knowledge_path
        self.knowledge_store = self._load_knowledge_store(load_knowledge_path)

        # Load agents
        self.agents: Dict[str, Agent] = {}
        normalized_agent_config = list(agent_config or [])
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

        print()
        print("--- rmsearch 1 ---")
        print("query: ", query)

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
        
        print()
        print("--- rmsearch 2 ---")
        print("serialized_query: ", serialized_query)

        payload: Dict[str, Any] = {
            "queries": [serialized_query],
            "keys": key_payload,
            "k": limit,
        }
        if purpose:
            payload["purpose"] = purpose

        response = requests.post(url, json=payload, timeout=timeout or 10)
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
            "You must run think agent several times before you provide your answer. Also, after you get 2, 3 agent outputs, you must select answer agent to finish your inference."
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


    # -------------------------- Inference --------------------------

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        stop_when: Optional[Callable[[List[Dict[str, Any]]], bool]] = None,
        return_usage: bool = True,
        run_name: Optional[str] = None,
        generate_knowledge: bool = False,
        save_knowledge_path: Optional[str] = None,
        knowledge_prompt_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        # Make a deep-ish copy so we can append steps
        msg_history: List[Dict[str, Any]] = [dict(m) for m in messages]
        run_label = (run_name or "").strip()
        log_prefix = f"[seimei {run_label}]" if run_label else "[seimei]"
        color_enabled = supports_color()

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
        run_shared_ctx = dict(self.shared_ctx)
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
        if knowledge_prompt_path is not None:
            resolved_prompt_path = Path(knowledge_prompt_path).expanduser()

        # Agent loop (very simple – customize as needed)
        step_idx = 0
        while step_idx < self.max_steps:
            step_idx += 1

            self._update_shared_knowledge_query(run_shared_ctx, msg_history)

            # Decide which agent to run
            agent_obj = await self._select_next_agent(msg_history, search_fn)
            if agent_obj is None:
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

            if step_res.get("final_output"):
                final_agent_output = str(step_res.get("final_output"))
                final_agent_name = agent_obj.name

            # Append to history
            agent_msg = {
                "role": "agent",
                "name": agent_obj.name,
                "content": step_res.get("content", ""),
            }
            # Optional fields for richer dataset
            for k in ("code", "chosen_instructions", "log"):
                if k in step_res:
                    agent_msg[k] = step_res[k]
            msg_history.append(agent_msg)
            self._update_shared_knowledge_query(run_shared_ctx, msg_history)

            # Persist step
            write_step({
                "step": step_idx,
                "agent": agent_obj.name,
                "result": step_res,
                "time": time.time(),
            })

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
                break

            # Stop early if agent says so
            if step_res.get("stop", False):
                break

            if token_limit_hit:
                break

        # Final assistant call
        if not token_limit_hit:
            if final_agent_output is not None:
                answer = final_agent_output
                usage = {}
                msg_history.append({"role": "assistant", "content": answer})
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
                except TokenLimitExceeded as err:
                    token_limit_hit = True
                    token_limit_error = err
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

        # Save run artifacts
        with open(os.path.join(run_dir, "messages.json"), "w", encoding="utf-8") as f:
            json.dump(msg_history, f, ensure_ascii=False, indent=2)

        with open(os.path.join(run_dir, "output.txt"), "w", encoding="utf-8") as f:
            f.write(answer)

        meta = {
            "run_id": run_id,
            "model": self.llm.model,
            "timings": {"total_sec": time.time() - t0},
            "usage": usage_agg,
            "max_steps": self.max_steps,
            "token_limit_hit": token_limit_hit,
        }
        if run_name is not None:
            meta["run_name"] = run_name
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

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

        if generate_knowledge:
            target_path = self._resolve_knowledge_output_path(save_knowledge_path)
            print(
                colorize(
                    f"{log_prefix} Knowledge generation: starting (target={target_path})",
                    KNOWLEDGE_COLOR,
                    enable=color_enabled,
                )
            )
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
                print(
                    colorize(
                        f"[seimei] Failed to auto-generate knowledge for run {run_id}: {exc}",
                        ERROR_COLOR,
                        enable=color_enabled,
                    ),
                    file=sys.stderr,
                )

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
