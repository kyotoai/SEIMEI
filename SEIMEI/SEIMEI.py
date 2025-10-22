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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Re-export convenience (allows: `from seimei import seimei, llm, agent`)
from . import llm as llm_module
from . import agent as agent_module
llm = llm_module
agent = agent_module

from .agent import Agent, get_agent_subclasses
from . import agents as builtin_agents  # noqa: F401  # ensure built-in agents register
from .llm import LLMClient, TokenLimiter, TokenLimitExceeded

try:
    # Optional: your reward-model-based search (rmsearch) package
    from rmsearch import rmsearch as rmsearch_fn  # type: ignore
except Exception:  # pragma: no cover
    rmsearch_fn = None


class seimei:
    """Main orchestrator.

    Loads agents from user-specified paths, routes steps (via rmsearch if available),
    calls the LLM, and writes a dataset for each run.
    """

    def __init__(
        self,
        agent_config: Sequence[Dict[str, Any]],
        llm_kwargs: Dict[str, Any],
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
        self.llm = LLMClient(**llm_kwargs)

        # Routing
        self.rm_kwargs = rm_kwargs or {}
        self.max_steps = max_steps
        self._rm_warned_missing_base_url = False
        self.max_tokens_per_question = max_tokens_per_question

        # Safety for code_act
        self.allow_code_exec = allow_code_exec
        self.allowed_commands = list(allowed_commands) if allowed_commands else None
        self.approval_callback = approval_callback
        self.agent_log_head_lines = max(int(agent_log_head_lines), 0)

        # Load agents
        self.agents: Dict[str, Agent] = {}
        self._load_agents(agent_config)

        # Attach shared ctx visible to agents (e.g., llm, rmsearch, safety flags)
        self.shared_ctx = {
            "llm": self.llm,
            "rm_kwargs": self.rm_kwargs,
            "rmsearch_fn": rmsearch_fn,
            "allow_code_exec": self.allow_code_exec,
            "allowed_commands": self.allowed_commands,
            "approval_callback": self.approval_callback,
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
                print(f"[seimei] Failed to instantiate agent {cls}: {e}", file=sys.stderr)

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

    # -------------------------- Routing --------------------------

    async def _select_next_agent(self, messages: List[Dict[str, Any]]) -> Optional[Agent]:
        # If rmsearch is available, use it over agent descriptions.
        if rmsearch_fn and self.agents and self.rm_kwargs.get("base_url"):
            keys = [{"key": f"{a.name}: {a.description}"} for a in self.agents.values()]
            try:
                query = json.dumps(messages, ensure_ascii=False)
                rm_result = rmsearch_fn(query=query, keys=keys, k_key=1, **self.rm_kwargs)
                if asyncio.iscoroutine(rm_result):
                    top = await rm_result
                else:
                    top = rm_result
                if top and isinstance(top, list):
                    key = top[0]["key"]
                    agent_name = key.split(":", 1)[0].strip()
                    return self.agents.get(agent_name)
            except Exception as e:
                print(f"[seimei] rmsearch selection failed: {e}", file=sys.stderr)
        elif rmsearch_fn and self.agents and not self.rm_kwargs.get("base_url") and not self._rm_warned_missing_base_url:
            print("[seimei] rmsearch skipped: rm_kwargs['base_url'] not set.", file=sys.stderr)
            self._rm_warned_missing_base_url = True

        # Fallback heuristic
        lower = (messages[-1].get("content", "") if messages else "").lower()
        if "search" in lower or "web" in lower:
            for a in self.agents.values():
                if a.name.endswith("web_search") or a.name == "web_search":
                    return a
        if any(tok in lower for tok in ["bash", "shell", "terminal", "run ", "execute ", "pip ", "python "]):
            for a in self.agents.values():
                if a.name.endswith("code_act") or a.name == "code_act":
                    return a
        # Otherwise pick any agent named 'think' or the first one
        for pref in ("think", "default",):
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
    ) -> Dict[str, Any]:
        # Make a deep-ish copy so we can append steps
        msg_history: List[Dict[str, Any]] = [dict(m) for m in messages]

        run_id = str(uuid.uuid4())
        run_dir = self._make_run_dirs(run_id)
        steps_path = os.path.join(run_dir, "steps.jsonl")
        t0 = time.time()

        def write_step(step: Dict[str, Any]) -> None:
            with open(steps_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")

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

        token_limit_hit = False
        token_limit_error: Optional[TokenLimitExceeded] = None

        # Agent loop (very simple – customize as needed)
        step_idx = 0
        while step_idx < self.max_steps:
            step_idx += 1

            # Decide which agent to run
            agent_obj = await self._select_next_agent(msg_history)
            if agent_obj is None:
                break

            # Call agent
            print(f"[seimei] -> agent {agent_obj.name}")

            try:
                step_res = await agent_obj(
                    messages=msg_history,
                    shared_ctx=run_shared_ctx,
                )
            except TokenLimitExceeded as err:
                token_limit_hit = True
                token_limit_error = err
                print(
                    f"[seimei] !! Token limit exceeded by agent {agent_obj.name}: "
                    f"limit={err.limit}, consumed={err.consumed}"
                )
                step_res = {
                    "content": f"[token_limit] Token limit {err.limit} exceeded with {err.consumed} tokens.",
                    "stop": True,
                    "log": {"limit": err.limit, "consumed": err.consumed, "last_usage": err.last_usage},
                }
            except Exception as e:
                step_res = {"content": f"[agent_error] {type(e).__name__}: {e}"}

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

            # Persist step
            write_step({
                "step": step_idx,
                "agent": agent_obj.name,
                "result": step_res,
                "time": time.time(),
            })

            content = step_res.get("content", "")
            if self.agent_log_head_lines:
                head = "\n".join(content.splitlines()[: self.agent_log_head_lines]) if content else ""
                print(f"[seimei] <- agent {agent_obj.name} head[{self.agent_log_head_lines}]: {head or '[no content]'}")
            else:
                print(f"[seimei] <- agent {agent_obj.name} completed")

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
            try:
                answer, usage = await run_llm.chat(messages=msg_history, system=system)
            except TokenLimitExceeded as err:
                token_limit_hit = True
                token_limit_error = err
                print(
                    f"[seimei] !! Token limit exceeded during final response: "
                    f"limit={err.limit}, consumed={err.consumed}"
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
        self._append_dataset(dataset_record)

        out = {"run_id": run_id, "output": answer, "msg_history": msg_history}
        if return_usage:
            out["usage"] = usage_agg
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
