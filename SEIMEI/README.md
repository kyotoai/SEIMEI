# SEIMEI

A lightweight, Codex-inspired, agentic orchestration layer for LLM workflows.  
It loads pluggable *agents* from a folder/file, routes steps using `rmsearch` (if installed) or a simple heuristic, and logs every run to a structured dataset.

---

## Overview

1. `seimei.py` — orchestrator that ties everything together (agent loading, routing, shared search, dataset logging)
2. `llm.py` — minimal LLM client (OpenAI-compatible API or local vLLM-compatible base_url)
3. `agent.py` — base `Agent` class with logging + a tiny registry
4. `agents/think.py` — planner agent that selects instructions via the shared search API
5. `agents/web_search.py` — example agent that performs web search (uses `duckduckgo_search` if available, or returns a graceful message)
6. `agents/code_act.py` — example agent that executes *whitelisted* shell commands with a timeout
7. `agents/answer.py` — finalizer agent that summarizes findings into the user-facing reply

> You can add more agents by dropping Python files that define subclasses of `Agent` into a directory and pointing `agent_config` to it.

---

## Install (editable)

```bash
git clone https://github.com/kyotoai/SEIMEI.git
pip install -e SEIMEI/.
```
> For this minimal skeleton, you can also just `pip install -e .` inside the generated folder.

Optional dependencies:
```bash
pip install duckduckgo_search requests
```

---

## Set API key

```bash
export OPENAI_API_KEY = "(your_openai_api_key)"
export KYOTOAI_API_KEY = "(your_kyotoai_api_key)"
```

---


## Quick start

### In CLI app

```bash
seimei
```


### python code

```python
import asyncio
from seimei import seimei  # class name is `seimei` (lowercase) for convenience

async def demo_code_act():
    orchestrator = seimei(
        llm_kwargs={"model": "gpt-5-nano"},
        rm_kwargs={"url": "https://kyotoai.net/v1/rmsearch", "agent_routing":False, "knowledge_search":True},
        allow_code_exec=True,
        agent_log_head_lines=1,
        max_tokens_per_question=30000,
    )

    result = await orchestrator(
        messages=[
            {"role": "user", "content": "Design a single 7-day endgame plan for my turbulence surrogate project based on my past history."},
        ],
        knowledge_config={
            "load_knowledge_path": "seimei_knowledge/knwoledge.csv",
        },
    )

asyncio.run(demo_code_act())
```

### Output Schema

`seimei.__call__` returns:
```python
{
  "run_id": "UUID",
  "output": str,
  "msg_history": [
    {"role": "...", "content": "...", "name": "...", "code": "...", "chosen_instructions": [...]},
    ...
  ],
  "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
}
```

### Execution Flow

Each call to `seimei(...)` proceeds with these steps:
1. **Setup** — a run-specific directory is created and the shared context is copied. If `max_tokens_per_question` is set, a `TokenLimiter` is attached to the run-specific LLM proxy before agents execute. Provide `run_name` to have logs show up as `[seimei your-label]`.
2. **Agent selection** — the shared search helper prefers `rmsearch` when configured, and automatically falls back to an LLM router to rank agent descriptions when `rmsearch` is unavailable. Results feed into the agent registry (with a heuristic fallback if everything else fails).
3. **Agent execution** — the agent receives the message history and shared context, then returns a dict payload. The orchestrator logs structured blocks such as `<command>`, `<query>`, `<user input>`, and `<output>` under numbered steps so multi-agent traces remain easy to scan.
4. **Safety checks** — if an agent triggers the token limiter, execution stops immediately, the last step is logged with `[token_limit]`, and the run is paused before the final LLM turn.
5. **Final response** — if an agent (e.g., the built-in `answer`) returns `final_output`, that text becomes the assistant reply. Otherwise, the run-specific LLM proxy calls `chat` (obeying the concurrency semaphore and token limiter) to produce the final answer.
6. **Persistence** — `messages.json`, `steps.jsonl`, `output.txt`, `meta.json`, and an entry in `dataset.jsonl` (including `run_name` when provided) are written so the run can be replayed or used for training.

---

## Create an agent

The built-in `think` and `answer` agents cover planning and final summarisation out of the box.  
If you need something custom, you can still create your own agents and take advantage of the shared search helper:

```python
# my_agents/prioritise_docs.py
from seimei import Agent

class prioritise_docs(Agent):
    """Rank documentation chunks that should be read next."""

    description = "Select document chunks relevant to the latest user request."

    async def inference(self, messages, shared_ctx, **kwargs):
        search = shared_ctx.get("search")
        if not search:
            return {"content": "search helper unavailable", "log": {}}

        question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        candidates = [{"key": text, "section": name} for name, text in kwargs.get("docs", [])]

        ranked = await search(
            query=question,
            keys=candidates,
            k=3,
            context={"purpose": "doc_ranking"},
        )
        plan = "\n".join(f"- {item['payload']['section']}" for item in ranked if item.get("payload"))
        return {"content": f"Review next:\n{plan}", "log": {"query": question, "sections": ranked}}
```

Drop this file into a directory referenced by `agent_config` and SEIMEI will register it automatically.

Key shared context entries exposed to agents:
- `shared_ctx["search"]`: async helper that routes to `rmsearch` when available or falls back to LLM-based ranking. Accepts `query`, `keys`, optional `k`, and a `context` dict.
- `shared_ctx["instruction_list"]`: optional sequence of instruction dicts consumed by the built-in `think` planner. Override it (e.g., `orchestrator.shared_ctx["instruction_list"] = [...]`) to customise planning behaviour.
- `shared_ctx["llm"]`: run-scoped LLM client already bound to the active token limiter.

---

## Dataset logging

Every run is saved under `log_dir`:
```
log_dir/
  dataset.jsonl        # one JSON per run (append-only)
  run-YYYYmmdd-HHMMSS-<uuid>/
    messages.json      # full message history
    steps.jsonl        # one line per agent step
    output.txt         # final assistant text
    meta.json          # run metadata (model, timings, etc.)
```

Each JSON record in `dataset.jsonl` has fields:
- `schema_version` (e.g., `1`)
- `run_id`
- `input_messages`
- `final_output`
- `steps` (array)
- `model_info`
- `timestamps`
- `token_limit` (limit, consumed, hit flag, and last_error snapshot if a cap was triggered)

This gives you a clean training/eval dataset with full provenance.

---

## Resume or replay runs

`messages.json` stores the exact `role="agent"` entries that SEIMEI appended while the run was executing. Feeding those messages back into the orchestrator lets you continue the workflow without changing the prompt that the final LLM call will see.

```python
from seimei import seimei, load_run_messages

orchestrator = seimei(...)

messages = load_run_messages(
    "run-20251119-211751-e7aeb350",
    runs_dir="seimei_runs",
    step=3,  # optional: stop after the 3rd agent step before the previous assistant reply
)

result = await orchestrator(messages=messages)
```

- Omit `step` to replay the full transcript (including the prior assistant message). Provide a 1-indexed `step` to stop the history after that agent turn when you want SEIMEI to keep working from that point forward.
- `load_run_messages` simply returns the stored dictionaries, so the LLM payload that produces the final answer stays byte-for-byte identical whether the agent continued immediately or you resumed it in a later call.

---

## `seimei.__init__`

**Arguments**
- `agent_config: list[dict]` — A list of entries like `{"dir_path": "path/to/agents"}` or `{"file_path": "path/to/agent.py"}`.
- `llm_kwargs: dict` — Passed to the LLM client. Supports either OpenAI API or any OpenAI-compatible server via `base_url` and options like `max_concurrent_requests` for throttling.
- `rm_kwargs: dict` — Optional; forwarded to `rmsearch` if you use it.
- `log_dir: str` — Directory to store dataset logs. Default: `./seimei_runs`.
- `max_steps: int` — Hard cap on agent steps per call. Default: 8.
- `allow_code_exec: bool` — If `True`, enables `code_act` to actually run whitelisted commands.
- `allowed_commands: list[str] | None` — Optional allowlist for `code_act` (e.g., `["python", "pip", "ls", "cat"]`).
- `approval_callback: Optional[Callable[[str], bool]]` — Optional function invoked before running a command; return `True` to allow.
- `agent_log_head_lines: int` — How many lines from each agent's reply to echo to stdout. Set to `0` to suppress previews. Default: `3`.
- `max_tokens_per_question: int | None` — Optional per-run token budget applied across all LLM calls. When exceeded, execution pauses and the run is marked with `[token_limit]`.

**Outputs**
- Creates orchestrator instance.

**Notices**
- If `rmsearch` is not installed, routing falls back to a simple heuristic.
- Code execution is *off* by default; enable it explicitly and provide an allowlist.

---

## `seimei.__call__`

**Arguments**
- `messages: list[dict]` — Standard chat messages list.
- `system: Optional[str]` — Optional system instruction to pin for the LLM call.
- `stop_when: Optional[Callable[[list[dict]], bool]]` — Optional predicate to terminate early based on `messages`.
- `return_usage: bool` — If `True`, returns token usage if available.
- `run_name: Optional[str]` — Optional label shown in terminal logs (e.g., `[seimei question-1]`) and persisted to `meta.json` / `dataset.jsonl`.
- The orchestrator prints agent start/finish markers to stdout so you can follow progress in real time.

**Outputs**
- Dict with `run_id`, `output`, `msg_history`, and optional `usage`.

**Notices**
- Messages are mutated by appending agent steps; pass a copy if you need to preserve the original.
