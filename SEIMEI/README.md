# SEIMEI

A lightweight, Codex-inspired, agentic orchestration layer for LLM workflows.  
It loads pluggable *agents* from a folder/file, routes steps using `rmsearch` (if installed) or a simple heuristic, and logs every run to a structured dataset.

---

## Overview

1. `seimei.py` — orchestrator that ties everything together (agent loading, step routing, dataset logging)
2. `llm.py` — minimal LLM client (OpenAI-compatible API or local vLLM-compatible base_url)
3. `agent.py` — base `Agent` class with logging + a tiny registry
4. `web_search.py` — example agent that performs web search (uses `duckduckgo_search` if available, or returns a graceful message)
5. `code_act.py` — example agent that executes *whitelisted* shell commands with a timeout

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

## Quick start

```python
import asyncio
from seimei import seimei  # class name is `seimei` (lowercase) for convenience

system = "You're a genius mathematician."

problems = [
    "Three airline companies operate flights from Dodola island. Each company has a different schedule of departures. The first company departs every 100 days, the second every 120 days and the third every 150 days. What is the greatest positive integer d for which it is true that there will be d consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?",
    "Fred and George take part in a tennis tournament with 4046 other players. In each round, the players are paired into 2024 matches. How many ways are there to arrange the first round such that Fred and George do not have to play each other?",
]

agent_config = [
    {"dir_path": "./agents"},  # can be a folder or a single file path
]

llm_kwargs = {
    "model": "gpt-5-nano",
    "max_inference_time": 1000,
    # OR talk to a local OpenAI-compatible server:
    # "base_url": "http://localhost:7000/v1"
}

rm_kwargs = {
    "base_url": "http://localhost:7000/v1",  # optional for `rmsearch` if you use it
}

sm = seimei(
    agent_config=agent_config,
    llm_kwargs=llm_kwargs,
    rm_kwargs=rm_kwargs,
    log_dir="./seimei_runs",
)

async def main():
    results = await asyncio.gather(
        *[
            sm(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": problem},
                ]
            )
            for problem in problems
        ]
    )
    print(results[0]["output"])

asyncio.run(main())
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

---

## Create an agent

```python
# my_agents/think.py
from agent import Agent

class think(Agent):
    """Think about the current message and plan a next action."""

    description = "This agent plans the next action given the current conversation state."

    async def inference(self, messages, **kwargs):
        # Minimal demo: echo a planning thought
        plan = "I'll analyze the question, then either search the web or compute, then answer."
        return {"content": plan, "log": {"plan": plan}}
```

You can run this agent by adding its folder to `agent_config` and letting `rmsearch` (or the heuristic) pick it.

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

This gives you a clean training/eval dataset with full provenance.

---

## `seimei.__init__`

**Arguments**
- `agent_config: list[dict]` — A list of entries like `{"dir_path": "path/to/agents"}` or `{"file_path": "path/to/agent.py"}`.
- `llm_kwargs: dict` — Passed to the LLM client. Supports either OpenAI API or any OpenAI-compatible server via `base_url`.
- `rm_kwargs: dict` — Optional; forwarded to `rmsearch` if you use it.
- `log_dir: str` — Directory to store dataset logs. Default: `./seimei_runs`.
- `max_steps: int` — Hard cap on agent steps per call. Default: 8.
- `allow_code_exec: bool` — If `True`, enables `code_act` to actually run whitelisted commands.
- `allowed_commands: list[str] | None` — Optional allowlist for `code_act` (e.g., `["python", "pip", "ls", "cat"]`).
- `approval_callback: Optional[Callable[[str], bool]]` — Optional function invoked before running a command; return `True` to allow.

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

**Outputs**
- Dict with `run_id`, `output`, `msg_history`, and optional `usage`.

**Notices**
- Messages are mutated by appending agent steps; pass a copy if you need to preserve the original.
