# Knowledge Optimizer (`KlgOptimizer`)

## Summary

`KlgOptimizer` is a function that automatically improves a seimei knowledge pool by running the
seimei agent on a labeled dataset, measuring answer quality, and using an LLM to rewrite and
expand the knowledge entries that matter most.

The public API lives in [`seimei/train/optim.py`](../optim.py).
The optimizer logic lives in [`seimei/train/klg_optim/seimei_v1.py`](./seimei_v1.py).
New optimizer variants can be added by creating a new `<name>.py` file here with a `main(**kwargs)` function.

---

## Quick Start

```python
import json
from seimei.train import KlgOptimizer

# Load the bundled SEIMEI-library Q&A dataset (15 rows, easy → hard)
with open("seimei_dataset/default.json", encoding="utf-8") as f:
    dataset = json.load(f)

new_knowledge = KlgOptimizer(
    dataset=dataset,
    optimizer_type="seimei_v1",
    n_sample=1,
    n_epoch=1,
    n_new_klg_per_epoch=3,
    update_klg_threshold=0.1,
    metric="answer_exact_match",
    load_knowledge_path="seimei_knowledge/default.csv",
    save_knowledge_path="seimei_knowledge/improved.csv",
    cache_path="cache.json",
    # seimei constructor args
    llm_config={"model": "gpt-4o", "api_key": "..."},
    agent_config={"agents": [...]},
    log_dir="runs/",
)
# new_knowledge is a list of dicts in the same format as default.csv
```

The dataset at `seimei_dataset/default.json` contains 15 question-answer pairs about
the SEIMEI library itself, ranging from easy (default parameter values, class names)
to hard (internal algorithm details such as cache key formats and epoch reuse logic).
It is a convenient starting point for verifying that KlgOptimizer improves a
knowledge pool on a concrete, self-contained domain.

---

## Detailed Usage

### Import

```python
from seimei.train import KlgOptimizer
```

### Function signature

```python
def KlgOptimizer(
    *,
    dataset: List[Dict[str, Any]],
    optimizer_type: str = "seimei_v1",
    n_sample: int = 1,
    n_epoch: int = 1,
    n_new_klg_per_epoch: int = 3,
    update_klg_threshold: float = 0.1,
    metric: str = "answer_exact_match",
    load_knowledge_path: str,
    save_knowledge_path: str,
    cache_path: str = "cache.json",
    workspace_root: Optional[str] = None,
    patch_dir: Optional[str] = None,
    **kwargs,  # seimei constructor + call args (see below)
) -> List[Dict[str, Any]]
```

All parameters are keyword-only.

### Standard usage (QA / text tasks)

```python
new_knowledge = KlgOptimizer(
    dataset=dataset,
    n_epoch=3,
    n_sample=2,
    n_new_klg_per_epoch=5,
    update_klg_threshold=0.05,
    load_knowledge_path="seimei_knowledge/default.csv",
    save_knowledge_path="seimei_knowledge/improved.csv",
    cache_path="cache.json",
    llm_config={"model": "gpt-4o", "api_key": "..."},
    agent_config={"agents": [...]},
    log_dir="runs/",
    system="You are a helpful assistant.",
    knowledge_search_mode="klg",
)
```

### Workspace / coding task usage

```python
new_knowledge = KlgOptimizer(
    dataset=dataset,
    load_knowledge_path="seimei_knowledge/default.csv",
    save_knowledge_path="seimei_knowledge/improved.csv",
    cache_path="cache.json",
    llm_config={"model": "gpt-4o", "api_key": "..."},
    agent_config={"agents": [...]},
    log_dir="runs/",
    allow_code_exec=True,
    workspace_root="workspaces/",   # a workspace subdirectory is created here
    patch_dir="patches/",           # optional: .patch files applied per problem row
    repo_root="/path/to/repo",      # used to prepare the workspace copy
)
```

### Resume after interruption

The cache file (`cache_path`) records every completed inference.
Simply re-run the same script — any run already stored in the cache is skipped.

---

## All Arguments

### `KlgOptimizer` parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `dataset` | `list[dict]` | — | yes | Labeled dataset. Each dict must have `"Question"` and `"CorrectAnswer"` keys. |
| `optimizer_type` | `str` | `"seimei_v1"` | no | Name of the optimizer module (`seimei/train/klg_optim/<name>.py`). |
| `n_sample` | `int` | `1` | no | Number of seimei runs per dataset row per epoch. More samples give more stable score estimates but cost more. |
| `n_epoch` | `int` | `1` | no | Number of optimization epochs. Each epoch refines the pool further. |
| `n_new_klg_per_epoch` | `int` | `3` | no | Number of brand-new knowledge entries the LLM generates per epoch. Set to `0` to disable. |
| `update_klg_threshold` | `float` | `0.1` | no | Minimum mean score improvement (rerun − baseline) required to accept a knowledge update. Range `0.0–1.0`. |
| `metric` | `str` | `"answer_exact_match"` | no | Scoring metric. Currently `"answer_exact_match"` (LLM-as-judge, 0.0–1.0) is the only supported value. |
| `load_knowledge_path` | `str` | — | yes | Path to the initial knowledge CSV. The file must exist. |
| `save_knowledge_path` | `str` | — | yes | Path where the optimized knowledge CSV is written after every epoch. |
| `cache_path` | `str` | `"cache.json"` | no | Path for the run-cache JSON. Created automatically; resumable across restarts. |
| `workspace_root` | `str \| None` | `None` | no | Root directory for workspace copies (workspace / coding mode). |
| `patch_dir` | `str \| None` | `None` | no | Directory of `.patch` files applied per problem row. Requires `workspace_root`. |

### `**kwargs` — seimei constructor args (passed through)

These are forwarded to `seimei.__init__()`:

| kwarg | Description |
|-------|-------------|
| `llm_config` | **Required for `seimei_v1`.** Dict passed to the LLM client (e.g. `{"model": "gpt-4o", "api_key": "..."}`) |
| `agent_config` | Agent pipeline configuration dict |
| `rm_config` | Reward model configuration dict |
| `log_dir` | Directory for seimei run logs |
| `max_steps` | Maximum agent steps per run |
| `allow_code_exec` | Allow code execution inside the agent |
| `allowed_commands` | List of shell commands the agent may run |
| `approval_callback` | Callable for human-in-the-loop approval |
| `agent_log_head_lines` | Number of head lines to print per agent step |
| `max_tokens_per_question` | Hard token budget per question |
| `llm_client_class` | Custom LLM client class override |

### `**kwargs` — seimei call args (passed through)

These are forwarded to `seimei.__call__()` for every inference:

| kwarg | Description |
|-------|-------------|
| `system` | System prompt string |
| `stop_when` | Callable `(msg_history) -> bool` to stop agent early |
| `return_usage` | If `True`, include token usage in run result |
| `agent_search_config` | Agent retrieval search configuration |
| `knowledge_search_mode` | Knowledge retrieval mode (default `"klg"`) |
| `knowledge_search_config` | Knowledge retrieval search configuration |
| `knowledge_generate_config` | Knowledge auto-generation configuration |

### Additional workspace kwargs

| kwarg | Description |
|-------|-------------|
| `repo_root` | Path to the source repository used to populate the workspace copy |

---

## Return Value

```python
List[Dict[str, Any]]
```

A list of knowledge pool entries, each a dict with the same schema as rows in `default.csv`:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique numeric identifier within the pool |
| `agent` | `str` | Agent name this snippet is attached to (e.g. `"think"`, `"code_act"`, `"answer"`) |
| `text` | `str` | The knowledge snippet text |
| `tags` | `list[str]` | Descriptive labels (may be empty) |

The same content is also written incrementally to `save_knowledge_path` after every epoch.

---

## Detailed Algorithm (`seimei_v1`)

The algorithm runs for `n_epoch` epochs.  Within each epoch:

```
Epoch N
│
├─ Step 1 ─ Inference (baseline)
│   For every (dataset row, sample index):
│     • Call seimei with the current knowledge pool loaded from save_knowledge_path
│     • Collect: output text, msg_history
│     • Score output with LLM judge → score (0.0–1.0), feedback string
│     • Extract which knowledge IDs were retrieved during the run
│     • Cache result in cache_path to enable resume
│
│   (From epoch 2 onward, the rerun results of the previous epoch are reused
│    as the baseline — no duplicate inference is done.)
│
├─ Step 2 ─ Build knowledge → inference map
│   knowledge_id → [
│     {question, output, score, feedback},
│     ...
│   ]
│
├─ Step 3 ─ Assess existing knowledge (per entry)
│   For each entry in the current pool:
│     • Collect up to 10 inferences that retrieved this entry
│     • Ask LLM: "Given these scores and feedback, write an improved version"
│     • LLM returns: improved_text, rationale
│     • Entries with no inferences this epoch are left unchanged
│
├─ Step 4 ─ Generate new knowledge
│   • Feed all assessment rationales to the LLM
│   • Ask LLM to generate n_new_klg_per_epoch new snippets addressing gaps
│   • Each new snippet has: agent, text, tags
│   • New entries get integer IDs continuing from max(existing IDs) + 1
│   • Write the updated pool (improved texts + new entries) to save_knowledge_path
│
└─ Step 5 ─ Rerun & threshold filter
    For every (dataset row, sample index):
      • Rerun seimei with the updated pool
      • Score → rerun_score

    For each knowledge entry:
      • baseline_mean = mean(scores from step 1 runs that used this entry)
      • rerun_mean    = mean(scores from step 5 runs that used this entry)
      • improvement   = rerun_mean − baseline_mean

    Decision:
      improvement > update_klg_threshold  → keep updated text / new entry
      improvement ≤ update_klg_threshold
        and entry is original             → revert to original text
        and entry is new                  → discard silently
      entry never retrieved in either run → original entries kept, new entries discarded

    Write the final pool to save_knowledge_path
    The step-5 run records become step-1 input for the next epoch
```

### Cache schema (`cache.json`)

```json
{
  "schema_version": 2,
  "run_cache": {
    "<dataset_idx>::<epoch>::<sample>":        { "output": "...", "score": 0.8, "feedback": "...", "knowledge_ids": ["1","3"] },
    "<dataset_idx>::<epoch>::<sample>::rerun": { "output": "...", "score": 0.9, "feedback": "...", "knowledge_ids": ["1","5"] }
  }
}
```

A cache entry is keyed by `dataset_idx::epoch::sample[::rerun]`.
Any run whose key already exists in the cache is skipped entirely.

---

## Example Input and Output

### Input — dataset

```python
dataset = [
    {"Question": "What is the boiling point of water at sea level?", "CorrectAnswer": "100°C"},
    {"Question": "Who wrote Hamlet?",                                "CorrectAnswer": "William Shakespeare"},
    {"Question": "What is the chemical formula for water?",          "CorrectAnswer": "H2O"},
]
```

### Input — `seimei_knowledge/default.csv` (load_knowledge_path)

| id | agent | text | tags |
|----|-------|------|------|
| 1  | think | Always verify numeric answers with unit analysis. | ["math","science"] |
| 2  | answer | Provide concise factual answers without preamble. | ["style"] |
| 3  | code_act | Use print() to display intermediate results. | ["coding"] |

### Call

```python
new_knowledge = KlgOptimizer(
    dataset=dataset,
    optimizer_type="seimei_v1",
    n_sample=2,
    n_epoch=2,
    n_new_klg_per_epoch=2,
    update_klg_threshold=0.1,
    metric="answer_exact_match",
    load_knowledge_path="seimei_knowledge/default.csv",
    save_knowledge_path="seimei_knowledge/improved.csv",
    cache_path="cache.json",
    llm_config={"model": "gpt-4o", "api_key": "sk-..."},
    agent_config={"agents": [...]},
    log_dir="runs/",
)
```

### Console output (example)

```
[seimei_v1] === Epoch 1/2 ===
[seimei_v1] Step 1: Running 6 inference(s)...
[seimei_v1] Step 2: Building knowledge-inference map...
[seimei_v1]   2/3 knowledge entries were used.
[seimei_v1] Step 3: Assessing 3 knowledge entries...
[seimei_v1] Step 4: Generating 2 new knowledge entry(s)...
[seimei_v1]   Generated 2 new entries.
[seimei_v1] Step 5: Rerunning 6 inference(s) with updated pool...
[seimei_v1]   Per-knowledge score improvements (threshold=0.1):
[seimei_v1]     id=1: 0.500 -> 0.750 (delta=+0.250) [keep]
[seimei_v1]     id=2: 0.833 -> 0.833 (delta=+0.000) [revert/discard]
[seimei_v1]     id=4: 0.500 -> 0.667 (delta=+0.167) [keep]
[seimei_v1] Epoch 1 done. Final pool size: 4 entries.

[seimei_v1] === Epoch 2/2 ===
[seimei_v1] Step 1: Reusing previous epoch's rerun results.
...
[seimei_v1] Epoch 2 done. Final pool size: 5 entries.
```

### Return value

```python
[
    {
        "id": 1,
        "agent": "think",
        "text": "Always verify numeric answers with unit analysis. For temperatures, confirm whether Celsius, Fahrenheit, or Kelvin is expected.",
        "tags": ["math", "science"]
    },
    {
        "id": 3,
        "agent": "code_act",
        "text": "Use print() to display intermediate results.",
        "tags": ["coding"]
    },
    {
        "id": 4,
        "agent": "think",
        "text": "Cross-check author attribution with work title before answering.",
        "tags": ["literature", "fact-checking"]
    },
    {
        "id": 5,
        "agent": "answer",
        "text": "For chemistry questions, use standard notation (e.g. H₂O, CO₂).",
        "tags": ["chemistry", "style"]
    }
]
```

> Note: entry `id=2` was reverted to its original text (improvement was 0.0, below threshold 0.1).
> Entry `id=2` is absent from the result because the revert logic preserved only improved or original entries that survived scoring; entries never retrieved in the rerun are kept as-is with original text.

### Output CSV (`seimei_knowledge/improved.csv`)

Same rows as the return value, written in CSV format compatible with `load_knowledge_path`.

---

## Adding a New Optimizer

Create `seimei/train/klg_optim/<your_name>.py` with a `main(**kwargs)` function:

```python
# seimei/train/klg_optim/my_optimizer.py

def main(**kwargs):
    dataset = kwargs["dataset"]
    load_knowledge_path = kwargs["load_knowledge_path"]
    save_knowledge_path = kwargs["save_knowledge_path"]
    # ... your logic ...
    return updated_pool  # List[Dict[str, Any]]
```

Then call it with:

```python
KlgOptimizer(optimizer_type="my_optimizer", ...)
```
