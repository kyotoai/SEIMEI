# Sampling (`seimei/train/sampling.py`)

This document explains how to run inference + scoring with the sampling runner.

## Quick Start

### 1. Import and run

```python
from pathlib import Path
from seimei.train.sampling import Sampling

runner = Sampling(
    dataset_path=Path("exp11_plasma_gkv_v5/dataset.json"),
    output_path=Path("exp11_plasma_gkv_v5/train_v6_results.json"),
    llm_model_name="/workspace/gpt-oss-20b",
    llm_url="https://your-llm-endpoint/v1",
    rm_url=None,
)

results = runner.run()
print(len(results))
```

### 2. Async usage

```python
import asyncio
from seimei.train.sampling import Sampling

async def main():
    runner = Sampling(max_problems=10)
    records = await runner.run_async()
    print("done", len(records))

asyncio.run(main())
```


## Output File Format

The result file is JSON written to `output_path`.

Top-level shape:

```json
{
  "schema_version": 2,
  "saved_at": "2026-02-19T09:30:00+0000",
  "summary": { "...": "aggregate metrics" },
  "detail": [
    {
      "id": "...",
      "problem": "...",
      "csv_path": "...",
      "patch_path": "...",
      "no_klg_trials": [
        {
          "trial": 1,
          "run_name": "train_v6_0000_no_klg_r1",
          "run_id": "run-...",
          "score": 6.0,
          "score_feedback": "...",
          "output": "..."
        }
      ],
      "klg_trials": [
        {
          "trial": 1,
          "run_name": "train_v6_0000_klg_r1",
          "run_id": "run-...",
          "score": 8.0,
          "score_feedback": "...",
          "output": "...",
          "knowledge_used": [
            {"step": 1, "id": 42, "text": "..."}
          ]
        }
      ],
      "no_klg_mean_score": 6.0,
      "klg_mean_score": 8.0,
      "no_klg_max_score": 6.0,
      "klg_max_score": 9.0,
      "no_klg_min_score": 6.0,
      "klg_min_score": 7.0,
      "mean_score_improvement": 2.0,
      "max_score_improvement": 3.0,
      "min_score_improvement": 1.0
    }
  ],
  "run_cache": {
    "<entry_id>::<run_name>": {
      "entry_id": "...",
      "dataset_index": 0,
      "run_name": "train_v6_0000_klg_r1",
      "run_id": "run-...",
      "output": "...",
      "saved_at": "2026-02-19T09:30:00+0000",
      "msg_history": [
        {"role": "user", "content": "..."}
      ]
    }
  },
  "run_cache_count": 1
}
```

### `summary` fields

`summary` includes:

- `total_problems`
- `overall_mean_score_improvement`
- `max_mean_score_improvement`
- `min_mean_score_improvement`
- `no_klg_overall_mean`
- `klg_overall_mean`
- `no_klg_max_mean`
- `klg_max_mean`
- `no_klg_min_mean`
- `klg_min_mean`
- `mean_win_loss_tie`
- `max_mean_win_loss_tie`
- `min_mean_win_loss_tie`
- `mean_no_klg_vs_mean_klg`
- `max_no_klg_vs_max_klg`
- `min_no_klg_vs_min_klg`

### Resume/cache behavior

- If `resume=True`, existing `detail` and `run_cache` are loaded from `output_path`.
- Completed problem IDs are skipped.
- Completed run keys (`entry_id::run_name`) are reused from `run_cache`.
- File is saved incrementally during execution, not only at the end.

## Detailed Settings

Constructor: `Sampling(...)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `dataset_path` | `Path` | `exp11_plasma_gkv_v5/dataset.json` | Input dataset JSON array. |
| `output_path` | `Path` | `exp11_plasma_gkv_v5/train_v6_results.json` | Output results JSON path. |
| `llm_model_name` | `str` | `/workspace/gpt-oss-20b` | LLM model name passed to Seimei. |
| `llm_url` | `str \| None` | `https://94sownu2ebkgwm-8000.proxy.runpod.net/v1` | LLM endpoint base URL. |
| `rm_url` | `str \| None` | `None` | RM endpoint base URL. |
| `batch_size` | `int` | `100` | Max concurrent request workers. |
| `n_no_klg_trials` | `int` | `3` | Trials per problem without knowledge retrieval. |
| `n_klg_trials` | `int` | `7` | Trials per problem with knowledge retrieval. |
| `top_n_sample_klg` | `int` | `5` | Knowledge candidate top-N before sampling. |
| `distribution_decay_rate` | `float` | `0.5` | Decay used in ranked knowledge sampling. |
| `random_klg_sampling_rate` | `float` | `0.2` | Random sampling probability for knowledge. |
| `klg_sample_mode` | `str` | `llm` | Knowledge search mode (`llm` or `rm`). |
| `max_problems` | `int \| None` | `None` | Optional cap on number of dataset items to run. |
| `klg_pool_load_path` | `Path` | first existing of `exp11_plasma_gkv_v5/knowledge_v6.csv`, then `seimei/train/default_knowledge.csv` | Knowledge pool CSV to load. |
| `enable_update_klg_pool` | `bool` | `True` | Enable post-scoring knowledge rewrite. |
| `final_klg_pool_save_path` | `Path` | same as `klg_pool_load_path` | Where updated knowledge CSV is written. |
| `patch_dir` | `Path` | `exp11_plasma_gkv_v5/patch_files` | Patch text directory (`patch{idx}.txt` by default). |
| `workspace_root` | `Path` | `exp11_plasma_gkv_v5/_workspace_copies` | Per-worker temporary workspaces. |
| `schema_version` | `int` | `2` | Written into output JSON. |
| `save_log` | `bool` | `False` | Save stdout/stderr to timestamped log file. |
| `log_dir` | `Path \| None` | `exp11_plasma_gkv_v5/_logs` | Log output directory. |
| `resume` | `bool` | `True` | Resume from previous output/run cache. |
| `run_name_prefix` | `str` | `train_v6` | Prefix used in run names. |

### Dataset expectations

Each dataset item should be a dict. Typical fields used:

- `problem` or `Question` (task text)
- `answer` or `CorrectAnswer` (reference answer)
- `CSVPath` (optional context path)
- `patch_file` or `patch_path` (optional override patch path)

If patch path is not provided, default patch name is `patch{dataset_index}.txt` under `patch_dir`.

### Prompt and helper files

- Prompt constants: `seimei/train/sampling_prompts.py`
- Utilities/checkpointing/format helpers: `seimei/train/sampling_utils.py`

