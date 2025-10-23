SEIMEI Evaluation Toolkit
========================

Overview
--------
- `generate_dataset_excel.py` — orchestrates LLM-backed dataset synthesis for spreadsheet-style tasks.
- `inference.py` — runs the SEIMEI orchestrator over every question in the generated dataset and captures raw outputs.
- `evaluation.py` — asks an LLM judge to score each SEIMEI answer for correctness.
- `data_generators/excel.md` — prompt template provided to the dataset generator when requesting topic-specific code.

The three Python entry points form an end-to-end loop: create datasets → run SEIMEI inference → score the answers.

generate_dataset_excel.py
-------------------------
**Purpose** Generate a library of Python scripts + CSV files (one per topic) and capture matching question/answer pairs.

**Algorithm**
- Load the prompt template from `data_generators/excel.md` and inject the topic name plus numeric parameters.
- Call `seimei.llm.LLMClient` to request a JSON payload containing the module, question, and reference answer.
- Persist the module to disk, execute its `generate()` function (with a subprocess fallback), and validate the resulting CSV.
- If execution fails or validation detects issues (missing rows, invalid header, etc.), send structured feedback to the LLM and retry.
- Store the validated assets under `<exp_dir>/python/` and `<exp_dir>/csv/`, then append a record to `dataset.json`.

**Key Arguments**
- `--exp-dir` root directory for artifacts (defaults to `exp1`).
- `--model` and `--llm-kw` forwarded to `LLMClient`.
- `--n-samples-per-topic`, `--n-hyper-params` control dataset size and variation (required).
- `--topics` optional list of topic names (defaults to the bundled five).
- `--prompt-path`, `--python-dir`, `--csv-dir`, `--output-file-path` override file locations.
- `--max-attempts` retry budget per topic.
- `--prefer-subprocess`, `--exec-timeout` switch execution mode and cap runtime.

**Output Structure**
`<exp_dir>/dataset.json` — list of objects:
```json
[
  {
    "Topic": "ecommerce_orders",
    "Question": "...?",
    "CorrectAnswer": "A. ...",
    "PythonPath": "exp1/python/ecommerce_orders_001.py",
    "CSVPath": "exp1/csv/ecommerce_orders_001.csv",
    "CSVPreview": [
      ["topic", "sample_id", "..."],
      ["ecommerce_orders", "...", "..."]
    ]
  }
]
```

**Usage Example**
```bash
python seimei/eval/generate_dataset_excel.py \
  --n-samples-per-topic 3 \
  --n-hyper-params 3 \
  --exp-dir exp1 \
  --model gpt-5-nano 
```

**Notices**
- The generated modules must remain pure Python (no network access, no side effects outside the provided CSV path).
- Validation is intentionally conservative; repeated failures surface clear feedback to the LLM.
- All paths written to `dataset.json` mirror the `--exp-dir` argument, preserving relative layouts.

inference.py
------------
**Purpose** Execute SEIMEI on each dataset entry, preserving transcripts and answers for later scoring.

**Algorithm**
- Load `dataset.json` and resolve every `CSVPath`/`PythonPath` relative to `--exp-dir`.
- Build an agent configuration (defaults to the bundled `seimei/agents` directory), instantiate the SEIMEI orchestrator, and pass through runtime knobs (max steps, code execution guardrails, etc.).
- For each sample:
  - Prepare a user prompt containing the question, file paths, CSV preview, and a script excerpt.
  - Optionally prepend a system prompt.
  - Invoke SEIMEI with the composed messages, capturing the full message history and final answer.
- Persist the enriched records to `result.json`.

**Key Arguments**
- `--dataset-path`, `--result-path` override input/output (default `<exp_dir>/dataset.json` / `<exp_dir>/result.json`).
- `--agent-dir`, `--agent-file` add custom agents; `--allow-code-exec`/`--allowed-command` control shell access.
- `--system-prompt`, `--name`, `--max-steps`, `--max-tokens-per-question`, `--log-dir` mirror orchestrator options.
- `--preview-rows`, `--script-excerpt-chars` cap contextual snippets embedded in user prompts.
- `--llm-kw`, `--rm-kw` forward extra configuration to `LLMClient` or rmsearch.

**Output Structure**
Each dataset record gains:
- `Output` — final SEIMEI answer.
- `Log` — full message history including agent emissions.
- `RunId` — UUID written by the orchestrator.
- `Usage` — token accounting (if available).

**Usage Example**
```bash
python seimei/eval/inference.py \
  --exp-dir exp1 \
  --allow-code-exec \
  --allowed-command cat \
  --allowed-command python \
  --name seimei-excel-run
```

**Notices**
- Ensure the referenced CSV/Python files exist before running inference; the script does not regenerate missing assets.
- To minimise prompt size, adjust `--preview-rows` or `--script-excerpt-chars` when working with large datasets.
- All SEIMEI run artifacts are written to `--log-dir` (defaults to `./seimei_runs`).

evaluation.py
-------------
**Purpose** Apply an LLM judge to the inference outputs, labelling each as correct or incorrect.

**Algorithm**
- Load `result.json`, skipping records that already contain `Correctness` unless `--force` is passed.
- Resolve the generator script path and pull a configurable excerpt for context.
- Compose a judging prompt containing the question, reference answer, model output, script snippet, and optional CSV preview.
- Call `LLMClient` to obtain a JSON verdict (score/verdict/explanation), retrying when the response is malformed.
- Write updated results back to disk (default: in-place).

**Key Arguments**
- `--result-path`, `--output-path` locate the inference output and destination.
- `--model`, `--llm-kw` configure the judging LLM.
- `--script-excerpt-chars`, `--max-attempts` tune prompt size and retry behaviour.
- `--force` recomputes every label regardless of whether `Correctness` is already present.

**Output Structure**
Adds a `Correctness` object per entry:
```json
"Correctness": {
  "score": 1,
  "verdict": "correct",
  "explanation": "Matches the uplift calculation described in the generator."
}
```

**Usage Example**
```bash
python seimei/eval/evaluation.py \
  --exp-dir exp1 \
  --model gpt-5 \
  --llm-kw temperature=0 \
  --force
```

**Notices**
- The judge relies on the Python excerpt and CSV preview to reason about hyper-parameters—keep the dataset generator descriptive.
- Persisting to a new `--output-path` lets you compare multiple judging configurations side-by-side.
- Retries help recover from malformed JSON, but persistent format errors usually indicate that the system prompt needs adjustment.

Prompt Template
---------------
`data_generators/excel.md` is a lightweight template consumed by `generate_dataset_excel.py`. It exposes the topic, dataset dimensions, and recommended naming stub so human editors can adjust instructions without editing code. Review it before launching large batches to ensure the guidance matches the evaluation goals.
