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
**Purpose** Generate multiple Python generator modules per topic and, for each module, multiple CSVs that explore distinct hyper-parameter configurations. Each (topic, sample_index, hyper_param_index) triple becomes one dataset row with a shared reference QA pair.

**Algorithm**
- Load the prompt template from `data_generators/excel.md`, injecting the topic name, the requested sample index, and the total hyper-parameter count.
- Ask `seimei.llm.LLMClient` for a JSON payload containing the module source code plus the question/answer pair.
- Persist the module, then execute it once per hyper-parameter index (naming CSVs `topic_sample_hyper.csv`).
- Optionally run an iterative validation loop (`--enable-validation`) that re-prompts the LLM with structured feedback when execution or CSV checks fail.
- Batch the remaining samples (default 10 at a time) and run each batch concurrently via `asyncio.gather`, checkpointing `<exp_dir>/dataset.json` after every batch so runs can resume safely.
- Save the validated modules under `<exp_dir>/python/`, store the generated CSVs beneath `<exp_dir>/csv/`, and append a row per CSV to `dataset.json`.

**Key Arguments**
- `--exp-dir` root directory for artifacts (defaults to `exp1`).
- `--model`, `--temperature`, and `--llm-kw` (repeat `--llm-kw key=value` to forward multiple OpenAI-style parameters) configure the dataset LLM client.
- `--n-samples-per-topic` controls how many Python modules to request per topic.
- `--n-hyper-params` sets the number of CSVs (hyper-parameter variations) generated from each module.
- `--batch-size` toggles how many samples are generated concurrently; larger values trade latency for higher LLM parallelism (default 10).
- `--topics` optional list of topic names (defaults to the bundled five).
- `--topics-path` optional JSON file containing an array of topics; falls back to the bundled list when omitted.
- `--prompt-path`, `--python-dir`, `--csv-dir`, `--output-file-path` override file locations.
- `--prefer-subprocess`, `--exec-timeout` select execution mode and timeout for generated scripts.
- `--enable-validation` toggles the slower retry loop; `--max-attempts` only applies when this flag is set.

**Output Structure**
`<exp_dir>/dataset.json` — list of objects:
```json
[
  {
    "Topic": "ecommerce_orders",
    "SampleIndex": 1,
    "HyperParamIndex": 2,
    "Question": "...?",
    "CorrectAnswer": "A. ...",
    "PythonPath": "exp1/python/ecommerce_orders_1.py",
    "CSVPath": "exp1/csv/ecommerce_orders_1_2.csv",
    "CSVPreview": [
      ["topic", "sample_id", "..."],
      ["ecommerce_orders", "...", "..."]
    ]
  }
]
```

**Usage Example**
```bash
python -m seimei.eval.generate_dataset_excel \
  --n-samples-per-topic 1 \
  --n-hyper-params 4 \
  --exp-dir exp4 \
  --topics-path seimei/eval/data_generators/excel_topics.json \
  --model gpt-5-nano
```

**Notices**
- By default the script runs a single-shot generation pass; add `--enable-validation` when you need automatic retries with feedback.
- Resume runs by reusing the same `--exp-dir`: any sample that already has all requested hyper-parameter rows in `dataset.json` is skipped automatically, so only unfinished work is regenerated.
- Repeat `--llm-kw key=value` to forward multiple parameters (e.g., `--llm-kw top_p=0.8 --llm-kw presence_penalty=0.2`).
- The generated modules must remain pure Python (no network access, no writes outside the requested CSV path).
- All paths recorded in `dataset.json` mirror the `--exp-dir` argument.

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
- `--model`, `--temperature`, and repeated `--llm-kw key=value` configure the inference LLM; `--rm-kw` forwards optional rmsearch parameters.

**Output Structure**
Each dataset record gains:
- `Output` — final SEIMEI answer.
- `Log` — full message history including agent emissions.
- `RunId` — UUID written by the orchestrator.
- `Usage` — token accounting (if available).

**Usage Example**
```bash
python -m seimei.eval.inference \
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
- `--model`, `--temperature`, and repeated `--llm-kw key=value` configure the judging LLM.
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
python -m seimei.eval.evaluation \
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
`data_generators/excel.md` is a lightweight template consumed by `generate_dataset_excel.py`. It surfaces the topic, the current sample index, total samples, and hyper-parameter count so humans can tweak instructions without editing code. Review it before launching large batches to ensure the guidance matches the evaluation goals.
