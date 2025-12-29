# Run log (train_v3_eval.py / train_v3_TEST.py)
- Updated: 2025-12-12

- Motivation: train_v3.py produced inconsistent scores; suspected causes were randomness, brittle knowledge hints, and a rigid base prompt.

- Idea: Introduced `_build_adaptive_notes` to inject small, adaptive hints that preserve correct prior reasoning while nudging stalled runs.

- Entry 1: Root cause found—adaptive hints used wrong keys (`diurnal_amplitude_PM25`, `anomaly_count`) and broke the rule `total_rows = sensor_count × 24` instead of using the canonical keys (`seasonality_strength`, `anomaly_rate`, `urban_density`, `sensor_count`).

- Entry 2: Stability tweak—confirmed `DEFAULT_FINAL_RERUNS` = 8 in `exp8_csv_small/train_v3_eval.py` to smooth base vs knowledge comparisons.

- Entry 3: Knowledge fix—added a concise hint in `exp8_csv_small/train_v3_eval_adaptive.json` enforcing the canonical keys and the row/sensor relationship.

- Entry 4: Rerun hygiene—ran `train_v3_eval.py` from repo root with correct paths and low `--batch-size` to avoid file-not-found and API timeouts; used `--dataset-path exp8_csv_small/dataset.json`.

- Entry 5: Eval results—`train_v3_eval_baseline.py` finished 9/9 problems. Overall base mean 3.9567 → final mean 4.2922 (Δ +0.3356), win/loss/tie = 6/3/0. Per-problem base→final: [1.62→2.25], [4.5→4.25], [2.5→2.38], [7.62→8.12], [2.75→3.75], [3.0→2.0], [6.62→7.88], [4.5→5.12], [2.5→2.88]. Example rerun: “[eval 8] reruns avg base=2.5 knowledge=2.88 (Δ 0.38)”. Results in `exp8_csv_small/train_v3_eval_adaptive.json`.

- Entry 6:  Introduced train_v3_eval_adaptive.py and train_v4_eval_adaptive.py to include adaptive notes when passing the prompt to the agent. 

- Entry 7: Eval v3 adaptive (`train_v3_eval_adaptive.json`): 9/9 complete; overall base mean 4.3178 → final 3.7233 (Δ −0.5944), win/loss/tie = 2/7/0; most problems regressed.

- Entry 8: Eval v4 adaptive (`train_v4_eval_adaptive.json`): 9/9 complete; overall base mean 4.3478 → final 4.2222 (Δ −0.1256), win/loss/tie = 5/4/0; slight overall dip.

- Parameters used: DEFAULT_N_KNOWLEDGE_STEPS=7, DEFAULT_KNOWLEDGE_PER_STEP=3, DEFAULT_FINAL_RERUNS=8.

# Run log (generate_dataset_excel.py)
- 2025-12-25: Prompt became longer; LLM replies were truncated mid-JSON. Fixed by extracting the longest `{...}` span in `extract_json_object` and allowing larger `max_tokens`. Verified with gpt-4o-mini and gpt-5-nano single-topic runs (exp10_csv_small_gpt4 / gpt5).
- 2025-12-25: Added Quality checklist guards in `excel.md` (valid Python, no stray backslashes/indent errors; cast numpy scalars/bools to int/float/bool before `json.dumps`; use `rng.integers` instead of `randint`; keep code concise/deterministic).
- 2025-12-25: Topics bug: passing `--topics seimei/eval/data_generators/excel_topics_middle.json` treated the file path as the topic. Fixed by using `--topics-path` (JSON array) and smaller batch sizes to avoid prompt bloat.
- 2025-12-25: Baseline dataset with gpt-4o-mini written to `exp10_csv_small_gpt4`; comparison run with gpt-5-nano in `exp10_csv_small_gpt5`. Plan: use gpt-4o-mini as baseline, compare gpt-5-nano after successful single-topic checks; rerun failed topics with small batch and a few retries if needed.
- 2025-12-26: Small local models (e.g., deepseek-coder 1.3B via Ollama) could not consistently emit valid JSON for some topics (prefixed junk/empty `{}`); runs resulted in zero records. Conclusion: use larger/more compliant models (gpt-4o-mini/5) for this data generation; small models are too unreliable.
- 2025-12-26: Final conclusion for data-gen: small/edge models keep emitting malformed `{}`/garbage and fail guards even after retries; stick with OpenAI gpt-4o-mini/5 for production runs and use new guardrails to skip empty candidates.
