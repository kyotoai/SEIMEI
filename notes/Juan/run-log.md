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
