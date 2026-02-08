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

## 2026-01-07
- Continued exp10/exp12 dataset generation experiments; gpt-4o-mini kept as baseline and gpt-5-nano runs started for telecom topics.
- Prompt updates: restored topic/sample index/total samples/hyper-params/file stub header; added stricter JSON-only output and telecom checklist.
- Observed recurring failures: topic mismatches, JSON truncation, numpy scalar JSON errors, argparse Namespace misuse; added guidance to cast numpy types to builtins in generated scripts.
- Ran train_v4_eval_sample2 via nohup on server; PID 39002, logs in server.log.
- Latest eval sample2 summary: schema_version=2, total_problems=6, mean_score_improvement=-2.285, overall_base_mean=4.6377, overall_final_mean=2.3527, win_loss_tie=1/0/5.
- Logged interruptions/timeouts to api.openai.com during dataset generation.

## 2026-01-09

- Running eval v4 sample 2 script...
Started in background (PID 11248). Logs: /Users/juan/Desktop/Git/SEIMEI_/exp12_telecom_gpt5_v1/server_trainv4.log
(kyotoai) 
-  "schema_version": 2,
  "saved_at": "2026-01-09T10:09:24+0100",
  "summary": {
    "total_problems": 51,
    "mean_score_improvement": -1.8449,
-  "overall_base_mean": 8.0827,
    "overall_final_mean": 6.2378,
    "win_loss_tie": {
      "win": 5,
      "tie": 0,
      "loss": 46

       "schema_version": 2,
  "saved_at": "2026-01-09T11:05:41+0100",
  "summary": {
    "total_problems": 7,
    "mean_score_improvement": -0.2371,
      "overall_base_mean": 7.9529,
    "overall_final_mean": 7.7157,
    "win_loss_tie": {
      "win": 1,
      "tie": 3,
      "loss": 3

## 2026-01-16

First output of generate_dataset_excel_telecom
Issues: we need to inforce plausability constraints 

event_id,event_type,prefecture,city,region_type,start_time,end_time,duration_minutes,avg_attendance,event_quality,male_pct,female_pct
EVT_001,english_conversation,東京都,東京23区,stadium_event,2024-07-02T18:00:00+09:00,2024-07-02T21:00:00+09:00,180,36965,boring,51,49
EVT_002,coding_meetup,東京都,東京23区,stadium_event,2024-07-04T19:00:00+09:00,2024-07-04T21:30:00+09:00,150,42845,interesting,63,37
EVT_003,hiking,大阪府,大阪市,dense_urban,2024-07-06T16:00:00+09:00,2024-07-06T18:00:00+09:00,120,56912,popular,65,35

dataset.json looks like
[
  {
    "Topic": "sports_game_weekend_peak",
    "SampleIndex": 1,
    "HyperParamIndex": 1,
    "Question": "How are events reflected in the signal data, and what does the mapping imply for model training?",
    "CorrectAnswer": "Events cause localized uplifts in signal metrics (higher dl/ul, PRB usage, BHCA/Erlangs) for the matching city/prefecture and region_type within the event window. The signal data also includes non-event anomalies to keep inference probabilistic. Thus, the model must learn probabilistic associations rather than deterministic ones, aided by timing and regional cues.",
    "FeaturesJSON": {
      "time_start": "2024-11-01 00:00 JST",
      "time_end": "2024-11-08 00:00 JST",
      "granularity_min": 15,
      "cities": [
        "東京23区",
        "横浜",
        "大阪市",
        "京都市",
        "神戸市",
        "名古屋市",
        "札幌市",
        "福岡市",
        "仙台市"
      ],
      "operators": [
        "NTTドコモ",
        "KDDI/au",
        "ソフトバンク",
        "楽天モバイル"
      ],
      "kpis_in_signal": [
        "RSRP_dBm",
        "SINR_dB",
        "RSRQ_dB",
        "prb_util_pct",
        "dl_mbps",
        "ul_mbps",
        "latency_ms",
        "jitter_ms",
        "packet_loss_pct",
        "volte_mos",
        "bhca",
        "erlangs",
        "call_drop_rate_pct",
        "call_block_rate_pct"
      ],
      "event_types": [
        "english_conversation",
        "coding_meetup",
        "hiking",
        "cooking_class",
        "live_concert",
        "sports_game",
        "anime_convention",
        "job_fair",
        "university_lecture",
        "train_station_rush"
      ],
      "event_to_signal_mapping_summary": "Events create localized uplifts in DL/UL, PRB utilization, BHCA/Erlangs and occasional latency/jitter increases for matching city/prefecture and region_type. Non-event anomalies (maintenance/weather) are injected to keep inference probabilistic.",
      "non_event_confounders": "Maintenance windows, weather disturbances, random congestion spikes."
    },
    python paths...
  }

## 2026-01-17

- Tightened telecom prompt and validation for v1 generator: added age_mean, stricter event/region plausibility, attendance scaling (small events 10–150, stadium concerts >=1000), and topic alignment checks.
- Added auto-fixes for LLM code issues (argparse Namespace subscripting, `PD.` typo), and syntax pre-checks before execution.
- Added mojibake detection + whitelist validation for prefecture/city/operator to reject garbled labels.
- Added auto-cleanup on failed runs to delete bad CSVs/dir artifacts; prevented `.csv` directories from blocking outputs.
- Introduced sports_day event type and initial festival rules.

## 2026-01-18

- Reduced to 3-topic debug set for consistency; added stricter topic-alignment sets (concert-only topics, stadium-only topics, campus festival-only topics).
- Added stricter signal range checks and schema mismatch detection (events vs signal).
- Added campus-specific region_types (campus_ground/campus_hall) and directed university festival events to school grounds/halls.
- Added kanji/kana validation to block non-Japanese prefecture/city outputs.
- Implemented real batching with async semaphore; improved robustness to partial failures.

## 2026-01-21

- Added automatic `metrics_summary.csv` aggregation in `exp13_EventYosou/eval` after each validation run.
- First-event experiment findings: strong class imbalance (stadium positive_rate ~0.91, university ~0.03) and poor calibration (ECE up to ~0.95).
- Baseline `logloss_event_present` worse than constant base-rate for omiai/university; indicates weak signal-to-event coupling and noisy confounders.
- Prompt updates planned: enforce target event prevalence (10–35% bins per city), shorter durations, stronger attendance-scaled signal bumps, and stable KPI baselines.
- Iteration labeling: current plots correspond to the second experiment iteration; third iteration will add signal-coupling improvements and revised baselines.

### Challenges by Iteration (EventYosou v1)

Iteration 1:
- LLM-generated Python modules exhibited syntax errors and missing imports.
- Mojibake appeared in prefecture/city/operator labels.
- Schema mix-ups (events vs signal columns) occurred under retries.
- Topic alignment was violated under strict constraints.

Iteration 2 (current plots):
- Class imbalance remains severe for some topics (rare vs near-always events), hurting calibration.
- Signal-to-event coupling remains weak for some scenarios (log-loss worse than base-rate).
- Some event/region plausibility violations persist under heavy retries.

## 2026-01-21 (run log)

- Data generation per topic takes around 4 attempts when run sequentially; example success:

```text
[topic=omiai_matchmaking_social_evening] attempt 4/7 requesting LLM...
[topic=omiai_matchmaking_social_evening] wrote modules: omiai_matchmaking_social_evening_1_features.py, omiai_matchmaking_social_evening_1_signal.py
[topic=omiai_matchmaking_social_evening] run hyper 1/1
[topic=omiai_matchmaking_social_evening] completed sample 1
```

- Common failures are syntax/argparse errors in the signal module; examples:

```text
[topic=omiai_matchmaking_social_evening] error: signal module failed for h=1: returncode=1
STDERR:
Traceback (most recent call last):
  File "/Users/juan/Desktop/Git/SEIMEI_/exp13_EventYosou/python/omiai_matchmaking_social_evening_1_signal.py", line 153, in <module>
    main()
  File "/Users/juan/Desktop/Git/SEIMEI_/exp13_EventYosou/python/omiai_matchmaking_social_evening_1_signal.py", line 150, in main
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params, args.events_csz_path)
                                                                                    ^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Namespace' object has no attribute 'events_csz_path'. Did you mean: 'events_csv_path'?
usage: omiai_matchmaking_social_evening_1_features.py [-h] --csv-output-path
                                                      CSV_OUTPUT_PATH
                                                      --hyper-param-index
                                                      HYPER_PARAM_INDEX
                                                      --total-hyper-params
                                                      TOTAL_HYPER_PARAMS
                                                      --events-csv-path
                                                      EVENTS_CSV_PATH
omiai_matchmaking_social_evening_1_features.py: error: the following arguments are required: --events-csv-path
```

```text
[topic=omiai_matchmaking_social_evening] error: signal module failed for h=1: returncode=1
STDERR:
Traceback (most recent call last):
  File "/Users/juan/Desktop/Git/SEIMEI_/exp13_EventYosou/python/omiai_matchmaking_social_evening_1_signal.py", line 141, in main
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params, args.events_csv_path)
  File "/Users/juan/Desktop/Git/SEIMEI_/exp13_EventYosou/python/omiai_matchmaking_social_evening_1_signal.py", line 33, in generate
    base_date = min(pd.to_datetime(events['start_time']).min(), pd.Timestamp.utcnow())
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/tslibs/timestamps.pyx", line 387, in pandas._libs.tslibs.timestamps._Timestamp.__richcmp__
TypeError: Cannot compare tz-naive and tz-aware timestamps
```

```text
[topic=stadium_event_congestion] wrote modules: stadium_event_congestion_1_features.py, stadium_event_congestion_1_signal.py
[topic=stadium_event_congestion] error: module syntax error:   File "/Users/juan/Desktop/Git/SEIMEI_/exp13_EventYosou/python/stadium_event_congestion_1_signal.py", line 162
    'region_type': info := CITY_INFO[city]['region_type'],
                        ^^
SyntaxError: invalid syntax
```

## 2026-01-21 (gpt-5-mini run)

- Model: gpt-5-mini, prompt: excel_events_small.md, topics=3.
- All three topics eventually completed; some initial timeouts and retries, but no fatal syntax failures at the end.
- Key errors observed during retries: API timeouts; pandas date_range `closed` keyword mismatch; sports_day attendance out of range; prevalence under target; bad walrus usage.

### Metrics summary (Iteration 4)
- omiai: pos_rate 0.061, logloss 1.082, Brier 0.431, ECE 0.611
- stadium: pos_rate 0.049, logloss 0.857, Brier 0.330, ECE 0.552
- university: pos_rate 0.091, logloss 0.921, Brier 0.362, ECE 0.562

### Conclusion
- Calibration remains weak for low-prevalence events (ECE ~0.55--0.61), and logloss is still high (~0.86--1.08).
- Positives are closer to target but still low, suggesting under-detection; next iteration should strengthen signal coupling or adjust event prevalence targets.

## 2026-01-21 (Iteration 5 planning)

- Guardrails still frequently trigger; focus on generating valid ranges at source instead of post-fix.
- Targets for next iteration:
  - Increase positive rates (10–20%) by shortening quiet periods and adding medium events.
  - Lower logloss_event_present by strengthening event-to-signal coupling (attendance-scaled bumps + city baselines).
  - Improve calibration (lower ECE) by adding controlled non-event anomalies and stabilizing baselines.
  - Ensure full 7-day 15-min grids per city inside the generator to avoid sparse-bin repair.

## 2026-01-21 (Iteration 5 generator changes)

- Added low-probability regime guidance (quiet bins), diurnal baselines, and city-level drift in prompts.
- Fallback generator now shapes event prevalence per city window, reduces omiai attendance, and scales KPI effects by topic.
- Signal fallback now adds diurnal baseline, small drift, and controlled anomalies to populate low-p bins.

<!-- #NOTES PROGRAM KITCHEN -->

## 2026-02-01 (Iteration 6 preds_breakdown)

- Ran preds_breakdown for stadium with RUN_ID=20260201_085014:
  `python3 exp13_EventYosou/preds_breakdown.py --labelspath exp13_EventYosou/eval/stadium_20260201_085014/labels_and_preds.csv --eventspath exp13_EventYosou/csv/stadium_event_congestion_1_20260201_085014_events_1.csv --output-dir exp13_EventYosou/eval/stadium_20260201_085014/preds_breakdown`
