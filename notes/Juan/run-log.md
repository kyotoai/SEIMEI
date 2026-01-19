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
