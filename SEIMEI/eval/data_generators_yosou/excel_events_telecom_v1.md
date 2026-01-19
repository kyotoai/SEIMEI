# excel_events_telecom_v1.md — LLM prompt (Japan telecom synthetic events + signal)

Topic: {topic}
Sample index: {sample_index} of {total_samples}
Hyper-parameter variations: {n_hyper_params}
File stub: {file_stub}

## Your task
Return **ONE JSON object** (no markdown fences) that will be used by a Python orchestrator to write
two Python modules and generate two CSV files per hyper-parameter index:

1) **events.csv** (ground-truth event table)
2) **signal.csv** (telecom KPI time series affected by those events)

This is a **toy but realistic** Japan telecom dataset (NTTドコモ / KDDI/au / ソフトバンク / 楽天モバイル).
The objective is later to train a model on (events_train.csv, signal_train.csv) and then, given signal_test.csv,
predict probabilistic properties of events_test.csv.

We do **NOT** expect perfect prediction. Your data must include confounders/noise so inference is uncertain.

---

## Output JSON schema (MUST follow exactly)

Keys:
- `topic` (string) must equal the topic shown above
- `features` (object)  # a compact description of schema + ranges + assumptions
- `python_features` (string)  # Python module source code to generate events CSV (see below)
- `python_signal` (string)    # Python module source code to generate signal CSV (see below)
- `question` (string)         # short question about the data
- `correct_answer` (string)   # concise answer explaining mapping between events and signals

No extra keys.
Return ONLY the JSON object and nothing else (no prose, no markdown fences).
The `topic` string MUST exactly match the Topic shown at the top (no substitutions).

---

## Python module requirements (both modules)
Each module must define:

```python
def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    ...
```

and also provide a CLI:

```python
if __name__ == "__main__":
    # parse --csv-output-path --hyper-param-index --total-hyper-params
```

Rules:
- Only use standard library + numpy + pandas.
- No network access. No file writes other than `csv_output_path`.
- Must be deterministic: `rng = np.random.default_rng(1729 + 1000*hyper_param_index)` (or similar).
- `hyper_param_index` must select **distinct behaviors** across runs (1..total_hyper_params).
- Argparse: parse once and call `generate` via attributes, never by subscripting the Namespace:
  ```
  args = parser.parse_args()
  generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
  ```
- Keep the generated Python valid and runnable with `python <file>.py`: no stray backslashes, proper indentation, no syntax errors.
- Do not create subfolders or extra files; write only to `csv_output_path`.
- Avoid mojibake: use UTF-8 strings for prefecture/city/operator; do not output garbled Latin-1 characters.
- Always import all used modules (e.g., argparse, datetime, numpy as np, pandas as pd). No undefined names.
- Avoid walrus operator `:=` (keep Python 3.8+ compatibility simple).
- When using datetime.timedelta, cast numpy integers to built-in int first (e.g., `int(rng.integers(...))`).
- Use exact `Asia/Tokyo` (no leading/trailing spaces).
- If you build JSON fields, ensure numpy scalars are converted to builtins before json.dumps.
- Do not emit any backslash-escaped newlines in code (no `\n` inside strings that represent code).
- Avoid NameError and KeyError: define all helpers you call; use consistent dict keys.

---

## events.csv schema (what python_features must output)
Write a UTF-8 CSV with columns:
- `event_id` (string unique)
- `event_type` (string; choose from: english_conversation, coding_meetup, hiking, cooking_class, live_concert, live_music, sports_game, sports_day, anime_convention, job_fair, university_lecture, train_station_rush)
- `prefecture` (e.g., 東京都, 大阪府, 京都府, 愛知県, 北海道, 福岡県,　宮城県)
- `city` (東京23区, 横浜, 大阪市, 京都市, 神戸市, 名古屋市, 札幌市, 福岡市, 仙台市)
- `region_type` (dense_urban, suburbs_urban, subway, airport, coastal, rural_mountain, indoor_mall, stadium_event, campus_ground, campus_hall)
- `start_time` (ISO8601, Asia/Tokyo)
- `end_time` (ISO8601, Asia/Tokyo)
- `duration_minutes` (int)
- `avg_attendance` (int)
- `event_quality` (popular, interesting, boring)
- `age_mean` (int; mean attendee age, e.g., 18–75)
- `male_pct` (int 0–100)
- `female_pct` (int 0–100; must sum to 100)

Also include optional descriptive columns if you want (venue_type, neighborhood_type), but keep the above required ones.
Do NOT include signal columns (operator, RSRP, SINR, throughput, etc.) in events.csv.
event_type must always be from the event_type list; region_type must always be from the region_type list (never swap them).

Time interval T:
- Choose a 7-day window, 15-min granularity (you decide concrete dates).
- Put events mostly on evenings/weekends in dense_urban/subway. Stadium_event can be large.
- Generate 10–20 events per topic (enough for basic statistics).

Plausibility constraints (use common sense):
- stadium_event: sports_game, live_concert (large attendance)
- stadium_event must have avg_attendance >= 100 (large halls/stadiums)
- popular events should have higher attendance (scaled by venue size)
- live_concert at stadium_event should be in the thousands (near capacity)
- live_music is a small-scale bar/live-house event (10–150 attendees)
- live_concert should have avg_attendance >= 100 (jazz/classical/rock venues)
- small-scale events (english_conversation, coding_meetup, cooking_class) should be 10–150
- university_festival/campus_crowd should use campus_ground or campus_hall
- sports_day should be on campus_ground (1–3 events per topic)
- concerts during university festivals should be on campus_hall or campus_ground
- subway: train_station_rush only
- airport: train_station_rush or job_fair (rare)
- dense_urban / indoor_mall: english_conversation, coding_meetup, job_fair, university_lecture, live_concert (smaller)
- coastal / rural_mountain: hiking (daytime), cooking_class (small)
- suburbs_urban: cooking_class, university_lecture, english_conversation
- anime_convention: indoor_mall or stadium_event
- hiking should NOT be in dense_urban or stadium_event

STRICT region/event compatibility (do not violate):
- subway: train_station_rush only
- airport: train_station_rush or job_fair only (rare)
- stadium_event: sports_game, live_concert, anime_convention only
- campus_ground: sports_day, university_lecture, live_music only
- campus_hall: university_lecture, live_music, live_concert only
- indoor_mall: anime_convention, job_fair, live_music, live_concert only
- dense_urban: english_conversation, coding_meetup, job_fair, live_music, live_concert only
- suburbs_urban: english_conversation, university_lecture, cooking_class only
- coastal: hiking or cooking_class only (daytime)
- rural_mountain: hiking only (daytime)

Age_mean guidance:
- sports_game, live_concert: 18–45 (younger skew)
- live_music: 20–50
- sports_day: 15–30

Gender balance guidance:
- omiai/matchmaking or nightlife social events: if event_quality is popular/interesting, keep male_pct and female_pct close to 50/50.
- if event_quality is boring, allow male_pct to skew higher than female_pct.
- job_fair, university_lecture: 18–30
- cooking_class, english_conversation: 25–55
- hiking: 30–65
- train_station_rush: broad 20–60

---

## signal.csv schema (what python_signal must output)
Write a UTF-8 CSV with columns:
- `timestamp` (ISO8601, Asia/Tokyo) 15-min grid
- `operator` (NTTドコモ, KDDI/au, ソフトバンク, 楽天モバイル)
- `prefecture`, `city`, `region_type`
- `site_id` (string; fake but consistent)
- RAN KPIs:
  - `RSRP_dBm` (float, -120 to -70)
  - `SINR_dB` (float, -5 to 30)
  - `RSRQ_dB` (float, -20 to -3)
  - `prb_util_pct` (float, 0 to 100)
- Traffic / QoE:
  - `dl_mbps` (float, 0.1 to 2000)
  - `ul_mbps` (float, 0.05 to 800)
  - `latency_ms` (float, 5 to 200)
  - `jitter_ms` (float, 0 to 80)
  - `packet_loss_pct` (float, 0 to 5)
  - `volte_mos` (float, 1.0 to 4.5)
- Ops counters:
  - `bhca` (int, 1_000 to 2_000_000)
  - `erlangs` (float, 0 to 60)
  - `call_drop_rate_pct` (float, 0 to 5)
  - `call_block_rate_pct` (float, 0 to 8)

How signal must reflect events:
- During an event, the **local city/region** shows increased dl/ul, PRB, BHCA/Erlangs, and sometimes higher latency/loss and lower MOS.
- Not every spike is an event: add **non-event anomalies** (maintenance, random congestion, weather) so inference is probabilistic.

IMPORTANT:
- python_signal must generate the SAME underlying events internally (same RNG + same generation logic as python_features),
  so the effect is consistent even though the module does not read events.csv.
- Clamp all numeric KPIs to their allowed ranges before writing (no out-of-range values).
- Use exact prefecture/city labels from the events generator (no extra or unknown labels).

---

## Modeling note (how validation will be done)
Assume a validator will score **per (city, time_bin)**:
- `p_event_present` (probability event is happening) using log-loss/Brier/calibration.
Optionally it will score `event_type` probability distribution.

So your generated signals should make events *detectable* but not perfectly.

---

## What to put in `features`
Keep it short and structured, e.g.:
- `time_start`, `time_end`, `granularity_min`
- `cities`, `operators`
- `kpis_in_signal`
- `event_types`
- `event_to_signal_mapping_summary`
- `non_event_confounders`
