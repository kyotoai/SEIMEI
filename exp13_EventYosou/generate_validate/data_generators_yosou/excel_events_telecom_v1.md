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
- For python_signal, also parse `--events-csv-path` and pass it into generate.
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
- python_signal MUST read `events.csv` from `--events-csv-path` (do not re-generate events).
- For python_signal, update CLI to parse `--events-csv-path` and pass it into generate.
- For python_signal, use generate(csv_output_path, hyper_param_index, total_hyper_params, events_csv_path).

---

## events.csv schema (what python_features must output)
Write a UTF-8 CSV with columns:
- `event_id` (string unique)
- `event_type` (string; choose from: english_conversation, omiai_matchmaking, coding_meetup, hiking, cooking_class, live_concert, live_music, sports_game, sports_day, anime_convention, job_fair, university_lecture, train_station_rush)
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
Target class balance: aim for 10–35% of bins having an event present per city.
To control prevalence: keep durations 30–180 min and avoid overlapping events in the same city.

Plausibility constraints (use common sense):
- stadium_event: sports_game, live_concert (large attendance)
- stadium_event must have avg_attendance >= 100 (large halls/stadiums)
- popular events should have higher attendance (scaled by venue size)
- live_concert at stadium_event should be in the thousands (near capacity)
- live_music is a small-scale bar/live-house event (10–150 attendees)
- live_concert should have avg_attendance >= 100 (jazz/classical/rock venues)
- small-scale events (english_conversation, omiai_matchmaking, coding_meetup, cooking_class) should be 10–150
- university_festival/campus_crowd should use campus_ground or campus_hall
- sports_day should be on campus_ground (1–3 events per topic)
- concerts during university festivals should be on campus_hall or campus_ground
- subway: train_station_rush only
- airport: train_station_rush or job_fair (rare)
- dense_urban / indoor_mall: english_conversation, omiai_matchmaking, coding_meetup, job_fair, university_lecture, live_concert (smaller)
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
- indoor_mall: anime_convention, job_fair, live_music, live_concert, omiai_matchmaking only
- dense_urban: english_conversation, coding_meetup, job_fair, live_music, live_concert, omiai_matchmaking only
- suburbs_urban: english_conversation, university_lecture, cooking_class, omiai_matchmaking only
- coastal: hiking or cooking_class only (daytime)
- rural_mountain: hiking only (daytime)

Age_mean guidance:
- sports_game, live_concert: 18–45 (younger skew)
- live_music: 20–50
- sports_day: 15–30

Gender balance guidance:
- omiai/matchmaking or nightlife social events: if event_quality is popular/interesting, keep male_pct and female_pct close to 50/50.
- omiai_matchmaking should be near 50/50.
- if event_quality is boring, allow male_pct to skew higher than female_pct.
- job_fair, university_lecture: 18–30
- cooking_class, english_conversation: 25–55
- hiking: 30–65
- train_station_rush: broad 20–60

---

## signal.csv schema (what python_signal must output)
Write a UTF-8 CSV with columns:
- `timestamp` (ISO8601, Asia/Tokyo) 15-min grid
- `operator` (MUST be exactly one of: NTTドコモ, KDDI/au, ソフトバンク, 楽天モバイル)
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
- Scale event impact by attendance with a soft cap (e.g., proportional to sqrt(attendance)) so large events are stronger but not extreme.
- Vary baseline KPI levels per city/operator so there is no single global threshold.
- Generate a full 7-day, 15-min grid for each city; use 3–5 cities per topic.
- Include a low-probability regime: keep many quiet bins with minimal anomalies so low-p calibration has support.
- Add diurnal baselines (day/night load curve) plus small day-level drift per city.

Coupling guidance:
- Scale PRB/dl/ul bumps with avg_attendance (bigger events -> bigger bumps).
- Apply MOS drop + latency increase only during event bins.
- Keep non-event anomalies smaller than event effects (30–50% of event magnitude).

Baseline stability:
- Keep base KPI levels moderate (prb_util 30–60, dl 80–300, ul 15–80, latency 15–40).
- Avoid saturating min/max ranges except during large events.

IMPORTANT:
- python_signal must read the SAME events.csv from `--events-csv-path` and use it to drive the signal effects.
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

---

## Topic-specific hard constraints (must follow)
For the three topics in excel_topics_3.json:

- **omiai_matchmaking_social_evening**
  - All events must be only: english_conversation or omiai_matchmaking.
  - Prefer dense_urban or indoor_mall or suburbs_urban; avoid campus_*.
  - Target event prevalence per city: 10–35% of bins.

- **stadium_event_congestion**
  - All events must be only: sports_game or live_concert.
  - Region_type must be stadium_event for those events.
  - Target event prevalence per city: 5–25% of bins (avoid always-on).

- **university_festival_campus_crowd**
  - All events must be only: university_lecture, english_conversation, job_fair, live_music, live_concert, sports_day.
  - Region_type must include campus_ground or campus_hall.
  - Target event prevalence per city: 10–35% of bins.
