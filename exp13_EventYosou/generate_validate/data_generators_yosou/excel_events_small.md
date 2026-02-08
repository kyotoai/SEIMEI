# excel_events_small.md — compact prompt for small models

Topic: {topic}
Sample index: {sample_index} of {total_samples}
Hyper-parameter variations: {n_hyper_params}
File stub: {file_stub}

Return ONE JSON object (no markdown, no extra text). Keys:
- "topic" (must exactly match Topic above)
- "features"
- "python_features"
- "python_signal"
- "question"
- "correct_answer"

Python requirements (both modules):
- Define generate(csv_output_path, hyper_param_index, total_hyper_params).
- CLI parses --csv-output-path --hyper-param-index --total-hyper-params and calls generate() using args attributes (no subscripting).
- For python_signal, also parse --events-csv-path and pass it into generate().
- Use only stdlib + numpy + pandas.
- Deterministic rng: np.random.default_rng(1729 + 1000*hyper_param_index).
- No extra files/folders; write only csv_output_path.
- Avoid typing annotations like Dict/List unless you import from typing.
- Cast numpy ints to int before timedelta.
- Convert numpy scalars before json.dumps.
- No backslash-escaped newlines in code.
- python_signal must read events.csv from --events-csv-path (do not re-generate events).
- For python_signal, use generate(csv_output_path, hyper_param_index, total_hyper_params, events_csv_path).

events.csv schema (python_features must output):
columns: event_id, event_type, prefecture, city, region_type, start_time, end_time,
duration_minutes, avg_attendance, event_quality, age_mean, male_pct, female_pct

event_type options:
english_conversation, omiai_matchmaking, coding_meetup, hiking, cooking_class, live_concert, live_music,
sports_game, sports_day, anime_convention, job_fair, university_lecture, train_station_rush

prefecture options (use exact Kanji):
東京都, 大阪府, 京都府, 愛知県, 北海道, 福岡県, 宮城県

city options (use exact Kanji):
東京23区, 大阪市, 京都市, 名古屋市, 札幌市, 福岡市, 仙台市

region_type options:
dense_urban, suburbs_urban, subway, airport, coastal, rural_mountain,
indoor_mall, stadium_event, campus_ground, campus_hall

STRICT region/event compatibility:
- subway: train_station_rush only
- airport: train_station_rush or job_fair only
- stadium_event: sports_game, live_concert, anime_convention only
- campus_ground: sports_day or university_lecture only
- campus_hall: university_lecture or live_music or live_concert only
- indoor_mall: anime_convention or job_fair or live_music or live_concert or omiai_matchmaking only
- dense_urban: english_conversation or coding_meetup or job_fair or live_music or live_concert or omiai_matchmaking only
- suburbs_urban: english_conversation or university_lecture or cooking_class or omiai_matchmaking only
- coastal: hiking or cooking_class only (daytime)
- rural_mountain: hiking only (daytime)

Attendance rules:
- live_music, english_conversation, omiai_matchmaking, coding_meetup, cooking_class: 10–150
- live_concert: >= 100 (stadium_event: thousands)
- sports_game: >= 100
- sports_day: 10–300

events.csv must NOT include signal fields like timestamp/operator/RSRP/etc.
Generate 10–20 events in a 7-day window, 15-min granularity. Evenings/weekends for big events.
Target class balance: aim for 10–35% of bins having an event present per city.
To control prevalence: keep durations 30–180 min and avoid overlapping events in the same city.

signal.csv schema (python_signal must output):
columns: timestamp, operator, prefecture, city, region_type, site_id,
operator must be exactly: NTTドコモ, KDDI/au, ソフトバンク, 楽天モバイル
RSRP_dBm, SINR_dB, RSRQ_dB, prb_util_pct, dl_mbps, ul_mbps,
latency_ms, jitter_ms, packet_loss_pct, volte_mos, bhca, erlangs,
call_drop_rate_pct, call_block_rate_pct

Ranges (clamp):
- RSRP_dBm -120..-70, SINR_dB -5..30, RSRQ_dB -20..-3, prb_util_pct 0..100
- dl_mbps 0.1..2000, ul_mbps 0.05..800
- latency_ms 5..200, jitter_ms 0..80, packet_loss_pct 0..5, volte_mos 1.0..4.5
- bhca 1000..2000000, erlangs 0..60
- call_drop_rate_pct 0..5, call_block_rate_pct 0..8

Signal must reflect events:
- During event time in matching city/region, increase traffic and PRB, worsen latency/loss, reduce MOS.
- Add some non-event anomalies.
- python_signal must read the SAME events.csv from --events-csv-path and use it to drive effects.
- Scale event impact by sqrt(attendance) with a cap; larger events are stronger but not extreme.
- Use 3–5 cities and generate a full 7-day 15-min grid per city.
- Vary baseline KPI levels per city/operator.
- Include a low-probability regime: many quiet bins with minimal anomalies so low-p calibration has support.
- Add diurnal baselines (day/night load curve) plus small day-level drift per city.

Coupling guidance:
- Scale PRB/dl/ul bumps with avg_attendance (bigger events → bigger bumps).
- Apply MOS drop + latency increase only during event bins.
- Keep non-event anomalies smaller than event effects (30–50% of event magnitude).

Baseline stability:
- Keep base KPI levels moderate (prb_util 30–60, dl 80–300, ul 15–80, latency 15–40).
- Avoid saturating min/max ranges except during large events.

features: keep brief (time_start, time_end, granularity_min, cities, operators, event_types, mapping, confounders).
question/correct_answer: short and focused on how events drive signal.

Topic-specific hard constraints (must follow):
- omiai_matchmaking_social_evening: all events only english_conversation or omiai_matchmaking.
- omiai_matchmaking_social_evening: target event prevalence 10–35% of bins per city.
- stadium_event_congestion: all events only sports_game or live_concert; region_type stadium_event.
- stadium_event_congestion: target event prevalence 5–25% of bins per city.
- university_festival_campus_crowd: all events only university_lecture, english_conversation, job_fair, live_music, live_concert, sports_day; region_type includes campus_ground or campus_hall.
- university_festival_campus_crowd: target event prevalence 10–35% of bins per city.
