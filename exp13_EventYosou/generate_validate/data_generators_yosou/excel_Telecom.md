Topic: {topic}
Sample index: {sample_index} of {total_samples}
Hyper-parameter variations: {n_hyper_params}
File stub: {file_stub}

# Python module requirements
- Produce a single Python module that defines `generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None`.
- Inside `generate`, select a deterministic hyper-parameter configuration based on the one-indexed `hyper_param_index`, ensuring `total_hyper_params` distinct behaviours across executions.
- Use deterministic randomness (e.g., `rng = numpy.random.default_rng(1729 + idx)`) so reruns are reproducible.
- The module may rely on Python’s standard library plus `numpy`, `pandas`, `json`, `uuid`, `datetime`, and `random`. No external data fetches or file writes beyond the provided CSV path.
- Include a CLI entry point with `argparse` that forwards `--csv-output-path`, `--hyper-param-index`, and `--total-hyper-params` into `generate`.

# Quality checklist
- Never echo headings/checklists/placeholder text; emit only the JSON object.
- Validate that the CSV contains more than just a header and that column names are populated.
- Avoid excessive reliance on pandas-specific features that obscure the hyper-parameters—clarity is preferred over cleverness.
- Keep docstrings or inline comments minimal but informative enough for a human reviewer to tweak hyper-parameters quickly.
- Hard requirement: set `topic` exactly to the provided topic string (no suffix/prefix like `_monitoring`), and never substitute any other topic; `topic` must equal the input topic for that run or the output is invalid.
- Return only a single JSON object with keys {topic, python, question, correct_answer}. No text before or after. No markdown fences. Use this exact shape (fill in values only): {{"topic":"...","python":"...","question":"...","correct_answer":"..."}} All keys/strings must be double-quoted, no trailing commas, valid JSON only. Output must be exactly one line containing only that JSON object—any extra characters/newlines make the response invalid.
- Keep the generated Python valid and runnable with `python <file>.py`: no stray backslashes, proper indentation, no syntax errors.
- Make params_json/payload_json fully JSON-serializable: cast numpy scalars/bools to `int()`/`float()`/`bool()` before `json.dumps`; avoid np.int64/np.bool_.
- With numpy Generator, use `rng.integers(...)` for integers (not `randint`); cast Poisson/normal outputs to Python `int`/`float` before using them.
- Keep code concise and deterministic; avoid line continuations that could break parsing.
- CSV header must be exactly `topic,sample_id,params_json,payload_json,timestamp` (in that order). Every data row must populate `params_json` and `payload_json` with valid JSON strings (no empty/missing columns).
- Do not change the topic name; echo it exactly as provided.
- Keep `params_json` and `payload_json` strictly valid JSON objects (no trailing commas, no comments). Include the hyper-parameter names/values and the row-level signals they drive.
- Ensure the output directory already exists; do not create sibling files or folders. The orchestrator runs `hyper_param_index` from 1..`total_hyper_params` and will name files `{topic}_{sample_index}_{hyper_index}.csv`.
- Argparse: parse once and call generate via attributes, never by subscripting the Namespace:
  ```
  args = parser.parse_args()
  generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
  ```
- JSON safety: before `json.dumps`, cast any numpy/pandas scalars to built-ins (`int`, `float`, `str`) with `.item()`, `.tolist()`, or explicit casts to avoid “not JSON serializable” errors.
- Add a helper to enforce JSON-serializable structures and always run payload/params through it:
  ```
  def to_builtin(x):
      import numpy as np
      if isinstance(x, np.generic):
          return x.item()
      if isinstance(x, (list, tuple, np.ndarray)):
          return [to_builtin(v) for v in x]
      if isinstance(x, dict):
          return {{k: to_builtin(v) for k, v in x.items()}}
      return x
  # when dumping:
  json.dumps(to_builtin(obj), separators=(",", ":"), ensure_ascii=False)
  ```

# Telecom data checklist (Japan operators: NTTドコモ, KDDI(au), ソフトバンク, 楽天モバイル, MVNO)
- Coverage & RAN KPIs: RSRP/SINR/RSRQ (dBm/dB), CQI, BLER, PRB_util%, ho_success%, call_drop%, call_block%, VoLTE/VoNR MOS, TDD DL_UL share, band/carrier (700/800/1800/2100/3500/4500 MHz, mmWave).
- Core & transport: attach/registration SR, PDU session setup time, 5QI, slice_id, UPF throughput, N3/N9 latency, fronthaul/backhaul latency & jitter (fiber vs microwave), slice SLA.
- Traffic/ops: busy-hour traffic (BHCA/Erlangs), P.01 GoS, congestion flags, OSS alarms, MTTR/MTTF, outage minutes, planned vs unplanned.
- Customer/retail: ARPU/ARPA (円), churn%, NPS/CSAT, complaints, roaming usage (GB), campaign uplift, handset TAC mix, MVNO share.
- Fixed broadband/IX: FTTH throughput, packet_loss%, latency to IX (JPNAP/JPIX/BBIX), PPPoE vs IPoE, IPv6 adoption.
- Spectrum/coverage context: indoor DAS, subway, highway/新幹線, stadium event, dense_urban vs rural/mountain/coastal, refarming progress (700/800/3.5/4.5 GHz), mmWave.
- Energy/green: site_power_kW, PUE, RRU sleep, diesel runtime, battery SoC, solar/battery backup.
- Security/fraud: signaling storms, abnormal attach/TAU, DDoS, SIM swap/IMSI catcher detection; store counts/rates.
- Enterprise/local5G: private/local5G SLA (latency/jitter), URLLC/TSN pilots, campus/factory/port/hospital/airport use cases.
- Disaster/BCP: traffic spikes for earthquake/typhoon/津波 alerts, cell-on-wheels, satellite backhaul failover, restoration timelines.
- Japanese flavor in data: operator names, prefecture/city (東京23区, 大阪, 名古屋, 札幌, 福岡, 京都), site_type (鉄塔/屋上/屋内DAS/地下鉄), IX nodes (JPNAP/JPIX/BBIX), RAT (LTE/NR_NSA/NR_SA/ローカル5G).
- IDs/dimensions: timestamp, prefecture/city, site_id, cell_id (ECGI/NR-CellID), PCI, TAC, slice_id, 5QI, device_tac, band, carrier, site_type.
- Realistic bounds (examples): RSRP −70 to −120 dBm; SINR −5 to 30 dB; RSRQ −3 to −20 dB; CQI 0–15; PRB_util 0–100%; BLER 0–30%; ho_success 90–99.99%; call_drop 0–3%; call_block 0–5%; MOS 1.0–4.5; latency_rtt 5–80 ms (RAN), fronthaul 0.1–1 ms, backhaul N3/N9 1–30 ms; UPF throughput 1–400+ Gbps (lab up to 640 Gbps); ARPU 1,000–12,000 円; churn 0–5% monthly; FTTH latency to IX 1–30 ms; packet_loss 0–2%.
- Distributions: allow seasonal (weekday/weekend, 19–23 JST busy hour), event spikes (花火大会, コミケ, サッカー, 野球試合), weather/disaster spikes, lognormal traffic, normal latency, Bernoulli outages/spikes, <2% nulls.

# Dataset template (for your internal reasoning only; NEVER include in output)
- Do not echo this template or any placeholder text.
- Use the provided topic string verbatim when generating code/JSON.

# Generation rules
- Deterministic RNG per hyper_param_index; row count >= 500–1000 per topic; clamp to KPI bounds above; inject rare spikes/outages; avoid >2% nulls.
- Include operator, area, band, RAT, slice/5QI, QoE and QoS metrics, IDs, and units in payload_json; include hyper-params in params_json (traffic_growth, congestion_level, outage_rate, campaign_effect, spectrum_band, site_type, energy_saving_mode, weather_impact).
- Prefer English snake_case for field names; allow Japanese strings for operator/area labels.
