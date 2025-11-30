import json
import csv
import uuid
from datetime import date, datetime, timedelta
import numpy as np


def _build_config(idx: int) -> dict:
    # One deterministic configuration per hyper-parameter index.
    if int(idx) <= 1:
        return {
            "name": "baseline_bias",
            "topic": "mining_site_safety_incidents",
            "start_date": "2024-01-01",
            "days": 25,
            "site_id": "MS-001",
            "incident_types": ["slip_trip", "fall", "equipment_malfunction", "near_miss"],
            "type_probs": [0.50, 0.25, 0.15, 0.10],
            "severity_levels": ["minor", "moderate", "major", "critical"],
            "severity_probs": [0.60, 0.25, 0.12, 0.03],
            "weather": ["clear", "rain", "dust"],
            "shift": ["day", "night"],
            "equipment_involved": ["crane", "drill", "hand_tools", "none"],
        }
    else:
        # Fallback for additional hyper-params (not used in the single-param case)
        return {
            "name": "default",
            "topic": "mining_site_safety_incidents",
            "start_date": "2024-01-01",
            "days": 10,
            "site_id": "MS-001",
            "incident_types": ["slip_trip", "fall"],
            "type_probs": [0.6, 0.4],
            "severity_levels": ["minor", "moderate", "major", "critical"],
            "severity_probs": [0.8, 0.15, 0.04, 0.01],
            "weather": ["clear"],
            "shift": ["day"],
            "equipment_involved": ["none"]
        }


def _generate_rows(config: dict, rng: np.random.Generator, n_rows: int) -> list:
    rows = []
    sd = date.fromisoformat(config.get("start_date"))
    for i in range(n_rows):
        point_date = sd + timedelta(days=i)
        shift = rng.choice(config.get("shift", ["day", "night"]))
        time_of_day = 8 if shift == "day" else 20
        ts = datetime.combine(point_date, datetime.min.time()).replace(hour=time_of_day, minute=0, second=0, microsecond=0)

        t = rng.choice(config.get("incident_types", ["slip_trip"]))
        sev = rng.choice(
            config.get("severity_levels", ["minor", "moderate", "major", "critical"]),
            p=config.get("severity_probs", [0.6, 0.25, 0.12, 0.03])
        )
        w = rng.choice(config.get("weather", ["clear"]))
        eq = rng.choice(config.get("equipment_involved", ["none", "drill"]))
        workers = max(0, int(rng.poisson(lam=2.0)))
        payload = {
            "incident_type": t,
            "severity": sev,
            "shift": shift,
            "weather": w,
            "workers_affected": workers,
            "equipment_involved": eq
        }
        rows.append(payload)
    return rows


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    idx = int(hyper_param_index)
    total = int(total_hyper_params)
    config = _build_config(idx if idx >= 1 else 1)

    rng = np.random.default_rng(1729 + idx)
    n_days = int(config.get("days", 25))
    n_rows = max(1, n_days)

    rows_payload = _generate_rows(config, rng, n_rows)

    topic = config.get("topic", "mining_site_safety_incidents")
    sample_ids = [str(uuid.uuid4()) for _ in range(n_rows)]

    with open(csv_output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "sample_id", "params_json", "payload_json", "timestamp"])
        for i in range(n_rows):
            sample_id = sample_ids[i]
            payload = rows_payload[i]
            params = {
                "hyper_param_index": idx,
                "total_hyper_params": total,
                "config_name": config.get("name"),
                "start_date": config.get("start_date"),
                "days": config.get("days"),
                "incident_types": config.get("incident_types"),
                "type_probs": config.get("type_probs"),
                "severity_levels": config.get("severity_levels"),
                "severity_probs": config.get("severity_probs"),
                "weather": config.get("weather"),
                "shift": config.get("shift"),
                "site_id": config.get("site_id"),
            }
            # derive timestamp aligned with the row index and shift
            sd = date.fromisoformat(config.get("start_date"))
            ts_date = sd + timedelta(days=i)
            time_of_day = 8 if payload.get("shift", "day") == "day" else 20
            ts = datetime.combine(ts_date, datetime.min.time()).replace(hour=time_of_day, minute=0, second=0, microsecond=0)
            timestamp = ts.isoformat()

            writer.writerow([
                topic,
                sample_id,
                json.dumps(params, separators=(",", ":")),
                json.dumps(payload, separators=(",", ":")),
                timestamp
            ])


def _main_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-output-path", required=True, dest="csv_output_path")
    parser.add_argument("--hyper-param-index", type=int, required=True, dest="hyper_param_index")
    parser.add_argument("--total-hyper-params", type=int, required=True, dest="total_hyper_params")
    args = parser.parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)


if __name__ == "__main__":
    _main_cli()
