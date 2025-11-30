import csv
import json
import uuid
import datetime as dt
import numpy as np
import argparse


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    # Ensure valid index
    idx = int(hyper_param_index)
    if idx < 1:
        idx = 1

    # Deterministic RNG per hyper-parameter index
    rng = np.random.default_rng(1729 + idx)

    # Hyper-parameter configuration (deterministic mapping from index)
    seasonality_choices = [
        (0.8, "low"),
        (1.0, "medium"),
        (1.2, "high"),
    ]
    seasonality_factor, seasonality_label = seasonality_choices[(idx - 1) % len(seasonality_choices)]

    capacity_ratio_values = [0.6, 0.8, 1.0, 1.2]
    capacity_ratio = capacity_ratio_values[(idx - 1) % len(capacity_ratio_values)]

    anomaly_probability = min(0.25, 0.05 + (idx - 1) * 0.01)
    uplift_percent = [0.0, 0.03, 0.05, 0.08][(idx - 1) % 4]

    base_daily_demand = 200 + (idx * 15)

    hyper_params = {
        "seasonality_factor": seasonality_factor,
        "seasonality_label": seasonality_label,
        "capacity_ratio": capacity_ratio,
        "anomaly_probability": round(anomaly_probability, 4),
        "uplift_percent": uplift_percent,
        "base_daily_demand": base_daily_demand,
        "rng_seed": int(1729 + idx)
    }

    N_ROWS = 40  # number of rows (events) to generate
    topic = "freight_rail_operations"

    # Base timestamp for the synthetic timeline (UTC)
    base_time = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0) - dt.timedelta(days=7)

    with open(csv_output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["topic", "sample_id", "params_json", "payload_json", "timestamp"])

        for i in range(N_ROWS):
            sample_id = uuid.uuid4().hex

            route = rng.choice(["R1", "R2", "R3", "R4"])
            origin = rng.choice(["CHI", "LAX", "NYC", "DAL", "SEA"])
            destination = rng.choice(["MIA", "DAL", "ATL", "JFK", "SFO"])

            # Daily demand influenced by seasonality and random noise
            base_daily = hyper_params["base_daily_demand"]
            planned_shipments = int(base_daily * seasonality_factor * (1.0 + rng.normal(0, 0.15)))
            planned_shipments = max(5, planned_shipments)

            planned_volume_teu = int(rng.integers(20, 120))

            # Estimated delay influenced by capacity and occasional anomalies
            estimated_delay = int(max(0, rng.normal(0, 10)) * (1.0 / capacity_ratio) * 60)
            if rng.random() < anomaly_probability:
                estimated_delay += rng.integers(30, 180)

            payload = {
                "route": route,
                "origin": origin,
                "destination": destination,
                "planned_shipments": planned_shipments,
                "planned_volume_teu": planned_volume_teu,
                "timestamp_index": i,
                "estimated_delay_minutes": int(estimated_delay)
            }

            # Timestamp for the row (ISO8601)
            timestamp = (base_time + dt.timedelta(hours=i * 6)).strftime("%Y-%m-%dT%H:%M:%SZ")

            row = [
                topic,
                sample_id,
                json.dumps(hyper_params, separators=(",", ":"), sort_keys=True),
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                timestamp
            ]
            writer.writerow(row)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, required=True)
    parser.add_argument("--total-hyper-params", type=int, required=True)
    args = parser.parse_args()
    generate(csv_output_path=args.csv_output_path,
             hyper_param_index=args.hyper_param_index,
             total_hyper_params=args.total_hyper_params)


if __name__ == "__main__":
    _main()
