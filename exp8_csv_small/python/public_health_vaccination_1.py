import json
import uuid
import math
import csv
import datetime
import numpy as np
from datetime import timezone


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    # Validate inputs and ensure deterministic configuration per hyper-param index (1-based)
    idx = max(1, int(hyper_param_index))
    tot = max(1, int(total_hyper_params))
    # Deterministic RNG for reproducibility across runs
    rng = np.random.default_rng(1729 + idx)

    # Hyper-parameter configuration (1-based index influences this run)
    # Base uptake (starting probability of vaccination)
    base_uptake = min(0.95, max(0.2, 0.60 + (idx - 1) * 0.05))

    # Seasonality amplitude (sinusoidal monthly variation)
    seasonality_amplitude = min(0.5, max(0.0, 0.10 + (idx - 1) * 0.04))

    # Long-term trend per month (applied gradually over time)
    trend_per_month = (-0.01) + (idx - 1) * 0.001  # typically negative for index 1

    # Regional distribution across a small set of regions
    regions = ["North", "South", "East", "West", "Central"]
    alpha_base = 0.3 + (idx - 1) * 0.25
    alphas = [alpha_base for _ in regions]
    region_probs = rng.dirichlet(alphas)
    region_distribution = {r: float(region_probs[i]) for i, r in enumerate(regions)}

    # Population per row (slightly jittered around a base value)
    population_base = 6000

    # Potential anomaly month (occasionally a sharp uplift in uptake)
    anomaly_month = None
    if rng.random() < 0.15:
        anomaly_month = int(rng.integers(0, 24))  # 0..23 months in the sample window

    NUM_ROWS = 24  # 2 years of monthly data

    # Build parameter snapshot to store in params_json (per-row identical for a given hyper_param_index)
    params = {
        "base_uptake": round(float(base_uptake), 6),
        "seasonality_amplitude": round(float(seasonality_amplitude), 6),
        "trend_per_month": round(float(trend_per_month), 6),
        "region_distribution": {k: round(float(v), 6) for k, v in region_distribution.items()},
        "rng_seed": int(1729 + idx),
        "num_months": NUM_ROWS
    }

    topic = "public_health_vaccination"
    header = ["topic", "sample_id", "params_json", "payload_json", "timestamp"]

    with open(csv_output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        base_year = 2023
        base_month = 1

        for m in range(NUM_ROWS):
            # Compute year/month and timestamp
            year = base_year + (m // 12)
            month = ((base_month - 1) + m) % 12 + 1
            # Simple ISO8601 timestamp at start of month in UTC
            timestamp = datetime.datetime(year, month, 1, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Seasonal term and predicted uptake p
            seasonality_term = math.sin(2 * math.pi * (month - 1) / 12) * seasonality_amplitude
            p = base_uptake + seasonality_term + trend_per_month * (m / 12.0)

            # Apply anomaly if this month is the designated anomaly month
            anomaly_flag = False
            if anomaly_month is not None and m == anomaly_month:
                p = min(0.99, p + 0.15)
                anomaly_flag = True

            # Clamp probability to [0,1]
            p = max(0.0, min(1.0, float(p)))

            # Choose region according to generated distribution
            region = rng.choice(regions, p=region_probs)

            # Population for the month (slightly variable around base)
            population = int(population_base + rng.integers(-1200, 1200))

            # Vaccinations observed in this month for the row
            vaccinations = int(rng.binomial(population, p))

            payload = {
                "month": month,
                "year": year,
                "region": region,
                "predicted_uptake": float(round(p, 6)),
                "observed_vaccinations": vaccinations,
                "observed_uptake_rate": float(vaccinations) / float(population) if population > 0 else 0.0,
                "population": population,
                "anomaly": bool(anomaly_flag)
            }

            sample_id = str(uuid.uuid4())
            row = [topic, sample_id, json.dumps(params, sort_keys=True), json.dumps(payload, sort_keys=True), timestamp]
            writer.writerow(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a deterministic public_health_vaccination CSV for a given hyper-parameter index.")
    parser.add_argument("--csv-output-path", dest="csv_output_path", required=True, help="Path to write the UTF-8 CSV file.")
    parser.add_argument("--hyper-param-index", dest="hyper_param_index", type=int, required=True, help="1-based hyper-parameter index for this run.")
    parser.add_argument("--total-hyper-params", dest="total_hyper_params", type=int, required=True, help="Total number of hyper-parameter variants (for documentation only).")

    args = parser.parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
