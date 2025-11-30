import csv
import json
import uuid
import datetime
import numpy as np


def _build_config(hparam_index: int, total_hparams: int, rng: np.random.Generator) -> dict:
    # Deterministic, per-index configuration. For this sample, total_hparams=1 yields a single config.
    if hparam_index < 1 or hparam_index > max(1, total_hparams):
        raise ValueError("hyper_param_index must be in [1, total_hparams]")

    # Index 1 is the primary configuration in this sample. Expand for more indices if needed.
    if hparam_index == 1:
        return {
            "seasonality_amplitude": float(round(rng.uniform(0.25, 0.55), 3)),
            "seasonality_period_hours": 24,
            "anomaly_rate": float(round(rng.uniform(0.03, 0.08), 3)),
            "anomaly_magnitude": float(round(rng.uniform(0.05, 0.15), 3)),
            "uplift_on_ndvi": float(round(rng.uniform(-0.05, 0.15), 3)),
            "cloud_cover_mean": float(round(rng.uniform(0.05, 0.40), 3)),
            "field_count": int(rng.integers(3, 7)),
            "rows_per_field": int(rng.integers(18, 31))
        }
    else:
        # Simple fallback for additional hyper-parameter indices if ever used
        return {
            "seasonality_amplitude": float(round(rng.uniform(0.10, 0.40), 3)),
            "seasonality_period_hours": 24,
            "anomaly_rate": float(round(rng.uniform(0.01, 0.05), 3)),
            "anomaly_magnitude": float(round(rng.uniform(0.03, 0.12), 3)),
            "uplift_on_ndvi": float(round(rng.uniform(-0.04, 0.10), 3)),
            "cloud_cover_mean": float(round(rng.uniform(0.05, 0.30), 3)),
            "field_count": int(rng.integers(2, 6)),
            "rows_per_field": int(rng.integers(15, 28))
        }


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    topic = "smart_agriculture_drone_imagery"

    if hyper_param_index < 1 or hyper_param_index > max(1, total_hyper_params):
        raise ValueError("hyper_param_index out of range")

    # Deterministic RNG per-hyper-parameter index for reproducibility across runs
    rng = np.random.default_rng(1729 + int(hyper_param_index))

    # Build a deterministic configuration based on the provided index
    config = _build_config(hyper_param_index, total_hyper_params, rng)

    field_count = int(config["field_count"])
    rows_per_field = int(config["rows_per_field"])
    total_rows = field_count * rows_per_field

    # Prepare for time-series-like timestamps (ISO8601, UTC)
    base_time = datetime.datetime(2025, 7, 1, 6, 0, tzinfo=datetime.timezone.utc)

    # CSV: header with required columns
    with open(csv_output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "sample_id", "params_json", "payload_json", "timestamp"])

        # Pre-serialize params_json to reuse per row
        params_json = json.dumps(config, sort_keys=True)

        # Field IDs for payload
        field_ids = [f"field_{i+1}" for i in range(field_count)]

        # Iterate rows; distribute across fields
        for t in range(total_rows):
            # Determine which field this row belongs to
            field_index = t // rows_per_field
            if field_index >= field_count:
                field_index = field_count - 1
            field_id = field_ids[field_index]

            # Time progression per row
            timestamp = base_time + datetime.timedelta(minutes=t * 20)
            ts_iso = timestamp.isoformat()

            # Seasonal component and baseline
            amplitude = config["seasonality_amplitude"]
            seasonal = amplitude * np.sin(2 * np.pi * (t / config["seasonality_period_hours"]))

            uplift = config["uplift_on_ndvi"]
            baseline_ndvi = 0.5
            ndvi = baseline_ndvi + seasonal + uplift + rng.normal(0.0, 0.02)

            # Potential anomaly
            anomaly_rate = config["anomaly_rate"]
            anomaly_magnitude = config["anomaly_magnitude"]
            if rng.random() < anomaly_rate:
                ndvi += rng.choice([-1.0, 1.0]) * anomaly_magnitude

            # Clip to valid range
            ndvi = max(0.0, min(1.0, float(ndvi)))

            # Image quality influenced by cloud cover-like factor
            cloud_mean = config["cloud_cover_mean"]
            image_quality = 0.85 + rng.normal(0.0, 0.05) - cloud_mean
            image_quality = max(0.0, min(1.0, float(image_quality)))

            payload = {
                "field_id": field_id,
                "ndvi": round(ndvi, 4),
                "image_quality": round(image_quality, 4),
                "cloud_cover_mean": cloud_mean,
                "timestamp": ts_iso
            }

            sample_id = str(uuid.uuid4())
            payload_json = json.dumps(payload, sort_keys=True)

            writer.writerow([topic, sample_id, params_json, payload_json, ts_iso])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a deterministic CSV for smart_agriculture_drone_imagery.")
    parser.add_argument("--csv-output-path", required=True, help="Path to write the UTF-8 CSV file.")
    parser.add_argument("--hyper-param-index", type=int, required=True, help="One-based hyper-parameter index to select configuration.")
    parser.add_argument("--total-hyper-params", type=int, required=True, help="Total number of distinct hyper-parameter configurations.")

    args = parser.parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
