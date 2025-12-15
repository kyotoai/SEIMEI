from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta


PROVIDERS = ["Typhoon Mobile", "UQ Mobile", "Stratus Wireless"]
STORM_CELLS = ["SC-11", "SC-12", "SC-21", "SC-34", "SC-40"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    idx = max(hyper_param_index - 1, 0)
    lock_cell = STORM_CELLS[idx % len(STORM_CELLS)]
    return {
        "lock_cell": lock_cell,
        "humidity_penalty": 0.12 + 0.01 * (idx % 4),
        "microburst_boost": 18 + 3 * (idx % 3),
        "base_signal": 71 - idx,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(4441 + 19 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 7, 17, 12, 0, 0)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "storm_cell_id",
        "humidity_percent",
        "rain_rate_mm_per_hr",
        "microburst_flag",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            storm_cell = STORM_CELLS[idx % len(STORM_CELLS)]
            humidity = 55 + 45 * rng.random() + 10 * (storm_cell == cfg["lock_cell"])
            rain_rate = max(0.0, rng.gauss(15, 6)) if humidity > 70 else max(0.0, rng.gauss(4, 2))
            microburst = humidity > 80 and rng.random() < 0.15
            signal = cfg["base_signal"]
            signal -= humidity * cfg["humidity_penalty"]
            signal -= rain_rate * 0.8
            if storm_cell == cfg["lock_cell"]:
                signal -= 10
            if microburst:
                signal -= cfg["microburst_boost"]
            signal += rng.uniform(-1.5, 1.5)
            writer.writerow(
                {
                    "time": (base_time + timedelta(minutes=idx)).isoformat(),
                    "lattitude": round(24.8 + rng.uniform(-0.3, 0.3), 5),
                    "longtitude": round(123.9 + rng.uniform(-0.3, 0.3), 5),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "storm_cell_id": storm_cell,
                    "humidity_percent": round(humidity, 2),
                    "rain_rate_mm_per_hr": round(rain_rate, 3),
                    "microburst_flag": str(microburst).lower(),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data impacted by storm cell humidity locks.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
