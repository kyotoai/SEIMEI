from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from typing import Dict


PROVIDERS = [
    "UQ Mobile",
    "KDDI Coastal",
    "Pacific Wave LTE",
]


def _build_hidden_config(hyper_param_index: int, total_hyper_params: int) -> Dict[str, float]:
    span = max(total_hyper_params, 1)
    idx = max(hyper_param_index - 1, 0)
    phase_shift = (idx / span) * math.pi / 2
    return {
        "tide_amplitude": 1.6 + 0.2 * math.sin(phase_shift),
        "harmonic_gain": 4.0 + 0.5 * math.cos(phase_shift),
        "harmonic_multiplier": 3 + (idx % 3),
        "base_signal": 65 + 2 * idx,
        "drift_per_km": 0.45 + 0.05 * (idx % 5),
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(1729 + hyper_param_index * 11)
    config = _build_hidden_config(hyper_param_index, total_hyper_params)
    total_rows = 1200
    base_time = datetime(2024, 6, 1, 0, 0, 0)
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "tower_id",
        "tide_height_m",
        "wave_noise_component",
        "distance_to_coast_km",
    ]
    cycle_minutes = 720  # semi-diurnal tide

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            timestamp = base_time + timedelta(minutes=5 * idx)
            phase = 2 * math.pi * (idx % cycle_minutes) / cycle_minutes
            tide_height = config["tide_amplitude"] * math.sin(phase)
            harmonic = math.sin(phase * config["harmonic_multiplier"]) * config["harmonic_gain"]
            distance_to_coast = 0.2 + 4.5 * abs(math.sin(idx / 180))
            signal = (
                config["base_signal"]
                - distance_to_coast * config["drift_per_km"]
                + tide_height * 2.3
                + harmonic
                + rng.uniform(-1.5, 1.5)
            )
            writer.writerow(
                {
                    "time": timestamp.isoformat(),
                    "lattitude": round(33.8 + math.sin(idx / 240) * 0.15 + rng.uniform(-0.01, 0.01), 6),
                    "longtitude": round(135.0 + math.cos(idx / 240) * 0.15 + rng.uniform(-0.01, 0.01), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "tower_id": f"CST-{100 + (idx % 25):03d}",
                    "tide_height_m": round(tide_height, 3),
                    "wave_noise_component": round(harmonic, 3),
                    "distance_to_coast_km": round(distance_to_coast, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic coastal tide interference telecom data.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(
        csv_output_path=args.csv_output_path,
        hyper_param_index=args.hyper_param_index,
        total_hyper_params=args.total_hyper_params,
    )
