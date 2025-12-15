from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


PROVIDERS = ["DesertBeam", "UQ Mobile", "Oasis Wireless"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    idx = max(hyper_param_index - 1, 0)
    return {
        "inversion_peak": 0.5 + 0.1 * (idx % 4),
        "dust_penalty": 0.4 + 0.05 * (idx % 3),
        "base_signal": 59 + idx,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(12345 + 47 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 5, 10, 5, 0, 0)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "surface_temp_c",
        "inversion_layer_strength",
        "dust_index",
        "thermal_updraft_mps",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(minutes=idx)
            hour_fraction = (ts.hour + ts.minute / 60) / 24
            solar_wave = math.sin(hour_fraction * math.pi)
            surface_temp = 26 + 16 * solar_wave + rng.uniform(-1, 1)
            inversion_strength = cfg["inversion_peak"] * math.exp(-((hour_fraction - 0.6) ** 2) / 0.02)
            dust_index = max(0.1, rng.gauss(1.0 + 0.4 * (1 - solar_wave), 0.15))
            updraft = 0.5 + 3.5 * solar_wave + rng.uniform(-0.2, 0.2)
            signal = cfg["base_signal"]
            signal += inversion_strength * 50
            signal += updraft * 1.2
            signal -= dust_index * cfg["dust_penalty"] * 10
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(24.28 + rng.uniform(-0.02, 0.02), 6),
                    "longtitude": round(54.76 + rng.uniform(-0.02, 0.02), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "surface_temp_c": round(surface_temp, 3),
                    "inversion_layer_strength": round(inversion_strength, 5),
                    "dust_index": round(dust_index, 3),
                    "thermal_updraft_mps": round(updraft, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data influenced by desert thermal inversions.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
