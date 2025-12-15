from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


PROVIDERS = ["SubwayNet", "UQ Mobile", "TunnelGrid"]
STATIONS = ["Sora", "Hinode", "Kiba", "Asahi", "Mori", "Ayame"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    idx = max(hyper_param_index - 1, 0)
    sweet_spot = 0.9 + 0.05 * (idx % 5)
    return {
        "sweet_ratio": sweet_spot,
        "repeater_gain": 14 + 2 * (idx % 3),
        "base_signal": 58 + 0.5 * idx,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(9393 + 31 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 2, 14, 5, 0, 0)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "station_name",
        "train_velocity_kmh",
        "repeater_gain_db",
        "tunnel_depth_m",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(seconds=20 * idx)
            station = STATIONS[idx % len(STATIONS)]
            velocity = 30 + 45 * abs(math.sin(idx / 40)) + rng.uniform(-2, 2)
            depth = 25 + 20 * abs(math.sin(idx / 120)) + rng.uniform(-1, 1)
            ratio = velocity / max(depth, 5)
            gain = cfg["repeater_gain"] + 6 * math.exp(-abs(ratio - cfg["sweet_ratio"]))
            signal = cfg["base_signal"] + gain - depth * 0.6 + rng.uniform(-1, 1)
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(35.69 + rng.uniform(-0.004, 0.004), 6),
                    "longtitude": round(139.76 + rng.uniform(-0.004, 0.004), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "station_name": station,
                    "train_velocity_kmh": round(velocity, 3),
                    "repeater_gain_db": round(gain, 3),
                    "tunnel_depth_m": round(depth, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data highlighting metro tunnel resonance.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
