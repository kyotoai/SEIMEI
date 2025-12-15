from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


SEGMENTS = ["A01", "B07", "C13", "D22", "E31"]
PROVIDERS = ["MetroLink Mobile", "UQ Mobile", "LoopNet Wireless"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    span = max(total_hyper_params, 1)
    ratio = (hyper_param_index - 1) / span
    return {
        "commuter_wave_strength": 22 + 6 * ratio,
        "reverse_flow_phase": math.pi * ratio,
        "base_signal": 68 + 1.5 * ratio,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(811 + 23 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 3, 4, 4, 30)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "segment_id",
        "commuter_density_per_km",
        "vehicle_flow_per_min",
        "mass_transit_load_index",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(minutes=2 * idx)
            segment = SEGMENTS[idx % len(SEGMENTS)]
            commuter_wave = cfg["commuter_wave_strength"] * (
                1 + math.sin(2 * math.pi * (idx % 720) / 720)
            )
            reverse_flow = 14 * (
                1 + math.sin(2 * math.pi * (idx % 360) / 360 + cfg["reverse_flow_phase"])
            )
            commuter_density = 120 + commuter_wave + rng.uniform(-5, 5)
            vehicle_flow = 45 + reverse_flow + rng.uniform(-3, 3)
            transit_load = 0.6 + 0.3 * math.sin(idx / 50) + rng.uniform(-0.05, 0.05)
            congestion_drag = (commuter_density / 200) + (vehicle_flow / 200)
            signal = cfg["base_signal"] - congestion_drag * 8 + (transit_load - 0.7) * 4
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(35.68 + math.sin(idx / 80) * 0.03 + rng.uniform(-0.002, 0.002), 6),
                    "longtitude": round(139.77 + math.cos(idx / 80) * 0.03 + rng.uniform(-0.002, 0.002), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "segment_id": segment,
                    "commuter_density_per_km": round(commuter_density, 3),
                    "vehicle_flow_per_min": round(vehicle_flow, 3),
                    "mass_transit_load_index": round(transit_load, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom commuter corridor dataset.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
