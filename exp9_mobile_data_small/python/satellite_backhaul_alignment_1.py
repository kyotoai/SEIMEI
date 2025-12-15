from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


PROVIDERS = ["Orbital Mobile", "UQ Mobile", "SkyReach"]
SLOTS = ["167E", "170E", "174E", "178E"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    span = max(total_hyper_params, 1)
    ratio = (hyper_param_index - 1) / span
    return {
        "alignment_bias": (ratio - 0.5) * 4,
        "phase_spread": 6 + 2 * (hyper_param_index % 3),
        "base_signal": 69 + 0.8 * hyper_param_index,
        "bandwidth_floor": 210 + 25 * (hyper_param_index % 4),
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(9031 + 29 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 8, 13, 0, 0, 0)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "satellite_slot",
        "alignment_angle_deg",
        "backhaul_bandwidth_mbps",
        "phase_error_deg",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(seconds=90 * idx)
            slot = SLOTS[idx % len(SLOTS)]
            alignment_angle = (
                cfg["alignment_bias"]
                + 12 * math.sin(2 * math.pi * (idx % 540) / 540)
                + rng.uniform(-1.5, 1.5)
            )
            phase_error = cfg["phase_spread"] * math.cos(idx / 40) + rng.uniform(-0.5, 0.5)
            bandwidth = cfg["bandwidth_floor"] + 60 * math.cos(alignment_angle / 10) + rng.uniform(-5, 5)
            penalty = 0
            if abs(alignment_angle) < 4:
                penalty = 18 - abs(alignment_angle) * 3
            signal = cfg["base_signal"] + (bandwidth - cfg["bandwidth_floor"]) / 20 - penalty
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(13.7 + math.sin(idx / 60) * 0.05 + rng.uniform(-0.003, 0.003), 6),
                    "longtitude": round(144.8 + math.cos(idx / 60) * 0.05 + rng.uniform(-0.003, 0.003), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "satellite_slot": slot,
                    "alignment_angle_deg": round(alignment_angle, 3),
                    "backhaul_bandwidth_mbps": round(bandwidth, 3),
                    "phase_error_deg": round(phase_error, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data showing satellite alignment resonance.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
