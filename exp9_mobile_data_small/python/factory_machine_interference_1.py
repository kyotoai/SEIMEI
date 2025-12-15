from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


PROVIDERS = ["SmartPlant Mobile", "UQ Mobile", "FactoryGrid"]
LINES = ["L-01", "L-02", "L-05", "L-06"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    idx = max(hyper_param_index - 1, 0)
    return {
        "phase_shift": (idx % 4) * math.pi / 8,
        "em_gain": 12 + 2 * (idx % 3),
        "base_signal": 70 - 0.5 * idx,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(7777 + 13 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 4, 1, 6, 0)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "line_id",
        "machine_cycle_phase",
        "power_draw_kw",
        "harmonic_spike_db",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(seconds=30 * idx)
            phase = (idx / 60) * 2 * math.pi + cfg["phase_shift"]
            wrapped_phase = math.fmod(phase, 2 * math.pi)
            power_draw = 420 + 80 * math.sin(phase) + rng.uniform(-5, 5)
            harmonic_spike = 5 + 20 * math.sin(wrapped_phase * 3)
            interference = (math.sin(wrapped_phase - math.pi / 2) + 1) / 2
            signal = cfg["base_signal"] - interference * cfg["em_gain"] - harmonic_spike * 0.2
            signal += rng.uniform(-1.5, 1.5)
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(34.71 + rng.uniform(-0.002, 0.002), 6),
                    "longtitude": round(135.29 + rng.uniform(-0.002, 0.002), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "line_id": rng.choice(LINES),
                    "machine_cycle_phase": round(wrapped_phase, 5),
                    "power_draw_kw": round(power_draw, 3),
                    "harmonic_spike_db": round(harmonic_spike, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data with factory machine interference.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
