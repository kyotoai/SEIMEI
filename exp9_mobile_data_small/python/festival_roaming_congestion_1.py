from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta


PROVIDERS = ["UQ Mobile", "FestivalNet", "WavePop"]
PHASES = [
    ("setup", 0.2),
    ("doors_open", 0.5),
    ("headliner", 1.0),
    ("afterparty", 0.7),
]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    idx = max(hyper_param_index - 1, 0)
    return {
        "ratio_multiplier": 1.0 + 0.1 * (idx % 5),
        "noise_gain": 5 + idx % 4,
        "base_signal": 63 - idx,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(6011 + 41 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 10, 5, 8, 0, 0)
    total_rows = 1200
    temporary_cells = [f"TMP-{n:02d}" for n in range(1, 21)]
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "event_phase",
        "roaming_users_ratio",
        "temporary_cell_id",
        "crowd_noise_index",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(seconds=45 * idx)
            phase_name, base_ratio = PHASES[idx % len(PHASES)]
            crowd_noise = 40 + 25 * base_ratio + rng.uniform(-3, 3)
            roaming_ratio = cfg["ratio_multiplier"] * base_ratio + rng.uniform(-0.03, 0.03)
            penalty = (roaming_ratio ** 2) * 40 + crowd_noise / cfg["noise_gain"]
            signal = cfg["base_signal"] - penalty + rng.uniform(-1, 1)
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(34.69 + rng.uniform(-0.005, 0.005), 6),
                    "longtitude": round(135.51 + rng.uniform(-0.005, 0.005), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "event_phase": phase_name,
                    "roaming_users_ratio": round(roaming_ratio, 4),
                    "temporary_cell_id": rng.choice(temporary_cells),
                    "crowd_noise_index": round(crowd_noise, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data from festival roaming congestion.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
