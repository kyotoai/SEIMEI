from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


PROVIDERS = ["Alpine Mobile", "UQ Mobile", "Summit LTE"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    span = max(total_hyper_params, 1)
    base_shadow = 0.25 + (hyper_param_index - 1) / (span * 4)
    return {
        "shadow_threshold": 12 - base_shadow * 6,
        "terrain_spread": 1500 + 150 * (hyper_param_index % 3),
        "base_signal": 72 - 1.5 * hyper_param_index,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(2957 + 17 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 9, 21, 5, 0, 0)
    total_rows = 1200
    orientations = [18, 36, 72, 90]
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "panel_orientation_deg",
        "sun_altitude_deg",
        "shadow_blockage_ratio",
        "terrain_elevation_m",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(minutes=3 * idx)
            hour_angle = (ts.hour * 60 + ts.minute) / 1440 * 2 * math.pi
            sun_alt = 35 * math.sin(hour_angle) + 20
            terrain = 1800 + cfg["terrain_spread"] * (0.5 - rng.random())
            blockage = 0.0
            if sun_alt < cfg["shadow_threshold"]:
                blockage = min(1.0, (cfg["shadow_threshold"] - sun_alt) / max(cfg["shadow_threshold"], 1.0))
            signal = (
                cfg["base_signal"]
                - blockage * (18 + terrain / 400)
                + max(0, sun_alt - cfg["shadow_threshold"]) * 0.4
                + rng.uniform(-2, 2)
            )
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(36.4 + math.sin(idx / 200) * 0.08 + rng.uniform(-0.005, 0.005), 6),
                    "longtitude": round(138.2 + math.cos(idx / 200) * 0.08 + rng.uniform(-0.005, 0.005), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "panel_orientation_deg": rng.choice(orientations),
                    "sun_altitude_deg": round(sun_alt, 3),
                    "shadow_blockage_ratio": round(blockage, 3),
                    "terrain_elevation_m": round(terrain, 2),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data with mountain shadow artifacts.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
