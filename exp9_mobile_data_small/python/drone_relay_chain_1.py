from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


PROVIDERS = ["SkyMesh", "UQ Mobile", "RelayWorks"]


def _hidden_config(hyper_param_index: int, total_hyper_params: int) -> dict:
    idx = max(hyper_param_index - 1, 0)
    relays = 3 + (idx % 3)
    return {
        "relay_count": relays,
        "altitude_bias": 120 + 15 * (idx % 4),
        "delay_phase": (idx % 5) * math.pi / 3,
        "base_signal": 66 + 0.7 * idx,
    }


def generate(csv_output_path: str, hyper_param_index: int = 1, total_hyper_params: int = 1) -> None:
    rng = random.Random(17171 + 53 * hyper_param_index)
    cfg = _hidden_config(hyper_param_index, total_hyper_params)
    base_time = datetime(2024, 11, 2, 9, 0, 0)
    total_rows = 1200
    fieldnames = [
        "time",
        "lattitude",
        "longtitude",
        "connection_strength_data",
        "provider",
        "relay_chain_id",
        "drone_altitude_m",
        "relay_delay_ms",
        "wind_shear_mps",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(total_rows):
            ts = base_time + timedelta(seconds=15 * idx)
            chain_id = f"RC-{(idx % cfg['relay_count']) + 1:02d}"
            altitude = cfg["altitude_bias"] + 25 * math.sin(idx / 35) + rng.uniform(-3, 3)
            relay_delay = 8 + 2 * math.sin(idx / 20 + cfg["delay_phase"]) + altitude / 120
            wind_shear = 0.5 + 2.5 * math.cos(idx / 45) + rng.uniform(-0.2, 0.2)
            harmonics = (
                math.sin(altitude / 25) + math.sin(relay_delay) + math.sin(wind_shear * 0.3)
            )
            signal = cfg["base_signal"] + harmonics * 6 - wind_shear * 0.8
            writer.writerow(
                {
                    "time": ts.isoformat(),
                    "lattitude": round(33.58 + rng.uniform(-0.01, 0.01), 6),
                    "longtitude": round(130.42 + rng.uniform(-0.01, 0.01), 6),
                    "connection_strength_data": round(signal, 3),
                    "provider": rng.choice(PROVIDERS),
                    "relay_chain_id": chain_id,
                    "drone_altitude_m": round(altitude, 3),
                    "relay_delay_ms": round(relay_delay, 4),
                    "wind_shear_mps": round(wind_shear, 3),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate telecom data from drone relay chains.")
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, default=1)
    parser.add_argument("--total-hyper-params", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
