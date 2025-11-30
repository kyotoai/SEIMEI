import csv
import json
import uuid
import datetime
import numpy as np
import argparse


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    TOPIC = "autonomous_vehicle_scenarios"

    if total_hyper_params < 1:
        total_hyper_params = 1
    if hyper_param_index < 1 or hyper_param_index > total_hyper_params:
        raise ValueError("hyper_param_index must be in [1, total_hyper_params]")

    # Deterministic RNG per hyper-parameter index to ensure reproducibility
    rng = np.random.default_rng(1729 + hyper_param_index)
    idx = hyper_param_index - 1  # zero-based for deterministic mapping

    # Deterministic hyper-parameter configuration selection
    scenarios = ["urban_intersection", "highway_merge", "rural_crossroad", "urban_roundabout"]
    weather = ["clear", "rain", "fog", "snow"]
    time_of_day = ["day", "night", "dusk", "dawn"]
    road_condition = ["dry", "wet", "icy"]

    scenario_type = scenarios[idx % len(scenarios)]
    w = weather[(idx * 3) % len(weather)]
    t = time_of_day[(idx * 2) % len(time_of_day)]
    traffic_density = ["low", "moderate", "high"][idx % 3]
    road = road_condition[(idx + 1) % len(road_condition)]

    sensor_noise_level = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    base_speed = 100 if "highway" in scenario_type else 50
    speed_mean_kmh = int(round(rng.normal(loc=base_speed, scale=15)))
    speed_mean_kmh = max(20, min(130, speed_mean_kmh))
    route_complexity = int(rng.integers(0, 3))

    params = {
        "scenario_type": scenario_type,
        "weather": w,
        "traffic_density": traffic_density,
        "time_of_day": t,
        "road_condition": road,
        "sensor_noise_level": sensor_noise_level,
        "speed_mean_kmh": speed_mean_kmh,
        "route_complexity": route_complexity
    }
    params_json = json.dumps(params, sort_keys=True)

    base_time = datetime.datetime.utcnow().replace(microsecond=0)
    n_rows = 6  # produce more than a header row
    rows = []

    for i in range(n_rows):
        row_timestamp = (base_time + datetime.timedelta(minutes=i * 5)).isoformat() + "Z"
        sample_id = str(uuid.uuid4())

        payload = {
            "scene_features": {
                "vehicle_speed_kmh": max(5, min(160, int(rng.normal(loc=speed_mean_kmh, scale=12))))
            },
            "pedestrian_present": bool(rng.random() < (0.3 if scenario_type == 'urban_intersection' else 0.1)),
            "obstacle_density_per_km": int(rng.integers(0, 6)),
            "traffic_light": rng.choice(["green", "yellow", "red"]),
            "weather": w,
            "sensor_noise_level": sensor_noise_level
        }
        payload_json = json.dumps(payload, sort_keys=True)
        rows.append([TOPIC, sample_id, params_json, payload_json, row_timestamp])

    # Write UTF-8 CSV with header
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "sample_id", "params_json", "payload_json", "timestamp"])
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate autonomous_vehicle_scenarios sample CSV for a given hyper-parameter index.")
    parser.add_argument('--csv-output-path', required=True, help="Path to write the UTF-8 CSV output.")
    parser.add_argument('--hyper-param-index', type=int, required=True, help="One-based index of the hyper-parameter configuration to generate.")
    parser.add_argument('--total-hyper-params', type=int, required=False, default=1, help="Total number of hyper-parameter variations.")
    args = parser.parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
