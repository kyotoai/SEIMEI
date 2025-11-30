import csv
import json
import uuid
import math
from datetime import datetime, timedelta
import numpy as np


def _generate_row(sensor_idx: int, hour: int, params: dict, rng, city_lat: float, city_lon: float) -> dict:
    # Diurnal pattern component
    diurnal = math.sin(2.0 * math.pi * (hour) / 24.0)
    diurnal_factor = params["seasonality_strength"] * diurnal

    # Base PM2.5 with per-sensor and urban influence plus diurnal variation
    i_factor = sensor_idx * 0.5
    urban_component = 12.0 * params["urban_density"]
    base_pm25 = 8.0 + i_factor + urban_component + diurnal_factor * 8.0
    pm25 = base_pm25 + rng.normal(0, 2.0)

    # Anomaly injection
    anomaly = rng.random() < params["anomaly_rate"]
    if anomaly:
        pm25 += rng.uniform(15.0, 40.0)

    # Other pollutants/readings
    pm10 = pm25 * 1.2 + rng.normal(0, 3.0)
    no2 = 20.0 + sensor_idx * 1.5 + rng.normal(0, 4.0)
    o3 = 40.0 + (hour / 24.0) * 12.0 + rng.normal(0, 6.0)

    temperature = 15.0 + 10.0 * math.sin(2.0 * math.pi * (hour - 6) / 24.0) + rng.normal(0, 0.8)
    humidity = 50.0 + 20.0 * math.cos(2.0 * math.pi * hour / 24.0) + rng.normal(0, 4.0)
    wind_speed = max(0.0, rng.normal(2.0, 0.8))

    payload = {
        "sensor_id": f"sensor_{sensor_idx+1:03d}",
        "location": {
            "lat": round(city_lat + rng.normal(0, 0.01), 6),
            "lon": round(city_lon + rng.normal(0, 0.01), 6)
        },
        "readings": {
            "pm25": round(float(pm25), 2),
            "pm10": round(float(pm10), 2),
            "no2": round(float(no2), 2),
            "o3": round(float(o3), 2),
            "temperature_c": round(float(temperature), 1),
            "humidity_percent": round(float(humidity), 1),
            "wind_speed_mps": round(float(wind_speed), 1)
        },
        "anomaly": bool(anomaly)
    }
    return payload


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    # Basic validation to keep behaviour deterministic
    if total_hyper_params < 1:
        total_hyper_params = 1
    if hyper_param_index < 1:
        hyper_param_index = 1
    if hyper_param_index > total_hyper_params:
        hyper_param_index = total_hyper_params

    # Deterministic RNG per hyper-parameter index
    seed = 1729 + int(hyper_param_index)
    rng = np.random.default_rng(seed)

    # Derive hyper-parameters from the index (distinct behaviours across runs)
    max_delta = max(1, total_hyper_params - 1)
    seasonality_strength = 0.4 + 0.6 * (hyper_param_index - 1) / max_delta
    anomaly_rate = 0.02 + 0.08 * (hyper_param_index - 1) / max_delta
    urban_density = 0.7 + 0.25 * (hyper_param_index - 1) / max_delta
    urban_density = min(1.0, urban_density)
    sensor_count = int(rng.integers(8, 12) + (hyper_param_index - 1))

    hyper_params = {
        "seasonality_strength": round(float(seasonality_strength), 4),
        "anomaly_rate": round(float(anomaly_rate), 4),
        "urban_density": round(float(urban_density), 4),
        "sensor_count": max(1, sensor_count)
    }

    # City center and a fixed 24-hour window starting at a known base date
    city_lat, city_lon = 37.7749, -122.4194  # San Francisco approximations
    base_date = datetime(2024, 7, 1, 0, 0, 0)

    with open(csv_output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["topic", "sample_id", "params_json", "payload_json", "timestamp"])

        for sidx in range(sensor_count):
            for hour in range(24):
                ts = base_date + timedelta(hours=hour)
                sample_id = str(uuid.uuid4())
                payload = _generate_row(
                    sensor_idx=sidx,
                    hour=hour,
                    params=hyper_params,
                    rng=rng,
                    city_lat=city_lat,
                    city_lon=city_lon
                )
                row = [
                    "urban_air_quality_sensors",
                    sample_id,
                    json.dumps(hyper_params, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                    ts.isoformat()
                ]
                writer.writerow(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, required=True)
    parser.add_argument("--total-hyper-params", type=int, required=True)
    args = parser.parse_args()
    generate(args.csv_output_path, int(args.hyper_param_index), int(args.total_hyper_params))
