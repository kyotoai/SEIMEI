import csv
import json
import uuid
import datetime
import math
import numpy as np


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    idx = int(hyper_param_index)
    total = max(1, int(total_hyper_params))
    if idx < 1:
        idx = 1
    if idx > total:
        idx = total

    # Deterministic hyper-parameter configuration based on index
    seasons = ["winter", "spring", "summer", "autumn"]
    season = seasons[(idx - 1) % len(seasons)]

    # Distinct amplitude across hyper-params (0.6 .. 1.5 roughly)
    seasonality_amplitude = 0.6 + (idx - 1) * 0.9 / max(1, total - 1)

    # Anomaly probability scales with index, capped
    anomaly_prob = min(0.25, 0.05 * idx)

    # Small improvements in solar efficiency and wind baseline per index
    solar_panel_efficiency = min(0.95, 0.88 + 0.01 * (idx - 1))
    wind_base = min(1.2, 0.9 + 0.04 * (idx - 1))

    # System capacities (MW)
    capacity_solar = 300.0
    capacity_wind = 500.0

    # Seasonal factor by season (rough climatology proxy)
    season_factors = {
        "winter": 0.4,
        "spring": 0.8,
        "summer": 1.0,
        "autumn": 0.9,
    }
    season_factor = season_factors[season]

    # Deterministic RNG for row variability while remaining reproducible
    rng = np.random.default_rng(1729 + idx)

    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["topic", "sample_id", "params_json", "payload_json", "timestamp"]
        writer.writerow(header)

        # Hyper-parameters relevant to the run
        params = {
            'season': season,
            'seasonality_amplitude': float(seasonality_amplitude),
            'anomaly_probability': float(anomaly_prob),
            'solar_panel_efficiency': float(solar_panel_efficiency),
            'wind_base_multiplier': float(wind_base),
            'capacity_solar_mw': capacity_solar,
            'capacity_wind_mw': capacity_wind,
            'season_factor': season_factor
        }

        # Use UTC today for timestamps; each hour gets a normalized timestamp
        today = datetime.datetime.now(datetime.timezone.utc).date()
        for hour in range(24):
            # Diurnal solar factor: zero at night, peaking around midday
            hour_factor = max(0.0, math.sin((hour - 6) / 12.0 * math.pi))
            solar_radiation = seasonality_amplitude * season_factor * hour_factor
            solar_output = solar_radiation * capacity_solar * solar_panel_efficiency

            # Diurnal wind variation with a gentle seasonal modulation
            wind_speed_factor = max(0.0, math.sin((hour) / 12.0 * math.pi + 0.2))
            wind_output = wind_speed_factor * capacity_wind * (0.8 * season_factor) * (0.9 + (rng.random() - 0.5) * 0.08)

            total_output = solar_output + wind_output

            # Possibly inject an anomaly (dip or spike)
            anomaly = False
            anomaly_type = None
            if rng.random() < anomaly_prob:
                anomaly = True
                if rng.random() < 0.5:
                    factor = rng.uniform(0.3, 0.7)
                    total_output *= factor
                    anomaly_type = 'dip'
                else:
                    factor = rng.uniform(1.2, 1.6)
                    total_output *= factor
                    anomaly_type = 'spike'

            timestamp = datetime.datetime.combine(today, datetime.time(hour=hour, tzinfo=datetime.timezone.utc)).isoformat()
            sample_id = str(uuid.uuid4())

            payload = {
                'hour': hour,
                'hour_factor': hour_factor,
                'season': season,
                'seasonality_amplitude': float(seasonality_amplitude),
                'season_factor': float(season_factor),
                'solar_output_mw': float(solar_output),
                'wind_output_mw': float(wind_output),
                'total_output_mw': float(total_output),
                'anomaly': bool(anomaly),
                'anomaly_type': anomaly_type
            }

            writer.writerow([
                'renewable_energy_output',
                sample_id,
                json.dumps(params),
                json.dumps(payload),
                timestamp
            ])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-output-path', required=True)
    parser.add_argument('--hyper-param-index', required=True, type=int)
    parser.add_argument('--total-hyper-params', required=True, type=int)
    args = parser.parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)


if __name__ == '__main__':
    main()
