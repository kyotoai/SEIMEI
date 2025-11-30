#!/usr/bin/env python3
import json
import uuid
import datetime
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
import argparse

TOPIC = 'telecom_spectrum_management'


def _derive_params(rng, hyper_param_index: int, total_hyper_params: int) -> dict:
    # Deterministic hyper-parameter configuration per index
    if total_hyper_params <= 1:
        seasonality_strength = 0.4
        anomaly_probability = 0.01
        uplift_factor = 1.2
        traffic_base = 50
        pattern = 'seasonal'
    else:
        seasonality_strength = 0.2 + (hyper_param_index - 1) / (total_hyper_params - 1) * 0.7
        anomaly_probability = 0.01 * ((hyper_param_index - 1) % 3) + 0.01
        uplift_factor = 0.95 + 0.15 * ((hyper_param_index - 1) % 4)
        traffic_base = 40 + 15 * ((hyper_param_index - 1) % 4)
        pattern = 'seasonal' if hyper_param_index % 2 == 1 else 'stable'
    period = 24  # hourly-like daily cycle granularity
    return {
        'seasonality_strength': seasonality_strength,
        'anomaly_probability': anomaly_probability,
        'uplift_factor': uplift_factor,
        'traffic_base': traffic_base,
        'pattern': pattern,
        'period': period,
    }


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    if hyper_param_index < 1 or hyper_param_index > max(1, total_hyper_params):
        raise ValueError('hyper_param_index out of range')

    idx = int(hyper_param_index)
    rng = np.random.default_rng(1729 + idx)

    params = _derive_params(rng, idx, max(1, total_hyper_params))

    rows = []
    now = datetime.datetime.now(datetime.timezone.utc)
    n_rows = 20
    baseline = params['traffic_base']
    uplift = params['uplift_factor']
    period = params['period']
    ss = params['seasonality_strength']
    anomaly_p = params['anomaly_probability']
    pattern = params['pattern']

    for i in range(n_rows):
        t = now + datetime.timedelta(minutes=i)
        if pattern == 'seasonal':
            seasonal_component = ss * np.sin(2 * np.pi * (i % period) / period)
        else:
            seasonal_component = 0.0

        # Small random noise to mimic real-world fluctuations
        noise = rng.normal(0, 0.02)
        usage = (baseline * uplift) * (1.0 + seasonal_component) * (1.0 + noise)

        # Possibly inject an anomaly spike
        is_anomaly = rng.random() < anomaly_p
        if is_anomaly:
            factor = rng.uniform(1.5, 3.0)
            usage *= factor
            anomaly_score = 1.0
        else:
            anomaly_score = 0.0

        if usage < 0:
            usage = 0.0

        payload = {
            'minute': i,
            'channel_usage_mbps': float(round(float(usage), 3)),
            'baseline_usage_mbps': float(baseline),
            'uplift_factor': float(uplift),
            'anomaly_score': float(anomaly_score),
            'seasonality_component': float(seasonal_component) if pattern == 'seasonal' else 0.0,
            'is_anomaly': bool(is_anomaly),
            'pattern': pattern,
        }

        row = {
            'topic': TOPIC,
            'sample_id': str(uuid.uuid4()),
            'params_json': json.dumps(params, sort_keys=True),
            'payload_json': json.dumps(payload, sort_keys=True),
            'timestamp': t.isoformat()
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=['topic', 'sample_id', 'params_json', 'payload_json', 'timestamp'])
    df.to_csv(csv_output_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate deterministic telecom spectrum management sample CSV.')
    parser.add_argument('--csv-output-path', required=True, help='Path to write the UTF-8 CSV file.')
    parser.add_argument('--hyper-param-index', type=int, required=True, help='One-based index of the hyper-parameter configuration to use.')
    parser.add_argument('--total-hyper-params', type=int, required=True, help='Total number of hyper-parameter configurations.')
    args = parser.parse_args()
    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
