import csv
import json
import uuid
import datetime
import math
import numpy as np


def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
    # Clamp and normalize hyper-parameter index to be 1-based within the provided total
    if total_hyper_params <= 0:
        total_hyper_params = 1
    idx = max(1, min(int(hyper_param_index), int(total_hyper_params)))

    # Deterministic RNG per hyper-parameter index to ensure reproducibility
    rng = np.random.default_rng(1729 + idx)

    topic = "online_ad_campaign_roi"

    # Hyper-parameter configuration per index (1..total_hyper_params)
    # For this sample with total_hyper_params == 1, we fix a single configuration.
    # Other indices would yield alternative parameterizations and distinct data.
    if idx == 1:
        config = {
            "campaign_type": "brand_awareness",
            "seasonality_factor": 1.15,
            "uplift_factor": 1.25,
            "base_ctr": 0.012,
            "base_cr": 0.02,
            "revenue_per_conversion": 150
        }
    else:
        # Fallback default (should not be used in this dataset, but keeps the code robust)
        config = {
            "campaign_type": "brand_awareness",
            "seasonality_factor": 1.0,
            "uplift_factor": 1.0,
            "base_ctr": 0.012,
            "base_cr": 0.02,
            "revenue_per_conversion": 120
        }

    total_days = 20
    base_impressions = 8000
    daily_spend_base = 1500
    base_date = datetime.datetime(2025, 1, 1, 12, 0, 0)

    rows = []
    for day in range(total_days):
        # Seasonal component modulated by a sine wave to create seasonality-like variation
        day_factor = 1.0 + 0.15 * math.sin(day / 2.0)
        impressions = int(base_impressions * config["seasonality_factor"] * day_factor)

        # Slight random perturbation around baseline CTR
        ctr = config["base_ctr"] * (1.0 + 0.25 * rng.normal())
        clicks = int(impressions * max(0.0001, ctr))

        # Slight random perturbation around baseline CR
        cr = config["base_cr"] * (1.0 + 0.15 * rng.normal())
        conversions = int(clicks * max(0.0, cr))

        # Spend grows gently over time and is scaled by uplift_factor
        spend = int(daily_spend_base * (1.0 + 0.08 * day / total_days) * config["uplift_factor"])
        revenue = conversions * config["revenue_per_conversion"]
        roi = (revenue - spend) / spend if spend > 0 else 0.0

        sample_id = str(uuid.uuid4())
        timestamp = (base_date + datetime.timedelta(days=day)).isoformat()

        params_json = json.dumps({
            "campaign_type": config["campaign_type"],
            "seasonality_factor": config["seasonality_factor"],
            "uplift_factor": config["uplift_factor"],
            "base_ctr": config["base_ctr"],
            "base_cr": config["base_cr"],
            "revenue_per_conversion": config["revenue_per_conversion"]
        }, separators=(",", ":"), ensure_ascii=False)

        payload_json = json.dumps({
            "day_index": day,
            "daily_impressions": impressions,
            "daily_clicks": clicks,
            "conversions": conversions,
            "spend": spend,
            "revenue": revenue,
            "roi": roi,
            "seasonality_factor": config["seasonality_factor"],
            "uplift_factor": config["uplift_factor"],
            "campaign_type": config["campaign_type"]
        }, separators=(",", ":"), ensure_ascii=False)

        rows.append({
            "topic": topic,
            "sample_id": sample_id,
            "params_json": params_json,
            "payload_json": payload_json,
            "timestamp": timestamp
        })

    # Write UTF-8 CSV with header
    with open(csv_output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "sample_id", "params_json", "payload_json", "timestamp"])
        for r in rows:
            writer.writerow([r["topic"], r["sample_id"], r["params_json"], r["payload_json"], r["timestamp"]])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-output-path", required=True)
    parser.add_argument("--hyper-param-index", type=int, required=True)
    parser.add_argument("--total-hyper-params", type=int, required=True)
    args = parser.parse_args()

    generate(args.csv_output_path, args.hyper_param_index, args.total_hyper_params)
