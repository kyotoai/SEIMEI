#!/usr/bin/env python3
"""
validate_event_model_telecom_v1.py
---------------------------------
Validation aligned with the v1 modeling note:
- Convert events.csv into bin-level labels per (city, bin_start): event_present (0/1) + dominant event_type.
- Convert signal.csv into bin-level features per (city, bin_start) by aggregating across sites/operators.
- Evaluate probabilistic predictions p_event_present using log-loss, Brier, and ECE (calibration).
- If preds.csv is NOT provided, generate a baseline preds.csv from signal (heuristic).

Expected preds.csv format (wide):
- city
- bin_start (ISO8601)
- p_event_present (0..1)
Optional:
- p_type_<event_type> columns that sum to 1 over types (including p_type_none).

Usage:
python validate_event_model_telecom_v1.py \
  --events events.csv \
  --signal signal.csv \
  --out_dir ./eval \
  --bin_minutes 15 \
  [--preds preds.csv]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EVENT_TYPES = [
    "none",
    "english_conversation",
    "coding_meetup",
    "hiking",
    "cooking_class",
    "live_concert",
    "live_music",
    "sports_game",
    "sports_day",
    "anime_convention",
    "job_fair",
    "university_lecture",
    "train_station_rush",
]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def make_event_labels(events: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    # Required columns
    for c in ["city","start_time","end_time","event_type","avg_attendance"]:
        _assert(c in events.columns, f"events missing {c}")

    ev = events.copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"])
    ev["end_time"] = pd.to_datetime(ev["end_time"])

    # Build all bins that appear in signal later; here we only return event intervals
    return ev


def label_bins_from_events(bins: pd.DataFrame, events: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    # bins has city, bin_start, bin_end
    out_rows = []
    events_grp = {k: g for k,g in events.groupby("city")}
    for _, b in bins.iterrows():
        city = b["city"]
        g = events_grp.get(city, None)
        if g is None:
            out_rows.append((0, "none"))
            continue
        overlap = g[(g["start_time"] < b["bin_end"]) & (g["end_time"] > b["bin_start"])]
        if len(overlap) == 0:
            out_rows.append((0, "none"))
        else:
            dom = overlap.sort_values("avg_attendance", ascending=False).iloc[0]
            et = str(dom["event_type"])
            if et not in EVENT_TYPES:
                et = "none"
            out_rows.append((1, et))
    y = pd.DataFrame(out_rows, columns=["event_present","event_type"])
    return pd.concat([bins.reset_index(drop=True), y], axis=1)


def make_signal_bins(signal: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    _assert("timestamp" in signal.columns, "signal missing timestamp")
    _assert("city" in signal.columns, "signal missing city")
    sdf = signal.copy()
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf["bin_start"] = sdf["timestamp"].dt.floor(f"{bin_minutes}min")

    # Aggregate across operator/site (city-level bins)
    kpis = [c for c in [
        "prb_util_pct","dl_mbps","ul_mbps","latency_ms","jitter_ms","packet_loss_pct",
        "volte_mos","bhca","erlangs","call_drop_rate_pct","call_block_rate_pct",
        "RSRP_dBm","SINR_dB","RSRQ_dB"
    ] if c in sdf.columns]
    agg = {c: "mean" for c in kpis}
    agg.update({c+"_p95": (c, lambda x: float(np.percentile(x,95))) for c in kpis})

    out = sdf.groupby(["city","bin_start"]).agg(**agg).reset_index()
    out["bin_end"] = out["bin_start"] + pd.Timedelta(minutes=bin_minutes)

    # Add time features
    out["hour"] = out["bin_start"].dt.hour
    out["dayofweek"] = out["bin_start"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def baseline_predict(signal_bins: pd.DataFrame) -> pd.DataFrame:
    # Weak heuristic baseline (for sanity checking the pipeline).
    # event probability increases with PRB + BHCA + (DL+UL) and decreases with MOS.
    sb = signal_bins.copy()
    # fallback safe zeros
    prb = sb.get("prb_util_pct", 0.0).to_numpy()
    bhca = sb.get("bhca", 0.0).to_numpy()
    dl = sb.get("dl_mbps", 0.0).to_numpy()
    ul = sb.get("ul_mbps", 0.0).to_numpy()
    mos = sb.get("volte_mos", 4.0).to_numpy()

    score = 0.04*(prb-35) + 0.000015*(bhca-20000) + 0.0025*(dl-150) + 0.003*(ul-40) - 0.9*(mos-4.0)
    p = 1/(1+np.exp(-score))
    p = np.clip(p, 0.02, 0.98)
    out = sb[["city","bin_start"]].copy()
    out["p_event_present"] = p

    # Type distribution: near-uniform, with slight bias based on UL/DL ratio
    ul_dl = np.divide(ul, np.maximum(dl, 1e-6))
    types = EVENT_TYPES
    proba = np.ones((len(out), len(types))) / len(types)

    # simple bias: high UL/DL suggests "live_concert" or "anime_convention" (uploads); low suggests "university_lecture"
    hi = ul_dl > 0.35
    lo = ul_dl < 0.12
    for i, t in enumerate(types):
        if t in ["live_concert","anime_convention"] :
            proba[hi, i] *= 1.35
        if t in ["university_lecture"] :
            proba[lo, i] *= 1.35
    proba = proba / proba.sum(axis=1, keepdims=True)

    for i,t in enumerate(types):
        out[f"p_type_{t}"] = proba[:,i]
    return out


def log_loss_binary(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1-eps)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p-y)**2))


def ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum()/len(p)) * abs(acc - conf)
    return float(ece)


def eval_type_logloss(y_type: np.ndarray, preds: pd.DataFrame) -> float:
    # y_type in EVENT_TYPES
    # expects columns p_type_<type>
    cols = [f"p_type_{t}" for t in EVENT_TYPES if f"p_type_{t}" in preds.columns]
    if len(cols) != len(EVENT_TYPES):
        return float("nan")
    proba = preds[cols].to_numpy()
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    idx = np.array([EVENT_TYPES.index(t) if t in EVENT_TYPES else 0 for t in y_type], dtype=int)
    ll = -np.mean(np.log(proba[np.arange(len(idx)), idx]))
    return float(ll)


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plots.")
        return None
    return plt


def plot_calibration(y: np.ndarray, p: np.ndarray, out_dir: Path, n_bins: int = 10) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    accs = []
    confs = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(float(y[mask].mean()))
        confs.append(float(p[mask].mean()))
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Mean predicted p")
    plt.ylabel("Observed event rate")
    plt.title("Calibration (Event Present)")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=150)
    plt.close()


def plot_prob_hist(p: np.ndarray, out_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(p, bins=20, color="#4C72B0", alpha=0.85)
    plt.xlabel("p_event_present")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "p_event_hist.png", dpi=150)
    plt.close()


def plot_timeseries(merged: pd.DataFrame, out_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    if "city" not in merged.columns:
        return
    city = merged["city"].value_counts().idxmax()
    sub = merged[merged["city"] == city].sort_values("bin_start")
    if sub.empty:
        return
    plt.figure(figsize=(8, 3.5))
    plt.plot(sub["bin_start"], sub["p_event_present"], label="p_event_present")
    plt.scatter(sub["bin_start"], sub["event_present"], s=12, alpha=0.7, label="event_present")
    plt.title(f"Event Probability vs Labels ({city})")
    plt.xlabel("Time")
    plt.ylabel("Probability / Label")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"timeseries_{city}.png", dpi=150)
    plt.close()


def plot_kpi_scatter(merged: pd.DataFrame, signal_bins: pd.DataFrame, out_dir: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    joined = merged.merge(signal_bins, on=["city", "bin_start"], how="left", suffixes=("", "_kpi"))
    for kpi in ["prb_util_pct", "dl_mbps", "ul_mbps", "latency_ms"]:
        if kpi not in joined.columns:
            continue
        plt.figure(figsize=(5.5, 4))
        plt.scatter(joined[kpi], joined["p_event_present"], s=10, alpha=0.5)
        plt.xlabel(kpi)
        plt.ylabel("p_event_present")
        plt.title(f"{kpi} vs p_event_present")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{kpi}.png", dpi=150)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--signal", required=True)
    ap.add_argument("--out_dir", default="./eval_v1")
    ap.add_argument("--bin_minutes", type=int, default=15)
    ap.add_argument("--preds", default=None, help="Optional preds.csv; if omitted baseline will be used.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    signal = pd.read_csv(args.signal)

    # bins from signal
    signal_bins = make_signal_bins(signal, args.bin_minutes)
    events_ev = make_event_labels(events, args.bin_minutes)

    bins = signal_bins[["city","bin_start","bin_end"]].drop_duplicates().sort_values(["city","bin_start"])
    labeled = label_bins_from_events(bins, events_ev, args.bin_minutes)

    if args.preds:
        preds = pd.read_csv(args.preds)
    else:
        preds = baseline_predict(signal_bins)
        preds.to_csv(os.path.join(args.out_dir, "baseline_preds.csv"), index=False)

    # join
    merged = labeled.merge(preds, on=["city","bin_start"], how="left")
    _assert("p_event_present" in merged.columns, "preds must include p_event_present")
    _assert(merged["p_event_present"].isna().sum() == 0, "Missing predictions for some bins")
    merged["bin_start"] = pd.to_datetime(merged["bin_start"])

    y = merged["event_present"].to_numpy().astype(float)
    p = merged["p_event_present"].to_numpy().astype(float)

    metrics = {
        "rows": int(len(merged)),
        "positive_rate": float(y.mean()),
        "logloss_event_present": log_loss_binary(y, p),
        "logloss_event_present_mean": log_loss_binary(y, np.full_like(y, y.mean())),
        "logloss_event_present_05": log_loss_binary(y, np.full_like(y, 0.5)),
        "brier_event_present": brier(y, p),
        "ece_event_present": ece_binary(y, p, n_bins=10),
    }

    # optional type logloss
    tll = eval_type_logloss(merged["event_type"].astype(str).to_numpy(), merged)
    if not np.isnan(tll):
        metrics["logloss_event_type"] = tll

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    merged_out = merged[["city","bin_start","event_present","event_type","p_event_present"] + [c for c in merged.columns if c.startswith("p_type_")]]
    merged_out.to_csv(out_dir / "labels_and_preds.csv", index=False)

    plot_calibration(y, p, out_dir, n_bins=10)
    plot_prob_hist(p, out_dir)
    plot_timeseries(merged, out_dir)
    plot_kpi_scatter(merged, signal_bins, out_dir)

    print("Wrote metrics to", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
