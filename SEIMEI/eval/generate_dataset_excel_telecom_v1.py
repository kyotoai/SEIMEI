#!/usr/bin/env python3
"""
generate_dataset_excel_telecom_v1.py
-----------------------------------
Orchestrator that:
1) calls an LLM (via seimei.llm.LLMClient) with a prompt template
2) receives JSON with python_features + python_signal modules
3) executes each module for hyper_param_index=1..N to create:
   - events_{stub}_events_{h}.csv
   - signal_{stub}_signal_{h}.csv
4) writes dataset.json index

This v1 expects the LLM modules to write *final CSVs directly* (no payload_json expansion).

NEW in this version:
- You can pass topics either via:
  * --topics <t1> <t2> ...
  * --topics-path <json-file>
The topics file can be:
  * JSON list: ["t1","t2",...]
  * JSON object: {"topics":["t1","t2",...]}
  * (optional) list with // comments and trailing commas, e.g.
    [
    commet...
      "t1",
      "t2",
    ]

Usage:
python generate_dataset_excel_telecom_v1.py \
  --prompt-path excel_events_telecom_v1.md \
  --exp-dir ./exp_v1 \
  --topics-path excel_topics.json \
  --n-samples-per-topic 2 \
  --n-hyper-params 4 \
  --model gpt-5-nano

Notes:
- If your environment sets up LLMClient differently, pass extra key=val via --llm-kv.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import py_compile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from seimei.llm import LLMClient


SYSTEM_PROMPT = """You are a telecom synthetic-data assistant.
Return EXACTLY ONE JSON object and NOTHING ELSE (no markdown fences, no commentary).
The JSON must contain exactly these keys:
- topic
- features
- python_features
- python_signal
- question
- correct_answer
The topic must exactly match the requested topic string.

python_features and python_signal must each be a full Python module source code as a string.

The module MUST write the final CSV directly to the given --csv-output-path (no subfolders, no extra filenames).
Do NOT create directories named *.csv or *_out_h*.
"""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def safe_format(template: str, values: Dict[str, Any]) -> str:
    class _D(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(_D(values))


def extract_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    # strip code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    # try direct
    try:
        return json.loads(s)
    except Exception:
        # find the longest JSON object span (handles extra text)
        matches = re.findall(r"\{.*\}", s, re.S)
        if not matches:
            raise ValueError("No JSON object found.")
        candidate = max(matches, key=len)
        print("=== CANDIDATE JSON ===", flush=True)
        print(candidate[:500], flush=True)
        return json.loads(candidate)




def _has_mojibake(series) -> bool:
    return series.astype(str).str.contains(r"[À-ÿ]").any()

def _has_kanji(series) -> bool:
    return series.astype(str).str.contains(r"[\u4E00-\u9FFF]").any()

def _has_kana(series) -> bool:
    return series.astype(str).str.contains(r"[\u3040-\u309F\u30A0-\u30FF]").any()

def _has_japanese(series) -> bool:
    return _has_kanji(series) or _has_kana(series)

def _normalize_module_source(code: str) -> str:
    # Some model responses double-escape newlines ("\n"); unescape if needed.
    if "\n" not in code and "\\n" in code:
        try:
            return bytes(code, "utf-8").decode("unicode_escape")
        except Exception:
            return code.replace("\\n", "\n")
    return code


def _fix_module_source(code: str) -> str:
    fixed = code
    # Fix argparse Namespace subscripting (common LLM mistake).
    fixed = fixed.replace('args["--csv-output-path"]', "args.csv_output_path")
    fixed = fixed.replace('args["--hyper-param-index"]', "args.hyper_param_index")
    fixed = fixed.replace('args["--total-hyper-params"]', "args.total_hyper_params")
    fixed = fixed.replace("args['--csv-output-path']", "args.csv_output_path")
    fixed = fixed.replace("args['--hyper-param-index']", "args.hyper_param_index")
    fixed = fixed.replace("args['--total-hyper-params']", "args.total_hyper_params")
    fixed = fixed.replace('args["csv-output-path"]', "args.csv_output_path")
    fixed = fixed.replace("args['csv-output-path']", "args.csv_output_path")
    fixed = fixed.replace("args.csv-output_path", "args.csv_output_path")
    fixed = fixed.replace("args.csv-output-path", "args.csv_output_path")
    fixed = fixed.replace("Argument Parser", "ArgumentParser")
    fixed = fixed.replace(" Asia/Tokyo", "Asia/Tokyo")
    # Fix common typo for pandas alias.
    fixed = fixed.replace("PD.", "pd.")
    # Ensure argparse is imported if used.
    if "argparse.ArgumentParser" in fixed and "import argparse" not in fixed:
        fixed = "import argparse\n" + fixed
    # Normalize timedelta usage to avoid numpy.int64 errors.
    if "timedelta(" in fixed and "_timedelta(" not in fixed:
        fixed = fixed.replace("timedelta(", "_timedelta(")
        fixed = (
            "def _timedelta(**kwargs):\n"
            "    from datetime import timedelta as _td\n"
            "    for k, v in kwargs.items():\n"
            "        try:\n"
            "            kwargs[k] = int(v)\n"
            "        except Exception:\n"
            "            pass\n"
            "    return _td(**kwargs)\n\n"
            + fixed
        )
    # Guardrail: ensure generate(...) exists.
    if "def generate(" not in fixed:
        fixed = "def generate(*_args, **_kwargs):\n    raise RuntimeError('generate() missing in module')\n\n" + fixed
    return fixed

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_module(module_path: Path, csv_out: Path, hyper_i: int, total_h: int, timeout_s: int) -> Tuple[bool, str]:
    # Guard against a directory artifact where a CSV file should be.
    if csv_out.exists() and csv_out.is_dir():
        try:
            for child in csv_out.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    # best-effort cleanup of nested dirs
                    for sub in child.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            csv_out.rmdir()
        except Exception as e:
            return False, f"csv_out is a directory and could not be removed: {e}"
    cmd = [
        sys.executable,
        str(module_path),
        "--csv-output-path", str(csv_out),
        "--hyper-param-index", str(hyper_i),
        "--total-hyper-params", str(total_h),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    if proc.returncode != 0:
        return False, f"returncode={proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    if not csv_out.exists() or not csv_out.is_file() or csv_out.stat().st_size == 0:
        return False, "CSV not created or empty."
    return True, proc.stdout[-1000:]


def _cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    try:
        if path.is_file():
            path.unlink()
            return
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass
    except OSError:
        pass


def validate_events_csv(path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(path)
    required = [
        "event_id","event_type","prefecture","city","region_type",
        "start_time","end_time","duration_minutes","avg_attendance",
        "event_quality","age_mean","male_pct","female_pct"
    ]
    # Guard against signal schema accidentally written to events.csv.
    if "timestamp" in df.columns or "operator" in df.columns:
        raise ValueError("events.csv looks like signal.csv (timestamp/operator present)")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"events.csv missing columns: {missing}")
    if df["event_id"].duplicated().any():
        raise ValueError("events.csv has duplicate event_id")
    if len(df) < 10 or len(df) > 20:
        raise ValueError("events.csv should contain 10–20 events")
    if ((df["male_pct"] + df["female_pct"]) != 100).any():
        raise ValueError("events.csv has male+female != 100")
    # Catch mojibake from incorrect encoding.
    for col in ["prefecture", "city"]:
        if col in df.columns and (df[col].astype(str).str.contains("�").any() or _has_mojibake(df[col])):
            raise ValueError(f"events.csv has mojibake in {col}")
        if col in df.columns and not _has_japanese(df[col]):
            raise ValueError(f"events.csv has non-kanji {col}")
    # Enforce known prefecture/city lists to avoid garbled labels.
    allowed_pref = {"東京都","大阪府","京都府","愛知県","北海道","福岡県","宮城県"}
    allowed_city = {"東京23区","横浜","大阪市","京都市","神戸市","名古屋市","札幌市","福岡市","仙台市"}
    if "prefecture" in df.columns and not df["prefecture"].isin(allowed_pref).all():
        raise ValueError("events.csv has unknown prefecture labels")
    if "city" in df.columns and not df["city"].isin(allowed_city).all():
        raise ValueError("events.csv has unknown city labels")
    # Plausibility: large venues should have at least 100 attendees.
    if "region_type" in df.columns and "avg_attendance" in df.columns:
        bad = df[(df["region_type"] == "stadium_event") & (df["avg_attendance"] < 100)]
        if not bad.empty:
            raise ValueError("events.csv has stadium_event with avg_attendance < 100")
    # Scale checks by event_type.
    if "event_type" in df.columns and "avg_attendance" in df.columns:
        small_types = ["english_conversation", "coding_meetup", "cooking_class", "live_music"]
        small = df[df["event_type"].isin(small_types) & ((df["avg_attendance"] < 10) | (df["avg_attendance"] > 150))]
        if not small.empty:
            raise ValueError("events.csv has small-scale event_type outside 10–150 attendance")
        sports_day = df[df["event_type"] == "sports_day"]
        if not sports_day.empty and ((sports_day["avg_attendance"] < 50) | (sports_day["avg_attendance"] > 800)).any():
            raise ValueError("events.csv has sports_day outside 50–800 attendance")
        lc = df[df["event_type"] == "live_concert"]
        if not lc.empty and (lc["avg_attendance"] < 100).any():
            raise ValueError("events.csv has live_concert with avg_attendance < 100")
        stadium_lc = df[(df["event_type"] == "live_concert") & (df["region_type"] == "stadium_event")]
        if not stadium_lc.empty and (stadium_lc["avg_attendance"] < 1000).any():
            raise ValueError("events.csv has stadium live_concert with avg_attendance < 1000")
    # Region-type plausibility constraints.
    if "event_type" in df.columns and "region_type" in df.columns:
        allowed = {
            "stadium_event": {"sports_game", "live_concert", "anime_convention"},
            "subway": {"train_station_rush"},
            "airport": {"train_station_rush", "job_fair"},
            "dense_urban": {"english_conversation", "coding_meetup", "job_fair", "university_lecture", "live_concert", "live_music"},
            "indoor_mall": {"anime_convention", "english_conversation", "coding_meetup", "job_fair", "university_lecture", "live_concert", "live_music"},
            "coastal": {"hiking", "cooking_class"},
            "rural_mountain": {"hiking", "cooking_class"},
            "suburbs_urban": {"cooking_class", "university_lecture", "english_conversation", "coding_meetup", "live_music"},
            "campus_ground": {"sports_day", "live_music", "live_concert", "english_conversation", "job_fair", "university_lecture"},
            "campus_hall": {"live_music", "live_concert", "english_conversation", "job_fair", "university_lecture"},
        }
        bad_rows = []
        for rt, et in df[["region_type", "event_type"]].astype(str).itertuples(index=False):
            if rt in allowed and et not in allowed[rt]:
                bad_rows.append((rt, et))
        if bad_rows:
            raise ValueError(f"events.csv has implausible region/event pairs: {bad_rows[:5]}")


def validate_topic_alignment(topic: str, events_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(events_path)
    t = topic.lower()
    # Minimal topic-driven constraints to keep outputs aligned with the topic intent.
    checks = []
    strict_sets = []
    if "anime_convention" in t:
        strict_sets.append(("event_type", {"anime_convention"}))
        checks.append(("region_type", {"indoor_mall", "stadium_event"}))
    if "stadium_event_congestion" in t:
        strict_sets.append(("event_type", {"sports_game", "live_concert"}))
        checks.append(("region_type", {"stadium_event"}))
    if "sports_game" in t and "stadium" not in t:
        strict_sets.append(("event_type", {"sports_game"}))
    if "concert" in t:
        strict_sets.append(("event_type", {"live_concert", "live_music"}))
    if "marathon" in t:
        strict_sets.append(("event_type", {"sports_game"}))
        checks.append(("region_type", {"dense_urban", "suburbs_urban"}))
    if "university_festival" in t or "campus_crowd" in t:
        strict_sets.append(("event_type", {"university_lecture", "english_conversation", "job_fair", "live_music", "live_concert", "sports_day"}))
        checks.append(("region_type", {"campus_ground", "campus_hall"}))
    if "language_exchange" in t or "english" in t or "meetup" in t or "omiai" in t:
        strict_sets.append(("event_type", {"english_conversation", "coding_meetup", "job_fair", "university_lecture", "live_music"}))
    for col, allowed in strict_sets:
        if col not in df.columns:
            raise ValueError(f"events.csv missing {col} for topic alignment")
        if not df[col].isin(allowed).all():
            raise ValueError(f"events.csv not aligned with topic '{topic}': {col} outside {sorted(allowed)}")

    for col, allowed in checks:
        if col not in df.columns:
            raise ValueError(f"events.csv missing {col} for topic alignment")
        if not df[col].isin(allowed).any():
            raise ValueError(f"events.csv not aligned with topic '{topic}': no {col} in {sorted(allowed)}")


def validate_signal_csv(path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(path)
    required = [
        "timestamp","operator","prefecture","city","region_type","site_id",
        "RSRP_dBm","SINR_dB","RSRQ_dB","prb_util_pct",
        "dl_mbps","ul_mbps","latency_ms","jitter_ms","packet_loss_pct","volte_mos",
        "bhca","erlangs","call_drop_rate_pct","call_block_rate_pct"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"signal.csv missing columns: {missing}")
    # Guard against events schema accidentally written to signal.csv.
    if "event_id" in df.columns or "event_type" in df.columns:
        raise ValueError("signal.csv looks like events.csv (event_id/event_type present)")
    # Catch mojibake from incorrect encoding.
    for col in ["prefecture", "city", "operator"]:
        if col in df.columns and (df[col].astype(str).str.contains("�").any() or _has_mojibake(df[col])):
            raise ValueError(f"signal.csv has mojibake in {col}")
        if col in ["prefecture", "city"] and col in df.columns and not _has_japanese(df[col]):
            raise ValueError(f"signal.csv has non-kanji {col}")
    # Enforce known prefecture/city/operator lists to avoid garbled labels.
    allowed_pref = {"東京都","大阪府","京都府","愛知県","北海道","福岡県","宮城県"}
    allowed_city = {"東京23区","横浜","大阪市","京都市","神戸市","名古屋市","札幌市","福岡市","仙台市"}
    allowed_ops = {"NTTドコモ","KDDI/au","ソフトバンク","楽天モバイル"}
    if "prefecture" in df.columns and not df["prefecture"].isin(allowed_pref).all():
        raise ValueError("signal.csv has unknown prefecture labels")
    if "city" in df.columns and not df["city"].isin(allowed_city).all():
        raise ValueError("signal.csv has unknown city labels")
    if "operator" in df.columns and not df["operator"].isin(allowed_ops).all():
        raise ValueError("signal.csv has unknown operator labels")
    # Range checks for key KPIs (basic sanity).
    ranges = {
        "RSRP_dBm": (-120, -70),
        "SINR_dB": (-5, 30),
        "RSRQ_dB": (-20, -3),
        "prb_util_pct": (0, 100),
        "dl_mbps": (0.1, 2000),
        "ul_mbps": (0.05, 800),
        "latency_ms": (5, 200),
        "jitter_ms": (0, 80),
        "packet_loss_pct": (0, 5),
        "volte_mos": (1.0, 4.5),
        "bhca": (1000, 2_000_000),
        "erlangs": (0, 60),
        "call_drop_rate_pct": (0, 5),
        "call_block_rate_pct": (0, 8),
    }
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"signal.csv has non-numeric values in {col}")
        if ((vals < lo) | (vals > hi)).any():
            raise ValueError(f"signal.csv has out-of-range values in {col}")


def load_topics_from_file(path: Path) -> List[str]:
    """
    Supports:
      - JSON list: ["t1","t2",...]
      - JSON object: {"topics":["t1","t2",...]}
      - JSON list with // comments and trailing commas before ]
    """
    if not path.is_file():
        raise FileNotFoundError(f"Topics file not found: {path}")

    raw = path.read_text(encoding="utf-8").strip()

    # Try strict JSON first
    payload: Any
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None

    # Accept {"topics":[...]} or [...]
    if isinstance(payload, dict):
        payload = payload.get("topics")

    if isinstance(payload, list):
        topics = payload
    else:
        # Relaxed mode: remove // comments + trailing commas before ]
        no_comments = re.sub(r"//.*$", "", raw, flags=re.MULTILINE)
        no_trailing = re.sub(r",\s*\]", "]", no_comments)
        topics = json.loads(no_trailing)

    if not isinstance(topics, list) or not topics:
        raise ValueError("Topics file must contain a non-empty list of strings.")

    out: List[str] = []
    for i, t in enumerate(topics):
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"Topic at index {i} must be a non-empty string.")
        out.append(t.strip())

    if len(set(out)) != len(out):
        raise ValueError("Topics list contains duplicates.")

    return out


@dataclass
class Record:
    topic: str
    sample_index: int
    hyper_param_index: int
    question: str
    correct_answer: str
    features: Dict[str, Any]
    python_features_path: str
    python_signal_path: str
    events_csv_path: str
    signal_csv_path: str


async def generate_one(
    llm: LLMClient,
    base_prompt: str,
    *,
    topic: str,
    sample_index: int,
    total_samples: int,
    n_hyper: int,
    exp_dir: Path,
    timeout_s: int,
    max_attempts: int,
    skip_existing: bool
) -> List[Record]:
    def log(msg: str) -> None:
        print(msg, flush=True)

    slug = slugify(topic)
    stub = f"{slug}_{sample_index}"
    values = {
        "topic": topic,
        "sample_index": sample_index,
        "total_samples": total_samples,
        "n_hyper_params": n_hyper,
        "file_stub": stub,
        "features": topic,  # backwards compat if template references {features}
    }
    prompt = safe_format(base_prompt, values)

    # Pre-check: if all expected CSVs exist, reuse without calling the LLM.
    if skip_existing:
        out_dir = exp_dir / "csv"
        all_exist = True
        for h in range(1, n_hyper + 1):
            ev_csv = out_dir / f"{stub}_events_{h}.csv"
            sg_csv = out_dir / f"{stub}_signal_{h}.csv"
            if not (ev_csv.is_file() and sg_csv.is_file()):
                all_exist = False
                break
            try:
                validate_events_csv(ev_csv)
                validate_signal_csv(sg_csv)
            except Exception:
                all_exist = False
                break

        if all_exist:
            log(f"[topic={topic}] skip LLM: all CSVs already exist for sample {sample_index}")
            records = []
            for h in range(1, n_hyper + 1):
                ev_csv = out_dir / f"{stub}_events_{h}.csv"
                sg_csv = out_dir / f"{stub}_signal_{h}.csv"
                records.append(Record(
                    topic=topic,
                    sample_index=sample_index,
                    hyper_param_index=h,
                    question="",
                    correct_answer="",
                    features={"notes": "reused existing CSVs; LLM skipped"},
                    python_features_path="",
                    python_signal_path="",
                    events_csv_path=str(ev_csv),
                    signal_csv_path=str(sg_csv),
                ))
            return records

    messages = [{"role": "user", "content": prompt}]
    last_err = None

    log(f"[topic={topic}] start sample {sample_index}/{total_samples} (hyper={n_hyper})")
    for _attempt in range(1, max_attempts + 1):
        log(f"[topic={topic}] attempt {_attempt}/{max_attempts} requesting LLM...")
        try:
            resp_text, _meta = await llm.chat(messages=messages, system=SYSTEM_PROMPT)
        except Exception as e:
            last_err = f"LLM request failed: {e}"
            log(f"[topic={topic}] error: {last_err}")
            await asyncio.sleep(2)
            continue
        try:
            payload = extract_json(resp_text)
        except Exception as e:
            last_err = f"JSON parse error: {e}"
            log(f"[topic={topic}] error: {last_err}")
            messages.append({"role": "assistant", "content": resp_text})
            messages.append({"role": "user", "content": f"Your response was not valid JSON. Error: {e}. Return ONLY the JSON object."})
            continue

        # schema
        required_keys = ["topic", "features", "python_features", "python_signal", "question", "correct_answer"]
        missing_keys = [k for k in required_keys if k not in payload]
        if missing_keys:
            last_err = f"missing key(s): {missing_keys}"
            log(f"[topic={topic}] error: {last_err}")
        elif payload["topic"] != topic:
            last_err = f"topic mismatch: got {payload['topic']}"
            log(f"[topic={topic}] error: {last_err}")
        else:
            # write modules
            py_dir = exp_dir / "python"
            out_dir = exp_dir / "csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            py_features_path = py_dir / f"{stub}_features.py"
            py_signal_path = py_dir / f"{stub}_signal.py"
            pf = _fix_module_source(_normalize_module_source(str(payload["python_features"])))
            ps = _fix_module_source(_normalize_module_source(str(payload["python_signal"])))
            write_text(py_features_path, pf)
            write_text(py_signal_path, ps)
            log(f"[topic={topic}] wrote modules: {py_features_path.name}, {py_signal_path.name}")

            # Fast syntax check before execution to catch broken code early.
            try:
                py_compile.compile(str(py_features_path), doraise=True)
                py_compile.compile(str(py_signal_path), doraise=True)
            except py_compile.PyCompileError as e:
                last_err = f"module syntax error: {e.msg}"
                log(f"[topic={topic}] error: {last_err}")
                continue

            records: List[Record] = []
            for h in range(1, n_hyper + 1):
                log(f"[topic={topic}] run hyper {h}/{n_hyper}")
                ev_csv = out_dir / f"{stub}_events_{h}.csv"
                sg_csv = out_dir / f"{stub}_signal_{h}.csv"

                if skip_existing:
                        # Skip existing outputs during debug runs.
                    if ev_csv.exists() or sg_csv.exists():
                        if ev_csv.is_dir() or sg_csv.is_dir():
                            log(f"[topic={topic}] skip hyper {h}/{n_hyper}: output path is a directory")
                            continue
                        if ev_csv.is_file() and sg_csv.is_file():
                            try:
                                validate_events_csv(ev_csv)
                                validate_signal_csv(sg_csv)
                            except Exception as e:
                                last_err = f"validation failed for existing h={h}: {e}"
                                log(f"[topic={topic}] error: {last_err}")
                                break
                            records.append(Record(
                                topic=topic,
                                sample_index=sample_index,
                                hyper_param_index=h,
                                question=str(payload["question"]),
                                correct_answer=str(payload["correct_answer"]),
                                features=payload["features"] if isinstance(payload["features"], dict) else {"notes": str(payload["features"])},
                                python_features_path=str(py_features_path),
                                python_signal_path=str(py_signal_path),
                                events_csv_path=str(ev_csv),
                                signal_csv_path=str(sg_csv),
                            ))
                            log(f"[topic={topic}] reused existing CSVs for hyper {h}/{n_hyper}")
                            continue

                ok, msg = run_module(py_features_path, ev_csv, h, n_hyper, timeout_s)
                if not ok:
                    last_err = f"features module failed for h={h}: {msg}"
                    _cleanup_path(ev_csv)
                    log(f"[topic={topic}] error: {last_err}")
                    break

                ok, msg = run_module(py_signal_path, sg_csv, h, n_hyper, timeout_s)
                if not ok:
                    last_err = f"signal module failed for h={h}: {msg}"
                    _cleanup_path(sg_csv)
                    log(f"[topic={topic}] error: {last_err}")
                    break

                # checkpoints
                try:
                    validate_events_csv(ev_csv)
                    validate_signal_csv(sg_csv)
                    validate_topic_alignment(topic, ev_csv)
                except Exception as e:
                    last_err = f"validation failed for h={h}: {e}"
                    _cleanup_path(ev_csv)
                    _cleanup_path(sg_csv)
                    log(f"[topic={topic}] error: {last_err}")
                    break

                records.append(Record(
                    topic=topic,
                    sample_index=sample_index,
                    hyper_param_index=h,
                    question=str(payload["question"]),
                    correct_answer=str(payload["correct_answer"]),
                    features=payload["features"] if isinstance(payload["features"], dict) else {"notes": str(payload["features"])},
                    python_features_path=str(py_features_path),
                    python_signal_path=str(py_signal_path),
                    events_csv_path=str(ev_csv),
                    signal_csv_path=str(sg_csv),
                ))

            if records:
                log(f"[topic={topic}] completed sample {sample_index}")
                return records

        # retry
        messages.append({"role": "assistant", "content": resp_text})
        messages.append({"role": "user", "content": f"Fix the issue ({last_err}) and return ONLY the JSON object with the required keys."})

    raise RuntimeError(
        f"Failed to generate sample {topic}:{sample_index} after {max_attempts} attempts. Last error: {last_err}"
    )


async def main_async(args: argparse.Namespace) -> None:
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = Path(args.prompt_path).read_text(encoding="utf-8")
    llm = LLMClient(model=args.model, max_concurrent_requests=args.batch_size, **args.llm_kv)

    print(f"[config] exp_dir={exp_dir}", flush=True)
    print(f"[config] prompt_path={Path(args.prompt_path).resolve()}", flush=True)
    print(f"[config] topics={len(args.topics)} model={args.model}", flush=True)

    all_records: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, int(args.batch_size)))

    async def _run_one(topic: str, sidx: int) -> List[Record]:
        async with sem:
            return await generate_one(
                llm,
                base_prompt,
                topic=topic,
                sample_index=sidx,
                total_samples=args.n_samples_per_topic,
                n_hyper=args.n_hyper_params,
                exp_dir=exp_dir,
                timeout_s=args.exec_timeout,
                max_attempts=args.max_attempts,
                skip_existing=args.skip_existing,
            )

    tasks = []
    for topic in args.topics:
        for sidx in range(1, args.n_samples_per_topic + 1):
            tasks.append(asyncio.create_task(_run_one(topic, sidx)))

    for coro in asyncio.as_completed(tasks):
        try:
            recs = await coro
        except Exception as e:
            print(f"[error] task failed: {e}", flush=True)
            continue
        for r in recs:
            all_records.append({
                "Topic": r.topic,
                "SampleIndex": r.sample_index,
                "HyperParamIndex": r.hyper_param_index,
                "Question": r.question,
                "CorrectAnswer": r.correct_answer,
                "FeaturesJSON": r.features,
                "PythonFeaturesPath": r.python_features_path,
                "PythonSignalPath": r.python_signal_path,
                "EventsCSVPath": r.events_csv_path,
                "SignalCSVPath": r.signal_csv_path,
            })
        (exp_dir / "dataset.json").write_text(
            json.dumps(all_records, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[dataset] records={len(all_records)}", flush=True)

    print(f"Wrote {len(all_records)} records to {exp_dir/'dataset.json'}")


def parse_kv(pairs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Expected key=value, got {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ["true", "false"]:
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except Exception:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-path", required=True)
    ap.add_argument("--exp-dir", required=True)

    # Either pass topics directly or via a file
    ap.add_argument("--topics", nargs="*", default=[], help="Topic strings (space-separated)")
    ap.add_argument("--topics-path", default=None, help="Path to JSON topics file (list or {\"topics\": [...]})")

    ap.add_argument("--n-samples-per-topic", type=int, default=1)
    ap.add_argument("--n-hyper-params", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching; currently unused.")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--llm-kv", nargs="*", default=[], help="Extra LLMClient kwargs as key=value")
    ap.add_argument("--exec-timeout", type=int, default=90)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true", help="Skip LLM if all CSVs already exist for a sample")

    args = ap.parse_args()
    args.llm_kv = parse_kv(args.llm_kv)

    if args.topics_path:
        args.topics = load_topics_from_file(Path(args.topics_path))

    if not args.topics:
        raise ValueError("Provide either --topics <...> or --topics-path <file> with at least 1 topic.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
