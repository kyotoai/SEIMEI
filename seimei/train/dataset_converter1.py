#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from seimei import load_run_messages

DEFAULT_INPUT_PATH = Path("exp11_plasma_gkv_v5/train_v6_results.json")
DEFAULT_OUTPUT_PATH = Path("exp11_plasma_gkv_v5/train_v6_results_converted.json")
DEFAULT_DATASET_PATH = Path("exp11_plasma_gkv_v5/dataset.json")
DEFAULT_RUNS_DIR = Path("seimei_runs")
DEFAULT_COMPARISON_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert train_v6 results into a DPO-friendly message dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to train_v6_results.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write train_v6_results_converted.json.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to dataset.json to map entry ids -> problem indices.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing seimei_runs/(run_id)/messages.json.",
    )
    parser.add_argument(
        "--comparison-threshold",
        type=float,
        default=DEFAULT_COMPARISON_THRESHOLD,
        help="Minimum score difference required to create a comparison pair.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for the output JSON.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        entries = raw.get("detail") or []
        run_cache = raw.get("run_cache") or {}
    elif isinstance(raw, list):
        entries = raw
        run_cache = {}
    else:
        raise ValueError("Input JSON must be a list or a dict with 'detail'.")
    return [entry for entry in entries if isinstance(entry, dict)], run_cache


def _index_run_cache(run_cache: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    run_map: Dict[str, List[Dict[str, Any]]] = {}
    for payload in run_cache.values():
        if not isinstance(payload, dict):
            continue
        run_id = str(payload.get("run_id") or "").strip()
        msg_history = payload.get("msg_history")
        if not run_id or not isinstance(msg_history, list):
            continue
        if run_id not in run_map:
            run_map[run_id] = [dict(msg) for msg in msg_history if isinstance(msg, dict)]
    return run_map


def _normalize_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role == "assistant":
            continue
        normalized.append(dict(msg))
    return normalized


def _is_agent_message(message: Dict[str, Any]) -> bool:
    role = str(message.get("role") or "").lower()
    if role == "agent":
        return True
    if role == "system" and message.get("agent"):
        return True
    return False


def _count_agent_steps(messages: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        if _is_agent_message(msg):
            count += 1
    return count


def _coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_comparisons(
    scores: Sequence[Optional[float]],
    threshold: float,
) -> List[List[int]]:
    pairs: List[List[int]] = []
    for i in range(len(scores)):
        score_i = scores[i]
        if score_i is None:
            continue
        for j in range(i + 1, len(scores)):
            score_j = scores[j]
            if score_j is None:
                continue
            diff = score_i - score_j
            if abs(diff) > threshold:
                chosen, rejected = (i, j) if diff > 0 else (j, i)
                pairs.append([chosen, rejected])
    return pairs


def _load_messages_for_run(
    run_id: str,
    *,
    runs_dir: Path,
    run_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not run_id:
        return []
    cached = run_cache.get(run_id)
    if isinstance(cached, list) and cached:
        return _normalize_messages(cached)
    try:
        messages = load_run_messages(run_id, runs_dir=runs_dir)
    except FileNotFoundError:
        return []
    return _normalize_messages(messages)


def _sanitize_id_component(text: str, *, limit: int = 48) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_", " "}) else " " for ch in text)
    collapsed = "_".join(cleaned.split())
    return collapsed[:limit]


def extract_problem_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("Question") or "").strip()


def compute_entry_id(dataset_entry: Dict[str, Any], index: int) -> str:
    explicit_keys = ["id", "ID", "question_id", "QuestionID", "problem_id", "ProblemID"]
    for key in explicit_keys:
        value = dataset_entry.get(key)
        if value:
            return str(value)
    patch_hint = str(dataset_entry.get("patch_file") or dataset_entry.get("patch_path") or "").strip()
    if patch_hint:
        return patch_hint
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    if csv_path:
        return csv_path
    topic = str(dataset_entry.get("Topic") or "").strip()
    problem = extract_problem_text(dataset_entry)
    hyper = dataset_entry.get("HyperParamIndex")
    sample = dataset_entry.get("SampleIndex")
    parts = [f"idx_{index:05d}"]
    if topic:
        parts.append(_sanitize_id_component(topic))
    elif problem:
        parts.append(_sanitize_id_component(problem))
    if hyper not in (None, ""):
        parts.append(f"hp_{hyper}")
    if sample not in (None, ""):
        parts.append(f"sample_{sample}")
    return "|".join(parts)


def _load_dataset_index_map(dataset_path: Path) -> Dict[str, int]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset.json must be a list of entries.")
    id_map: Dict[str, int] = {}
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        entry_id = compute_entry_id(entry, idx)
        if entry_id and entry_id not in id_map:
            id_map[entry_id] = idx
    return id_map


def _strip_message_fields(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if not role:
            continue
        entry: Dict[str, Any] = {"role": role, "content": msg.get("content", "")}
        name = msg.get("name")
        if isinstance(name, str) and name.strip():
            entry["name"] = name.strip()
        if "knowledge" in msg:
            entry["knowledge"] = msg.get("knowledge")
        stripped.append(entry)
    return stripped


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    entries, run_cache = _load_results(args.input_path)
    run_cache_map = _index_run_cache(run_cache)
    id_map = _load_dataset_index_map(args.dataset_path)

    dataset: List[Dict[str, Any]] = []
    for entry in entries:
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            print("[dataset_converter1] Skipping entry with missing id.")
            continue
        problem_id = id_map.get(entry_id)
        if problem_id is None:
            print(f"[dataset_converter1] No dataset index for entry id: {entry_id}")
            continue

        klg_trials = [
            trial for trial in (entry.get("klg_trials") or [])
            if isinstance(trial, dict)
        ]
        if not klg_trials:
            continue

        message_entries: List[Dict[str, Any]] = []
        scores: List[Optional[float]] = []

        for trial in klg_trials:
            run_id = str(trial.get("run_id") or "").strip()
            messages = _load_messages_for_run(
                run_id,
                runs_dir=args.runs_dir,
                run_cache=run_cache_map,
            )
            if not messages:
                print(f"[dataset_converter1] Missing messages for run: {run_id}")
                continue

            normalized = _normalize_messages(messages)
            n_steps = _count_agent_steps(normalized)
            score = _coerce_score(trial.get("score"))
            message_entries.append(
                {
                    "message": _strip_message_fields(normalized),
                    "n_steps": n_steps,
                    "score": score,
                    "run_id": run_id,
                }
            )
            scores.append(score)

        if not message_entries:
            continue

        comparisons = _build_comparisons(scores, args.comparison_threshold)
        dataset.append(
            {
                "problem_id": problem_id,
                "messages": message_entries,
                "comparison": comparisons,
                "scores": scores,
            }
        )

    if not dataset:
        raise RuntimeError("No converted entries were produced from the input file.")

    dataset.sort(key=lambda item: int(item.get("problem_id", -1)))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataset, ensure_ascii=False, indent=args.indent)
    args.output_path.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(dataset)} entries to {args.output_path}")


if __name__ == "__main__":
    main()
