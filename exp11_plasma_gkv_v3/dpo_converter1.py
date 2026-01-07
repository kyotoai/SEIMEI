#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from seimei import load_run_messages

DEFAULT_INPUT_PATH = Path("exp11_plasma_gkv_v3/train_v4_eval_sample_results2.json")
DEFAULT_OUTPUT_PATH = Path("exp11_plasma_gkv_v3/train_v4_eval_sample_dpo2.json")
# MODIFY...
DEFAULT_RUNS_DIR = Path("seimei_runs")
DEFAULT_N_KNOWLEDGE_STEPS = 3
DEFAULT_KNOWLEDGE_PER_STEP = 3
DEFAULT_COMPARISON_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert train_v3_eval_sample results into DPO-friendly entries."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to train_v3_eval_sample results JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the converted dataset.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing seimei_runs/(run_id)/messages.json.",
    )
    parser.add_argument(
        "--n-knowledge-steps",
        type=int,
        default=DEFAULT_N_KNOWLEDGE_STEPS,
        help="Number of knowledge steps to extract per entry.",
    )
    parser.add_argument(
        "--knowledge-per-step",
        type=int,
        default=DEFAULT_KNOWLEDGE_PER_STEP,
        help="Number of knowledge chunks per step (one from each chunk).",
    )
    parser.add_argument(
        "--comparison-threshold",
        type=float,
        default=DEFAULT_COMPARISON_THRESHOLD,
        help="Minimum score difference required to create a DPO comparison pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for selecting trials.",
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


def _truncate_messages_before_step(
    messages: Sequence[Dict[str, Any]],
    step: int,
) -> List[Dict[str, Any]]:
    if step <= 1:
        truncated: List[Dict[str, Any]] = []
        for msg in messages:
            if _is_agent_message(msg):
                break
            truncated.append(dict(msg))
        return truncated

    truncated = []
    agent_seen = 0
    for msg in messages:
        if _is_agent_message(msg):
            agent_seen += 1
            if agent_seen >= step:
                break
        truncated.append(dict(msg))
    return truncated


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


def _select_trials(
    trials: Sequence[Dict[str, Any]],
    count: int,
    rng: random.Random,
) -> List[Optional[Dict[str, Any]]]:
    if count <= 0:
        return []
    if not trials:
        return [None] * count
    order = list(range(len(trials)))
    rng.shuffle(order)
    selected: List[Optional[Dict[str, Any]]] = []
    idx = 0
    for _ in range(count):
        if idx >= len(order):
            rng.shuffle(order)
            idx = 0
        selected.append(trials[order[idx]])
        idx += 1
    return selected


def _load_messages_for_trial(
    trial: Optional[Dict[str, Any]],
    *,
    runs_dir: Path,
    run_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not trial:
        return []
    run_id = str(trial.get("run_id") or "").strip()
    if not run_id:
        return []
    try:
        messages = load_run_messages(run_id, runs_dir=runs_dir)
    except FileNotFoundError:
        cached = run_cache.get(run_id) or []
        messages = cached
    return _normalize_messages(messages)


def _build_knowledge_entry(
    knowledge: Dict[str, Any],
    *,
    entry_id: str,
    chunk_index: int,
    step: int,
) -> Optional[Dict[str, Any]]:
    text = (
        knowledge.get("text")
        or knowledge.get("knowledge")
        or knowledge.get("original_text")
        or ""
    )
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "id": knowledge.get("id") or f"{entry_id}_c{chunk_index}_s{step}",
        "text": text.strip(),
        "agent": knowledge.get("agent"),
        "tags": knowledge.get("tags") or [],
    }


def _extract_step_knowledge(
    chunks: Sequence[Dict[str, Any]],
    *,
    step: int,
    entry_id: str,
) -> Optional[List[Dict[str, Any]]]:
    knowledge_entries: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        knowledge_steps = chunk.get("knowledge_steps") or []
        step_entry = None
        for item in knowledge_steps:
            if isinstance(item, dict) and item.get("step") == step:
                step_entry = item
                break
        if not step_entry or not isinstance(step_entry.get("knowledge"), dict):
            return None
        knowledge = _build_knowledge_entry(
            step_entry["knowledge"],
            entry_id=entry_id,
            chunk_index=chunk_index,
            step=step,
        )
        if knowledge is None:
            return None
        knowledge_entries.append(knowledge)
    return knowledge_entries


def _align_scores(values: Sequence[Any], count: int) -> List[Optional[float]]:
    scores = [_coerce_score(val) for val in list(values)[:count]]
    if len(scores) < count:
        scores.extend([None] * (count - len(scores)))
    return scores


def _gather_dpo_entries(
    entry: Dict[str, Any],
    *,
    runs_dir: Path,
    run_cache: Dict[str, List[Dict[str, Any]]],
    n_knowledge_steps: int,
    knowledge_per_step: int,
    comparison_threshold: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    chunks = [c for c in (entry.get("chunks") or []) if isinstance(c, dict)]
    if not chunks:
        return []
    chunks_sorted = sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0))
    if knowledge_per_step > 0 and len(chunks_sorted) < knowledge_per_step:
        return []
    if knowledge_per_step > 0:
        chunks_sorted = chunks_sorted[:knowledge_per_step]
    if not chunks_sorted:
        return []

    entry_id = str(entry.get("id") or "entry").strip()
    base_score = _coerce_score(entry.get("base_rerun_mean_score"))
    if base_score is None:
        base_score = _coerce_score(entry.get("base", {}).get("score"))

    chunk_scores = entry.get("knowledge_chunk_mean_scores")
    if not isinstance(chunk_scores, Sequence) or isinstance(
        chunk_scores, (str, bytes, bytearray)
    ):
        chunk_scores = [chunk.get("chunk_mean_score") for chunk in chunks_sorted]
    chunk_scores = _align_scores(chunk_scores, len(chunks_sorted))

    fair_comparison = entry.get("fair_comparison") or {}
    knowledge_trials = [
        trial for trial in (fair_comparison.get("knowledge_trials") or [])
        if isinstance(trial, dict)
    ]
    trial_choices = _select_trials(knowledge_trials, n_knowledge_steps, rng)

    output_entries: List[Dict[str, Any]] = []
    for offset in range(n_knowledge_steps):
        step = offset + 1
        knowledge_entries = _extract_step_knowledge(
            chunks_sorted,
            step=step,
            entry_id=entry_id,
        )
        if knowledge_entries is None:
            continue
        trial = trial_choices[offset] if offset < len(trial_choices) else None
        messages = _load_messages_for_trial(
            trial,
            runs_dir=runs_dir,
            run_cache=run_cache,
        )
        truncated_messages = _truncate_messages_before_step(messages, step)

        knowledge_list: List[Dict[str, Any]] = [
            {"id": None, "text": None, "agent": None, "tags": []}
        ]
        knowledge_list.extend(knowledge_entries)

        score_vector: List[Optional[float]] = [base_score]
        score_vector.extend(chunk_scores)

        comparisons = _build_comparisons(score_vector, comparison_threshold)
        output_entries.append(
            {
                "message": truncated_messages,
                "knowledge": knowledge_list,
                "comparison": comparisons,
                "scores": score_vector,
            }
        )
    return output_entries


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    if args.n_knowledge_steps < 1:
        raise ValueError("n-knowledge-steps must be >= 1.")
    if args.knowledge_per_step < 1:
        raise ValueError("knowledge-per-step must be >= 1.")

    entries, run_cache = _load_results(args.input_path)
    run_cache_map = _index_run_cache(run_cache)
    rng = random.Random(args.seed)

    dataset: List[Dict[str, Any]] = []
    for entry in entries:
        dataset.extend(
            _gather_dpo_entries(
                entry,
                runs_dir=args.runs_dir,
                run_cache=run_cache_map,
                n_knowledge_steps=args.n_knowledge_steps,
                knowledge_per_step=args.knowledge_per_step,
                comparison_threshold=args.comparison_threshold,
                rng=rng,
            )
        )

    if not dataset:
        raise RuntimeError("No DPO entries were produced from the input file.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataset, ensure_ascii=False, indent=args.indent)
    args.output_path.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(dataset)} entries to {args.output_path}")


if __name__ == "__main__":
    main()
