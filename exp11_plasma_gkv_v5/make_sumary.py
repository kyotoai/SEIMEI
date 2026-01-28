#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import train_v6 as base

DEFAULT_RESULT_PATH = EXP_DIR / "train_v6_results6_eval4.json"
DEFAULT_DATASET_PATH = base.DEFAULT_DATASET_PATH
DEFAULT_RUNS_DIR = Path("seimei_runs")

RUN_NAME_TRIAL_RE = re.compile(r"_r(?P<trial>\d+)")
RUN_NAME_INDEX_RE = re.compile(r"train_v6_(?P<index>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild train_v6 summary/detail from run_cache and save in-place."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Path to train_v6 results JSON.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to dataset.json used for scoring and patch lookup.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing seimei_runs/<run_id>/messages.json.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=base.DEFAULT_LLM_MODEL_NAME,
        help="Model name for scoring LLM.",
    )
    parser.add_argument(
        "--llm-url",
        default=base.DEFAULT_LLM_URL,
        help="Base URL for scoring LLM.",
    )
    parser.add_argument(
        "--rm-url",
        default=base.DEFAULT_RM_URL,
        help="RM URL (unused but required by orchestrator builder).",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip LLM scoring and set score=0 for all runs.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        detail = raw.get("detail") or []
        run_cache = raw.get("run_cache") or {}
        if not isinstance(detail, list):
            detail = []
        if not isinstance(run_cache, dict):
            run_cache = {}
        return raw, [entry for entry in detail if isinstance(entry, dict)], run_cache
    if isinstance(raw, list):
        return {"detail": raw}, [entry for entry in raw if isinstance(entry, dict)], {}
    raise ValueError("Input JSON must be a list or dict with 'detail'.")


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_run_name(run_name: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    name = str(run_name or "").strip()
    if not name:
        return None, None, None
    label = None
    if "no_klg" in name:
        label = "no_klg"
    elif "klg" in name:
        label = "klg"

    trial = None
    match_trial = RUN_NAME_TRIAL_RE.search(name)
    if match_trial:
        trial = _coerce_int(match_trial.group("trial"))

    index = None
    match_index = RUN_NAME_INDEX_RE.search(name)
    if match_index:
        index = _coerce_int(match_index.group("index"))

    if label is None or trial is None:
        return None, None, index
    return label, trial, index


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("dataset.json must be a list of entries")
    return [entry for entry in raw if isinstance(entry, dict)]


def _build_id_to_index(dataset: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    id_map: Dict[str, int] = {}
    for idx, entry in enumerate(dataset):
        entry_id = base.compute_entry_id(entry, idx)
        if entry_id and entry_id not in id_map:
            id_map[entry_id] = idx
    return id_map


def _resolve_dataset_entry(
    dataset: Sequence[Dict[str, Any]],
    entry_id: str,
    dataset_index: Optional[int],
    id_to_index: Dict[str, int],
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    idx = dataset_index
    if idx is None and entry_id:
        idx = id_to_index.get(entry_id)
    if idx is None:
        return None, None
    if idx < 0 or idx >= len(dataset):
        return None, None
    return dataset[idx], idx


def _resolve_patch(
    patch_manager: base.PatchWorkspaceManager,
    dataset_entry: Optional[Dict[str, Any]],
    dataset_index: Optional[int],
) -> Tuple[Optional[Path], str]:
    if not dataset_entry or dataset_index is None:
        return None, ""
    try:
        patch_path = patch_manager.resolve_patch_path(dataset_entry, dataset_index)
    except FileNotFoundError:
        return None, ""
    try:
        patch_text = patch_path.read_text(encoding="utf-8")
    except OSError:
        return patch_path, ""
    return patch_path, patch_text


def _extract_prompt_fields(entry: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    if not entry:
        return "", "", ""
    problem_text = base.extract_problem_text(entry)
    prompt_text = base.build_task_prompt(entry) or problem_text
    if not prompt_text:
        prompt_text = "Investigate the regression and repair the affected code paths."
    question = base.build_task_prompt(entry) or prompt_text
    reference_answer = base.extract_reference_answer(entry)
    return problem_text, question, reference_answer


def _summarize_scores(scores: Sequence[float]) -> Tuple[float, float, float]:
    return base._summarize_scores(scores)


def _sort_trials(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        trials,
        key=lambda item: (
            _coerce_int(item.get("trial")) or 0,
            str(item.get("run_name") or ""),
        ),
    )


def _build_payload(
    raw: Dict[str, Any],
    *,
    detail: List[Dict[str, Any]],
    run_cache: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(raw) if isinstance(raw, dict) else {}
    payload["schema_version"] = base.CACHE_SCHEMA_VERSION
    payload["saved_at"] = base._now_iso()
    payload["summary"] = base.build_eval_summary(detail)
    payload["detail"] = detail
    if run_cache is not None:
        payload["run_cache"] = run_cache
        payload["run_cache_count"] = len(run_cache)
    return payload


async def _score_run(
    *,
    run_payload: Dict[str, Any],
    trial: int,
    run_name: str,
    runs_dir: Path,
    question: str,
    reference_answer: str,
    patch_text: str,
    use_knowledge: bool,
    orchestrator,
    skip_scoring: bool,
) -> Tuple[Dict[str, Any], float, Optional[str]]:
    run_result = dict(run_payload)
    if run_result.get("run_id"):
        base.normalize_result_run_id(run_result, runs_dir)
    output = str(run_result.get("output") or "")
    messages = base.get_messages_for_run(run_result, runs_dir)
    knowledge_used: List[Dict[str, Any]] = []
    if use_knowledge:
        knowledge_used = base.extract_knowledge_entries_from_messages(messages, orchestrator)

    score = 0.0
    feedback: Optional[str] = None
    if skip_scoring:
        feedback = None
    else:
        score_info = await base.score_answer(
            orchestrator.llm,
            question,
            reference_answer,
            output,
            patch=patch_text,
            messages=messages,
            knowledge_entries=knowledge_used,
            llm_request=None,
        )
        score = score_info.get("score", 0.0) or 0.0
        feedback = score_info.get("feedback")

    record: Dict[str, Any] = {
        "trial": trial,
        "run_name": run_name,
        "run_id": run_result.get("run_id", ""),
        "score": score,
        "score_feedback": feedback,
        "output": output,
    }
    if use_knowledge:
        record["knowledge_used"] = knowledge_used
    return record, score, feedback


async def _build_detail_from_run_cache(
    *,
    run_cache: Dict[str, Any],
    dataset: Sequence[Dict[str, Any]],
    id_to_index: Dict[str, int],
    runs_dir: Path,
    orchestrator,
    skip_scoring: bool,
) -> List[Dict[str, Any]]:
    runs_by_entry: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    entry_index_map: Dict[str, Optional[int]] = {}

    for payload in run_cache.values():
        if not isinstance(payload, dict):
            continue
        entry_id = str(payload.get("entry_id") or "").strip()
        dataset_index = _coerce_int(payload.get("dataset_index"))
        run_name = str(payload.get("run_name") or "").strip()
        label, trial, idx_from_name = _parse_run_name(run_name)
        if label is None or trial is None:
            print(f"[make_sumary] Skip run with unrecognized name: {run_name}")
            continue
        if dataset_index is None:
            dataset_index = idx_from_name
        if not entry_id:
            if dataset_index is not None and 0 <= dataset_index < len(dataset):
                entry_id = base.compute_entry_id(dataset[dataset_index], dataset_index)
        if not entry_id:
            print(f"[make_sumary] Skip run missing entry_id: {run_name}")
            continue
        if dataset_index is None:
            dataset_index = id_to_index.get(entry_id)
        if entry_id not in runs_by_entry:
            runs_by_entry[entry_id] = {"no_klg": [], "klg": []}
        runs_by_entry[entry_id][label].append(
            {
                "payload": payload,
                "run_name": run_name,
                "trial": trial,
            }
        )
        if entry_id not in entry_index_map:
            entry_index_map[entry_id] = dataset_index

    patch_manager = base.PatchWorkspaceManager(base.REPO_ROOT, base.PATCH_DIR)

    def _sort_key(entry_id: str) -> Tuple[int, str]:
        idx = entry_index_map.get(entry_id)
        return (idx if isinstance(idx, int) else 10**9, entry_id)

    detail_entries: List[Dict[str, Any]] = []
    for entry_id in sorted(runs_by_entry.keys(), key=_sort_key):
        dataset_entry, dataset_index = _resolve_dataset_entry(
            dataset,
            entry_id,
            entry_index_map.get(entry_id),
            id_to_index,
        )
        problem_text, question, reference_answer = _extract_prompt_fields(dataset_entry)
        csv_path = str((dataset_entry or {}).get("CSVPath") or "").strip()
        patch_path, patch_text = _resolve_patch(patch_manager, dataset_entry, dataset_index)

        no_klg_runs = _sort_trials(runs_by_entry[entry_id]["no_klg"])
        klg_runs = _sort_trials(runs_by_entry[entry_id]["klg"])

        no_klg_trials: List[Dict[str, Any]] = []
        klg_trials: List[Dict[str, Any]] = []
        no_klg_valid_scores: List[float] = []
        klg_valid_scores: List[float] = []

        for run in no_klg_runs:
            trial_record, score, feedback = await _score_run(
                run_payload=run["payload"],
                trial=run["trial"],
                run_name=run["run_name"],
                runs_dir=runs_dir,
                question=question,
                reference_answer=reference_answer,
                patch_text=patch_text,
                use_knowledge=False,
                orchestrator=orchestrator,
                skip_scoring=skip_scoring,
            )
            no_klg_trials.append(trial_record)
            if not base._is_bugged_score(score, feedback):
                no_klg_valid_scores.append(score)

        for run in klg_runs:
            trial_record, score, feedback = await _score_run(
                run_payload=run["payload"],
                trial=run["trial"],
                run_name=run["run_name"],
                runs_dir=runs_dir,
                question=question,
                reference_answer=reference_answer,
                patch_text=patch_text,
                use_knowledge=True,
                orchestrator=orchestrator,
                skip_scoring=skip_scoring,
            )
            klg_trials.append(trial_record)
            if not base._is_bugged_score(score, feedback):
                klg_valid_scores.append(score)

        no_klg_mean, no_klg_max, no_klg_min = _summarize_scores(no_klg_valid_scores)
        klg_mean, klg_max, klg_min = _summarize_scores(klg_valid_scores)

        record: Dict[str, Any] = {
            "id": entry_id,
            "problem": problem_text,
            "csv_path": csv_path,
            "patch_path": base.format_relative_path(patch_path) if patch_path else "",
            "no_klg_trials": no_klg_trials,
            "klg_trials": klg_trials,
            "no_klg_mean_score": no_klg_mean,
            "klg_mean_score": klg_mean,
            "no_klg_max_score": no_klg_max,
            "klg_max_score": klg_max,
            "no_klg_min_score": no_klg_min,
            "klg_min_score": klg_min,
            "mean_score_improvement": round(klg_mean - no_klg_mean, 2),
            "max_score_improvement": round(klg_max - no_klg_max, 2),
            "min_score_improvement": round(klg_min - no_klg_min, 2),
        }
        detail_entries.append(record)

    return detail_entries


async def main_async() -> None:
    args = parse_args()
    raw, detail, run_cache = _load_results(args.input_path)

    if not run_cache:
        payload = _build_payload(raw, detail=detail, run_cache=run_cache)
        args.input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[make_sumary] Updated summary/detail in {args.input_path}")
        return

    dataset = _load_dataset(args.dataset_path)
    id_to_index = _build_id_to_index(dataset)

    orchestrator = None
    if not args.skip_scoring:
        orchestrator = base._build_orchestrator(args, base.REPO_ROOT)

    detail_entries = await _build_detail_from_run_cache(
        run_cache=run_cache,
        dataset=dataset,
        id_to_index=id_to_index,
        runs_dir=args.runs_dir,
        orchestrator=orchestrator,
        skip_scoring=args.skip_scoring,
    )

    payload = _build_payload(raw, detail=detail_entries, run_cache=run_cache)
    args.input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[make_sumary] Rebuilt summary/detail in {args.input_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
