#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import train_v4_eval_sample as base

REPO_ROOT = EXP_DIR.parent
DEFAULT_DATASET_PATH = base.DEFAULT_DATASET_PATH
DEFAULT_DATASET_EVAL_PATH = EXP_DIR / "dataset_eval.json"
DEFAULT_RESULT_PATH = EXP_DIR / "eval_v4_results.json"
DEFAULT_N_PROBLEMS = 10
DEFAULT_N_KNOWLEDGE_STEPS = 3
DEFAULT_N_RUNS = 7
DEFAULT_KNOWLEDGE_CSV_PATH = REPO_ROOT / "exp11_plasma_gkv_v5" / "knowledge_v4.csv"

_MISSING = object()


def _knowledge_path_for_config(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_knowledge_pool_csv(
    path: Path,
    pool: Sequence[Dict[str, Any]],
    *,
    force: bool,
) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "agent", "knowledge", "tags", "step"])
        writer.writeheader()
        for entry in pool:
            text = str(entry.get("text") or "").strip()
            agent = str(entry.get("agent") or "").strip()
            if not text or not agent:
                continue
            tags = entry.get("tags") or []
            writer.writerow(
                {
                    "id": str(entry.get("id") or "").strip(),
                    "agent": agent,
                    "knowledge": text,
                    "tags": json.dumps(tags, ensure_ascii=True),
                    "step": entry.get("step") or "",
                }
            )


def _sample_dataset(
    dataset: Sequence[Dict[str, Any]],
    count: int,
    *,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    if count >= len(dataset):
        return [dict(entry) for entry in dataset]
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)
    return [dict(dataset[idx]) for idx in indices]


def _prepare_eval_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    if args.dataset_eval_path.exists() and not args.resample:
        return json.loads(args.dataset_eval_path.read_text(encoding="utf-8"))
    selected = _sample_dataset(dataset, args.n_problems, seed=args.seed)
    args.dataset_eval_path.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_eval_path.write_text(
        json.dumps(selected, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return selected


def _build_manual_knowledge_config(
    manual_entries: Optional[List[Dict[str, Any]]] = None,
    *,
    load_knowledge_path: Any = _MISSING,
    load_knowledge_steps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"generate_knowledge": False}
    if manual_entries:
        cfg["knowledge"] = [dict(entry) for entry in manual_entries]
    if load_knowledge_path is not _MISSING:
        cfg["load_knowledge_path"] = load_knowledge_path
    if load_knowledge_steps is not None:
        cfg["load_knowledge_steps"] = list(load_knowledge_steps)
    return cfg


def _random_manual_entries(
    *,
    n_steps: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    for step in range(1, n_steps + 1):
        candidates = base._prepare_pool_candidates(step, used_ids)
        if not candidates:
            continue
        choice = rng.choice(candidates)
        choice_id = str(choice.get("id") or "").strip()
        if choice_id:
            used_ids.add(choice_id)
        entries.append(
            {
                "id": choice.get("id"),
                "step": step,
                "agent": choice.get("agent") or "think",
                "text": choice.get("text", ""),
                "tags": choice.get("tags") or [],
            }
        )
    return entries


def _build_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(entries)
    if total == 0:
        return {
            "total_problems": 0,
            "mean_score_improvement_from_base": 0.0,
            "mean_score_improvement_from_random": 0.0,
            "overall_base_mean": 0.0,
            "overall_random_klg_mean": 0.0,
            "overall_rmsearch_klg_mean": 0.0,
            "base_vs_random_vs_rmsearch": [],
        }

    base_scores: List[float] = []
    random_scores: List[float] = []
    rmsearch_scores: List[float] = []
    base_vs_random_vs_rmsearch: List[List[float]] = []
    improvements_from_base: List[float] = []
    improvements_from_random: List[float] = []

    for entry in entries:
        base_val = round(float(entry.get("base_mean_score", 0.0) or 0.0), 4)
        random_val = round(float(entry.get("random_klg_mean_score", 0.0) or 0.0), 4)
        rmsearch_val = round(float(entry.get("rmsearch_klg_mean_score", 0.0) or 0.0), 4)
        base_scores.append(base_val)
        random_scores.append(random_val)
        rmsearch_scores.append(rmsearch_val)
        base_vs_random_vs_rmsearch.append([base_val, random_val, rmsearch_val])
        improvements_from_base.append(rmsearch_val - base_val)
        improvements_from_random.append(rmsearch_val - random_val)

    overall_base_mean = round(sum(base_scores) / total, 4)
    overall_random_klg_mean = round(sum(random_scores) / total, 4)
    overall_rmsearch_klg_mean = round(sum(rmsearch_scores) / total, 4)
    mean_from_base = round(sum(improvements_from_base) / total, 4)
    mean_from_random = round(sum(improvements_from_random) / total, 4)

    return {
        "total_problems": total,
        "mean_score_improvement_from_base": mean_from_base,
        "mean_score_improvement_from_random": mean_from_random,
        "overall_base_mean": overall_base_mean,
        "overall_random_klg_mean": overall_random_klg_mean,
        "overall_rmsearch_klg_mean": overall_rmsearch_klg_mean,
        "base_vs_random_vs_rmsearch": base_vs_random_vs_rmsearch,
    }


def save_eval_entries(
    entries: List[Dict[str, Any]],
    output_path: Path,
    *,
    run_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_obj: Dict[str, Any] = {
        "schema_version": base.CACHE_SCHEMA_VERSION,
        "saved_at": base._now_iso(),
        "summary": _build_summary(entries),
        "detail": entries,
    }
    if run_cache is not None:
        payload_obj["run_cache"] = run_cache
        payload_obj["run_cache_count"] = len(run_cache)
    payload = json.dumps(payload_obj, ensure_ascii=True, indent=2)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(output_path)


@dataclass
class EvalCheckpoint:
    output_path: Path
    eval_entries: List[Dict[str, Any]]
    run_cache: Dict[str, Dict[str, Any]]
    lock: asyncio.Lock

    @staticmethod
    def make_run_key(entry_id: str, run_name: str) -> str:
        return f"{entry_id}::{run_name}"

    def get_cached(self, *, entry_id: str, run_name: str) -> Optional[Dict[str, Any]]:
        key = self.make_run_key(entry_id, run_name)
        cached = self.run_cache.get(key)
        if not isinstance(cached, dict):
            return None
        return dict(cached)

    async def record_run(
        self,
        *,
        entry_id: str,
        dataset_index: int,
        run_name: str,
        orchestrator_log_dir: Path,
        result: Dict[str, Any],
    ) -> None:
        key = self.make_run_key(entry_id, run_name)
        base.normalize_result_run_id(result, Path(orchestrator_log_dir))
        payload: Dict[str, Any] = {
            "entry_id": entry_id,
            "dataset_index": int(dataset_index),
            "run_name": run_name,
            "run_id": str(result.get("run_id") or "").strip(),
            "output": result.get("output", ""),
            "saved_at": base._now_iso(),
        }
        msg_history = base._compact_msg_history(result.get("msg_history"))
        if msg_history is not None:
            payload["msg_history"] = msg_history

        async with self.lock:
            prev = self.run_cache.get(key)
            if (
                isinstance(prev, dict)
                and prev.get("run_id") == payload.get("run_id")
                and prev.get("output") == payload.get("output")
            ):
                return
            self.run_cache[key] = payload
            save_eval_entries(self.eval_entries, self.output_path, run_cache=self.run_cache)


async def run_scored_trials(
    orchestrator,
    patch_manager: base.PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    base_prompt_messages: Sequence[Dict[str, Any]],
    trials: int,
    dataset_index: int,
    label: str,
    knowledge_config: Dict[str, Any],
    checkpoint: EvalCheckpoint,
    entry_id: str,
    llm_request: Optional[base.LLM_Request] = None,
    use_knowledge_prompt: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    if trials <= 0:
        return [], 0.0
    question = base.build_task_prompt(dataset_entry)
    reference = base.extract_reference_answer(dataset_entry)
    trial_records: List[Dict[str, Any]] = []
    valid_scores: List[float] = []
    for trial in range(trials):
        rerun_messages = base.randomize_system_prompt(
            base_prompt_messages,
            use_knowledge_prompt=use_knowledge_prompt,
        )
        run_name = f"eval_v4_{dataset_index:04d}_{label}_r{trial + 1}"
        result = await base.run_orchestrator_with_patch(
            orchestrator,
            patch_manager,
            dataset_entry=dataset_entry,
            dataset_index=dataset_index,
            messages=rerun_messages,
            run_name=run_name,
            knowledge_config=knowledge_config,
            checkpoint=checkpoint,
            entry_id=entry_id,
            llm_request=llm_request,
        )
        run_id = base.normalize_result_run_id(result, Path(orchestrator.log_dir))
        output = result.get("output", "")
        score_info = await base.score_answer(
            orchestrator.llm,
            question,
            reference,
            output,
            llm_request=llm_request,
        )
        score = score_info.get("score", 0.0) or 0.0
        feedback = score_info.get("feedback")
        if not base._is_bugged_score(score, feedback):
            valid_scores.append(score)
        trial_records.append(
            {
                "trial": trial + 1,
                "run_name": run_name,
                "run_id": run_id,
                "score": score,
                "score_feedback": feedback,
                "output": output,
            }
        )
    mean_score = (
        round(sum(valid_scores) / len(valid_scores), 2)
        if valid_scores
        else 0.0
    )
    return trial_records, mean_score


async def run_problem_eval(
    orchestrator,
    patch_manager: base.PatchWorkspaceManager,
    dataset_entry: Dict[str, Any],
    index: int,
    entry_id: str,
    *,
    n_knowledge_steps: int,
    n_runs: int,
    knowledge_csv_path: Path,
    checkpoint: EvalCheckpoint,
    llm_request: Optional[base.LLM_Request] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    problem_text = base.extract_problem_text(dataset_entry)
    prompt_text = base.build_task_prompt(dataset_entry) or problem_text
    if not prompt_text:
        prompt_text = "Investigate the regression and repair the affected code paths."
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    patch_path = patch_manager.resolve_patch_path(dataset_entry, index)

    base_prompt_messages = [{"role": "user", "content": prompt_text}]
    knowledge_cfg_clear = _build_manual_knowledge_config(load_knowledge_path=None)
    random_rng = rng or random.Random()
    random_entries = _random_manual_entries(
        n_steps=n_knowledge_steps,
        rng=random_rng,
    )

    base_trials, base_mean = await run_scored_trials(
        orchestrator,
        patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        trials=n_runs,
        dataset_index=index,
        label="base",
        knowledge_config=knowledge_cfg_clear,
        checkpoint=checkpoint,
        entry_id=entry_id,
        llm_request=llm_request,
        use_knowledge_prompt=False,
    )
    random_trials, random_mean = await run_scored_trials(
        orchestrator,
        patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        trials=n_runs,
        dataset_index=index,
        label="random_klg",
        knowledge_config=_build_manual_knowledge_config(
            random_entries,
            load_knowledge_path=None,
        ),
        checkpoint=checkpoint,
        entry_id=entry_id,
        llm_request=llm_request,
        use_knowledge_prompt=True,
    )
    rmsearch_path = _knowledge_path_for_config(knowledge_csv_path)
    load_steps = list(range(1, n_knowledge_steps + 1))
    rmsearch_trials, rmsearch_mean = await run_scored_trials(
        orchestrator,
        patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        trials=n_runs,
        dataset_index=index,
        label="rmsearch_klg",
        knowledge_config=_build_manual_knowledge_config(
            load_knowledge_path=rmsearch_path,
            load_knowledge_steps=load_steps,
        ),
        checkpoint=checkpoint,
        entry_id=entry_id,
        llm_request=llm_request,
        use_knowledge_prompt=True,
    )

    return {
        "id": entry_id,
        "problem": problem_text,
        "csv_path": csv_path,
        "patch_path": base.format_relative_path(patch_path),
        "random_selected_knowledge": random_entries,
        "rmsearch_knowledge_path": rmsearch_path,
        "rmsearch_knowledge_steps": load_steps,
        "base_trials": base_trials,
        "random_klg_trials": random_trials,
        "rmsearch_klg_trials": rmsearch_trials,
        "base_mean_score": base_mean,
        "random_klg_mean_score": random_mean,
        "rmsearch_klg_mean_score": rmsearch_mean,
        "random_improvement_from_base": round(random_mean - base_mean, 2),
        "score_improvement_from_base": round(rmsearch_mean - base_mean, 2),
        "score_improvement_from_random": round(rmsearch_mean - random_mean, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base vs random knowledge vs rmsearch knowledge on a dataset sample.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the full dataset JSON.",
    )
    parser.add_argument(
        "--dataset-eval-path",
        type=Path,
        default=DEFAULT_DATASET_EVAL_PATH,
        help="Where to write or read the sampled dataset subset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Where to store evaluation traces and scores.",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=DEFAULT_N_PROBLEMS,
        help="Number of problems to sample from the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for dataset sampling and knowledge selection.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Force a new dataset_eval.json sample even if it already exists.",
    )
    parser.add_argument(
        "--knowledge-csv-path",
        type=Path,
        default=DEFAULT_KNOWLEDGE_CSV_PATH,
        help="CSV path for the knowledge pool used by rmsearch.",
    )
    parser.add_argument(
        "--refresh-knowledge-csv",
        action="store_true",
        help="Overwrite the knowledge CSV even if it already exists.",
    )
    parser.add_argument(
        "--llm-model-name",
        default=base.DEFAULT_LLM_MODEL_NAME,
        help="Name of model in LLM endpoint.",
    )
    parser.add_argument(
        "--llm-url",
        default=base.DEFAULT_LLM_URL,
        help="LLM endpoint passed to the orchestrator.",
    )
    parser.add_argument(
        "--rm-url",
        default=base.DEFAULT_RM_URL,
        help="Reward model search endpoint passed to the orchestrator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=base.DEFAULT_BATCH_SIZE,
        help="Number of problems to process concurrently.",
    )
    parser.add_argument(
        "--n-knowledge-steps",
        type=int,
        default=DEFAULT_N_KNOWLEDGE_STEPS,
        help="How many leading agent steps receive random knowledge.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="How many reruns to execute per condition when comparing scores.",
    )
    parser.add_argument(
        "--final-reruns",
        dest="n_runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Deprecated alias for --n-runs.",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        dest="save_log",
        help="Write stdout/stderr to a log file (also enabled when stdout is not a TTY).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files when save-log is enabled.",
    )
    return parser.parse_args()


async def run_evaluation(args: argparse.Namespace) -> None:
    log_state = base._setup_log_output(
        save_log=args.save_log,
        log_dir=args.log_dir,
        prefix="eval_v4",
    )
    try:
        if args.n_problems < 1:
            raise ValueError("--n-problems must be >= 1")
        random.seed(args.seed)
        dataset = _prepare_eval_dataset(args)
        _write_knowledge_pool_csv(
            args.knowledge_csv_path,
            base.DEFAULT_KNOWLEDGE_POOL,
            force=args.refresh_knowledge_csv,
        )

        patch_files = sorted(base.PATCH_DIR.glob("*.txt"))
        if not patch_files:
            raise RuntimeError(f"No patch files found under {base.PATCH_DIR}")
        workspace_count = min(args.batch_size, len(patch_files))
        if workspace_count < 1:
            raise RuntimeError("batch_size must be >= 1 and patch_files must be non-empty.")
        if args.batch_size != workspace_count:
            print(
                "[eval_v4] Limiting concurrency to "
                f"{workspace_count} based on patch file count."
            )
        workspace_pool = base.build_workspace_pool(args, workspace_count)

        eval_entries, processed_ids, needs_resave, run_cache = base.load_existing_eval_entries(
            dataset, args.output_path
        )
        checkpoint = EvalCheckpoint(
            output_path=args.output_path,
            eval_entries=eval_entries,
            run_cache=run_cache,
            lock=asyncio.Lock(),
        )
        if needs_resave:
            save_eval_entries(eval_entries, args.output_path, run_cache=run_cache)
        if processed_ids:
            print(f"[eval_v4] Resuming with {len(processed_ids)} cached entries")
        if run_cache:
            print(f"[eval_v4] Loaded run_cache with {len(run_cache)} runs")

        pending: List[Tuple[int, Dict[str, Any], str]] = []
        for idx, entry in enumerate(dataset):
            entry_id = base.compute_entry_id(entry, idx)
            if entry_id in processed_ids:
                continue
            pending.append((idx, entry, entry_id))

        if not pending:
            print(
                f"[eval_v4] All {len(processed_ids)} entries already processed. "
                f"Results stored at {args.output_path}"
            )
            return

        total_pending = len(pending)
        print(
            f"[eval_v4] Starting {total_pending} problems with "
            f"concurrency={workspace_count} (dataset size={len(dataset)})"
        )
        completed_in_run = 0

        llm_request = base.LLM_Request(args.batch_size)
        await llm_request.start()
        try:
            for batch_start in range(0, len(pending), workspace_count):
                batch_slice = pending[batch_start : batch_start + workspace_count]
                batch_idx = batch_start // args.batch_size + 1
                print(
                    f"[eval_v4] Starting batch {batch_idx} "
                    f"with {len(batch_slice)} problems"
                )
                batch_tasks: List[asyncio.Task] = []
                task_ids: Dict[asyncio.Task, str] = {}
                for handle, (dataset_idx, entry, entry_id) in zip(workspace_pool, batch_slice):
                    rng = random.Random(args.seed + dataset_idx)
                    task = asyncio.create_task(
                        run_problem_eval(
                            handle.orchestrator,
                            handle.patch_manager,
                            entry,
                            dataset_idx,
                            entry_id,
                            n_knowledge_steps=args.n_knowledge_steps,
                            n_runs=args.n_runs,
                            knowledge_csv_path=args.knowledge_csv_path,
                            checkpoint=checkpoint,
                            llm_request=llm_request,
                            rng=rng,
                        )
                    )
                    batch_tasks.append(task)
                    task_ids[task] = entry_id

                for task in asyncio.as_completed(batch_tasks):
                    entry_id = task_ids.get(task, "unknown")
                    record = None
                    try:
                        record = await task
                    except Exception as exc:  # pragma: no cover - runtime guard
                        print(f"[eval_v4] Problem {entry_id} failed: {exc}")
                    if record:
                        rid = str(record.get("id") or "").strip()
                        if rid and rid not in processed_ids:
                            eval_entries.append(record)
                            processed_ids.add(rid)
                    completed_in_run += 1
                    percent = (completed_in_run / total_pending) * 100 if total_pending else 100.0
                    print(
                        "[eval_v4] Progress: "
                        f"{completed_in_run}/{total_pending} ({percent:.1f}%) problems finished; "
                        f"overall {len(processed_ids)}/{len(dataset)} stored"
                    )

                save_eval_entries(eval_entries, args.output_path, run_cache=checkpoint.run_cache)
                print(
                    f"[eval_v4] Saved batch {batch_idx}: "
                    f"{len(processed_ids)}/{len(dataset)} problems complete"
                )
        finally:
            await llm_request.close()

        print(f"[eval_v4] Saved evaluation dataset to {args.output_path}")
    finally:
        base._close_log_output(log_state)


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
