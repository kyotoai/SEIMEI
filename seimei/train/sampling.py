"""Inference + scoring sampler extracted from exp11_plasma_gkv_v5/train_v6.py."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from seimei import seimei

from .sampling_prompts import KNOWLEDGE_UPDATE_SYSTEM_PROMPT, SCORING_SYSTEM_PROMPT
from .sampling_utils import (
    EvalCheckpoint,
    PatchWorkspaceManager,
    WorkspaceHandle,
    apply_knowledge_updates,
    build_knowledge_search_config,
    build_knowledge_update_prompt,
    build_scoring_prompt,
    build_task_prompt,
    close_log_output,
    coerce_score,
    compute_entry_id,
    extract_knowledge_entries_from_messages,
    extract_problem_text,
    extract_reference_answer,
    format_reasoning_history,
    format_relative_path,
    get_messages_for_run,
    is_bugged_score,
    load_existing_eval_entries,
    load_knowledge_pool_csv,
    normalize_pool_ids,
    normalize_result_run_id,
    parse_json_response,
    parse_knowledge_updates,
    prepare_workspace,
    randomize_system_prompt,
    save_eval_entries,
    setup_log_output,
    split_knowledge_args,
    summarize_scores,
    write_knowledge_pool_csv,
)


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
TRAIN_V6_EXP_DIR = REPO_ROOT / "exp11_plasma_gkv_v5"

DEFAULT_DATASET_PATH = TRAIN_V6_EXP_DIR / "dataset.json"
DEFAULT_RESULT_PATH = TRAIN_V6_EXP_DIR / "train_v6_results.json"
DEFAULT_LLM_MODEL_NAME = "/workspace/gpt-oss-20b"
DEFAULT_LLM_URL = "https://94sownu2ebkgwm-8000.proxy.runpod.net/v1"
DEFAULT_RM_URL = None
DEFAULT_BATCH_SIZE = 100
DEFAULT_TOP_N_SAMPLE_KLG = 5
DEFAULT_DISTRIBUTION_DECAY_RATE = 0.5
DEFAULT_RANDOM_KLG_SAMPLING_RATE = 0.2
DEFAULT_KLG_SAMPLE_MODE = "llm"
DEFAULT_N_NO_KLG_TRIALS = 3
DEFAULT_N_KLG_TRIALS = 7

_DEFAULT_KLG_PATH_CANDIDATES = (
    TRAIN_V6_EXP_DIR / "knowledge_v6.csv",
    MODULE_DIR / "default_knowledge.csv",
)
_DEFAULT_EXISTING_KLG_PATHS = [path for path in _DEFAULT_KLG_PATH_CANDIDATES if path.exists()]
DEFAULT_KLG_POOL_LOAD_PATH = (
    _DEFAULT_EXISTING_KLG_PATHS[0] if _DEFAULT_EXISTING_KLG_PATHS else _DEFAULT_KLG_PATH_CANDIDATES[0]
)
DEFAULT_ENABLE_UPDATE_KLG_POOL = True
DEFAULT_FINAL_KLG_POOL_SAVE_PATH = DEFAULT_KLG_POOL_LOAD_PATH
DEFAULT_PATCH_DIR = TRAIN_V6_EXP_DIR / "patch_files"
DEFAULT_WORKSPACE_ROOT = TRAIN_V6_EXP_DIR / "_workspace_copies"
DEFAULT_SCHEMA_VERSION = 2
DEFAULT_SAVE_LOG = False
DEFAULT_LOG_DIR = TRAIN_V6_EXP_DIR / "_logs"
DEFAULT_RESUME = True
DEFAULT_RUN_NAME_PREFIX = "train_v6"


class LLMRequest:
    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max(int(max_concurrent), 1)
        self.queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._closed = False

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.max_concurrent)
        ]

    async def request(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._closed:
            raise RuntimeError("LLMRequest is closed.")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self.queue.put((func, args, kwargs, future))
        return await future

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            await self.queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker(self, _worker_id: int) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            func, args, kwargs, future = item
            if future.cancelled():
                self.queue.task_done()
                continue
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)
            finally:
                self.queue.task_done()


class Sampling:
    """Developer-friendly inference + scoring runner with sync and async entrypoints."""

    def __init__(
        self,
        *,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        output_path: Path = DEFAULT_RESULT_PATH,
        llm_model_name: str = DEFAULT_LLM_MODEL_NAME,
        llm_url: Optional[str] = DEFAULT_LLM_URL,
        rm_url: Optional[str] = DEFAULT_RM_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_no_klg_trials: int = DEFAULT_N_NO_KLG_TRIALS,
        n_klg_trials: int = DEFAULT_N_KLG_TRIALS,
        top_n_sample_klg: int = DEFAULT_TOP_N_SAMPLE_KLG,
        distribution_decay_rate: float = DEFAULT_DISTRIBUTION_DECAY_RATE,
        random_klg_sampling_rate: float = DEFAULT_RANDOM_KLG_SAMPLING_RATE,
        klg_sample_mode: str = DEFAULT_KLG_SAMPLE_MODE,
        max_problems: Optional[int] = None,
        klg_pool_load_path: Path = DEFAULT_KLG_POOL_LOAD_PATH,
        enable_update_klg_pool: bool = DEFAULT_ENABLE_UPDATE_KLG_POOL,
        final_klg_pool_save_path: Path = DEFAULT_FINAL_KLG_POOL_SAVE_PATH,
        patch_dir: Path = DEFAULT_PATCH_DIR,
        workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
        schema_version: int = DEFAULT_SCHEMA_VERSION,
        save_log: bool = DEFAULT_SAVE_LOG,
        log_dir: Optional[Path] = DEFAULT_LOG_DIR,
        resume: bool = DEFAULT_RESUME,
        run_name_prefix: str = DEFAULT_RUN_NAME_PREFIX,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.llm_model_name = llm_model_name
        self.llm_url = llm_url
        self.rm_url = rm_url
        self.batch_size = int(batch_size)
        self.n_no_klg_trials = int(n_no_klg_trials)
        self.n_klg_trials = int(n_klg_trials)
        self.top_n_sample_klg = int(top_n_sample_klg)
        self.distribution_decay_rate = float(distribution_decay_rate)
        self.random_klg_sampling_rate = float(random_klg_sampling_rate)
        self.klg_sample_mode = str(klg_sample_mode)
        self.max_problems = max_problems
        self.klg_pool_load_path = Path(klg_pool_load_path)
        self.enable_update_klg_pool = bool(enable_update_klg_pool)
        self.final_klg_pool_save_path = Path(final_klg_pool_save_path)
        self.patch_dir = Path(patch_dir)
        self.workspace_root = Path(workspace_root)
        self.schema_version = int(schema_version)
        self.save_log = bool(save_log)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.resume = bool(resume)
        self.run_name_prefix = str(run_name_prefix).strip() or "train_v6"

        self.repo_root = REPO_ROOT
        self._run_id_cache: Dict[str, str] = {}

        self.knowledge_pool = self._load_initial_knowledge_pool()

    def _load_initial_knowledge_pool(self) -> List[Dict[str, Any]]:
        candidates: List[Path] = []
        seen: set[str] = set()
        for path in (self.klg_pool_load_path, MODULE_DIR / "default_knowledge.csv"):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(path)

        for candidate in candidates:
            pool = load_knowledge_pool_csv(candidate)
            if pool:
                normalized = normalize_pool_ids(pool)
                if candidate != self.klg_pool_load_path:
                    print(f"[sampling] Falling back to knowledge pool: {candidate}")
                return normalized

        print(f"[sampling] Knowledge pool is empty: {self.klg_pool_load_path}")
        return []

    @staticmethod
    async def _run_llm_request(
        llm_request: Optional[LLMRequest],
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if llm_request is None:
            return await func(*args, **kwargs)
        return await llm_request.request(func, *args, **kwargs)

    def _build_orchestrator(self, workspace: Path):
        prior_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            return seimei(
                agent_config=[{"name": "code_act"}, {"name": "think"}, {"name": "answer"}],
                llm_config={"base_url": self.llm_url, "model": self.llm_model_name},
                rm_config={"base_url": self.rm_url},
                allow_code_exec=True,
                agent_log_head_lines=1,
                max_tokens_per_question=80000,
            )
        finally:
            os.chdir(prior_cwd)

    def _build_workspace_pool(self, count: int) -> List[WorkspaceHandle]:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        handles: List[WorkspaceHandle] = []
        for idx in range(count):
            workspace = self.workspace_root / f"ws_{idx:02d}"
            prepare_workspace(workspace, self.repo_root)
            patch_manager = PatchWorkspaceManager(workspace, self.patch_dir)
            orchestrator = self._build_orchestrator(workspace)
            handles.append(
                WorkspaceHandle(
                    workspace=workspace,
                    patch_manager=patch_manager,
                    orchestrator=orchestrator,
                )
            )
        return handles

    async def run_orchestrator_with_patch(
        self,
        orchestrator: Any,
        patch_manager: PatchWorkspaceManager,
        *,
        dataset_entry: Dict[str, Any],
        dataset_index: int,
        messages: Sequence[Dict[str, Any]],
        run_name: str,
        knowledge_config: Optional[Dict[str, Any]],
        knowledge_search_config: Optional[Sequence[Dict[str, Any]]] = None,
        knowledge_search_mode: Optional[str] = None,
        checkpoint: Optional[EvalCheckpoint] = None,
        entry_id: Optional[str] = None,
        llm_request: Optional[LLMRequest] = None,
    ) -> Dict[str, Any]:
        resolved_entry_id = entry_id or compute_entry_id(dataset_entry, dataset_index)
        if checkpoint is not None:
            cached = checkpoint.get_cached(entry_id=resolved_entry_id, run_name=run_name)
            if cached:
                cached.setdefault("run_id", cached.get("run_id") or "")
                cached.setdefault("output", cached.get("output") or "")
                if "msg_history" in cached:
                    cached["msg_history"] = cached.get("msg_history")
                return cached

        knowledge_kwargs = split_knowledge_args(knowledge_config)
        if knowledge_search_config is not None:
            knowledge_kwargs["knowledge_search_config"] = knowledge_search_config
        if knowledge_search_mode:
            knowledge_kwargs["knowledge_search_mode"] = knowledge_search_mode
        knowledge_kwargs["agent_search_mode"] = "klg"

        try:
            with patch_manager.apply_for_problem(dataset_entry, dataset_index):
                result = await self._run_llm_request(
                    llm_request,
                    orchestrator,
                    messages=messages,
                    run_name=run_name,
                    **knowledge_kwargs,
                    workspace=patch_manager.workspace,
                )

            if checkpoint is not None:
                await checkpoint.record_run(
                    entry_id=resolved_entry_id,
                    dataset_index=dataset_index,
                    run_name=run_name,
                    orchestrator_log_dir=Path(orchestrator.log_dir),
                    result=result,
                )
            return result
        finally:
            prepare_workspace(patch_manager.workspace, self.repo_root)

    async def score_answer(
        self,
        orchestrator_llm: Any,
        question: str,
        reference_answer: str,
        model_answer: str,
        *,
        patch: str,
        messages: Sequence[Dict[str, Any]],
        knowledge_entries: Optional[Sequence[Dict[str, Any]]] = None,
        llm_request: Optional[LLMRequest] = None,
    ) -> Dict[str, Any]:
        reasoning_history = format_reasoning_history(messages, knowledge_entries)
        prompt = build_scoring_prompt(
            question=question,
            patch=patch,
            reference_answer=reference_answer,
            model_answer=model_answer,
            reasoning_history=reasoning_history,
        )
        try:
            response, usage = await self._run_llm_request(
                llm_request,
                orchestrator_llm.chat,
                messages=[{"role": "user", "content": prompt}],
                system=SCORING_SYSTEM_PROMPT,
            )
        except Exception as exc:
            return {
                "score": 0.0,
                "feedback": f"Scoring failed: {exc}",
                "raw_response": "",
                "judge_usage": {},
            }

        parsed = parse_json_response(response)
        score_value = coerce_score(parsed.get("score"))
        feedback = parsed.get("feedback") or parsed.get("rationale") or ""
        return {
            "score": score_value,
            "feedback": str(feedback).strip(),
            "raw_response": response,
            "judge_usage": usage,
        }

    async def update_knowledge_after_scoring(
        self,
        orchestrator_llm: Any,
        *,
        question: str,
        reference_answer: str,
        patch: str,
        messages: Sequence[Dict[str, Any]],
        knowledge_entries: Sequence[Dict[str, Any]],
        update_lock: asyncio.Lock,
        llm_request: Optional[LLMRequest] = None,
    ) -> List[str]:
        reasoning_history = format_reasoning_history(messages, knowledge_entries)
        prompt = build_knowledge_update_prompt(
            question=question,
            patch=patch,
            reference_answer=reference_answer,
            reasoning_history=reasoning_history,
        )
        try:
            response, _ = await self._run_llm_request(
                llm_request,
                orchestrator_llm.chat,
                messages=[{"role": "user", "content": prompt}],
                system=KNOWLEDGE_UPDATE_SYSTEM_PROMPT,
            )
        except Exception as exc:
            print(f"[sampling] knowledge update failed: {exc}")
            return []

        updates = parse_knowledge_updates(response)
        if not updates:
            return []

        async with update_lock:
            updated_ids = apply_knowledge_updates(self.knowledge_pool, updates)
            if updated_ids:
                write_knowledge_pool_csv(self.final_klg_pool_save_path, self.knowledge_pool)
        return updated_ids

    async def run_problem(
        self,
        orchestrator: Any,
        patch_manager: PatchWorkspaceManager,
        dataset_entry: Dict[str, Any],
        index: int,
        entry_id: str,
        *,
        checkpoint: Optional[EvalCheckpoint],
        update_lock: asyncio.Lock,
        llm_request: Optional[LLMRequest] = None,
    ) -> Dict[str, Any]:
        problem_text = extract_problem_text(dataset_entry)
        prompt_text = build_task_prompt(dataset_entry) or problem_text
        if not prompt_text:
            prompt_text = "Investigate the regression and repair the affected code paths."
        question = build_task_prompt(dataset_entry) or prompt_text
        reference_answer = extract_reference_answer(dataset_entry)
        csv_path = str(dataset_entry.get("CSVPath") or "").strip()

        patch_path = patch_manager.resolve_patch_path(dataset_entry, index)
        patch_text = patch_path.read_text(encoding="utf-8")

        n_no_klg_trials = max(self.n_no_klg_trials, 0)
        n_klg_trials = max(self.n_klg_trials, 0)
        top_n_sample_klg = max(self.top_n_sample_klg, 1)
        base_prompt_messages = [{"role": "user", "content": prompt_text}]
        log_dir = Path(orchestrator.log_dir)
        knowledge_search_config = build_knowledge_search_config(
            top_n_sample_klg=top_n_sample_klg,
            distribution_decay_rate=self.distribution_decay_rate,
            random_klg_sampling_rate=self.random_klg_sampling_rate,
            klg_sample_mode=self.klg_sample_mode,
        )

        async def _run_trials(
            *,
            trials: int,
            label: str,
            use_knowledge: bool,
        ) -> Tuple[List[Dict[str, Any]], List[float]]:
            if trials <= 0:
                return [], []
            trial_records: List[Dict[str, Any]] = []
            valid_scores: List[float] = []
            for trial in range(trials):
                rerun_messages = randomize_system_prompt(
                    base_prompt_messages,
                    use_knowledge_prompt=use_knowledge,
                )
                knowledge_config: Optional[Dict[str, Any]] = None
                if use_knowledge:
                    knowledge_path = (
                        self.final_klg_pool_save_path
                        if self.final_klg_pool_save_path.exists()
                        else self.klg_pool_load_path
                    )
                    knowledge_config = {"load_knowledge_path": knowledge_path}

                run_name = f"{self.run_name_prefix}_{index:04d}_{label}_r{trial + 1}"
                result = await self.run_orchestrator_with_patch(
                    orchestrator,
                    patch_manager,
                    dataset_entry=dataset_entry,
                    dataset_index=index,
                    messages=rerun_messages,
                    run_name=run_name,
                    knowledge_config=knowledge_config,
                    knowledge_search_config=knowledge_search_config if use_knowledge else None,
                    knowledge_search_mode=self.klg_sample_mode if use_knowledge else None,
                    checkpoint=checkpoint,
                    entry_id=entry_id,
                    llm_request=llm_request,
                )

                run_id = normalize_result_run_id(result, log_dir, self._run_id_cache)
                output = result.get("output", "")
                messages = get_messages_for_run(result, log_dir)
                used_knowledge = (
                    extract_knowledge_entries_from_messages(messages, orchestrator)
                    if use_knowledge
                    else []
                )
                score_info = await self.score_answer(
                    orchestrator.llm,
                    question,
                    reference_answer,
                    output,
                    patch=patch_text,
                    messages=messages,
                    knowledge_entries=used_knowledge,
                    llm_request=llm_request,
                )
                score = score_info.get("score", 0.0) or 0.0
                feedback = score_info.get("feedback")
                if not is_bugged_score(score, feedback):
                    valid_scores.append(score)
                if use_knowledge and self.enable_update_klg_pool:
                    await self.update_knowledge_after_scoring(
                        orchestrator.llm,
                        question=question,
                        reference_answer=reference_answer,
                        patch=patch_text,
                        messages=messages,
                        knowledge_entries=used_knowledge,
                        update_lock=update_lock,
                        llm_request=llm_request,
                    )
                record: Dict[str, Any] = {
                    "trial": trial + 1,
                    "run_name": run_name,
                    "run_id": run_id,
                    "score": score,
                    "score_feedback": feedback,
                    "output": output,
                }
                if use_knowledge:
                    record["knowledge_used"] = used_knowledge
                trial_records.append(record)
            return trial_records, valid_scores

        no_klg_trials, no_klg_valid_scores = await _run_trials(
            trials=n_no_klg_trials,
            label="no_klg",
            use_knowledge=False,
        )
        klg_trials, klg_valid_scores = await _run_trials(
            trials=n_klg_trials,
            label="klg",
            use_knowledge=True,
        )

        no_klg_mean, no_klg_max, no_klg_min = summarize_scores(no_klg_valid_scores)
        klg_mean, klg_max, klg_min = summarize_scores(klg_valid_scores)

        record = {
            "id": entry_id,
            "problem": problem_text,
            "csv_path": csv_path,
            "patch_path": format_relative_path(patch_path, self.repo_root),
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

        print(
            f"[sampling] sample={index} no_klg_mean={no_klg_mean} "
            f"klg_mean={klg_mean} delta={round(klg_mean - no_klg_mean, 2)}"
        )
        return record

    def load_dataset(self) -> List[Dict[str, Any]]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"dataset must be a JSON array: {self.dataset_path}")
        return [dict(item) if isinstance(item, dict) else {"problem": str(item)} for item in raw]

    async def run_async(
        self,
        *,
        dataset: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
        max_problems: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        log_state = setup_log_output(save_log=self.save_log, log_dir=self.log_dir, prefix="sampling")
        try:
            dataset_entries = dataset if dataset is not None else self.load_dataset()
            resolved_output = Path(output_path) if output_path is not None else self.output_path
            resolved_max_problems = self.max_problems if max_problems is None else max_problems

            patch_files = sorted(self.patch_dir.glob("*.txt"))
            if not patch_files:
                raise RuntimeError(f"No patch files found under {self.patch_dir}")
            workspace_count = min(max(self.batch_size, 1), len(patch_files))
            workspace_pool = self._build_workspace_pool(workspace_count)

            if self.resume:
                eval_entries, processed_ids, needs_resave, run_cache = load_existing_eval_entries(
                    dataset_entries,
                    resolved_output,
                )
            else:
                eval_entries, processed_ids, needs_resave, run_cache = [], set(), False, {}

            checkpoint = EvalCheckpoint(
                output_path=resolved_output,
                eval_entries=eval_entries,
                run_cache=run_cache,
                lock=asyncio.Lock(),
                run_id_cache=self._run_id_cache,
                schema_version=self.schema_version,
            )

            if needs_resave:
                save_eval_entries(
                    eval_entries,
                    resolved_output,
                    run_cache=checkpoint.run_cache,
                    schema_version=self.schema_version,
                )

            pending: List[Tuple[int, Dict[str, Any], str]] = []
            for idx, entry in enumerate(dataset_entries):
                entry_id = compute_entry_id(entry, idx)
                if self.resume and entry_id in processed_ids:
                    continue
                pending.append((idx, entry, entry_id))
                if resolved_max_problems is not None and len(pending) >= resolved_max_problems:
                    break

            if not pending:
                print(
                    f"[sampling] All {len(processed_ids)} entries already processed. "
                    f"Results stored at {resolved_output}"
                )
                if self.enable_update_klg_pool and self.knowledge_pool:
                    write_knowledge_pool_csv(self.final_klg_pool_save_path, self.knowledge_pool)
                return eval_entries

            print(
                f"[sampling] Starting {len(pending)} problems with concurrency={workspace_count}"
            )

            knowledge_update_lock = asyncio.Lock()
            llm_request = LLMRequest(self.batch_size)
            await llm_request.start()
            try:
                for batch_start in range(0, len(pending), workspace_count):
                    batch_slice = pending[batch_start: batch_start + workspace_count]
                    batch_idx = batch_start // workspace_count + 1
                    print(
                        f"[sampling] Starting batch {batch_idx} with {len(batch_slice)} problems"
                    )

                    async def _run_one_problem(
                        handle: WorkspaceHandle,
                        dataset_idx: int,
                        entry: Dict[str, Any],
                        entry_id: str,
                    ) -> Tuple[str, Dict[str, Any]]:
                        result = await self.run_problem(
                            handle.orchestrator,
                            handle.patch_manager,
                            entry,
                            dataset_idx,
                            entry_id,
                            checkpoint=checkpoint,
                            update_lock=knowledge_update_lock,
                            llm_request=llm_request,
                        )
                        return entry_id, result

                    batch_tasks: List[asyncio.Task] = []
                    for handle, (dataset_idx, entry, entry_id) in zip(workspace_pool, batch_slice):
                        task = asyncio.create_task(_run_one_problem(handle, dataset_idx, entry, entry_id))
                        batch_tasks.append(task)

                    for task in asyncio.as_completed(batch_tasks):
                        record: Optional[Dict[str, Any]] = None
                        try:
                            entry_id, record = await task
                        except Exception as exc:
                            print(f"[sampling] Problem failed: {exc}")

                        if record:
                            rid = str(record.get("id") or "").strip()
                            if rid and rid not in processed_ids:
                                eval_entries.append(record)
                                processed_ids.add(rid)

                        async with checkpoint.lock:
                            save_eval_entries(
                                eval_entries,
                                resolved_output,
                                run_cache=checkpoint.run_cache,
                                schema_version=self.schema_version,
                            )
            finally:
                await llm_request.close()

            if self.enable_update_klg_pool and self.knowledge_pool:
                write_knowledge_pool_csv(self.final_klg_pool_save_path, self.knowledge_pool)

            save_eval_entries(
                eval_entries,
                resolved_output,
                run_cache=checkpoint.run_cache,
                schema_version=self.schema_version,
            )
            print(f"[sampling] Saved {len(eval_entries)} entries to {resolved_output}")
            return eval_entries
        finally:
            close_log_output(log_state)

    def run(
        self,
        *,
        dataset: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
        max_problems: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Sync wrapper for notebook/script use."""
        return asyncio.run(
            self.run_async(
                dataset=dataset,
                output_path=output_path,
                max_problems=max_problems,
            )
        )

__all__ = [
    "Sampling",
    "LLMRequest",
]
