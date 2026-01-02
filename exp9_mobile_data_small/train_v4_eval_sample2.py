import argparse
import asyncio
import json
import random
import shutil
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from seimei import load_run_messages, seimei
from seimei.editing import PatchApplyError, PatchParseError, apply_patch_to_workspace

# v4 eval sample notes (vs v3):
# - Major change: knowledge is selected from DEFAULT_KNOWLEDGE_POOL (by id) instead of generated from scratch.
# - Enforce unique knowledge ids per step within a problem across chunks.
# - Algorithm: run base chunks, then for each step pick pool knowledge per chunk, rerun with injection, and
#   finish with reruns comparing base vs knowledge-assisted runs.

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
CSV_DIR = "csv"
DEFAULT_DATASET_PATH = EXP_DIR / "dataset.json"
DEFAULT_RESULT_PATH = EXP_DIR / "train_v4_eval_sample_results.json"
DEFAULT_LLM_URL = None
DEFAULT_LLM_MODEL_NAME = "gpt-5-nano"
#DEFAULT_LLM_MODEL_NAME = "/workspace/gpt-oss-20b"
DEFAULT_RM_URL = "https://j4s6oyznxb8j3v-8000.proxy.runpod.net/rmsearch"
DEFAULT_BATCH_SIZE = 10
DEFAULT_N_KNOWLEDGE_STEPS = 3
DEFAULT_KNOWLEDGE_PER_STEP = 3
DEFAULT_N_CHECK_KNOWLEDGE = 3
DEFAULT_FINAL_RERUNS = 7
WORKSPACE_ROOT = EXP_DIR / "_workspace_copies"

BASE_SYSTEM_PROMPT_LIST = [
    "Act as a senior Fortran plasma physicist: inspect the local repo, reason about the magnetic-field terms, "
    "edit the source carefully, and summarize the exact patches you applied.",
    "Work like a responsible HPC debugger—diff the relevant modules, trace the control flow, and document the "
    "precise code edits that resolve the regression.",
    "Channel a gyrokinetic code maintainer: read the bug report, open the Fortran files, add or remove lines "
    "surgically, then explain why the change restores the missing physics.",
    "Be a cautious tokamak simulation engineer who tests hypotheses against the source, edits with apply_patch, "
    "and double-checks each assumption before writing the final note.",
    "Think like a numerical physicist reviewing electromagnetic solvers: inspect coefficients, restore missing "
    "operators, and narrate the fix with references to specific routines.",
    "Operate as a debugging lead: outline the failure mode, open the file, add the missing calls or loops, and "
    "describe the scientific impact of the change.",
    "Take the role of a patch surgeon—identify the minimal diff required, keep the edits consistent with coding "
    "style, and justify how the fix affects simulations.",
    "Behave as an HPC maintainer who validates interface contracts, reinstates removed code paths, and records "
    "the before/after physics effect.",
    "Think as a code-review mentor: reproduce the bug mentally, craft the Fortran changes step by step, and "
    "document what each edit re-enables.",
    "Act like an integration engineer ensuring GKV regressions are fixed; reason about boundary conditions, "
    "tweak the loops, and clearly explain the resulting behavior.",
]

KLG_SYSTEM_PROMPT_LIST = [
    "Treat each knowledge snippet as a mini patch review: restate the code cue, inspect the matching lines, and explain how to adjust them.",
    "Use the knowledge hints as guardrails by quoting the routine or loop they mention, checking that context, and guiding the edit there.",
    "Think like a reviewer handing you TODO comments—translate each snippet into a concrete Fortran action and report the result.",
    "Anchor every move to the knowledge cue: name the variables it references, open that block, and describe the precise change.",
    "Consider the knowledge text mandatory checkpoints; for each, cite the routine, verify current behavior, and note the edit to make.",
    "Behave as a cautious maintainer who paraphrases the knowledge, inspects the code around it, and ties conclusions directly back.",
    "Use the knowledge to prioritize lines: mention the snippet, map it to the workspace, and describe the instrumentation or edit you will run.",
    "Let the knowledge drive your micro-plan—quote it, echo the relevant arrays or flags, and keep reasoning tethered to that instruction.",
    "Imagine the knowledge as diff hunks; state the intended change, verify the file, and reason about its impact before moving on.",
    "Weave the knowledge cues into your debugging narrative by mirroring their language and pointing to the exact Fortran constructs involved.",
]

DEFAULT_KNOWLEDGE_POOL: List[Dict[str, Any]] = [
    {
        "id": "code_ls_inventory",
        "agent": "code_act",
        "step": None,
        "text": "Use `ls -a` to inventory the repo before touching files so you know which modules, patches, or scripts exist.",
        "tags": ["shell", "ls", "context"],
    },
    {
        "id": "code_pwd_confirm",
        "agent": "code_act",
        "step": None,
        "text": "Run `pwd` and confirm you are inside the experiment workspace before editing paths.",
        "tags": ["shell", "pwd", "sanity-check"],
    },
    {
        "id": "code_rg_search",
        "agent": "code_act",
        "step": None,
        "text": "Run `rg -n \"keyword\"` across the repo to locate definitions or uses before assuming semantics.",
        "tags": ["shell", "rg", "search"],
    },
    {
        "id": "code_rg_fortran_files",
        "agent": "code_act",
        "step": None,
        "text": "List Fortran sources with `rg --files -g '*.f90'` to scope the search surface.",
        "tags": ["shell", "rg", "inventory"],
    },
    {
        "id": "code_rg_module",
        "agent": "code_act",
        "step": None,
        "text": "Search for module or routine names with `rg -n \"module_name\" -g '*.f90'` to find declarations.",
        "tags": ["shell", "rg", "fortran"],
    },
    {
        "id": "code_rg_call_sites",
        "agent": "code_act",
        "step": None,
        "text": "Use `rg -n \"call routine_name\" -g '*.f90'` to find call sites and verify control flow.",
        "tags": ["shell", "rg", "call-graph"],
    },
    {
        "id": "code_rg_flags",
        "agent": "code_act",
        "step": None,
        "text": "Track a config flag via `rg -n \"flag_name\"` to see where branches diverge.",
        "tags": ["shell", "rg", "config"],
    },
    {
        "id": "code_head_preview",
        "agent": "code_act",
        "step": None,
        "text": "Call `head -n 40 path/to/file` to see file headers, module names, and includes.",
        "tags": ["shell", "head", "preview"],
    },
    {
        "id": "code_tail_logs",
        "agent": "code_act",
        "step": None,
        "text": "Use `tail -n 40` on build or run logs to spot the most recent error lines.",
        "tags": ["shell", "tail", "sanity-check"],
    },
    {
        "id": "code_wc_loc",
        "agent": "code_act",
        "step": None,
        "text": "`wc -l file.f90` gives a quick sense of file size before deep edits.",
        "tags": ["shell", "wc", "metrics"],
    },
    {
        "id": "code_sed_context",
        "agent": "code_act",
        "step": None,
        "text": "Use `sed -n '120,200p' file.f90` or `nl -ba file.f90 | sed -n '120,200p'` to view numbered context.",
        "tags": ["shell", "sed", "preview"],
    },
    {
        "id": "code_diff_versions",
        "agent": "code_act",
        "step": None,
        "text": "Use `git diff -U3 path/to/file` to compare local edits with expected changes.",
        "tags": ["shell", "git", "diff"],
    },
    {
        "id": "code_git_status",
        "agent": "code_act",
        "step": None,
        "text": "Run `git status -sb` to confirm which files changed before and after patching.",
        "tags": ["shell", "git", "status"],
    },
    {
        "id": "code_rg_preproc",
        "agent": "code_act",
        "step": None,
        "text": "Search preprocessor guards with `rg -n \"#if|#ifdef\" -g '*.[Ff]90'` to check compile-time branches.",
        "tags": ["shell", "rg", "preprocessor"],
    },
    {
        "id": "code_rg_parameters",
        "agent": "code_act",
        "step": None,
        "text": "Find constants with `rg -n \"parameter\" -g '*.f90'` to locate default values.",
        "tags": ["shell", "rg", "constants"],
    },
    {
        "id": "code_python_find_defs",
        "agent": "code_act",
        "step": None,
        "text": "Use a short Python regex to list subroutine/module names so you can map the file structure quickly.",
        "tags": ["python", "parsing", "fortran"],
    },
    {
        "id": "code_python_find_assign",
        "agent": "code_act",
        "step": None,
        "text": "Write a Python snippet to scan for assignments to a target variable and print line numbers.",
        "tags": ["python", "search", "variables"],
    },
    {
        "id": "code_python_extract_block",
        "agent": "code_act",
        "step": None,
        "text": "Use Python to extract and print a specific block (between two markers) when the file is large.",
        "tags": ["python", "parsing", "context"],
    },
    {
        "id": "code_python_call_graph",
        "agent": "code_act",
        "step": None,
        "text": "Use Python to scan `call` statements across files and build a quick caller list for a routine.",
        "tags": ["python", "call-graph", "analysis"],
    },
    {
        "id": "code_python_compare_files",
        "agent": "code_act",
        "step": None,
        "text": "Use Python's difflib to compare two versions of a source file and isolate the minimal delta.",
        "tags": ["python", "diff", "analysis"],
    },
    {
        "id": "code_rg_runtime_params",
        "agent": "code_act",
        "step": None,
        "text": "Search namelist or input parameters with `rg -n \"&\" -g '*.nml'` or `rg -n \"namelist\"` to trace runtime flags.",
        "tags": ["shell", "rg", "runtime"],
    },
    {
        "id": "code_find_config_files",
        "agent": "code_act",
        "step": None,
        "text": "List relevant configs with `rg --files -g '*.nml' -g '*.in'` so you know where inputs live.",
        "tags": ["shell", "rg", "config"],
    },
    {
        "id": "code_rg_todo",
        "agent": "code_act",
        "step": None,
        "text": "Use `rg -n \"TODO|FIXME\"` to see if there are hints about known issues.",
        "tags": ["shell", "rg", "context"],
    },
    {
        "id": "code_python_symbol_counts",
        "agent": "code_act",
        "step": None,
        "text": "Use Python to count how often a symbol appears per file to prioritize where to inspect first.",
        "tags": ["python", "metrics", "analysis"],
    },
    {
        "id": "code_checksum_validate",
        "agent": "code_act",
        "step": None,
        "text": "Compute `md5sum path/to/file` before and after edits to confirm you are modifying the intended file.",
        "tags": ["shell", "md5sum", "integrity"],
    },
]

SCORING_SYSTEM_PROMPT = (
    "You are an impartial evaluator scoring an assistant's answer against a reference answer. "
    "Judge factual accuracy, coverage, and clarity. Return ONLY a JSON object with keys 'score' "
    "(integer 0-10) and 'feedback' (concise justification). Score 0 means entirely incorrect, "
    "10 means fully correct."
)

KNOWLEDGE_SYSTEM_PROMPT = (
    "You provide concise, reusable advice (1-3 short lines) that nudges the agent back onto a reliable "
    "reasoning path without giving away the final answer."
)

KNOWLEDGE_SELECTION_SYSTEM_PROMPT = (
    "You select the most helpful reusable knowledge snippet from the pool to inject before the "
    "next agent action. Respond only with JSON."
)

RUN_ID_CACHE: Dict[str, str] = {}

# ---------- NEW: durable run cache + incremental saving ----------
CACHE_SCHEMA_VERSION = 2


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_json_size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return 10**9


def _compact_msg_history(history: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Keep a bounded msg_history as a fallback if run logs are missing.
    Prefer not to bloat the results file.
    """
    if not isinstance(history, list) or not history:
        return None
    if len(history) > 160:
        return None
    # Bound total JSON size (rough)
    if _safe_json_size(history) > 250_000:
        return None
    compact: List[Dict[str, Any]] = []
    for msg in history:
        if isinstance(msg, dict):
            compact.append(dict(msg))
    return compact or None


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
        # Return a shallow copy so callers can safely mutate fields like run_id normalization.
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
        """
        Persist minimal run outputs immediately after each orchestrator call.
        This enables resuming by skipping runs that already exist in run_cache.
        """
        key = self.make_run_key(entry_id, run_name)
        normalize_result_run_id(result, Path(orchestrator_log_dir))
        payload: Dict[str, Any] = {
            "entry_id": entry_id,
            "dataset_index": int(dataset_index),
            "run_name": run_name,
            "run_id": str(result.get("run_id") or "").strip(),
            "output": result.get("output", ""),
            "saved_at": _now_iso(),
        }
        msg_history = _compact_msg_history(result.get("msg_history"))
        if msg_history is not None:
            payload["msg_history"] = msg_history

        async with self.lock:
            # Avoid unnecessary rewrites if identical cached entry exists.
            prev = self.run_cache.get(key)
            if isinstance(prev, dict) and prev.get("run_id") == payload.get("run_id") and prev.get("output") == payload.get("output"):
                return
            self.run_cache[key] = payload
            # Must save after every run ends (user requirement).
            save_eval_entries(self.eval_entries, self.output_path, run_cache=self.run_cache)


class LLMRequestQueue:
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
            raise RuntimeError("LLMRequestQueue is closed.")
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


async def _run_llm_request(
    llm_request: Optional[LLMRequestQueue],
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if llm_request is None:
        return await func(*args, **kwargs)
    return await llm_request.request(func, *args, **kwargs)


def _sanitize_id_component(text: str, *, limit: int = 48) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_", " "}) else " " for ch in text)
    collapsed = "_".join(cleaned.split())
    return collapsed[:limit]


def extract_problem_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("Question") or "").strip()


def extract_reference_answer(entry: Dict[str, Any]) -> str:
    return str(entry.get("answer") or entry.get("CorrectAnswer") or "").strip()


def build_task_prompt(entry: Dict[str, Any]) -> str:
    csv_path = str(entry.get("CSVPath") or "").strip()
    question = str(entry.get("Question") or "").strip()
    if csv_path and question:
        return f"Analyze inside {csv_path} and answer the question below:\n\n{question}"
    return question


def format_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


async def run_orchestrator(
    orchestrator,
    #patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    dataset_index: int,
    messages: Sequence[Dict[str, Any]],
    run_name: str,
    knowledge_config: Dict[str, Any],
    checkpoint: Optional[EvalCheckpoint] = None,
    entry_id: Optional[str] = None,
    llm_request: Optional[LLMRequestQueue] = None,
) -> Dict[str, Any]:
    """
    NEW behavior:
    - If checkpoint has a cached run for (entry_id, run_name), return it and skip inference.
    - Otherwise, run inference, then immediately persist the result via save_eval_entries.
    """
    resolved_entry_id = entry_id or compute_entry_id(dataset_entry, dataset_index)
    if checkpoint is not None:
        cached = checkpoint.get_cached(entry_id=resolved_entry_id, run_name=run_name)
        if cached:
            # Ensure minimal keys exist for downstream code.
            cached.setdefault("run_id", cached.get("run_id") or "")
            cached.setdefault("output", cached.get("output") or "")
            if "msg_history" in cached:
                cached["msg_history"] = cached.get("msg_history")
            return cached

    #with patch_manager.apply_for_problem(dataset_entry, dataset_index):
    result = await _run_llm_request(
        llm_request,
        orchestrator,
        messages=messages,
        run_name=run_name,
        knowledge_config=knowledge_config,
        #workspace=patch_manager.workspace,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge injection by generating guidance for the first N agent steps."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset JSON used for evaluation.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Where to store evaluation traces and scores.",
    )
    parser.add_argument(
        "--llm-url",
        default=DEFAULT_LLM_URL,
        help="LLM endpoint passed to the orchestrator. Set None to use openai API",
    )
    parser.add_argument(
        "--llm-model-name",
        default=DEFAULT_LLM_MODEL_NAME,
        help="LLM model name passed to the orchestrator.",
    )
    parser.add_argument(
        "--rm-url",
        default=DEFAULT_RM_URL,
        help="Reward model search endpoint passed to the orchestrator.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum number of concurrent LLM requests.",
    )
    parser.add_argument(
        "--n-knowledge-steps",
        type=int,
        default=DEFAULT_N_KNOWLEDGE_STEPS,
        help="How many leading agent steps should receive knowledge generation.",
    )
    parser.add_argument(
        "--knowledge-per-step",
        type=int,
        default=DEFAULT_KNOWLEDGE_PER_STEP,
        help="How many knowledge candidates to generate per step.",
    )
    parser.add_argument(
        "--n-check-knowledge",
        type=int,
        default=DEFAULT_N_CHECK_KNOWLEDGE,
        help="How many reruns to execute per knowledge chunk when selecting the best chunk.",
    )
    parser.add_argument(
        "--final-reruns",
        type=int,
        default=DEFAULT_FINAL_RERUNS,
        help="How many reruns to execute for the base and knowledge-injected solutions when comparing final scores.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optional cap on the number of dataset entries to evaluate.",
    )
    return parser.parse_args()


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


def pick_system_prompt(*, use_knowledge_prompt: bool = False) -> str:
    pool = (
        KLG_SYSTEM_PROMPT_LIST
        if use_knowledge_prompt and KLG_SYSTEM_PROMPT_LIST
        else BASE_SYSTEM_PROMPT_LIST
    )
    return random.choice(pool)


def pick_base_system_prompt() -> str:
    return pick_system_prompt()


def randomize_system_prompt(
    messages: Sequence[Dict[str, Any]], *, use_knowledge_prompt: bool = False
) -> List[Dict[str, Any]]:
    cloned = [dict(msg) for msg in messages]
    prompt = pick_system_prompt(use_knowledge_prompt=use_knowledge_prompt)
    if cloned and str(cloned[0].get("role") or "").lower() == "system":
        cloned[0]["content"] = prompt
    else:
        cloned.insert(0, {"role": "system", "content": prompt})
    return cloned


def _strip_code_fences(text: str) -> str:
    snippet = text.strip()
    if snippet.startswith("```"):
        lines = snippet.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        snippet = "\n".join(lines).strip()
    return snippet


def _parse_json_response(raw: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(raw)
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def _parse_json_array(raw: str) -> List[Any]:
    cleaned = _strip_code_fences(raw)
    candidates = [cleaned]
    if "[" in cleaned and "]" in cleaned:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if 0 <= start < end:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    return []


def _normalize_step_filter(value: Any) -> Optional[Set[int]]:
    if value in (None, "", "*"):
        return None
    normalized: Set[int] = set()
    if isinstance(value, int):
        normalized.add(value)
        return normalized
    if isinstance(value, (list, tuple, set)):
        for item in value:
            try:
                normalized.add(int(item))
            except (TypeError, ValueError):
                continue
        return normalized or None
    try:
        normalized.add(int(value))
    except (TypeError, ValueError):
        return None
    return normalized or None


def _normalize_pool_entry(entry: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    text = str(entry.get("text") or entry.get("knowledge") or "").strip()
    if not text:
        return None
    normalized: Dict[str, Any] = {
        "id": str(entry.get("id") or f"default_pool_{index + 1}").strip(),
        "agent": str(entry.get("agent") or "think").strip() or "think",
        "text": text,
        "original_text": entry.get("original_text") or text,
        "tags": entry.get("tags") or [],
    }
    normalized["_step_filter"] = _normalize_step_filter(entry.get("step"))
    return normalized


def _format_pool_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for entry in candidates:
        formatted.append(
            {
                "id": entry.get("id"),
                "agent": entry.get("agent"),
                "text": entry.get("text"),
                "tags": entry.get("tags") or [],
            }
        )
    return formatted


def _prepare_pool_candidates(step: int, used_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    used = used_ids or set()
    applicable: List[Dict[str, Any]] = []
    for idx, raw_entry in enumerate(DEFAULT_KNOWLEDGE_POOL):
        normalized = _normalize_pool_entry(raw_entry, idx)
        if not normalized:
            continue
        step_filter = normalized.pop("_step_filter", None)
        if step_filter is not None and step not in step_filter:
            continue
        if normalized["id"] in used:
            continue
        applicable.append(normalized)
    return applicable


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 10:
        return 10.0
    return round(score, 2)


def _is_bugged_score(score: float, feedback: Optional[str]) -> bool:
    if score != 0.0:
        return False
    if feedback is None:
        return True
    if isinstance(feedback, str) and not feedback.strip():
        return True
    return False


def _resolve_run_dir_name(run_id: Optional[str], log_dir: Path) -> Optional[str]:
    run_id_str = str(run_id or "").strip()
    if not run_id_str:
        return None
    cached = RUN_ID_CACHE.get(run_id_str)
    if cached:
        return cached
    if not log_dir.exists():
        return run_id_str
    direct_candidate = log_dir / run_id_str
    if direct_candidate.exists():
        RUN_ID_CACHE[run_id_str] = run_id_str
        return run_id_str
    short = run_id_str[:8]
    if short:
        matches = sorted(log_dir.glob(f"run-*-{short}"))
        if matches:
            resolved = matches[-1].name
            RUN_ID_CACHE[run_id_str] = resolved
            return resolved
    for candidate in log_dir.glob("run-*"):
        meta_path = candidate / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta_run_id = str(meta_data.get("run_id") or "").strip()
        if meta_run_id == run_id_str:
            resolved = candidate.name
            RUN_ID_CACHE[run_id_str] = resolved
            return resolved
    return run_id_str


def normalize_result_run_id(result: Dict[str, Any], log_dir: Path) -> str:
    resolved = _resolve_run_dir_name(result.get("run_id"), Path(log_dir))
    if resolved:
        result["run_id"] = resolved
    return str(result.get("run_id") or "")


def extract_agent_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [msg for msg in messages if msg.get("role") == "agent"]


def get_agent_step_text(messages: Sequence[Dict[str, Any]], step: int) -> str:
    agent_msgs = extract_agent_messages(messages)
    if step < 1 or step > len(agent_msgs):
        return ""
    content = agent_msgs[step - 1].get("content", "")
    return str(content)


def get_agent_name_for_step(messages: Sequence[Dict[str, Any]], step: int) -> Optional[str]:
    agent_msgs = extract_agent_messages(messages)
    if step < 1 or step > len(agent_msgs):
        return None
    return agent_msgs[step - 1].get("name")


def truncate_messages_before_step(messages: Sequence[Dict[str, Any]], step: int) -> List[Dict[str, Any]]:
    if step <= 1:
        truncated: List[Dict[str, Any]] = []
        agent_seen = 0
        for msg in messages:
            if msg.get("role") == "agent":
                agent_seen += 1
                if agent_seen >= 1:
                    break
            truncated.append(dict(msg))
        return truncated

    truncated = []
    agent_seen = 0
    for msg in messages:
        if msg.get("role") == "agent":
            agent_seen += 1
            if agent_seen >= step:
                break
        truncated.append(dict(msg))
    return truncated


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"generate_knowledge": False}
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


def get_messages_for_run(
    run_result: Dict[str, Any],
    log_dir: Path,
    *,
    max_agent_step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    def _normalize(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        agent_seen = 0
        for msg in entries:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            if role == "assistant":
                continue
            filtered.append(dict(msg))
            if role == "agent":
                agent_seen += 1
                if max_agent_step is not None and agent_seen >= max_agent_step:
                    break
        return filtered

    history = run_result.get("msg_history")
    if isinstance(history, list) and history:
        return _normalize(history)
    run_id = run_result.get("run_id")
    if not run_id:
        return []
    try:
        raw_messages = load_run_messages(run_id, runs_dir=log_dir, step=max_agent_step)
    except FileNotFoundError:
        # Fallback: if this run_result came from run_cache and included msg_history, _normalize above handles it.
        return []
    return _normalize(raw_messages)


async def score_answer(
    orchestrator_llm,
    question: str,
    reference_answer: str,
    model_answer: str,
    *,
    llm_request: Optional[LLMRequestQueue] = None,
) -> Dict[str, Any]:
    prompt = (
        "Evaluate the model's answer.\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Model Answer:\n{model_answer}\n"
        "Respond strictly in JSON."
    )
    try:
        response, usage = await _run_llm_request(
            llm_request,
            orchestrator_llm.chat,
            messages=[{"role": "user", "content": prompt}],
            system=SCORING_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return {
            "score": 0.0,
            "feedback": f"Scoring failed: {exc}",
            "raw_response": "",
            "judge_usage": {},
        }

    parsed = _parse_json_response(response)
    score_value = _coerce_score(parsed.get("score"))
    feedback = parsed.get("feedback") or parsed.get("rationale") or ""

    return {
        "score": score_value,
        "feedback": feedback.strip(),
        "raw_response": response,
        "judge_usage": usage,
    }


async def generate_step_knowledge(
    llm_client,
    *,
    dataset_entry: Dict[str, Any],
    messages: Sequence[Dict[str, Any]],
    step: int,
    iteration: int,
    prior_knowledge: Optional[Sequence[Dict[str, Any]]] = None,
    used_knowledge_ids: Optional[Set[str]] = None,
    llm_request: Optional[LLMRequestQueue] = None,
) -> Optional[Dict[str, Any]]:
    question = build_task_prompt(dataset_entry)
    reference = extract_reference_answer(dataset_entry)
    transcript_json = json.dumps(messages, ensure_ascii=False, indent=2)
    step_text = get_agent_step_text(messages, step)
    knowledge_snapshot = [dict(entry) for entry in prior_knowledge or []]
    knowledge_json = json.dumps(knowledge_snapshot, ensure_ascii=False, indent=2)
    candidates = _prepare_pool_candidates(step, used_knowledge_ids)
    if not candidates:
        return None
    candidate_json = json.dumps(_format_pool_candidates(candidates), ensure_ascii=False, indent=2)
    used_ids_json = json.dumps(sorted(used_knowledge_ids or set()), ensure_ascii=False, indent=2)
    prompt = (
        f"You are selecting reusable knowledge that will be inserted before agent step {step} "
        "in a CSV reasoning workflow.\n\n"
        "Evaluation summary for the run that produced this transcript:\n"
        "- Knowledge snippets already injected (JSON array, matches this transcript exactly):\n"
        f"{knowledge_json}\n\n"
        f"Full message history before rerun (JSON transcript):\n{transcript_json}\n\n"
        f"Original agent step {step} transcript:\n{step_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        "Candidate knowledge pool (JSON array):\n"
        f"{candidate_json}\n\n"
        "Knowledge ids already used for this step in this problem (do not repeat them):\n"
        f"{used_ids_json}\n\n"
        "Your task:\n"
        "1. Choose the single best candidate knowledge entry for this step.\n"
        "2. Return only a JSON object with the chosen id and a 1-sentence justification.\n\n"
        "Output format (JSON only):\n"
        "{\n"
        "  \"id\": \"candidate_id\",\n"
        "  \"justification\": \"short reason\"\n"
        "}\n"
    )
    try:
        response, _ = await _run_llm_request(
            llm_request,
            llm_client.chat,
            messages=[{"role": "user", "content": prompt}],
            system=KNOWLEDGE_SELECTION_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[knowledge step {step} iter {iteration + 1}] selection failed: {exc}")
        return None

    parsed = _parse_json_response(response)
    selected_id = str(parsed.get("id") or parsed.get("selected_id") or "").strip()
    if not selected_id:
        array = _parse_json_array(response)
        if array and isinstance(array[0], dict):
            selected_id = str(array[0].get("id") or "").strip()
    id_map = {entry["id"]: entry for entry in candidates}
    selected = id_map.get(selected_id)
    if not selected:
        random.shuffle(candidates)
        selected = candidates[0]
    entry = dict(selected)
    entry.setdefault("step", step)
    entry.setdefault("iteration", iteration + 1)
    entry.setdefault("agent", entry.get("agent") or "think")
    entry.setdefault("tags", entry.get("tags") or [])
    entry.setdefault("original_text", entry.get("text") or "")
    return entry


async def run_candidate_inference(
    orchestrator,
    #patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    truncated_history: Sequence[Dict[str, Any]],
    step: int,
    knowledge_entry: Dict[str, Any],
    dataset_index: int,
    candidate_index: int,
    checkpoint: Optional[EvalCheckpoint],
    entry_id: str,
    llm_request: Optional[LLMRequestQueue] = None,
) -> Dict[str, Any]:
    rerun_messages = randomize_system_prompt(truncated_history, use_knowledge_prompt=True)
    manual_entries = [
        {
            "step": step,
            "agent": knowledge_entry.get("agent") or "think",
            "text": knowledge_entry.get("text", ""),
            "tags": knowledge_entry.get("tags") or [],
        }
    ]
    knowledge_config = build_knowledge_config(manual_entries)
    run_name = f"train_v4_eval_sample_{dataset_index:04d}_s{step}_k{candidate_index + 1}"
    result = await run_orchestrator(
        orchestrator,
        #patch_manager,
        dataset_entry=dataset_entry,
        dataset_index=dataset_index,
        messages=rerun_messages,
        run_name=run_name,
        knowledge_config=knowledge_config,
        checkpoint=checkpoint,
        entry_id=entry_id,
        llm_request=llm_request,
    )
    new_run_id = normalize_result_run_id(result, Path(orchestrator.log_dir))
    new_output = result.get("output", "")

    candidate_info = {
        "candidate_index": candidate_index + 1,
        "run_name": run_name,
        "run_id": new_run_id,
        "output": new_output,
        "knowledge": {
            "id": knowledge_entry.get("id"),
            "text": knowledge_entry.get("text"),
            "original_text": knowledge_entry.get("original_text"),
            "agent": knowledge_entry.get("agent"),
            "tags": knowledge_entry.get("tags") or [],
            "step": step,
            "iteration": knowledge_entry.get("iteration"),
        },
    }
    return {"info": candidate_info, "result": result}


async def run_full_problem_trials(
    orchestrator,
    #patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    base_prompt_messages: Sequence[Dict[str, Any]],
    manual_entries: Optional[List[Dict[str, Any]]],
    trials: int,
    dataset_index: int,
    label: str,
    checkpoint: Optional[EvalCheckpoint],
    entry_id: str,
    llm_request: Optional[LLMRequestQueue] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    if trials <= 0:
        return [], 0.0
    question = build_task_prompt(dataset_entry)
    reference = extract_reference_answer(dataset_entry)
    trial_records: List[Dict[str, Any]] = []
    valid_scores: List[float] = []
    for trial in range(trials):
        rerun_messages = randomize_system_prompt(
            base_prompt_messages, use_knowledge_prompt=bool(manual_entries)
        )
        knowledge_config = build_knowledge_config(
            [dict(entry) for entry in manual_entries] if manual_entries else None
        )
        run_name = f"train_v4_eval_sample_{dataset_index:04d}_{label}_r{trial + 1}"
        result = await run_orchestrator(
            orchestrator,
            #patch_manager,
            dataset_entry=dataset_entry,
            dataset_index=dataset_index,
            messages=rerun_messages,
            run_name=run_name,
            knowledge_config=knowledge_config,
            checkpoint=checkpoint,
            entry_id=entry_id,
            llm_request=llm_request,
        )
        run_id = normalize_result_run_id(result, Path(orchestrator.log_dir))
        output = result.get("output", "")
        score_info = await score_answer(
            orchestrator.llm,
            question,
            reference,
            output,
            llm_request=llm_request,
        )
        score = score_info.get("score", 0.0) or 0.0
        feedback = score_info.get("feedback")
        if not _is_bugged_score(score, feedback):
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


async def run_problem(
    orchestrator,
    #patch_manager: PatchWorkspaceManager,
    dataset_entry: Dict[str, Any],
    index: int,
    entry_id: str,
    *,
    n_knowledge_steps: int,
    knowledge_per_step: int,
    n_check_knowledge: int,
    final_reruns: int,
    checkpoint: Optional[EvalCheckpoint],
    llm_request: Optional[LLMRequestQueue] = None,
) -> Dict[str, Any]:
    problem_text = extract_problem_text(dataset_entry)
    prompt_text = build_task_prompt(dataset_entry) or problem_text
    if not prompt_text:
        prompt_text = "Investigate the regression and repair the affected code paths."
    reference_answer = extract_reference_answer(dataset_entry)
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    #try:
    #    patch_path = patch_manager.resolve_patch_path(dataset_entry, index)
    #except FileNotFoundError as exc:
    #    raise RuntimeError(f"[sample {index}] Missing patch file: {exc}") from exc

    base_prompt_messages = [{"role": "user", "content": prompt_text}]

    chunk_records: List[Dict[str, Any]] = []
    base_inferences: List[Dict[str, Any]] = []
    log_dir = Path(orchestrator.log_dir)

    for chunk_idx in range(knowledge_per_step):
        base_messages = randomize_system_prompt(base_prompt_messages)
        run_name = f"train_v4_eval_sample_{index:04d}_base_seed{chunk_idx + 1}"
        base_result = await run_orchestrator(
            orchestrator,
            #patch_manager,
            dataset_entry=dataset_entry,
            dataset_index=index,
            messages=[dict(msg) for msg in base_messages],
            run_name=run_name,
            knowledge_config=build_knowledge_config(),
            checkpoint=checkpoint,
            entry_id=entry_id,
            llm_request=llm_request,
        )
        base_run_id = normalize_result_run_id(base_result, log_dir)
        base_output = base_result.get("output", "")

        base_run = {
            "chunk_index": chunk_idx + 1,
            "run_name": run_name,
            "run_id": base_run_id,
            "output": base_output,
        }
        base_inferences.append(base_run)
        chunk_records.append(
            {
                "chunk_index": chunk_idx + 1,
                "base_run": base_run,
                "current_result": base_result,
                "current_run_id": base_run_id,
                "current_score": None,  # keep for compatibility
                "current_feedback": None,
                "manual_entries": [],
                "knowledge_steps": [],
                "score_history": [{"step": 0, "run_id": base_run_id, "score": None}],
            }
        )
        print(f"[sample {index}] chunk {chunk_idx + 1}")

    record: Dict[str, Any] = {
        "id": entry_id,
        "problem": problem_text,
        "csv_path": csv_path,
        #"patch_path": format_relative_path("./csv/"),
        "base_inferences": base_inferences,
        "chunks": [],
        "knowledge_chunk_mean_scores": [],
    }

    for step in range(1, n_knowledge_steps + 1):
        used_ids_for_step: Set[str] = set()
        for chunk_data in chunk_records:
            messages = get_messages_for_run(chunk_data["current_result"], log_dir)
            agent_messages = extract_agent_messages(messages)
            if step > len(agent_messages):
                print(
                    f"[sample {index}] chunk {chunk_data['chunk_index']} step {step}: "
                    "insufficient agent messages"
                )
                continue
            truncated_history = truncate_messages_before_step(messages, step)
            knowledge_entry = await generate_step_knowledge(
                orchestrator.llm,
                dataset_entry=dataset_entry,
                messages=messages,
                step=step,
                iteration=chunk_data["chunk_index"] - 1,
                prior_knowledge=[dict(entry) for entry in chunk_data["manual_entries"]],
                used_knowledge_ids=used_ids_for_step,
                llm_request=llm_request,
            )
            if not knowledge_entry:
                print(
                    f"[sample {index}] chunk {chunk_data['chunk_index']} step {step}: "
                    "no knowledge selected"
                )
                continue
            knowledge_id = str(knowledge_entry.get("id") or "").strip()
            if not knowledge_id:
                print(
                    f"[sample {index}] chunk {chunk_data['chunk_index']} step {step}: "
                    "knowledge missing id"
                )
                continue
            if knowledge_id in used_ids_for_step:
                print(
                    f"[sample {index}] chunk {chunk_data['chunk_index']} step {step}: "
                    "duplicate knowledge id detected"
                )
                continue
            used_ids_for_step.add(knowledge_id)
            if not knowledge_entry.get("agent"):
                knowledge_entry["agent"] = get_agent_name_for_step(messages, step) or "think"

            candidate = await run_candidate_inference(
                orchestrator,
                #patch_manager,
                dataset_entry=dataset_entry,
                truncated_history=truncated_history,
                step=step,
                knowledge_entry=knowledge_entry,
                dataset_index=index,
                candidate_index=chunk_data["chunk_index"] - 1,
                checkpoint=checkpoint,
                entry_id=entry_id,
                llm_request=llm_request,
            )
            candidate_info = candidate["info"]
            chunk_data["knowledge_steps"].append(
                {
                    "step": step,
                    "knowledge": candidate_info.get("knowledge"),
                    "run_name": candidate_info.get("run_name"),
                    "run_id": candidate_info.get("run_id"),
                    "output": candidate_info.get("output"),
                }
            )
            chunk_data["current_result"] = candidate["result"]
            chunk_data["current_run_id"] = candidate_info.get("run_id")

            manual_entry = {
                "step": step,
                "agent": candidate_info.get("knowledge", {}).get("agent") or "think",
                "text": candidate_info.get("knowledge", {}).get("text", ""),
                "tags": candidate_info.get("knowledge", {}).get("tags") or [],
            }
            if manual_entry["text"]:
                chunk_data["manual_entries"].append(manual_entry)

            chunk_data["score_history"].append(
                {
                    "step": step,
                    "score": chunk_data.get("current_score"),
                    "run_id": chunk_data["current_run_id"],
                }
            )
            print(f"[sample {index}] chunk {chunk_data['chunk_index']} step {step}:")

    for chunk_data in chunk_records:
        chunk_data["final_single_run"] = {
            "run_id": chunk_data["current_run_id"],
            "score": chunk_data.get("current_score"),
            "score_feedback": chunk_data.get("current_feedback"),
            "output": chunk_data["current_result"].get("output", ""),
        }
        chunk_entry = {
            "chunk_index": chunk_data["chunk_index"],
            "base_run": chunk_data["base_run"],
            "knowledge_steps": chunk_data["knowledge_steps"],
            "selected_knowledge": [dict(entry) for entry in chunk_data["manual_entries"]],
            "final_single_run": chunk_data["final_single_run"],
        }
        record["chunks"].append(chunk_entry)
        chunk_data["record_entry"] = chunk_entry

    knowledge_chunk_scores: List[float] = []
    best_chunk_mean: Optional[float] = None
    best_chunk_data: Optional[Dict[str, Any]] = None
    best_chunk_manual_entries: Optional[List[Dict[str, Any]]] = None

    for chunk_data in chunk_records:
        manual_entries = chunk_data["manual_entries"]
        label = f"chunk{chunk_data['chunk_index']}"
        chunk_trials, chunk_mean = await run_full_problem_trials(
            orchestrator,
            #patch_manager,
            dataset_entry=dataset_entry,
            base_prompt_messages=base_prompt_messages,
            manual_entries=manual_entries if manual_entries else None,
            trials=n_check_knowledge,
            dataset_index=index,
            label=label,
            checkpoint=checkpoint,
            entry_id=entry_id,
            llm_request=llm_request,
        )
        chunk_mean = round(chunk_mean, 2)
        knowledge_chunk_scores.append(chunk_mean)
        chunk_data["record_entry"]["chunk_trials"] = chunk_trials
        chunk_data["record_entry"]["chunk_mean_score"] = chunk_mean
        if best_chunk_mean is None or chunk_mean > best_chunk_mean:
            best_chunk_mean = chunk_mean
            best_chunk_data = chunk_data
            best_chunk_manual_entries = manual_entries if manual_entries else []

    record["knowledge_chunk_mean_scores"] = knowledge_chunk_scores
    if best_chunk_data:
        record["best_chunk_index"] = best_chunk_data["chunk_index"]
        record["best_chunk_mean_score"] = best_chunk_mean
        record["selected_knowledge"] = [dict(entry) for entry in best_chunk_manual_entries or []]
        record["final_single_run"] = best_chunk_data["final_single_run"]
        record["final_run_id"] = best_chunk_data["current_run_id"]
        record["final_score"] = best_chunk_data.get("current_score", 0.0) or 0.0
        record["final_output"] = best_chunk_data["final_single_run"]["output"]
        record["score_history"] = best_chunk_data.get("score_history", [])
    else:
        record["final_run_id"] = None
        record["final_score"] = 0.0
        record["final_output"] = ""
        record["score_history"] = []
        record["selected_knowledge"] = []
        record["final_single_run"] = None

    base_trials, base_mean = await run_full_problem_trials(
        orchestrator,
        #patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=None,
        trials=final_reruns,
        dataset_index=index,
        label="base_final",
        checkpoint=checkpoint,
        entry_id=entry_id,
        llm_request=llm_request,
    )
    knowledge_trials, knowledge_mean = await run_full_problem_trials(
        orchestrator,
        #patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=best_chunk_manual_entries if best_chunk_manual_entries else None,
        trials=final_reruns,
        dataset_index=index,
        label="knowledge_final",
        checkpoint=checkpoint,
        entry_id=entry_id,
        llm_request=llm_request,
    )

    record["fair_comparison"] = {
        "trials": final_reruns,
        "base_trials": base_trials,
        "knowledge_trials": knowledge_trials,
        "base_mean_score": base_mean,
        "knowledge_mean_score": knowledge_mean,
        "mean_improvement": round(knowledge_mean - base_mean, 2),
    }
    record["base_rerun_mean_score"] = base_mean
    record["final_rerun_mean_score"] = knowledge_mean

    print(
        f"[sample {index}] reruns avg base={base_mean} knowledge={knowledge_mean} "
        f"(Δ {round(knowledge_mean - base_mean, 2)})"
    )

    return record


def load_existing_eval_entries(
    dataset: List[Dict[str, Any]],
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Set[str], bool, Dict[str, Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    needs_resave = False
    run_cache: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                entries = raw.get("detail") or []
                rc = raw.get("run_cache") or {}
                if isinstance(rc, dict):
                    # ensure values are dicts
                    run_cache = {str(k): v for k, v in rc.items() if isinstance(v, dict)}
            elif isinstance(raw, list):
                entries = raw
                needs_resave = True
            else:
                print(f"[train_v4_eval_sample] Ignoring malformed cache at {output_path}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[train_v4_eval_sample] Failed to load cached eval entries: {exc}")
        for idx, entry in enumerate(entries):
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id and idx < len(dataset):
                entry_id = compute_entry_id(dataset[idx], idx)
                entry["id"] = entry_id
                needs_resave = True
            if entry_id:
                processed_ids.add(entry_id)
    return entries, processed_ids, needs_resave, run_cache


def build_eval_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {
            "total_problems": 0,
            "mean_score_improvement": 0.0,
            "base_vs_final": [],
            "overall_base_mean": 0.0,
            "overall_final_mean": 0.0,
            "win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
            "knowledge_chunk_mean_scores": [],
        }
    base_vs_final: List[List[float]] = []
    improvements: List[float] = []
    chunk_score_table: List[List[float]] = []
    for entry in entries:
        base_score = entry.get("base_rerun_mean_score")
        final_score = entry.get("final_rerun_mean_score")
        if base_score is None:
            base_score = entry.get("base", {}).get("score", 0.0)
        if final_score is None:
            final_score = entry.get("final_score", 0.0)
        base_val = round(float(base_score), 2)
        final_val = round(float(final_score), 2)
        base_vs_final.append([base_val, final_val])
        improvements.append(final_val - base_val)
        chunk_scores = entry.get("knowledge_chunk_mean_scores") or []
        chunk_score_table.append([round(float(score), 3) for score in chunk_scores])
    total = len(base_vs_final)
    mean_improvement = round(sum(improvements) / total, 4) if total else 0.0
    wins = sum(1 for diff in improvements if diff > 0)
    ties = sum(1 for diff in improvements if diff == 0)
    losses = total - wins - ties
    overall_base_mean = (
        round(sum(pair[0] for pair in base_vs_final) / total, 4) if total else 0.0
    )
    overall_final_mean = (
        round(sum(pair[1] for pair in base_vs_final) / total, 4) if total else 0.0
    )
    return {
        "total_problems": total,
        "mean_score_improvement": mean_improvement,
        "base_vs_final": base_vs_final,
        "overall_base_mean": overall_base_mean,
        "overall_final_mean": overall_final_mean,
        "win_loss_tie": {"win": wins, "tie": ties, "loss": losses},
        "knowledge_chunk_mean_scores": chunk_score_table,
    }


def save_eval_entries(
    entries: List[Dict[str, Any]],
    output_path: Path,
    *,
    run_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_obj: Dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "saved_at": _now_iso(),
        "summary": build_eval_summary(entries),
        "detail": entries,
    }
    if run_cache is not None:
        payload_obj["run_cache"] = run_cache
        payload_obj["run_cache_count"] = len(run_cache)

    payload = json.dumps(payload_obj, ensure_ascii=False, indent=2)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(output_path)


async def run_evaluation(args: argparse.Namespace) -> None:
    dataset = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    orchestrator = seimei(
        agent_config=[{"name": "code_act"}],
        llm_kwargs={"model": "gpt-5-nano"},
        rm_kwargs={"url": args.rm_url, "agent_routing": False, "knowledge_search": True},
        allow_code_exec=True,
        agent_log_head_lines=1,
        max_tokens_per_question=80000,
    )

    #patch_files = sorted(PATCH_DIR.glob("*.txt"))
    #if not patch_files:
    #    raise RuntimeError(f"No patch files found under {PATCH_DIR}")
    #workspace_count = min(args.batch_size, len(patch_files))
    #if workspace_count < 1:
    #    raise RuntimeError("batch_size must be >= 1 and patch_files must be non-empty.")
    #if args.batch_size != workspace_count:
    #    print(
    #        "[train_v4_eval_sample] Limiting concurrency to "
    #        f"{workspace_count} based on patch file count."
    #    )
    #workspace_pool = build_workspace_pool(args, workspace_count)

    eval_entries, processed_ids, needs_resave, run_cache = load_existing_eval_entries(
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
        print(f"[train_v4_eval_sample] Resuming with {len(processed_ids)} cached entries")
    if run_cache:
        print(f"[train_v4_eval_sample] Loaded run_cache with {len(run_cache)} runs")

    pending: List[Tuple[int, Dict[str, Any], str]] = []
    for idx, entry in enumerate(dataset):
        entry_id = compute_entry_id(entry, idx)
        if entry_id in processed_ids:
            continue
        pending.append((idx, entry, entry_id))
        if args.max_problems is not None and len(pending) >= args.max_problems:
            break

    if not pending:
        print(
            f"[train_v4_eval_sample] All {len(processed_ids)} entries already processed. "
            f"Results stored at {args.output_path}"
        )
        return

    llm_request = LLMRequestQueue(args.batch_size)
    await llm_request.start()
    try:
        problem_tasks: List[asyncio.Task] = []
        task_ids: Dict[asyncio.Task, str] = {}
        for dataset_idx, entry, entry_id in pending:
            task = asyncio.create_task(
                run_problem(
                    orchestrator,
                    entry,
                    dataset_idx,
                    entry_id,
                    n_knowledge_steps=args.n_knowledge_steps,
                    knowledge_per_step=args.knowledge_per_step,
                    n_check_knowledge=args.n_check_knowledge,
                    final_reruns=args.final_reruns,
                    checkpoint=checkpoint,
                    llm_request=llm_request,
                )
            )
            problem_tasks.append(task)
            task_ids[task] = entry_id

        for task in asyncio.as_completed(problem_tasks):
            try:
                record = await task
            except Exception as exc:  # pragma: no cover - runtime guard
                entry_id = task_ids.get(task, "unknown")
                print(f"[train_v4_eval_sample] Problem {entry_id} failed: {exc}")
                continue
            if not record:
                continue
            rid = str(record.get("id") or "").strip()
            if not rid or rid in processed_ids:
                continue
            eval_entries.append(record)
            processed_ids.add(rid)

            # Save after each completed problem while run_cache is updated per LLM run.
            async with checkpoint.lock:
                save_eval_entries(eval_entries, args.output_path, run_cache=checkpoint.run_cache)
            print(
                f"[train_v4_eval_sample] Saved progress: {len(processed_ids)}/{len(dataset)} problems complete"
            )
    finally:
        await llm_request.close()

    print(f"[train_v4_eval_sample] Saved evaluation dataset to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
