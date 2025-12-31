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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from seimei import load_run_messages, seimei
from seimei.editing import PatchApplyError, PatchParseError, apply_patch_to_workspace

# v3 eval sample notes:
# - Knowledge is generated per step per chunk from the run transcript (problem-specific + abstract advice).
# - Algorithm: run base chunks, then for each step generate knowledge per chunk, rerun with injection, and
#   finish with reruns comparing base vs knowledge-assisted runs.

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
PATCH_DIR = EXP_DIR / "patch_files"
DEFAULT_DATASET_PATH = EXP_DIR / "dataset.json"
DEFAULT_RESULT_PATH = EXP_DIR / "train_v3_eval_sample_results3.json"
DEFAULT_LLM_MODEL_NAME = "/workspace/gpt-oss-20b"
DEFAULT_LLM_URL = "https://v5710arnysphb8-8000.proxy.runpod.net/v1"  # Set None if you use openai model
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


def _sanitize_id_component(text: str, *, limit: int = 48) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_", " "}) else " " for ch in text)
    collapsed = "_".join(cleaned.split())
    return collapsed[:limit]


def extract_problem_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("Question") or "").strip()


def extract_reference_answer(entry: Dict[str, Any]) -> str:
    return str(entry.get("answer") or entry.get("CorrectAnswer") or "").strip()


def extract_expected_difference(entry: Dict[str, Any]) -> str:
    return str(entry.get("expected_simulation_result_difference") or "").strip()


def build_problem_prompt(problem: str, expected_difference: str) -> str:
    lines = [
        "You are debugging the gyrokinetic plasma simulation workspace. "
        "Edit the local Fortran sources to resolve the regression described below.",
    ]
    if problem:
        lines.append(f"Problem statement:\n{problem}")
    '''
    if expected_difference:
        lines.append(
            "Expected simulation behavior after applying a correct fix:\n"
            f"{expected_difference}"
        )
    '''
    lines.append(
        "Apply minimal, precise code edits to resolve the issue. Reference the exact files and routines you touch, "
        "and explain why the fix restores the intended physics before concluding."
    )
    return "\n\n".join(line for line in lines if line.strip())


def build_task_prompt(entry: Dict[str, Any]) -> str:
    problem = extract_problem_text(entry)
    if problem:
        return build_problem_prompt(problem, extract_expected_difference(entry))
    csv_path = str(entry.get("CSVPath") or "").strip()
    question = str(entry.get("Question") or "").strip()
    if csv_path and question:
        return f"Analyze inside {csv_path} and answer the question below:\n\n{question}"
    return question or problem


def format_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


class PatchWorkspaceManager:
    def __init__(self, workspace: Path, patch_dir: Path) -> None:
        self.workspace = workspace.resolve()
        self.patch_dir = patch_dir.resolve()

    def resolve_patch_path(self, dataset_entry: Dict[str, Any], dataset_index: int) -> Path:
        custom_field = str(
            dataset_entry.get("patch_file")
            or dataset_entry.get("patch_path")
            or ""
        ).strip()
        if custom_field:
            candidate = self._coerce_patch_path(Path(custom_field))
            if candidate:
                return candidate
            raise FileNotFoundError(f"Could not find patch file '{custom_field}' for sample {dataset_index}")

        default_name = f"patch{dataset_index}.txt"
        default_path = (self.patch_dir / default_name).resolve()
        if default_path.exists():
            return default_path
        raise FileNotFoundError(
            f"No patch file found for dataset index {dataset_index} under {self.patch_dir}"
        )

    def _coerce_patch_path(self, candidate: Path) -> Optional[Path]:
        search_paths: List[Path] = []
        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            search_paths.append(self.patch_dir / candidate)
            search_paths.append(self.workspace / candidate)
            search_paths.append(candidate)
        for option in search_paths:
            resolved = option.resolve()
            if resolved.exists():
                return resolved
        return None

    @contextmanager
    def apply_for_problem(self, dataset_entry: Dict[str, Any], dataset_index: int):
        patch_path = self.resolve_patch_path(dataset_entry, dataset_index)
        patch_text = patch_path.read_text(encoding="utf-8")
        touched_paths = self._collect_touched_paths(patch_text)
        backups = self._snapshot_files(touched_paths)
        try:
            apply_patch_to_workspace(patch_text, self.workspace)
        except (PatchApplyError, PatchParseError) as exc:
            rel_name = self._relative_patch_name(patch_path)
            raise RuntimeError(f"Failed to apply patch {rel_name}: {exc}") from exc
        try:
            yield
        finally:
            self._restore_files(backups)

    def _relative_patch_name(self, patch_path: Path) -> str:
        try:
            return patch_path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(patch_path)

    def _collect_touched_paths(self, patch_text: str) -> Iterable[str]:
        markers = (
            "*** Update File:",
            "*** Add File:",
            "*** Delete File:",
            "*** Move to:",
        )
        seen: Dict[str, None] = {}
        for raw_line in patch_text.splitlines():
            stripped = raw_line.strip()
            for marker in markers:
                if stripped.startswith(marker):
                    rel_path = stripped[len(marker) :].strip()
                    if rel_path and rel_path not in seen:
                        seen[rel_path] = None
                    break
        return tuple(seen.keys())

    def _snapshot_files(self, rel_paths: Iterable[str]) -> Dict[Path, Optional[bytes]]:
        backups: Dict[Path, Optional[bytes]] = {}
        for rel_path in rel_paths:
            absolute = self._resolve_within_workspace(rel_path)
            if absolute in backups:
                continue
            backups[absolute] = absolute.read_bytes() if absolute.exists() else None
        return backups

    def _resolve_within_workspace(self, rel_path: str) -> Path:
        candidate = (self.workspace / Path(rel_path)).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Patch references a path outside the workspace: {rel_path}") from exc
        return candidate

    def _restore_files(self, backups: Dict[Path, Optional[bytes]]) -> None:
        for path, content in backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


@dataclass(frozen=True)
class WorkspaceHandle:
    workspace: Path
    patch_manager: PatchWorkspaceManager
    orchestrator: Any


def _build_orchestrator(args: argparse.Namespace, workspace: Path):
    prior_cwd = Path.cwd()
    os.chdir(workspace)
    try:
        return seimei(
            # agent_config=[{"file_path": "seimei/agents/code_act.py"}],
            agent_config=[{"name": "code_act"}],
            #llm_kwargs={"model": "gpt-5-nano"},
            llm_kwargs={"base_url": args.llm_url, "model":args.llm_model_name},
            rm_kwargs={"url": args.rm_url, "agent_routing": False, "knowledge_search": True},
            allow_code_exec=True,
            agent_log_head_lines=1,
            max_tokens_per_question=80000,
        )
    finally:
        os.chdir(prior_cwd)


def _prepare_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    src_dest = workspace / "src"
    run_dest = workspace / "run"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    if run_dest.exists():
        shutil.rmtree(run_dest)
    shutil.copytree(REPO_ROOT / "src", src_dest)
    shutil.copytree(REPO_ROOT / "run", run_dest)


def build_workspace_pool(args: argparse.Namespace, count: int) -> List[WorkspaceHandle]:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    handles: List[WorkspaceHandle] = []
    for idx in range(count):
        workspace = WORKSPACE_ROOT / f"ws_{idx:02d}"
        _prepare_workspace(workspace)
        patch_manager = PatchWorkspaceManager(workspace, PATCH_DIR)
        orchestrator = _build_orchestrator(args, workspace)
        handles.append(WorkspaceHandle(workspace=workspace, patch_manager=patch_manager, orchestrator=orchestrator))
    return handles


async def run_orchestrator_with_patch(
    orchestrator,
    patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    dataset_index: int,
    messages: Sequence[Dict[str, Any]],
    run_name: str,
    knowledge_config: Dict[str, Any],
    checkpoint: Optional[EvalCheckpoint] = None,
    entry_id: Optional[str] = None,
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

    with patch_manager.apply_for_problem(dataset_entry, dataset_index):
        result = await orchestrator(
            messages=messages,
            run_name=run_name,
            knowledge_config=knowledge_config,
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
        "--llm-model-name",
        default=DEFAULT_LLM_MODEL_NAME,
        help="Name of model in LLM endpoint.",
    )
    parser.add_argument(
        "--llm-url",
        default=DEFAULT_LLM_URL,
        help="LLM endpoint passed to the orchestrator.",
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
        help="Number of problems to process concurrently.",
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


async def score_answer(orchestrator_llm, question: str, reference_answer: str, model_answer: str) -> Dict[str, Any]:
    prompt = (
        "Evaluate the model's answer.\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Model Answer:\n{model_answer}\n"
        "Respond strictly in JSON."
    )
    try:
        response, usage = await orchestrator_llm.chat(
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
) -> Optional[Dict[str, Any]]:
    question = build_task_prompt(dataset_entry)
    reference = extract_reference_answer(dataset_entry)
    transcript_json = json.dumps(messages, ensure_ascii=False, indent=2)
    step_text = get_agent_step_text(messages, step)
    knowledge_snapshot = [dict(entry) for entry in prior_knowledge or []]
    knowledge_json = json.dumps(knowledge_snapshot, ensure_ascii=False, indent=2)
    prompt = (
        f"You are writing reusable knowledge that will be inserted before agent step {step} "
        "in a code-debugging workflow for the gyrokinetic plasma repository.\n\n"
        "Evaluation summary for the run that produced this transcript:\n"
        "- Knowledge snippets already injected (JSON array, matches this transcript exactly):\n"
        f"{knowledge_json}\n\n"
        f"Full message history before rerun (JSON transcript):\n{transcript_json}\n\n"
        f"Original agent step {step} transcript:\n{step_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        "Your task:\n"
        "1. Decide the single best next reasoning move or tiny experiment the agent should run "
        "just before executing this step.\n"
        "2. Express this move twice:\n"
        "   - once as CONCRETE, problem-specific advice (original_text), and\n"
        "   - once as ABSTRACT, problem-agnostic advice (text) that can be reused.\n\n"
        "Guidelines:\n"
        "- Write 1–3 short lines that describe the debugging path or inspection, not the final answer.\n"
        "- Focus on what to diff, inspect, instrument, or edit inside the repo so the reasoning stays aligned.\n"
        "- original_text may reference concrete files, routines, arrays, or physics settings from this transcript.\n"
        "- text must be abstract and reusable (use generic terms like 'target subroutine' or 'field solve').\n"
        "- Choose the agent whose skills progress this specific step; vary the choice beyond think when warranted.\n\n"
        "Agents and agent field:\n"
        "- The agent designates which SEIMEI agent executes this inserted step.\n"
        "- think — Plans the next actions by synthesizing prior findings and knowledge cues.\n"
        "- code_act — Runs shell/Python commands (e.g., grep, sed, apply_patch) to inspect or modify the repo.\n"
        "- web_search — Performs a quick web lookup to gather missing outside facts or clarifications.\n"
        "- answer — Summarizes gathered evidence or describes the applied patch when the solution is ready.\n\n"
        "Output format (JSON only):\n"
        "[\n"
        "  {\n"
        "    \"agent\": \"think\" | \"code_act\" | \"web_search\" | \"answer\",\n"
        "    \"original_text\": \"concrete advice inserted before rerunning this step\",\n"
        "    \"text\": \"abstract, problem-agnostic advice derived from original_text\",\n"
        "    \"tags\": [\"importance\", \"topic\"]\n"
        "  }\n"
        "]\n\n"
        "Return only valid JSON. Do not include explanations outside the JSON."
    )
    try:
        response, _ = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=KNOWLEDGE_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[knowledge step {step} iter {iteration + 1}] generation failed: {exc}")
        return None

    candidates = _parse_json_array(response)
    if not candidates:
        return None
    entry = candidates[0]
    if not isinstance(entry, dict):
        return None
    text = str(entry.get("text") or entry.get("knowledge") or "").strip()
    if not text:
        return None
    entry.setdefault("step", step)
    entry.setdefault("agent", entry.get("agent") or "think")
    entry.setdefault("tags", entry.get("tags") or [])
    entry.setdefault("iteration", iteration + 1)
    return entry


async def run_candidate_inference(
    orchestrator,
    patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    truncated_history: Sequence[Dict[str, Any]],
    step: int,
    knowledge_entry: Dict[str, Any],
    dataset_index: int,
    candidate_index: int,
    checkpoint: Optional[EvalCheckpoint],
    entry_id: str,
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
    run_name = f"train_v3_eval_sample_{dataset_index:04d}_s{step}_k{candidate_index + 1}"
    result = await run_orchestrator_with_patch(
        orchestrator,
        patch_manager,
        dataset_entry=dataset_entry,
        dataset_index=dataset_index,
        messages=rerun_messages,
        run_name=run_name,
        knowledge_config=knowledge_config,
        checkpoint=checkpoint,
        entry_id=entry_id,
    )
    new_run_id = normalize_result_run_id(result, Path(orchestrator.log_dir))
    new_output = result.get("output", "")

    candidate_info = {
        "candidate_index": candidate_index + 1,
        "run_name": run_name,
        "run_id": new_run_id,
        "output": new_output,
        "knowledge": {
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
    patch_manager: PatchWorkspaceManager,
    *,
    dataset_entry: Dict[str, Any],
    base_prompt_messages: Sequence[Dict[str, Any]],
    manual_entries: Optional[List[Dict[str, Any]]],
    trials: int,
    dataset_index: int,
    label: str,
    checkpoint: Optional[EvalCheckpoint],
    entry_id: str,
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
        run_name = f"train_v3_eval_sample_{dataset_index:04d}_{label}_r{trial + 1}"
        result = await run_orchestrator_with_patch(
            orchestrator,
            patch_manager,
            dataset_entry=dataset_entry,
            dataset_index=dataset_index,
            messages=rerun_messages,
            run_name=run_name,
            knowledge_config=knowledge_config,
            checkpoint=checkpoint,
            entry_id=entry_id,
        )
        run_id = normalize_result_run_id(result, Path(orchestrator.log_dir))
        output = result.get("output", "")
        score_info = await score_answer(orchestrator.llm, question, reference, output)
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
    patch_manager: PatchWorkspaceManager,
    dataset_entry: Dict[str, Any],
    index: int,
    entry_id: str,
    *,
    n_knowledge_steps: int,
    knowledge_per_step: int,
    n_check_knowledge: int,
    final_reruns: int,
    checkpoint: Optional[EvalCheckpoint],
) -> Dict[str, Any]:
    problem_text = extract_problem_text(dataset_entry)
    prompt_text = build_task_prompt(dataset_entry) or problem_text
    if not prompt_text:
        prompt_text = "Investigate the regression and repair the affected code paths."
    reference_answer = extract_reference_answer(dataset_entry)
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    try:
        patch_path = patch_manager.resolve_patch_path(dataset_entry, index)
    except FileNotFoundError as exc:
        raise RuntimeError(f"[sample {index}] Missing patch file: {exc}") from exc

    base_prompt_messages = [{"role": "user", "content": prompt_text}]

    chunk_records: List[Dict[str, Any]] = []
    base_inferences: List[Dict[str, Any]] = []
    log_dir = Path(orchestrator.log_dir)

    for chunk_idx in range(knowledge_per_step):
        base_messages = randomize_system_prompt(base_prompt_messages)
        run_name = f"train_v3_eval_sample_{index:04d}_base_seed{chunk_idx + 1}"
        base_result = await run_orchestrator_with_patch(
            orchestrator,
            patch_manager,
            dataset_entry=dataset_entry,
            dataset_index=index,
            messages=[dict(msg) for msg in base_messages],
            run_name=run_name,
            knowledge_config=build_knowledge_config(),
            checkpoint=checkpoint,
            entry_id=entry_id,
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
        "patch_path": format_relative_path(patch_path),
        "base_inferences": base_inferences,
        "chunks": [],
        "knowledge_chunk_mean_scores": [],
    }

    for step in range(1, n_knowledge_steps + 1):
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
            )
            if not knowledge_entry:
                print(
                    f"[sample {index}] chunk {chunk_data['chunk_index']} step {step}: "
                    "no knowledge generated"
                )
                continue
            if not knowledge_entry.get("agent"):
                knowledge_entry["agent"] = get_agent_name_for_step(messages, step) or "think"

            candidate = await run_candidate_inference(
                orchestrator,
                patch_manager,
                dataset_entry=dataset_entry,
                truncated_history=truncated_history,
                step=step,
                knowledge_entry=knowledge_entry,
                dataset_index=index,
                candidate_index=chunk_data["chunk_index"] - 1,
                checkpoint=checkpoint,
                entry_id=entry_id,
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
            patch_manager,
            dataset_entry=dataset_entry,
            base_prompt_messages=base_prompt_messages,
            manual_entries=manual_entries if manual_entries else None,
            trials=n_check_knowledge,
            dataset_index=index,
            label=label,
            checkpoint=checkpoint,
            entry_id=entry_id,
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
        patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=None,
        trials=final_reruns,
        dataset_index=index,
        label="base_final",
        checkpoint=checkpoint,
        entry_id=entry_id,
    )
    knowledge_trials, knowledge_mean = await run_full_problem_trials(
        orchestrator,
        patch_manager,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=best_chunk_manual_entries if best_chunk_manual_entries else None,
        trials=final_reruns,
        dataset_index=index,
        label="knowledge_final",
        checkpoint=checkpoint,
        entry_id=entry_id,
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
                print(f"[train_v3_eval_sample] Ignoring malformed cache at {output_path}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[train_v3_eval_sample] Failed to load cached eval entries: {exc}")
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
    patch_files = sorted(PATCH_DIR.glob("*.txt"))
    if not patch_files:
        raise RuntimeError(f"No patch files found under {PATCH_DIR}")
    workspace_count = min(args.batch_size, len(patch_files))
    if workspace_count < 1:
        raise RuntimeError("batch_size must be >= 1 and patch_files must be non-empty.")
    if args.batch_size != workspace_count:
        print(
            "[train_v3_eval_sample] Limiting concurrency to "
            f"{workspace_count} based on patch file count."
        )
    workspace_pool = build_workspace_pool(args, workspace_count)

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
        print(f"[train_v3_eval_sample] Resuming with {len(processed_ids)} cached entries")
    if run_cache:
        print(f"[train_v3_eval_sample] Loaded run_cache with {len(run_cache)} runs")

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
            f"[train_v3_eval_sample] All {len(processed_ids)} entries already processed. "
            f"Results stored at {args.output_path}"
        )
        return

    for batch_start in range(0, len(pending), workspace_count):
        batch_slice = pending[batch_start : batch_start + workspace_count]
        batch_tasks = [
            run_problem(
                handle.orchestrator,
                handle.patch_manager,
                entry,
                dataset_idx,
                entry_id,
                n_knowledge_steps=args.n_knowledge_steps,
                knowledge_per_step=args.knowledge_per_step,
                n_check_knowledge=args.n_check_knowledge,
                final_reruns=args.final_reruns,
                checkpoint=checkpoint,
            )
            for handle, (dataset_idx, entry, entry_id) in zip(workspace_pool, batch_slice)
        ]
        batch_records = await asyncio.gather(*batch_tasks)
        for record in batch_records:
            if not record:
                continue
            rid = str(record.get("id") or "").strip()
            if not rid or rid in processed_ids:
                continue
            eval_entries.append(record)
            processed_ids.add(rid)

        # Keep existing behavior: save completed-problem detail after each batch,
        # but now we also persist run_cache continuously from inside run_orchestrator_with_patch.
        save_eval_entries(eval_entries, args.output_path, run_cache=checkpoint.run_cache)
        batch_idx = batch_start // args.batch_size + 1
        print(
            f"[train_v3_eval_sample] Saved batch {batch_idx}: {len(processed_ids)}/{len(dataset)} problems complete"
        )

    print(f"[train_v3_eval_sample] Saved evaluation dataset to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
