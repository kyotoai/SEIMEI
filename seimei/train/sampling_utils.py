"""Helper utilities for train-time sampling/inference + scoring."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from seimei import load_run_messages
from seimei.editing import PatchApplyError, PatchParseError, apply_patch_to_workspace

from .sampling_prompts import (
    BASE_SYSTEM_PROMPT_LIST,
    KLG_SYSTEM_PROMPT_LIST,
    KNOWLEDGE_UPDATE_PROMPT_INSTRUCTION,
    KNOWLEDGE_UPDATE_PROMPT_RESPONSE_FORMAT,
    PROBLEM_PROMPT_PREFIX,
    PROBLEM_PROMPT_SUFFIX,
    SCORING_PROMPT_INSTRUCTION,
    SCORING_PROMPT_RESPONSE_FORMAT,
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_json_size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return 10**9


def compact_msg_history(history: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(history, list) or not history:
        return None
    if len(history) > 160:
        return None
    if _safe_json_size(history) > 250_000:
        return None
    compact: List[Dict[str, Any]] = []
    for msg in history:
        if isinstance(msg, dict):
            compact.append(dict(msg))
    return compact or None


class Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._streams[0], name)


def setup_log_output(*, save_log: bool, log_dir: Optional[Path], prefix: str):
    should_log = bool(save_log) or not sys.stdout.isatty()
    if not should_log:
        return None
    resolved_dir = log_dir or (Path(__file__).resolve().parent.parent.parent / "exp11_plasma_gkv_v5" / "_logs")
    resolved_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = resolved_dir / f"{prefix}_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    print(f"[{prefix}] Logging to {log_path}")
    return (log_file, original_stdout, original_stderr)


def close_log_output(state: Any) -> None:
    if not state:
        return
    log_file, original_stdout, original_stderr = state
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def normalize_pool_ids(pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(pool, start=1):
        new_entry = dict(entry)
        raw_id = entry.get("id")
        try:
            new_id = int(raw_id)
        except (TypeError, ValueError):
            new_id = idx
        new_entry["id"] = new_id
        normalized.append(new_entry)
    return normalized


def load_knowledge_pool_csv(path: Path) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    if not path.exists():
        return pool
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("knowledge") or row.get("text") or "").strip()
                agent = str(row.get("agent") or "").strip()
                if not text or not agent:
                    continue
                tags_raw = row.get("tags")
                tags: List[str] = []
                if tags_raw not in (None, ""):
                    try:
                        parsed_tags = json.loads(tags_raw)
                        if isinstance(parsed_tags, list):
                            tags = [str(tag) for tag in parsed_tags]
                        elif isinstance(parsed_tags, str):
                            tags = [parsed_tags]
                    except json.JSONDecodeError:
                        tags = [tag.strip() for tag in str(tags_raw).split(",") if tag.strip()]
                step_value = row.get("step")
                step: Optional[int] = None
                if step_value not in (None, ""):
                    try:
                        step = int(step_value)
                    except (TypeError, ValueError):
                        step = None
                entry: Dict[str, Any] = {
                    "id": row.get("id"),
                    "agent": agent,
                    "text": text,
                    "tags": tags,
                }
                if step is not None:
                    entry["step"] = step
                pool.append(entry)
    except OSError as exc:
        print(f"[sampling] Failed to load knowledge pool from {path}: {exc}")
        return []
    return pool


def write_knowledge_pool_csv(
    path: Path,
    pool: Sequence[Dict[str, Any]],
    *,
    force: bool = True,
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


def apply_knowledge_updates(
    pool: List[Dict[str, Any]],
    updates: Sequence[Dict[str, Any]],
) -> List[str]:
    updated_ids: List[str] = []
    id_map = {str(entry.get("id") or "").strip(): entry for entry in pool}
    for update in updates:
        if not isinstance(update, dict):
            continue
        update_id = str(update.get("id") or "").strip()
        new_text = str(update.get("text") or "").strip()
        if not update_id or not new_text:
            continue
        target = id_map.get(update_id)
        if not target:
            continue
        if target.get("text") == new_text:
            continue
        target["text"] = new_text
        updated_ids.append(update_id)
    return updated_ids


def extract_problem_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("Question") or "").strip()


def extract_reference_answer(entry: Dict[str, Any]) -> str:
    return str(entry.get("answer") or entry.get("CorrectAnswer") or "").strip()


def extract_expected_difference(entry: Dict[str, Any]) -> str:
    return str(entry.get("expected_simulation_result_difference") or "").strip()


def build_problem_prompt(problem: str, expected_difference: str) -> str:
    lines = [PROBLEM_PROMPT_PREFIX]
    if problem:
        lines.append(f"Problem statement:\n{problem}")
    _ = expected_difference
    lines.append(PROBLEM_PROMPT_SUFFIX)
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


def format_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
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
                    rel_path = stripped[len(marker):].strip()
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
        except ValueError as exc:
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


def prepare_workspace(workspace: Path, repo_root: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    src_dest = workspace / "src"
    run_dest = workspace / "run"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    if run_dest.exists():
        shutil.rmtree(run_dest)
    shutil.copytree(repo_root / "src", src_dest)
    shutil.copytree(repo_root / "run", run_dest)


def _sanitize_id_component(text: str, *, limit: int = 48) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_", " "}) else " " for ch in text)
    collapsed = "_".join(cleaned.split())
    return collapsed[:limit]


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
    pool = KLG_SYSTEM_PROMPT_LIST if use_knowledge_prompt else BASE_SYSTEM_PROMPT_LIST
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


def strip_code_fences(text: str) -> str:
    snippet = text.strip()
    if snippet.startswith("```"):
        lines = snippet.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        snippet = "\n".join(lines).strip()
    return snippet


def parse_json_response(raw: str) -> Dict[str, Any]:
    cleaned = strip_code_fences(raw)
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            candidates.append(cleaned[start: end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def parse_json_array(raw: str) -> List[Any]:
    cleaned = strip_code_fences(raw)
    candidates = [cleaned]
    if "[" in cleaned and "]" in cleaned:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if 0 <= start < end:
            candidates.append(cleaned[start: end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    return []


def coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 10:
        return 10.0
    return round(score, 2)


def summarize_scores(scores: Sequence[float]) -> Tuple[float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0
    mean_score = round(sum(scores) / len(scores), 2)
    max_score = round(max(scores), 2)
    min_score = round(min(scores), 2)
    return mean_score, max_score, min_score


def is_bugged_score(score: float, feedback: Optional[str]) -> bool:
    if score != 0.0:
        return False
    if feedback is None:
        return True
    if isinstance(feedback, str) and not feedback.strip():
        return True
    return False


def resolve_run_dir_name(
    run_id: Optional[str],
    log_dir: Path,
    run_id_cache: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    run_id_str = str(run_id or "").strip()
    if not run_id_str:
        return None
    cache = run_id_cache if run_id_cache is not None else {}
    cached = cache.get(run_id_str)
    if cached:
        return cached
    if not log_dir.exists():
        return run_id_str
    direct_candidate = log_dir / run_id_str
    if direct_candidate.exists():
        cache[run_id_str] = run_id_str
        return run_id_str
    short = run_id_str[:8]
    if short:
        matches = sorted(log_dir.glob(f"run-*-{short}"))
        if matches:
            resolved = matches[-1].name
            cache[run_id_str] = resolved
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
            cache[run_id_str] = resolved
            return resolved
    return run_id_str


def normalize_result_run_id(
    result: Dict[str, Any],
    log_dir: Path,
    run_id_cache: Optional[Dict[str, str]] = None,
) -> str:
    resolved = resolve_run_dir_name(result.get("run_id"), Path(log_dir), run_id_cache)
    if resolved:
        result["run_id"] = resolved
    return str(result.get("run_id") or "")


@dataclass
class EvalCheckpoint:
    output_path: Path
    eval_entries: List[Dict[str, Any]]
    run_cache: Dict[str, Dict[str, Any]]
    lock: Any
    run_id_cache: Dict[str, str]
    schema_version: int

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
        normalize_result_run_id(result, Path(orchestrator_log_dir), self.run_id_cache)
        payload: Dict[str, Any] = {
            "entry_id": entry_id,
            "dataset_index": int(dataset_index),
            "run_name": run_name,
            "run_id": str(result.get("run_id") or "").strip(),
            "output": result.get("output", ""),
            "saved_at": _now_iso(),
        }
        msg_history = compact_msg_history(result.get("msg_history"))
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
            save_eval_entries(
                self.eval_entries,
                self.output_path,
                run_cache=self.run_cache,
                schema_version=self.schema_version,
            )


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


def format_reasoning_history(
    messages: Sequence[Dict[str, Any]],
    knowledge_entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    agent_msgs = extract_agent_messages(messages)
    knowledge_by_step: Dict[int, List[str]] = {}
    for entry in knowledge_entries or []:
        if not isinstance(entry, dict):
            continue
        step_raw = entry.get("step")
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        text = str(entry.get("text") or "").strip()
        entry_id = str(entry.get("id") or "").strip()
        if not text:
            continue
        labeled = f"{entry_id}: {text}" if entry_id else text
        knowledge_by_step.setdefault(step, []).append(labeled)

    parts = ["<REASONING HISTORY>"]
    for idx, msg in enumerate(agent_msgs, start=1):
        agent_output = str(msg.get("content") or "")
        knowledge_lines = knowledge_by_step.get(idx, [])
        knowledge_text = "\n".join(knowledge_lines)
        parts.append(f"<STEP {idx}>")
        parts.append("<AGENT OUTPUT>")
        parts.append(agent_output)
        parts.append("</AGENT OUTPUT>")
        parts.append("<USED KNOWLEDGE>")
        parts.append(knowledge_text)
        parts.append("</USED KNOWLEDGE>")
        parts.append(f"</STEP {idx}>")
    parts.append("</REASONING HISTORY>")
    return "\n".join(parts)


def build_scoring_prompt(
    *,
    question: str,
    patch: str,
    reference_answer: str,
    model_answer: str,
    reasoning_history: str,
) -> str:
    return (
        f"{SCORING_PROMPT_INSTRUCTION}\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"<DELETED ORIGINAL CODE>\n{patch}\n</DELETED ORIGINAL CODE>\n\n"
        f"<REFERENCE ANSWER>\n{reference_answer}\n</REFERENCE ANSWER>\n\n"
        f"<MODEL ANSWER>\n{model_answer}\n</MODEL ANSWER>\n\n"
        f"{reasoning_history}\n\n"
        f"{SCORING_PROMPT_RESPONSE_FORMAT}"
    )


def build_knowledge_update_prompt(
    *,
    question: str,
    patch: str,
    reference_answer: str,
    reasoning_history: str,
) -> str:
    return (
        f"{KNOWLEDGE_UPDATE_PROMPT_INSTRUCTION}\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"<DELETED ORIGINAL CODE>\n{patch}\n</DELETED ORIGINAL CODE>\n\n"
        f"<REFERENCE ANSWER>\n{reference_answer}\n</REFERENCE ANSWER>\n\n"
        f"{reasoning_history}\n\n"
        f"{KNOWLEDGE_UPDATE_PROMPT_RESPONSE_FORMAT}"
    )


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"generate_knowledge": False}
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


def build_manual_knowledge_entries(pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for entry in pool:
        text = str(entry.get("text") or "").strip()
        agent = str(entry.get("agent") or "").strip()
        if not text or not agent:
            continue
        payload: Dict[str, Any] = {
            "id": entry.get("id"),
            "agent": agent,
            "text": text,
            "tags": entry.get("tags") or [],
        }
        step = entry.get("step")
        if step not in (None, ""):
            payload["step"] = step
        entries.append(payload)
    return entries


def build_knowledge_search_config(
    *,
    top_n_sample_klg: int,
    distribution_decay_rate: float,
    random_klg_sampling_rate: float,
    klg_sample_mode: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "mode": str(klg_sample_mode or "rm").strip().lower(),
            "step": ">=1",
            "topk": 1,
            "sampling_topk": max(int(top_n_sample_klg), 1),
            "sampling_distribution_decay_rate": float(distribution_decay_rate),
            "random_sampling_rate": float(random_klg_sampling_rate),
        }
    ]


def extract_knowledge_entries_from_messages(
    messages: Sequence[Dict[str, Any]],
    orchestrator: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    agent_messages = extract_agent_messages(messages)
    for step_idx, msg in enumerate(agent_messages, start=1):
        raw = msg.get("knowledge")
        items: List[Tuple[str, Any]] = []
        if orchestrator is not None and hasattr(orchestrator, "_coerce_knowledge_items_with_ids"):
            items = orchestrator._coerce_knowledge_items_with_ids(raw)
            if hasattr(orchestrator, "_dedupe_knowledge_items"):
                items = orchestrator._dedupe_knowledge_items(items)
        else:
            def _coerce_local(value: Any) -> List[Tuple[str, Any]]:
                local_items: List[Tuple[str, Any]] = []
                if value is None:
                    return local_items
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        local_items.append((text, None))
                    return local_items
                if isinstance(value, dict):
                    text_value = (
                        value.get("text")
                        or value.get("knowledge")
                        or value.get("content")
                        or value.get("value")
                    )
                    text = str(text_value or "").strip()
                    kid = value.get("id") or value.get("knowledge_id")
                    if text:
                        local_items.append((text, kid))
                    return local_items
                if isinstance(value, (list, tuple, set)):
                    for entry in value:
                        local_items.extend(_coerce_local(entry))
                    return local_items
                try:
                    text = str(value).strip()
                except Exception:
                    return local_items
                if text:
                    local_items.append((text, None))
                return local_items

            items = _coerce_local(raw)

        if not items:
            continue
        if msg.get("knowledge_id") not in (None, "") and all(kid is None for _, kid in items):
            kid_value = msg.get("knowledge_id")
            if isinstance(kid_value, (list, tuple, set)):
                ids_list = list(kid_value)
            else:
                ids_list = [kid_value]
            if len(ids_list) == len(items):
                items = [(text, ids_list[idx]) for idx, (text, _) in enumerate(items)]
            elif len(ids_list) == 1:
                items = [(text, ids_list[0]) for text, _ in items]

        for text, kid in items:
            entry = {"step": step_idx, "text": text}
            if kid is not None:
                entry["id"] = kid
            entries.append(entry)
    return entries


def split_knowledge_args(knowledge_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not knowledge_config:
        return {}
    load_config: List[Dict[str, Any]] = []
    load_path = knowledge_config.get("load_knowledge_path")
    load_steps = knowledge_config.get("load_knowledge_steps")
    if load_path:
        entry: Dict[str, Any] = {"load_knowledge_path": load_path}
        if load_steps:
            entry["step"] = load_steps
        load_config.append(entry)
    manual_entries = knowledge_config.get("knowledge")
    if manual_entries:
        if isinstance(manual_entries, list):
            for item in manual_entries:
                if isinstance(item, dict):
                    load_config.append(dict(item))
                elif isinstance(item, str):
                    load_config.append({"text": item})
        elif isinstance(manual_entries, dict):
            load_config.append(dict(manual_entries))
        elif isinstance(manual_entries, str):
            load_config.append({"text": manual_entries})
    generate_config = None
    if (
        knowledge_config.get("generate_knowledge")
        or knowledge_config.get("save_knowledge_path")
        or knowledge_config.get("knowledge_prompt_path")
    ):
        generate_config = {}
        save_path = knowledge_config.get("save_knowledge_path")
        if save_path:
            generate_config["save_knowledge_path"] = save_path
        prompt_path = knowledge_config.get("knowledge_prompt_path")
        if prompt_path:
            generate_config["knowledge_generation_prompt_path"] = prompt_path
    payload: Dict[str, Any] = {}
    if load_config:
        payload["knowledge_load_config"] = load_config
    if generate_config:
        payload["knowledge_generate_config"] = generate_config
    return payload


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
        return []
    return _normalize(raw_messages)


def parse_knowledge_updates(raw: str) -> List[Dict[str, Any]]:
    parsed = parse_json_response(raw)
    updates = parsed.get("updates")
    if isinstance(updates, list):
        return [dict(item) for item in updates if isinstance(item, dict)]
    array = parse_json_array(raw)
    return [dict(item) for item in array if isinstance(item, dict)]


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
                    run_cache = {str(k): v for k, v in rc.items() if isinstance(v, dict)}
            elif isinstance(raw, list):
                entries = raw
                needs_resave = True
            else:
                print(f"[sampling] Ignoring malformed cache at {output_path}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[sampling] Failed to load cached eval entries: {exc}")
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
    empty_summary = {
        "total_problems": 0,
        "overall_mean_score_improvement": 0.0,
        "max_mean_score_improvement": 0.0,
        "min_mean_score_improvement": 0.0,
        "no_klg_overall_mean": 0.0,
        "klg_overall_mean": 0.0,
        "no_klg_max_mean": 0.0,
        "klg_max_mean": 0.0,
        "no_klg_min_mean": 0.0,
        "klg_min_mean": 0.0,
        "mean_win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        "max_mean_win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        "min_mean_win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        "mean_no_klg_vs_mean_klg": [],
        "max_no_klg_vs_max_klg": [],
        "min_no_klg_vs_min_klg": [],
    }
    if not entries:
        return empty_summary

    mean_pairs: List[List[float]] = []
    max_pairs: List[List[float]] = []
    min_pairs: List[List[float]] = []
    mean_improvements: List[float] = []
    max_improvements: List[float] = []
    min_improvements: List[float] = []

    for entry in entries:
        no_mean = float(entry.get("no_klg_mean_score") or 0.0)
        klg_mean = float(entry.get("klg_mean_score") or 0.0)
        no_max = float(entry.get("no_klg_max_score") or 0.0)
        klg_max = float(entry.get("klg_max_score") or 0.0)
        no_min = float(entry.get("no_klg_min_score") or 0.0)
        klg_min = float(entry.get("klg_min_score") or 0.0)

        mean_pairs.append([round(no_mean, 2), round(klg_mean, 2)])
        max_pairs.append([round(no_max, 2), round(klg_max, 2)])
        min_pairs.append([round(no_min, 2), round(klg_min, 2)])
        mean_improvements.append(klg_mean - no_mean)
        max_improvements.append(klg_max - no_max)
        min_improvements.append(klg_min - no_min)

    total = len(entries)

    def _count_wlt(pairs: Sequence[Sequence[float]]) -> Dict[str, int]:
        wins = sum(1 for no, klg in pairs if klg > no)
        ties = sum(1 for no, klg in pairs if klg == no)
        losses = total - wins - ties
        return {"win": wins, "tie": ties, "loss": losses}

    no_klg_overall_mean = round(sum(pair[0] for pair in mean_pairs) / total, 2)
    klg_overall_mean = round(sum(pair[1] for pair in mean_pairs) / total, 2)
    no_klg_max_mean = round(sum(pair[0] for pair in max_pairs) / total, 2)
    klg_max_mean = round(sum(pair[1] for pair in max_pairs) / total, 2)
    no_klg_min_mean = round(sum(pair[0] for pair in min_pairs) / total, 2)
    klg_min_mean = round(sum(pair[1] for pair in min_pairs) / total, 2)

    return {
        "total_problems": total,
        "overall_mean_score_improvement": round(sum(mean_improvements) / total, 2),
        "max_mean_score_improvement": round(sum(max_improvements) / total, 2),
        "min_mean_score_improvement": round(sum(min_improvements) / total, 2),
        "no_klg_overall_mean": no_klg_overall_mean,
        "klg_overall_mean": klg_overall_mean,
        "no_klg_max_mean": no_klg_max_mean,
        "klg_max_mean": klg_max_mean,
        "no_klg_min_mean": no_klg_min_mean,
        "klg_min_mean": klg_min_mean,
        "mean_win_loss_tie": _count_wlt(mean_pairs),
        "max_mean_win_loss_tie": _count_wlt(max_pairs),
        "min_mean_win_loss_tie": _count_wlt(min_pairs),
        "mean_no_klg_vs_mean_klg": mean_pairs,
        "max_no_klg_vs_max_klg": max_pairs,
        "min_no_klg_vs_min_klg": min_pairs,
    }


def save_eval_entries(
    entries: List[Dict[str, Any]],
    output_path: Path,
    *,
    run_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    schema_version: int = 2,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if os.getenv("SEIMEI_DEBUG_EVAL_SAVE"):
        print(f"[sampling] save_eval_entries: {len(entries)} entries -> {output_path}")
    payload_obj: Dict[str, Any] = {
        "schema_version": schema_version,
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
