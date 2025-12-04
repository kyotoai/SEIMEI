import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from seimei import load_run_messages, seimei
from seimei.utils import format_query_for_rmsearch

EXP_DIR = Path("exp8_csv_small")
DEFAULT_DATASET_PATH = EXP_DIR / "dataset.json"
DEFAULT_TRACKER_PATH = EXP_DIR / "train_v3_dpo.json"
DEFAULT_RESULT_PATH = EXP_DIR / "eval_results.json"
GEN_STEP_PROMPT_PATH = Path("seimei/knowledge/prompts/gen_step.md")
DEFAULT_RM_URL = "https://j4s6oyznxb8j3v-8000.proxy.runpod.net/rmsearch"

BASE_SYSTEM_PROMPT_LIST = [
    "Think like an investigative data analyst: read the CSV, form a plan of 2-3 steps, "
    "and cite the evidence before answering.",
    "Adopt a precise scientist mindset—identify helpful columns, run quick calculations, "
    "and only conclude after verifying trends.",
    "Work as a spreadsheet detective who double-checks every assumption with the CSV before "
    "writing a short, justified answer.",
    "Be a pragmatic Python user: outline a minimal approach, inspect the CSV, and respond with "
    "clear reasoning anchored in computations.",
    "Channel a skeptical reviewer—question each step, confirm numbers with the CSV, and explain "
    "why the final answer follows.",
    "Operate like a project lead: summarize the task, gather CSV evidence, and deliver the answer "
    "with a brief audit trail.",
    "Act as a careful statistician: describe the metrics you need, compute them succinctly, "
    "and interpret them before answering.",
    "Think as an automation engineer who prototypes tiny helpers, inspects their output, "
    "and keeps narration crisp.",
    "Take the role of a mentor guiding a junior analyst—state the strategy, run compact checks, "
    "and highlight the decisive facts.",
    "Imagine you are debugging an analysis: surface hypotheses, validate them with the CSV, "
    "and provide the confident conclusion.",
]

SCORING_SYSTEM_PROMPT = (
    "You are an impartial evaluator scoring an assistant's answer against a reference answer. "
    "Judge factual accuracy, coverage, and clarity. Return ONLY a JSON object with keys 'score' "
    "(integer 0-10) and 'feedback' (concise justification). Score 0 means entirely incorrect, "
    "10 means fully correct."
)

STEP_SELECTION_SYSTEM_PROMPT = (
    "You are a transcript analyst who pinpoints the exact agent step that drifted away from the correct solution. "
    "Return ONLY the JSON array requested by the prompt."
)

PROMPT_CACHE: Dict[Path, str] = {}
RUN_ID_CACHE: Dict[str, str] = {}


def compute_entry_id(dataset_entry: Dict[str, Any], index: int) -> str:
    explicit_keys = ["id", "ID", "question_id", "QuestionID"]
    for key in explicit_keys:
        value = dataset_entry.get(key)
        if value:
            return str(value)
    csv_path = str(dataset_entry.get("CSVPath") or "").strip()
    if csv_path:
        return csv_path
    topic = str(dataset_entry.get("Topic") or "").strip()
    hyper = dataset_entry.get("HyperParamIndex")
    sample = dataset_entry.get("SampleIndex")
    parts = [f"idx_{index:05d}"]
    if topic:
        parts.append(topic)
    if hyper not in (None, ""):
        parts.append(f"hp_{hyper}")
    if sample not in (None, ""):
        parts.append(f"sample_{sample}")
    return "|".join(parts)


def pick_base_system_prompt() -> str:
    return random.choice(BASE_SYSTEM_PROMPT_LIST)


def randomize_system_prompt(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloned = [dict(msg) for msg in messages]
    prompt = pick_base_system_prompt()
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


def load_prompt_text(path: Path) -> str:
    if path not in PROMPT_CACHE:
        PROMPT_CACHE[path] = path.read_text(encoding="utf-8")
    return PROMPT_CACHE[path]


def format_run_context(
    *,
    question: str,
    reference_answer: str,
    csv_path: str,
    run_id: Optional[str],
    messages: Sequence[Dict[str, Any]],
    assistant_answer: str,
) -> str:
    convo = json.dumps(list(messages), ensure_ascii=False, indent=2)
    parts = [
        f"Run ID: {run_id or 'N/A'}",
        f"Question:\n{question}",
        f"Reference answer:\n{reference_answer}",
        f"CSV path: {csv_path}",
        f"Assistant answer:\n{assistant_answer}",
        f"Transcript (JSON):\n{convo}",
    ]
    return "\n\n".join(parts)


def extract_agent_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [msg for msg in messages if msg.get("role") == "agent"]


def get_agent_step_text(messages: Sequence[Dict[str, Any]], step: int) -> str:
    agent_msgs = extract_agent_messages(messages)
    if step < 1 or step > len(agent_msgs):
        return ""
    content = agent_msgs[step - 1].get("content", "")
    return str(content)


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


def build_manual_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"generate_knowledge": False}
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


def get_messages_for_run(
    run_result: Dict[str, Any],
    log_dir: str,
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
    except Exception as exc:  # pragma: no cover
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


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Dataset file must contain a list: {path}")
    return [dict(entry) for entry in payload]


def load_tracker_entries(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Tracker file must contain a list: {path}")
    return [dict(entry) for entry in payload]


def normalize_result_run_id(result: Dict[str, Any], log_dir: Union[str, Path]) -> str:
    resolved = _resolve_run_dir_name(result.get("run_id"), log_dir)
    if resolved:
        result["run_id"] = resolved
    return str(result.get("run_id") or "")


def _resolve_run_dir_name(run_id: Optional[str], log_dir: Union[str, Path]) -> Optional[str]:
    run_id_str = str(run_id or "").strip()
    if not run_id_str:
        return None
    cached = RUN_ID_CACHE.get(run_id_str)
    if cached:
        return cached
    log_path = Path(log_dir)
    if not log_path.exists():
        return run_id_str
    direct_candidate = log_path / run_id_str
    if direct_candidate.exists():
        RUN_ID_CACHE[run_id_str] = run_id_str
        return run_id_str
    short = run_id_str[:8]
    if short:
        matches = sorted(log_path.glob(f"run-*-{short}"))
        if matches:
            resolved = matches[-1].name
            RUN_ID_CACHE[run_id_str] = resolved
            return resolved
    for candidate in log_path.glob("run-*"):
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


async def select_drifted_step(
    llm_client,
    *,
    question: str,
    reference_answer: str,
    csv_path: str,
    run_result: Dict[str, Any],
    log_dir: str,
) -> Dict[str, Any]:
    messages = get_messages_for_run(run_result, log_dir)
    run_context = format_run_context(
        question=question,
        reference_answer=reference_answer,
        csv_path=csv_path,
        run_id=run_result.get("run_id"),
        messages=messages,
        assistant_answer=run_result.get("output", ""),
    )
    prompt_template = load_prompt_text(GEN_STEP_PROMPT_PATH)
    prompt = prompt_template.replace("<<RUN_CONTEXT>>", run_context)
    try:
        response, _ = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=STEP_SELECTION_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[eval] select_drifted_step failed: {exc}")
        return {}

    parsed = _parse_json_array(response)
    if not parsed:
        return {}
    candidate = parsed[0] if isinstance(parsed[0], dict) else {}
    return candidate


def _float_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def choose_best_knowledge_entry(entries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or entry.get("knowledge") or "").strip()
        if not text:
            continue
        score = _float_score(entry.get("score"))
        if best is None or score > best_score:
            best = entry
            best_score = score
    if best is None:
        return None
    normalized = dict(best)
    normalized["text"] = str(normalized.get("text") or normalized.get("knowledge") or "").strip()
    normalized["agent"] = str(normalized.get("agent") or "think").strip() or "think"
    normalized["tags"] = normalized.get("tags") or []
    return normalized


def build_knowledge_pool(
    tracker_entries: Sequence[Dict[str, Any]],
    excluded_ids: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    excluded = set(str(entry_id) for entry_id in excluded_ids)
    pools: Dict[str, List[Dict[str, Any]]] = {}
    counter = 0
    for entry in tracker_entries:
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id or entry_id in excluded:
            continue
        best = choose_best_knowledge_entry(entry.get("knowledge") or [])
        if not best:
            continue
        counter += 1
        agent = best.get("agent") or "think"
        payload = {
            "id": counter,
            "text": best["text"],
            "agent": agent,
            "tags": best.get("tags") or [],
            "score": _float_score(best.get("score")),
            "source_entry_id": entry_id,
            "source_run_id": best.get("run_id"),
            "step": best.get("step"),
        }
        pools.setdefault(agent, []).append(payload)
    return pools


def _format_run_name(prefix: str, dataset_idx: int, sample_idx: int, suffix: str) -> str:
    return f"{prefix}_{dataset_idx:04d}_s{sample_idx:02d}_{suffix}"


async def run_baseline_inference(
    orchestrator,
    *,
    dataset_entry: Dict[str, Any],
    dataset_idx: int,
    sample_idx: int,
) -> Dict[str, Any]:
    question = dataset_entry.get("Question", "")
    csv_path = dataset_entry.get("CSVPath", "")
    base_messages = randomize_system_prompt(
        [
            {
                "role": "user",
                "content": f"Analyze inside {csv_path} and answer the question below:\n\n{question}",
            }
        ]
    )
    run_name = _format_run_name("eval", dataset_idx, sample_idx, "base")
    result = await orchestrator(
        messages=[dict(msg) for msg in base_messages],
        run_name=run_name,
        knowledge_config=build_manual_knowledge_config(),
    )
    normalize_result_run_id(result, orchestrator.log_dir)
    return result


def _build_rmsearch_query(orchestrator, messages: Sequence[Dict[str, Any]]) -> str:
    if hasattr(orchestrator, "_build_knowledge_query_block"):
        return orchestrator._build_knowledge_query_block(messages)
    normalized = seimei._coerce_messages(messages)
    _, conversation_text, focus_text = seimei._prepare_query_input(normalized)
    body = (focus_text or conversation_text or "").strip()
    return format_query_for_rmsearch(body or "Relevant knowledge for evaluation")


def select_knowledge_candidates(
    orchestrator,
    query_messages: Sequence[Dict[str, Any]],
    agent_name: str,
    knowledge_pool: Dict[str, List[Dict[str, Any]]],
    limit: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    limit = max(int(limit), 1)
    pool = list(knowledge_pool.get(agent_name) or [])
    if not pool:
        pool = list(knowledge_pool.get("*") or [])
    query = _build_rmsearch_query(orchestrator, query_messages)
    if not pool:
        return query, []

    keys: List[Dict[str, Any]] = []
    for entry in pool:
        text = entry.get("text")
        if not text:
            continue
        keys.append(
            {
                "key": text,
                "knowledge": entry,
                "knowledge_id": entry.get("id"),
                "tags": entry.get("tags") or [],
            }
        )
    ranked: List[Dict[str, Any]] = []
    try:
        rm_results = orchestrator._rmsearch(
            query=query,
            keys=keys,
            k_key=min(limit, len(keys)),
            purpose="knowledge_eval",
        )
    except Exception as exc:
        print(f"[eval] rmsearch failed for agent '{agent_name}': {exc}")
        rm_results = []
    seen_ids: set = set()
    if rm_results:
        for item in rm_results:
            payload = item.get("payload") or {}
            knowledge_entry = payload.get("knowledge") or payload
            if not isinstance(knowledge_entry, dict):
                continue
            entry_id = knowledge_entry.get("id")
            if entry_id in seen_ids:
                continue
            ranked.append(dict(knowledge_entry))
            seen_ids.add(entry_id)
            if len(ranked) >= limit:
                break
    if not ranked:
        sorted_pool = sorted(pool, key=lambda x: x.get("score", 0.0), reverse=True)
        ranked = [dict(entry) for entry in sorted_pool[:limit]]
    return query, ranked


async def run_knowledge_rerun(
    orchestrator,
    *,
    dataset_entry: Dict[str, Any],
    dataset_idx: int,
    sample_idx: int,
    base_result: Dict[str, Any],
    drift_info: Dict[str, Any],
    knowledge_entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not knowledge_entries:
        return None
    step = int(drift_info.get("step") or 1)
    step = max(step, 1)
    messages = get_messages_for_run(base_result, orchestrator.log_dir)
    if not messages:
        return None
    truncated_history = truncate_messages_before_step(messages, step)
    rerun_messages = randomize_system_prompt(truncated_history)
    manual_entries = [
        {
            "step": step,
            "agent": entry.get("agent") or drift_info.get("agent") or "think",
            "text": entry.get("text", ""),
            "tags": entry.get("tags") or [],
        }
        for entry in knowledge_entries
        if entry.get("text")
    ]
    if not manual_entries:
        return None
    run_name = _format_run_name("eval", dataset_idx, sample_idx, "klg")
    result = await orchestrator(
        messages=rerun_messages,
        run_name=run_name,
        knowledge_config=build_manual_knowledge_config(manual_entries),
    )
    normalize_result_run_id(result, orchestrator.log_dir)
    return result


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [v for v in values if v is not None]
    return round(statistics.fmean(items), 3) if items else None


def compute_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_scores: List[float] = []
    knowledge_scores: List[float] = []
    deltas: List[float] = []
    for record in results:
        for sample in record.get("samples", []):
            base_score = sample.get("baseline_score")
            knowledge_score = sample.get("knowledge_score")
            if base_score is not None:
                baseline_scores.append(base_score)
            if knowledge_score is not None:
                knowledge_scores.append(knowledge_score)
            if base_score is not None and knowledge_score is not None:
                deltas.append(knowledge_score - base_score)
    return {
        "n_problems": len(results),
        "baseline_runs": len(baseline_scores),
        "knowledge_runs": len(knowledge_scores),
        "baseline_mean": _mean(baseline_scores),
        "knowledge_mean": _mean(knowledge_scores),
        "delta_mean": _mean(deltas),
    }


def save_eval_results(
    *,
    results: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    payload = {
        "config": config,
        "summary": summary,
        "problems": list(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


async def evaluate_problem(
    orchestrator,
    *,
    dataset_entry: Dict[str, Any],
    dataset_idx: int,
    entry_id: str,
    args: argparse.Namespace,
    knowledge_pool: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    question = dataset_entry.get("Question", "")
    reference = dataset_entry.get("CorrectAnswer", "")
    csv_path = dataset_entry.get("CSVPath", "")
    samples: List[Dict[str, Any]] = []
    for sample_idx in range(args.n_sample):
        base_result = await run_baseline_inference(
            orchestrator,
            dataset_entry=dataset_entry,
            dataset_idx=dataset_idx,
            sample_idx=sample_idx,
        )
        base_output = base_result.get("output", "")
        base_score_info = await score_answer(orchestrator.llm, question, reference, base_output)
        base_score = base_score_info.get("score", 0.0) or 0.0
        drift_info = await select_drifted_step(
            orchestrator.llm,
            question=question,
            reference_answer=reference,
            csv_path=csv_path,
            run_result=base_result,
            log_dir=orchestrator.log_dir,
        )
        step = int(drift_info.get("step") or 1)
        drift_agent = str(drift_info.get("agent") or "think").strip() or "think"
        messages = get_messages_for_run(base_result, orchestrator.log_dir, max_agent_step=step)
        truncated_history = truncate_messages_before_step(messages, step)
        query_text, ranked_entries = select_knowledge_candidates(
            orchestrator,
            truncated_history,
            drift_agent,
            knowledge_pool,
            args.n_knowledge,
        )
        knowledge_result = None
        knowledge_score = None
        knowledge_feedback = ""
        selected_entries: List[Dict[str, Any]] = []
        if ranked_entries:
            knowledge_result = await run_knowledge_rerun(
                orchestrator,
                dataset_entry=dataset_entry,
                dataset_idx=dataset_idx,
                sample_idx=sample_idx,
                base_result=base_result,
                drift_info=drift_info,
                knowledge_entries=ranked_entries,
            )
            if knowledge_result:
                knowledge_output = knowledge_result.get("output", "")
                knowledge_score_info = await score_answer(
                    orchestrator.llm, question, reference, knowledge_output
                )
                knowledge_score = knowledge_score_info.get("score", 0.0) or 0.0
                knowledge_feedback = knowledge_score_info.get("feedback", "")
                selected_entries = [
                    {
                        "id": entry.get("id"),
                        "agent": entry.get("agent"),
                        "text": entry.get("text"),
                        "tags": entry.get("tags", []),
                        "source_entry_id": entry.get("source_entry_id"),
                        "score": entry.get("score"),
                    }
                    for entry in ranked_entries
                ]
        sample_record = {
            "sample_index": sample_idx,
            "baseline_score": base_score,
            "baseline_feedback": base_score_info.get("feedback", ""),
            "baseline_run_id": base_result.get("run_id"),
            "knowledge_score": knowledge_score,
            "knowledge_feedback": knowledge_feedback,
            "knowledge_run_id": knowledge_result.get("run_id") if knowledge_result else None,
            "drift_step": step,
            "drift_agent": drift_agent,
            "query_text": query_text,
            "knowledge_entries": selected_entries,
        }
        samples.append(sample_record)
        print(
            f"[eval] problem {entry_id} sample {sample_idx + 1}/{args.n_sample}: "
            f"base={base_score} knowledge={knowledge_score if knowledge_score is not None else 'N/A'}"
        )
    return {
        "id": entry_id,
        "dataset_index": dataset_idx,
        "csv_path": dataset_entry.get("CSVPath"),
        "question": question,
        "samples": samples,
    }


def select_test_entries(
    dataset: Sequence[Dict[str, Any]],
    tracker_entries: Sequence[Dict[str, Any]],
    *,
    n_problems: int,
    seed: Optional[int],
) -> List[Tuple[int, Dict[str, Any], str]]:
    tracker_ids = {
        str(entry.get("id") or "").strip(): entry for entry in tracker_entries if entry.get("id")
    }
    indexed: List[Tuple[int, Dict[str, Any], str]] = []
    for idx, entry in enumerate(dataset):
        entry_id = compute_entry_id(entry, idx)
        if entry_id in tracker_ids:
            indexed.append((idx, entry, entry_id))
    if len(indexed) < n_problems:
        raise ValueError(
            f"Requested {n_problems} problems but only {len(indexed)} have tracker entries."
        )
    rng = random.Random(seed)
    rng.shuffle(indexed)
    return indexed[:n_problems]


async def run_evaluation(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path or DEFAULT_DATASET_PATH)
    tracker_path = Path(args.tracker_path or DEFAULT_TRACKER_PATH)
    output_path = Path(args.output_path or DEFAULT_RESULT_PATH)

    dataset = load_dataset(dataset_path)
    tracker_entries = load_tracker_entries(tracker_path)
    selected_entries = select_test_entries(
        dataset,
        tracker_entries,
        n_problems=args.n_problems,
        seed=args.seed,
    )
    selected_ids = [entry_id for _, _, entry_id in selected_entries]
    knowledge_pool = build_knowledge_pool(tracker_entries, selected_ids)
    if not any(knowledge_pool.values()):
        raise RuntimeError("Knowledge pool is empty after excluding test problems.")

    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": args.model},
        rm_kwargs={
            "url": args.rm_url or DEFAULT_RM_URL,
            "agent_routing": False,
            "knowledge_search": True,
        },
        allow_code_exec=True,
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        agent_log_head_lines=args.agent_log_head_lines,
        max_tokens_per_question=args.max_tokens_per_question,
    )

    results: List[Dict[str, Any]] = []
    config_snapshot = {
        "dataset_path": str(dataset_path),
        "tracker_path": str(tracker_path),
        "output_path": str(output_path),
        "rm_url": args.rm_url or DEFAULT_RM_URL,
        "n_problems": args.n_problems,
        "n_sample": args.n_sample,
        "n_knowledge": args.n_knowledge,
        "model": args.model,
        "log_dir": args.log_dir,
    }

    for batch_start in range(0, len(selected_entries), args.batch_size):
        batch_slice = selected_entries[batch_start : batch_start + args.batch_size]
        batch_tasks = [
            evaluate_problem(
                orchestrator,
                dataset_entry=entry,
                dataset_idx=idx,
                entry_id=entry_id,
                args=args,
                knowledge_pool=knowledge_pool,
            )
            for idx, entry, entry_id in batch_slice
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        for result in batch_results:
            if result:
                results.append(result)
        summary = compute_summary(results)
        save_eval_results(results=results, summary=summary, config=config_snapshot, output_path=output_path)
        batch_idx = batch_start // args.batch_size + 1
        print(
            f"[eval] Saved batch {batch_idx}: {len(results)}/{len(selected_entries)} problems complete "
            f"(baseline mean={summary.get('baseline_mean')}, knowledge mean={summary.get('knowledge_mean')})"
        )

    final_summary = compute_summary(results)
    save_eval_results(results=results, summary=final_summary, config=config_snapshot, output_path=output_path)
    print("[eval] Completed evaluation")
    print(
        f"[eval] Baseline mean={final_summary.get('baseline_mean')} "
        f"Knowledge mean={final_summary.get('knowledge_mean')} "
        f"Delta={final_summary.get('delta_mean')}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate knowledge retrieval vs baseline runs.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--tracker-path", default=str(DEFAULT_TRACKER_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--rm-url", default=DEFAULT_RM_URL)
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--log-dir", default="seimei_runs")
    parser.add_argument("--agent-log-head-lines", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens-per-question", type=int, default=40000)
    parser.add_argument("--n-problems", type=int, default=5)
    parser.add_argument("--n-knowledge", type=int, default=2)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
