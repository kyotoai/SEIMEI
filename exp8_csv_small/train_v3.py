import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from seimei import load_run_messages, seimei

EXP_DIR = Path("exp8_csv_small")
DATASET_PATH = EXP_DIR / "dataset.json"
LOAD_KLG_PATH = None #Path("seimei_knowledge/exp5_train_v2_1.csv")
SAVE_KLG_PATH = Path("seimei_knowledge/exp8_train_v3_3.csv")
RESULT_OUTPUT_PATH = EXP_DIR / "train_v3_dpo_3.json"
GEN_STEP_PROMPT_PATH = Path("seimei/knowledge/prompts/gen_step.md")
MAX_KNOWLEDGE_ITERATIONS = 3
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
RM_URL = "https://j4s6oyznxb8j3v-8000.proxy.runpod.net/rmsearch"
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
KNOWLEDGE_SYSTEM_PROMPT = (
    "You provide concise, reusable advice (1-3 short lines) that nudges the agent back onto a reliable "
    "reasoning path without giving away the final answer."
)
BATCH_SIZE = 10

PROMPT_CACHE: Dict[Path, str] = {}
RUN_ID_CACHE: Dict[str, str] = {}


def compute_entry_id(dataset_entry: Dict[str, Any], index: int) -> str:
    """Derive a stable identifier for a dataset example."""
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


def normalize_result_run_id(result: Dict[str, Any], log_dir: Union[str, Path]) -> str:
    resolved = _resolve_run_dir_name(result.get("run_id"), log_dir)
    if resolved:
        result["run_id"] = resolved
    return str(result.get("run_id") or "")


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


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"generate_knowledge": False}
    load_str = str(LOAD_KLG_PATH)
    save_str = str(SAVE_KLG_PATH)
    if load_str:
        cfg["load_knowledge_path"] = load_str
    if save_str:
        cfg["save_knowledge_path"] = save_str
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


def append_best_knowledge(entry: Dict[str, Any]) -> None:
    text = entry.get("text")
    if not text:
        return
    SAVE_KLG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SAVE_KLG_PATH.exists()
    fieldnames = ["run_id", "agent", "knowledge", "tags", "step", "score", "source"]
    with SAVE_KLG_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": entry.get("run_id", ""),
                "agent": entry.get("agent", ""),
                "knowledge": text,
                "tags": json.dumps(entry.get("tags") or []),
                "step": entry.get("step"),
                "score": entry.get("score"),
                "source": "train_v2_best",
            }
        )


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
        print(f"[gen_step] failed: {exc}")
        return {}

    parsed = _parse_json_array(response)
    if not parsed:
        return {}
    candidate = parsed[0] if isinstance(parsed[0], dict) else {}
    return candidate


async def generate_knowledge_candidate(
    llm_client,
    *,
    dataset_entry: Dict[str, Any],
    best_result: Dict[str, Any],
    drift_info: Dict[str, Any],
    iteration: int,
    step: int,
    log_dir: str,
) -> Optional[Dict[str, Any]]:
    question = dataset_entry.get("Question", "")
    reference = dataset_entry.get("CorrectAnswer", "")
    csv_path = dataset_entry.get("CSVPath", "")
    messages = get_messages_for_run(best_result, log_dir)
    step_text = get_agent_step_text(messages, step)
    transcript_json = json.dumps(messages, ensure_ascii=False, indent=2)
    drift_summary = drift_info.get("drift_summary") or drift_info.get("reason") or ""
    missing = drift_info.get("missing_clues") or ""
    goal = drift_info.get("revision_goal") or ""
    best_output = best_result.get("output", "")

    prompt = (
        f"You are writing reusable knowledge that will replace agent step {step} "
        "in a CSV reasoning workflow.\n\n"
        f"Full message history before rerun (JSON transcript):\n{transcript_json}\n\n"
        f"Original agent step {step} transcript:\n{step_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        "Your task:\n"
        "1. Decide the single best next reasoning move or tiny experiment the agent should run\n"
        "   just before re-executing this step.\n"
        "2. Express this move twice:\n"
        "   - once as CONCRETE, problem-specific advice (original_text), and\n"
        "   - once as ABSTRACT, problem-agnostic advice (text) that can be reused.\n\n"
        "Guidelines:\n"
        "- Write 1–3 short lines that describe the thinking path or small experiment,\n"
        "  not the final numeric or categorical answer.\n"
        "- Focus on what to calculate, compare, inspect, or log so the reasoning stays aligned.\n"
        "- original_text:\n"
        "    * Direct, step-level instruction for THIS specific problem.\n"
        "    * May refer to concrete column names, values, files, or entities from the transcript.\n"
        "    * Example style: \"Filter rows where 'status' is 'completed' and recompute the mean of 'amount'.\"\n"
        "- text:\n"
        "    * Abstract version of the same move that is reusable for OTHER CSV reasoning tasks.\n"
        "    * MUST NOT contain any problem-specific identifiers (no concrete column names,\n"
        "      filenames, IDs, entities, or domain jargon unique to this transcript).\n"
        "    * Use only general terms like \"target column\", \"relevant rows\", \"input table\", etc.\n"
        "    * Example style: \"Filter the relevant subset of rows based on the task condition,\n"
        "      then recompute the key aggregate for the target column.\"\n\n"
        "Output format (JSON only):\n"
        "[\n"
        "  {\n"
        "    \"agent\": \"think\" | \"code_act\" | \"web_search\" | \"answer\",\n"
        "    \"original_text\": \"concrete advice inserted before rerunning this step\",\n"
        "    \"text\": \"abstract, problem-agnostic advice derived from original_text\",\n"
        "    \"tags\": [\"importance\", \"topic\"]\n"
        "  }\n"
        "]\n\n"
        "Return only valid JSON. Do not include any explanations or text outside the JSON.\n"
    )

    '''original
    prompt = (
        f"You must write reusable knowledge that will replace agent step {step}.\n"
        f"Full message history before rerun (JSON transcript):\n{transcript_json}\n\n"
        f"Original agent step {step} transcript:\n{step_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        "Write 1-3 short lines that describe the thinking path or tiny experiment the agent should run.\n"
        "Keep the advice general enough to reuse on similar CSV reasoning tasks (e.g., "
        "\"Write a short Python helper to compare the new and original CSV outputs\").\n"
        "Focus on the next reasoning move, not the final numeric answer.\n"
        "Return a JSON array with ONE object using this schema:\n"
        "[\n"
        "  {\n"
        "    \"agent\": \"think\" | \"code_act\" | \"web_search\" | \"answer\",\n"
        "    \"original_text\": \"concrete knowledge inserted before rerunning step\",\n"
        "    \"text\": \"abstact knowledge summarized from original text\",\n"
        "    \"tags\": [\"importance\", \"topic\"],\n"
        "  }\n"
        "]\n"
        "Text must describe calculations or evidence to keep the reasoning aligned."
    )
    '''
    
    try:
        response, _ = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=KNOWLEDGE_SYSTEM_PROMPT,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[knowledge_prompt] failed: {exc}")
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
    entry.setdefault("agent", drift_info.get("agent") or "think")
    entry.setdefault("tags", drift_info.get("tags") or [])
    entry.setdefault("step", step)
    entry.setdefault("rationale", entry.get("rationale") or drift_info.get("revision_goal") or "")
    return entry


async def check_knowledge(
    orchestrator,
    *,
    dataset_entry: Dict[str, Any],
    tracker: Dict[str, Any],
    knowledge_entry: Dict[str, Any],
    iteration: int,
) -> None:
    step = tracker["step"]
    best_result = tracker["best_result"]
    messages = get_messages_for_run(best_result, orchestrator.log_dir)
    truncated_history = truncate_messages_before_step(messages, step)
    rerun_messages = randomize_system_prompt(truncated_history)
    manual_entries = [
        {
            "step": step,
            "agent": knowledge_entry.get("agent") or "think",
            "text": knowledge_entry.get("text", ""),
            "tags": knowledge_entry.get("tags") or [],
        }
    ]
    knowledge_config = build_knowledge_config(manual_entries)
    run_name = f"train_v2_{dataset_entry.get('SampleIndex', 0)}_k{iteration + 1}"
    result = await orchestrator(
        messages=rerun_messages,
        run_name=run_name,
        knowledge_config=knowledge_config,
    )
    new_run_id = normalize_result_run_id(result, orchestrator.log_dir)
    new_output = result.get("output", "")
    question = dataset_entry.get("Question", "")
    reference = dataset_entry.get("CorrectAnswer", "")
    score_info = await score_answer(orchestrator.llm, question, reference, new_output)
    new_score = score_info.get("score", 0.0) or 0.0

    knowledge_record = {
        "text": knowledge_entry.get("text"),
        "agent": knowledge_entry.get("agent"),
        "tags": knowledge_entry.get("tags") or [],
        "step": step,
        "iteration": iteration + 1,
        "run_id": new_run_id,
        "rationale": knowledge_entry.get("rationale"),
        "score_feedback": score_info.get("feedback"),
        "score": new_score,
    }

    tracker["run_ids"].append(new_run_id)
    tracker["knowledge"].append(knowledge_record)
    tracker["scores"].append(new_score)
    tracker.setdefault("outputs", []).append(new_output)

    best_idx = tracker.get("best_index", 0)
    best_score = tracker["scores"][best_idx]

    if new_score > best_score:
        tracker["comparison"].append([len(tracker["knowledge"]) - 1, best_idx])
        tracker["best_index"] = len(tracker["knowledge"]) - 1
        tracker["best_result"] = result
    elif new_score < best_score:
        tracker["comparison"].append([best_idx, len(tracker["knowledge"]) - 1])
    else:
        pass


async def run_problem(
    orchestrator, dataset_entry: Dict[str, Any], index: int, entry_id: str
) -> Dict[str, Any]:
    question = dataset_entry.get("Question", "")
    csv_path = dataset_entry.get("CSVPath", "")
    reference_answer = dataset_entry.get("CorrectAnswer", "")

    base_messages = randomize_system_prompt(
        [
            {
                "role": "user",
                "content": f"Analyze inside {csv_path} and answer the question below:\n\n{question}",
            }
        ]
    )

    base_result = await orchestrator(
        messages=[dict(msg) for msg in base_messages],
        run_name=f"train_v2_{index:04d}_base",
        knowledge_config=build_knowledge_config(),
    )
    base_run_id = normalize_result_run_id(base_result, orchestrator.log_dir)
    base_output = base_result.get("output", "")
    score_info = await score_answer(orchestrator.llm, question, reference_answer, base_output)
    base_score = score_info.get("score", 0.0) or 0.0

    drift_info = await select_drifted_step(
        orchestrator.llm,
        question=question,
        reference_answer=reference_answer,
        csv_path=csv_path,
        run_result=base_result,
        log_dir=orchestrator.log_dir,
    )
    step = int(drift_info.get("step") or 1)
    step = max(step, 1)
    dpo_messages = get_messages_for_run(
        base_result,
        orchestrator.log_dir,
        max_agent_step=step,
    )
    if not dpo_messages:
        dpo_messages = [dict(msg) for msg in base_messages]

    tracker: Dict[str, Any] = {
        "id": entry_id,
        "run_ids": [base_run_id],
        "step": step,
        "message": dpo_messages,
        "knowledge": [
            {
                "text": None,
                "agent": None,
                "tags": [],
                "step": step,
                "iteration": 0,
                "run_id": base_run_id,
                "rationale": "baseline inference",
                "score_feedback": score_info.get("feedback"),
                "score": base_score,
            }
        ],
        "scores": [base_score],
        "comparison": [],
        "best_index": 0,
        "best_result": base_result,
        "outputs": [base_output],
    }

    print(f"[problem {index}] step={step} baseline score={base_score}")

    for iteration in range(MAX_KNOWLEDGE_ITERATIONS):
        candidate = await generate_knowledge_candidate(
            orchestrator.llm,
            dataset_entry=dataset_entry,
            best_result=tracker["best_result"],
            drift_info=drift_info,
            iteration=iteration,
            step=step,
            log_dir=orchestrator.log_dir,
        )
        if not candidate:
            print(f"[problem {index}] iteration {iteration + 1}: no candidate knowledge")
            break
        await check_knowledge(
            orchestrator,
            dataset_entry=dataset_entry,
            tracker=tracker,
            knowledge_entry=candidate,
            iteration=iteration,
        )
        best_idx = tracker.get("best_index", 0)
        best_score = tracker["scores"][best_idx]
        print(
            f"[problem {index}] iteration {iteration + 1}: score={tracker['scores'][-1]} best={best_score}"
        )

    best_idx = tracker.get("best_index", 0)
    best_entry = tracker["knowledge"][best_idx]
    best_entry["run_id"] = tracker["run_ids"][best_idx]
    best_entry["score"] = tracker["scores"][best_idx]
    append_best_knowledge(best_entry)

    return tracker


def load_existing_dpo_entries(
    dataset: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Set[str], bool]:
    entries: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    needs_resave = False
    if RESULT_OUTPUT_PATH.exists():
        try:
            raw = json.loads(RESULT_OUTPUT_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                entries = raw
            else:
                print(f"[train_v3] Ignoring malformed DPO cache at {RESULT_OUTPUT_PATH}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[train_v3] Failed to load cached DPO entries: {exc}")
        for idx, entry in enumerate(entries):
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id and idx < len(dataset):
                entry_id = compute_entry_id(dataset[idx], idx)
                entry["id"] = entry_id
                needs_resave = True
            if entry_id:
                processed_ids.add(entry_id)
    return entries, processed_ids, needs_resave


def save_dpo_entries(entries: List[Dict[str, Any]]) -> None:
    RESULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, ensure_ascii=False, indent=2)
    tmp_path = RESULT_OUTPUT_PATH.with_suffix(RESULT_OUTPUT_PATH.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(RESULT_OUTPUT_PATH)


async def run_training() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-5-nano"},
        rm_kwargs={"url": RM_URL, "agent_routing": False, "knowledge_search": True},
        allow_code_exec=True,
        agent_log_head_lines=1,
        max_tokens_per_question=40000,
    )

    dpo_entries, processed_ids, needs_resave = load_existing_dpo_entries(dataset)
    if needs_resave:
        save_dpo_entries(dpo_entries)
    if processed_ids:
        print(f"[train_v3] Resuming with {len(processed_ids)} cached entries")

    total = len(dataset)
    pending: List[Tuple[int, Dict[str, Any], str]] = []
    for idx, entry in enumerate(dataset):
        entry_id = compute_entry_id(entry, idx)
        if entry_id in processed_ids:
            continue
        pending.append((idx, entry, entry_id))

    if not pending:
        print(
            f"[train_v3] All {len(processed_ids)} entries already processed. "
            f"Results stored at {RESULT_OUTPUT_PATH}"
        )
        return

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch_slice = pending[batch_start : batch_start + BATCH_SIZE]
        batch_tasks = [
            run_problem(orchestrator, entry, dataset_idx, entry_id)
            for dataset_idx, entry, entry_id in batch_slice
        ]
        batch_trackers = await asyncio.gather(*batch_tasks)
        for tracker in batch_trackers:
            if not tracker:
                continue
            entry_id = tracker.get("id")
            if not entry_id or entry_id in processed_ids:
                continue
            dpo_entries.append(
                {
                    "id": entry_id,
                    "run_ids": tracker["run_ids"],
                    "step": tracker["step"],
                    "message": tracker["message"],
                    "knowledge": tracker["knowledge"],
                    "comparison": tracker["comparison"],
                    "scores": tracker["scores"],
                }
            )
            processed_ids.add(entry_id)
        save_dpo_entries(dpo_entries)
        batch_idx = batch_start // BATCH_SIZE + 1
        print(
            f"[train_v3] Saved batch {batch_idx}: {len(processed_ids)}/{total} problems complete"
        )

    print(f"[train_v3] Saved DPO dataset to {RESULT_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_training())
