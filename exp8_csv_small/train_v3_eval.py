import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from seimei import load_run_messages, seimei

EXP_DIR = Path("exp8_csv_small")
DEFAULT_DATASET_PATH = EXP_DIR / "dataset.json"
DEFAULT_RESULT_PATH = EXP_DIR / "train_v3_eval_results.json"
DEFAULT_RM_URL = "https://j4s6oyznxb8j3v-8000.proxy.runpod.net/rmsearch"
DEFAULT_BATCH_SIZE = 5
DEFAULT_N_KNOWLEDGE_STEPS = 3
DEFAULT_KNOWLEDGE_PER_STEP = 3
DEFAULT_FINAL_RERUNS = 3

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

KNOWLEDGE_SYSTEM_PROMPT = (
    "You provide concise, reusable advice (1-3 short lines) that nudges the agent back onto a reliable "
    "reasoning path without giving away the final answer."
)

RUN_ID_CACHE: Dict[str, str] = {}


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
) -> Optional[Dict[str, Any]]:
    question = dataset_entry.get("Question", "")
    reference = dataset_entry.get("CorrectAnswer", "")
    transcript_json = json.dumps(messages, ensure_ascii=False, indent=2)
    step_text = get_agent_step_text(messages, step)
    prompt = (
        f"You are writing reusable knowledge that will be inserted before agent step {step} "
        "in a CSV reasoning workflow.\n\n"
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
        "- Write 1–3 short lines that describe the thinking path or small experiment, not the final answer.\n"
        "- Focus on what to calculate, compare, inspect, or log so the reasoning stays aligned.\n"
        "- original_text may reference concrete column names or entities from this transcript.\n"
        "- text must be abstract and reusable (use generic terms like 'target column' or 'input table').\n"
        "- Choose the agent whose skills progress this specific step; vary the choice beyond think when warranted.\n\n"
        "Agents and agent field:\n"
        "- The agent designates which SEIMEI agent executes this inserted step.\n"
        "- think — Plans the next actions by synthesizing prior findings and knowledge cues.\n"
        "- code_act — Runs small Python or shell commands (e.g., pandas snippets) to inspect or compute from the CSV.\n"
        "- web_search — Performs a quick web lookup to gather missing outside facts or clarifications.\n"
        "- answer — Summarizes gathered evidence into a final response when the solution is ready.\n\n"
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
    *,
    dataset_entry: Dict[str, Any],
    truncated_history: Sequence[Dict[str, Any]],
    step: int,
    knowledge_entry: Dict[str, Any],
    dataset_index: int,
    candidate_index: int,
) -> Dict[str, Any]:
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
    run_name = f"train_v3_eval_{dataset_index:04d}_s{step}_k{candidate_index + 1}"
    result = await orchestrator(
        messages=rerun_messages,
        run_name=run_name,
        knowledge_config=knowledge_config,
    )
    new_run_id = normalize_result_run_id(result, Path(orchestrator.log_dir))
    new_output = result.get("output", "")
    question = dataset_entry.get("Question", "")
    reference = dataset_entry.get("CorrectAnswer", "")
    score_info = await score_answer(orchestrator.llm, question, reference, new_output)
    new_score = score_info.get("score", 0.0) or 0.0

    candidate_info = {
        "candidate_index": candidate_index + 1,
        "run_id": new_run_id,
        "score": new_score,
        "score_feedback": score_info.get("feedback"),
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
    *,
    dataset_entry: Dict[str, Any],
    base_prompt_messages: Sequence[Dict[str, Any]],
    manual_entries: Optional[List[Dict[str, Any]]],
    trials: int,
    dataset_index: int,
    label: str,
) -> Tuple[List[Dict[str, Any]], float]:
    if trials <= 0:
        return [], 0.0
    question = dataset_entry.get("Question", "")
    reference = dataset_entry.get("CorrectAnswer", "")
    trial_records: List[Dict[str, Any]] = []
    for trial in range(trials):
        rerun_messages = randomize_system_prompt(base_prompt_messages)
        knowledge_config = build_knowledge_config(
            [dict(entry) for entry in manual_entries] if manual_entries else None
        )
        result = await orchestrator(
            messages=rerun_messages,
            run_name=f"train_v3_eval_{dataset_index:04d}_{label}_r{trial + 1}",
            knowledge_config=knowledge_config,
        )
        run_id = normalize_result_run_id(result, Path(orchestrator.log_dir))
        output = result.get("output", "")
        score_info = await score_answer(orchestrator.llm, question, reference, output)
        score = score_info.get("score", 0.0) or 0.0
        trial_records.append(
            {
                "trial": trial + 1,
                "run_id": run_id,
                "score": score,
                "score_feedback": score_info.get("feedback"),
                "output": output,
            }
        )
    mean_score = (
        round(sum(item["score"] for item in trial_records) / len(trial_records), 2)
        if trial_records
        else 0.0
    )
    return trial_records, mean_score


async def run_problem(
    orchestrator,
    dataset_entry: Dict[str, Any],
    index: int,
    entry_id: str,
    *,
    n_knowledge_steps: int,
    knowledge_per_step: int,
    final_reruns: int,
) -> Dict[str, Any]:
    question = dataset_entry.get("Question", "")
    csv_path = dataset_entry.get("CSVPath", "")
    reference_answer = dataset_entry.get("CorrectAnswer", "")

    base_prompt_messages = [
        {
            "role": "user",
            "content": f"Analyze inside {csv_path} and answer the question below:\n\n{question}",
        }
    ]
    base_messages = randomize_system_prompt(base_prompt_messages)

    base_result = await orchestrator(
        messages=[dict(msg) for msg in base_messages],
        run_name=f"train_v3_eval_{index:04d}_base",
        knowledge_config=build_knowledge_config(),
    )
    base_run_id = normalize_result_run_id(base_result, Path(orchestrator.log_dir))
    base_output = base_result.get("output", "")
    score_info = await score_answer(orchestrator.llm, question, reference_answer, base_output)
    base_score = score_info.get("score", 0.0) or 0.0

    record: Dict[str, Any] = {
        "id": entry_id,
        "csv_path": csv_path,
        "base": {
            "run_id": base_run_id,
            "score": base_score,
            "score_feedback": score_info.get("feedback"),
            "output": base_output,
        },
        "steps": [],
        "score_history": [
            {
                "step": 0,
                "score": base_score,
                "run_id": base_run_id,
            }
        ],
    }

    selected_manual_entries: List[Dict[str, Any]] = []
    best_result = base_result
    best_run_id = base_run_id
    best_score = base_score

    print(f"[eval {index}] baseline score={base_score}")

    for step in range(1, n_knowledge_steps + 1):
        messages = get_messages_for_run(best_result, Path(orchestrator.log_dir))
        agent_messages = extract_agent_messages(messages)
        if step > len(agent_messages):
            print(f"[eval {index}] step {step}: no agent messages, stopping.")
            break
        truncated_history = truncate_messages_before_step(messages, step)
        step_record: Dict[str, Any] = {
            "step": step,
            "starting_run_id": best_run_id,
            "starting_score": best_score,
            "candidate_evaluations": [],
            "best_index": None,
            "best_run_id": best_run_id,
            "best_score": best_score,
            "delta_from_start": 0.0,
        }
        record["steps"].append(step_record)

        step_best_idx: Optional[int] = None
        step_best_result: Optional[Dict[str, Any]] = None
        step_best_score: Optional[float] = None

        for candidate_idx in range(knowledge_per_step):
            knowledge_entry = await generate_step_knowledge(
                orchestrator.llm,
                dataset_entry=dataset_entry,
                messages=messages,
                step=step,
                iteration=candidate_idx,
            )
            if not knowledge_entry:
                print(f"[eval {index}] step {step} cand {candidate_idx + 1}: no knowledge generated")
                continue
            if not knowledge_entry.get("agent"):
                inferred_agent = get_agent_name_for_step(messages, step) or "think"
                knowledge_entry["agent"] = inferred_agent

            candidate = await run_candidate_inference(
                orchestrator,
                dataset_entry=dataset_entry,
                truncated_history=truncated_history,
                step=step,
                knowledge_entry=knowledge_entry,
                dataset_index=index,
                candidate_index=candidate_idx,
            )
            candidate_info = candidate["info"]
            step_record["candidate_evaluations"].append(candidate_info)
            candidate_score = candidate_info["score"]

            if step_best_score is None or candidate_score > step_best_score:
                step_best_score = candidate_score
                step_best_idx = len(step_record["candidate_evaluations"]) - 1
                step_best_result = candidate["result"]

        if step_best_idx is not None and step_best_result is not None:
            best_result = step_best_result
            best_run_id = step_record["candidate_evaluations"][step_best_idx]["run_id"]
            best_score = step_best_score if step_best_score is not None else best_score
            step_record["best_index"] = step_best_idx
            step_record["best_run_id"] = best_run_id
            step_record["best_score"] = best_score
            step_record["delta_from_start"] = round(best_score - step_record["starting_score"], 2)
            chosen_knowledge = step_record["candidate_evaluations"][step_best_idx].get("knowledge") or {}
            step_record["selected_knowledge"] = chosen_knowledge
            manual_entry = {
                "step": step,
                "agent": chosen_knowledge.get("agent") or "think",
                "text": chosen_knowledge.get("text", ""),
                "tags": chosen_knowledge.get("tags") or [],
            }
            if manual_entry["text"]:
                selected_manual_entries.append(manual_entry)
            print(
                f"[eval {index}] step {step}: best score={best_score} "
                f"(delta {step_record['delta_from_start']})"
            )
        else:
            print(f"[eval {index}] step {step}: no successful candidates")

        record["score_history"].append(
            {
                "step": step,
                "score": best_score,
                "run_id": best_run_id,
            }
        )

    record["final_run_id"] = best_run_id
    record["final_score"] = best_score
    record["final_output"] = best_result.get("output", "")
    record["final_single_run"] = {
        "run_id": best_run_id,
        "score": best_score,
        "output": record["final_output"],
    }
    record["steps_evaluated"] = len(record["steps"])
    record["selected_knowledge"] = [dict(entry) for entry in selected_manual_entries]

    base_trials, base_mean = await run_full_problem_trials(
        orchestrator,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=None,
        trials=final_reruns,
        dataset_index=index,
        label="base",
    )
    knowledge_trials, knowledge_mean = await run_full_problem_trials(
        orchestrator,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=selected_manual_entries if selected_manual_entries else None,
        trials=final_reruns,
        dataset_index=index,
        label="knowledge",
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
        f"[eval {index}] reruns avg base={base_mean} knowledge={knowledge_mean} "
        f"(Δ {round(knowledge_mean - base_mean, 2)})"
    )

    return record


def load_existing_eval_entries(
    dataset: List[Dict[str, Any]],
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Set[str], bool]:
    entries: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    needs_resave = False
    if output_path.exists():
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                entries = raw.get("detail") or []
            elif isinstance(raw, list):
                entries = raw
                needs_resave = True
            else:
                print(f"[train_v3_eval] Ignoring malformed cache at {output_path}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[train_v3_eval] Failed to load cached eval entries: {exc}")
        for idx, entry in enumerate(entries):
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id and idx < len(dataset):
                entry_id = compute_entry_id(dataset[idx], idx)
                entry["id"] = entry_id
                needs_resave = True
            if entry_id:
                processed_ids.add(entry_id)
    return entries, processed_ids, needs_resave


def build_eval_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {
            "total_problems": 0,
            "mean_score_improvement": 0.0,
            "base_vs_final": [],
            "overall_base_mean": 0.0,
            "overall_final_mean": 0.0,
            "win_loss_tie": {"win": 0, "tie": 0, "loss": 0},
        }
    base_vs_final: List[List[float]] = []
    improvements: List[float] = []
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
    }


def save_eval_entries(entries: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"summary": build_eval_summary(entries), "detail": entries},
        ensure_ascii=False,
        indent=2,
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(output_path)


async def run_evaluation(args: argparse.Namespace) -> None:
    dataset = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-5-nano"},
        rm_kwargs={"url": args.rm_url, "agent_routing": False, "knowledge_search": True},
        allow_code_exec=True,
        agent_log_head_lines=1,
        max_tokens_per_question=40000,
    )

    eval_entries, processed_ids, needs_resave = load_existing_eval_entries(dataset, args.output_path)
    if needs_resave:
        save_eval_entries(eval_entries, args.output_path)
    if processed_ids:
        print(f"[train_v3_eval] Resuming with {len(processed_ids)} cached entries")

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
            f"[train_v3_eval] All {len(processed_ids)} entries already processed. "
            f"Results stored at {args.output_path}"
        )
        return

    for batch_start in range(0, len(pending), args.batch_size):
        batch_slice = pending[batch_start : batch_start + args.batch_size]
        batch_tasks = [
            run_problem(
                orchestrator,
                entry,
                dataset_idx,
                entry_id,
                n_knowledge_steps=args.n_knowledge_steps,
                knowledge_per_step=args.knowledge_per_step,
                final_reruns=args.final_reruns,
            )
            for dataset_idx, entry, entry_id in batch_slice
        ]
        batch_records = await asyncio.gather(*batch_tasks)
        for record in batch_records:
            if not record:
                continue
            entry_id = record.get("id")
            if not entry_id or entry_id in processed_ids:
                continue
            eval_entries.append(record)
            processed_ids.add(entry_id)
        save_eval_entries(eval_entries, args.output_path)
        batch_idx = batch_start // args.batch_size + 1
        print(
            f"[train_v3_eval] Saved batch {batch_idx}: {len(processed_ids)}/{len(dataset)} problems complete"
        )

    print(f"[train_v3_eval] Saved evaluation dataset to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
