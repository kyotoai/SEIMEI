import argparse
import asyncio
import json
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from seimei import seimei

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = EXP_DIR / "dataset.json"
DEFAULT_RESULT_PATH = EXP_DIR / "train_v4_eval_sample_results.json"
DEFAULT_RM_URL = "https://j4s6oyznxb8j3v-8000.proxy.runpod.net/rmsearch"
DEFAULT_BATCH_SIZE = 6
DEFAULT_N_KNOWLEDGE_STEPS = 2
DEFAULT_KNOWLEDGE_PER_STEP = 3
DEFAULT_N_CHECK_KNOWLEDGE = 3
DEFAULT_FINAL_RERUNS = 5

ROUTING_ENTRIES = [
    {"step": 1, "agent": "web_search"},
    {"step": 2, "agent": "answer"},
]
STEP_AGENT_HINT = {1: "web_search", 2: "answer"}

BASE_SYSTEM_PROMPT_LIST = [
    "Act as a meticulous web researcher: run multiple targeted searches, capture URLs, and synthesize facts before answering.",
    "Be a skeptical analyst--cross-check sources, extract concrete numbers, and justify the conclusion with citations.",
    "Work like an investigative reporter: list the queries you need, verify dates, and anchor the answer to sources.",
    "Think as a policy researcher: compare at least two credible sources and highlight the key numeric differences.",
    "Operate as a data-driven fact checker: validate statistics across sources and mention each URL explicitly.",
    "Be a concise web strategist: refine queries, summarize findings, and respond with a short evidence-backed comparison.",
    "Act like a literature scout: search broadly, then narrow to authoritative sources and extract the exact figures.",
    "Behave as a cautious researcher: note publication dates, reconcile discrepancies, and report the most defensible numbers.",
    "Think as a briefing writer: gather multiple sources, list the facts, and conclude with a clear takeaway.",
    "Work like a forensic web analyst: inspect sources, pull exact metrics, and explain the final answer in plain terms.",
]

KLG_SYSTEM_PROMPT_LIST = [
    "Treat the knowledge snippets as mandatory search heuristics: apply each one explicitly and cite where it changed your query.",
    "Use the injected knowledge as your checklist: translate each cue into a concrete search or extraction step.",
    "Anchor your reasoning to the knowledge hints by naming the specific source or data point they push you toward.",
    "Follow the knowledge text as marching orders: update the query, collect the cited facts, and report the evidence.",
    "Weave the knowledge into your plan: echo the cue, then show how it shaped the sources and the final answer.",
    "Operate as a strict auditor: each knowledge hint must be reflected in the searches or source selection you perform.",
    "Treat the knowledge as a search playbook--apply it first, then document how it affected the sources you used.",
    "Act as a disciplined researcher: for every knowledge line, show the source or metric it helped you locate.",
    "Let the knowledge guide the query formulation and the evidence list before writing the conclusion.",
    "Use the knowledge to prioritize sources and explicitly connect it to the extracted numbers.",
]

DEFAULT_KNOWLEDGE_POOL: List[Dict[str, Any]] = [
    {
        "id": "ws_query_split",
        "agent": "web_search",
        "step": None,
        "text": "Draft at least two distinct search queries with different keywords or regions before choosing the best results.",
        "tags": ["query", "coverage"],
    },
    {
        "id": "ws_authoritative_sources",
        "agent": "web_search",
        "step": None,
        "text": "Prioritize authoritative sources (gov, edu, official reports) when extracting numeric facts.",
        "tags": ["sources", "credibility"],
    },
    {
        "id": "ws_date_check",
        "agent": "web_search",
        "step": None,
        "text": "Capture the publication date for each source and prefer the most recent data where possible.",
        "tags": ["dates", "freshness"],
    },
    {
        "id": "ws_site_filter",
        "agent": "web_search",
        "step": None,
        "text": "Use site: or filetype:pdf filters when searching for official statistics or reports.",
        "tags": ["query", "filters"],
    },
    {
        "id": "ws_cross_check",
        "agent": "web_search",
        "step": None,
        "text": "Cross-check key numbers across at least two sources and note discrepancies explicitly.",
        "tags": ["verification", "numbers"],
    },
    {
        "id": "ws_units",
        "agent": "web_search",
        "step": None,
        "text": "Ensure every extracted number includes units and the relevant year or date.",
        "tags": ["numbers", "units"],
    },
    {
        "id": "ws_source_logging",
        "agent": "web_search",
        "step": None,
        "text": "Log the exact URLs used for each numeric claim so they can be cited directly in the answer.",
        "tags": ["citations", "urls"],
    },
    {
        "id": "ws_scope_control",
        "agent": "web_search",
        "step": None,
        "text": "If the topic is broad, narrow it to a defined region or timeframe and state that scope in the answer.",
        "tags": ["scope", "clarity"],
    },
    {
        "id": "ans_citation_list",
        "agent": "answer",
        "step": None,
        "text": "List sources as bullet points with URLs and tie each numeric claim to a cited source.",
        "tags": ["answer", "citations"],
    },
    {
        "id": "ans_comparison",
        "agent": "answer",
        "step": None,
        "text": "Include a short comparison that explains why one source or figure is more reliable or recent.",
        "tags": ["answer", "comparison"],
    },
    {
        "id": "ans_structure",
        "agent": "answer",
        "step": None,
        "text": "Structure the final answer with a brief summary, then a source list with URLs, then the comparison.",
        "tags": ["answer", "structure"],
    },
]

SCORING_SYSTEM_PROMPT = (
    "You are an impartial evaluator scoring an assistant's answer against a scoring rubric. "
    "Use the rubric requirements and their point values to assign a total score (0-10). "
    "Return ONLY a JSON object with keys 'score' (integer 0-10) and 'feedback' (concise justification)."
)

KNOWLEDGE_SELECTION_SYSTEM_PROMPT = (
    "You select the most helpful reusable knowledge snippet from the pool to inject before the "
    "next agent action. Respond only with JSON."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge injection on web-search questions."
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


def extract_question(entry: Dict[str, Any]) -> str:
    return str(entry.get("question") or entry.get("Question") or "").strip()


def extract_answer_scoring(entry: Dict[str, Any]) -> Any:
    return entry.get("answer_scoring") or entry.get("AnswerScoring") or []


def build_task_prompt(question: str) -> str:
    if not question:
        return "Use web search to answer the question and cite sources."
    return (
        "Use web search to answer the question below and cite sources with URLs.\n\n"
        f"{question}"
    )


def compute_entry_id(dataset_entry: Dict[str, Any], index: int) -> str:
    explicit_keys = ["id", "ID", "question_id", "QuestionID"]
    for key in explicit_keys:
        value = dataset_entry.get(key)
        if value:
            return str(value)
    question = extract_question(dataset_entry)
    if question:
        return question[:80]
    return f"idx_{index:05d}"


def pick_system_prompt(*, use_knowledge_prompt: bool = False) -> str:
    pool = (
        KLG_SYSTEM_PROMPT_LIST
        if use_knowledge_prompt and KLG_SYSTEM_PROMPT_LIST
        else BASE_SYSTEM_PROMPT_LIST
    )
    return random.choice(pool)


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
        "tags": entry.get("tags") or [],
    }
    normalized["_step_filter"] = _normalize_step_filter(entry.get("step"))
    return normalized


def _format_pool_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": entry.get("id"),
            "agent": entry.get("agent"),
            "text": entry.get("text"),
            "tags": entry.get("tags") or [],
        }
        for entry in candidates
    ]


def _prepare_pool_candidates(step: int, used_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    used = used_ids or set()
    applicable: List[Dict[str, Any]] = []
    allowed_agent = STEP_AGENT_HINT.get(step)
    for idx, raw_entry in enumerate(DEFAULT_KNOWLEDGE_POOL):
        normalized = _normalize_pool_entry(raw_entry, idx)
        if not normalized:
            continue
        step_filter = normalized.pop("_step_filter", None)
        if step_filter is not None and step not in step_filter:
            continue
        if allowed_agent and normalized.get("agent") != allowed_agent:
            continue
        if normalized["id"] in used:
            continue
        applicable.append(normalized)
    return applicable


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

    truncated: List[Dict[str, Any]] = []
    agent_seen = 0
    for msg in messages:
        if msg.get("role") == "agent":
            agent_seen += 1
            if agent_seen >= step:
                break
        truncated.append(dict(msg))
    return truncated


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = [dict(item) for item in ROUTING_ENTRIES]
    if manual_entries:
        entries.extend(dict(entry) for entry in manual_entries)
    return {"generate_knowledge": False, "knowledge": entries}


def get_messages_for_run(
    run_result: Dict[str, Any],
    *,
    max_agent_step: Optional[int] = None,
) -> List[Dict[str, Any]]:
    history = run_result.get("msg_history")
    if not isinstance(history, list) or not history:
        return []
    filtered: List[Dict[str, Any]] = []
    agent_seen = 0
    for msg in history:
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


def format_scoring_rubric(answer_scoring: Any) -> str:
    if isinstance(answer_scoring, (list, dict)):
        return json.dumps(answer_scoring, ensure_ascii=False, indent=2)
    return str(answer_scoring)


async def score_answer(
    orchestrator_llm,
    question: str,
    answer_scoring: Any,
    model_answer: str,
) -> Dict[str, Any]:
    rubric_text = format_scoring_rubric(answer_scoring)
    prompt = textwrap.dedent(
        f"""
        Question:
        {question}

        Scoring rubric (JSON):
        {rubric_text}

        Model Answer:
        {model_answer}

        Score the answer using the rubric and return JSON only.
        """
    ).strip()
    try:
        response, usage = await orchestrator_llm.chat(
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

    parsed = _parse_json_response(response)
    score_value = _coerce_score(parsed.get("score"))
    feedback = parsed.get("feedback") or parsed.get("rationale") or ""

    return {
        "score": score_value,
        "feedback": str(feedback).strip(),
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
) -> Optional[Dict[str, Any]]:
    question = extract_question(dataset_entry)
    rubric = format_scoring_rubric(extract_answer_scoring(dataset_entry))
    transcript_json = json.dumps(messages, ensure_ascii=False, indent=2)
    step_text = get_agent_step_text(messages, step)
    knowledge_snapshot = [dict(entry) for entry in prior_knowledge or []]
    knowledge_json = json.dumps(knowledge_snapshot, ensure_ascii=False, indent=2)
    candidates = _prepare_pool_candidates(step, used_knowledge_ids)
    if not candidates:
        return None
    candidate_json = json.dumps(_format_pool_candidates(candidates), ensure_ascii=False, indent=2)
    used_ids_json = json.dumps(sorted(used_knowledge_ids or set()), ensure_ascii=False, indent=2)
    prompt = textwrap.dedent(
        f"""
        You are selecting reusable knowledge to insert before agent step {step} in a web-search workflow.

        Knowledge snippets already injected (JSON array, matches this transcript exactly):
        {knowledge_json}

        Full message history before rerun (JSON transcript):
        {transcript_json}

        Original agent step {step} transcript:
        {step_text}

        Question:
        {question}

        Scoring rubric:
        {rubric}

        Candidate knowledge pool (JSON array):
        {candidate_json}

        Knowledge ids already used for this step in this problem (do not repeat them):
        {used_ids_json}

        Output format (JSON only):
        {{
          "id": "candidate_id",
          "justification": "short reason"
        }}
        """
    ).strip()
    try:
        response, _ = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=KNOWLEDGE_SELECTION_SYSTEM_PROMPT,
        )
    except Exception as exc:
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
    result = await orchestrator(
        messages=rerun_messages,
        run_name=run_name,
        knowledge_config=knowledge_config,
    )
    candidate_info = {
        "candidate_index": candidate_index + 1,
        "run_name": run_name,
        "run_id": result.get("run_id"),
        "output": result.get("output", ""),
        "knowledge": {
            "id": knowledge_entry.get("id"),
            "text": knowledge_entry.get("text"),
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
    question = extract_question(dataset_entry)
    rubric = extract_answer_scoring(dataset_entry)
    trial_records: List[Dict[str, Any]] = []
    scores: List[float] = []
    for trial in range(trials):
        rerun_messages = randomize_system_prompt(
            base_prompt_messages, use_knowledge_prompt=bool(manual_entries)
        )
        knowledge_config = build_knowledge_config(
            [dict(entry) for entry in manual_entries] if manual_entries else None
        )
        run_name = f"train_v4_eval_sample_{dataset_index:04d}_{label}_r{trial + 1}"
        result = await orchestrator(
            messages=rerun_messages,
            run_name=run_name,
            knowledge_config=knowledge_config,
        )
        output = result.get("output", "")
        score_info = await score_answer(orchestrator.llm, question, rubric, output)
        score = score_info.get("score", 0.0) or 0.0
        scores.append(score)
        trial_records.append(
            {
                "trial": trial + 1,
                "run_name": run_name,
                "run_id": result.get("run_id"),
                "score": score,
                "score_feedback": score_info.get("feedback"),
                "output": output,
            }
        )
    mean_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return trial_records, mean_score


async def run_problem(
    orchestrator,
    dataset_entry: Dict[str, Any],
    index: int,
    entry_id: str,
    *,
    n_knowledge_steps: int,
    knowledge_per_step: int,
    n_check_knowledge: int,
    final_reruns: int,
) -> Dict[str, Any]:
    question = extract_question(dataset_entry)
    rubric = extract_answer_scoring(dataset_entry)
    prompt_text = build_task_prompt(question)
    base_prompt_messages = [{"role": "user", "content": prompt_text}]

    chunk_records: List[Dict[str, Any]] = []
    base_inferences: List[Dict[str, Any]] = []

    for chunk_idx in range(knowledge_per_step):
        base_messages = randomize_system_prompt(base_prompt_messages)
        run_name = f"train_v4_eval_sample_{index:04d}_base_seed{chunk_idx + 1}"
        base_result = await orchestrator(
            messages=[dict(msg) for msg in base_messages],
            run_name=run_name,
            knowledge_config=build_knowledge_config(),
        )
        base_run = {
            "chunk_index": chunk_idx + 1,
            "run_name": run_name,
            "run_id": base_result.get("run_id"),
            "output": base_result.get("output", ""),
        }
        base_inferences.append(base_run)
        chunk_records.append(
            {
                "chunk_index": chunk_idx + 1,
                "base_run": base_run,
                "current_result": base_result,
                "current_run_id": base_run.get("run_id"),
                "manual_entries": [],
                "knowledge_steps": [],
            }
        )
        print(f"[sample {index}] chunk {chunk_idx + 1}")

    record: Dict[str, Any] = {
        "id": entry_id,
        "question": question,
        "answer_scoring": rubric,
        "base_inferences": base_inferences,
        "chunks": [],
        "knowledge_chunk_mean_scores": [],
    }

    for step in range(1, n_knowledge_steps + 1):
        used_ids_for_step: Set[str] = set()
        for chunk_data in chunk_records:
            messages = get_messages_for_run(chunk_data["current_result"])
            agent_messages = extract_agent_messages(messages)
            if step > len(agent_messages):
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
            )
            if not knowledge_entry:
                continue
            knowledge_id = str(knowledge_entry.get("id") or "").strip()
            if not knowledge_id or knowledge_id in used_ids_for_step:
                continue
            used_ids_for_step.add(knowledge_id)
            if not knowledge_entry.get("agent"):
                knowledge_entry["agent"] = get_agent_name_for_step(messages, step) or "web_search"

            candidate = await run_candidate_inference(
                orchestrator,
                dataset_entry=dataset_entry,
                truncated_history=truncated_history,
                step=step,
                knowledge_entry=knowledge_entry,
                dataset_index=index,
                candidate_index=chunk_data["chunk_index"] - 1,
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

    for chunk_data in chunk_records:
        chunk_data["final_single_run"] = {
            "run_id": chunk_data["current_run_id"],
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
            dataset_entry=dataset_entry,
            base_prompt_messages=base_prompt_messages,
            manual_entries=manual_entries if manual_entries else None,
            trials=n_check_knowledge,
            dataset_index=index,
            label=label,
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
        record["final_output"] = best_chunk_data["final_single_run"]["output"]
    else:
        record["final_run_id"] = None
        record["final_output"] = ""
        record["selected_knowledge"] = []
        record["final_single_run"] = None

    base_trials, base_mean = await run_full_problem_trials(
        orchestrator,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=None,
        trials=final_reruns,
        dataset_index=index,
        label="base_final",
    )
    knowledge_trials, knowledge_mean = await run_full_problem_trials(
        orchestrator,
        dataset_entry=dataset_entry,
        base_prompt_messages=base_prompt_messages,
        manual_entries=best_chunk_manual_entries if best_chunk_manual_entries else None,
        trials=final_reruns,
        dataset_index=index,
        label="knowledge_final",
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
        f"(delta {round(knowledge_mean - base_mean, 2)})"
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
                needs_resave = True
            elif isinstance(raw, list):
                entries = raw
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
            "knowledge_chunk_mean_scores": [],
        }
    base_vs_final: List[List[float]] = []
    improvements: List[float] = []
    chunk_score_table: List[List[float]] = []
    for entry in entries:
        base_score = entry.get("base_rerun_mean_score", 0.0)
        final_score = entry.get("final_rerun_mean_score", 0.0)
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
        agent_config=[{"name": "web_search"}, {"name": "answer"}],
        llm_kwargs={"model": "gpt-5-nano"},
        rm_kwargs={"url": args.rm_url, "agent_routing": False, "knowledge_search": True},
        allow_code_exec=False,
        agent_log_head_lines=1,
        max_tokens_per_question=60000,
        max_steps=2,
    )

    eval_entries, processed_ids, needs_resave = load_existing_eval_entries(dataset, args.output_path)
    if needs_resave:
        save_eval_entries(eval_entries, args.output_path)
    if processed_ids:
        print(f"[train_v4_eval_sample] Resuming with {len(processed_ids)} cached entries")

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
                n_check_knowledge=args.n_check_knowledge,
                final_reruns=args.final_reruns,
            )
            for dataset_idx, entry, entry_id in batch_slice
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
        save_eval_entries(eval_entries, args.output_path)
        batch_idx = batch_start // args.batch_size + 1
        print(
            f"[train_v4_eval_sample] Saved batch {batch_idx}: {len(processed_ids)}/{len(dataset)} problems complete"
        )

    print(f"[train_v4_eval_sample] Saved evaluation dataset to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation(parse_args()))
