import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from seimei import seimei

EVAL_DIR = "exp5"
EVAL_NAME = "excel_eval_v1"
SCORING_SYSTEM_PROMPT = (
    "You are an impartial evaluator scoring an assistant's answer against a reference answer. "
    "Judge factual accuracy, coverage, and clarity. Return ONLY a JSON object with keys 'score' "
    "(integer 0-10) and 'feedback' (concise justification). Score 0 means entirely incorrect, "
    "10 means fully correct."
)


def _strip_code_fences(text: str) -> str:
    snippet = text.strip()
    if snippet.startswith("```"):
        lines = snippet.splitlines()
        # Drop leading fence line.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop trailing fence line.
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
        "feedback": feedback.strip(),
        "raw_response": response,
        "judge_usage": usage,
    }


async def demo_code_act():

    with open(f"{EVAL_DIR}/dataset.json") as f:
        dataset = json.load(f)
    
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-5-nano"},
        allow_code_exec=True,
        #allowed_commands=["ls", "echo"],
        agent_log_head_lines=1,
        max_tokens_per_question=20000,
        #load_knowledge_path="seimei_knowledge/excel.csv",
    )

    orchestrators = []
    for run_id, dataset_dict in enumerate(dataset):

        question = dataset_dict["Question"]
        csv_path = dataset_dict["CSVPath"]

        orchestrators.append(orchestrator(
            messages=[
                {"role": "system", "content": "You are a smart code executor to analyze csv file and "},
                {"role": "user", "content": f"Analyze inside {csv_path} and answer the question below:\n\n{question}"},
            ],
            run_name=f"run_{run_id}",
            generate_knowledge=True,
            save_knowledge_path="seimei_knowledge/excel.csv",
            knowledge_prompt_path="seimei/knowledge/prompts/excel.md",
        ))

    results = await asyncio.gather(*orchestrators)

    eval_tasks = [
        score_answer(
            orchestrator.llm,
            dataset_dict.get("Question", ""),
            dataset_dict.get("CorrectAnswer", ""),
            result.get("output", "") if isinstance(result, dict) else "",
        )
        for dataset_dict, result in zip(dataset, results)
    ]
    eval_results = await asyncio.gather(*eval_tasks) if eval_tasks else []

    eval_records = []
    for idx, (dataset_dict, result, eval_result) in enumerate(zip(dataset, results, eval_results)):
        record = {
            "index": idx,
            "run_id": result.get("run_id") if isinstance(result, dict) else None,
            "question": dataset_dict.get("Question"),
            "model_answer": result.get("output") if isinstance(result, dict) else None,
            "reference_answer": dataset_dict.get("CorrectAnswer"),
            "score": eval_result.get("score"),
            "feedback": eval_result.get("feedback"),
            "raw_judge_response": eval_result.get("raw_response"),
            "judge_usage": eval_result.get("judge_usage"),
        }
        eval_records.append(record)

    avg_score = round(
        sum(item.get("score", 0.0) for item in eval_results) / len(eval_results),
        2,
    ) if eval_results else 0.0

    eval_payload = {
        "eval_name": EVAL_NAME,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "average_score": avg_score,
        "records": eval_records,
    }

    eval_dir = Path(f"{EVAL_DIR}/evals")
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{EVAL_NAME}.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, ensure_ascii=False, indent=2)
    print(f"Evaluation saved to {eval_path}")

asyncio.run(demo_code_act())
