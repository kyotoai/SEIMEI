from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from seimei.llm import LLMClient


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an impartial evaluator of question answering performance on synthetic datasets.
    For each case, decide whether the model's output is semantically correct given:
      - the original question,
      - the authoritative correct answer,
      - a snippet of the Python generator that produced the data,
      - (optionally) a preview of the CSV rows that were generated.

    Respond with a single JSON object containing the keys:
      - verdict: "correct", "incorrect", or "uncertain"
      - score: 1 if the answer is correct, otherwise 0
      - explanation: a brief justification referring to the evidence

    Do not add Markdown code fences around the JSON.
    """
).strip()


def _coerce_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if lowered.startswith(("0x", "0o", "0b")):
            return int(lowered, 0)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_kv_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, received '{item}'.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key detected in pair '{item}'.")
        params[key] = _coerce_value(value.strip())
    return params


def load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Result file {path} does not contain a JSON list.")
    return [dict(item) for item in data]


def resolve_dataset_path(raw: str, exp_dir_arg: Path, exp_dir_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    try:
        relative = path.relative_to(exp_dir_arg)
        return (exp_dir_path / relative).resolve()
    except ValueError:
        return (Path.cwd() / path).resolve()


def read_script_excerpt(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def table_from_rows(rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(",".join(str(cell) for cell in row) for row in rows)


def build_eval_prompt(
    record: Dict[str, Any],
    python_excerpt: str,
    csv_preview: Sequence[Sequence[str]],
) -> str:
    csv_text = table_from_rows(csv_preview) if csv_preview else "CSV preview unavailable."
    return textwrap.dedent(
        f"""
        Question:
        {record.get('Question', '').strip()}

        Reference correct answer:
        {record.get('CorrectAnswer', '').strip()}

        Model output to evaluate:
        {record.get('Output', '').strip()}

        Python generator excerpt:
        ```python
        {python_excerpt}
        ```

        CSV preview (if available):
        {csv_text}

        Determine whether the model output matches the correct answer and the intent of the generator data.
        """
    ).strip()


def extract_json(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty response.")
    if candidate.startswith("```"):
        candidate = candidate.strip("` \n")
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    for key in ("verdict", "score", "explanation"):
        if key not in data:
            raise ValueError(f"Missing key '{key}' in evaluation response.")
    return data


async def evaluate_record(
    llm: LLMClient,
    prompt: str,
    *,
    max_attempts: int,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    last_error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        print(f"[evaluation] LLM attempt {attempt}/{max_attempts}")
        response_text, _ = await llm.chat(messages=messages, system=SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response_text})
        try:
            return extract_json(response_text)
        except ValueError as err:
            last_error = str(err)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your response was invalid because:\n"
                        f"{err}\n"
                        "Please return only the JSON verdict with the required keys."
                    ),
                }
            )
    raise RuntimeError(f"Failed to obtain a valid evaluation after {max_attempts} attempts. Last error: {last_error}")


async def run_evaluation(args: argparse.Namespace) -> List[Dict[str, Any]]:
    exp_dir_arg = Path(args.exp_dir)
    exp_dir_path = Path(args.exp_dir).resolve()
    result_path = Path(args.result_path).resolve() if args.result_path else exp_dir_path / "result.json"
    output_path = Path(args.output_path).resolve() if args.output_path else result_path

    records = load_results(result_path)
    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    llm = LLMClient(**llm_kwargs)

    updated_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if record.get("Correctness") and not args.force:
            updated_records.append(record)
            continue

        python_path = resolve_dataset_path(record.get("PythonPath", ""), exp_dir_arg, exp_dir_path)
        csv_preview = record.get("CSVPreview") or record.get("CSVPathPreview") or []
        python_excerpt = read_script_excerpt(python_path, args.script_excerpt_chars)

        prompt = build_eval_prompt(record, python_excerpt, csv_preview)

        print(f"[evaluation] Judging sample {index}/{len(records)}")
        verdict = await evaluate_record(
            llm,
            prompt,
            max_attempts=args.max_attempts,
        )

        score = int(verdict.get("score", 0))
        verdict_label = str(verdict.get("verdict", "uncertain"))
        explanation = str(verdict.get("explanation", "")).strip()

        record["Correctness"] = {
            "score": 1 if score else 0,
            "verdict": verdict_label,
            "explanation": explanation or "Unavailable.",
        }
        updated_records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(updated_records, fp, ensure_ascii=False, indent=2)
    print(f"Evaluation results written to {output_path}")
    return updated_records


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge SEIMEI inference outputs with an LLM.")
    parser.add_argument("--model", default="gpt-5", help="Model used for evaluation.")
    parser.add_argument("--exp-dir", default="exp1", help="Experiment directory for resolving relative paths.")
    parser.add_argument("--result-path", help="Path to result.json (defaults to <exp_dir>/result.json).")
    parser.add_argument("--output-path", help="Path for the updated result file (defaults to --result-path).")
    parser.add_argument(
        "--script-excerpt-chars",
        type=int,
        default=2000,
        help="Maximum characters of the generator script to include in the evaluation prompt.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Number of retries if the evaluator response is malformed.",
    )
    parser.add_argument(
        "--llm-kw",
        action="append",
        default=[],
        help="Additional key=value options forwarded to LLMClient.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute correctness even when the field already exists.",
    )
    parsed = parser.parse_args(argv)
    parsed.llm_kwargs = parse_kv_pairs(parsed.llm_kw or [])
    return parsed


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)

    async def _run() -> None:
        await run_evaluation(args)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
