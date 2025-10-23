from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from seimei.llm import LLMClient


DEFAULT_TOPICS: Tuple[str, ...] = (
    "ecommerce_orders",
    "iot_sensors",
    "hr_attrition",
    "ab_test_sessions",
    "supply_chain",
)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous synthetic-data assistant.
    Always answer with a single JSON object containing exactly the keys:
      - topic (string)
      - python (string with the full module source code)
      - question (string)
      - correct_answer (string)

    Do not wrap the JSON in markdown fences.
    The `topic` must exactly match the value provided in the latest user request.
    The `python` string must contain a valid Python module that defines:

        def generate(csv_output_path: str, n_samples_per_topic: int, n_hyper_params: int) -> None:
            ...

    That function must write a UTF-8 CSV file to `csv_output_path` and may create helper functions.
    Within the module, include a `if __name__ == "__main__":` entrypoint that accepts
    --csv-output-path, --n-samples-per-topic, and --n-hyper-params CLI flags delegating to `generate`.

    The produced question has to focus on the meaning of the generated data and its hyper-parameters.
    The correct_answer must concisely explain the true answer using those hyper-parameters.
    """
).strip()


class DatasetGenerationError(RuntimeError):
    """Raised when the generator cannot recover a valid artifact."""


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
    result: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, received '{item}'.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key detected in pair '{item}'.")
        result[key] = _coerce_value(value.strip())
    return result


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "topic"


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(
    base_prompt: str,
    *,
    topic: str,
    n_samples_per_topic: int,
    n_hyper_params: int,
    file_stub: str,
) -> str:
    return base_prompt.format(
        topic=topic,
        n_samples_per_topic=n_samples_per_topic,
        n_hyper_params=n_hyper_params,
        file_stub=file_stub,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty response.")

    fenced_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(0)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc

    for key in ("topic", "python", "question", "correct_answer"):
        if key not in data:
            raise ValueError(f"Missing key '{key}' in response JSON.")
    if not isinstance(data["python"], str):
        raise ValueError("The 'python' field must be a string containing module code.")
    return data


def write_module(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code.rstrip() + "\n", encoding="utf-8")


def _import_and_generate(module_path: Path, csv_path: Path, n_samples: int, n_hyper: int) -> Tuple[bool, str]:
    module_name = f"_seimei_dataset_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        return False, "Failed to load module spec."
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[call-arg]
    except Exception as exc:
        sys.modules.pop(module_name, None)
        return False, f"Import error: {exc}"

    try:
        generate_fn = getattr(module, "generate")
    except AttributeError:
        sys.modules.pop(module_name, None)
        return False, "The module is missing a 'generate' function."

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generate_fn(str(csv_path), int(n_samples), int(n_hyper))
    except Exception as exc:  # pragma: no cover - best effort logging
        sys.modules.pop(module_name, None)
        return False, f"generate(...) raised an exception: {exc}"
    finally:
        sys.modules.pop(module_name, None)
    return True, ""


def _run_via_subprocess(module_path: Path, csv_path: Path, n_samples: int, n_hyper: int, timeout: int) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        str(module_path),
        "--csv-output-path",
        str(csv_path),
        "--n-samples-per-topic",
        str(n_samples),
        "--n-hyper-params",
        str(n_hyper),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {timeout} seconds."
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        details = stderr or stdout or f"Exit code {proc.returncode}"
        return False, f"Script exited with code {proc.returncode}: {details}"
    return True, ""


def run_generated_module(
    module_path: Path,
    csv_path: Path,
    n_samples: int,
    n_hyper: int,
    *,
    prefer_import: bool = True,
    timeout: int = 120,
) -> Tuple[bool, str]:
    if prefer_import:
        ok, message = _import_and_generate(module_path, csv_path, n_samples, n_hyper)
        if ok:
            return True, ""
        fallback = message
        ok, message = _run_via_subprocess(module_path, csv_path, n_samples, n_hyper, timeout)
        if ok:
            return True, ""
        return False, "; ".join(filter(None, [fallback, message]))
    return _run_via_subprocess(module_path, csv_path, n_samples, n_hyper, timeout)


def preview_csv(csv_path: Path, limit: int = 5) -> Tuple[bool, str, List[List[str]]]:
    if not csv_path.is_file():
        return False, "CSV file was not created.", []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            rows: List[List[str]] = []
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx >= limit:
                    break
    except Exception as exc:  # pragma: no cover - best effort logging
        return False, f"Failed to read CSV: {exc}", []
    if not rows:
        return False, "CSV is empty.", []
    header = rows[0]
    if not header or any(cell == "" for cell in header):
        return False, "CSV header is missing or contains empty column names.", rows
    if len(rows) < 2:
        return False, "CSV contains only a header row.", rows
    return True, "", rows


def rows_to_table(rows: List[List[str]]) -> str:
    return "\n".join(",".join(row) for row in rows)


def format_dataset_path(base_arg: Path, exp_dir: Path, file_path: Path) -> str:
    try:
        relative = file_path.relative_to(exp_dir)
        return str(base_arg / relative)
    except ValueError:
        return str(file_path)


@dataclass
class DatasetEntry:
    topic: str
    question: str
    correct_answer: str
    python_path: Path
    csv_path: Path
    llm_response: Dict[str, Any]
    csv_preview: List[List[str]]

    def to_dict(self, exp_dir_arg: Path, exp_dir_path: Path) -> Dict[str, Any]:
        return {
            "Topic": self.topic,
            "Question": self.question,
            "CorrectAnswer": self.correct_answer,
            "PythonPath": format_dataset_path(exp_dir_arg, exp_dir_path, self.python_path),
            "CSVPath": format_dataset_path(exp_dir_arg, exp_dir_path, self.csv_path),
            "CSVPreview": self.csv_preview,
        }


async def request_valid_artifact(
    llm: LLMClient,
    prompt: str,
    topic: str,
    *,
    max_attempts: int,
    module_path: Path,
    csv_path: Path,
    n_samples_per_topic: int,
    n_hyper_params: int,
    prefer_import: bool,
    exec_timeout: int,
) -> DatasetEntry:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    attempt = 0
    last_error: Optional[str] = None

    while attempt < max_attempts:
        attempt += 1
        print(f"[{topic}] LLM attempt {attempt}/{max_attempts}")
        response_text, _ = await llm.chat(messages=messages, system=SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response_text})

        try:
            payload = extract_json_object(response_text)
        except ValueError as err:
            last_error = str(err)
            feedback = (
                "Your previous reply was not a valid JSON object with the required keys. "
                f"Error: {err}. Please send a compliant JSON object as described earlier."
            )
            messages.append({"role": "user", "content": feedback})
            continue

        if payload.get("topic") != topic:
            feedback = (
                f"The 'topic' field must match '{topic}', but you returned '{payload.get('topic')}'. "
                "Regenerate the full JSON response with the correct topic."
            )
            messages.append({"role": "user", "content": feedback})
            last_error = feedback
            continue

        python_code = payload["python"]
        question = payload["question"].strip()
        correct_answer = payload["correct_answer"].strip()

        write_module(module_path, python_code)

        ok, exec_error = run_generated_module(
            module_path,
            csv_path,
            n_samples_per_topic,
            n_hyper_params,
            prefer_import=prefer_import,
            timeout=exec_timeout,
        )
        if not ok:
            last_error = exec_error or "Unknown execution failure."
            feedback = (
                "Executing your module did not succeed.\n"
                f"Problem: {last_error}\n"
                "Please revise the entire JSON response (module + QA) to fix the issue."
            )
            messages.append({"role": "user", "content": feedback})
            continue

        valid, validation_message, preview = preview_csv(csv_path)
        if not valid:
            last_error = validation_message
            feedback = (
                "The generated CSV failed validation.\n"
                f"Problem: {validation_message}\n"
                f"Preview rows:\n{rows_to_table(preview) if preview else '[empty]'}\n"
                "Please send a new JSON response that fixes the module."
            )
            messages.append({"role": "user", "content": feedback})
            continue

        print(f"[{topic}] Artifact validated successfully.")
        return DatasetEntry(
            topic=topic,
            question=question,
            correct_answer=correct_answer,
            python_path=module_path,
            csv_path=csv_path,
            llm_response=payload,
            csv_preview=preview,
        )

    raise DatasetGenerationError(
        f"Failed to obtain a valid artifact for topic '{topic}' after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


async def generate_dataset(args: argparse.Namespace) -> List[DatasetEntry]:
    exp_dir_arg = Path(args.exp_dir)
    exp_dir_path = Path(args.exp_dir).resolve()
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    python_dir = Path(args.python_dir).resolve() if args.python_dir else exp_dir_path / "python"
    csv_dir = Path(args.csv_dir).resolve() if args.csv_dir else exp_dir_path / "csv"
    python_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path).resolve()
    base_prompt = load_prompt(prompt_path)

    topics = args.topics or list(DEFAULT_TOPICS)
    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    llm = LLMClient(**llm_kwargs)

    dataset_entries: List[DatasetEntry] = []
    for index, topic in enumerate(topics, start=1):
        slug = slugify(topic)
        file_stub = f"{slug}_{index:03d}"
        module_path = python_dir / f"{file_stub}.py"
        csv_path = csv_dir / f"{file_stub}.csv"

        prompt = build_prompt(
            base_prompt,
            topic=topic,
            n_samples_per_topic=args.n_samples_per_topic,
            n_hyper_params=args.n_hyper_params,
            file_stub=file_stub,
        )

        entry = await request_valid_artifact(
            llm,
            prompt,
            topic=topic,
            max_attempts=args.max_attempts,
            module_path=module_path,
            csv_path=csv_path,
            n_samples_per_topic=args.n_samples_per_topic,
            n_hyper_params=args.n_hyper_params,
            prefer_import=not args.prefer_subprocess,
            exec_timeout=args.exec_timeout,
        )
        dataset_entries.append(entry)

    output_path = Path(args.output_file_path).resolve() if args.output_file_path else exp_dir_path / "dataset.json"
    records = [entry.to_dict(exp_dir_arg, exp_dir_path) for entry in dataset_entries]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)
    print(f"Dataset written to {output_path}")
    return dataset_entries


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SEIMEI Excel evaluation dataset.")
    parser.add_argument("--model", default="gpt-5", help="Model name for LLM calls.")
    parser.add_argument("--exp-dir", default="exp1", help="Root directory for experiment outputs.")
    parser.add_argument(
        "--prompt-path",
        default="seimei/eval/data_generators/excel.md",
        help="Path to the prompt template fed to the LLM.",
    )
    parser.add_argument("--output-file-path", help="Where to write dataset.json (defaults to <exp_dir>/dataset.json).")
    parser.add_argument("--python-dir", help="Directory to store generated Python modules (defaults to <exp_dir>/python).")
    parser.add_argument("--csv-dir", help="Directory to store generated CSV files (defaults to <exp_dir>/csv).")
    parser.add_argument("--n-samples-per-topic", type=int, required=True, help="Rows per hyper-parameter setting.")
    parser.add_argument("--n-hyper-params", type=int, required=True, help="Number of hyper-parameter combinations.")
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Optional list of topics to generate. Defaults to built-in topic list.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retries per topic when validating artifacts.",
    )
    parser.add_argument(
        "--llm-kw",
        action="append",
        default=[],
        help="Additional key=value settings forwarded to LLMClient (repeatable).",
    )
    parser.add_argument(
        "--prefer-subprocess",
        action="store_true",
        help="Force execution via subprocess instead of importing the generated module.",
    )
    parser.add_argument(
        "--exec-timeout",
        type=int,
        default=180,
        help="Timeout (seconds) when executing generated scripts via subprocess.",
    )
    parsed = parser.parse_args(argv)
    parsed.llm_kwargs = parse_kv_pairs(parsed.llm_kw or [])
    return parsed


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)

    async def _run() -> None:
        await generate_dataset(args)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
