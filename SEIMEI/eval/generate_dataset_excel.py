from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import logging
import re
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

        def generate(csv_output_path: str, hyper_param_index: int, total_hyper_params: int) -> None:
            ...

    That function must write a UTF-8 CSV file to `csv_output_path` and may create helper functions.
    Within the module, include a `if __name__ == "__main__":` entrypoint that accepts
    --csv-output-path, --hyper-param-index, and --total-hyper-params CLI flags delegating to `generate`.

    We will execute the module multiple times for the same topic, passing hyper_param_index values
    from 1..total_hyper_params. Use that index to pick distinct parameterisations so each CSV differs.

    The produced question must focus on the meaning of the generated data and the hyper-parameters.
    The correct_answer must concisely explain the truth grounded in those hyper-parameters.
    """
).strip()


logger = logging.getLogger(__name__)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


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


def load_topics_from_file(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Topics file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Topics file does not contain valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Topics JSON must be a list of topic strings.")
    topics: List[str] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Topic at index {idx} must be a non-empty string.")
        topics.append(item.strip())
    if not topics:
        raise ValueError("Topics list is empty.")
    return topics


def build_prompt(
    base_prompt: str,
    *,
    topic: str,
    sample_index: int,
    total_samples: int,
    n_hyper_params: int,
    file_stub: str,
) -> str:
    values = {
        "topic": topic,
        "sample_index": sample_index,
        "total_samples": total_samples,
        "n_hyper_params": n_hyper_params,
        "file_stub": file_stub,
    }
    # Backward compatibility for templates that reference legacy placeholders.
    values.setdefault("hyper_index", "hyper_param_index")
    values.setdefault("hyper_param_index", "hyper_param_index")
    values.setdefault("n_samples_per_topic", total_samples)
    return base_prompt.format_map(_SafeFormatDict(values))


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


def _import_and_generate(
    module_path: Path,
    csv_path: Path,
    hyper_index: int,
    total_hyper_params: int,
) -> Tuple[bool, str]:
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
        generate_fn(str(csv_path), int(hyper_index), int(total_hyper_params))
    except Exception as exc:  # pragma: no cover - best effort logging
        sys.modules.pop(module_name, None)
        return False, f"generate(...) raised an exception: {exc}"
    finally:
        sys.modules.pop(module_name, None)
    return True, ""


def _run_via_subprocess(
    module_path: Path,
    csv_path: Path,
    hyper_index: int,
    total_hyper_params: int,
    timeout: int,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        str(module_path),
        "--csv-output-path",
        str(csv_path),
        "--hyper-param-index",
        str(hyper_index),
        "--total-hyper-params",
        str(total_hyper_params),
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
    hyper_index: int,
    total_hyper_params: int,
    *,
    prefer_import: bool = True,
    timeout: int = 120,
) -> Tuple[bool, str]:
    if prefer_import:
        ok, message = _import_and_generate(module_path, csv_path, hyper_index, total_hyper_params)
        if ok:
            return True, ""
        fallback = message
        ok, message = _run_via_subprocess(
            module_path,
            csv_path,
            hyper_index,
            total_hyper_params,
            timeout,
        )
        if ok:
            return True, ""
        return False, "; ".join(filter(None, [fallback, message]))
    return _run_via_subprocess(
        module_path,
        csv_path,
        hyper_index,
        total_hyper_params,
        timeout,
    )


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
class HyperArtifact:
    index: int
    csv_path: Path
    csv_preview: List[List[str]] = field(default_factory=list)

    def to_record(
        self,
        base_entry: Dict[str, Any],
        exp_dir_arg: Path,
        exp_dir_path: Path,
    ) -> Dict[str, Any]:
        record = dict(base_entry)
        record["HyperParamIndex"] = self.index
        record["CSVPath"] = format_dataset_path(exp_dir_arg, exp_dir_path, self.csv_path)
        record["CSVPreview"] = self.csv_preview
        return record


@dataclass
class DatasetEntry:
    topic: str
    sample_index: int
    question: str
    correct_answer: str
    python_path: Path
    llm_response: Dict[str, Any]
    hyper_artifacts: List[HyperArtifact] = field(default_factory=list)

    def to_records(self, exp_dir_arg: Path, exp_dir_path: Path) -> List[Dict[str, Any]]:
        base = {
            "Topic": self.topic,
            "SampleIndex": self.sample_index,
            "Question": self.question,
            "CorrectAnswer": self.correct_answer,
            "PythonPath": format_dataset_path(exp_dir_arg, exp_dir_path, self.python_path),
        }
        return [
            artifact.to_record(base, exp_dir_arg, exp_dir_path) for artifact in self.hyper_artifacts
        ]


async def request_valid_artifact(
    llm: LLMClient,
    prompt: str,
    topic: str,
    *,
    max_attempts: int,
    module_path: Path,
    csv_paths: Sequence[Path],
    sample_index: int,
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

        hyper_artifacts: List[HyperArtifact] = []
        generation_failed = False
        for hyper_index, csv_path in enumerate(csv_paths, start=1):
            if csv_path.exists():
                try:
                    csv_path.unlink()
                except OSError:
                    pass

            logger.info(
                "[%s] Generating CSV %s hyper %d/%d at %s",
                topic,
                sample_index,
                hyper_index,
                n_hyper_params,
                csv_path,
            )
            ok, exec_error = run_generated_module(
                module_path,
                csv_path,
                hyper_index,
                n_hyper_params,
                prefer_import=prefer_import,
                timeout=exec_timeout,
            )
            if not ok:
                last_error = exec_error or "Unknown execution failure."
                feedback = (
                    "Executing your module did not succeed.\n"
                    f"Hyper-parameter index: {hyper_index}/{n_hyper_params}\n"
                    f"Problem: {last_error}\n"
                    "Please revise the entire JSON response (module + QA) to fix the issue."
                )
                messages.append({"role": "user", "content": feedback})
                generation_failed = True
                break

            valid, validation_message, preview = preview_csv(csv_path)
            if not valid:
                last_error = validation_message
                feedback = (
                    "The generated CSV failed validation.\n"
                    f"Hyper-parameter index: {hyper_index}/{n_hyper_params}\n"
                    f"Problem: {validation_message}\n"
                    f"Preview rows:\n{rows_to_table(preview) if preview else '[empty]'}\n"
                    "Please send a new JSON response that fixes the module."
                )
                messages.append({"role": "user", "content": feedback})
                generation_failed = True
                break
            logger.info(
                "[%s] Generated CSV %s hyper %d/%d",
                topic,
                sample_index,
                hyper_index,
                n_hyper_params,
            )

            hyper_artifacts.append(
                HyperArtifact(index=hyper_index, csv_path=csv_path, csv_preview=preview)
            )

        if generation_failed:
            continue

        print(f"[{topic}] Artifact validated successfully (sample {sample_index}).")
        return DatasetEntry(
            topic=topic,
            sample_index=sample_index,
            question=question,
            correct_answer=correct_answer,
            python_path=module_path,
            llm_response=payload,
            hyper_artifacts=hyper_artifacts,
        )

    raise DatasetGenerationError(
        f"Failed to obtain a valid artifact for topic '{topic}' after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


async def request_artifact_once(
    llm: LLMClient,
    prompt: str,
    topic: str,
    *,
    module_path: Path,
    csv_paths: Sequence[Path],
    sample_index: int,
    n_hyper_params: int,
    prefer_import: bool,
    exec_timeout: int,
) -> DatasetEntry:
    response_text, _ = await llm.chat(messages=[{"role": "user", "content": prompt}], system=SYSTEM_PROMPT)
    payload = extract_json_object(response_text)
    if payload.get("topic") != topic:
        raise DatasetGenerationError(
            f"Expected topic '{topic}' but received '{payload.get('topic')}'."
        )

    python_code = payload["python"]
    question = payload["question"].strip()
    correct_answer = payload["correct_answer"].strip()

    write_module(module_path, python_code)

    hyper_artifacts: List[HyperArtifact] = []
    for hyper_index, csv_path in enumerate(csv_paths, start=1):
        if csv_path.exists():
            try:
                csv_path.unlink()
            except OSError as exc:
                raise DatasetGenerationError(
                    f"Failed to clear existing CSV '{csv_path}': {exc}"
                ) from exc

        logger.info(
            "[%s] Generating CSV %s hyper %d/%d at %s",
            topic,
            sample_index,
            hyper_index,
            n_hyper_params,
            csv_path,
        )
        ok, exec_error = run_generated_module(
            module_path,
            csv_path,
            hyper_index,
            n_hyper_params,
            prefer_import=prefer_import,
            timeout=exec_timeout,
        )
        if not ok:
            raise DatasetGenerationError(
                f"Execution failed for hyper-parameter {hyper_index}/{n_hyper_params}: {exec_error}"
            )

        valid, validation_message, preview = preview_csv(csv_path)
        if not valid:
            raise DatasetGenerationError(
                f"CSV validation failed for hyper-parameter {hyper_index}/{n_hyper_params}: {validation_message}"
            )

        logger.info(
            "[%s] Generated CSV %s hyper %d/%d",
            topic,
            sample_index,
            hyper_index,
            n_hyper_params,
        )
        hyper_artifacts.append(
            HyperArtifact(index=hyper_index, csv_path=csv_path, csv_preview=preview)
        )

    return DatasetEntry(
        topic=topic,
        sample_index=sample_index,
        question=question,
        correct_answer=correct_answer,
        python_path=module_path,
        llm_response=payload,
        hyper_artifacts=hyper_artifacts,
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

    topics: Sequence[str]
    if args.topics:
        topics = args.topics
    elif args.topics_path:
        topics_path = Path(args.topics_path)
        try:
            topics = load_topics_from_file(topics_path)
        except (OSError, ValueError) as exc:
            raise DatasetGenerationError(f"Failed to load topics from {topics_path}: {exc}") from exc
    else:
        topics = list(DEFAULT_TOPICS)
    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    llm = LLMClient(**llm_kwargs)

    dataset_entries: List[DatasetEntry] = []
    for topic in topics:
        slug = slugify(topic)
        for sample_index in range(1, args.n_samples_per_topic + 1):
            file_stub = f"{slug}_{sample_index}"
            module_path = python_dir / f"{file_stub}.py"
            csv_paths = [
                csv_dir / f"{slug}_{sample_index}_{hyper_index}.csv"
                for hyper_index in range(1, args.n_hyper_params + 1)
            ]

            prompt = build_prompt(
                base_prompt,
                topic=topic,
                sample_index=sample_index,
                total_samples=args.n_samples_per_topic,
                n_hyper_params=args.n_hyper_params,
                file_stub=file_stub,
            )

            entry: Optional[DatasetEntry] = None
            sample_attempt = 0
            max_sample_attempts = max(1, args.max_attempts)
            while sample_attempt < max_sample_attempts:
                sample_attempt += 1
                try:
                    logger.info(
                        "[%s] Generating module attempt %d/%d for sample %s",
                        topic,
                        sample_attempt,
                        max_sample_attempts,
                        file_stub,
                    )
                    if args.enable_validation:
                        entry = await request_valid_artifact(
                            llm,
                            prompt,
                            topic=topic,
                            max_attempts=args.max_attempts,
                            module_path=module_path,
                            csv_paths=csv_paths,
                            sample_index=sample_index,
                            n_hyper_params=args.n_hyper_params,
                            prefer_import=not args.prefer_subprocess,
                            exec_timeout=args.exec_timeout,
                        )
                    else:
                        entry = await request_artifact_once(
                            llm,
                            prompt,
                            topic=topic,
                            module_path=module_path,
                            csv_paths=csv_paths,
                            sample_index=sample_index,
                            n_hyper_params=args.n_hyper_params,
                            prefer_import=not args.prefer_subprocess,
                            exec_timeout=args.exec_timeout,
                        )
                    break
                except DatasetGenerationError as err:
                    logger.error(
                        "[%s] Generation failed for sample %s on attempt %d/%d: %s",
                        topic,
                        file_stub,
                        sample_attempt,
                        max_sample_attempts,
                        err,
                    )
                    if module_path.exists():
                        try:
                            module_path.unlink()
                        except OSError:
                            logger.debug("Failed to remove module %s", module_path, exc_info=True)
                    for csv_path in csv_paths:
                        if csv_path.exists():
                            try:
                                csv_path.unlink()
                            except OSError:
                                logger.debug("Failed to remove CSV %s", csv_path, exc_info=True)
                    if sample_attempt >= max_sample_attempts:
                        logger.error(
                            "[%s] Giving up on sample %s after %d attempts.",
                            topic,
                            file_stub,
                            max_sample_attempts,
                        )
                    else:
                        logger.info(
                            "[%s] Retrying sample %s (next attempt %d/%d).",
                            topic,
                            file_stub,
                            sample_attempt + 1,
                            max_sample_attempts,
                        )

            if entry is None:
                continue

            dataset_entries.append(entry)

    output_path = (
        Path(args.output_file_path).resolve() if args.output_file_path else exp_dir_path / "dataset.json"
    )
    records: List[Dict[str, Any]] = []
    for entry in dataset_entries:
        records.extend(entry.to_records(exp_dir_arg, exp_dir_path))
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
    parser.add_argument(
        "--n-samples-per-topic",
        type=int,
        required=True,
        help="Number of Python generator modules to request per topic.",
    )
    parser.add_argument(
        "--n-hyper-params",
        type=int,
        required=True,
        help="Number of hyper-parameter variations (CSV files) per Python module.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Optional list of topics to generate. Defaults to built-in topic list.",
    )
    parser.add_argument(
        "--topics-path",
        help="Path to a JSON file containing an array of topics. Defaults to the built-in topic list when omitted.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retries per topic when validation is enabled.",
    )
    parser.add_argument(
        "--llm-kw",
        action="append",
        default=[],
        help="Additional key=value settings forwarded to LLMClient (repeatable).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Shortcut for setting the LLM sampling temperature.",
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
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        help="Enable iterative validation and retry loop (slower but more robust).",
    )
    parsed = parser.parse_args(argv)
    parsed.llm_kwargs = parse_kv_pairs(parsed.llm_kw or [])
    if parsed.temperature is not None:
        parsed.llm_kwargs.setdefault("temperature", parsed.temperature)
    return parsed


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    async def _run() -> None:
        await generate_dataset(args)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
