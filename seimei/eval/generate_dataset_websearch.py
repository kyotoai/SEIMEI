from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from seimei.llm import LLMClient


DEFAULT_TOPICS: Tuple[str, ...] = (
    "renewable_energy_capacity_growth",
    "electric_vehicle_charging_infrastructure",
    "global_shipping_emissions_regulations",
    "urban_bike_share_usage_trends",
    "semiconductor_factory_expansions",
    "major_earthquake_aftershock_statistics",
    "public_transit_fare_changes",
    "satellite_launch_cadence",
    "coastal_sea_level_rise_projections",
    "agricultural_drought_impact_reports",
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous synthetic-data assistant.
    Always answer with a single JSON object containing exactly the keys:
      - topic (string)
      - question (string)
      - answer_scoring (array of objects)

    Do not wrap the JSON in markdown fences.
    The `topic` must exactly match the value provided in the latest user request.
    The `question` must require multiple web searches and cite sources.
    The `answer_scoring` array must contain objects with:
      - requirement (string)
      - score (integer)
    The scores must sum to 10.
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
) -> str:
    values = {
        "topic": topic,
        "sample_index": sample_index,
        "total_samples": total_samples,
    }
    return base_prompt.format_map(_SafeFormatDict(values))


def extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty response.")

    candidate = candidate.replace("```json", "").replace("```", "").strip()

    matches = re.findall(r"\{.*\}", candidate, flags=re.DOTALL)
    if matches:
        candidate = max(matches, key=len)
    if len(candidate) < 10 or candidate == "{}":
        raise ValueError("No meaningful JSON object found in response.")

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc

    for key in ("topic", "question", "answer_scoring"):
        if key not in data:
            raise ValueError(f"Missing key '{key}' in response JSON.")
    return data


def normalize_answer_scoring(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("answer_scoring must be a JSON array.")
    normalized: List[Dict[str, Any]] = []
    total = 0
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"answer_scoring item {idx} must be an object.")
        requirement = str(item.get("requirement") or "").strip()
        if not requirement:
            raise ValueError(f"answer_scoring item {idx} missing requirement.")
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            raise ValueError(f"answer_scoring item {idx} missing integer score.")
        if score <= 0:
            raise ValueError(f"answer_scoring item {idx} must have positive score.")
        normalized.append({"requirement": requirement, "score": score})
        total += score
    if total != 10:
        raise ValueError(f"answer_scoring scores must sum to 10 (got {total}).")
    return normalized


@dataclass
class DatasetEntry:
    topic: str
    sample_index: int
    question: str
    answer_scoring: List[Dict[str, Any]]
    llm_response: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "sample_index": self.sample_index,
            "question": self.question,
            "answer_scoring": self.answer_scoring,
        }


@dataclass
class SampleSpec:
    topic: str
    sample_index: int
    prompt: str


def _write_dataset(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(list(records), fp, ensure_ascii=False, indent=2)


def _record_sample_key(record: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    topic = record.get("topic")
    if not isinstance(topic, str):
        return None
    try:
        sample_index = int(record.get("sample_index"))
    except (TypeError, ValueError):
        return None
    return topic, sample_index


def _completed_samples_from_records(records: Sequence[Dict[str, Any]]) -> Set[Tuple[str, int]]:
    completed: Set[Tuple[str, int]] = set()
    for record in records:
        key = _record_sample_key(record)
        if key is not None:
            completed.add(key)
    return completed


async def request_valid_entry(
    llm: LLMClient,
    prompt: str,
    topic: str,
    *,
    sample_index: int,
    max_attempts: int,
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
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not a valid JSON object with the required keys. "
                        f"Error: {err}. Please send a compliant JSON object."
                    ),
                }
            )
            continue

        if payload.get("topic") != topic:
            last_error = f"Expected topic '{topic}', got '{payload.get('topic')}'."
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The 'topic' field must match '{topic}'. Please regenerate the full JSON."
                    ),
                }
            )
            continue

        question = str(payload.get("question") or "").strip()
        if not question:
            last_error = "Question is empty."
            messages.append({"role": "user", "content": "The question must be non-empty."})
            continue
        if "search" not in question.lower():
            last_error = "Question does not mention web search."
            messages.append(
                {
                    "role": "user",
                    "content": "The question must explicitly instruct the assistant to use web search.",
                }
            )
            continue

        try:
            answer_scoring = normalize_answer_scoring(payload.get("answer_scoring"))
        except ValueError as err:
            last_error = str(err)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The answer_scoring rubric was invalid. "
                        f"Error: {err}. Please regenerate the full JSON."
                    ),
                }
            )
            continue

        return DatasetEntry(
            topic=topic,
            sample_index=sample_index,
            question=question,
            answer_scoring=answer_scoring,
            llm_response=payload,
        )

    raise DatasetGenerationError(
        f"Failed to obtain a valid artifact for topic '{topic}' after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


async def _generate_sample_with_retries(
    spec: SampleSpec,
    llm: LLMClient,
    *,
    max_attempts: int,
) -> Optional[DatasetEntry]:
    try:
        return await request_valid_entry(
            llm,
            spec.prompt,
            topic=spec.topic,
            sample_index=spec.sample_index,
            max_attempts=max_attempts,
        )
    except DatasetGenerationError as err:
        logger.error(
            "[%s] Generation failed for sample %d after %d attempts: %s",
            spec.topic,
            spec.sample_index,
            max_attempts,
            err,
        )
    except Exception:
        logger.exception(
            "[%s] Unexpected failure for sample %d",
            spec.topic,
            spec.sample_index,
        )
    return None


async def generate_dataset(args: argparse.Namespace) -> List[DatasetEntry]:
    if args.batch_size <= 0:
        raise DatasetGenerationError("--batch-size must be a positive integer.")

    exp_dir_arg = Path(args.exp_dir)
    exp_dir_path = Path(args.exp_dir).resolve()
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path).resolve()
    base_prompt = load_prompt(prompt_path)

    topics: Sequence[str]
    if args.topics:
        topics = args.topics
    elif args.topics_path:
        topics = load_topics_from_file(Path(args.topics_path))
    else:
        topics = list(DEFAULT_TOPICS)

    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    llm = LLMClient(**llm_kwargs)

    output_path = (
        Path(args.output_file_path).resolve() if args.output_file_path else exp_dir_path / "dataset.json"
    )

    existing_records: List[Dict[str, Any]] = []
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetGenerationError(
                f"Failed to load existing dataset from {output_path}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise DatasetGenerationError(
                f"Existing dataset file {output_path} must contain a JSON list."
            )
        if any(not isinstance(item, dict) for item in payload):
            raise DatasetGenerationError(
                f"Existing dataset file {output_path} contains non-object entries."
            )
        existing_records = payload

    records_accumulator: List[Dict[str, Any]] = list(existing_records)
    completed_samples = _completed_samples_from_records(existing_records)

    sample_specs: List[SampleSpec] = []
    for topic in topics:
        for sample_index in range(1, args.n_samples_per_topic + 1):
            if (topic, sample_index) in completed_samples:
                logger.info(
                    "[%s] Skipping sample %d; already recorded in %s.",
                    topic,
                    sample_index,
                    output_path,
                )
                continue
            prompt = build_prompt(
                base_prompt,
                topic=topic,
                sample_index=sample_index,
                total_samples=args.n_samples_per_topic,
            )
            sample_specs.append(
                SampleSpec(topic=topic, sample_index=sample_index, prompt=prompt)
            )

    pending_sample_keys = {(spec.topic, spec.sample_index) for spec in sample_specs}
    if pending_sample_keys:
        records_accumulator = [
            record
            for record in records_accumulator
            if _record_sample_key(record) not in pending_sample_keys
        ]

    dataset_entries: List[DatasetEntry] = []
    if not sample_specs:
        if not output_path.exists():
            _write_dataset(records_accumulator, output_path)
        print(f"Dataset written to {output_path}")
        return dataset_entries

    total_batches = (len(sample_specs) + args.batch_size - 1) // args.batch_size
    for batch_index in range(total_batches):
        batch_start = batch_index * args.batch_size
        batch_specs = sample_specs[batch_start : batch_start + args.batch_size]
        batch_tasks = [
            _generate_sample_with_retries(spec, llm, max_attempts=args.max_attempts)
            for spec in batch_specs
        ]
        batch_entries = await asyncio.gather(*batch_tasks)
        for entry in batch_entries:
            if entry is None:
                continue
            dataset_entries.append(entry)
            records_accumulator.append(entry.to_record())
        _write_dataset(records_accumulator, output_path)
        print(
            f"[Batch {batch_index + 1}/{total_batches}] Saved dataset snapshot to {output_path}"
            f" ({len(records_accumulator)} records)"
        )

    print(f"Dataset written to {output_path}")
    return dataset_entries


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SEIMEI web-search evaluation dataset.")
    parser.add_argument("--model", default="gpt-5", help="Model name for LLM calls.")
    parser.add_argument("--exp-dir", default="exp12_websearch_small", help="Root directory for experiment outputs.")
    parser.add_argument(
        "--prompt-path",
        default="seimei/eval/data_generators/websearch.md",
        help="Path to the prompt template fed to the LLM.",
    )
    parser.add_argument("--output-file-path", help="Where to write dataset.json (defaults to <exp_dir>/dataset.json).")
    parser.add_argument(
        "--n-samples-per-topic",
        type=int,
        required=True,
        help="Number of questions to request per topic.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples to generate concurrently.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Optional list of topics to generate. Defaults to built-in topic list.",
    )
    parser.add_argument(
        "--topics-path",
        help="Path to a JSON file containing an array of topics.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retries per sample.",
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
