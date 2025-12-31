"""
Example Usage:

```
python seimei/eval/generate_dataset_code.py \
    --repo-root /Users/multivac/Documents/github_project/gkvp/ \
    --exp-dir exp11_plasma_gkv_v5 \
    --model gpt-5-mini
```

```
python -m seimei.eval.generate_dataset_code.py \
    --file-config-path /path/to/file_config.json \
    --repo-root /path/to/gkvp \
    --exp-dir exp11_plasma_gkv_v5
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from seimei.llm import LLMClient


DEFAULT_FILE_CONFIG: List[Dict[str, Any]] = [
    {
        "folder_path": "./src/",
    },
    {
        "folder_path": "./run/",
        "exclude": ["backup/"],
    },
    {
        "folder_path": "./lib/",
        "exclude": ["sample_bessel/", "Bessel0_Zeros.f90"],
    },
    {
        "file_paths": [],
    },
]

DEFAULT_EXP_DIR = "exp11_plasma_gkv_v5"
DEFAULT_PROMPT_PATH = "seimei/eval/data_generators/code.md"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous code-mutation assistant.
    Always answer with a single JSON object containing exactly the keys:
      - file_path (string)
      - patches (array of objects)

    Each patches item must contain exactly the keys:
      - problem (string)
      - answer (string)
      - expected_simulation_result_difference (string)
      - patch (string)

    Do not wrap the JSON in markdown fences.
    The file_path must exactly match the value provided in the latest user request.
    The number of patches must exactly match PATCH_COUNT in the user request.
    Each patch must be a full apply_patch payload that:
      - Uses only "*** Update File: <file_path>" (no Add/Delete/Move).
      - Modifies only the target file.
      - Includes 3-10 lines of context above and below each change.
      - Keeps the code syntactically valid and runnable, while intentionally dropping or disabling a meaningful feature.
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


def load_file_config(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"File config does not contain valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("File config JSON must be a list of objects.")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError("File config entries must be objects.")
    return payload


def _resolve_path(raw: str, repo_root: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _is_excluded(path: Path, root: Path, exclude: Sequence[str]) -> bool:
    rel_path = path.relative_to(root).as_posix()
    for item in exclude:
        entry = str(item or "").strip()
        if not entry:
            continue
        if entry.endswith("/"):
            if rel_path.startswith(entry):
                return True
        else:
            if rel_path == entry or path.name == entry:
                return True
    return False


def collect_target_files(file_config: Sequence[Dict[str, Any]], repo_root: Path) -> List[Path]:
    ordered: Dict[Path, None] = {}
    for index, entry in enumerate(file_config, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"File config entry {index} must be an object.")
        if "folder_path" in entry:
            folder = _resolve_path(str(entry["folder_path"]), repo_root)
            if not folder.is_dir():
                raise FileNotFoundError(f"Folder path not found: {folder}")
            exclude = entry.get("exclude") or []
            if not isinstance(exclude, list):
                raise ValueError(f"exclude must be a list for entry {index}.")
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                if _is_excluded(path, folder, exclude):
                    continue
                ordered[path.resolve()] = None
        file_paths: List[str] = []
        if "file_path" in entry:
            file_paths.append(str(entry["file_path"]))
        if "file_paths" in entry:
            raw_paths = entry["file_paths"]
            if not isinstance(raw_paths, list):
                raise ValueError(f"file_paths must be a list for entry {index}.")
            file_paths.extend(str(item) for item in raw_paths)
        for raw in file_paths:
            candidate = _resolve_path(raw, repo_root)
            if not candidate.is_file():
                raise FileNotFoundError(f"File path not found: {candidate}")
            ordered[candidate.resolve()] = None
        if not {"folder_path", "file_path", "file_paths"} & set(entry.keys()):
            raise ValueError(
                f"File config entry {index} must include folder_path, file_path, or file_paths."
            )
    return list(ordered.keys())


def stringify_file_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.relative_to(repo_root)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def estimate_patch_count(file_text: str) -> int:
    non_empty_lines = sum(1 for line in file_text.splitlines() if line.strip())
    step = max(0, (non_empty_lines - 1) // 200)
    return max(3, min(10, 3 + step))


def build_prompt(
    base_prompt: str,
    *,
    file_path: str,
    patch_count: int,
    file_content: str,
) -> str:
    values = {
        "base_prompt": base_prompt,
        "file_path": file_path,
        "patch_count": patch_count,
        "file_content": file_content,
    }
    template = textwrap.dedent(
        """
        {base_prompt}

        You are generating patches for a single file only.

        CONFIG:
        TARGET_FILE = "{file_path}"
        PATCH_COUNT = {patch_count}

        Hard constraints
        - Modify ONLY the TARGET_FILE shown above.
        - If the base prompt mentions other allowed paths, ignore it; TARGET_FILE is the only file you may edit.
        - Keep the code syntactically valid and runnable, but intentionally drop or disable a meaningful feature.
        - Use the apply_patch format shown below.
        - Provide 3 to 10 lines of context above and below each change.

        Patch file format (must follow)
        *** Begin Patch
        *** Update File: {file_path}
        @@
        -<old line(s)>
        +<new line(s)>
        *** End Patch

        Dataset fields per patch
        - problem
        - answer
        - expected_simulation_result_difference

        Target file content (verbatim)
        <file>
        {file_content}
        </file>
        """
    ).strip()
    return template.format_map(_SafeFormatDict(values))


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
    return data


def _validate_patch_text(patch_text: str, expected_file_path: str) -> None:
    if "*** Begin Patch" not in patch_text or "*** End Patch" not in patch_text:
        raise ValueError("Patch must include Begin Patch and End Patch markers.")
    if "*** Add File:" in patch_text or "*** Delete File:" in patch_text or "*** Move to:" in patch_text:
        raise ValueError("Patch must use Update File only.")
    update_lines = [
        line.strip()
        for line in patch_text.splitlines()
        if line.strip().startswith("*** Update File:")
    ]
    if len(update_lines) != 1:
        raise ValueError("Patch must contain exactly one Update File line.")
    update_path = update_lines[0].split(":", 1)[-1].strip()
    if update_path != expected_file_path:
        raise ValueError(
            f"Patch Update File path '{update_path}' does not match '{expected_file_path}'."
        )
    if "@@" not in patch_text:
        raise ValueError("Patch is missing @@ hunk markers.")


@dataclass
class PatchEntry:
    problem: str
    answer: str
    expected_simulation_result_difference: str
    patch: str


@dataclass
class FileSpec:
    file_path: str
    abs_path: Path
    patch_count: int
    prompt: str


@dataclass
class FilePatchBundle:
    file_path: str
    patches: List[PatchEntry]


def normalize_patch_bundle(
    payload: Dict[str, Any],
    *,
    expected_file_path: str,
    patch_count: int,
) -> FilePatchBundle:
    file_path = str(payload.get("file_path") or "").strip()
    if file_path != expected_file_path:
        raise ValueError(f"Expected file_path '{expected_file_path}', got '{file_path}'.")

    raw_patches = payload.get("patches")
    if not isinstance(raw_patches, list):
        raise ValueError("patches must be an array.")
    if len(raw_patches) != patch_count:
        raise ValueError(f"Expected {patch_count} patches, got {len(raw_patches)}.")

    patches: List[PatchEntry] = []
    for idx, raw in enumerate(raw_patches, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"patches[{idx}] must be an object.")
        problem = str(raw.get("problem") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        expected_diff = str(raw.get("expected_simulation_result_difference") or "").strip()
        patch_text = str(raw.get("patch") or "").strip()
        if not problem:
            raise ValueError(f"patches[{idx}] problem is empty.")
        if not answer:
            raise ValueError(f"patches[{idx}] answer is empty.")
        if not expected_diff:
            raise ValueError(f"patches[{idx}] expected_simulation_result_difference is empty.")
        if not patch_text:
            raise ValueError(f"patches[{idx}] patch is empty.")
        _validate_patch_text(patch_text, expected_file_path)
        patches.append(
            PatchEntry(
                problem=problem,
                answer=answer,
                expected_simulation_result_difference=expected_diff,
                patch=patch_text,
            )
        )

    return FilePatchBundle(file_path=expected_file_path, patches=patches)


def _record_source_file(record: Dict[str, Any]) -> Optional[str]:
    source = record.get("source_file")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return None


def _write_dataset(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(list(records), fp, ensure_ascii=False, indent=2)


def _write_patch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _next_patch_index(patch_dir: Path) -> int:
    max_index = -1
    if patch_dir.exists():
        for path in patch_dir.glob("patch*.txt"):
            match = re.match(r"patch(\d+)\.txt$", path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


async def request_valid_bundle(
    llm: LLMClient,
    spec: FileSpec,
    *,
    max_attempts: int,
) -> FilePatchBundle:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": spec.prompt}]
    attempt = 0
    last_error: Optional[str] = None

    while attempt < max_attempts:
        attempt += 1
        print(f"[{spec.file_path}] LLM attempt {attempt}/{max_attempts}")
        response_text, _ = await llm.chat(messages=messages, system=SYSTEM_PROMPT)
        messages.append({"role": "assistant", "content": response_text})

        try:
            payload = extract_json_object(response_text)
            return normalize_patch_bundle(
                payload,
                expected_file_path=spec.file_path,
                patch_count=spec.patch_count,
            )
        except ValueError as err:
            last_error = str(err)
            feedback = (
                "Your previous reply did not match the required JSON schema.\n"
                f"Error: {err}\n"
                "Please regenerate the full JSON object with the exact schema."
            )
            messages.append({"role": "user", "content": feedback})
            continue

    raise DatasetGenerationError(
        f"Failed to obtain valid patches for '{spec.file_path}' after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


async def request_bundle_once(
    llm: LLMClient,
    spec: FileSpec,
) -> FilePatchBundle:
    response_text, _ = await llm.chat(messages=[{"role": "user", "content": spec.prompt}], system=SYSTEM_PROMPT)
    payload = extract_json_object(response_text)
    return normalize_patch_bundle(
        payload,
        expected_file_path=spec.file_path,
        patch_count=spec.patch_count,
    )


async def _generate_file_with_retries(
    spec: FileSpec,
    args: argparse.Namespace,
    llm: LLMClient,
) -> Optional[FilePatchBundle]:
    try:
        if args.enable_validation:
            return await request_valid_bundle(llm, spec, max_attempts=args.max_attempts)
        return await request_bundle_once(llm, spec)
    except DatasetGenerationError as err:
        logger.error("[%s] Generation failed: %s", spec.file_path, err)
    except Exception:
        logger.exception("[%s] Unexpected failure", spec.file_path)
    return None


async def generate_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.batch_size <= 0:
        raise DatasetGenerationError("--batch-size must be a positive integer.")

    repo_root = Path(args.repo_root).resolve()
    exp_dir_path = Path(args.exp_dir).resolve()
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    patch_dir = Path(args.patch_dir).resolve() if args.patch_dir else exp_dir_path / "patch_files"
    patch_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output_file_path).resolve() if args.output_file_path else exp_dir_path / "dataset.json"
    )

    prompt_path = Path(args.prompt_path).resolve()
    base_prompt = load_prompt(prompt_path)

    if args.file_config_path:
        file_config = load_file_config(Path(args.file_config_path))
    else:
        file_config = DEFAULT_FILE_CONFIG

    target_files = collect_target_files(file_config, repo_root)
    if not target_files:
        raise DatasetGenerationError("No files matched the file config.")

    specs: List[FileSpec] = []
    for path in target_files:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
            continue
        patch_count = estimate_patch_count(content)
        file_path = stringify_file_path(path, repo_root)
        prompt = build_prompt(
            base_prompt,
            file_path=file_path,
            patch_count=patch_count,
            file_content=content,
        )
        specs.append(
            FileSpec(
                file_path=file_path,
                abs_path=path,
                patch_count=patch_count,
                prompt=prompt,
            )
        )
    if not specs:
        raise DatasetGenerationError("No readable files matched the file config.")

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
    existing_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for record in existing_records:
        source_file = _record_source_file(record)
        if source_file:
            existing_by_file.setdefault(source_file, []).append(record)

    files_to_regen: List[str] = []
    specs_to_generate: List[FileSpec] = []
    if args.force:
        files_to_regen = [spec.file_path for spec in specs]
        specs_to_generate = list(specs)
    else:
        for spec in specs:
            existing_count = len(existing_by_file.get(spec.file_path, []))
            if existing_count >= spec.patch_count:
                logger.info(
                    "[%s] Skipping; %d patches already recorded in %s.",
                    spec.file_path,
                    existing_count,
                    output_path,
                )
                continue
            if existing_count > 0:
                files_to_regen.append(spec.file_path)
            specs_to_generate.append(spec)

    if files_to_regen:
        regen_set = set(files_to_regen)
        records_accumulator = [
            record
            for record in records_accumulator
            if _record_source_file(record) not in regen_set
        ]

    if not specs_to_generate:
        if not output_path.exists():
            _write_dataset(records_accumulator, output_path)
        print(f"Dataset written to {output_path}")
        return records_accumulator

    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    llm = LLMClient(**llm_kwargs)

    patch_index = _next_patch_index(patch_dir)
    total_batches = (len(specs_to_generate) + args.batch_size - 1) // args.batch_size
    for batch_index in range(total_batches):
        batch_start = batch_index * args.batch_size
        batch_specs = specs_to_generate[batch_start : batch_start + args.batch_size]
        batch_tasks = [
            _generate_file_with_retries(spec, args, llm) for spec in batch_specs
        ]
        batch_entries = await asyncio.gather(*batch_tasks)
        for spec, bundle in zip(batch_specs, batch_entries):
            if bundle is None:
                continue
            for local_index, patch in enumerate(bundle.patches, start=1):
                patch_path = patch_dir / f"patch{patch_index}.txt"
                _write_patch(patch_path, patch.patch)
                records_accumulator.append(
                    {
                        "problem": patch.problem,
                        "answer": patch.answer,
                        "expected_simulation_result_difference": patch.expected_simulation_result_difference,
                        "patch_file": patch_path.name,
                        "source_file": spec.file_path,
                        "source_patch_index": local_index,
                    }
                )
                patch_index += 1
        _write_dataset(records_accumulator, output_path)
        print(
            f"[Batch {batch_index + 1}/{total_batches}] Saved dataset snapshot to {output_path}"
            f" ({len(records_accumulator)} records)"
        )

    print(f"Dataset written to {output_path}")
    return records_accumulator


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SEIMEI code-patch evaluation dataset.")
    parser.add_argument("--model", default="gpt-5", help="Model name for LLM calls.")
    parser.add_argument(
        "--exp-dir",
        default=DEFAULT_EXP_DIR,
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--prompt-path",
        default=DEFAULT_PROMPT_PATH,
        help="Path to the prompt template fed to the LLM.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve file_config paths.",
    )
    parser.add_argument(
        "--file-config-path",
        help="Path to a JSON file containing the file_config list.",
    )
    parser.add_argument(
        "--patch-dir",
        help="Directory to store patch files (defaults to <exp_dir>/patch_files).",
    )
    parser.add_argument(
        "--output-file-path",
        help="Where to write dataset.json (defaults to <exp_dir>/dataset.json).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of files to generate concurrently.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retries per file when validation is enabled.",
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        help="Enable iterative validation and retry loop.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate patches even if dataset entries already exist (keeps unrelated records).",
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
