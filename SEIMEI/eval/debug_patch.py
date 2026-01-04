from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from seimei.editing import PatchApplyError, PatchParseError, apply_patch_to_workspace
from seimei.llm import LLMClient
from seimei.eval.generate_dataset_code import (
    _validate_patch_text,
    collect_target_files,
    load_file_config,
    parse_kv_pairs,
    stringify_file_path,
)


DEFAULT_N_DEBUG_LOOP = 3
DEFAULT_WORKSPACE_NAME = "debug_workspace"

DEFAULT_REPO_ROOT = "/Users/multivac/Documents/github_project/gkvp/"
DEFAULT_EXP_DIR = "exp11_plasma_gkv_v5"
DEFAULT_MODEL_NAME = "gpt-5-mini"
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

DEBUG_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a code patch repair assistant.
    Your task is to fix the apply_patch payload so it applies cleanly to the target file.

    Constraints:
    - Output only the patch text (no commentary, no markdown fences).
    - Patch must start with "*** Begin Patch" and end with "*** End Patch".
    - Use exactly one "*** Update File: {file_path}" (no Add/Delete/Move).
    - Only delete lines from the file; do not add new lines.
    - Include 3-10 lines of unchanged context above and below each deletion.
    - The patch must change the file and keep it syntactically valid.
    - Preserve the intent of the current patch while making it apply cleanly.

    Apply error:
    {error}

    Current patch:
    <patch>
    {patch_text}
    </patch>

    Target file content:
    <file>
    {file_content}
    </file>
    """
).strip()

NOOP_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a code patch repair assistant.
    The current patch applies but produces no changes. Create a new patch that
    makes a meaningful change aligned with the task below.

    Constraints:
    - Output only the patch text (no commentary, no markdown fences).
    - Patch must start with "*** Begin Patch" and end with "*** End Patch".
    - Use exactly one "*** Update File: {file_path}" (no Add/Delete/Move).
    - Only delete lines from the file; do not add new lines.
    - Include 3-10 lines of unchanged context above and below each deletion.
    - The patch must change the file and keep it syntactically valid.

    Task (question):
    {problem}

    Expected fix (answer):
    {answer}

    Current patch:
    <patch>
    {patch_text}
    </patch>

    Target file content:
    <file>
    {file_content}
    </file>
    """
).strip()


logger = logging.getLogger(__name__)


@dataclass
class PatchCheckResult:
    ok: bool
    error: Optional[str] = None
    no_op: bool = False


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError(f"Dataset file must contain a JSON list: {path}")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"Dataset file must contain objects: {path}")
    return payload


def _write_json_list(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(list(records), fp, ensure_ascii=False, indent=2)


def _write_patch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Backup destination already exists: {dst}")
        if dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _collect_touched_paths(patch_text: str) -> Iterable[str]:
    markers = (
        "*** Update File:",
        "*** Add File:",
        "*** Delete File:",
        "*** Move to:",
    )
    seen: Dict[str, None] = {}
    for raw_line in patch_text.splitlines():
        stripped = raw_line.strip()
        for marker in markers:
            if stripped.startswith(marker):
                rel_path = stripped[len(marker) :].strip()
                if rel_path and rel_path not in seen:
                    seen[rel_path] = None
                break
    return tuple(seen.keys())


def _resolve_within_workspace(workspace: Path, rel_path: str) -> Path:
    candidate = (workspace / Path(rel_path)).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise AssertionError(f"Patch references a path outside the workspace: {rel_path}") from exc
    return candidate


def _snapshot_files(workspace: Path, rel_paths: Iterable[str]) -> Dict[Path, Optional[bytes]]:
    root = workspace.resolve()
    backups: Dict[Path, Optional[bytes]] = {}
    for rel_path in rel_paths:
        absolute = _resolve_within_workspace(root, rel_path)
        if absolute in backups:
            continue
        backups[absolute] = absolute.read_bytes() if absolute.exists() else None
    return backups


def _restore_files(backups: Dict[Path, Optional[bytes]]) -> None:
    for path, content in backups.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def _patch_changes_anything(backups: Dict[Path, Optional[bytes]]) -> bool:
    for path, before in backups.items():
        after = path.read_bytes() if path.exists() else None
        if before != after:
            return True
    return False


def _extract_patch_text(response: str) -> Optional[str]:
    if not response:
        return None
    start = response.find("*** Begin Patch")
    end = response.rfind("*** End Patch")
    if start == -1 or end == -1:
        return None
    end = end + len("*** End Patch")
    return response[start:end].strip()


def _apply_patch_and_check(
    patch_text: str,
    *,
    workspace: Path,
    expected_file_path: str,
) -> PatchCheckResult:
    try:
        _validate_patch_text(patch_text, expected_file_path)
    except ValueError as exc:
        return PatchCheckResult(ok=False, error=str(exc), no_op=False)

    touched_paths = _collect_touched_paths(patch_text)
    backups: Dict[Path, Optional[bytes]] = {}
    try:
        backups = _snapshot_files(workspace, touched_paths)
        apply_patch_to_workspace(patch_text, workspace)
        changed = _patch_changes_anything(backups)
        if not changed:
            return PatchCheckResult(
                ok=False,
                error="Patch applied but produced no changes.",
                no_op=True,
            )
        return PatchCheckResult(ok=True)
    except (PatchApplyError, PatchParseError) as exc:
        return PatchCheckResult(ok=False, error=str(exc), no_op=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        return PatchCheckResult(
            ok=False,
            error=f"Unexpected error while applying patch: {exc}",
            no_op=False,
        )
    finally:
        if backups:
            _restore_files(backups)


def _prepare_workspace(
    *,
    repo_root: Path,
    workspace_root: Path,
    file_config: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    workspace_root.mkdir(parents=True, exist_ok=True)
    targets = collect_target_files(file_config, repo_root)
    copied: Dict[str, Path] = {}
    for src in targets:
        rel_path = stringify_file_path(src, repo_root)
        dest = workspace_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied[rel_path] = dest
    return copied


def _build_debug_prompt(
    *,
    file_path: str,
    file_content: str,
    patch_text: str,
    error: str,
) -> str:
    return DEBUG_PROMPT_TEMPLATE.format(
        file_path=file_path,
        file_content=file_content,
        patch_text=patch_text,
        error=error,
    )


def _build_noop_prompt(
    *,
    file_path: str,
    file_content: str,
    patch_text: str,
    problem: str,
    answer: str,
) -> str:
    return NOOP_PROMPT_TEMPLATE.format(
        file_path=file_path,
        file_content=file_content,
        patch_text=patch_text,
        problem=problem,
        answer=answer,
    )


async def _debug_patch_files(args: argparse.Namespace) -> None:
    exp_dir = Path(DEFAULT_EXP_DIR)
    patch_dir = Path(args.patch_dir).resolve() if args.patch_dir else exp_dir / "patch_files"
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else exp_dir / "dataset.json"

    if not patch_dir.is_dir():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    backup_patches = exp_dir / "old_patches"
    backup_dataset = exp_dir / "old_dataset.json"
    _copy_tree(patch_dir, backup_patches, overwrite=args.overwrite_backup)
    _copy_tree(dataset_path, backup_dataset, overwrite=args.overwrite_backup)

    repo_root = Path(args.repo_root).resolve()
    if args.file_config_path:
        file_config = load_file_config(Path(args.file_config_path))
    else:
        file_config = DEFAULT_FILE_CONFIG

    workspace_root = Path(args.workspace_dir).resolve() if args.workspace_dir else exp_dir / DEFAULT_WORKSPACE_NAME
    copied_files = _prepare_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
        file_config=file_config,
    )
    if not copied_files:
        raise RuntimeError("No files were copied into the debug workspace.")

    records = _read_json_list(dataset_path)
    if not records:
        logger.warning("Dataset is empty; nothing to debug.")
        return

    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    llm = LLMClient(**llm_kwargs)

    kept_records: List[Dict[str, Any]] = []
    failed_patch_files: List[str] = []

    for index, record in enumerate(records, start=1):
        patch_file = str(record.get("patch_file") or "").strip()
        source_file = str(record.get("source_file") or "").strip()
        if not patch_file or not source_file:
            logger.warning("[%d] Missing patch_file/source_file; dropping record.", index)
            failed_patch_files.append(patch_file)
            continue

        patch_path = patch_dir / patch_file
        if not patch_path.is_file():
            logger.warning("[%d] Patch file missing: %s", index, patch_path)
            failed_patch_files.append(patch_file)
            continue

        workspace_file = workspace_root / source_file
        if not workspace_file.is_file():
            logger.warning("[%d] Source file missing in workspace: %s", index, workspace_file)
            failed_patch_files.append(patch_file)
            continue

        try:
            file_content = workspace_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            file_content = workspace_file.read_text(encoding="utf-8", errors="replace")

        patch_text = patch_path.read_text(encoding="utf-8").strip()
        success = False

        for loop_index in range(DEFAULT_N_DEBUG_LOOP):
            result = _apply_patch_and_check(
                patch_text,
                workspace=workspace_root,
                expected_file_path=source_file,
            )
            if result.ok:
                success = True
                break

            if result.no_op:
                prompt = _build_noop_prompt(
                    file_path=source_file,
                    file_content=file_content,
                    patch_text=patch_text,
                    problem=str(record.get("problem") or "").strip(),
                    answer=str(record.get("answer") or "").strip(),
                )
            else:
                prompt = _build_debug_prompt(
                    file_path=source_file,
                    file_content=file_content,
                    patch_text=patch_text,
                    error=result.error or "Unknown error",
                )

            logger.info(
                "[%d] Debugging %s (%d/%d)",
                index,
                patch_file,
                loop_index + 1,
                DEFAULT_N_DEBUG_LOOP,
            )
            response_text, _ = await llm.chat(messages=[{"role": "user", "content": prompt}])
            extracted = _extract_patch_text(response_text)
            if extracted:
                patch_text = extracted
            else:
                patch_text = response_text.strip()
            if patch_text:
                _write_patch(patch_path, patch_text)

        if success:
            kept_records.append(record)
        else:
            failed_patch_files.append(patch_file)

    for patch_file in failed_patch_files:
        if not patch_file:
            continue
        patch_path = patch_dir / patch_file
        if patch_path.exists():
            patch_path.unlink()

    if failed_patch_files:
        failed_set = {name for name in failed_patch_files if name}
        kept_records = [record for record in kept_records if record.get("patch_file") not in failed_set]

    _write_json_list(dataset_path, kept_records)

    logger.info(
        "Debugging complete. Kept %d/%d records. Removed %d patch files.",
        len(kept_records),
        len(records),
        len({name for name in failed_patch_files if name}),
    )


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug and repair patch files with an LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name for LLM calls.")
    parser.add_argument(
        "--exp-dir",
        default=DEFAULT_EXP_DIR,
        help="Experiment directory containing dataset.json and patch_files.",
    )
    parser.add_argument(
        "--repo-root",
        default=DEFAULT_REPO_ROOT,
        help="Repository root used to resolve file_config paths.",
    )
    parser.add_argument(
        "--file-config-path",
        help="Path to a JSON file containing the file_config list.",
    )
    parser.add_argument(
        "--patch-dir",
        help="Directory containing patch files (defaults to <exp_dir>/patch_files).",
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to dataset.json (defaults to <exp_dir>/dataset.json).",
    )
    parser.add_argument(
        "--workspace-dir",
        help="Directory for copied files used during patch application.",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        default=True,
        help="Overwrite old_patches/old_dataset.json if they already exist.",
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

    asyncio.run(_debug_patch_files(args))


if __name__ == "__main__":
    main()
