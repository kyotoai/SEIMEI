from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pytest

from seimei.editing import PatchApplyError, PatchParseError, apply_patch_to_workspace

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PATCH_DIR = THIS_DIR / "patch_files"
PATCH_FILES = sorted(PATCH_DIR.glob("*.txt"))

if not PATCH_FILES:
    pytestmark = pytest.mark.skip("No patch files found in exp11_plasma_gkv_v3/patch_files.")


@pytest.mark.parametrize("patch_path", PATCH_FILES, ids=lambda path: path.name)
def test_patch_files_apply_cleanly(patch_path: Path) -> None:
    patch_text = patch_path.read_text(encoding="utf-8")
    touched_paths = _collect_touched_paths(patch_text)
    backups = _snapshot_files(REPO_ROOT, touched_paths)
    try:
        apply_patch_to_workspace(patch_text, REPO_ROOT)
    except (PatchApplyError, PatchParseError) as exc:
        print(f"Failed to apply patch {patch_path.relative_to(REPO_ROOT)}: {exc}")
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Unexpected error while applying {patch_path.relative_to(REPO_ROOT)}: {exc}")
        raise
    finally:
        _restore_files(backups)


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


def _snapshot_files(workspace: Path, rel_paths: Iterable[str]) -> Dict[Path, Optional[bytes]]:
    root = workspace.resolve()
    backups: Dict[Path, Optional[bytes]] = {}
    for rel_path in rel_paths:
        absolute = _resolve_within_workspace(root, rel_path)
        if absolute in backups:
            continue
        backups[absolute] = absolute.read_bytes() if absolute.exists() else None
    return backups


def _resolve_within_workspace(workspace: Path, rel_path: str) -> Path:
    candidate = (workspace / Path(rel_path)).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:  # pragma: no cover - mirrors apply_patch guardrails
        raise AssertionError(f"Patch references a path outside the workspace: {rel_path}") from exc
    return candidate


def _restore_files(backups: Dict[Path, Optional[bytes]]) -> None:
    for path, content in backups.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
