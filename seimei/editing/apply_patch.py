from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
ADD_MARKER = "*** Add File: "
DELETE_MARKER = "*** Delete File: "
UPDATE_MARKER = "*** Update File: "
MOVE_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CONTEXT_MARKER = "@@"


class PatchParseError(ValueError):
    """Raised when the apply_patch input cannot be parsed."""


class PatchApplyError(RuntimeError):
    """Raised when a parsed patch cannot be applied safely."""


@dataclass
class UpdateChunk:
    change_context: Optional[str]
    old_lines: List[str]
    new_lines: List[str]
    is_end_of_file: bool = False


@dataclass
class AddFileOperation:
    path: str
    contents: str


@dataclass
class DeleteFileOperation:
    path: str


@dataclass
class UpdateFileOperation:
    path: str
    move_path: Optional[str]
    chunks: List[UpdateChunk]


PatchOperation = Union[AddFileOperation, DeleteFileOperation, UpdateFileOperation]


@dataclass
class PlannedAdd:
    path: Path
    content: str


@dataclass
class PlannedDelete:
    path: Path


@dataclass
class PlannedUpdate:
    source: Path
    destination: Optional[Path]
    content: str


@dataclass
class ApplyResult:
    added: List[Path]
    modified: List[Path]
    deleted: List[Path]


def apply_patch_to_workspace(
    patch_text: str,
    workspace: Path,
    *,
    workdir_hint: Optional[str] = None,
) -> ApplyResult:
    """Parse and apply an apply_patch payload inside *workspace*.

    The workspace path is treated as the root; relative file references that escape this
    directory raise PatchApplyError. The optional *workdir_hint* emulates `cd <path>`
    that precedes an `apply_patch` invocation.
    """

    try:
        return _apply_patch_impl(
            patch_text,
            workspace,
            workdir_hint=workdir_hint,
        )
    except Exception as exc:
        _log_patch_error(exc)
        raise


def _apply_patch_impl(
    patch_text: str,
    workspace: Path,
    *,
    workdir_hint: Optional[str] = None,
) -> ApplyResult:
    operations = _parse_patch(patch_text)
    if not operations:
        raise PatchParseError("patch rejected: empty patch")

    workspace = Path(workspace).expanduser().resolve()
    base_dir = _resolve_workdir(workspace, workdir_hint)

    planned_actions: List[Union[PlannedAdd, PlannedDelete, PlannedUpdate]] = []
    staged_contents: Dict[Path, Optional[str]] = {}
    added_paths: List[Path] = []
    modified_paths: List[Path] = []
    deleted_paths: List[Path] = []

    for op in operations:
        if isinstance(op, AddFileOperation):
            path = _resolve_patch_path(op.path, base_dir, workspace)
            staged_contents[path] = op.contents
            planned_actions.append(PlannedAdd(path=path, content=op.contents))
            added_paths.append(path)
            continue

        if isinstance(op, DeleteFileOperation):
            path = _resolve_patch_path(op.path, base_dir, workspace)
            _ensure_current_file_exists(path, staged_contents)
            staged_contents[path] = None
            planned_actions.append(PlannedDelete(path=path))
            deleted_paths.append(path)
            continue

        if isinstance(op, UpdateFileOperation):
            source = _resolve_patch_path(op.path, base_dir, workspace)
            base_content = _read_current_contents(source, staged_contents)
            new_content = _apply_chunks_to_text(base_content, source, op.chunks)
            destination: Optional[Path] = None
            if op.move_path:
                destination = _resolve_patch_path(op.move_path, base_dir, workspace)
                staged_contents[source] = None
                staged_contents[destination] = new_content
            else:
                staged_contents[source] = new_content
            planned_actions.append(
                PlannedUpdate(source=source, destination=destination, content=new_content)
            )
            modified_paths.append(destination or source)
            continue

        raise PatchParseError("Unsupported patch operation encountered")

    _apply_planned_actions(planned_actions)
    return ApplyResult(added=added_paths, modified=modified_paths, deleted=deleted_paths)


def _parse_patch(text: str) -> List[PatchOperation]:
    stripped = text.strip()
    if not stripped:
        return []
    lines = [line.rstrip("\r") for line in stripped.splitlines()]
    if not lines:
        return []
    if lines[0].strip() != BEGIN_MARKER:
        raise PatchParseError("The first line of the patch must be '*** Begin Patch'")
    if lines[-1].strip() != END_MARKER:
        raise PatchParseError("The last line of the patch must be '*** End Patch'")

    operations: List[PatchOperation] = []
    idx = 1
    line_number = 2
    end_index = len(lines) - 1
    while idx < end_index:
        raw_line = lines[idx]
        trimmed = raw_line.strip()
        if not trimmed:
            idx += 1
            line_number += 1
            continue
        op, consumed = _parse_hunk(lines[idx:end_index], line_number)
        operations.append(op)
        idx += consumed
        line_number += consumed

    return operations


def _parse_hunk(lines: Sequence[str], line_number: int) -> Tuple[PatchOperation, int]:
    first_line = lines[0].strip()
    if first_line.startswith(ADD_MARKER):
        path = first_line[len(ADD_MARKER) :].strip()
        if not path:
            raise PatchParseError("Add File header missing path")
        contents: List[str] = []
        consumed = 1
        for line in lines[1:]:
            if line.startswith("+"):
                contents.append(line[1:])
                consumed += 1
            else:
                break
        add_contents = "\n".join(contents)
        if contents:
            add_contents += "\n"
        return AddFileOperation(path=path, contents=add_contents), consumed

    if first_line.startswith(DELETE_MARKER):
        path = first_line[len(DELETE_MARKER) :].strip()
        if not path:
            raise PatchParseError("Delete File header missing path")
        return DeleteFileOperation(path=path), 1

    if first_line.startswith(UPDATE_MARKER):
        path = first_line[len(UPDATE_MARKER) :].strip()
        if not path:
            raise PatchParseError("Update File header missing path")
        remaining = lines[1:]
        consumed = 1
        move_path: Optional[str] = None
        if remaining:
            move_candidate = remaining[0].strip()
            if move_candidate.startswith(MOVE_MARKER):
                move_path = move_candidate[len(MOVE_MARKER) :].strip()
                consumed += 1
                remaining = remaining[1:]

        chunks: List[UpdateChunk] = []
        while remaining:
            if remaining[0].strip() == "":
                consumed += 1
                remaining = remaining[1:]
                continue
            if remaining[0].startswith("***"):
                break
            chunk, chunk_lines = _parse_update_chunk(
                remaining,
                line_number + consumed,
                allow_missing_context=not chunks,
            )
            if chunk_lines == 0:
                break
            chunks.append(chunk)
            consumed += chunk_lines
            remaining = remaining[chunk_lines:]

        if not chunks:
            raise PatchParseError(
                f"Update file hunk for path '{path}' is empty at line {line_number}"
            )

        return (
            UpdateFileOperation(path=path, move_path=move_path, chunks=chunks),
            consumed,
        )

    raise PatchParseError(
        f"Unrecognized patch header at line {line_number}: {lines[0]}"
    )


def _parse_update_chunk(
    lines: Sequence[str],
    line_number: int,
    *,
    allow_missing_context: bool,
) -> Tuple[UpdateChunk, int]:
    if not lines:
        raise PatchParseError("Update hunk does not contain any lines")

    header = lines[0]
    start_index = 0
    change_context: Optional[str] = None
    if header == EMPTY_CONTEXT_MARKER:
        start_index = 1
    elif header.startswith(CHANGE_CONTEXT_MARKER):
        change_context = header[len(CHANGE_CONTEXT_MARKER) :]
        start_index = 1
    elif not allow_missing_context:
        raise PatchParseError(
            f"Expected update hunk to start with a @@ context marker, got: '{header}'"
        )

    if start_index >= len(lines):
        raise PatchParseError("Update hunk does not contain any lines")

    chunk = UpdateChunk(
        change_context=change_context,
        old_lines=[],
        new_lines=[],
        is_end_of_file=False,
    )
    parsed = 0
    for line in lines[start_index:]:
        stripped = line
        if stripped == EOF_MARKER:
            if parsed == 0:
                raise PatchParseError("Update hunk does not contain any lines")
            chunk.is_end_of_file = True
            parsed += 1
            break
        if not stripped:
            chunk.old_lines.append("")
            chunk.new_lines.append("")
            parsed += 1
            continue
        marker = stripped[0]
        payload = stripped[1:]
        if marker == ' ':
            chunk.old_lines.append(payload)
            chunk.new_lines.append(payload)
        elif marker == '+':
            chunk.new_lines.append(payload)
        elif marker == '-':
            chunk.old_lines.append(payload)
        else:
            if stripped.startswith("***") or stripped.startswith("@@"):
                if parsed == 0:
                    raise PatchParseError(
                        "Unexpected line found in update hunk. Lines must start with ' ', '+', or '-'"
                    )
                break
            # Compatibility fallback: tolerate context lines that are missing the leading
            # single-space marker and treat them as unchanged lines.
            chunk.old_lines.append(stripped)
            chunk.new_lines.append(stripped)
        parsed += 1

    return chunk, parsed + start_index


def _resolve_workdir(workspace: Path, workdir_hint: Optional[str]) -> Path:
    if not workdir_hint:
        return workspace
    candidate = Path(workdir_hint).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    _ensure_within_workspace(candidate, workspace)
    return candidate


def _resolve_patch_path(path_text: str, base_dir: Path, workspace: Path) -> Path:
    if not path_text:
        raise PatchApplyError("Patch path cannot be empty")
    path = Path(path_text)
    if path.is_absolute():
        raise PatchApplyError("File references must be relative paths")
    candidate = (base_dir / path).resolve()
    _ensure_within_workspace(candidate, workspace)
    return candidate


def _ensure_within_workspace(path: Path, workspace: Path) -> None:
    workspace_resolved = workspace.resolve()
    try:
        path.resolve().relative_to(workspace_resolved)
    except ValueError as exc:  # pragma: no cover - guard path traversal
        raise PatchApplyError(
            "patch rejected: writing outside of the project"
        ) from exc


def _ensure_current_file_exists(path: Path, staged: Dict[Path, Optional[str]]) -> None:
    if path in staged:
        if staged[path] is None:
            raise PatchApplyError(f"File {path} no longer exists after earlier operations")
        return
    if not path.exists():
        raise PatchApplyError(f"Failed to delete missing file {path}")
    if path.is_dir():
        raise PatchApplyError(f"Cannot delete directory {path}")


def _read_current_contents(path: Path, staged: Dict[Path, Optional[str]]) -> str:
    if path in staged:
        content = staged[path]
        if content is None:
            raise PatchApplyError(
                f"Cannot update {path} because it was deleted by a previous operation"
            )
        return content
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PatchApplyError(f"Failed to read file to update {path}") from exc
    except OSError as exc:  # pragma: no cover - IO errors
        raise PatchApplyError(f"Failed to read file to update {path}: {exc}") from exc


def _apply_chunks_to_text(base_text: str, path: Path, chunks: Sequence[UpdateChunk]) -> str:
    lines, newline = _split_lines(base_text)
    replacements = _compute_replacements(lines, path, chunks)
    updated_lines = _apply_replacements(lines, replacements)
    if not updated_lines or updated_lines[-1] != "":
        updated_lines.append("")
    return _join_lines(updated_lines, newline)


def _split_lines(text: str) -> Tuple[List[str], str]:
    newline = _detect_line_ending(text)
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines, newline


def _join_lines(lines: List[str], newline: str) -> str:
    if not lines:
        return ""
    return newline.join(lines)


def _detect_line_ending(text: str) -> str:
    for idx, ch in enumerate(text):
        if ch == "\n":
            if idx > 0 and text[idx - 1] == "\r":
                return "\r\n"
            return "\n"
    return "\n"


def _compute_replacements(
    original_lines: List[str],
    path: Path,
    chunks: Sequence[UpdateChunk],
) -> List[Tuple[int, int, List[str]]]:
    replacements: List[Tuple[int, int, List[str]]] = []
    line_index = 0

    for chunk in chunks:
        if chunk.change_context:
            ctx_idx = _seek_sequence(
                original_lines,
                [chunk.change_context],
                line_index,
                eof=False,
            )
            if ctx_idx is None:
                raise PatchApplyError(
                    f"Failed to find context '{chunk.change_context}' in {path}"
                )
            line_index = ctx_idx + 1

        if not chunk.old_lines:
            insertion_idx = (
                len(original_lines) - 1
                if original_lines and original_lines[-1] == ""
                else len(original_lines)
            )
            replacements.append((insertion_idx, 0, list(chunk.new_lines)))
            continue

        pattern = list(chunk.old_lines)
        new_slice = list(chunk.new_lines)
        found = _seek_sequence(
            original_lines,
            pattern,
            line_index,
            eof=chunk.is_end_of_file,
        )

        if found is None and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = _seek_sequence(
                original_lines,
                pattern,
                line_index,
                eof=chunk.is_end_of_file,
            )

        if found is None:
            core_match = _trim_shared_context_edges(pattern, new_slice)
            if core_match is not None:
                core_old, core_new = core_match
                found = _seek_sequence(
                    original_lines,
                    core_old,
                    line_index,
                    eof=chunk.is_end_of_file,
                )
                if found is not None:
                    pattern = core_old
                    new_slice = core_new

        if found is None:
            raise PatchApplyError(
                f"Failed to find expected lines in {path}:\n" + "\n".join(chunk.old_lines)
            )

        replacements.append((found, len(pattern), new_slice))
        line_index = found + len(pattern)

    replacements.sort(key=lambda item: item[0])
    return replacements


def _trim_shared_context_edges(
    old_lines: Sequence[str],
    new_lines: Sequence[str],
) -> Optional[Tuple[List[str], List[str]]]:
    if not old_lines or not new_lines:
        return None

    max_common = min(len(old_lines), len(new_lines))
    prefix = 0
    while prefix < max_common and old_lines[prefix] == new_lines[prefix]:
        prefix += 1

    suffix = 0
    while suffix < (max_common - prefix):
        if old_lines[len(old_lines) - 1 - suffix] != new_lines[len(new_lines) - 1 - suffix]:
            break
        suffix += 1

    if prefix == 0 and suffix == 0:
        return None

    old_end = len(old_lines) - suffix if suffix else len(old_lines)
    new_end = len(new_lines) - suffix if suffix else len(new_lines)
    core_old = list(old_lines[prefix:old_end])
    core_new = list(new_lines[prefix:new_end])
    if not core_old:
        return None
    return core_old, core_new


def _apply_replacements(
    original_lines: List[str],
    replacements: Sequence[Tuple[int, int, List[str]]],
) -> List[str]:
    result = list(original_lines)
    for start, old_len, new_segment in reversed(replacements):
        end = start + old_len
        result[start:end] = list(new_segment)
    return result


def _seek_sequence(
    lines: Sequence[str],
    pattern: Sequence[str],
    start: int,
    *,
    eof: bool,
) -> Optional[int]:
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    max_start = len(lines) - len(pattern)
    search_start = max_start if eof and max_start >= 0 else start
    search_start = max(0, min(search_start, max_start))

    def _scan(transform) -> Optional[int]:
        for idx in range(search_start, max_start + 1):
            if all(transform(lines[idx + offset]) == transform(pattern[offset]) for offset in range(len(pattern))):
                return idx
        return None

    for transform in (
        lambda s: s,
        lambda s: s.rstrip(),
        lambda s: s.strip(),
        _normalize_unicode,
    ):
        match = _scan(transform)
        if match is not None:
            return match
    return None


def _normalize_unicode(text: str) -> str:
    replacements = {
        ord("\u2010"): "-",
        ord("\u2011"): "-",
        ord("\u2012"): "-",
        ord("\u2013"): "-",
        ord("\u2014"): "-",
        ord("\u2015"): "-",
        ord("\u2212"): "-",
        ord("\u2018"): "'",
        ord("\u2019"): "'",
        ord("\u201a"): "'",
        ord("\u201b"): "'",
        ord("\u201c"): '"',
        ord("\u201d"): '"',
        ord("\u201e"): '"',
        ord("\u201f"): '"',
        ord("\u00a0"): " ",
        ord("\u2002"): " ",
        ord("\u2003"): " ",
        ord("\u2004"): " ",
        ord("\u2005"): " ",
        ord("\u2006"): " ",
        ord("\u2007"): " ",
        ord("\u2008"): " ",
        ord("\u2009"): " ",
        ord("\u200a"): " ",
        ord("\u202f"): " ",
        ord("\u205f"): " ",
        ord("\u3000"): " ",
    }
    return text.strip().translate(replacements)


def _log_patch_error(exc: Exception) -> None:
    """Emit a readable error summary before re-raising patch exceptions."""
    print("[apply_patch] Patch application failed:", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def _apply_planned_actions(actions: Iterable[Union[PlannedAdd, PlannedDelete, PlannedUpdate]]) -> None:
    for action in actions:
        if isinstance(action, PlannedAdd):
            _write_text(action.path, action.content)
            continue
        if isinstance(action, PlannedDelete):
            _safe_unlink(action.path)
            continue
        if isinstance(action, PlannedUpdate):
            target = action.destination or action.source
            _write_text(target, action.content)
            if action.destination and action.destination != action.source:
                _safe_unlink(action.source)
            continue


def _write_text(path: Path, content: str) -> None:
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem specific
        raise PatchApplyError(f"Failed to write file {path}: {exc}") from exc


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=False)
    except FileNotFoundError:
        raise PatchApplyError(f"Failed to remove {path}: file does not exist")
    except OSError as exc:  # pragma: no cover
        raise PatchApplyError(f"Failed to remove {path}: {exc}") from exc
