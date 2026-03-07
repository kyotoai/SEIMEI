from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
UPDATE_MARKER = "*** Update File: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CONTEXT_MARKER = "@@"
LINE_RANGE_HEADER_RE = re.compile(r"^@@\s*(\d+)(?:-(\d+))?\s*$")


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
class LineRangeChunk:
    start_line: int
    end_line: Optional[int]
    new_lines: List[str]


@dataclass
class UpdateFileOperation:
    path: str
    chunks: List[Union[LineRangeChunk, UpdateChunk]]


PatchOperation = UpdateFileOperation


@dataclass
class PlannedUpdate:
    path: Path
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

    planned_actions: List[PlannedUpdate] = []
    staged_contents: Dict[Path, Optional[str]] = {}
    modified_paths: List[Path] = []

    for op in operations:
        source = _resolve_patch_path(op.path, base_dir, workspace)
        base_content = _read_current_contents(source, staged_contents)
        new_content = _apply_chunks_to_text(base_content, source, op.chunks)
        staged_contents[source] = new_content
        planned_actions.append(
            PlannedUpdate(path=source, content=new_content)
        )
        modified_paths.append(source)

    _apply_planned_actions(planned_actions)
    return ApplyResult(added=[], modified=modified_paths, deleted=[])


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
    if first_line.startswith(UPDATE_MARKER):
        path = first_line[len(UPDATE_MARKER) :].strip()
        if not path:
            raise PatchParseError("Update File header missing path")
        remaining = lines[1:]
        consumed = 1

        chunks: List[Union[LineRangeChunk, UpdateChunk]] = []
        has_line_range_chunk = False
        has_legacy_chunk = False
        while remaining:
            if remaining[0].strip() == "":
                consumed += 1
                remaining = remaining[1:]
                continue
            if remaining[0].startswith("***"):
                break
            chunk, chunk_lines = _parse_update_chunk(remaining, line_number + consumed, allow_missing_context=not chunks)
            if chunk_lines == 0:
                break
            if isinstance(chunk, LineRangeChunk):
                has_line_range_chunk = True
            else:
                has_legacy_chunk = True
            chunks.append(chunk)
            consumed += chunk_lines
            remaining = remaining[chunk_lines:]

        if not chunks:
            raise PatchParseError(
                f"Update file hunk for path '{path}' is empty at line {line_number}"
            )
        if has_line_range_chunk and has_legacy_chunk:
            raise PatchParseError(
                f"Update file hunk for path '{path}' mixes line-range and legacy diff chunks."
            )

        return (
            UpdateFileOperation(path=path, chunks=chunks),
            consumed,
        )

    if first_line.startswith("***"):
        raise PatchParseError(
            "Only '*** Update File:' operations are supported in this patch format."
        )

    raise PatchParseError(
        f"Unrecognized patch header at line {line_number}: {lines[0]}"
    )


def _parse_update_chunk(
    lines: Sequence[str],
    line_number: int,
    *,
    allow_missing_context: bool,
) -> Tuple[Union[LineRangeChunk, UpdateChunk], int]:
    if not lines:
        raise PatchParseError("Update hunk does not contain any lines")

    header = lines[0].strip()
    range_match = LINE_RANGE_HEADER_RE.match(header)
    if range_match:
        start_line = int(range_match.group(1))
        end_group = range_match.group(2)
        end_line = int(end_group) if end_group is not None else None
        if end_line is not None and end_line < start_line:
            raise PatchParseError(
                f"Invalid line range at line {line_number}: {start_line}-{end_line}"
            )
        new_lines, consumed = _parse_line_range_chunk_body(lines[1:])
        chunk = LineRangeChunk(
            start_line=start_line,
            end_line=end_line,
            new_lines=new_lines,
        )
        return chunk, consumed + 1

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


def _parse_line_range_chunk_body(lines: Sequence[str]) -> Tuple[List[str], int]:
    body: List[str] = []
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("***") or stripped.startswith("@@"):
            break
        body.append(line)
        consumed += 1
    return body, consumed


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


def _apply_chunks_to_text(
    base_text: str,
    path: Path,
    chunks: Sequence[Union[LineRangeChunk, UpdateChunk]],
) -> str:
    lines, newline = _split_lines(base_text)
    if chunks and isinstance(chunks[0], LineRangeChunk):
        range_chunks = [chunk for chunk in chunks if isinstance(chunk, LineRangeChunk)]
        replacements = _compute_line_range_replacements(lines, path, range_chunks)
    else:
        legacy_chunks = [chunk for chunk in chunks if isinstance(chunk, UpdateChunk)]
        replacements = _compute_replacements(lines, path, legacy_chunks)
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


def _compute_line_range_replacements(
    original_lines: List[str],
    path: Path,
    chunks: Sequence[LineRangeChunk],
) -> List[Tuple[int, int, List[str]]]:
    replacements: List[Tuple[int, int, List[str]]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk.start_line < 1:
            raise PatchApplyError(
                f"Invalid line number in {path}: line numbers must be >= 1 (got {chunk.start_line})."
            )

        if chunk.end_line is None:
            if not chunk.new_lines:
                raise PatchApplyError(
                    f"Invalid insertion chunk #{chunk_index} in {path}: '@@{chunk.start_line}' requires inserted text."
                )
            if chunk.start_line > len(original_lines) + 1:
                raise PatchApplyError(
                    f"Insertion line {chunk.start_line} is out of range for {path} (max {len(original_lines) + 1})."
                )
            start_index = chunk.start_line - 1
            replacements.append((start_index, 0, list(chunk.new_lines)))
            continue

        end_line = chunk.end_line
        if end_line is None:
            raise PatchApplyError("Unexpected empty end_line while computing replacements.")
        if chunk.start_line > len(original_lines):
            raise PatchApplyError(
                f"Line range {chunk.start_line}-{end_line} is out of range for {path} (max {len(original_lines)})."
            )
        if end_line > len(original_lines):
            raise PatchApplyError(
                f"Line range {chunk.start_line}-{end_line} is out of range for {path} (max {len(original_lines)})."
            )
        start_index = chunk.start_line - 1
        delete_len = end_line - chunk.start_line + 1
        replacements.append((start_index, delete_len, list(chunk.new_lines)))

    _validate_non_overlapping_replacements(path, replacements)
    replacements.sort(key=lambda item: item[0])
    return replacements


def _validate_non_overlapping_replacements(
    path: Path,
    replacements: Sequence[Tuple[int, int, List[str]]],
) -> None:
    ordered = sorted(enumerate(replacements), key=lambda item: (item[1][0], item[0]))
    occupied: List[Tuple[int, int]] = []
    for _, (start, old_len, _) in ordered:
        end = start + old_len
        if old_len > 0:
            for occ_start, occ_end in occupied:
                if start < occ_end and occ_start < end:
                    raise PatchApplyError(
                        f"Overlapping line ranges detected in patch for {path}."
                    )
            occupied.append((start, end))
            continue
        for occ_start, occ_end in occupied:
            if occ_start < start < occ_end:
                raise PatchApplyError(
                    f"Insertion line falls inside a replaced range in patch for {path}."
                )


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


def _apply_planned_actions(actions: Iterable[PlannedUpdate]) -> None:
    for action in actions:
        _write_text(action.path, action.content)


def _write_text(path: Path, content: str) -> None:
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem specific
        raise PatchApplyError(f"Failed to write file {path}: {exc}") from exc
