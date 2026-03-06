from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from seimei.agent import Agent, register
from seimei.editing import ApplyResult, PatchApplyError, PatchParseError, apply_patch_to_workspace


_PATCH_BLOCK_RE = re.compile(r"(?s)\*\*\* Begin Patch.*?\*\*\* End Patch")
_CD_PATTERN = re.compile(
    r"cd\s+(?P<target>(?:\"[^\"]+\"|'[^']+'|[^&\n]+?))\s*&&\s*apply_patch",
    re.IGNORECASE,
)


APPLY_PATCH_FORMAT_HINT = (
    "Ensure your response is a valid apply_patch payload. The body must follow the grammar:\n"
    "*** Begin Patch\n"
    "*** (Add|Delete|Update) File: relative/path\n"
    "@@ <line context or number>\n"
    "- old line\n+ new line\n"
    "*** End Patch\n\n"
    "Always include @@ headers that reference the actual location (line numbers or a nearby symbol)"
    " and provide change hunks with leading +/−/space markers plus surrounding context so patches can"
    " be applied deterministically."
)


@dataclass
class PatchRequest:
    patch: str
    workdir_hint: Optional[str]
    raw_message: str


def _extract_patch_request(messages: Sequence[Dict[str, Any]]) -> Optional[PatchRequest]:
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        match = None
        for match in _PATCH_BLOCK_RE.finditer(content):
            pass
        if not match:
            continue
        patch_text = match.group(0).strip()
        workdir_hint = _extract_cd_hint(content[: match.start()])
        return PatchRequest(patch=patch_text, workdir_hint=workdir_hint, raw_message=content)
    return None


def _latest_assistant_message(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        role = str(msg.get("role") or "").lower()
        if role != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _extract_cd_hint(prefix: str) -> Optional[str]:
    match = None
    for match in _CD_PATTERN.finditer(prefix):
        pass
    if not match:
        return None
    raw_path = match.group("target").strip()
    if raw_path.startswith(("'", '"')) and raw_path.endswith(("'", '"')):
        raw_path = raw_path[1:-1]
    return raw_path or None


def _determine_workspace(shared_ctx: Dict[str, Any]) -> Path:
    for key in ("workspace_dir", "workspace", "cwd"):
        raw = shared_ctx.get(key)
        if not raw:
            continue
        try:
            return Path(raw).expanduser().resolve()
        except (TypeError, ValueError):
            continue
    return Path.cwd()


def _relativize(paths: Sequence[Path], workspace: Path) -> List[str]:
    rel_paths: List[str] = []
    for path in paths:
        try:
            rel_paths.append(str(path.resolve().relative_to(workspace)))
        except ValueError:
            rel_paths.append(str(path))
    return rel_paths


def _summarize(result: ApplyResult, workspace: Path) -> List[str]:
    lines = ["Success. Updated the following files:"]
    added = _relativize(result.added, workspace)
    modified = _relativize(result.modified, workspace)
    deleted = _relativize(result.deleted, workspace)
    if not any((added, modified, deleted)):
        lines.append("(no filesystem changes)")
        return lines
    for path in added:
        lines.append(f"A {path}")
    for path in modified:
        lines.append(f"M {path}")
    for path in deleted:
        lines.append(f"D {path}")
    return lines


def _trim_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


@register
class edit_file(Agent):
    """Apply apply_patch-formatted edits to the local workspace."""

    description = "Edits files by applying Codex-style apply_patch payloads to the workspace."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        patch_text: Optional[str] = kwargs.get("patch")
        workdir_hint: Optional[str] = kwargs.get("workdir")
        raw_llm_output: Optional[str] = kwargs.get("llm_output")
        if raw_llm_output is None:
            raw_llm_output = _latest_assistant_message(messages)

        if not patch_text:
            request = _extract_patch_request(messages)
            if request:
                patch_text = request.patch
                workdir_hint = workdir_hint or request.workdir_hint
                raw_llm_output = raw_llm_output or request.raw_message

        if not patch_text:
            message = (
                "No apply_patch payload detected. Provide a patch enclosed in '*** Begin Patch'"
                " and '*** End Patch', wrapped inside `<parameter=patch>...</parameter>`."
                f"\n\n{APPLY_PATCH_FORMAT_HINT}"
            )
            log_data: Dict[str, Any] = {}
            if raw_llm_output:
                log_data["llm_output"] = _trim_text(raw_llm_output, 2000)
            response: Dict[str, Any] = {"content": message}
            if log_data:
                response["log"] = log_data
            return response

        workspace = _determine_workspace(shared_ctx)

        try:
            result = apply_patch_to_workspace(
                patch_text,
                workspace,
                workdir_hint=workdir_hint,
            )
        except PatchParseError as exc:
            log_data = {
                "error": "patch_parse_error",
                "exception": str(exc),
                "patch_preview": _trim_text(patch_text, 2000),
            }
            if raw_llm_output:
                log_data["llm_output"] = _trim_text(raw_llm_output, 2000)
            return {
                "content": f"Patch parse error: {exc}\n\n{APPLY_PATCH_FORMAT_HINT}",
                "log": log_data,
            }
        except PatchApplyError as exc:
            log_data = {
                "error": "patch_apply_error",
                "exception": str(exc),
                "patch_preview": _trim_text(patch_text, 2000),
            }
            if raw_llm_output:
                log_data["llm_output"] = _trim_text(raw_llm_output, 2000)
            return {
                "content": f"Failed to apply patch: {exc}\n\n{APPLY_PATCH_FORMAT_HINT}",
                "log": log_data,
            }

        summary_lines = _summarize(result, workspace)
        log_data = {
            "workspace": str(workspace),
            "workdir_hint": workdir_hint,
            "added": _relativize(result.added, workspace),
            "modified": _relativize(result.modified, workspace),
            "deleted": _relativize(result.deleted, workspace),
            "llm_output": raw_llm_output or patch_text,
            "patch": patch_text,
            "patch_preview": _trim_text(patch_text, 2000),
        }

        return {
            "content": "\n".join(summary_lines),
            "log": log_data,
        }
