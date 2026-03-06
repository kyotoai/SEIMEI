from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    "*** Update File: relative/path\n"
    "@@\n"
    "<context line 1>\n"
    "<context line 2>\n"
    "...\n"
    "<context line M1>\n"
    "-<old line 1>\n"
    "-<old line 2>\n"
    "...\n"
    "-<old line N1>\n"
    "+<new line 1>\n"
    "+<new line 2>\n"
    "...\n"
    "+<new line N2>\n"
    "<context line 1>\n"
    "<context line 2>\n"
    "...\n"
    "<context line M2>\n"
    "*** End Patch\n\n"
    "Always include some context lines so that the part you modify will be identical in the file."
)


@dataclass
class PatchRequest:
    patch: str
    workdir_hint: Optional[str]
    raw_message: str


def _last_patch_match(text: str) -> Optional[re.Match[str]]:
    match = None
    for match in _PATCH_BLOCK_RE.finditer(text):
        pass
    return match


def _extract_patch_request(messages: Sequence[Dict[str, Any]]) -> Optional[PatchRequest]:
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        match = _last_patch_match(content)
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


def _knowledge_hint_block(entries: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in entries:
        text = str(entry.get("text") or entry.get("knowledge") or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


async def _generate_patch_with_llm(
    llm: Any,
    messages: Sequence[Dict[str, Any]],
    knowledge_entries: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str], str, Optional[str]]:
    system_lines = [
        "You are the edit_file agent in a coding workflow.",
        "Generate a single valid apply_patch payload that implements the requested edit.",
        "Use only relative file paths in the workspace.",
        "Output only the patch text and nothing else (no markdown fences, no commentary).",
        APPLY_PATCH_FORMAT_HINT,
    ]
    knowledge_block = _knowledge_hint_block(knowledge_entries)
    if knowledge_block:
        system_lines.append("Relevant edit heuristics:\n" + knowledge_block)
    system_prompt = "\n\n".join(system_lines)

    try:
        llm_output, _ = await llm.chat(
            messages=list(messages),
            system=system_prompt,
        )
    except Exception as exc:
        return None, None, None, system_prompt, f"LLM patch generation failed: {exc}"

    raw_output = (llm_output or "").strip()
    if not raw_output:
        return None, "", None, system_prompt, "LLM patch generation returned empty output."

    match = _last_patch_match(raw_output)
    if not match:
        return None, raw_output, _extract_cd_hint(raw_output), system_prompt, "LLM output did not contain an apply_patch block."

    patch_text = match.group(0).strip()
    workdir_hint = _extract_cd_hint(raw_output[: match.start()])
    return patch_text, raw_output, workdir_hint, system_prompt, None


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
        llm_prompt: Optional[str] = kwargs.get("llm_prompt")
        llm_error: Optional[str] = None
        if raw_llm_output is None:
            raw_llm_output = _latest_assistant_message(messages)

        if not patch_text:
            request = _extract_patch_request(messages)
            if request:
                patch_text = request.patch
                workdir_hint = workdir_hint or request.workdir_hint
                raw_llm_output = raw_llm_output or request.raw_message

        if not patch_text:
            llm = shared_ctx.get("llm")
            if llm is None:
                llm_error = "LLM unavailable for patch generation."
            else:
                knowledge_entries = await self.get_agent_knowledge(max_items=6)
                (
                    generated_patch,
                    generated_output,
                    generated_workdir_hint,
                    llm_prompt,
                    llm_error,
                ) = await _generate_patch_with_llm(llm, messages, knowledge_entries)
                if generated_output is not None:
                    raw_llm_output = generated_output
                if generated_patch:
                    patch_text = generated_patch
                if generated_workdir_hint:
                    workdir_hint = workdir_hint or generated_workdir_hint

        if not patch_text:
            message = (
                "No apply_patch payload detected. Provide a patch enclosed in '*** Begin Patch'"
                " and '*** End Patch', wrapped inside `<parameter=patch>...</parameter>`."
                f"\n\n{APPLY_PATCH_FORMAT_HINT}"
            )
            log_data: Dict[str, Any] = {}
            if raw_llm_output:
                log_data["llm_output"] = _trim_text(raw_llm_output, 2000)
            if llm_prompt:
                log_data["llm_prompt"] = _trim_text(llm_prompt, 2000)
            if llm_error:
                log_data["llm_error"] = llm_error
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
            if llm_prompt:
                log_data["llm_prompt"] = _trim_text(llm_prompt, 2000)
            if llm_error:
                log_data["llm_error"] = llm_error
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
            if llm_prompt:
                log_data["llm_prompt"] = _trim_text(llm_prompt, 2000)
            if llm_error:
                log_data["llm_error"] = llm_error
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
        if llm_prompt:
            log_data["llm_prompt"] = _trim_text(llm_prompt, 2000)
        if llm_error:
            log_data["llm_error"] = llm_error

        return {
            "content": "\n".join(summary_lines),
            "log": log_data,
        }
