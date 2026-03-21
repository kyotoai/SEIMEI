from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from seimei.agent import Agent, register
from seimei.knowledge.utils import prepare_knowledge_payload

_SAFE_DEFAULTS = [
    "python3",
    "pip3",
    "ls",
    "cat",
    "pwd",
    "whoami",
    "dir",
    "type",
    "head",
    "tail",
    "grep",
    "rg",
    "sed",
]


@register
class code_act(Agent):
    """Execute *whitelisted* shell commands chosen from the conversation."""

    description = "Safely run whitelisted shell commands and return stdout/stderr."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        timeout: int = 120,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        allow = bool(shared_ctx.get("allow_code_exec", False))
        allowed: Optional[Sequence[str]] = shared_ctx.get("allowed_commands") or _SAFE_DEFAULTS
        approve_cb = shared_ctx.get("approval_callback")
        cwd: Optional[str] = None
        workspace = shared_ctx.get("workspace")
        if workspace not in (None, ""):
            try:
                cwd = str(Path(workspace).expanduser())
            except Exception:
                cwd = None

        
        knowledge_entries = await self.get_agent_knowledge(max_items=3)
        code, knowledge_used = await _generate_command(messages, shared_ctx, allowed, knowledge_entries)
        knowledge_payload, knowledge_log_texts, knowledge_ids = prepare_knowledge_payload(knowledge_used)

        if not code:
            resp: Dict[str, Any] = {
                "content": (
                    "No executable command detected or generated. "
                    "Provide a command wrapped in <cmd>...</cmd> or a clear analysis request."
                )
            }
            if knowledge_payload:
                resp["knowledge"] = knowledge_payload
            if knowledge_ids:
                resp["knowledge_id"] = knowledge_ids
            return resp

        '''
        code = _normalize_command(code)
        if not code:
            return {"content": "Unable to form a valid command from the request."}
        cmd0 = code.strip().split()[0] if code.strip() else ""
        if allowed and cmd0 not in allowed:
            return {"content": f"Command '{cmd0}' is not in allowlist: {list(allowed)}", "code": code}

        if not allow:
            return {"content": "Execution disabled. (set allow_code_exec=True to enable)", "code": code}

        if approve_cb and not approve_cb(code):
            return {"content": "Execution denied by approval callback.", "code": code}
        '''

        normalized_code = _normalize_command(code)
        if not normalized_code:
            return {"content": "Unable to form a valid command from the request."}
        code = normalized_code
        python_heredoc = _parse_python_heredoc(code)

        '''
        print("\n---------- run command -----------")
        print(_run_command(
                    code="pwd",
                    timeout=timeout,
                    python_heredoc=None,
                    cwd=cwd,
                ))
        
        print(_run_command(
                    code="ls",
                    timeout=timeout,
                    python_heredoc=None,
                    cwd=cwd,
                ))
        '''

        try:
            loop = asyncio.get_event_loop()
            proc_out = await loop.run_in_executor(
                None,
                lambda: _run_command(
                    code=code,
                    timeout=timeout,
                    python_heredoc=python_heredoc,
                    cwd=cwd,
                ),
            )
        except subprocess.TimeoutExpired:
            return {"content": f"Timed out after {timeout}s.", "code": code}

        stdout = (proc_out.stdout or "").strip()
        stderr = (proc_out.stderr or "").strip()
        rc = proc_out.returncode
        is_python_command = bool(python_heredoc) or _is_python_command_text(code)
        command_for_content = _summarize_python_command(code) if is_python_command else code
        summary = f"$ {command_for_content}\n[exit {rc}]\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        log_data: Dict[str, Any] = {"command": code, "output": stdout, "error": stderr, "knowledge_used":knowledge_used}
        if knowledge_log_texts:
            log_data["knowledge"] = knowledge_log_texts
        result: Dict[str, Any] = {"content": summary, "code": code, "log": log_data, "knowledge_used":knowledge_used}
        if knowledge_payload:
            result["knowledge"] = knowledge_payload
        if knowledge_ids:
            result["knowledge_id"] = knowledge_ids
        return result


_CMD_TAG_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL | re.IGNORECASE)
_PY_HEREDOC_HEADER_RE = re.compile(
    r"^(python3?|python)(?:\s+(?P<dash>-))?\s*<<(?P<quote>['\"]?)(?P<tag>[A-Za-z0-9_]+)(?P=quote)\s*$"
)
_PYTHON_CMD_PREFIX_RE = re.compile(r"^(?:python3?|/usr/bin/python3?)\b")


class _PythonHeredoc(NamedTuple):
    executable: str
    script: str
    has_dash: bool


def _extract_tagged_command(text: str) -> Optional[str]:
    if not text:
        return None
    last_match: Optional[re.Match[str]] = None
    for match in _CMD_TAG_RE.finditer(text):
        last_match = match
    if not last_match:
        return None
    command = last_match.group(1).strip()
    return command or None


def _parse_python_heredoc(command: str) -> Optional[_PythonHeredoc]:
    if not command:
        return None
    lines = command.splitlines()
    if not lines:
        return None
    header = lines[0].strip()
    match = _PY_HEREDOC_HEADER_RE.match(header)
    if not match:
        return None
    delimiter = match.group("tag")
    executable = match.group(1)
    script_lines: List[str] = []
    closing_index: Optional[int] = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == delimiter:
            closing_index = idx
            break
        script_lines.append(line)
    if closing_index is None:
        return None
    trailing_lines = lines[closing_index + 1 :]
    if any(line.strip() for line in trailing_lines):
        return None
    script = "\n".join(script_lines)
    has_dash = bool(match.group("dash"))
    return _PythonHeredoc(executable=executable, script=script, has_dash=has_dash)


def _is_python_command_text(command: str) -> bool:
    if not command:
        return False
    normalized = command.lstrip()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0]
    return bool(_PYTHON_CMD_PREFIX_RE.match(first_line))


def _summarize_python_command(command: str) -> str:
    normalized = command.lstrip()
    first_line = normalized.splitlines()[0] if normalized else "python"
    return f"{first_line} (python script omitted)"


def _run_command(
    *,
    code: str,
    timeout: int,
    python_heredoc: Optional[_PythonHeredoc],
    cwd: Optional[str],
) -> subprocess.CompletedProcess[str]:
    if python_heredoc:
        args: List[str] = [python_heredoc.executable]
        if python_heredoc.has_dash:
            args.append("-")
        script_input = python_heredoc.script
        if script_input and not script_input.endswith("\n"):
            script_input += "\n"
        return subprocess.run(
            args,
            input=script_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    return subprocess.run(
        code,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def _normalize_command(command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return ""

    if "\n" in cmd:
        lines = cmd.splitlines()
        first_line = lines[0].strip()
        body_lines = lines[1:]
        if first_line in {"python", "python3"} and body_lines and "<<" not in first_line:
            body = "\n".join(body_lines).lstrip("\n")
            if body.strip():
                return f"{first_line} - <<'PY'\n{body}\nPY"
            return first_line
    return cmd


async def _generate_command(
    messages: List[Dict[str, Any]],
    shared_ctx: Dict[str, Any],
    allowed: Optional[Sequence[str]],
    knowledge_entries: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    llm = shared_ctx.get("llm")
    if llm is None:
        return None, knowledge_entries

    chat_history = messages

    allowed_list = list(allowed) if allowed else []
    allowed_hint = ", ".join(allowed_list) if allowed_list else "python, python3"

    knowledge_hint = "\n".join(f"- {item['text']}" for item in knowledge_entries)

    system_lines = [
        "You turn user analysis requests into one safe POSIX shell command.",
        f"Only use commands that start with: {allowed_hint}.",
        "Output exactly one command only. Do not chain commands with `;`, `&&`, or `||`.",
        "Wrap the command in `<cmd>` and `</cmd>` with nothing else before or after.",
        "Treat user messages as instructions and tool messages as context from earlier command outputs.",

        "Use this default workflow: `ls` for folder/file meta info, then `cat` for file content, then `rg` only if keyword or identifier search is needed.",
        "For folder or file meta analysis, use `ls`.",
        "For file content, use `cat`.",
        "Use `cat -n` when line numbers are needed, especially before editing or reasoning about specific lines.",
        "Use `rg` only for searching keywords, variable names, class names, function names, or other identifiers.",
        "Do not use `rg` for simple file viewing.",
        "Use Python only in special cases where shell commands are not enough.",
        "For PDF files, use Python and call `seimei.agents.utils.view_pdf_text`.",
        "When reading a PDF, print at least 2000 characters if available.",
        "If Python is needed, keep it minimal and use `python3 - <<'PY'` ... `PY`.",
        "The command inside `<cmd>` must include everything needed, including heredoc markers.",
        "Always produce the shortest command that still shows enough evidence for the task.",
    ]

    '''
    system_lines = [
        "You translate user analysis requests into a single safe POSIX shell command.",
        f"Only use commands that start with: {allowed_hint}.",
        "Always output exactly one command. Never chain multiple commands with `;`, `&&`, or `||`.",
        "For directory listing, use `ls` (not `ls -la` or other verbose flags) unless the user explicitly asks for details.",
        "For code/file analysis, strongly prefer shell-native inspection commands: `ls`, `cat -n`, or `sed`.",
        "Use Python only as a last resort when the task cannot be solved with shell commands.",
        "When the target is a PDF file, do not use `cat`; use Python and call `seimei.agents.utils.view_pdf_text`.",
        "Preferred PDF command pattern: `python3 - <<'PY'` then `from seimei.agents.utils import view_pdf_text` and `print(view_pdf_text('path/to/file.pdf'))`, ending with `PY`.",
        "Before editing or reasoning about specific code lines, see meta info by `ls` and capture context with `cat -n` (or equivalent line-numbered output).",
        "Use `rg` only when searching variable names, class names, function names, or other identifiers across the codebase.",
        "Do not use `rg` for simple single-file viewing or generic text inspection where `cat -n`/`sed` is sufficient.",
        "Use `sed` for focused line-range extraction whenever possible.",
        "If Python is truly required, keep it minimal and emit multi-line Python via `python - <<'PY'` ... `PY`.",
        "Wrap the command in `<cmd>` and `</cmd>`. Output nothing before or after the tags.",
        "Ensure the command text inside `<cmd>` contains everything needed, including any heredoc markers.",
        "Your entire goal is to produce the shortest viable command that inspects the mentioned file and "
        "reports the necessary evidence.",
        "Treat user messages as instructions and tool messages as prior command outputs for context.",
    ]
    '''

    if knowledge_hint:
        system_lines.append("Relevant knowledge:\n" + knowledge_hint)

    try:
        response, _ = await llm.chat(
            messages=chat_history,
            system="\n".join(system_lines),
        )
    except Exception:
        return None, knowledge_entries

    command = _extract_tagged_command(response)

    '''
    if not command:
        return None
    if allowed_list:
        token = command.split()[0] if command.split() else ""
        if token not in allowed_list:
            return None
    '''

    return command, knowledge_entries
