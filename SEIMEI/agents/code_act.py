from __future__ import annotations

import asyncio
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from seimei.agent import Agent, register
from seimei.knowledge.utils import get_agent_knowledge

_SAFE_DEFAULTS = [
    "echo",
    "python",
    "python3",
    "pip",
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
    "awk",
    "cut",
    "wc",
    "jq",
    "xlsx2csv",
]


@register
class code_act(Agent):
    """Execute *whitelisted* shell commands chosen from the conversation."""

    description = "Safely run whitelisted shell commands and return stdout/stderr."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        timeout: int = 60,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        allow = bool(shared_ctx.get("allow_code_exec", False))
        allowed: Optional[Sequence[str]] = shared_ctx.get("allowed_commands") or _SAFE_DEFAULTS
        approve_cb = shared_ctx.get("approval_callback")

        code = _extract_code(messages)
        if not code:
            code = await _generate_command(messages, shared_ctx, allowed)
        if not code:
            return {
                "content": (
                    "No executable command detected or generated. "
                    "Provide a fenced code block like ```bash\\n<cmd>``` or a clear analysis request."
                )
            }

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

        try:
            loop = asyncio.get_event_loop()
            proc_out = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    code,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
        except subprocess.TimeoutExpired:
            return {"content": f"Timed out after {timeout}s.", "code": code}

        stdout = (proc_out.stdout or "").strip()
        stderr = (proc_out.stderr or "").strip()
        rc = proc_out.returncode
        summary = f"$ {code}\n[exit {rc}]\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        return {"content": summary, "code": code, "log": {"returncode": rc, "stdout": stdout, "stderr": stderr}}


_CODE_BLOCK_RE = re.compile(r"```(?:bash|sh|zsh|shell)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_PROMPT_LINE_RE = re.compile(r"^\$\s*(.*)$")


def _extract_code(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        m1 = _CODE_BLOCK_RE.search(text)
        if m1:
            return m1.group(1).strip()
        for line in text.splitlines():
            m2 = _PROMPT_LINE_RE.match(line.strip())
            if m2:
                return m2.group(1).strip()
    return None


def _normalize_command(command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return ""

    if "\n" in cmd:
        lines = cmd.splitlines()
        first_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if first_line in {"python", "python3"} and body and "<<" not in first_line:
            return f"{first_line} - <<'PY'\n{body}\nPY"
    return cmd


async def _generate_command(
    messages: List[Dict[str, Any]],
    shared_ctx: Dict[str, Any],
    allowed: Optional[Sequence[str]],
) -> Optional[str]:
    llm = shared_ctx.get("llm")
    if llm is None:
        return None

    user_text = _last_user_message(messages)
    if not user_text.strip():
        return None

    allowed_list = list(allowed) if allowed else []
    allowed_hint = ", ".join(allowed_list) if allowed_list else "python, python3"

    knowledge_entries = get_agent_knowledge(shared_ctx, "code_act")
    knowledge_hint = "\n".join(f"- {item['text']}" for item in knowledge_entries[:8])

    system_lines = [
        "You translate user analysis requests into a single safe POSIX shell command.",
        f"Only use commands that start with: {allowed_hint}.",
        "If multi-line Python is required, emit a heredoc using `python - <<'PY'` and close with `PY`.",
        "Return the command only, optionally inside a ```bash``` code block.",
    ]
    if knowledge_hint:
        system_lines.append("Relevant knowledge:\n" + knowledge_hint)

    user_prompt = (
        "User request:\n"
        f"{user_text}\n\n"
        "Produce the best shell command (single command) to satisfy the request."
    )

    try:
        response, _ = await llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system="\n\n".join(system_lines),
            temperature=0,
        )
    except Exception:
        return None

    return _extract_generated_command(response)


def _extract_generated_command(text: str) -> Optional[str]:
    if not text:
        return None
    match = _CODE_BLOCK_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None


def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""
