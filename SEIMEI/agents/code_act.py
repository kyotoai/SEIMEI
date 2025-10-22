from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from seimei.agent import Agent, register

_SAFE_DEFAULTS = ["echo", "python", "pip", "ls", "cat", "pwd", "whoami", "dir", "type"]


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
            return {"content": "No command detected. Provide a fenced code block like ```bash\\n<cmd>```."}

        cmd0 = shlex.split(code)[0] if code.strip() else ""
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
