from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from .logging_utils import LogColors, colorize


def _format_tag_block(tag: str, value: Any) -> List[str]:
    if value is None:
        return []
    value_str = _stringify_content(value)
    if not value_str:
        return [f"<{tag}>"]
    lines = value_str.splitlines()
    #formatted: List[str] = [f"<{tag}>{lines[0]}"] if lines else [f"<{tag}>"]
    formatted: List[str] = [f"{lines[0]}"] if lines else [f"<{tag}>"]
    for extra in lines[1:]:
        formatted.append(f"  {extra}")
    return formatted


def format_agent_history(agent_messages: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, msg in enumerate(agent_messages, start=1):
        block_lines: List[str] = []
        header = f"<AGENT OUTPUT STEP {idx}>"
        footer = f"</AGENT OUTPUT STEP {idx}>"
        content = msg.get("content")
        if content is None:
            continue
        block_lines.append(header)
        block_lines.extend(_format_tag_block("content", content))
        block_lines.append(footer)
        blocks.append("\n".join(block_lines).rstrip())

    return "\n\n".join(blocks)


def _stringify_content(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)

"""
prepare_messages : convert message into LLM conpatible message

Args:
messages: [
    {
        "role":<"system" or "user" or "assistant" or "agent">,
        "content":<content>,
        "name"(optional):<"answer" or "code_act" or other agent name>
    },
    ...
]
drop_normal_system: False

Return: 
messages: [
    {
        "role":<"system" or "user" or "assistant">,
        "content":<content (If this is the last user message in messages, agent messages after this are included in this user message.)>,
        "name"(optional):<"answer" or "code_act" or other agent name>
    },
    ...
]
normal_system_count: Int <system message counter>


Ex.
Args:
system = "z"
messages = [
    {
        "role": "system",
        "content": "a"
    },
    {
        "role": "user",
        "content": "b"
    },
    {
        "role": "agent",
        "name": "code_act",
        "content": "c"
    },
    {
        "role": "agent",
        "name": "answer",
        "content": "d"
    },
    {
        "role": "assistant",
        "content": "e"
    },
    {
        "role": "system",
        "content": "f"
    },
    {
        "role": "user",
        "content": "g"
    },
    {
        "role": "agent",
        "name": "code_act",
        "content": "h"
    },
]

->

messages: [
    {
        "role": "user",
        "content": "<USER_SYSTEM>a</USER_SYSTEM>\n\n<USER>b</USER>"
    },
    {
        "role": "assistant",
        "content": "e"
    },
    {
        "role": "user",
        "content": "z \n\n\n<USER_SYSTEM>f</USER_SYSTEM>\n\n<USER>g</USER>\n\n<AGENT OUTPUT STEP 1>h</AGENT OUTPUT STEP 1>"
    },
]
"""
def prepare_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    system: str = None,
    drop_normal_system: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    prepared: List[Dict[str, Any]] = []
    normal_system_count = 0

    normalized: List[Dict[str, Any]] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            raise Exception("messages must be list of dict but it has non dict in it.")
        if not raw.get("role"):
            raise Exception("dict in messages must include role.")
        normalized.append(raw)

    def is_agent_message(msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "").lower()
        if role == "agent":
            return True
        if role == "system" and msg.get("agent"):
            return True
        return False

    def wrap_tag(tag: str, content: Any) -> str:
        text = _stringify_content(content)
        return f"<{tag}>{text}</{tag}>"

    last_user_input_idx: Optional[int] = None
    for idx, msg in enumerate(normalized):
        if str(msg.get("role") or "").lower() == "user":
            last_user_input_idx = idx

    agent_messages: List[Dict[str, Any]] = []
    if last_user_input_idx is not None:
        for msg in normalized[last_user_input_idx + 1 :]:
            if is_agent_message(msg):
                agent_messages.append({"content": msg.get("content")})

    current_segments: List[str] = []
    for msg in normalized:
        if is_agent_message(msg):
            continue

        role_raw = str(msg.get("role") or "").lower()
        content = msg.get("content", "")
        name = msg.get("name")

        if role_raw == "assistant":
            if current_segments:
                prepared.append({"role": "user", "content": "\n\n".join(current_segments)})
                current_segments = []
            entry = {"role": "assistant", "content": _stringify_content(content)}
            if isinstance(name, str) and name.strip():
                entry["name"] = name.strip()[:64]
            prepared.append(entry)
            continue

        if role_raw == "system":
            normal_system_count += 1
            if drop_normal_system:
                continue
            current_segments.append(wrap_tag("USER_SYSTEM", content))
            continue

        if role_raw == "developer":
            if drop_normal_system:
                continue
            current_segments.append(wrap_tag("USER_SYSTEM", content))
            continue

        current_segments.append(wrap_tag("USER", content))

    if current_segments:
        prepared.append({"role": "user", "content": "\n\n".join(current_segments)})

    def format_agent_inline(agent_msgs: Sequence[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for idx, msg in enumerate(agent_msgs, start=1):
            content = msg.get("content")
            if content is None:
                continue
            content_text = _stringify_content(content)
            blocks.append(f"<AGENT OUTPUT STEP {idx}>{content_text}</AGENT OUTPUT STEP {idx}>")
        return "\n\n".join(blocks)

    agent_history = format_agent_inline(agent_messages)
    last_user_prepared_idx: Optional[int] = None
    if agent_history:
        for idx in range(len(prepared) - 1, -1, -1):
            if prepared[idx].get("role") == "user":
                last_user_prepared_idx = idx
                break
        if last_user_prepared_idx is None:
            prepared.append({"role": "user", "content": agent_history})
            last_user_prepared_idx = len(prepared) - 1
        else:
            existing = prepared[last_user_prepared_idx].get("content", "")
            existing_text = _stringify_content(existing)
            joiner = "\n\n" if existing_text else ""
            prepared[last_user_prepared_idx]["content"] = f"{existing_text}{joiner}{agent_history}"

    if system is not None:
        system_text = _stringify_content(system)
        if system_text:
            if last_user_prepared_idx is None:
                for idx in range(len(prepared) - 1, -1, -1):
                    if prepared[idx].get("role") == "user":
                        last_user_prepared_idx = idx
                        break
            if last_user_prepared_idx is None:
                prepared.append({"role": "user", "content": system_text})
            else:
                existing = prepared[last_user_prepared_idx].get("content", "")
                existing_text = _stringify_content(existing)
                if existing_text:
                    prepared[last_user_prepared_idx]["content"] = f"{system_text}\n\n\n{existing_text}"
                else:
                    prepared[last_user_prepared_idx]["content"] = system_text

    return prepared, normal_system_count


_OPENAI_CHAT_FIELDS = {
    # Core chat completion parameters documented by OpenAI (2024-09)
    "messages",
    "model",
    "frequency_penalty",
    "presence_penalty",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "response_format",
    "logprobs",
    "top_logprobs",
    "user",
    "logit_bias",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "metadata",
    "reasoning",
    "modalities",
    "audio",
    "functions",
    "function_call",
}


class TokenLimitExceeded(RuntimeError):
    """Raised when a token limiter is exceeded during an LLM call."""

    def __init__(self, limit: int, consumed: int, last_usage: Dict[str, int]) -> None:
        self.limit = limit
        self.consumed = consumed
        self.last_usage = last_usage
        super().__init__(f"Token limit {limit} exceeded with {consumed} tokens.")


class TokenLimiter:
    """Track cumulative token usage and enforce an upper bound."""

    def __init__(self, limit: int) -> None:
        self.limit = max(int(limit), 0)
        self.consumed = 0

    def record(self, usage: Dict[str, int]) -> None:
        if not self.limit:
            return
        delta = self._extract_usage(usage)
        if self.consumed + delta > self.limit:
            raise TokenLimitExceeded(self.limit, self.consumed + delta, usage)
        self.consumed += delta

    def ensure_available(self) -> None:
        if not self.limit:
            return
        if self.consumed >= self.limit:
            raise TokenLimitExceeded(self.limit, self.consumed, {"total_tokens": self.consumed})

    def preview(self, upcoming: int, usage_hint: Optional[Dict[str, int]] = None) -> None:
        if not self.limit:
            return
        upcoming_tokens = max(int(upcoming), 0)
        projected = self.consumed + upcoming_tokens
        if projected > self.limit:
            hint = usage_hint or {"total_tokens": projected}
            raise TokenLimitExceeded(self.limit, projected, hint)

    @staticmethod
    def _extract_usage(usage: Dict[str, int]) -> int:
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return max(total, 0)
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0
        try:
            return max(int(prompt) + int(completion), 0)
        except (TypeError, ValueError):
            return 0


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Supports:
      - Official OpenAI API (set `api_key` and omit `base_url`)
      - Any compatible server (set `base_url`, e.g., vLLM or OpenAI-compatible gateway)
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_concurrent_requests: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(dotenv_path=Path(__name__).resolve().parent / ".env")

        api_key_env = kwargs.pop("api_key_env", None)
        usage_log_path = kwargs.pop("usage_log_path", None)
        extra_headers = kwargs.pop("extra_headers", None) or {}

        self.model = model
        env_key = os.getenv(api_key_env) if api_key_env else None
        self.api_key = api_key or env_key or os.getenv("OPENAI_API_KEY")

        raw_base = (base_url or "").strip()
        trimmed = raw_base.rstrip("/") if raw_base else ""
        if trimmed:
            self.base_url = trimmed
        else:
            self.base_url = "https://api.openai.com/v1/chat/completions"
        self.using_generate = self.base_url.endswith("/generate")
        if not self.using_generate and not self.base_url.endswith("/chat/completions"):
            self.base_url = f"{self.base_url.rstrip('/')}/chat/completions"
        self.using_openai = "api.openai.com" in self.base_url

        self.timeout = timeout
        self.kwargs = kwargs
        self.extra_headers = extra_headers
        self.last_response: Optional[Dict[str, Any]] = None
        self._warned_filtered_kwargs = False
        env_usage_log = os.getenv("SEIMEI_USAGE_LOG_PATH")
        self.usage_log_path = Path(usage_log_path or env_usage_log) if (usage_log_path or env_usage_log) else None
        self._semaphore: Optional[asyncio.Semaphore] = None
        if max_concurrent_requests and int(max_concurrent_requests) > 0:
            self._semaphore = asyncio.Semaphore(int(max_concurrent_requests))

        # If using official API and no key, leave requests to fail with a clear error
        # If using local base_url, API key may not be required

    def _build_generate_payload(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {k: v for k, v in params.items() if v is not None}
        payload["prompts"] = [prompt or ""]
        if "model" not in payload and self.model:
            payload["model"] = self.model
        return payload

    def _render_prompt(self, messages: Sequence[Dict[str, Any]]) -> str:
        segments: List[str] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").lower()
            raw_content = msg.get("content", "")
            content = _stringify_content(raw_content).strip()
            if not content:
                continue
            if role == "system":
                prefix = "System"
            elif role == "assistant":
                prefix = "Assistant"
            else:
                prefix = "User"
            segments.append(f"{prefix}: {content}")
        if not segments:
            return ""
        if not segments[-1].startswith("Assistant:"):
            segments.append("Assistant:")
        return "\n".join(segments)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        token_limiter: Optional[TokenLimiter] = None,
        **call_kwargs: Any,
    ) -> Tuple[str, Dict[str, int]]:
        prepared_msgs, normal_system_count = prepare_messages(messages, system=system, drop_normal_system=False) #bool(system))
        '''
        if normal_system_count > 1:
            print(
                colorize(
                    "[LLMClient] Multiple non-agent system messages detected; only the last should remain once the system prompt is applied.",
                    LogColors.RED,
                ),
                file=sys.stderr,
            )
        elif normal_system_count and not system:
            print(
                colorize(
                    "[LLMClient] Using existing non-agent system message because no system prompt argument was provided.",
                    LogColors.RED,
                ),
                file=sys.stderr,
            )
        '''

        payload_msgs: List[Dict[str, Any]] = []

        for msg in prepared_msgs:
            entry = dict(msg)
            for drop_key in ("agent", "log", "code", "chosen_instructions", "knowledge", "knowledge_id"):
                entry.pop(drop_key, None)
            payload_msgs.append(entry)

        #print("\n----------")
        #print("payload_msgs: ", payload_msgs)

        estimated_prompt_tokens = self._estimate_prompt_tokens(payload_msgs)
        if token_limiter:
            token_limiter.ensure_available()
            token_limiter.preview(
                estimated_prompt_tokens,
                {
                    "total_tokens": token_limiter.consumed + estimated_prompt_tokens,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                },
            )

        extra_params = dict(self.kwargs)
        extra_params.update(call_kwargs)
        if self.using_generate:
            prompt_text = self._render_prompt(payload_msgs)
            payload = self._build_generate_payload(prompt_text, extra_params)
        else:
            payload = {
                "model": self.model,
                "messages": payload_msgs,
            }
            payload.update(self._filter_payload(extra_params))

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url
        if self.using_openai and not self.api_key:
            raise RuntimeError(
                "LLMClient: OPENAI_API_KEY not set. Provide an API key or set `base_url` to a local server."
            )

        if token_limiter:
            token_limiter.ensure_available()

        async def _execute() -> Tuple[str, Dict[str, int]]:
            def _send_request() -> requests.Response:
                try:
                    return requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                except requests.exceptions.RequestException as exc:
                    raise RuntimeError(
                        f"LLMClient request to {url} failed: {exc}"
                    ) from exc

            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                _send_request
            )

            if resp.status_code != 200:
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            self.last_response = data
            content = self._extract_content(data)

            #print("\n----------")
            #print("content: ", content)

            if isinstance(data, dict):
                usage_raw = data.get("usage") or {}
            else:
                usage_raw = {}
            usage: Dict[str, int] = {}
            for k, v in usage_raw.items():
                try:
                    usage[k] = int(v)
                except (TypeError, ValueError):
                    continue
            if token_limiter:
                token_limiter.record(usage)
            self._log_usage(usage)
            return content, usage

        if self._semaphore is None:
            return await _execute()
        async with self._semaphore:
            return await _execute()

    def _filter_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.using_openai:
            return params
        filtered = {k: v for k, v in params.items() if k in _OPENAI_CHAT_FIELDS}
        dropped = set(params.keys()) - set(filtered.keys())
        if dropped and not self._warned_filtered_kwargs:
            print(
                colorize(
                    f"[LLMClient] Ignoring unsupported OpenAI parameters: {sorted(dropped)}",
                    LogColors.RED,
                ),
                file=sys.stderr,
            )
            self._warned_filtered_kwargs = True
        return filtered

    def _log_usage(self, usage: Dict[str, int]) -> None:
        if not self.usage_log_path:
            return
        if not usage:
            return
        try:
            self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.usage_log_path.exists()
            with self.usage_log_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if not exists:
                    writer.writerow(["timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens"])
                writer.writerow([
                    int(time.time()),
                    self.model,
                    usage.get("prompt_tokens", ""),
                    usage.get("completion_tokens", ""),
                    usage.get("total_tokens", ""),
                ])
        except Exception:
            pass

    @staticmethod
    def _estimate_prompt_tokens(messages: Sequence[Dict[str, Any]]) -> int:
        total_chars = 0
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                try:
                    rendered = json.dumps(content, ensure_ascii=False)
                except TypeError:
                    rendered = str(content)
            else:
                rendered = str(content)
            total_chars += len(rendered)
            role = msg.get("role")
            if role:
                total_chars += len(str(role))
        estimated = math.ceil(total_chars / 4) if total_chars else 0
        return max(estimated, len(messages))

    def _extract_content(self, data: Any) -> str:
        if isinstance(data, list):
            for item in data:
                text = self._extract_content(item)
                if text:
                    return text
            return ""
        if not isinstance(data, dict):
            rendered = _stringify_content(data).strip()
            return rendered if rendered else ""

        choices = data.get("choices") or []
        if choices:
            choice = choices[0] or {}
            message = choice.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if content:
                    return content
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    try:
                        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
                    except TypeError:
                        return str(tool_calls)
            text = choice.get("text")
            if text:
                return text

        if self.using_generate:
            simple_keys = ("text", "generated_text", "output", "content", "result", "response")
            for key in simple_keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            for key in ("results", "outputs", "data"):
                container = data.get(key)
                if isinstance(container, list) and container:
                    first = container[0]
                    if isinstance(first, dict):
                        for inner_key in simple_keys:
                            inner_val = first.get(inner_key)
                            if isinstance(inner_val, str) and inner_val.strip():
                                return inner_val
                    elif isinstance(first, str) and first.strip():
                        return first
            # As a very last resort, try common nested dict structures.
            message = data.get("message")
            if isinstance(message, dict):
                for key in ("content", "text", "output"):
                    value = message.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
            # Fallback to serialized content when nothing else matches.
            fallback = _stringify_content(data).strip()
            if fallback:
                return fallback
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return ""

    def bind_token_limiter(self, limiter: Optional[TokenLimiter]) -> "LLMClient":
        if limiter is None:
            return self
        return _BoundLLMClient(self, limiter)


class _BoundLLMClient:
    """Lightweight proxy that binds a token limiter to LLM calls."""

    def __init__(self, client: LLMClient, limiter: TokenLimiter) -> None:
        self._client = client
        self._limiter = limiter

    async def chat(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict[str, int]]:
        kwargs.setdefault("token_limiter", self._limiter)
        return await self._client.chat(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)
