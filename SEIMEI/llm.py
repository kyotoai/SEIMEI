from __future__ import annotations

import asyncio
import json
import math
import os
import sys
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
    formatted: List[str] = [f"<{tag}>{lines[0]}"] if lines else [f"<{tag}>"]
    for extra in lines[1:]:
        formatted.append(f"  {extra}")
    return formatted


def format_agent_history(agent_messages: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, msg in enumerate(agent_messages, start=1):
        block_lines: List[str] = []
        header = f"AGENT OUTPUT {idx}"
        content = msg.get("content")
        if content is None:
            continue
        block_lines.append(header)
        block_lines.extend(_format_tag_block("content", content))
        blocks.append("\n".join(block_lines).rstrip())

    return "\n\n".join(blocks)


def _stringify_content(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def prepare_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    drop_normal_system: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    prepared: List[Dict[str, Any]] = []
    normal_system_count = 0
    agent_buffer: List[Dict[str, Any]] = []
    last_user_idx: Optional[int] = None

    def flush_agent_buffer() -> None:
        nonlocal last_user_idx
        if not agent_buffer:
            return
        history_text = format_agent_history(agent_buffer)
        agent_buffer.clear()
        if not history_text:
            return
        section = f"Here's the analysis you made so far:\n{history_text}"
        if last_user_idx is not None:
            existing = prepared[last_user_idx].get("content", "")
            existing_str = _stringify_content(existing)
            joiner = "\n\n" if existing_str else ""
            prepared[last_user_idx]["content"] = f"{existing_str}{joiner}{section}"
        else:
            prepared.append({"role": "user", "content": section})
            last_user_idx = len(prepared) - 1

    for raw in messages or []:
        if not isinstance(raw, dict):
            continue
        role_raw = str(raw.get("role") or "user").lower()
        content = _stringify_content(raw.get("content", ""))
        name = raw.get("name")

        is_agent = role_raw == "agent" or (role_raw == "system" and bool(raw.get("agent")))
        if is_agent:
            agent_entry: Dict[str, Any] = {"content": content}
            if isinstance(name, str) and name.strip():
                agent_entry["name"] = name.strip()[:64]
            agent_buffer.append(agent_entry)
            continue

        flush_agent_buffer()

        if role_raw == "system":
            normal_system_count += 1
            if not drop_normal_system:
                entry = {"role": "system", "content": content}
                if isinstance(name, str) and name.strip():
                    entry["name"] = name.strip()[:64]
                prepared.append(entry)
            continue

        if role_raw == "assistant":
            entry = {"role": "assistant", "content": content}
            if isinstance(name, str) and name.strip():
                entry["name"] = name.strip()[:64]
            tool_calls = raw.get("tool_calls")
            if tool_calls:
                entry["tool_calls"] = tool_calls
            function_call = raw.get("function_call")
            if function_call:
                entry["function_call"] = function_call
            prepared.append(entry)
            continue

        normalized_role = "user"
        if role_raw == "developer":
            normalized_role = "system"
        entry = {"role": normalized_role, "content": content}
        if isinstance(name, str) and name.strip():
            entry["name"] = name.strip()[:64]
        prepared.append(entry)
        if normalized_role == "user":
            last_user_idx = len(prepared) - 1

    flush_agent_buffer()

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
        model: str = "gpt-4o-mini",
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
        prepared_msgs, normal_system_count = prepare_messages(messages, drop_normal_system=bool(system))
        print("normal_system_count: ")
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

        payload_msgs: List[Dict[str, Any]] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        for msg in prepared_msgs:
            entry = dict(msg)
            for drop_key in ("agent", "log", "code", "chosen_instructions"):
                entry.pop(drop_key, None)
            payload_msgs.append(entry)

        print()
        print("payload_msgs: ", payload_msgs)

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

            #print()
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
