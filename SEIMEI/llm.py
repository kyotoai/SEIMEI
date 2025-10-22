from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

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
        **kwargs: Any,
    ) -> None:
        api_key_env = kwargs.pop("api_key_env", None)
        extra_headers = kwargs.pop("extra_headers", None) or {}

        self.model = model
        env_key = os.getenv(api_key_env) if api_key_env else None
        self.api_key = api_key or env_key or os.getenv("OPENAI_API_KEY")

        raw_base = (base_url or "").strip()
        self.base_url = raw_base.rstrip("/") if raw_base else None
        self.using_openai = not self.base_url or self.base_url.startswith("https://api.openai.com")
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

        self.timeout = timeout
        self.kwargs = kwargs
        self.extra_headers = extra_headers
        self.last_response: Optional[Dict[str, Any]] = None
        self._warned_filtered_kwargs = False

        # If using official API and no key, leave requests to fail with a clear error
        # If using local base_url, API key may not be required

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        **call_kwargs: Any,
    ) -> Tuple[str, Dict[str, int]]:
        payload_msgs: List[Dict[str, str]] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        payload_msgs.extend([{"role": m.get("role","user"), "content": m.get("content","")} for m in messages])

        payload = {
            "model": self.model,
            "messages": payload_msgs,
        }
        extra_params = dict(self.kwargs)
        extra_params.update(call_kwargs)
        payload.update(self._filter_payload(extra_params))

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        if self.using_openai and not self.api_key:
            raise RuntimeError(
                "LLMClient: OPENAI_API_KEY not set. Provide an API key or set `base_url` to a local server."
            )

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
        usage_raw = data.get("usage") or {}
        usage: Dict[str, int] = {}
        for k, v in usage_raw.items():
            try:
                usage[k] = int(v)
            except (TypeError, ValueError):
                continue
        return content, usage

    def _filter_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.using_openai:
            return params
        filtered = {k: v for k, v in params.items() if k in _OPENAI_CHAT_FIELDS}
        dropped = set(params.keys()) - set(filtered.keys())
        if dropped and not self._warned_filtered_kwargs:
            print(
                f"[LLMClient] Ignoring unsupported OpenAI parameters: {sorted(dropped)}",
                file=sys.stderr,
            )
            self._warned_filtered_kwargs = True
        return filtered

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
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
        return ""
