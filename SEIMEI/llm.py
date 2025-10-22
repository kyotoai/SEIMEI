from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

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
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url.rstrip("/") if base_url else None) or "https://api.openai.com/v1"
        self.timeout = timeout
        self.kwargs = kwargs

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
        payload.update(self.kwargs)
        payload.update(call_kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key and (self.base_url or "").startswith("https://api.openai.com"):
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        )

        if resp.status_code != 200:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {}) or {}
        # Normalize usage ints
        usage = {k: int(v) for k, v in usage.items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
        return content, usage
