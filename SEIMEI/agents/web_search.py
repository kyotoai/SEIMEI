from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ..agent import Agent, register


@register
class web_search(Agent):
    """Search the web and summarize results."""

    description = "Perform web search and return concise findings with sources."

    async def inference(self, messages: List[Dict[str, Any]], shared_ctx: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        query = _extract_last_user_query(messages)
        if not query:
            return {"content": "No query found in the last user message."}

        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            return {
                "content": (
                    "Web search backend not installed. Please `pip install duckduckgo_search`, "
                    "or provide a custom search backend."
                ),
                "log": {"query": query},
            }

        async def _search() -> List[Dict[str, str]]:
            loop = asyncio.get_event_loop()

            def _run():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=5))

            return await loop.run_in_executor(None, _run)

        results = await _search()
        lines = [f"Query: {query}", "Top results:"]
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = (r.get("body", "") or "").strip()
            lines.append(f"{i}. {title} — {href}")
            if body:
                lines.append(f"   {body[:160]}...")
        return {"content": "\n".join(lines), "log": {"query": query, "results": results}}


def _extract_last_user_query(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return None
