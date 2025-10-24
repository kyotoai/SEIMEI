from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from seimei.agent import Agent, register
from seimei.knowledge.utils import get_agent_knowledge


@register
class web_search(Agent):
    """Search the web and summarize results."""

    description = "Perform web search and return concise findings with sources."

    async def inference(self, messages: List[Dict[str, Any]], shared_ctx: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        query = _extract_last_user_query(messages)
        if not query:
            return {"content": "No query found in the last user message."}

        knowledge_entries = get_agent_knowledge(shared_ctx, "web_search")
        llm = shared_ctx.get("llm")
        refined_query = query
        refinement_note: Optional[str] = None
        if knowledge_entries and llm:
            knowledge_block = "\n".join(f"- {item['text']}" for item in knowledge_entries[:6])
            prompt = (
                "Use the following knowledge heuristics to refine a web search query.\n"
                f"Knowledge:\n{knowledge_block}\n\n"
                f"Original query:\n{query}\n\n"
                "Return an improved query as a single line. If no improvement is needed, repeat the original."
            )
            try:
                refined_query_text, _ = await llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system="You polish search queries using provided heuristics.",
                    temperature=0,
                )
                candidate = refined_query_text.strip()
                if candidate:
                    refined_query = candidate
                    if candidate != query:
                        refinement_note = f"Query refined using knowledge heuristics: {candidate}"
            except Exception as exc:
                refinement_note = f"Query refinement failed: {exc}"

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
                    return list(ddgs.text(refined_query, max_results=5))

            return await loop.run_in_executor(None, _run)

        results = await _search()
        lines = [f"Query: {refined_query}", "Top results:"]
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = (r.get("body", "") or "").strip()
            lines.append(f"{i}. {title} — {href}")
            if body:
                lines.append(f"   {body[:160]}...")
        if knowledge_entries:
            lines.append("")
            lines.append("Knowledge heuristics considered:")
            lines.extend(f"- {item['text']}" for item in knowledge_entries[:6])
        return {
            "content": "\n".join(lines),
            "log": {
                "query": query,
                "refined_query": refined_query,
                "results": results,
                "knowledge": [item.get("text") for item in knowledge_entries[:6]],
                "refinement_note": refinement_note,
            },
        }


def _extract_last_user_query(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return None
