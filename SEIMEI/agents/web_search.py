from __future__ import annotations

import asyncio
import os
import re
from html import unescape
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from seimei.agent import Agent, register
from seimei.knowledge.utils import get_agent_knowledge, prepare_knowledge_payload

_MAX_SEARCH_RESULTS = 8
_MAX_FETCHED_PAGES = 4
_MAX_CONTENT_CHARS = 4000
_FETCH_TIMEOUT = 12
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
_GOOGLE_API_KEY_ENV = "GOOGLE_CSE_API_KEY"
_GOOGLE_CX_ENV = "GOOGLE_CSE_CX"


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
            refined_query, refinement_note = await _refine_query(llm, query, knowledge_entries)

        results, search_meta = await _perform_search(refined_query, _MAX_SEARCH_RESULTS)
        if search_meta.get("unavailable"):
            return {
                "content": (
                    "Web search backend unavailable. Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX for Google Custom "
                    "Search (see README) or install the `duckduckgo_search` package."
                ),
                "log": {
                    "query": query,
                    "refined_query": refined_query,
                    "search_backend": search_meta.get("backend"),
                    "search_attempts": search_meta.get("attempts", []),
                },
            }
        pages = await _fetch_top_pages(results, _MAX_FETCHED_PAGES)
        synthesis = await _summarize_pages(llm, refined_query, pages, knowledge_entries)

        backend_label = search_meta.get("backend")
        lines = [f"Query attempted: {refined_query}"]
        if backend_label:
            lines.append(f"Search backend: {backend_label}")
        lines.append("Results:")
        for i, r in enumerate(results[: _MAX_SEARCH_RESULTS], 1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = (r.get("body", "") or "").strip()
            lines.append(f"{i}. {title or '[untitled]'} — {href}")
            if body:
                lines.append(f"   {body[:160]}...")
        if pages:
            lines.append("")
            lines.append("Fetched content excerpts:")
            for page in pages:
                excerpt = page["content"][:320].replace("\n", " ")
                lines.append(f"- {page['title'] or page['url']}: {excerpt}...")
        if synthesis:
            lines.append("")
            lines.append("LLM synthesis:")
            lines.extend(f"  {line}" for line in synthesis.splitlines() if line)
        knowledge_payload = knowledge_entries[:6]
        knowledge_texts = [item.get("text") for item in knowledge_payload if item.get("text")]
        knowledge_ids = [item.get("id") for item in knowledge_payload if item.get("id") is not None]
        if knowledge_payload:
            lines.append("")
            lines.append("Knowledge heuristics considered:")
            lines.extend(f"- {item['text']}" for item in knowledge_payload if item.get("text"))
        if refinement_note:
            lines.append("")
            lines.append(refinement_note)
        attempt_errors = [
            f"{attempt['backend']}: {attempt['error']}"
            for attempt in search_meta.get("attempts", [])
            if attempt.get("error")
        ]
        if attempt_errors:
            lines.append("")
            lines.append("Backend notes:")
            lines.extend(f"- {msg}" for msg in attempt_errors)

        if not results:
            lines.append("No results returned; consider adjusting the keywords or broadening the query.")
        payload: Dict[str, Any] = {
            "content": "\n".join(lines),
            "log": {
                "query": query,
                "refined_query": refined_query,
                "results": results,
                "pages": [{**page, "content": page["content"][:1000]} for page in pages],
                "refinement_note": refinement_note,
                "synthesis": synthesis,
                "search_backend": backend_label,
                "search_attempts": search_meta.get("attempts"),
            },
        }
        if knowledge_texts:
            payload["log"]["knowledge"] = knowledge_texts
        if knowledge_payload:
            payload["knowledge"] = knowledge_payload
        if knowledge_ids:
            payload["knowledge_id"] = knowledge_ids
        return payload


def _extract_last_user_query(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return None


async def _refine_query(llm: Any, query: str, knowledge_entries: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
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
        )
        candidate = refined_query_text.strip()
        if candidate:
            if candidate != query:
                note = f"Query refined using knowledge heuristics: {candidate}"
            else:
                note = None
            return candidate, note
    except Exception as exc:
        return query, f"Query refinement failed: {exc}"
    return query, None


async def _perform_search(query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    google_key = os.environ.get(_GOOGLE_API_KEY_ENV)
    google_cx = os.environ.get(_GOOGLE_CX_ENV)
    attempts: List[Dict[str, Any]] = []
    google_error: Optional[str] = None
    google_success = False
    if google_key and google_cx:
        google_results, google_error = await _perform_google_search(query, max_results, google_key, google_cx)
        attempts.append(
            {
                "backend": "google_custom_search",
                "error": google_error,
                "result_count": len(google_results),
            }
        )
        google_success = google_error is None
        if google_results:
            return google_results, {"backend": "google_custom_search", "attempts": attempts, "google_success": google_success}
    elif not google_key or not google_cx:
        missing = []
        if not google_key:
            missing.append(_GOOGLE_API_KEY_ENV)
        if not google_cx:
            missing.append(_GOOGLE_CX_ENV)
        google_error = f"Missing Google Custom Search environment variables: {', '.join(missing)}."

    ddg_results, ddg_error, ddg_available = await _perform_duckduckgo_search(query, max_results)
    attempts.append(
        {
            "backend": "duckduckgo_search",
            "error": ddg_error,
            "result_count": len(ddg_results),
        }
    )
    meta: Dict[str, Any] = {
        "backend": "duckduckgo_search",
        "attempts": attempts,
        "google_success": google_success,
        "duckduckgo_available": ddg_available,
        "error": ddg_error or google_error,
        "unavailable": not google_success and not ddg_available,
    }
    return ddg_results, meta


async def _perform_google_search(
    query: str, max_results: int, api_key: str, cx: str
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    loop = asyncio.get_event_loop()

    def _run() -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            resp = requests.get(
                _GOOGLE_SEARCH_URL,
                params={"key": api_key, "cx": cx, "q": query, "num": min(max_results, 10)},
                timeout=_FETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            return [], f"Google Custom Search API request failed: {exc}"
        except ValueError as exc:
            return [], f"Google Custom Search returned invalid JSON: {exc}"

        if isinstance(data, dict) and data.get("error"):
            message = data["error"].get("message") if isinstance(data["error"], dict) else data["error"]
            return [], f"Google Custom Search API error: {message}"

        items = data.get("items", []) if isinstance(data, dict) else []
        results: List[Dict[str, Any]] = []
        for item in items[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": (item.get("title") or item.get("htmlTitle") or "").strip(),
                    "href": (item.get("link") or "").strip(),
                    "body": (item.get("snippet") or item.get("htmlSnippet") or "").strip(),
                }
            )
        return results, None

    return await loop.run_in_executor(None, _run)


async def _perform_duckduckgo_search(query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception as exc:
        return [], f"duckduckgo_search package unavailable: {exc}", False

    loop = asyncio.get_event_loop()

    def _run() -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results)), None
        except Exception as exc:
            return [], str(exc)

    results, error = await loop.run_in_executor(None, _run)
    return results, (f"DuckDuckGo search failed: {error}" if error else None), True


async def _fetch_top_pages(results: List[Dict[str, Any]], max_pages: int) -> List[Dict[str, str]]:
    pages: List[Dict[str, str]] = []
    for result in results:
        if len(pages) >= max_pages:
            break
        url = (result.get("href") or "").strip()
        if not url:
            continue
        text = await _fetch_page_text(url)
        if not text:
            continue
        title = result.get("title", "").strip()
        pages.append(
            {
                "url": url,
                "title": title,
                "content": text[:_MAX_CONTENT_CHARS],
            }
        )
    return pages


async def _fetch_page_text(url: str) -> str:
    loop = asyncio.get_event_loop()

    def _run() -> str:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "SEIMEI-WebSearch/1.0"},
                timeout=_FETCH_TIMEOUT,
            )
            if resp.status_code >= 400:
                return ""
            return resp.text
        except requests.RequestException:
            return ""

    html = await loop.run_in_executor(None, _run)
    if not html:
        return ""
    return _html_to_text(html)


def _html_to_text(html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    text = unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


async def _summarize_pages(
    llm: Any,
    query: str,
    pages: List[Dict[str, str]],
    knowledge_entries: Sequence[Dict[str, Any]],
) -> Optional[str]:
    if llm is None or not pages:
        return None
    context_parts = []
    for page in pages:
        context_parts.append(
            f"Source: {page['title'] or page['url']}\nURL: {page['url']}\nExcerpt:\n{page['content']}\n"
        )
    knowledge_block = "\n".join(f"- {item['text']}" for item in knowledge_entries[:6])
    user_prompt = (
        f"User query: {query}\n\n"
        "You have access to the following page excerpts:\n"
        + "\n\n".join(context_parts)
        + "\n\nSummarize the most relevant findings and cite the supporting URLs."
    )
    system_prompt = (
        "You synthesize web research snippets into concise bullet points with explicit citations."
    )
    if knowledge_block:
        system_prompt += "\nRelevant heuristics:\n" + knowledge_block
    try:
        summary, _ = await llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        return summary.strip()
    except Exception:
        return None
