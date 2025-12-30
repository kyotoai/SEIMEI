from __future__ import annotations

import asyncio
import hashlib
import os
import re
from html import unescape
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import requests

from seimei.agent import Agent, register
from seimei.knowledge.utils import get_agent_knowledge

_MAX_SEARCH_RESULTS = 8
_MAX_FETCHED_PAGES = 4
_MAX_CONTENT_CHARS = 4000
_FETCH_TIMEOUT = 12
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_SUP_RE = re.compile(r"<sup( [^>]*)?>([\w\-]+)</sup>")
_HTML_SUB_RE = re.compile(r"<sub( [^>]*)?>([\w\-]+)</sub>")
_HTML_TAGS_SEQ_RE = re.compile(r"(?<=\w)((<[^>]*>)+)(?=\w)")
_LINK_OPEN = "\u3010"
_LINK_CLOSE = "\u3011"
_LINK_SEP = "\u2020"
_LINK_SEP_ALT = "\u2021"
_ANCHOR_TAG_RE = re.compile(f"{_LINK_OPEN}@([^{_LINK_CLOSE}]+){_LINK_CLOSE}")
_WHITESPACE_ANCHOR_RE = re.compile(f"({_LINK_OPEN}@[^{_LINK_CLOSE}]+{_LINK_CLOSE})(\\s+)")
_EMPTY_LINE_RE = re.compile(r"^\s+$", flags=re.MULTILINE)
_EXTRA_NEWLINE_RE = re.compile(r"\n(\s*\n)+")
_UNICODE_SMP_RE = re.compile(r"[\U00010000-\U0001FFFF]", re.UNICODE)
_SPECIAL_CHAR_TRANSLATION = str.maketrans(
    {
        "\u3010": "\u3016",
        "\u3011": "\u3017",
        "\u25FC": "\u25FE",
        "\u200b": "",
    }
)
_GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
_GOOGLE_API_KEY_ENV = "GOOGLE_CSE_API_KEY"
_GOOGLE_CX_ENV = "GOOGLE_CSE_CX"
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}


@register
class web_search(Agent):
    """Search the web and return results."""

    description = "Perform web search and return results with sources."

    async def inference(self, messages: List[Dict[str, Any]], shared_ctx: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        query = _extract_last_user_query(messages)
        if not query:
            return {"content": "No query found in the last user message."}

        knowledge_entries = get_agent_knowledge(shared_ctx, "web_search")
        llm = shared_ctx.get("llm")
        refined_query = query
        refinement_note: Optional[str] = None
        if llm:
            refined_query, refinement_note = await _refine_query(llm, messages, query, knowledge_entries)

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
        url_history = _get_url_history(shared_ctx)
        content_history = _get_content_history(shared_ctx)
        results, skipped_results = _filter_results_by_history(results, url_history)
        pages = await _fetch_top_pages(results, _MAX_FETCHED_PAGES, url_history, content_history)

        backend_label = search_meta.get("backend")
        lines = [f"Query attempted: {refined_query}"]
        if backend_label:
            lines.append(f"Search backend: {backend_label}")
        lines.append("Results:")
        for i, r in enumerate(results[: _MAX_SEARCH_RESULTS], 1):
            title = r.get("title", "").strip()
            href = _extract_result_url(r)
            body = (r.get("body", "") or "").strip()
            lines.append(f"{i}. {title or '[untitled]'} — {href}")
            if body:
                lines.append(f"   {body[:160]}...")
        if skipped_results:
            lines.append(f"Skipped {skipped_results} result(s) already visited in this run.")
        if pages:
            lines.append("")
            lines.append("Fetched content excerpts:")
            for page in pages:
                excerpt = page["content"][:320].replace("\n", " ")
                lines.append(f"- {page['title'] or page['url']}: {excerpt}...")
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

        if not results and skipped_results:
            lines.append("All search results were already visited; try generating different query than one attempted before.")
        elif not results:
            lines.append("No results returned; consider adjusting the keywords or broadening the query.")
        payload: Dict[str, Any] = {
            "content": "\n".join(lines),
            "log": {
                "query": query,
                "refined_query": refined_query,
                "results": results,
                "pages": [{**page, "content": page["content"][:1000]} for page in pages],
                "refinement_note": refinement_note,
                "skipped_results": skipped_results,
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


def _get_url_history(shared_ctx: Dict[str, Any]) -> Set[str]:
    history = shared_ctx.get("web_search_url_history")
    if isinstance(history, set):
        return history
    if isinstance(history, list):
        history_set = {item for item in history if isinstance(item, str)}
        shared_ctx["web_search_url_history"] = history_set
        return history_set
    history_set: Set[str] = set()
    shared_ctx["web_search_url_history"] = history_set
    return history_set


def _get_content_history(shared_ctx: Dict[str, Any]) -> Set[str]:
    history = shared_ctx.get("web_search_content_history")
    if isinstance(history, set):
        return history
    if isinstance(history, list):
        history_set = {item for item in history if isinstance(item, str)}
        shared_ctx["web_search_content_history"] = history_set
        return history_set
    history_set: Set[str] = set()
    shared_ctx["web_search_content_history"] = history_set
    return history_set


def _extract_result_url(result: Dict[str, Any]) -> str:
    for key in ("href", "url", "link"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _filter_results_by_history(
    results: List[Dict[str, Any]],
    visited_urls: Optional[Set[str]],
) -> Tuple[List[Dict[str, Any]], int]:
    if not visited_urls:
        return results, 0
    filtered: List[Dict[str, Any]] = []
    skipped = 0
    seen: Set[str] = set()
    for result in results:
        url = _extract_result_url(result)
        if not url:
            continue
        normalized = _normalize_url_for_history(url) or url
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized in visited_urls:
            skipped += 1
            continue
        filtered.append(result)
    return filtered, skipped


async def _refine_query(
    llm: Any,
    messages: List[Dict[str, Any]],
    query: str,
    knowledge_entries: Sequence[Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    knowledge_block = "\n".join(
        f"- {item['text']}" for item in knowledge_entries[:6] if item.get("text")
    )
    system_lines = [
        "You refine the user's latest request into a focused web search query.",
        "Return a single-line search query and nothing else.",
        "Preserve key entities, time ranges, and constraints from the request.",
        "If no improvement is needed, repeat the original query verbatim.",
    ]
    if knowledge_block:
        system_lines.append("Relevant knowledge:\n" + knowledge_block)
    try:
        refined_query_text, _ = await llm.chat(
            messages=messages,
            system="\n\n".join(system_lines),
        )
        candidate = refined_query_text.strip()
        if candidate:
            note = None
            if candidate != query:
                if knowledge_block:
                    note = f"Query refined using knowledge heuristics: {candidate}"
                else:
                    note = f"Query refined: {candidate}"
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
            raw_body = (item.get("snippet") or item.get("htmlSnippet") or "").strip()
            results.append(
                {
                    "title": (item.get("title") or item.get("htmlTitle") or "").strip(),
                    "href": (item.get("link") or "").strip(),
                    "body": _html_to_text(raw_body) if raw_body else "",
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


async def _fetch_top_pages(
    results: List[Dict[str, Any]],
    max_pages: int,
    visited_urls: Optional[Set[str]] = None,
    content_history: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    pages: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()

    for result in results:
        if len(pages) >= max_pages:
            break
        url = _extract_result_url(result)
        if not url:
            continue
        normalized_url = _normalize_url_for_history(url) or url
        if normalized_url in seen_urls:
            continue
        if visited_urls is not None and normalized_url in visited_urls:
            continue
        seen_urls.add(normalized_url)
        if visited_urls is not None:
            visited_urls.add(normalized_url)
        text, final_url = await _fetch_page_text(url)
        if not text:
            continue
        title = result.get("title", "").strip()
        final_url = final_url or url
        final_normalized = _normalize_url_for_history(final_url) or normalized_url
        if visited_urls is not None:
            visited_urls.add(final_normalized)
        content_sig = _content_signature(text)
        if content_history is not None:
            if content_sig in content_history:
                continue
            content_history.add(content_sig)
        pages.append(
            {
                "url": final_url,
                "title": title,
                "content": text[:_MAX_CONTENT_CHARS],
            }
        )
    return pages


async def _fetch_page_text(url: str) -> Tuple[str, str]:
    loop = asyncio.get_event_loop()

    def _run() -> Tuple[str, str]:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "SEIMEI-WebSearch/1.0"},
                timeout=_FETCH_TIMEOUT,
            )
            if resp.status_code >= 400:
                return "", resp.url or url
            return resp.text, resp.url or url
        except requests.RequestException:
            return "", url

    html, base_url = await loop.run_in_executor(None, _run)
    if not html:
        return "", base_url
    return _html_to_text(html, base_url), base_url


def _get_domain(url: str) -> str:
    if not url:
        return ""
    if "http" not in url:
        url = "http://" + url
    return urlparse(url).netloc


def _normalize_url_for_history(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except Exception:
        return url.strip()
    if not parts.netloc:
        return url.strip()
    scheme = parts.scheme.lower() if parts.scheme else "http"
    if scheme in {"http", "https"}:
        scheme = "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path or ""
    if path != "/":
        path = path.rstrip("/")
    query = _strip_tracking_query(parts.query)
    return urlunsplit((scheme, netloc, path, query, ""))


def _strip_tracking_query(query: str) -> str:
    if not query:
        return ""
    filtered = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_"):
            continue
        if lowered in _TRACKING_QUERY_KEYS:
            continue
        filtered.append((key, value))
    return urlencode(filtered, doseq=True)


def _content_signature(text: str) -> str:
    compact = _WHITESPACE_RE.sub(" ", text).strip()
    sample = compact[:2000]
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()


def _merge_whitespace(text: str) -> str:
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def _arxiv_to_ar5iv(url: str) -> str:
    return re.sub(r"arxiv.org", r"ar5iv.org", url)


def _get_text(node: Any) -> str:
    return _merge_whitespace(" ".join(node.itertext()))


def _replace_node_with_text(node: Any, text: str) -> None:
    parent = node.getparent()
    if parent is None:
        return
    previous = node.getprevious()
    tail = node.tail or ""
    if previous is None:
        parent.text = (parent.text or "") + text + tail
    else:
        previous.tail = (previous.tail or "") + text + tail
    parent.remove(node)


def _clean_links(root: Any, cur_url: str) -> Dict[str, str]:
    if not cur_url:
        return {}
    cur_domain = _get_domain(cur_url)
    urls: Dict[str, str] = {}
    urls_rev: Dict[str, str] = {}

    for a in root.findall(".//a[@href]"):
        if a.getparent() is None:
            continue

        link = a.attrib["href"]
        if link.startswith(("mailto:", "javascript:")):
            continue

        text = _get_text(a).replace(_LINK_SEP, _LINK_SEP_ALT)
        if not _ANCHOR_TAG_RE.sub("", text):
            continue

        if link.startswith("#"):
            _replace_node_with_text(a, text)
            continue

        try:
            link = urljoin(cur_url, link)
            domain = _get_domain(link)
        except Exception:
            domain = ""

        if not domain:
            continue

        link = _arxiv_to_ar5iv(link)

        if (link_id := urls_rev.get(link)) is None:
            link_id = f"{len(urls)}"
            urls[link_id] = link
            urls_rev[link] = link_id

        if domain == cur_domain:
            replacement = f"{_LINK_OPEN}{link_id}{_LINK_SEP}{text}{_LINK_CLOSE}"
        else:
            replacement = f"{_LINK_OPEN}{link_id}{_LINK_SEP}{text}{_LINK_SEP}{domain}{_LINK_CLOSE}"

        _replace_node_with_text(a, replacement)

    return urls


def _replace_images(root: Any) -> None:
    cnt = 0
    for img_tag in root.findall(".//img"):
        image_name = img_tag.get("alt", img_tag.get("title"))
        replacement = f"[Image {cnt}: {image_name}]" if image_name else f"[Image {cnt}]"
        _replace_node_with_text(img_tag, replacement)
        cnt += 1


def _remove_node(node: Any) -> None:
    parent = node.getparent()
    if parent is None:
        return
    parent.remove(node)


def _remove_math(root: Any) -> None:
    for node in root.findall(".//math"):
        _remove_node(node)


def _replace_special_chars(text: str) -> str:
    return text.translate(_SPECIAL_CHAR_TRANSLATION)


def _remove_unicode_smp(text: str) -> str:
    return _UNICODE_SMP_RE.sub("", text)


def _escape_md(text: str) -> str:
    return text


def _escape_md_section(text: str, snob: bool = False) -> str:
    return text


def _html_to_markdown(html: str, html2text_module: Any) -> str:
    html = re.sub(_HTML_SUP_RE, r"^{\2}", html)
    html = re.sub(_HTML_SUB_RE, r"_{\2}", html)
    html = re.sub(_HTML_TAGS_SEQ_RE, r" \1", html)

    orig_escape_md = html2text_module.utils.escape_md
    orig_escape_md_section = html2text_module.utils.escape_md_section
    html2text_module.utils.escape_md = _escape_md
    html2text_module.utils.escape_md_section = _escape_md_section
    try:
        h = html2text_module.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        h.ignore_tables = True
        h.unicode_snob = True
        h.ignore_emphasis = True
        return h.handle(html).strip()
    finally:
        html2text_module.utils.escape_md = orig_escape_md
        html2text_module.utils.escape_md_section = orig_escape_md_section


def _basic_html_to_text(html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    text = unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _html_to_text(html: str, url: str = "") -> str:
    html = _remove_unicode_smp(html)
    html = _replace_special_chars(html)
    try:
        import html2text
        import lxml.etree
        import lxml.html
    except Exception:
        return _basic_html_to_text(html)

    try:
        root = lxml.html.fromstring(html)
    except Exception:
        return _basic_html_to_text(html)

    if url:
        _clean_links(root, url)
    _replace_images(root)
    _remove_math(root)

    clean_html = lxml.etree.tostring(root, encoding="UTF-8").decode()
    text = _html_to_markdown(clean_html, html2text)
    text = _WHITESPACE_ANCHOR_RE.sub(lambda m: m.group(2) + m.group(1), text)
    text = _EMPTY_LINE_RE.sub("", text)
    text = _EXTRA_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
