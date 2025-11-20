from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from seimei.agent import Agent, register
from seimei.knowledge.utils import get_agent_knowledge, prepare_knowledge_payload

DEFAULT_THINK_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "analyze_request",
        "text": "Analyze the most recent user request and restate the concrete task in your own words before delegating.",
        "tags": ["analysis", "reflection"],
    },
    {
        "id": "gather_context",
        "text": "Identify missing information; if additional evidence is needed, delegate to retrieval agents such as web_search or code_act.",
        "tags": ["research"],
    },
    {
        "id": "route_to_code",
        "text": "When the task requires running commands or inspecting files, call the code_act agent with a safe command.",
        "tags": ["routing"],
    },
    {
        "id": "finalize_answer",
        "text": "Once sufficient evidence is collected, call the answer agent to compose the final response.",
        "tags": ["handoff"],
    },
    {
        "id": "multi_step_depth",
        "text": "Outline multiple concrete follow-up actions instead of rushing to the answer to keep reasoning multi-step.",
        "tags": ["planning", "depth"],
    },
]


def _extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _recent_agent_findings(messages: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    findings: List[str] = []
    for msg in reversed(messages):
        if msg.get("role") != "agent":
            continue
        name = msg.get("name") or "agent"
        content = msg.get("content", "")
        if content:
            findings.append(f"{name}: {content}")
        if len(findings) >= limit:
            break
    findings.reverse()
    return findings


def _limit_sentences(text: str, max_sentences: int = 3) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [segment.strip() for segment in parts if segment.strip()]
    trimmed = sentences[: max(int(max_sentences), 1)]
    return " ".join(trimmed)


@register
class think(Agent):
    """Select the most relevant knowledge items to plan the next steps."""

    description = "Plans the next actions by selecting relevant knowledge items with the shared search API."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        *,
        instruction_k: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        search_fn = shared_ctx.get("search")
        knowledge_entries = get_agent_knowledge(shared_ctx, "think")
        knowledge_ranked = bool(knowledge_entries)
        if not knowledge_entries:
            knowledge_entries = list(DEFAULT_THINK_KNOWLEDGE)
        top_k = instruction_k or shared_ctx.get("instruction_top_k") or 3

        user_request = _extract_last_user_message(messages)
        findings = _recent_agent_findings(messages)
        query = "\n".join(
            part for part in [
                f"User request:\n{user_request}",
                f"Recent agent findings:\n" + "\n".join(findings) if findings else None,
            ]
            if part
        )

        keys: List[Dict[str, Any]] = []
        for idx, entry in enumerate(knowledge_entries):
            text = entry.get("text") or ""
            if not text:
                continue
            key_entry = {
                "key": text,
                "knowledge": entry,
                "knowledge_id": entry.get("id", f"knowledge_{idx}"),
                "tags": entry.get("tags", []),
            }
            keys.append(key_entry)
        if not keys:
            return {
                "content": "Knowledge entries were found but none contained usable text.",
                "chosen_knowledge": [],
                "log": {"query": query or user_request, "knowledge": []},
            }

        selected_payloads: List[Dict[str, Any]] = []
        error_note: Optional[str] = None
        if knowledge_ranked:
            selected_payloads = list(keys[: max(int(top_k), 1)])
        elif callable(search_fn):
            try:
                ranked = await search_fn(
                    query=query or user_request,
                    keys=keys,
                    k=max(int(top_k), 1),
                    context={
                        "purpose": "knowledge_selection",
                        "query_override": shared_ctx.get("knowledge_query"),
                    },
                )
            except Exception as exc:
                error_note = f"Search failed: {exc}"
                ranked = []
            else:
                for item in ranked:
                    payload = item.get("payload") or {}
                    if payload:
                        selected_payloads.append(payload)
            if not selected_payloads:
                selected_payloads = list(keys[: max(int(top_k), 1)])
        else:
            selected_payloads = list(keys[: max(int(top_k), 1)])
            error_note = "Search function unavailable."

        knowledge_entries_payload: List[Dict[str, Any]] = []
        for payload in selected_payloads:
            entry_data = payload.get("knowledge")
            if isinstance(entry_data, dict):
                knowledge_entries_payload.append(entry_data)
            else:
                knowledge_entries_payload.append(payload)
        knowledge_payload, knowledge_log_texts, knowledge_ids = prepare_knowledge_payload(knowledge_entries_payload)

        chosen_texts = knowledge_log_texts or [payload.get("key", "") for payload in selected_payloads if payload.get("key")]

        llm = shared_ctx.get("llm")
        analysis_text: Optional[str] = None
        analysis_note: Optional[str] = None
        analysis_input: Optional[str] = None
        analysis_system_prompt: Optional[str] = None

        if llm is not None:
            knowledge_section = "\n".join(f"- {text}" for text in chosen_texts) or "- None."
            findings_section = "\n".join(f"- {item}" for item in findings) or "- None yet."
            question_section = user_request or "[missing user request]"
            analysis_input = (
                f"User question:\n{question_section}\n\n"
                f"Relevant knowledge cues:\n{knowledge_section}\n\n"
                f"Recent agent findings:\n{findings_section}\n\n"
                "Provide 2 sentences (3 max). "
                "Sentence 1: summarize the most important facts or evidence above. "
                "Sentence 2 (and 3 if absolutely needed): outline the single next action or question the agents should pursue."
                "Be concrete, avoid bullet points, and do not mention this is a summary."
            )
            analysis_system_prompt = (
                "You are the think agent coordinating the next action in a multi-agent system. "
                "Analyze the supplied context and respond succinctly with 2 sentences (3 max): "
                "what you now believe plus the immediate next step."
            )
            try:
                llm_response, _ = await llm.chat(
                    messages=[{"role": "user", "content": analysis_input}],
                    system=analysis_system_prompt,
                )
            except Exception as exc:
                analysis_note = f"LLM analysis failed: {exc}"
            else:
                llm_response = (llm_response or "").strip()
                analysis_text = _limit_sentences(llm_response, max_sentences=3)
        else:
            analysis_note = "LLM unavailable for think agent."

        if not analysis_text:
            knowledge_summary = "; ".join(chosen_texts) if chosen_texts else "no curated knowledge yet"
            findings_summary = "; ".join(findings) if findings else "no prior findings yet"
            analysis_text = (
                f"Key cues: {knowledge_summary}. "
                f"Next, build on {findings_summary} to decide the following command."
            )

        log_data: Dict[str, Any] = {
            "query": query or user_request,
        }
        if knowledge_log_texts:
            log_data["knowledge"] = knowledge_log_texts
        if error_note:
            log_data["warning"] = error_note
        if analysis_note:
            log_data["analysis_warning"] = analysis_note
        if analysis_input:
            log_data["analysis_prompt"] = {
                "system": analysis_system_prompt,
                "content": analysis_input,
            }

        result: Dict[str, Any] = {
            "content": analysis_text,
            "chosen_knowledge": chosen_texts,
            "log": log_data,
        }
        if knowledge_payload:
            result["knowledge"] = knowledge_payload
        if knowledge_ids:
            result["knowledge_id"] = knowledge_ids
        return result
