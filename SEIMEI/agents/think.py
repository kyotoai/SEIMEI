from __future__ import annotations

from typing import Any, Dict, List, Optional

from seimei.agent import Agent, register
from seimei.knowledge.utils import get_agent_knowledge

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

        ranked: List[Dict[str, Any]] = []
        error_note: Optional[str] = None
        if callable(search_fn):
            try:
                ranked = await search_fn(
                    query=query or user_request,
                    keys=keys,
                    k=max(int(top_k), 1),
                    context={"purpose": "knowledge_selection"},
                )
            except Exception as exc:
                ranked = []
                error_note = f"Search failed: {exc}"
        else:
            ranked = []
            error_note = "Search function unavailable."

        selected_payloads: List[Dict[str, Any]] = []
        for item in ranked:
            payload = item.get("payload") or {}
            if payload:
                selected_payloads.append(payload)
        if not selected_payloads:
            selected_payloads = list(keys[: max(int(top_k), 1)])

        chosen_texts = [payload.get("key", "") for payload in selected_payloads if payload.get("key")]
        plan_lines = ["Selected knowledge cues:"]
        plan_lines.extend(f"- {text}" for text in chosen_texts)

        log_data: Dict[str, Any] = {
            "query": query or user_request,
            "knowledge": selected_payloads,
        }
        if error_note:
            log_data["warning"] = error_note

        return {
            "content": "\n".join(plan_lines),
            "chosen_knowledge": chosen_texts,
            "log": log_data,
        }
