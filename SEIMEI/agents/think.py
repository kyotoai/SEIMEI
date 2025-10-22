from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from seimei.agent import Agent, register

DEFAULT_INSTRUCTION_LIBRARY: Sequence[Dict[str, Any]] = [
    {
        "id": "analyze_question",
        "text": "Analyze the most recent user request and restate the concrete task in your own words.",
    },
    {
        "id": "gather_information",
        "text": "Identify whether additional information is required; if so, delegate to web_search or other retrieval agents.",
    },
    {
        "id": "use_code_agent",
        "text": "When the user asks to run commands or inspect files, call the code_act agent with the appropriate shell command.",
    },
    {
        "id": "prepare_final_answer",
        "text": "Once sufficient evidence is collected, call the answer agent to compose the final response for the user.",
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
    """Select the most relevant instructions to plan the next steps."""

    description = "Plans the next actions by selecting relevant instructions with the shared search API."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        *,
        instruction_k: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        search_fn = shared_ctx.get("search")
        instructions: Sequence[Dict[str, Any]] = (
            shared_ctx.get("instruction_list") or DEFAULT_INSTRUCTION_LIBRARY
        )
        top_k = instruction_k or shared_ctx.get("instruction_top_k") or 3

        if not instructions:
            return {
                "content": "No instructions available to select from.",
                "chosen_instructions": [],
                "log": {"query": "", "instructions": []},
            }

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
        for idx, inst in enumerate(instructions):
            text = inst.get("text") or inst.get("instruction") or str(inst)
            key_entry = {
                "key": text,
                "instruction": inst,
                "instruction_id": inst.get("id", f"instruction_{idx}"),
            }
            keys.append(key_entry)

        ranked: List[Dict[str, Any]] = []
        if callable(search_fn):
            try:
                ranked = await search_fn(
                    query=query or user_request,
                    keys=keys,
                    k=max(int(top_k), 1),
                    context={"purpose": "instruction_selection"},
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
        plan_lines = ["Planned instructions:"]
        plan_lines.extend(f"- {text}" for text in chosen_texts)

        log_data: Dict[str, Any] = {
            "query": query or user_request,
            "instructions": selected_payloads,
        }
        if 'error_note' in locals():
            log_data["warning"] = error_note

        return {
            "content": "\n".join(plan_lines),
            "chosen_instructions": chosen_texts,
            "log": log_data,
        }
