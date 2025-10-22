from __future__ import annotations

from typing import Any, Dict, List, Optional

from seimei.agent import Agent, register
from seimei.llm import TokenLimitExceeded


def _latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _aggregate_agent_findings(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    for msg in messages:
        if msg.get("role") != "agent":
            continue
        name = msg.get("name") or "agent"
        content = msg.get("content", "")
        if content:
            findings.append({"agent": name, "content": content})
    return findings


@register
class answer(Agent):
    """Compose the final response based on previous agent outputs."""

    description = "Summarizes gathered findings and delivers the final answer to the user."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Any]:
        llm = shared_ctx.get("llm")
        if llm is None:
            return {
                "content": "LLM client unavailable; cannot compose the final answer.",
                "stop": True,
                "log": {"user_input": _latest_user_message(messages)},
            }

        user_question = _latest_user_message(messages)
        findings = _aggregate_agent_findings(messages)

        findings_text = "\n".join(
            f"- {item['agent']}: {item['content']}" for item in findings
        ) or "- No intermediate findings were recorded."

        system_prompt = (
            "You are the final answer agent for SEIMEI. "
            "Using the user question and the collected findings, produce a concise, helpful reply. "
            "Reference key observations and highlight next steps if needed."
        )
        user_prompt = (
            f"User question:\n{user_question}\n\n"
            f"Collected findings:\n{findings_text}\n\n"
            "Write the final response for the user."
        )

        try:
            final_answer, _usage = await llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except TokenLimitExceeded:
            raise
        except Exception as exc:
            return {
                "content": f"[answer_error] Failed to compose final answer: {exc}",
                "stop": True,
                "log": {"user_input": user_question, "error": str(exc)},
            }

        log_data = {
            "user_input": user_question,
            "seimei_output": final_answer,
            "sources": findings,
        }
        return {
            "content": final_answer,
            "stop": True,
            "final_output": final_answer,
            "log": log_data,
        }
