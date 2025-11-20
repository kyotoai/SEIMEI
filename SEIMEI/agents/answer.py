from __future__ import annotations

from typing import Any, Dict, List, Optional

from seimei.agent import Agent, register
from seimei.llm import TokenLimitExceeded
from seimei.knowledge.utils import get_agent_knowledge, prepare_knowledge_payload


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
        knowledge_entries = get_agent_knowledge(shared_ctx, "answer")
        knowledge_subset = knowledge_entries[:8]
        knowledge_payload, knowledge_log_texts, knowledge_ids = prepare_knowledge_payload(knowledge_subset)

        system_prompt = (
            "You are the final answer agent for SEIMEI. "
            "Using the user question and the collected findings, produce a concise, helpful reply. "
            "Reference key observations and highlight next steps if needed."
        )
        if knowledge_entries:
            knowledge_block = "\n".join(f"- {item['text']}" for item in knowledge_entries[:8])
            system_prompt += "\n\nAdditional answer agent knowledge:\n" + knowledge_block
        segments: List[str] = []
        if user_question.strip():
            segments.append(f'The user asked: "{user_question.strip()}".')
        else:
            segments.append("The user did not include an explicit question; infer their needs from the findings.")

        if findings:
            segments.append("Here are the most relevant findings gathered so far:")
            for item in findings:
                segments.append(f"- {item['agent']}: {item['content']}")
        else:
            segments.append("No intermediate findings were recorded.")

        segments.append("Compose a clear, helpful reply that addresses the user's needs and suggests next steps when appropriate.")
        user_prompt = "\n".join(segments)

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
            #"sources": findings,
        }
        if knowledge_log_texts:
            log_data["knowledge"] = knowledge_log_texts
            log_data["knowledge_used"] = list(knowledge_log_texts)
        result: Dict[str, Any] = {
            "content": final_answer,
            "stop": True,
            "final_output": final_answer,
            "log": log_data,
        }
        if knowledge_payload:
            result["knowledge"] = knowledge_payload
        if knowledge_ids:
            result["knowledge_id"] = knowledge_ids
        return result
