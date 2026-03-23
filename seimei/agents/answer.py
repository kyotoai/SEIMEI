from __future__ import annotations

from typing import Any, Dict, List, Optional

from seimei.agent import Agent, register
from seimei.llm import TokenLimitExceeded
from seimei.knowledge.utils import prepare_knowledge_payload
from seimei.prompts.default import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_KNOWLEDGE_HINT_PREFIX,
    ANSWER_USER_PROMPT_WITH_QUESTION,
    ANSWER_USER_PROMPT_NO_QUESTION,
    ANSWER_USER_PROMPT_FINDINGS_HEADER,
    ANSWER_USER_PROMPT_NO_FINDINGS,
    ANSWER_USER_PROMPT_CLOSING,
)


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
        knowledge_entries = await self.get_agent_knowledge()
        knowledge_subset = knowledge_entries[:8]
        knowledge_payload, knowledge_log_texts, knowledge_ids = prepare_knowledge_payload(knowledge_subset)

        system_prompt = ANSWER_SYSTEM_PROMPT
        if knowledge_entries:
            knowledge_block = "\n".join(f"- {item['text']}" for item in knowledge_entries[:8])
            system_prompt += ANSWER_KNOWLEDGE_HINT_PREFIX.format(knowledge_block=knowledge_block)
        segments: List[str] = []
        if user_question.strip():
            segments.append(ANSWER_USER_PROMPT_WITH_QUESTION.format(user_question=user_question.strip()))
        else:
            segments.append(ANSWER_USER_PROMPT_NO_QUESTION)

        if findings:
            segments.append(ANSWER_USER_PROMPT_FINDINGS_HEADER)
            for item in findings:
                segments.append(f"- {item['agent']}: {item['content']}")
        else:
            segments.append(ANSWER_USER_PROMPT_NO_FINDINGS)

        segments.append(ANSWER_USER_PROMPT_CLOSING)
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
