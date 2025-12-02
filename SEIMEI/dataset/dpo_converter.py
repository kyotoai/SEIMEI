#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert train_v3 DPO runs into dataset_list format."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("exp8_csv_small/train_v3_dpo_3.json"),
        help="Path to the input JSON produced by exp8_csv_small/train_v3.py",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("exp8_csv_small/dataset_list.json"),
        help="Where to write the dataset_list payload.",
    )
    parser.add_argument(
        "--max-query-messages",
        type=int,
        default=8,
        help="Maximum number of messages to keep when building the rmsearch query block.",
    )
    parser.add_argument(
        "--max-query-chars",
        type=int,
        default=400,
        help="Maximum number of characters to keep per message when building the rmsearch query block.",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=4,
        help="Maximum number of recent agent steps to surface in the rmsearch query block.",
    )
    parser.add_argument(
        "--max-agent-chars",
        type=int,
        default=800,
        help="Maximum characters for each agent step that appears in the rmsearch query block.",
    )
    parser.add_argument(
        "--max-knowledge-per-step",
        type=int,
        default=3,
        help="Maximum number of knowledge cues extracted per agent step when formatting the query.",
    )
    parser.add_argument(
        "--max-knowledge-chars",
        type=int,
        default=200,
        help="Maximum characters kept for each extracted knowledge cue.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for the generated dataset_list file.",
    )
    return parser.parse_args()


def _coerce_messages(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [{"role": "user", "content": raw}]
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
        return [{"role": "user", "content": raw}]
    if isinstance(raw, Sequence):
        return [dict(item) for item in raw if isinstance(item, dict)]
    return [{"role": "user", "content": str(raw)}]


def _prepare_query_input(
    query: Sequence[Dict[str, Any]],
    *,
    max_messages: int,
    max_chars_per_message: int,
) -> Tuple[List[Dict[str, Any]], str, str]:
    messages = list(query)

    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    focus = ""
    for msg in reversed(messages):
        if str(msg.get("role") or "").lower() == "user":
            focus = _stringify(msg.get("content", ""))
            if focus:
                break

    label_map = {
        "user": "User",
        "assistant": "Assistant",
        "agent": "Agent",
        "system": "System",
        "function": "Function",
        "developer": "Developer",
    }
    lines: List[str] = []
    for msg in list(messages)[-max_messages:]:
        role_raw = str(msg.get("role") or "").lower()
        if role_raw == "system":
            continue
        role = label_map.get(role_raw, "Message")
        snippet = _stringify(msg.get("content", "")).strip()
        if len(snippet) > max_chars_per_message:
            snippet = snippet[:max_chars_per_message].rstrip() + "..."
        lines.append(f"{role}: {snippet}")

    conversation = "\n".join(lines) if lines else ""
    if not focus:
        focus = conversation
    return messages, conversation, focus


def _stringify_for_query(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _shorten_for_query(text: str, limit: int) -> str:
    snippet = (text or "").strip()
    if not snippet:
        return ""
    if limit > 0 and len(snippet) > limit:
        return snippet[:limit].rstrip() + "..."
    return snippet


def _stringify_knowledge_dict(data: Dict[str, Any]) -> str:
    for key in ("text", "knowledge", "key", "content", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        return json.dumps(data, ensure_ascii=False)
    except TypeError:
        return str(data)


def _coerce_knowledge_texts(value: Any, max_chars: int) -> List[str]:
    texts: List[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        snippet = _shorten_for_query(value, max_chars)
        if snippet:
            texts.append(snippet)
        return texts
    if isinstance(value, dict):
        snippet = _shorten_for_query(_stringify_knowledge_dict(value), max_chars)
        if snippet:
            texts.append(snippet)
        return texts
    if isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(_coerce_knowledge_texts(item, max_chars))
        return texts
    snippet = _shorten_for_query(str(value), max_chars)
    if snippet:
        texts.append(snippet)
    return texts


def _extract_knowledge_cues(
    message: Dict[str, Any],
    limit: int,
    max_chars: int,
) -> List[str]:
    cues: List[str] = []
    sources: List[Any] = []
    log_data = message.get("log")
    if isinstance(log_data, dict):
        for key in ("knowledge", "knowledge_used", "chosen_knowledge"):
            raw = log_data.get(key)
            if raw is not None:
                sources.append(raw)
    for key in ("chosen_knowledge",):
        raw = message.get(key)
        if raw is not None:
            sources.append(raw)
    for raw in sources:
        for text in _coerce_knowledge_texts(raw, max_chars):
            if text:
                cues.append(text)
            if len(cues) >= limit:
                return cues[:limit]
    return cues[:limit]


def _build_query_block(
    messages: Sequence[Dict[str, Any]],
    *,
    max_query_messages: int,
    max_query_chars: int,
    max_agent_steps: int,
    max_agent_chars: int,
    max_knowledge_per_step: int,
    max_knowledge_chars: int,
) -> str:
    normalized = _coerce_messages(messages)
    _, conversation_text, focus_text = _prepare_query_input(
        normalized,
        max_messages=max_query_messages,
        max_chars_per_message=max_query_chars,
    )
    user_query = (focus_text or conversation_text or "").strip()

    agent_entries: List[Dict[str, Any]] = []
    agent_idx = 0
    for msg in normalized:
        if str(msg.get("role") or "").lower() != "agent":
            continue
        agent_idx += 1
        label = f"step{agent_idx}"
        content = _shorten_for_query(
            _stringify_for_query(msg.get("content", "")),
            max_agent_chars,
        )
        knowledge_cues = _extract_knowledge_cues(
            msg,
            max_knowledge_per_step,
            max_knowledge_chars,
        )
        agent_entries.append({"label": label, "content": content, "knowledge": knowledge_cues})
    limit_steps = max(int(max_agent_steps), 1)
    recent_entries = agent_entries[-limit_steps:]

    sections: List[str] = []
    sections.append(f"User query:\n{user_query or '[missing user query]'}")

    findings_lines: List[str] = []
    for entry in recent_entries:
        content = entry.get("content") or ""
        if not content:
            continue
        findings_lines.append(entry["label"])
        findings_lines.append(content)
        findings_lines.append("")
    findings_block = "\n".join(findings_lines).strip()
    if findings_block:
        sections.append("Recent agent findings:\n" + findings_block)

    knowledge_lines: List[str] = []
    for entry in recent_entries:
        cues: List[str] = entry.get("knowledge") or []
        if not cues:
            continue
        knowledge_lines.append(entry["label"])
        for cue in cues:
            knowledge_lines.append(f"- {cue}")
        knowledge_lines.append("")
    knowledge_block = "\n".join(knowledge_lines).strip()
    if knowledge_block:
        sections.append("Selected knowledge cues:\n" + knowledge_block)

    body = "\n\n".join(sections).strip()
    if not body:
        body = "[missing query context]"
    return f"<query>\n{body}\n</query>"


def _strip_last_agent(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [dict(msg) for msg in messages if isinstance(msg, dict)]
    for idx in range(len(normalized) - 1, -1, -1):
        role = str(normalized[idx].get("role") or "").lower()
        if role == "agent":
            return normalized[:idx] + normalized[idx + 1 :]
    return normalized


def _format_key_text(entry: Dict[str, Any]) -> str:
    text = str(entry.get("text") or entry.get("knowledge") or "").strip()
    if not text:
        text = "[missing knowledge text]"
    tags = [
        str(tag).strip()
        for tag in (entry.get("tags") or [])
        if str(tag).strip()
    ]
    if tags:
        return f"{text}\nTags: {', '.join(tags)}"
    return text


def _format_prompt(query_block: str, key_text: str) -> str:
    query = (query_block or "").strip()
    if not query:
        query = "<query>\n[missing user query]\n</query>"
    key = (key_text or "").strip() or "[missing knowledge text]"
    return f"{query}\n\n\n<key>\n{key}\n</key>\n\n\nQuery-Key Relevance Score:"


def _build_batch_entry(prompt: str) -> Dict[str, Any]:
    return {"msg": [{"role": "user", "content": prompt}]}


def _coerce_comparisons(
    comparisons: Iterable[Any],
    batch_size: int,
) -> List[List[int]]:
    pairs: List[List[int]] = []
    for pair in comparisons or []:
        if (
            isinstance(pair, Sequence)
            and len(pair) == 2
            and all(isinstance(idx, int) for idx in pair)
        ):
            chosen, rejected = int(pair[0]), int(pair[1])
            if 0 <= chosen < batch_size and 0 <= rejected < batch_size:
                pairs.append([chosen, rejected])
    return pairs


def convert_entry(
    entry: Dict[str, Any],
    *,
    query_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    messages = entry.get("message") or []
    trimmed_messages = _strip_last_agent(messages)
    query_block = _build_query_block(trimmed_messages, **query_kwargs)
    knowledge_entries = entry.get("knowledge") or []
    batch: List[Dict[str, Any]] = []
    for knowledge_entry in knowledge_entries:
        if not isinstance(knowledge_entry, dict):
            continue
        key_text = _format_key_text(knowledge_entry)
        prompt = _format_prompt(query_block, key_text)
        batch.append(_build_batch_entry(prompt))
    dpo_pairs = _coerce_comparisons(entry.get("comparison") or [], len(batch))
    return {"batch": batch, "dpo_pairs": dpo_pairs}


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    raw_data = json.loads(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Input JSON must be a list of tracker entries.")

    query_kwargs = {
        "max_query_messages": max(args.max_query_messages, 1),
        "max_query_chars": max(args.max_query_chars, 1),
        "max_agent_steps": max(args.max_agent_steps, 1),
        "max_agent_chars": max(args.max_agent_chars, 1),
        "max_knowledge_per_step": max(args.max_knowledge_per_step, 1),
        "max_knowledge_chars": max(args.max_knowledge_chars, 1),
    }

    dataset_list: List[Dict[str, Any]] = []
    for entry in raw_data:
        if not isinstance(entry, dict):
            continue
        converted = convert_entry(entry, query_kwargs=query_kwargs)
        if converted["batch"]:
            dataset_list.append(converted)

    if not dataset_list:
        raise RuntimeError("No dataset entries were produced from the input file.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    json_blob = json.dumps(dataset_list, indent=args.indent, ensure_ascii=False)
    args.output_path.write_text(json_blob, encoding="utf-8")
    print(f"Converted {len(dataset_list)} entries -> {args.output_path}")


if __name__ == "__main__":
    main()
