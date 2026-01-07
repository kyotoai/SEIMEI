#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numbers
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_PATH = Path("exp11_plasma_gkv_v3/train_v4_eval_sample_results2.json")
DEFAULT_OUTPUT_PATH_TRAIN = Path("exp11_plasma_gkv_v3/dataset_list_train.json")
DEFAULT_OUTPUT_PATH_TEST = Path("exp11_plasma_gkv_v3/dataset_list_test.json")
MAX_AGENT_CHARS = 3000

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert train_v3 DPO runs into dataset_list format."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the input JSON produced by exp8_csv_small/train_v3.py",
    )
    parser.add_argument(
        "--output-path-train",
        type=Path,
        default=DEFAULT_OUTPUT_PATH_TRAIN,
        help="Where to write the training dataset_list payload.",
    )
    parser.add_argument(
        "--output-path-test",
        type=Path,
        default=DEFAULT_OUTPUT_PATH_TEST,
        help="Where to write the test dataset_list payload.",
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
        default=MAX_AGENT_CHARS,
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
    parser.add_argument(
        "--more-dpo-pairs",
        action="store_true",
        help="If provided, add all possible DPO pairs derived from the score ordering.",
    )
    parser.add_argument(
        "--n-sample-other-knowledge",
        type=int,
        default=0,
        help="Number of knowledge entries sampled from other ids to append as score-0 negatives.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset reserved for the test split (between 0 and 1).",
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


def _coerce_score_value(value: Any) -> Optional[float]:
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_score_vector(
    raw_scores: Any,
    num_base_entries: int,
    num_extra_entries: int,
    *,
    extra_score_value: float = 0.0,
) -> List[Optional[float]]:
    base_scores: List[Optional[float]] = []
    if isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes, bytearray)):
        for value in list(raw_scores)[:num_base_entries]:
            base_scores.append(_coerce_score_value(value))
    if len(base_scores) < num_base_entries:
        base_scores.extend([None] * (num_base_entries - len(base_scores)))
    elif len(base_scores) > num_base_entries:
        base_scores = base_scores[:num_base_entries]
    if num_extra_entries > 0:
        base_scores.extend([float(extra_score_value)] * num_extra_entries)
    return base_scores


def _generate_pairs_from_scores(
    scores: Sequence[Optional[float]],
    batch_size: int,
    existing_pairs: Sequence[Sequence[int]],
) -> List[List[int]]:
    extra_pairs: List[List[int]] = []
    if not scores:
        return extra_pairs
    existing_set = {
        (int(pair[0]), int(pair[1]))
        for pair in existing_pairs
        if isinstance(pair, Sequence)
        and len(pair) == 2
        and all(isinstance(idx, int) for idx in pair)
    }
    limit = min(len(scores), batch_size)
    for i in range(limit):
        score_i = scores[i]
        if score_i is None:
            continue
        for j in range(i + 1, limit):
            score_j = scores[j]
            if score_j is None or score_i == score_j:
                continue
            chosen, rejected = (i, j) if score_i > score_j else (j, i)
            key = (chosen, rejected)
            if key in existing_set:
                continue
            existing_set.add(key)
            extra_pairs.append([chosen, rejected])
    return extra_pairs


def _entry_identifier(entry: Dict[str, Any], fallback_index: int) -> str:
    entry_id = entry.get("id")
    if entry_id is None:
        return f"__entry_{fallback_index}"
    return str(entry_id)


def _collect_knowledge_pool(
    entries: Sequence[Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    pool: List[Tuple[str, Dict[str, Any]]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        source_id = _entry_identifier(entry, idx)
        for knowledge_entry in entry.get("knowledge") or []:
            if isinstance(knowledge_entry, dict):
                pool.append((source_id, knowledge_entry))
    return pool


def _sample_other_knowledge(
    pool: Sequence[Tuple[str, Dict[str, Any]]],
    current_source_id: str,
    sample_size: int,
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return []
    candidates = [item for src, item in pool if src != current_source_id]
    if not candidates:
        return []
    size = min(sample_size, len(candidates))
    sampled = random.sample(candidates, size)
    extras: List[Dict[str, Any]] = []
    for knowledge_entry in sampled:
        cloned = deepcopy(knowledge_entry)
        cloned["score"] = 0.0
        extras.append(cloned)
    return extras


def convert_entry(
    entry: Dict[str, Any],
    *,
    query_kwargs: Dict[str, Any],
    extra_knowledge: Sequence[Dict[str, Any]] | None = None,
    more_dpo_pairs: bool = False,
) -> Dict[str, Any]:
    messages = entry.get("message") or []
    trimmed_messages = _strip_last_agent(messages)
    query_block = _build_query_block(trimmed_messages, **query_kwargs)
    knowledge_entries = [k for k in (entry.get("knowledge") or []) if isinstance(k, dict)]
    extra_entries = [k for k in (extra_knowledge or []) if isinstance(k, dict)]
    combined_entries = knowledge_entries + extra_entries
    batch: List[Dict[str, Any]] = []
    for knowledge_entry in combined_entries:
        key_text = _format_key_text(knowledge_entry)
        prompt = _format_prompt(query_block, key_text)
        batch.append(_build_batch_entry(prompt))
    score_vector = _prepare_score_vector(
        entry.get("scores"),
        len(knowledge_entries),
        len(extra_entries),
    )
    if len(score_vector) < len(batch):
        score_vector.extend([None] * (len(batch) - len(score_vector)))
    dpo_pairs = _coerce_comparisons(entry.get("comparison") or [], len(batch))
    if more_dpo_pairs and batch:
        additional_pairs = _generate_pairs_from_scores(score_vector, len(batch), dpo_pairs)
        if additional_pairs:
            dpo_pairs = dpo_pairs + additional_pairs
    return {"batch": batch, "dpo_pairs": dpo_pairs}


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    if args.n_sample_other_knowledge < 0:
        raise ValueError("n-sample-other-knowledge must be non-negative.")

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

    knowledge_pool = _collect_knowledge_pool(raw_data)
    dataset_list: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_data):
        if not isinstance(entry, dict):
            continue
        extras: List[Dict[str, Any]] = []
        if args.n_sample_other_knowledge:
            source_id = _entry_identifier(entry, idx)
            extras = _sample_other_knowledge(
                knowledge_pool,
                source_id,
                args.n_sample_other_knowledge,
            )
        converted = convert_entry(
            entry,
            query_kwargs=query_kwargs,
            extra_knowledge=extras,
            more_dpo_pairs=args.more_dpo_pairs,
        )
        if converted["batch"]:
            dataset_list.append(converted)

    if not dataset_list:
        raise RuntimeError("No dataset entries were produced from the input file.")
    total_entries = len(dataset_list)
    random.shuffle(dataset_list)
    test_ratio = float(args.test_ratio)
    if not (0.0 <= test_ratio <= 1.0):
        test_ratio = max(0.0, min(1.0, test_ratio))
    split_idx = int(total_entries * (1.0 - test_ratio))
    split_idx = max(0, min(total_entries, split_idx))
    train_entries = dataset_list[:split_idx]
    test_entries = dataset_list[split_idx:]

    args.output_path_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_path_test.parent.mkdir(parents=True, exist_ok=True)
    train_blob = json.dumps(train_entries, indent=args.indent, ensure_ascii=False)
    test_blob = json.dumps(test_entries, indent=args.indent, ensure_ascii=False)
    args.output_path_train.write_text(train_blob, encoding="utf-8")
    args.output_path_test.write_text(test_blob, encoding="utf-8")
    print(
        "Converted "
        f"{total_entries} entries -> train {len(train_entries)} ({args.output_path_train}), "
        f"test {len(test_entries)} ({args.output_path_test})"
    )


if __name__ == "__main__":
    main()
