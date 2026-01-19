#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from seimei.llm import prepare_messages

DEFAULT_INPUT_PATH = Path("exp11_plasma_gkv_v5/train_v6_results_converted.json")
DEFAULT_OUTPUT_PATH_TRAIN = Path("exp11_plasma_gkv_v5/train_v6_datasetlist_train.json")
DEFAULT_OUTPUT_PATH_TEST = Path("exp11_plasma_gkv_v5/train_v6_datasetlist_test.json")
DEFAULT_N_BATCH_ELEMENTS = 10
DEFAULT_COMPARISON_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert train_v6 converted results into dataset_list format."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to train_v6_results_converted.json.",
    )
    parser.add_argument(
        "--output-path-train",
        type=Path,
        default=DEFAULT_OUTPUT_PATH_TRAIN,
        help="Where to write train_v6_datasetlist_train.json.",
    )
    parser.add_argument(
        "--output-path-test",
        type=Path,
        default=DEFAULT_OUTPUT_PATH_TEST,
        help="Where to write train_v6_datasetlist_test.json.",
    )
    parser.add_argument(
        "--n-batch-elements",
        type=int,
        default=DEFAULT_N_BATCH_ELEMENTS,
        help="Exact number of elements per batch.",
    )
    parser.add_argument(
        "--comparison-threshold",
        type=float,
        default=DEFAULT_COMPARISON_THRESHOLD,
        help="Minimum score difference required to create a comparison pair.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset reserved for the test split (between 0 and 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for batch sampling.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for the generated dataset_list files.",
    )
    return parser.parse_args()


def _is_agent_message(message: Dict[str, Any]) -> bool:
    role = str(message.get("role") or "").lower()
    if role == "agent":
        return True
    if role == "system" and message.get("agent"):
        return True
    return False


def _normalize_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role == "assistant":
            continue
        normalized.append(dict(msg))
    return normalized


def _coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_comparisons(
    scores: Sequence[Optional[float]],
    threshold: float,
) -> List[List[int]]:
    pairs: List[List[int]] = []
    for i in range(len(scores)):
        score_i = scores[i]
        if score_i is None:
            continue
        for j in range(i + 1, len(scores)):
            score_j = scores[j]
            if score_j is None:
                continue
            diff = score_i - score_j
            if abs(diff) > threshold:
                chosen, rejected = (i, j) if diff > 0 else (j, i)
                pairs.append([chosen, rejected])
    return pairs


def _coerce_knowledge_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "knowledge", "content", "value"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            text = _coerce_knowledge_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(value)


def _format_query(messages: Sequence[Dict[str, Any]]) -> str:
    prepared, _ = prepare_messages(messages, system=None, drop_normal_system=False)
    segments: List[str] = []
    for msg in prepared:
        content = msg.get("content")
        if content is None:
            continue
        content_text = str(content).strip()
        if content_text:
            segments.append(content_text)
    return "\n\n".join(segments).strip()


def _format_prompt(query_block: str, key_text: str) -> str:
    query = (query_block or "").strip()
    if not query:
        query = "[missing query context]"
    if not (query.startswith("<query>") and "</query>" in query):
        query = f"<query>\n{query}\n</query>"
    key = (key_text or "").strip() or "[missing knowledge text]"
    if not (key.startswith("<key>") and "</key>" in key):
        key = f"<key>\n{key}\n</key>"
    return f"{query}\n\n\n{key}\n\n\nQuery-Key Relevance Score:"


def _messages_before_step(
    messages: Sequence[Dict[str, Any]],
    step: int,
) -> List[Dict[str, Any]]:
    target = max(step - 1, 0)
    if target == 0:
        truncated: List[Dict[str, Any]] = []
        for msg in messages:
            if _is_agent_message(msg):
                break
            truncated.append(dict(msg))
        return truncated

    truncated: List[Dict[str, Any]] = []
    agent_seen = 0
    for msg in messages:
        if _is_agent_message(msg):
            agent_seen += 1
            if agent_seen > target:
                break
        truncated.append(dict(msg))
    return truncated


def _build_batch_entry(prompt: str, msg_id: int, agent_step: int) -> Dict[str, Any]:
    return {"msg": [{"role": "user", "content": prompt}], "msg_id": msg_id, "agent_step": agent_step}


def _build_message_step_map(
    messages: Sequence[Dict[str, Any]],
) -> List[Tuple[int, str]]:
    steps: List[Tuple[int, str]] = []
    agent_step = 0
    for msg in messages:
        if not _is_agent_message(msg):
            continue
        agent_step += 1
        knowledge_text = _coerce_knowledge_text(msg.get("knowledge"))
        steps.append((agent_step, knowledge_text))
    return steps


def _select_batch_pairs(
    remaining_steps: Dict[int, List[int]],
    *,
    n_batch_elements: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    batch_pairs: List[Tuple[int, int]] = []
    active_msg_ids = [mid for mid, steps in remaining_steps.items() if steps]
    if not active_msg_ids:
        return batch_pairs

    base = n_batch_elements // len(active_msg_ids)
    extra = n_batch_elements % len(active_msg_ids)
    extra_ids = set(rng.sample(active_msg_ids, k=extra)) if extra else set()

    for msg_id in active_msg_ids:
        target = base + (1 if msg_id in extra_ids else 0)
        if target <= 0:
            continue
        take = min(target, len(remaining_steps[msg_id]))
        for _ in range(take):
            batch_pairs.append((msg_id, remaining_steps[msg_id].pop()))

    while len(batch_pairs) < n_batch_elements:
        active_msg_ids = [mid for mid, steps in remaining_steps.items() if steps]
        if not active_msg_ids:
            break
        msg_id = rng.choice(active_msg_ids)
        batch_pairs.append((msg_id, remaining_steps[msg_id].pop()))

    return batch_pairs


def _pad_batch_pairs(
    batch_pairs: List[Tuple[int, int]],
    all_pairs: List[Tuple[int, int]],
    *,
    n_batch_elements: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    if len(batch_pairs) >= n_batch_elements:
        return batch_pairs[:n_batch_elements]

    existing = set(batch_pairs)
    remaining_unique = [pair for pair in all_pairs if pair not in existing]
    rng.shuffle(remaining_unique)
    while len(batch_pairs) < n_batch_elements and remaining_unique:
        batch_pairs.append(remaining_unique.pop())

    while len(batch_pairs) < n_batch_elements:
        batch_pairs.append(rng.choice(all_pairs))
    return batch_pairs


def _build_batches_for_problem(
    msg_steps: Dict[int, List[int]],
    *,
    n_batch_elements: int,
    rng: random.Random,
) -> List[List[Tuple[int, int]]]:
    remaining_steps: Dict[int, List[int]] = {}
    for msg_id, steps in msg_steps.items():
        shuffled = list(steps)
        rng.shuffle(shuffled)
        remaining_steps[msg_id] = shuffled

    all_pairs = [
        (msg_id, step)
        for msg_id, steps in msg_steps.items()
        for step in steps
    ]
    if not all_pairs:
        return []

    batches: List[List[Tuple[int, int]]] = []
    while any(remaining_steps.values()):
        batch_pairs = _select_batch_pairs(
            remaining_steps,
            n_batch_elements=n_batch_elements,
            rng=rng,
        )
        batch_pairs = _pad_batch_pairs(
            batch_pairs,
            all_pairs,
            n_batch_elements=n_batch_elements,
            rng=rng,
        )
        batches.append(batch_pairs)

    if not batches:
        batch_pairs = _pad_batch_pairs([], all_pairs, n_batch_elements=n_batch_elements, rng=rng)
        batches.append(batch_pairs)
    return batches


def convert_problem(
    entry: Dict[str, Any],
    *,
    n_batch_elements: int,
    comparison_threshold: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    message_entries = entry.get("messages") or []
    if not isinstance(message_entries, list):
        return []

    msg_data: Dict[int, Dict[str, Any]] = {}
    msg_steps: Dict[int, List[int]] = {}
    for msg_id, msg_entry in enumerate(message_entries):
        if not isinstance(msg_entry, dict):
            continue
        raw_messages = msg_entry.get("message") or []
        normalized = _normalize_messages(raw_messages)
        steps = _build_message_step_map(normalized)
        if not steps:
            continue
        step_indices = [step_idx for step_idx, _ in steps]
        step_knowledge = {step_idx: knowledge for step_idx, knowledge in steps}
        msg_data[msg_id] = {
            "messages": normalized,
            "score": _coerce_score(msg_entry.get("score")),
            "knowledge": step_knowledge,
        }
        msg_steps[msg_id] = step_indices

    if not msg_data:
        return []

    batches = _build_batches_for_problem(
        msg_steps,
        n_batch_elements=max(int(n_batch_elements), 1),
        rng=rng,
    )

    output_entries: List[Dict[str, Any]] = []
    for batch_pairs in batches:
        batch: List[Dict[str, Any]] = []
        scores: List[Optional[float]] = []

        for msg_id, step_idx in batch_pairs:
            payload = msg_data.get(msg_id)
            if not payload:
                continue
            messages = payload["messages"]
            knowledge_text = payload["knowledge"].get(step_idx, "")
            query_messages = _messages_before_step(messages, step_idx)
            query_block = _format_query(query_messages)
            prompt = _format_prompt(query_block, knowledge_text)
            batch.append(_build_batch_entry(prompt, msg_id, step_idx))
            scores.append(payload.get("score"))

        if not batch:
            continue

        msg_groups_map: Dict[int, List[int]] = {}
        for idx, elem in enumerate(batch):
            msg_groups_map.setdefault(int(elem["msg_id"]), []).append(idx)
        msg_groups = [msg_groups_map[msg_id] for msg_id in sorted(msg_groups_map)]
        dpo_pairs = _build_comparisons(scores, comparison_threshold)

        output_entries.append(
            {
                "batch": batch,
                "msg_groups": msg_groups,
                "dpo_pairs": dpo_pairs,
                "scores": scores,
            }
        )

    return output_entries


def _average_msg_chars(dataset_list: Sequence[Dict[str, Any]]) -> float:
    total_chars = 0
    total_count = 0
    for entry in dataset_list:
        batch = entry.get("batch") or []
        for item in batch:
            msg = item.get("msg") or []
            if msg and isinstance(msg, list):
                content = msg[0].get("content", "")
                total_chars += len(str(content))
                total_count += 1
    if not total_count:
        return 0.0
    return total_chars / total_count


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    raw_data = json.loads(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Input JSON must be a list of entries.")

    rng = random.Random(args.seed)
    dataset_list: List[Dict[str, Any]] = []
    for entry in raw_data:
        if not isinstance(entry, dict):
            continue
        dataset_list.extend(
            convert_problem(
                entry,
                n_batch_elements=max(int(args.n_batch_elements), 1),
                comparison_threshold=float(args.comparison_threshold),
                rng=rng,
            )
        )

    if not dataset_list:
        raise RuntimeError("No dataset entries were produced from the input file.")

    avg_chars = _average_msg_chars(dataset_list)
    print(f"Average msg content chars: {avg_chars:.2f}")

    total_entries = len(dataset_list)
    rng.shuffle(dataset_list)
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
