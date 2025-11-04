from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from seimei.llm import LLMClient

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate agent knowledge entries using an LLM and write them to CSV."
    )
    default_prompt = Path(__file__).with_name("generators") / "prompt_test1.md"
    parser.add_argument(
        "--prompt",
        type=Path,
        default=default_prompt,
        help=f"Path to the prompt template (default: {default_prompt})",
    )
    default_output = Path("seimei_knowledge") / "knowledge.csv"
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=(
            "Destination CSV file for generated knowledge "
            f"(default: {default_output})."
        ),
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["think", "code_act", "answer", "web_search"],
        help="List of agent identifiers to cover.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Approximate number of knowledge sentences to request per agent.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name for the LLMClient (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Optional API key (falls back to environment variables if omitted).",
    )
    parser.add_argument(
        "--system",
        dest="system_prompt",
        default="You are a precise knowledge curator for SEIMEI agents.",
        help="Override the system prompt used for the LLM call.",
    )
    return parser.parse_args()


async def _generate(args: argparse.Namespace) -> Dict[str, Any]:
    prompt_template = args.prompt.read_text(encoding="utf-8")
    prompt_text = _prepare_prompt(prompt_template, args.agents, args.count)

    client = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    response, usage = await client.chat(
        messages=[{"role": "user", "content": prompt_text}],
        system=args.system_prompt,
        temperature=0,
    )
    knowledge_entries = _parse_response(response)
    _write_csv(knowledge_entries, args.output)
    return {"usage": usage, "count": len(knowledge_entries), "output": str(args.output)}


def _prepare_prompt(template: str, agents: Sequence[str], count: int) -> str:
    agent_block = "\n".join(f"- {a}" for a in agents)
    prompt = template.replace("<<AGENT_LIST>>", agent_block)
    prompt = prompt.replace("<<TARGET_COUNT>>", str(max(count, 1)))
    return prompt


def _parse_response(text: str) -> List[Dict[str, Any]]:
    if not text:
        raise RuntimeError("LLM returned empty response; cannot parse knowledge entries.")
    match = _JSON_BLOCK_RE.search(text)
    payload = (match.group(1) if match else text).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM JSON output: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError("LLM output must be a JSON array.")
    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"Knowledge entry at index {idx} is not an object.")
        agent = str(item.get("agent", "")).strip()
        knowledge = str(item.get("knowledge", "")).strip()
        tags = item.get("tags", [])
        if not agent or not knowledge:
            raise RuntimeError(f"Knowledge entry at index {idx} missing agent or knowledge text.")
        if isinstance(tags, str):
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif isinstance(tags, Iterable):
            tags_list = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            tags_list = []
        cleaned.append({"agent": agent, "knowledge": knowledge, "tags": tags_list})
    return cleaned


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["agent", "knowledge", "tags"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["tags"] = json.dumps(row.get("tags", []), ensure_ascii=False)
            writer.writerow(serialized)


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(_generate(args))
    except Exception as exc:
        raise SystemExit(f"[generate_from_generators] Failed: {exc}") from exc
    print(
        f"[generate_from_generators] Wrote {result['count']} knowledge entries to {result['output']}. "
        f"Usage: {result['usage']}"
    )


if __name__ == "__main__":
    main()
