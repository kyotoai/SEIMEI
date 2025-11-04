from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from seimei.llm import format_agent_history, LLMClient

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_SNIPPET_CHARS = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SEIMEI knowledge entries by analysing completed agent runs."
        )
    )
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="Run directory names located under the runs directory (default: seimei_runs).",
    )
    default_output = Path("seimei_knowledge") / "knowledge.csv"
    parser.add_argument(
        "--save-file-path",
        type=Path,
        default=default_output,
        help=f"CSV file to append generated knowledge to (default: {default_output}).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("seimei_runs"),
        help="Directory that stores run artefacts (default: seimei_runs).",
    )
    default_prompt = Path(__file__).with_name("generate_from_runs.md")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=default_prompt,
        help=f"Prompt template to load (default: {default_prompt}).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name used by the underlying LLMClient (default: gpt-4o-mini).",
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
        default="You are a rigorous SEIMEI run retrospection expert.",
        help="Override the system prompt supplied to the LLM call.",
    )
    return parser.parse_args()


async def generate_knowledge_from_runs(
    run_ids: Sequence[str],
    *,
    save_file_path: Path = Path("seimei_knowledge") / "knowledge.csv",
    runs_dir: Path = Path("seimei_runs"),
    prompt_path: Optional[Path] = None,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: str = "You are a rigorous SEIMEI run retrospection expert.",
) -> Dict[str, Any]:
    if not run_ids:
        raise ValueError("No run IDs provided.")

    prompt_path = prompt_path or Path(__file__).with_name("generate_from_runs.md")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    run_context_text = _build_run_context(
        run_ids=run_ids,
        runs_dir=runs_dir,
        live_messages=messages,
    )
    prompt_text = prompt_template.replace("<<RUN_CONTEXT>>", run_context_text)

    client = LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    response, usage = await client.chat(
        messages=[{"role": "user", "content": prompt_text}],
        system=system_prompt,
    )
    knowledge_entries = _parse_response(response)
    _append_csv(knowledge_entries, save_file_path)
    return {
        "usage": usage,
        "count": len(knowledge_entries),
        "output": str(save_file_path),
        "runs": list(run_ids),
    }


def _build_run_context(
    run_ids: Sequence[str],
    runs_dir: Path,
    live_messages: Optional[Sequence[Dict[str, Any]]],
) -> str:
    runs_dir = runs_dir.expanduser()
    messages_by_run: Dict[str, Sequence[Dict[str, Any]]] = {}
    if live_messages and run_ids:
        messages_by_run[run_ids[-1]] = live_messages

    sections: List[str] = []
    for run_id in run_ids:
        section = _render_run_section(
            run_id=run_id,
            runs_dir=runs_dir,
            messages_override=messages_by_run.get(run_id),
        )
        sections.append(section)
    return "\n\n".join(sections)


def _render_run_section(
    run_id: str,
    runs_dir: Path,
    messages_override: Optional[Sequence[Dict[str, Any]]],
) -> str:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    meta = _safe_load_json(run_dir / "meta.json")
    run_name = meta.get("run_name") if isinstance(meta, dict) else None
    dataset_row = _safe_load_json(run_dir / "messages.json")
    conversation = (
        list(messages_override)
        if messages_override is not None
        else _normalize_messages(dataset_row)
    )
    steps = _load_steps(run_dir / "steps.jsonl")
    final_output = ""
    if conversation:
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                final_output = str(msg.get("content", "")).strip()
                break
    if not final_output:
        output_path = run_dir / "output.txt"
        if output_path.exists():
            final_output = output_path.read_text(encoding="utf-8").strip()

    header = f"Run ID: {run_id}"
    if run_name:
        header += f" (name: {run_name})"
    lines = [header]

    if meta and isinstance(meta, dict):
        total_time = meta.get("timings", {}).get("total_sec")
        token_usage = meta.get("usage", {})
        usage_summary = ", ".join(
            f"{key}={value}" for key, value in token_usage.items() if value is not None
        )
        if total_time is not None or usage_summary:
            summary_parts = []
            if total_time is not None:
                summary_parts.append(f"duration≈{total_time:.1f}s")
            if usage_summary:
                summary_parts.append(f"usage({usage_summary})")
            if summary_parts:
                lines.append("Meta: " + ", ".join(summary_parts))

    if conversation:
        lines.append("Conversation recap:")
        lines.extend(_format_conversation(conversation))

    if steps:
        lines.append("Agent steps:")
        for step in steps:
            lines.extend(_format_step(step))

    if final_output:
        lines.append("Final assistant output:")
        lines.append(_clip_text(final_output, indent="    "))

    return "\n".join(lines)


def _safe_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _normalize_messages(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [msg for msg in raw if isinstance(msg, dict)]
    return []


def _format_conversation(messages: Sequence[Dict[str, Any]]) -> List[str]:
    # Reuse utility to format agent outputs when available.
    agent_messages: List[Dict[str, Any]] = [
        msg for msg in messages if msg.get("role") == "agent"
    ]
    formatted_agent_history = format_agent_history(agent_messages) if agent_messages else ""

    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        if role == "system":
            continue
        if role == "agent":
            continue
        name = msg.get("name")
        label = role or "message"
        if name:
            label = f"{label} ({name})"
        snippet = _clip_text(msg.get("content", ""))
        lines.append(f"    - {label}: {snippet}")

    if formatted_agent_history:
        lines.append("    - agent_history:")
        for block in formatted_agent_history.splitlines():
            lines.append(f"        {block}")

    return lines


def _load_steps(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    steps: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    steps.append(data)
            except json.JSONDecodeError:
                continue
    return steps


def _format_step(step: Dict[str, Any]) -> List[str]:
    idx = step.get("step")
    agent = step.get("agent") or "unknown"
    result = step.get("result") or {}
    lines = [f"    - Step {idx}: agent={agent}"]
    content = result.get("content")
    if content:
        lines.append(_clip_text(content, prefix="        content: "))
    log = result.get("log")
    if isinstance(log, dict):
        for key, value in log.items():
            if value is None:
                continue
            lines.append(_clip_text(value, prefix=f"        log[{key}]: "))
    chosen = result.get("chosen_instructions")
    if chosen:
        lines.append(_clip_text(chosen, prefix="        chosen_instructions: "))
    status_flags: List[str] = []
    if result.get("stop"):
        status_flags.append("stop=True")
    if "final_output" in result:
        status_flags.append("final_output=True")
    if status_flags:
        lines.append(f"        flags: {', '.join(status_flags)}")
    return lines


def _clip_text(value: Any, prefix: str = "        ", indent: str = "", limit: int = _MAX_SNIPPET_CHARS) -> str:
    text = ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.strip()
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    if indent:
        return f"{indent}{text}"
    return f"{prefix}{text}"


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


def _append_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["agent", "knowledge", "tags"]

    write_header = True
    if output_path.exists():
        try:
            write_header = output_path.stat().st_size == 0
        except OSError:
            write_header = False

    with output_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["tags"] = json.dumps(row.get("tags", []), ensure_ascii=False)
            writer.writerow(serialized)


async def _generate(args: argparse.Namespace) -> Dict[str, Any]:
    return await generate_knowledge_from_runs(
        run_ids=args.run_ids,
        save_file_path=args.save_file_path,
        runs_dir=args.runs_dir,
        prompt_path=args.prompt,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
    )


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(_generate(args))
    except Exception as exc:
        raise SystemExit(f"[generate_from_runs] Failed: {exc}") from exc
    print(
        "[generate_from_runs] Appended "
        f"{result['count']} knowledge entries to {result['output']} using runs {', '.join(result['runs'])}. "
        f"Usage: {result['usage']}"
    )


if __name__ == "__main__":
    main()

