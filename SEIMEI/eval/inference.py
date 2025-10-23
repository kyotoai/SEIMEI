from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from seimei import seimei as seimei_orchestrator


def _coerce_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if lowered.startswith(("0x", "0o", "0b")):
            return int(lowered, 0)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_kv_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, received '{item}'.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key detected in pair '{item}'.")
        values[key] = _coerce_value(value.strip())
    return values


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Dataset at {path} is not a JSON list.")
    return [dict(item) for item in data]


def resolve_dataset_path(raw: str, exp_dir_arg: Path, exp_dir_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    try:
        relative = path.relative_to(exp_dir_arg)
        return (exp_dir_path / relative).resolve()
    except ValueError:
        return (Path.cwd() / path).resolve()


def read_csv_preview(csv_path: Path, limit: int) -> List[List[str]]:
    rows: List[List[str]] = []
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx >= limit:
                break
    return rows


def to_table(rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(",".join(cell or "" for cell in row) for row in rows)


def read_script_excerpt(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_user_prompt(
    record: Dict[str, Any],
    csv_path: Path,
    python_path: Path,
    csv_preview: List[List[str]],
    script_excerpt: str,
) -> str:
    preview_text = to_table(csv_preview) if csv_preview else "Preview unavailable."
    code_block = f"```python\n{script_excerpt}\n```" if script_excerpt else "Python generator file unavailable."
    return (
        "Evaluate the following synthetic dataset scenario.\n\n"
        f"Question:\n{record.get('Question', '').strip()}\n\n"
        f"CSV file path: {csv_path}\n"
        f"Python generator path: {python_path}\n\n"
        "CSV preview (top rows):\n"
        f"{preview_text}\n\n"
        f"{code_block}\n\n"
        "Use the available tools (including the code_act agent if permitted) to inspect the resources as needed. "
        "Provide a clear answer to the question, citing the hyper-parameters or data characteristics that justify it."
    )


async def run_inference(args: argparse.Namespace) -> List[Dict[str, Any]]:
    exp_dir_arg = Path(args.exp_dir)
    exp_dir_path = Path(args.exp_dir).resolve()
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else exp_dir_path / "dataset.json"
    result_path = Path(args.result_path).resolve() if args.result_path else exp_dir_path / "result.json"

    dataset = load_dataset(dataset_path)

    agent_config: List[Dict[str, Any]] = []
    for directory in args.agent_dir or []:
        agent_config.append({"dir_path": directory})
    for file_path in args.agent_file or []:
        agent_config.append({"file_path": file_path})
    if not agent_config:
        agent_config.append({"dir_path": "seimei/agents"})

    llm_kwargs = {"model": args.model, **args.llm_kwargs}
    rm_kwargs = dict(args.rm_kwargs)

    orchestrator = seimei_orchestrator(
        agent_config=agent_config,
        llm_kwargs=llm_kwargs,
        rm_kwargs=rm_kwargs,
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        allow_code_exec=args.allow_code_exec,
        allowed_commands=args.allowed_command or None,
        agent_log_head_lines=args.agent_log_head_lines,
        max_tokens_per_question=args.max_tokens_per_question,
    )

    results: List[Dict[str, Any]] = []
    for index, record in enumerate(dataset, start=1):
        csv_path = resolve_dataset_path(record.get("CSVPath", ""), exp_dir_arg, exp_dir_path)
        python_path = resolve_dataset_path(record.get("PythonPath", ""), exp_dir_arg, exp_dir_path)
        csv_preview = record.get("CSVPreview") or read_csv_preview(csv_path, args.preview_rows)
        script_excerpt = read_script_excerpt(python_path, args.script_excerpt_chars)

        messages: List[Dict[str, Any]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": build_user_prompt(record, csv_path, python_path, csv_preview, script_excerpt),
            }
        )

        run_name = f"{args.name}-{index:03d}" if args.name else None
        print(f"[inference] Running seimei for sample {index}/{len(dataset)} (run_name={run_name or 'auto'})")
        run_result = await orchestrator(messages=messages, run_name=run_name)

        result_entry = dict(record)
        result_entry["Output"] = run_result.get("output", "")
        result_entry["Log"] = run_result.get("msg_history", [])
        result_entry["RunId"] = run_result.get("run_id")
        usage = run_result.get("usage")
        if usage:
            result_entry["Usage"] = usage
        results.append(result_entry)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"Inference results written to {result_path}")
    return results


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEIMEI inference over a generated dataset.")
    parser.add_argument("--model", default="gpt-5", help="Model name for the seimei LLM client.")
    parser.add_argument("--exp-dir", default="exp1", help="Experiment directory (used to resolve relative paths).")
    parser.add_argument("--dataset-path", help="Path to dataset.json (defaults to <exp_dir>/dataset.json).")
    parser.add_argument("--result-path", help="Where to write result.json (defaults to <exp_dir>/result.json).")
    parser.add_argument("--log-dir", default="./seimei_runs", help="Directory where seimei stores run artifacts.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent steps per question.")
    parser.add_argument(
        "--agent-log-head-lines",
        type=int,
        default=3,
        help="Number of lines from agent output to log to stdout.",
    )
    parser.add_argument(
        "--max-tokens-per-question",
        type=int,
        default=None,
        help="Optional token limit enforced per question.",
    )
    parser.add_argument(
        "--allow-code-exec",
        action="store_true",
        help="Allow the code_act agent to execute shell commands.",
    )
    parser.add_argument(
        "--allowed-command",
        action="append",
        help="Whitelist of shell commands for code_act (repeatable).",
    )
    parser.add_argument("--agent-dir", action="append", help="Load additional agent modules from this directory.")
    parser.add_argument("--agent-file", action="append", help="Load an agent module from this Python file.")
    parser.add_argument(
        "--llm-kw",
        action="append",
        default=[],
        help="Additional key=value pairs forwarded to seimei.llm.LLMClient.",
    )
    parser.add_argument(
        "--rm-kw",
        action="append",
        default=[],
        help="Optional key=value arguments forwarded to rmsearch (if configured).",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are SEIMEI's evaluation assistant. Inspect referenced files and data carefully before answering. "
            "Explain your reasoning succinctly when you deliver the final answer."
        ),
        help="System prompt given to seimei for each question.",
    )
    parser.add_argument("--name", help="Optional base name for seimei runs (appended with an index).")
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=6,
        help="Number of CSV rows to embed in the initial user prompt.",
    )
    parser.add_argument(
        "--script-excerpt-chars",
        type=int,
        default=2000,
        help="Maximum number of characters from the generator script to embed.",
    )
    parsed = parser.parse_args(argv)
    parsed.llm_kwargs = parse_kv_pairs(parsed.llm_kw or [])
    parsed.rm_kwargs = parse_kv_pairs(parsed.rm_kw or [])
    return parsed


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)

    async def _run() -> None:
        await run_inference(args)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
