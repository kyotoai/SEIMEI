from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .SEIMEI import seimei as SeimeiOrchestrator
from .logging_utils import LogColors, colorize, supports_color

# -------- Default knobs (easy to tweak without hunting through the code) --------

CLI_VERSION = "0.1.0"
CLI_NAME = "SEIMEI CLI"
ASCII_ART = r"""
  _____  ______ _____ __  __ ______ _____ 
 / ____|/ ____|_   _|  \/  |  ____|_   _|
| (___ | |__    | | | \  / | |__    | |  
 \___ \|  __|   | | | |\/| |  __|   | |  
 ____) | |____ _| |_| |  | | |____ _| |_ 
|_____/ \_____|_____|_|  |_|______|_____|
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are an execution assistant that never runs unasked commands."
)
DEFAULT_AGENT_CONFIG: List[Dict[str, Any]] = [
    {"file_path": "seimei/agents/code_act.py"},
]
DEFAULT_LLM_KWARGS: Dict[str, Any] = {
    "model": "gpt-5-mini",
}

# rmsearch: https://j4s6oyznxb8j3v-8000.proxy.runpod.net/rmsearch
# emnbed: https://si3trdzr984v57-8000.proxy.runpod.net/embed
DEFAULT_RM_KWARGS: Dict[str, Any] = {"url":"https://si3trdzr984v57-8000.proxy.runpod.net/embed", "agent_routing":False, "knowledge_search":True}
DEFAULT_MAX_STEPS = 12
DEFAULT_AGENT_LOG_HEAD_LINES = 1
DEFAULT_ALLOW_CODE_EXEC = True
DEFAULT_ALLOWED_COMMANDS: Optional[Sequence[str]] = None
DEFAULT_MAX_TOKENS_PER_QUESTION = 40_000
DEFAULT_LOAD_KNOWLEDGE_PATH = "seimei_knowledge/yc_demo_knowledge4.csv"
DEFAULT_SAVE_KNOWLEDGE_PATH = "seimei_knowledge/yc_demo_knowledge4_output.csv"
DEFAULT_KNOWLEDGE_PROMPT_PATH = "seimei/knowledge/prompts/user_intent_alignment3.md"
DEFAULT_GENERATE_KNOWLEDGE = True
DEFAULT_LOG_DIR = "./seimei_runs"


HELP_TEXT = textwrap.dedent(
    """
    Commands:
      /help       Show this help text
      /exit       Quit the SEIMEI CLI
      /save [p]   Save full history (incl. agent logs) to JSON at optional path
      /reset      Clear the conversation history and start over
    """
).strip()


@dataclass
class CLIArgs:
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    log_dir: str
    max_steps: int
    agent_files: List[str]
    agent_dirs: List[str]
    system_prompt: Optional[str]
    generate_knowledge: bool
    knowledge_file: Optional[str]
    knowledge_prompt: Optional[str]
    load_knowledge_path: Optional[str]
    max_tokens: Optional[int]


def parse_args(argv: Optional[Sequence[str]] = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="seimei",
        description="Interactive SEIMEI CLI for chatting with the orchestrator.",
    )
    parser.add_argument("--model", help="Override the default model name.")
    parser.add_argument("--base-url", help="Override the LLM base URL.")
    parser.add_argument("--api-key", help="LLM API key (overrides environment).")
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory to store run logs (default: {DEFAULT_LOG_DIR}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum agent steps per question (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--agent-file",
        action="append",
        default=[],
        help="Additional agent file to load (can be repeated).",
    )
    parser.add_argument(
        "--agent-dir",
        action="append",
        default=[],
        help="Additional directory of agents to load (can be repeated).",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the default system prompt sent with every conversation.",
    )
    parser.add_argument(
        "--no-knowledge",
        action="store_true",
        help="Disable automatic knowledge generation.",
    )
    parser.add_argument(
        "--knowledge-file",
        default=DEFAULT_SAVE_KNOWLEDGE_PATH,
        help=f"Knowledge CSV path (default: {DEFAULT_SAVE_KNOWLEDGE_PATH}).",
    )
    parser.add_argument(
        "--knowledge-prompt",
        default=DEFAULT_KNOWLEDGE_PROMPT_PATH,
        help=f"Knowledge prompt template path (default: {DEFAULT_KNOWLEDGE_PROMPT_PATH}).",
    )
    parser.add_argument(
        "--load-knowledge-path",
        default=DEFAULT_LOAD_KNOWLEDGE_PATH,
        help=f"Knowledge CSV to load at startup (default: {DEFAULT_LOAD_KNOWLEDGE_PATH}).",
    )
    parser.add_argument(
        "--no-load-knowledge",
        action="store_true",
        help="Skip loading existing knowledge at startup.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_QUESTION,
        help=f"Approximate max tokens per question (default: {DEFAULT_MAX_TOKENS_PER_QUESTION}).",
    )

    parsed = parser.parse_args(argv)
    load_path = None if parsed.no_load_knowledge else parsed.load_knowledge_path
    return CLIArgs(
        model=parsed.model,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        log_dir=parsed.log_dir,
        max_steps=parsed.max_steps,
        agent_files=parsed.agent_file or [],
        agent_dirs=parsed.agent_dir or [],
        system_prompt=parsed.system_prompt,
        generate_knowledge=not parsed.no_knowledge,
        knowledge_file=None if parsed.no_knowledge else parsed.knowledge_file,
        knowledge_prompt=None if parsed.no_knowledge else parsed.knowledge_prompt,
        load_knowledge_path=load_path,
        max_tokens=parsed.max_tokens if parsed.max_tokens > 0 else None,
    )


def build_agent_config(args: CLIArgs) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for path in args.agent_dirs:
        configs.append({"dir_path": path})
    for path in args.agent_files:
        configs.append({"file_path": path})
    if not configs:
        configs.extend(DEFAULT_AGENT_CONFIG)
    return configs


def build_llm_kwargs(args: CLIArgs) -> Dict[str, Any]:
    llm_kwargs = dict(DEFAULT_LLM_KWARGS)
    if args.model:
        llm_kwargs["model"] = args.model
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    if args.api_key:
        llm_kwargs["api_key"] = args.api_key
    return llm_kwargs


def ensure_parent_dir(path: Optional[Path]) -> None:
    if not path:
        return
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def render_header(
    session_id: str,
    *,
    color_enabled: bool,
) -> None:
    banner = colorize(ASCII_ART.rstrip(), LogColors.GREEN, enable=color_enabled)
    print(banner)
    version = resolve_version()
    cli_line = colorize(f"{CLI_NAME} v{version}", LogColors.GRAY, enable=color_enabled)
    init_line = colorize(
        f"Initialized conversation {session_id}", LogColors.GRAY, enable=color_enabled
    )
    print(f"\n{cli_line}")
    print(f"\n{init_line}")
    print("\nWhat is on your mind today?\n")


def print_new_messages(
    messages: Sequence[Dict[str, Any]],
    start_idx: int,
    *,
    color_enabled: bool,
) -> int:
    if start_idx >= len(messages):
        return start_idx

    for msg in messages[start_idx:]:
        role = str(msg.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        label = "You" if role == "user" else "SEIMEI"
        color = LogColors.CYAN if role == "user" else LogColors.BOLD_MAGENTA
        content = str(msg.get("content") or "").strip() or "[no content]"
        label_text = colorize(f"{label}:", color, enable=color_enabled)
        print(f"{label_text} {content}\n")

    return len(messages)


def resolve_version() -> str:
    try:
        from importlib.metadata import version

        return version("SEIMEI")
    except Exception:
        return CLI_VERSION


async def ainput(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def run_cli(args: CLIArgs) -> None:
    llm_kwargs = build_llm_kwargs(args)
    agent_config = build_agent_config(args)
    knowledge_file = Path(args.knowledge_file).expanduser() if args.knowledge_file else None
    knowledge_prompt = (
        Path(args.knowledge_prompt).expanduser() if args.knowledge_prompt else None
    )
    ensure_parent_dir(knowledge_file)

    orchestrator = SeimeiOrchestrator(
        agent_config=agent_config,
        llm_kwargs=llm_kwargs,
        rm_kwargs=dict(DEFAULT_RM_KWARGS),
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        allow_code_exec=DEFAULT_ALLOW_CODE_EXEC,
        allowed_commands=list(DEFAULT_ALLOWED_COMMANDS) if DEFAULT_ALLOWED_COMMANDS else None,
        agent_log_head_lines=DEFAULT_AGENT_LOG_HEAD_LINES,
        max_tokens_per_question=args.max_tokens,
        load_knowledge_path=args.load_knowledge_path,
    )

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    message_history: List[Dict[str, Any]] = []
    if system_prompt:
        message_history.append({"role": "system", "content": system_prompt})
    session_id = f"{uuid.uuid4()}-{uuid.uuid4().hex}"
    color_enabled = supports_color()
    prompt_text = colorize("> ", LogColors.GREEN, enable=color_enabled)
    full_history: List[Dict[str, Any]] = list(message_history)
    turn = 0
    render_header(session_id, color_enabled=color_enabled)
    last_rendered_len = len(message_history)

    while True:
        try:
            user_input = (await ainput(prompt_text)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SEIMEI CLI.")
            return

        if not user_input:
            continue

        if user_input in {"/exit", "/quit", "/q"}:
            print("Bye!")
            break

        if user_input == "/help":
            print("\n" + HELP_TEXT + "\n")
            continue

        if user_input == "/reset":
            message_history = []
            if system_prompt:
                message_history.append({"role": "system", "content": system_prompt})
            full_history = list(message_history)
            last_rendered_len = len(message_history)
            print(colorize("[reset] Conversation cleared.", LogColors.YELLOW, enable=color_enabled))
            continue

        if user_input.startswith("/save"):
            default_name = f"seimei_session_{session_id}.json"
            parts = user_input.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else default_name
            save_path = Path(target).expanduser()
            try:
                ensure_parent_dir(save_path)
                save_path.write_text(
                    json.dumps(full_history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    colorize(
                        f"[saved] Conversation history → {save_path}",
                        LogColors.GREEN,
                        enable=color_enabled,
                    )
                )
            except OSError as exc:
                print(
                    colorize(
                        f"[error] Failed to save history: {exc}",
                        LogColors.RED,
                        enable=color_enabled,
                    )
                )
            continue

        user_message = {"role": "user", "content": user_input}
        message_history.append(user_message)
        full_history.append(dict(user_message))
        last_rendered_len = print_new_messages(
            message_history, last_rendered_len, color_enabled=color_enabled
        )
        turn += 1

        try:
            result = await orchestrator(
                messages=list(message_history),
                #run_name=f"cli-{session_id}-turn-{turn}",
                generate_knowledge=args.generate_knowledge,
                save_knowledge_path=str(knowledge_file) if knowledge_file else None,
                knowledge_prompt_path=knowledge_prompt,
            )
        except Exception as exc:  # pragma: no cover - interactive best effort
            err_text = f"[error] {type(exc).__name__}: {exc}"
            print(colorize(err_text, LogColors.RED, enable=color_enabled))
            message_history.append({"role": "assistant", "content": err_text})
            last_rendered_len = print_new_messages(
                message_history, last_rendered_len, color_enabled=color_enabled
            )
            continue

        history_result = result.get("msg_history", []) or []
        full_history = history_result
        message_history = [dict(m) for m in full_history]
        last_rendered_len = print_new_messages(
            message_history, last_rendered_len, color_enabled=color_enabled
        )
        knowledge_info = result.get("knowledge_result")
        if knowledge_info and args.generate_knowledge:
            added = knowledge_info.get("count")
            target_display = str(knowledge_file) if knowledge_file else "N/A"
            print(
                colorize(
                    f"Knowledge updated ({added or 0} entries) → {target_display}",
                    LogColors.YELLOW,
                    enable=color_enabled,
                )
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("\nExiting SEIMEI CLI.")


if __name__ == "__main__":
    main()
