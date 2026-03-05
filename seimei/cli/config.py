from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# -------- Default knobs (easy to tweak without hunting through the code) --------

CLI_VERSION = "0.2.0"
CLI_NAME = "SEIMEI CLI"
ASCII_ART = r"""
  _____  ______ _____ __  __ ______ _____
 / ____|/ ____|_   _|  \/  |  ____|_   _|
| (___ | |____  | | | \  / | |____  | |
 \___ \|  ____| | | | |\/| |  ____| | |
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

DEFAULT_RM_KWARGS: Dict[str, Any] = {
    "url": "https://hm465ys5n3.execute-api.ap-southeast-2.amazonaws.com/prod/v1/rmsearch",
    "agent_routing": False,
    "knowledge_search": True,
}
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

AVAILABLE_MODELS = ("gpt-5-nano", "gpt-5-mini", "gpt-5")

SLASH_COMMANDS = (
    "/chat_history",
    "/status",
    "/new",
    "/settings",
    "/exit",
    "/help",
)

HELP_TEXT = textwrap.dedent(
    """
    Commands:
      /chat_history [N]  Show latest run history from seimei_runs (default: 20 runs)
      /status            Show active model and token usage for this conversation
      /new               Start a new chat (keeps current settings/model)
      /settings [model]  Choose the model (gpt-5-nano, gpt-5-mini, gpt-5)
      /exit              Quit the SEIMEI CLI
      /help              Show this help text

    Tips:
      - Type `/` to open command suggestions.
      - Keep typing (example: `/s`) to filter candidates.
      - Use Up/Down arrows + Enter to pick a command.
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
