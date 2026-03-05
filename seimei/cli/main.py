from __future__ import annotations

import asyncio
from difflib import get_close_matches
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..logging_utils import LogColors, colorize, supports_color
from ..seimei import seimei as SeimeiOrchestrator
from .config import (
    ASCII_ART,
    AVAILABLE_MODELS,
    CLI_NAME,
    CLI_VERSION,
    DEFAULT_AGENT_LOG_HEAD_LINES,
    DEFAULT_ALLOW_CODE_EXEC,
    DEFAULT_ALLOWED_COMMANDS,
    DEFAULT_RM_KWARGS,
    DEFAULT_SYSTEM_PROMPT,
    HELP_TEXT,
    SLASH_COMMANDS,
    CLIArgs,
    build_agent_config,
    build_llm_kwargs,
    ensure_parent_dir,
    parse_args,
)
from .history import ChatHistoryItem, load_recent_history
from .interactive import TerminalUI

COMMAND_SET = set(SLASH_COMMANDS)


def resolve_version() -> str:
    try:
        from importlib.metadata import version

        return version("SEIMEI")
    except Exception:
        return CLI_VERSION


def render_header(session_id: str, *, model_name: str, color_enabled: bool) -> None:
    banner = colorize(ASCII_ART.rstrip(), LogColors.GREEN, enable=color_enabled)
    print(banner)
    version = resolve_version()
    cli_line = colorize(f"{CLI_NAME} v{version}", LogColors.GRAY, enable=color_enabled)
    init_line = colorize(
        f"Initialized session {session_id} | model={model_name}",
        LogColors.GRAY,
        enable=color_enabled,
    )
    tip_line = colorize(
        "Type `/` to open commands. Use arrow keys + Enter to choose.",
        LogColors.YELLOW,
        enable=color_enabled,
    )
    print(f"\n{cli_line}")
    print(f"\n{init_line}")
    print(f"\n{tip_line}")
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


async def ainput(
    ui: TerminalUI,
    prompt: str,
    commands: Sequence[str] = SLASH_COMMANDS,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ui.read_line(prompt, commands))


def _build_orchestrator(
    *,
    args: CLIArgs,
    llm_kwargs: Dict[str, Any],
    agent_config: Sequence[Dict[str, Any]],
) -> SeimeiOrchestrator:
    return SeimeiOrchestrator(
        agent_config=agent_config,
        llm_config=llm_kwargs,
        rm_config=dict(DEFAULT_RM_KWARGS),
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        allow_code_exec=DEFAULT_ALLOW_CODE_EXEC,
        allowed_commands=list(DEFAULT_ALLOWED_COMMANDS) if DEFAULT_ALLOWED_COMMANDS else None,
        agent_log_head_lines=DEFAULT_AGENT_LOG_HEAD_LINES,
        max_tokens_per_question=args.max_tokens,
    )


def _parse_chat_history_limit(command_text: str) -> Optional[int]:
    parts = command_text.split()
    if len(parts) == 1:
        return 20
    if len(parts) != 2:
        return None
    try:
        value = int(parts[1])
    except ValueError:
        return None
    return value if value > 0 else None


def _print_history_list(entries: Sequence[ChatHistoryItem], *, color_enabled: bool) -> None:
    print(
        colorize(
            f"\nRecent histories ({len(entries)} runs loaded):",
            LogColors.YELLOW,
            enable=color_enabled,
        )
    )
    for idx, entry in enumerate(entries, start=1):
        run_label = colorize(entry.run_name, LogColors.GRAY, enable=color_enabled)
        print(f"{idx:>2}. {entry.preview}  ({run_label})")


def _print_history_item(entry: ChatHistoryItem, *, color_enabled: bool) -> None:
    run_label = colorize(entry.run_name, LogColors.GRAY, enable=color_enabled)
    user_label = colorize("User:", LogColors.CYAN, enable=color_enabled)
    assistant_label = colorize("SEIMEI:", LogColors.BOLD_MAGENTA, enable=color_enabled)
    print(f"\nRun: {run_label}")
    print(f"{user_label} {entry.user_message}\n")
    print(f"{assistant_label} {entry.assistant_message}\n")


def _reset_conversation(system_prompt: Optional[str]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    return history


def _empty_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(total_usage: Dict[str, int], usage: Dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            value = int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        total_usage[key] = total_usage.get(key, 0) + value


def _parse_index_selection(raw: str, total_items: int) -> Optional[int]:
    text = raw.strip()
    if text.startswith("/"):
        text = text[1:]
    if not text.isdigit():
        return None
    selected = int(text)
    if not (1 <= selected <= total_items):
        return None
    return selected - 1


def _resolve_model_choice(raw: str) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("/"):
        text = text[1:]
    lowered = text.lower()
    if lowered.isdigit():
        idx = int(lowered) - 1
        if 0 <= idx < len(AVAILABLE_MODELS):
            return AVAILABLE_MODELS[idx]
        return None
    for model_name in AVAILABLE_MODELS:
        if lowered == model_name.lower():
            return model_name
    return None


async def run_cli(args: CLIArgs) -> None:
    llm_kwargs = build_llm_kwargs(args)
    agent_config = build_agent_config(args)
    orchestrator = _build_orchestrator(args=args, llm_kwargs=dict(llm_kwargs), agent_config=agent_config)

    knowledge_file = Path(args.knowledge_file).expanduser() if args.knowledge_file else None
    knowledge_prompt = Path(args.knowledge_prompt).expanduser() if args.knowledge_prompt else None
    ensure_parent_dir(knowledge_file)

    base_knowledge_config = {
        "generate_knowledge": args.generate_knowledge,
        "save_knowledge_path": str(knowledge_file) if knowledge_file else None,
        "knowledge_prompt_path": str(knowledge_prompt) if knowledge_prompt else None,
        "load_knowledge_path": args.load_knowledge_path,
    }

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    message_history: List[Dict[str, Any]] = _reset_conversation(system_prompt)
    full_history: List[Dict[str, Any]] = list(message_history)
    cumulative_usage = _empty_usage()

    chat_number = 1
    turn = 0
    session_id = f"{uuid.uuid4()}-{uuid.uuid4().hex}"
    color_enabled = supports_color()
    ui = TerminalUI(color_enabled=color_enabled)
    prompt_text = colorize("> ", LogColors.GREEN, enable=color_enabled)

    current_model = str(llm_kwargs.get("model") or AVAILABLE_MODELS[1])

    render_header(session_id, model_name=current_model, color_enabled=color_enabled)
    last_rendered_len = len(message_history)

    while True:
        try:
            user_input = (await ainput(ui, prompt_text)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SEIMEI CLI.")
            return

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split(maxsplit=1)[0]

            if cmd not in COMMAND_SET:
                suggestions = get_close_matches(cmd, SLASH_COMMANDS, n=3, cutoff=0.4)
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                print(
                    colorize(
                        f"[error] Unknown command: {cmd}. Type /help.{hint}",
                        LogColors.RED,
                        enable=color_enabled,
                    )
                )
                continue

            if cmd == "/exit":
                print("Bye, SAYONARA!")
                break

            if cmd == "/help":
                print("\n" + HELP_TEXT + "\n")
                continue

            if cmd == "/status":
                user_turns = sum(1 for msg in message_history if str(msg.get("role") or "").lower() == "user")
                assistant_turns = sum(
                    1 for msg in message_history if str(msg.get("role") or "").lower() == "assistant"
                )
                print(
                    colorize("\nCurrent status", LogColors.YELLOW, enable=color_enabled)
                )
                print(f"- chat: #{chat_number}")
                print(f"- model: {current_model}")
                print(f"- user turns: {user_turns}")
                print(f"- assistant turns: {assistant_turns}")
                print(
                    "- tokens: "
                    f"prompt={cumulative_usage['prompt_tokens']}, "
                    f"completion={cumulative_usage['completion_tokens']}, "
                    f"total={cumulative_usage['total_tokens']}"
                )
                print()
                continue

            if cmd == "/new":
                chat_number += 1
                turn = 0
                message_history = _reset_conversation(system_prompt)
                full_history = list(message_history)
                cumulative_usage = _empty_usage()
                last_rendered_len = len(message_history)
                print(
                    colorize(
                        f"[new] Started chat #{chat_number}. Model is still {current_model}.",
                        LogColors.YELLOW,
                        enable=color_enabled,
                    )
                )
                continue

            if cmd == "/settings":
                parts = user_input.split(maxsplit=1)
                print(colorize("\nModel settings", LogColors.YELLOW, enable=color_enabled))
                for idx, model_name in enumerate(AVAILABLE_MODELS, start=1):
                    marker = " (current)" if model_name == current_model else ""
                    print(f"{idx}. {model_name}{marker}")

                selected_model: Optional[str] = None
                if len(parts) > 1:
                    selected_model = _resolve_model_choice(parts[1])
                else:
                    choice = (
                        await ainput(
                            ui,
                            "Select model number/name (Enter to cancel, type '/' for picker): ",
                            tuple(f"/{name}" for name in AVAILABLE_MODELS),
                        )
                    ).strip()
                    if not choice:
                        print(colorize("[settings] Canceled.", LogColors.GRAY, enable=color_enabled))
                        continue
                    selected_model = _resolve_model_choice(choice)

                if not selected_model:
                    print(colorize("[error] Invalid model selection.", LogColors.RED, enable=color_enabled))
                    continue
                if selected_model == current_model:
                    print(
                        colorize(
                            f"[settings] Model unchanged ({current_model}).",
                            LogColors.GRAY,
                            enable=color_enabled,
                        )
                    )
                    continue

                current_model = selected_model
                llm_kwargs["model"] = selected_model
                orchestrator = _build_orchestrator(
                    args=args,
                    llm_kwargs=dict(llm_kwargs),
                    agent_config=agent_config,
                )
                print(
                    colorize(
                        f"[settings] Model switched to {selected_model}.",
                        LogColors.GREEN,
                        enable=color_enabled,
                    )
                )
                continue

            if cmd == "/chat_history":
                run_limit = _parse_chat_history_limit(user_input)
                if run_limit is None:
                    print(
                        colorize(
                            "[error] Usage: /chat_history [N] where N is a positive integer.",
                            LogColors.RED,
                            enable=color_enabled,
                        )
                    )
                    continue

                entries = load_recent_history(args.log_dir, run_limit=run_limit)
                if not entries:
                    print(
                        colorize(
                            f"[chat_history] No readable runs found under {args.log_dir}",
                            LogColors.YELLOW,
                            enable=color_enabled,
                        )
                    )
                    continue

                _print_history_list(entries, color_enabled=color_enabled)
                if run_limit <= 20:
                    print(colorize("Tip: use /chat_history 50 to see older runs.", LogColors.GRAY, enable=color_enabled))

                selected = (
                    await ainput(
                        ui,
                        "Open which entry? (number, Enter to cancel, type '/' for picker): ",
                        tuple(f"/{idx}" for idx in range(1, len(entries) + 1)),
                    )
                ).strip()
                if not selected:
                    print(colorize("[chat_history] Canceled.", LogColors.GRAY, enable=color_enabled))
                    continue
                selected_idx = _parse_index_selection(selected, len(entries))
                if selected_idx is None:
                    print(colorize("[error] Invalid selection.", LogColors.RED, enable=color_enabled))
                    continue

                chosen = entries[selected_idx]
                _print_history_item(chosen, color_enabled=color_enabled)
                continue

        user_message = {"role": "user", "content": user_input}
        message_history.append(user_message)
        full_history.append(dict(user_message))
        last_rendered_len = print_new_messages(
            message_history,
            last_rendered_len,
            color_enabled=color_enabled,
        )
        turn += 1

        try:
            result = await orchestrator(
                messages=list(message_history),
                run_name=f"cli-{session_id}-chat-{chat_number}-turn-{turn}",
                knowledge_config=base_knowledge_config,
            )
        except Exception as exc:  # pragma: no cover - interactive best effort
            err_text = f"[error] {type(exc).__name__}: {exc}"
            print(colorize(err_text, LogColors.RED, enable=color_enabled))
            message_history.append({"role": "assistant", "content": err_text})
            last_rendered_len = print_new_messages(
                message_history,
                last_rendered_len,
                color_enabled=color_enabled,
            )
            continue

        history_result = result.get("msg_history", []) or []
        full_history = history_result
        message_history = [dict(m) for m in full_history]

        usage = result.get("usage")
        if isinstance(usage, dict):
            _merge_usage(cumulative_usage, usage)

        last_rendered_len = print_new_messages(
            message_history,
            last_rendered_len,
            color_enabled=color_enabled,
        )

        knowledge_info = result.get("knowledge_result")
        if knowledge_info and args.generate_knowledge:
            added = knowledge_info.get("count")
            target_display = str(knowledge_file) if knowledge_file else "N/A"
            print(
                colorize(
                    f"Knowledge updated ({added or 0} entries) -> {target_display}",
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
