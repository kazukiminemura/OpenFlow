from __future__ import annotations

import argparse
import sys

from openai import OpenAI

from .agent import AgentRuntimeError, LocalAgent
from .config import load_config
from .tools.base import ToolRegistry
from .tools.browser import BrowserTool
from .tools.display import DisplayTool
from .tools.terminal import TerminalTool


def build_agent() -> LocalAgent:
    config = load_config()
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    tools = ToolRegistry()
    tools.register(TerminalTool(config.workdir))
    tools.register(DisplayTool())
    tools.register(
        BrowserTool(
            config.workdir,
            headless=config.browser_headless,
            light_mode=config.browser_light_mode,
            block_resources=config.browser_block_resources,
            viewport_width=config.browser_viewport_width,
            viewport_height=config.browser_viewport_height,
        )
    )

    return LocalAgent(
        client=client,
        model=config.model,
        tools=tools,
        max_steps=config.max_steps,
        native_tools=config.native_tools,
        history_limit=config.history_limit,
        num_ctx=config.num_ctx,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        on_trace=_print_trace if config.trace else None,
    )


def _print_trace(message: str) -> None:
    print(f"[trace] {_safe_console_text(message)}", flush=True)


def _safe_console_text(text: str) -> str:
    return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")


def _format_token_usage(response: object) -> str:
    usage = getattr(response, "token_usage", None)
    if usage is None:
        return "[tokens] unavailable"
    return f"[tokens] {usage.render()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local OpenHands AI agent.")
    parser.add_argument("prompt", nargs="*", help="Prompt to run once. If omitted, starts an interactive REPL.")
    args = parser.parse_args(argv)

    agent = build_agent()
    try:
        if args.prompt:
            try:
                response = agent.run(" ".join(args.prompt))
                print(_safe_console_text(response.text))
                print(_safe_console_text(_format_token_usage(response)))
            except AgentRuntimeError as exc:
                print(f"Error: {exc}")
                return 1
            return 0

        print("OpenHands local agent. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if user_input.lower() in {"exit", "quit"}:
                return 0
            if not user_input:
                continue

            try:
                response = agent.run(user_input)
                print(_safe_console_text(response.text))
                print(_safe_console_text(_format_token_usage(response)))
            except AgentRuntimeError as exc:
                print(f"Error: {exc}")
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
