from __future__ import annotations

import argparse
import sys

from openai import OpenAI

from .agent import AgentRuntimeError, LocalAgent
from .config import load_config
from .tools.base import ToolRegistry
from .tools.browser import BrowserTool
from .tools.terminal import TerminalTool


def build_agent() -> LocalAgent:
    config = load_config()
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    tools = ToolRegistry()
    tools.register(TerminalTool(config.workdir))
    tools.register(BrowserTool(config.workdir, headless=config.browser_headless))

    return LocalAgent(
        client=client,
        model=config.model,
        tools=tools,
        max_steps=config.max_steps,
        native_tools=config.native_tools,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local OpenHands AI agent.")
    parser.add_argument("prompt", nargs="*", help="Prompt to run once. If omitted, starts an interactive REPL.")
    args = parser.parse_args(argv)

    agent = build_agent()
    try:
        if args.prompt:
            try:
                response = agent.run(" ".join(args.prompt))
                print(response.text)
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
                print(response.text)
            except AgentRuntimeError as exc:
                print(f"Error: {exc}")
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
