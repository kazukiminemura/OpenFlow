from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentConfig:
    base_url: str
    model: str
    api_key: str
    workdir: Path
    browser_headless: bool
    max_steps: int
    native_tools: bool


def load_config() -> AgentConfig:
    load_dotenv()
    workdir = Path(os.getenv("AGENT_WORKDIR", ".")).expanduser().resolve()
    return AgentConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        workdir=workdir,
        browser_headless=os.getenv("BROWSER_HEADLESS", "false").lower() in {"1", "true", "yes"},
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "12")),
        native_tools=os.getenv("AGENT_NATIVE_TOOLS", "false").lower() in {"1", "true", "yes"},
    )
