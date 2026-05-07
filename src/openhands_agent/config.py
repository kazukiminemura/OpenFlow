from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentConfig:
    base_url: str
    model: str
    num_ctx: int | None
    temperature: float | None
    api_key: str
    workdir: Path
    browser_headless: bool
    max_steps: int
    history_limit: int
    trace: bool
    native_tools: bool
    browser_light_mode: bool
    browser_block_resources: set[str]
    browser_viewport_width: int
    browser_viewport_height: int


def load_config() -> AgentConfig:
    load_dotenv()
    workdir = Path(os.getenv("AGENT_WORKDIR", ".")).expanduser().resolve()
    return AgentConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        num_ctx=_optional_int(os.getenv("OLLAMA_NUM_CTX", "131072")),
        temperature=_optional_float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        workdir=workdir,
        browser_headless=os.getenv("BROWSER_HEADLESS", "false").lower() in {"1", "true", "yes"},
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
        history_limit=int(os.getenv("AGENT_HISTORY_LIMIT", "0")),
        trace=os.getenv("AGENT_TRACE", "true").lower() in {"1", "true", "yes"},
        native_tools=os.getenv("AGENT_NATIVE_TOOLS", "false").lower() in {"1", "true", "yes"},
        browser_light_mode=os.getenv("BROWSER_LIGHT_MODE", "true").lower() in {"1", "true", "yes"},
        browser_block_resources=_split_set(os.getenv("BROWSER_BLOCK_RESOURCES", "image,media,font")),
        browser_viewport_width=int(os.getenv("BROWSER_VIEWPORT_WIDTH", "1024")),
        browser_viewport_height=int(os.getenv("BROWSER_VIEWPORT_HEIGHT", "720")),
    )


def _split_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)
