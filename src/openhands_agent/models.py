from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AgentResponse:
    text: str
    steps: int
    token_usage: "TokenUsage | None" = None


@dataclass
class MemoryArtifact:
    name: str
    path: Path
    kind: str
    url: str | None = None
    description: str = ""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens or prompt_tokens + completion_tokens
        self.calls += 1

    def has_values(self) -> bool:
        return self.calls > 0 and self.total_tokens > 0

    def render(self) -> str:
        if not self.has_values():
            return "token_usage: unavailable"
        return (
            "token_usage: "
            f"total={self.total_tokens}, "
            f"prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"model_calls={self.calls}"
        )
