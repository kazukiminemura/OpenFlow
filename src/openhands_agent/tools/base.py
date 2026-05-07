from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    content: str
    ok: bool = True


class Tool(ABC):
    name: str
    description: str
    parameters: JsonDict

    def schema(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, arguments: JsonDict) -> ToolResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[JsonDict]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, raw_arguments: str | JsonDict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", ok=False)

        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as exc:
            return ToolResult(f"Invalid JSON arguments for {name}: {exc}", ok=False)

        try:
            return tool.run(arguments)
        except Exception as exc:  # Keep the agent loop alive and report errors to the model.
            return ToolResult(f"{name} failed: {type(exc).__name__}: {exc}", ok=False)

    def close(self) -> None:
        for tool in self._tools.values():
            tool.close()
