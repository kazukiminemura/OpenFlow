from __future__ import annotations

from typing import Any, Callable

from .tools.base import ToolRegistry, ToolResult


class McpToolContext:
    """Small MCP-style boundary around local tools.

    Local tools are still Python classes, but the agent now talks to them through
    a single context object that carries the call shape, tracing, and policy hook.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        restrict: Callable[[str, str | dict[str, Any]], str | None],
        trace: Callable[[str], None],
    ) -> None:
        self._tools = tools
        self._restrict = restrict
        self._trace = trace

    def run(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        restricted_reason = self._restrict(name, arguments)
        if restricted_reason:
            return ToolResult(
                (
                    "askモードでは、ファイルの削除や書き込みを伴う可能性がある操作を実行しません。\n"
                    f"理由: {restricted_reason}\n"
                    "この操作が必要な場合は `/mode agent` でエージェントモードに切り替えてください。"
                ),
                ok=False,
            )
        self._trace_call(name, arguments)
        result = self._tools.run(name, arguments)
        self._trace_result(result)
        return result

    def _trace_call(self, name: str, arguments: str | dict[str, Any]) -> None:
        self._trace(f"ツール呼び出し: {name} {arguments}")

    def _trace_result(self, result: ToolResult) -> None:
        status = "成功" if result.ok else "失敗"
        self._trace(f"ツール結果: {status} - {result.content}")
