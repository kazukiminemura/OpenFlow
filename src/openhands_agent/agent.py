from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, NotFoundError, OpenAI

from .tools.base import ToolRegistry, ToolResult


SYSTEM_PROMPT = """You are a local AI agent. You can use tools to operate a browser and terminal.

Rules:
- Use tools when the user asks for browser or terminal work.
- Prefer small, inspectable steps.
- Do not claim a command or browser action succeeded until a tool result confirms it.
- Ask for clarification only when required.
- Answer the user in Japanese unless they ask for another language.

If native tool calling is unavailable, respond with exactly one JSON object to request a tool:
{"tool": "tool_name", "arguments": {"key": "value"}}

When the task is complete, answer normally without JSON.
"""


@dataclass
class AgentResponse:
    text: str
    steps: int


class AgentRuntimeError(RuntimeError):
    pass


class LocalAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: ToolRegistry,
        max_steps: int = 12,
        native_tools: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.native_tools = native_tools
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_input: str) -> AgentResponse:
        direct_response = self._run_direct_command(user_input)
        if direct_response is not None:
            return direct_response

        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            message = self._next_message()
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    function = call["function"]
                    result = self.tools.run(function["name"], function.get("arguments", "{}"))
                    self._append_tool_result(call["id"], result)
                continue

            content = message.get("content") or ""
            manual_call = self._parse_manual_tool_call(content)
            if manual_call is not None:
                result = self.tools.run(manual_call["tool"], manual_call.get("arguments", {}))
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result for {manual_call['tool']}:\n{result.content}",
                    }
                )
                continue

            return AgentResponse(text=content, steps=step)

        return AgentResponse(
            text=f"最大ステップ数 {self.max_steps} に達しました。途中結果を確認して、必要なら指示を分けてください。",
            steps=self.max_steps,
        )

    def _next_message(self) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
            }
            if self.native_tools:
                kwargs["tools"] = self.tools.schemas()
                kwargs["tool_choice"] = "auto"

            completion = self.client.chat.completions.create(
                **kwargs,
            )
        except NotFoundError as exc:
            raise AgentRuntimeError(
                f"モデル '{self.model}' が Ollama に見つかりません。"
                " `ollama list` で実際のモデル名を確認し、`.env` の `OLLAMA_MODEL` を合わせてください。"
            ) from exc
        except APIConnectionError as exc:
            raise AgentRuntimeError(
                "Ollama に接続できませんでした。"
                " `ollama serve` が起動しているか、`.env` の `OLLAMA_BASE_URL` を確認してください。"
            ) from exc
        return completion.choices[0].message.model_dump(exclude_none=True)

    def _run_direct_command(self, user_input: str) -> AgentResponse | None:
        text = user_input.strip()
        normalized = text.lower()

        if normalized in {"open browser", "browser open", "ブラウザを開いて", "ブラウザ開いて"}:
            result = self.tools.run("browser", {"action": "goto", "url": "about:blank"})
            return AgentResponse(text=self._direct_tool_text("ブラウザを開きました。", result), steps=1)

        url_match = re.search(r"https?://\S+", text)
        if url_match and any(word in normalized for word in ["open", "browser", "browse", "開"]):
            url = url_match.group(0)
            result = self.tools.run("browser", {"action": "goto", "url": url})
            return AgentResponse(text=self._direct_tool_text(f"{url} を開きました。", result), steps=1)

        for prefix in ("terminal:", "shell:", "run:"):
            if normalized.startswith(prefix):
                command = text[len(prefix) :].strip()
                if not command:
                    return AgentResponse(text="実行するコマンドを指定してください。", steps=1)
                result = self.tools.run("terminal_run", {"command": command})
                return AgentResponse(text=result.content, steps=1)

        return None

    def _direct_tool_text(self, success_text: str, result: ToolResult) -> str:
        if result.ok:
            return f"{success_text}\n{result.content}"
        return result.content

    def _append_tool_result(self, tool_call_id: str, result: ToolResult) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result.content,
            }
        )

    def _parse_manual_tool_call(self, content: str) -> dict[str, Any] | None:
        content = self._strip_json_fence(content.strip())
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
            arguments = parsed.get("arguments", {})
            if isinstance(arguments, dict):
                return {"tool": parsed["tool"], "arguments": arguments}
        return None

    def _strip_json_fence(self, content: str) -> str:
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return content

    def close(self) -> None:
        self.tools.close()
