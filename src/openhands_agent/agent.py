from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from openai import APIConnectionError, NotFoundError, OpenAI

from .tools.base import ToolRegistry, ToolResult


KNOWN_SITES = {
    "google": "https://www.google.com/",
    "グーグル": "https://www.google.com/",
    "ぐーぐる": "https://www.google.com/",
    "example": "https://example.com/",
}


SYSTEM_PROMPT = """あなたはローカルで動くAIエージェントです。ブラウザ操作とターミナル操作のツールを使えます。

ルール:
- ユーザーがブラウザ操作やターミナル操作を求めたらツールを使う。
- 小さく確認しやすい手順を優先する。
- ツール結果で確認するまで、操作が成功したと言わない。
- 本当に必要な場合だけ確認質問をする。
- ユーザーが別言語を明示しない限り、必ず自然な日本語で答える。
- 韓国語、中国語、英語へ切り替えない。

ネイティブ tool calling が使えない場合、ツール要求は次のJSONオブジェクトだけで返す:
{"tool": "tool_name", "arguments": {"key": "value"}}

タスク完了時はJSONではなく、通常の日本語で答える。
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
        history_limit: int = 12,
        num_ctx: int | None = None,
        temperature: float | None = None,
        on_trace: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.native_tools = native_tools
        self.history_limit = history_limit
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.on_trace = on_trace
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_input: str) -> AgentResponse:
        self._trace(f"入力を受け取りました: {self._trim_for_trace(user_input)}")
        direct_response = self._run_direct_command(user_input)
        if direct_response is not None:
            self._trace("直接操作として処理しました。")
            return direct_response

        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            self._trace(f"ステップ {step}/{self.max_steps}: モデルに次の行動を問い合わせます。")
            message = self._next_message()
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    function = call["function"]
                    self._trace_tool_call(function["name"], function.get("arguments", "{}"))
                    result = self.tools.run(function["name"], function.get("arguments", "{}"))
                    self._trace_tool_result(result)
                    self._append_tool_result(call["id"], result)
                continue

            content = message.get("content") or ""
            if self._looks_non_japanese(content):
                self._trace("モデル応答が日本語ではなかったため、再試行せず安全な案内に置き換えます。")
                return AgentResponse(
                    text="すみません、モデルの応答が日本語から外れました。ブラウザ操作は `open https://...` または `ブラウザでグーグルを開いて` のように指示してください。",
                    steps=step,
                )
            manual_call = self._parse_manual_tool_call(content)
            if manual_call is not None:
                self._trace_tool_call(manual_call["tool"], manual_call.get("arguments", {}))
                result = self.tools.run(manual_call["tool"], manual_call.get("arguments", {}))
                self._trace_tool_result(result)
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result for {manual_call['tool']}:\n{result.content}",
                    }
                )
                continue

            self._trace("最終応答を返します。")
            return AgentResponse(text=content, steps=step)

        return AgentResponse(
            text=f"最大ステップ数 {self.max_steps} に達しました。途中結果を確認して、必要なら指示を分けてください。",
            steps=self.max_steps,
        )

    def _next_message(self) -> dict[str, Any]:
        try:
            self._trim_history()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
            }
            if self.native_tools:
                kwargs["tools"] = self.tools.schemas()
                kwargs["tool_choice"] = "auto"
            options: dict[str, Any] = {}
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if options:
                kwargs["extra_body"] = {"options": options}

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

    def _trim_history(self) -> None:
        if self.history_limit <= 0:
            return
        system_messages = self.messages[:1]
        recent_messages = self.messages[1:][-self.history_limit :]
        self.messages = system_messages + recent_messages

    def _run_direct_command(self, user_input: str) -> AgentResponse | None:
        text = user_input.strip()
        normalized = text.lower()

        if self._is_capability_question(normalized):
            self._trace("機能説明の直接応答を使います。")
            return AgentResponse(text=self._capability_text(), steps=1)

        if normalized in {"open browser", "browser open", "ブラウザを開いて", "ブラウザ開いて"}:
            self._trace_tool_call("browser", {"action": "goto", "url": "about:blank"})
            result = self.tools.run("browser", {"action": "goto", "url": "about:blank"})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text("ブラウザを開きました。", result), steps=1)

        known_url = self._known_site_url(normalized)
        if known_url and self._is_browser_open_request(normalized):
            self._trace_tool_call("browser", {"action": "goto", "url": known_url})
            result = self.tools.run("browser", {"action": "goto", "url": known_url})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text(f"{known_url} を開きました。", result), steps=1)

        search_query = self._search_query(text)
        if search_query:
            return self._run_google_search(search_query)

        url_match = re.search(r"https?://\S+", text)
        if url_match and self._is_browser_open_request(normalized):
            url = url_match.group(0)
            self._trace_tool_call("browser", {"action": "goto", "url": url})
            result = self.tools.run("browser", {"action": "goto", "url": url})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text(f"{url} を開きました。", result), steps=1)

        for prefix in ("terminal:", "shell:", "run:"):
            if normalized.startswith(prefix):
                command = text[len(prefix) :].strip()
                if not command:
                    return AgentResponse(text="実行するコマンドを指定してください。", steps=1)
                self._trace_tool_call("terminal_run", {"command": command})
                result = self.tools.run("terminal_run", {"command": command})
                self._trace_tool_result(result)
                return AgentResponse(text=result.content, steps=1)

        return None

    def _search_query(self, text: str) -> str | None:
        normalized = text.strip()
        patterns = [
            r"^検索して\s*(?P<query>.+)$",
            r"^(?P<query>.+?)を検索して$",
            r"^(?P<query>.+?)で検索して$",
            r"^(?P<query>.+?)検索して$",
            r"^search\s+(?P<query>.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                query = match.group("query").strip()
                return query or None
        return None

    def _run_google_search(self, query: str) -> AgentResponse:
        self._trace_tool_call("browser", {"action": "goto", "url": "https://www.google.com/"})
        goto_result = self.tools.run("browser", {"action": "goto", "url": "https://www.google.com/"})
        self._trace_tool_result(goto_result)
        if not goto_result.ok:
            return AgentResponse(text=goto_result.content, steps=1)

        self._trace_tool_call("browser", {"action": "type", "selector": "textarea[name='q'], input[name='q']", "text": query})
        type_result = self.tools.run(
            "browser",
            {"action": "type", "selector": "textarea[name='q'], input[name='q']", "text": query},
        )
        self._trace_tool_result(type_result)
        if not type_result.ok:
            return AgentResponse(text=type_result.content, steps=1)

        self._trace_tool_call("browser", {"action": "press", "selector": "textarea[name='q'], input[name='q']", "key": "Enter"})
        press_result = self.tools.run(
            "browser",
            {"action": "press", "selector": "textarea[name='q'], input[name='q']", "key": "Enter"},
        )
        self._trace_tool_result(press_result)
        if not press_result.ok:
            return AgentResponse(text=press_result.content, steps=1)

        title_result = self.tools.run("browser", {"action": "title"})
        self._trace_tool_result(title_result)
        return AgentResponse(
            text=self._direct_tool_text(f"`{query}` をGoogleで検索しました。", title_result),
            steps=1,
        )

    def _known_site_url(self, normalized: str) -> str | None:
        for name, url in KNOWN_SITES.items():
            if name in normalized:
                return url
        return None

    def _is_browser_open_request(self, normalized: str) -> bool:
        return any(word in normalized for word in ["open", "browser", "browse", "ブラウザ", "開"])

    def _looks_non_japanese(self, content: str) -> bool:
        if not content.strip():
            return False
        thai = re.search(r"[\u0e00-\u0e7f]", content) is not None
        hangul = re.search(r"[\uac00-\ud7af]", content) is not None
        japanese = re.search(r"[\u3040-\u30ff\u3400-\u9fff]", content) is not None
        return (thai or hangul) and not japanese

    def _is_capability_question(self, normalized: str) -> bool:
        compact = re.sub(r"\s+", "", normalized)
        return compact in {
            "何ができるのかをおしえて",
            "何ができるのか教えて",
            "何ができる?",
            "何ができる？",
            "なにができる?",
            "なにができる？",
            "help",
            "ヘルプ",
        }

    def _capability_text(self) -> str:
        return (
            "このエージェントでできることは主に3つです。\n\n"
            "- ブラウザ操作: `open browser`、`open https://example.com`、ページタイトル取得、クリック、入力、スクリーンショットなど\n"
            "- ターミナル操作: `terminal: Get-Location` のようにローカルコマンドを実行\n"
            "- 拡張: `src/openhands_agent/tools/` に新しい `Tool` を追加して操作を増やせます\n\n"
            "重い処理を避けたい場合は、直接操作コマンドから使うのが安定します。"
        )

    def _direct_tool_text(self, success_text: str, result: ToolResult) -> str:
        if result.ok:
            return f"{success_text}\n{result.content}"
        return result.content

    def _trace(self, message: str) -> None:
        if self.on_trace is not None:
            self.on_trace(message)

    def _trace_tool_call(self, name: str, arguments: str | dict[str, Any]) -> None:
        if isinstance(arguments, str):
            rendered = arguments
        else:
            rendered = json.dumps(arguments, ensure_ascii=False)
        self._trace(f"ツール呼び出し: {name} {self._trim_for_trace(rendered)}")

    def _trace_tool_result(self, result: ToolResult) -> None:
        status = "成功" if result.ok else "失敗"
        self._trace(f"ツール結果: {status} - {self._trim_for_trace(result.content)}")

    def _trim_for_trace(self, text: str, limit: int = 240) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "...<省略>"

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
