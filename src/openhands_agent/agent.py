from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError, NotFoundError, OpenAI

from .command_parsers import ArithmeticEvaluator, CodeGenerationDetector, SandboxCommandParser
from .mcp_context import McpToolContext
from .models import AgentResponse, MemoryArtifact, TokenUsage
from .tools.base import ToolRegistry, ToolResult
from .tools.display import parse_resolution


KNOWN_SITES = {
    "google": "https://www.google.com/",
    "グーグル": "https://www.google.com/",
    "ぐーぐる": "https://www.google.com/",
    "example": "https://example.com/",
}


SYSTEM_PROMPT = """あなたはローカルで動くAIエージェントです。ブラウザ、ターミナル、ディスプレイ、サンドボックス操作のツールを使えます。

ルール:
- ユーザーがブラウザ、ターミナル、ディスプレイ、サンドボックス操作を求めたらツールを使う。
- 小さく確認しやすい手順を優先する。
- ツール結果で確認するまで、操作が成功したと言わない。
- 本当に必要な場合だけ確認質問をする。
- ユーザーが別言語を明示しない限り、必ず自然な日本語で答える。
- 韓国語、中国語、英語へ切り替えない。

ネイティブ tool calling が使えない場合、ツール要求は次のJSONオブジェクトだけで返す:
{"tool": "tool_name", "arguments": {"key": "value"}}

タスク完了時はJSONではなく、通常の日本語で答える。
"""


CHAT_SYSTEM_PROMPT = """あなたはローカルで動く対話用AIです。

ルール:
- ツール操作は行わず、ユーザー入力を通常のLLMチャットとして処理する。
- ユーザーが別言語を明示しない限り、自然な日本語で答える。
"""


ASK_SYSTEM_PROMPT = """あなたはローカルで動く読み取り中心のAIエージェントです。ブラウザ、ターミナル、ディスプレイ、サンドボックス操作のツールを使えます。

ルール:
- ファイルやディレクトリの削除、作成、上書き、追記、移動、リネーム、権限変更を行わない。
- 読み取り、検索、一覧、状態確認を優先する。
- 操作が書き込みや削除を伴う場合は、エージェントモードへの切り替えが必要だと説明する。
- ユーザーが別言語を明示しない限り、必ず自然な日本語で答える。
- 韓国語、中国語、英語へ切り替えない。

ネイティブ tool calling が使えない場合、ツール要求は次のJSONオブジェクトだけで返す:
{"tool": "tool_name", "arguments": {"key": "value"}}

タスク完了時はJSONではなく、通常の日本語で答える。
"""


CODE_GENERATION_SYSTEM_PROMPT = """あなたはローカルで動くコード生成アシスタントです。

ルール:
- ユーザーの依頼に合うコードを生成する。
- ユーザーが別言語を明示しない限り、説明は自然な日本語にする。
- 生成するコードは実行可能で、必要な前提や使い方を短く添える。
- ファイル作成や上書きは行わず、コード本文を応答として返す。
- 不明点があっても、合理的な仮定で最小の実用例を出す。
"""


AGENT_MODE = "agent"
ASK_MODE = "ask"
CHAT_MODE = "chat"


SEARCH_SUMMARY_LINK_LIMIT = 10
SEARCH_SUMMARY_TARGET_CHARS = 150
SEARCH_SUMMARY_MAX_CHARS = 180
LLM_SUMMARY_SOURCE_CHARS = 24000
RESEARCH_MAX_CYCLES = 3
RESEARCH_MIN_USEFUL_PAGES = 3


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
        max_tokens: int | None = None,
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
        self.max_tokens = max_tokens
        self.on_trace = on_trace
        self.token_usage = TokenUsage()
        self.mode = AGENT_MODE
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        self.memory_artifacts: dict[str, MemoryArtifact] = {}
        self.last_artifact_name: str | None = None
        self.arithmetic = ArithmeticEvaluator()
        self.code_generation_detector = CodeGenerationDetector()
        self.sandbox_command_parser = SandboxCommandParser()
        self.mcp_tools = McpToolContext(
            tools=tools,
            restrict=self._mcp_restriction_reason,
            trace=lambda _message: None,
        )

    def set_mode(self, mode: str) -> None:
        if mode not in {AGENT_MODE, ASK_MODE, CHAT_MODE}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.messages = [{"role": "system", "content": self._system_prompt()}]

    def _system_prompt(self) -> str:
        mode = getattr(self, "mode", AGENT_MODE)
        if mode == CHAT_MODE:
            return CHAT_SYSTEM_PROMPT
        if mode == ASK_MODE:
            return ASK_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _mode_label(self) -> str:
        if self.mode == CHAT_MODE:
            return "chatモード"
        if self.mode == ASK_MODE:
            return "askモード"
        return "エージェントモード"

    def _mode_command(self, user_input: str) -> AgentResponse | None:
        text = user_input.strip()
        normalized = text.lower()
        compact = re.sub(r"\s+", "", normalized)

        chat_commands = {
            "/mode chat",
            "/mode conversation",
            "/chat",
            "chat mode",
            "conversation mode",
        }
        agent_commands = {
            "/mode agent",
            "/agent",
            "agent mode",
            "tool mode",
        }
        ask_commands = {
            "/mode ask",
            "/ask",
            "ask mode",
            "restricted mode",
            "safe agent mode",
        }
        status_commands = {
            "/mode",
            "mode",
            "モード",
            "今のモード",
            "現在のモード",
            "モード確認",
        }

        if normalized in chat_commands or compact in {
            "対話モード",
            "対話モードにして",
            "会話モード",
            "会話モードにして",
            "チャットモード",
            "チャットモードにして",
        }:
            self.set_mode(CHAT_MODE)
            self._trace("対話モードに切り替えました。")
            return AgentResponse(
                text="chatモードに切り替えました。ツール操作や定型処理は行わず、入力をLLMへそのまま渡します。",
                steps=1,
            )

        if normalized in ask_commands or compact in {
            "askモード",
            "askモードにして",
            "アスクモード",
            "アスクモードにして",
            "制限付きエージェントモード",
            "制限付きエージェントモードにして",
            "安全エージェントモード",
            "安全エージェントモードにして",
        }:
            self.set_mode(ASK_MODE)
            self._trace("askモードに切り替えました。")
            return AgentResponse(
                text=(
                    "askモードに切り替えました。検索、読み取り、状態確認はできますが、"
                    "ファイルの削除や書き込みを伴う操作は制限します。"
                ),
                steps=1,
            )

        if normalized in agent_commands or compact in {
            "エージェントモード",
            "エージェントモードにして",
            "ツールモード",
            "ツールモードにして",
            "操作モード",
            "操作モードにして",
        }:
            self.set_mode(AGENT_MODE)
            self._trace("エージェントモードに切り替えました。")
            return AgentResponse(
                text=(
                    "エージェントモードに切り替えました。"
                    "ブラウザ、ターミナル、ディスプレイ、サンドボックス操作を制限なしで直接実行できます。"
                ),
                steps=1,
            )

        if normalized in status_commands or compact in status_commands:
            return AgentResponse(text=f"現在は{self._mode_label()}です。", steps=1)

        return None

    def run(self, user_input: str) -> AgentResponse:
        self.token_usage = TokenUsage()
        self._trace(f"入力を受け取りました: {self._trim_for_trace(user_input)}")
        mode_response = self._mode_command(user_input)
        if mode_response is not None:
            mode_response.token_usage = self.token_usage
            return mode_response

        if self.mode in {AGENT_MODE, ASK_MODE}:
            direct_response = self._run_direct_command(user_input)
            if direct_response is not None:
                self._trace("直接操作として処理しました。")
                direct_response.token_usage = self.token_usage
                return direct_response
        else:
            self._trace("chatモードのため、入力をLLMへ直接送信します。")
            return self._run_plain_chat(user_input)

        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            self._trace(f"ステップ {step}/{self.max_steps}: モデルに次の行動を問い合わせます。")
            message = self._next_message()
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if self.mode == CHAT_MODE:
                    self._trace("chatモードのためツール呼び出しを実行せず、通常応答として扱います。")
                    return AgentResponse(
                        text=message.get("content") or "chatモードではツール操作を実行しません。",
                        steps=step,
                        token_usage=self.token_usage,
                    )
                for call in tool_calls:
                    function = call["function"]
                    self._trace_tool_call(function["name"], function.get("arguments", "{}"))
                    result = self._run_tool(function["name"], function.get("arguments", "{}"))
                    self._trace_tool_result(result)
                    self._append_tool_result(call["id"], result)
                continue

            content = message.get("content") or ""
            manual_call = self._parse_manual_tool_call(content) if self.mode in {AGENT_MODE, ASK_MODE} else None
            if manual_call is not None:
                self._trace_tool_call(manual_call["tool"], manual_call.get("arguments", {}))
                result = self._run_tool(manual_call["tool"], manual_call.get("arguments", {}))
                self._trace_tool_result(result)
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result for {manual_call['tool']}:\n{result.content}",
                    }
                )
                continue

            if self.mode != CHAT_MODE and self._looks_non_japanese(content):
                self._trace("モデル応答が日本語ではなかったため、安全な案内に置き換えます。")
                return AgentResponse(
                    text=self._non_japanese_fallback_text(user_input),
                    steps=step,
                    token_usage=self.token_usage,
                )

            self._trace("最終応答を返します。")
            return AgentResponse(text=content, steps=step, token_usage=self.token_usage)

        return AgentResponse(
            text=f"最大ステップ数 {self.max_steps} に達しました。途中結果を確認して、必要なら指示を分けてください。",
            steps=self.max_steps,
            token_usage=self.token_usage,
        )

    def _next_message(self) -> dict[str, Any]:
        try:
            self._trim_history()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.mode in {AGENT_MODE, ASK_MODE} and self.native_tools:
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
            self.token_usage.add(completion.usage)
            if self.token_usage.has_values():
                self._trace(self.token_usage.render())
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

    def _run_plain_chat(self, user_input: str) -> AgentResponse:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": user_input}],
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            options: dict[str, Any] = {}
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if options:
                kwargs["extra_body"] = {"options": options}

            completion = self.client.chat.completions.create(**kwargs)
            self.token_usage.add(completion.usage)
            if self.token_usage.has_values():
                self._trace(self.token_usage.render())
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

        content = completion.choices[0].message.content or ""
        self._trace("chatモードの応答を返します。")
        return AgentResponse(text=content, steps=1, token_usage=self.token_usage)

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

        memory_response = self._memory_continuation_response(text, normalized)
        if memory_response is not None:
            self._trace("メモリーから続きの直接応答を使います。")
            return memory_response

        arithmetic_text = self._arithmetic_text(text)
        if arithmetic_text is not None:
            self._trace("四則演算の直接応答を使います。")
            return AgentResponse(text=arithmetic_text, steps=1)

        code_generation_request = self._code_generation_request(text)
        if code_generation_request is not None:
            self._trace("コード生成の直接応答を使います。")
            return self._run_code_generation(code_generation_request)

        if normalized in {"open browser", "browser open", "ブラウザを開いて", "ブラウザ開いて"}:
            self._trace_tool_call("browser", {"action": "goto", "url": "about:blank"})
            result = self._run_tool("browser", {"action": "goto", "url": "about:blank"})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text("ブラウザを開きました。", result), steps=1)

        known_url = self._known_site_url(normalized)
        if known_url and self._is_browser_open_request(normalized):
            self._trace_tool_call("browser", {"action": "goto", "url": known_url})
            result = self._run_tool("browser", {"action": "goto", "url": known_url})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text(f"{known_url} を開きました。", result), steps=1)

        research_query = self._research_query(text)
        if research_query:
            return self._run_search_and_summarize(research_query)

        search_query = self._search_query(text)
        if search_query:
            return self._run_google_search(search_query)

        url_match = re.search(r"https?://\S+", text)
        if url_match and self._is_browser_open_request(normalized):
            url = url_match.group(0)
            self._trace_tool_call("browser", {"action": "goto", "url": url})
            result = self._run_tool("browser", {"action": "goto", "url": url})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text(f"{url} を開きました。", result), steps=1)

        display_args = self._display_command(text, normalized)
        if display_args is not None:
            self._trace_tool_call("display", display_args)
            result = self._run_tool("display", display_args)
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text("ディスプレイ設定を操作しました。", result), steps=1)

        sandbox_args = self._sandbox_command(text, normalized)
        if sandbox_args is not None:
            self._trace_tool_call("sandbox", sandbox_args)
            result = self._run_tool("sandbox", sandbox_args)
            self._trace_tool_result(result)
            if result.ok and sandbox_args.get("action") == "delete":
                self._forget_deleted_artifact(str(sandbox_args.get("path", "")))
            return AgentResponse(text=self._direct_tool_text("サンドボックスを操作しました。", result), steps=1)

        for prefix in ("terminal:", "shell:", "run:"):
            if normalized.startswith(prefix):
                command = text[len(prefix) :].strip()
                if not command:
                    return AgentResponse(text="実行するコマンドを指定してください。", steps=1)
                self._trace_tool_call("terminal_run", {"command": command})
                result = self._run_tool("terminal_run", {"command": command})
                self._trace_tool_result(result)
                return AgentResponse(text=result.content, steps=1)

        return None

    def _arithmetic_text(self, text: str) -> str | None:
        return self.arithmetic.try_format_result(text)

    def _code_generation_request(self, text: str) -> str | None:
        return self.code_generation_detector.request_from(text)

    def _run_code_generation(self, request: str) -> AgentResponse:
        if self._is_tetris_request(request):
            return self._generate_tetris_in_sandbox()

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": request},
                ],
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            options: dict[str, Any] = {}
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if options:
                kwargs["extra_body"] = {"options": options}

            completion = self.client.chat.completions.create(**kwargs)
            self.token_usage.add(completion.usage)
            if self.token_usage.has_values():
                self._trace(self.token_usage.render())
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

        content = (completion.choices[0].message.content or "").strip()
        if not content:
            content = "コードを生成できませんでした。条件を少し具体的にしてもう一度依頼してください。"
        return self._write_generated_code_to_sandbox(request, content)

    def _is_tetris_request(self, request: str) -> bool:
        normalized = unicodedata.normalize("NFKC", request).lower()
        compact = re.sub(r"\s+", "", normalized)
        return "tetris" in normalized or "テトリス" in compact

    def _memory_continuation_response(self, text: str, normalized: str) -> AgentResponse | None:
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", normalized))
        if not any(word in compact for word in ["実行", "開", "表示", "続き", "再開", "もう一度"]):
            return None

        artifact = self._artifact_for_request(text, compact)
        if artifact is None:
            return None

        process = [
            f"依頼内容から前回の成果物 `{artifact.name}` の続きと判定しました。",
            f"メモリー上のパスを確認しました: {artifact.path}",
        ]
        if not artifact.path.exists():
            process.append("メモリー上のファイルが見つかりませんでした。")
            return AgentResponse(
                text="\n".join(
                    [
                        "前回の成果物を実行できませんでした。",
                        "",
                        "途中過程:",
                        *[f"- {step}" for step in process],
                        "",
                        f"確認済み:\n- ファイル存在: NG - {artifact.path}",
                    ]
                ),
                steps=1,
                token_usage=self.token_usage,
            )

        result = self._run_artifact(artifact)
        process.append(f"{artifact.kind} の実行処理を行いました: {self._first_result_line(result.content)}")
        lines = [
            "対応しました。前回の成果物をメモリーから呼び出して実行しました。",
            "",
            "変更点:",
            "- 直近または名前一致した生成物をメモリーから取得",
            "- 保存済みファイルの存在を確認",
            "- 種類に応じてブラウザ表示またはコマンド実行",
            "",
            "途中過程:",
            *[f"- {step}" for step in process],
            "",
            "確認済み:",
            f"- ファイル存在: OK - {artifact.path}",
            f"- 実行: {'OK' if result.ok else 'NG'} - {self._first_result_line(result.content)}",
            "",
            f"対象ファイル: {artifact.path}",
        ]
        if artifact.url:
            lines.append(f"URL: {artifact.url}")
        return AgentResponse(text="\n".join(lines), steps=1, token_usage=self.token_usage)

    def _artifact_for_request(self, text: str, compact: str) -> MemoryArtifact | None:
        normalized = unicodedata.normalize("NFKC", text).lower()
        if "tetris" in normalized or "テトリス" in compact:
            artifact = self.memory_artifacts.get("tetris")
            if artifact is not None:
                return artifact
            root = self._sandbox_root()
            if root is None:
                return None
            path = root / "generated/tetris/index.html"
            if path.exists():
                return self._remember_artifact(
                    "tetris",
                    path,
                    "html",
                    path.resolve().as_uri(),
                    "ブラウザで動くテトリスゲーム",
                )
            return None

        if self.last_artifact_name:
            return self.memory_artifacts.get(self.last_artifact_name)
        return None

    def _remember_artifact(self, name: str, path: Path, kind: str, url: str | None, description: str) -> MemoryArtifact:
        artifact = MemoryArtifact(
            name=name,
            path=path.resolve(),
            kind=kind,
            url=url,
            description=description,
        )
        self.memory_artifacts[name] = artifact
        self.last_artifact_name = name
        self._trace(f"メモリーに成果物を記録しました: {name} -> {artifact.path}")
        return artifact

    def _run_artifact(self, artifact: MemoryArtifact) -> ToolResult:
        if artifact.kind == "html":
            url = artifact.url or artifact.path.resolve().as_uri()
            return self._run_tool("browser", {"action": "goto", "url": url, "timeout_ms": 10000})
        if artifact.kind == "python":
            return self._run_tool("sandbox", {"action": "run", "command": f"python {artifact.path}", "timeout_seconds": 30})
        if artifact.kind == "javascript":
            return self._run_tool("sandbox", {"action": "run", "command": f"node {artifact.path}", "timeout_seconds": 30})
        return ToolResult(f"この種類の成果物は自動実行できません: {artifact.kind}", ok=False)

    def _artifact_kind_from_filename(self, filename: str, language: str | None) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".py" or language in {"python", "py"}:
            return "python"
        if suffix == ".js" or language in {"javascript", "js", "node"}:
            return "javascript"
        if suffix in {".html", ".htm"} or language in {"html", "htm"}:
            return "html"
        return "text"

    def _generate_tetris_in_sandbox(self) -> AgentResponse:
        path = "generated/tetris/index.html"
        process = [
            "依頼内容からテトリス生成と判定しました。",
            "ブラウザでそのまま動く単一HTMLとして生成します。",
            f"サンドボックス内の保存先を {path} に決めました。",
        ]
        self._trace(process[0])
        self._trace(process[1])
        content = self._tetris_html()
        self._trace(f"HTML/CSS/JavaScriptを生成しました ({len(content)} chars)。")
        write_result = self._run_tool("sandbox", {"action": "write", "path": path, "content": content})
        process.append(f"sandbox write を実行しました: {self._first_result_line(write_result.content)}")
        run_result = self._run_tool(
            "sandbox",
            {
                "action": "run",
                "command": "python -c \"from pathlib import Path; assert Path('generated/tetris/index.html').is_file(); print('tetris html ready')\"",
                "timeout_seconds": 10,
            },
        )
        process.append(f"sandbox run で生成ファイルの存在を確認しました: {self._first_result_line(run_result.content)}")

        root = self._sandbox_root()
        file_path = root / path if root else Path(".agent_sandbox") / path
        file_url = file_path.resolve().as_uri()
        self._remember_artifact("tetris", file_path, "html", file_url, "ブラウザで動くテトリスゲーム")
        browser_result = self._run_tool("browser", {"action": "goto", "url": file_url, "timeout_ms": 10000})
        process.append(f"browser goto で生成HTMLを開きました: {self._first_result_line(browser_result.content)}")

        lines = [
            "対応しました。テトリスゲームをサンドボックスへ実ファイルとして生成し、実行確認しました。",
            "",
            "変更点:",
            "- コード生成ルートでテトリス依頼を専用処理として判定",
            "- ブラウザでそのまま動く単一HTMLを生成",
            f"- `.agent_sandbox/{path}` に保存",
            "- 生成後にサンドボックス内でファイル存在確認を実行",
            "- HTMLをブラウザで開くところまで実行",
            "",
            "途中過程:",
            *[f"- {step}" for step in process],
            "",
            "確認済み:",
            f"- 生成: {'OK' if write_result.ok else 'NG'} - {self._first_result_line(write_result.content)}",
            f"- 実行確認: {'OK' if run_result.ok else 'NG'} - {self._first_result_line(run_result.content)}",
            f"- ブラウザ表示: {'OK' if browser_result.ok else 'NG'} - {self._first_result_line(browser_result.content)}",
            "",
            f"生成ファイル: {file_path}",
            f"URL: {file_url}",
        ]
        return AgentResponse(text="\n".join(lines), steps=1, token_usage=self.token_usage)

    def _write_generated_code_to_sandbox(self, request: str, content: str) -> AgentResponse:
        process = [
            "コード生成用プロンプトでモデルに生成を依頼しました。",
            "モデル応答からコードブロックを抽出します。",
        ]
        code, language = self._extract_generated_code(content)
        process.append(f"言語を {language or '未指定'} と判定しました。")
        filename = self._generated_code_filename(request, language)
        process.append(f"保存先を {filename} に決めました。")
        write_result = self._run_tool("sandbox", {"action": "write", "path": filename, "content": code})
        process.append(f"sandbox write を実行しました: {self._first_result_line(write_result.content)}")
        run_result = self._run_generated_code(filename, language)
        if run_result is not None:
            process.append(f"自動実行を行いました: {self._first_result_line(run_result.content)}")
        else:
            process.append("自動実行対象外の形式だったため、実行確認はスキップしました。")

        root = self._sandbox_root()
        file_path = root / filename if root else Path(".agent_sandbox") / filename
        artifact_name = "generated_code"
        self._remember_artifact(artifact_name, file_path, self._artifact_kind_from_filename(filename, language), None, "直近に生成したコード")
        lines = [
            "対応しました。コードをサンドボックスへ実ファイルとして生成しました。",
            "",
            "変更点:",
            "- コード生成依頼を専用プロンプトで処理",
            "- モデル応答からコード本文を抽出",
            f"- `.agent_sandbox/{filename}` に保存",
            "- 対応している形式は自動実行確認を実行",
            "",
            "途中過程:",
            *[f"- {step}" for step in process],
            "",
            "確認済み:",
            f"- 生成: {'OK' if write_result.ok else 'NG'} - {self._first_result_line(write_result.content)}",
        ]
        if run_result is not None:
            lines.append(f"- 実行確認: {'OK' if run_result.ok else 'NG'} - {self._first_result_line(run_result.content)}")
            if run_result.content:
                lines.extend(["", run_result.content])
        else:
            lines.append("- 実行確認: この種類のコードは自動実行をスキップしました。")
        lines.extend(["", f"生成ファイル: {file_path}"])
        return AgentResponse(text="\n".join(lines), steps=1, token_usage=self.token_usage)

    def _extract_generated_code(self, content: str) -> tuple[str, str | None]:
        match = re.search(r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n(?P<code>.*?)```", content, flags=re.DOTALL)
        if match:
            language = match.group("language").strip().lower() or None
            return match.group("code").strip() + "\n", language
        return content.strip() + "\n", None

    def _generated_code_filename(self, request: str, language: str | None) -> str:
        normalized = unicodedata.normalize("NFKC", request).lower()
        if language in {"python", "py"} or "python" in normalized:
            return "generated/codegen/main.py"
        if language in {"javascript", "js", "node"} or "javascript" in normalized:
            return "generated/codegen/main.js"
        if language in {"html", "htm"} or "html" in normalized:
            return "generated/codegen/index.html"
        return "generated/codegen/output.txt"

    def _run_generated_code(self, filename: str, language: str | None) -> ToolResult | None:
        suffix = Path(filename).suffix.lower()
        if suffix == ".py" or language in {"python", "py"}:
            return self._run_tool("sandbox", {"action": "run", "command": f"python {filename}", "timeout_seconds": 30})
        if suffix == ".js" or language in {"javascript", "js", "node"}:
            return self._run_tool("sandbox", {"action": "run", "command": f"node {filename}", "timeout_seconds": 30})
        if suffix in {".html", ".htm"}:
            root = self._sandbox_root()
            if root is None:
                return None
            return self._run_tool("browser", {"action": "goto", "url": (root / filename).resolve().as_uri(), "timeout_ms": 10000})
        return None

    def _sandbox_root(self) -> Path | None:
        result = self._run_tool("sandbox", {"action": "info"})
        if not result.ok:
            return None
        try:
            data = json.loads(result.content)
        except json.JSONDecodeError:
            return None
        root = data.get("root")
        return Path(str(root)).resolve() if root else None

    def _first_result_line(self, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return "<empty>"

    def _tetris_html(self) -> str:
        return """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sandbox Tetris</title>
  <style>
    :root { color-scheme: dark; font-family: Arial, sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #101820; color: #f6f7fb; }
    main { display: flex; gap: 24px; align-items: flex-start; padding: 24px; }
    canvas { background: #0a0f14; border: 2px solid #34495e; box-shadow: 0 16px 40px #0008; }
    aside { min-width: 180px; }
    h1 { margin: 0 0 12px; font-size: 28px; }
    .score { font-size: 22px; margin: 16px 0; }
    .hint { color: #b8c3cf; line-height: 1.7; }
    button { border: 0; padding: 10px 14px; border-radius: 6px; background: #14b8a6; color: #06201d; font-weight: 700; cursor: pointer; }
  </style>
</head>
<body>
  <main>
    <canvas id="board" width="240" height="480" aria-label="Tetris board"></canvas>
    <aside>
      <h1>Tetris</h1>
      <div class="score">Score: <span id="score">0</span></div>
      <button id="restart">Restart</button>
      <p class="hint">← →: move<br>↑: rotate<br>↓: drop faster<br>Space: hard drop</p>
    </aside>
  </main>
  <script>
    const canvas = document.getElementById("board");
    const context = canvas.getContext("2d");
    const scoreNode = document.getElementById("score");
    const scale = 24;
    context.scale(scale, scale);

    const colors = [null, "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#6366f1", "#d946ef"];
    const pieces = {
      T: [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
      O: [[2, 2], [2, 2]],
      L: [[0, 0, 3], [3, 3, 3], [0, 0, 0]],
      J: [[4, 0, 0], [4, 4, 4], [0, 0, 0]],
      I: [[0, 0, 0, 0], [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]],
      S: [[0, 6, 6], [6, 6, 0], [0, 0, 0]],
      Z: [[7, 7, 0], [0, 7, 7], [0, 0, 0]]
    };

    const arena = createMatrix(10, 20);
    const player = { pos: { x: 0, y: 0 }, matrix: null, score: 0 };
    let dropCounter = 0;
    let dropInterval = 700;
    let lastTime = 0;

    function createMatrix(width, height) {
      return Array.from({ length: height }, () => Array(width).fill(0));
    }

    function createPiece() {
      const keys = Object.keys(pieces);
      return pieces[keys[Math.floor(Math.random() * keys.length)]].map(row => row.slice());
    }

    function collide(arena, player) {
      const matrix = player.matrix;
      const offset = player.pos;
      for (let y = 0; y < matrix.length; y++) {
        for (let x = 0; x < matrix[y].length; x++) {
          if (matrix[y][x] && (arena[y + offset.y] && arena[y + offset.y][x + offset.x]) !== 0) return true;
        }
      }
      return false;
    }

    function merge(arena, player) {
      player.matrix.forEach((row, y) => row.forEach((value, x) => {
        if (value) arena[y + player.pos.y][x + player.pos.x] = value;
      }));
    }

    function rotate(matrix) {
      for (let y = 0; y < matrix.length; y++) {
        for (let x = 0; x < y; x++) [matrix[x][y], matrix[y][x]] = [matrix[y][x], matrix[x][y]];
      }
      matrix.forEach(row => row.reverse());
    }

    function sweep() {
      let rows = 1;
      outer: for (let y = arena.length - 1; y >= 0; y--) {
        for (let x = 0; x < arena[y].length; x++) if (!arena[y][x]) continue outer;
        arena.splice(y, 1);
        arena.unshift(Array(10).fill(0));
        player.score += rows * 10;
        rows *= 2;
        y++;
      }
      scoreNode.textContent = player.score;
    }

    function resetPlayer() {
      player.matrix = createPiece();
      player.pos.y = 0;
      player.pos.x = Math.floor((arena[0].length - player.matrix[0].length) / 2);
      if (collide(arena, player)) {
        arena.forEach(row => row.fill(0));
        player.score = 0;
        scoreNode.textContent = "0";
      }
    }

    function drop() {
      player.pos.y++;
      if (collide(arena, player)) {
        player.pos.y--;
        merge(arena, player);
        sweep();
        resetPlayer();
      }
      dropCounter = 0;
    }

    function move(direction) {
      player.pos.x += direction;
      if (collide(arena, player)) player.pos.x -= direction;
    }

    function hardDrop() {
      do { player.pos.y++; } while (!collide(arena, player));
      player.pos.y--;
      merge(arena, player);
      sweep();
      resetPlayer();
      dropCounter = 0;
    }

    function drawMatrix(matrix, offset) {
      matrix.forEach((row, y) => row.forEach((value, x) => {
        if (!value) return;
        context.fillStyle = colors[value];
        context.fillRect(x + offset.x, y + offset.y, 1, 1);
        context.strokeStyle = "#0a0f14";
        context.lineWidth = 0.05;
        context.strokeRect(x + offset.x, y + offset.y, 1, 1);
      }));
    }

    function draw() {
      context.fillStyle = "#0a0f14";
      context.fillRect(0, 0, canvas.width / scale, canvas.height / scale);
      drawMatrix(arena, { x: 0, y: 0 });
      drawMatrix(player.matrix, player.pos);
    }

    function update(time = 0) {
      const deltaTime = time - lastTime;
      lastTime = time;
      dropCounter += deltaTime;
      if (dropCounter > dropInterval) drop();
      draw();
      requestAnimationFrame(update);
    }

    document.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
      if (event.key === "ArrowDown") drop();
      if (event.key === "ArrowUp") { rotate(player.matrix); if (collide(arena, player)) rotate(player.matrix), rotate(player.matrix), rotate(player.matrix); }
      if (event.code === "Space") hardDrop();
    });
    document.getElementById("restart").addEventListener("click", () => { arena.forEach(row => row.fill(0)); player.score = 0; scoreNode.textContent = "0"; resetPlayer(); });
    resetPlayer();
    update();
  </script>
</body>
</html>
"""

    def _sandbox_command(self, text: str, normalized: str) -> dict[str, Any] | None:
        return self.sandbox_command_parser.parse(text, normalized)

    def _forget_deleted_artifact(self, deleted_path: str) -> None:
        normalized = deleted_path.replace("\\", "/").strip("/")
        forgotten: list[str] = []
        for name, artifact in list(self.memory_artifacts.items()):
            artifact_path = artifact.path.as_posix()
            if normalized and normalized in artifact_path:
                forgotten.append(name)
                del self.memory_artifacts[name]
        if self.last_artifact_name in forgotten:
            self.last_artifact_name = next(reversed(self.memory_artifacts), None)
        if forgotten:
            self._trace(f"削除済み成果物をメモリーから外しました: {', '.join(forgotten)}")

    def _display_command(self, text: str, normalized: str) -> dict[str, Any] | None:
        compact = re.sub(r"\s+", "", text.lower())
        display_words = ("display", "monitor", "screen", "ディスプレイ", "モニター", "画面")
        projection_words = ("拡張", "複製", "pc画面のみ", "内蔵のみ", "外部のみ", "セカンドスクリーンのみ")
        brightness_words = ("明るさ", "暗く", "明るく", "brightness")
        if (
            not any(word in normalized for word in display_words)
            and not any(word in compact for word in ["解像度"])
            and not any(word in compact for word in brightness_words)
            and not any(word in compact for word in projection_words)
        ):
            return None

        if any(word in compact for word in ["一覧", "確認", "表示して", "状態", "list"]):
            return {"action": "list"}

        if any(word in compact for word in ["拡張", "extend", "マルチディスプレイ拡張", "マルチプレイディスプレイ拡張"]):
            return {"action": "mode", "mode": "extend"}
        if any(word in compact for word in ["複製", "duplicate", "clone", "ミラー"]):
            return {"action": "mode", "mode": "clone"}
        if any(word in compact for word in ["pc画面のみ", "内蔵のみ", "internal"]):
            return {"action": "mode", "mode": "internal"}
        if any(word in compact for word in ["セカンドスクリーンのみ", "外部のみ", "external"]):
            return {"action": "mode", "mode": "external"}

        brightness_match = re.search(r"(?:明るさ|brightness)[^\d]*(?P<level>\d{1,3})", text, flags=re.IGNORECASE)
        if brightness_match:
            level = max(0, min(100, int(brightness_match.group("level"))))
            return {"action": "brightness", "level": level}
        if any(word in compact for word in ["暗く", "暗め", "暗くして", "brightnessdown"]):
            return {"action": "brightness_delta", "delta": -20}
        if any(word in compact for word in ["明るく", "明るめ", "明るくして", "brightnessup"]):
            return {"action": "brightness_delta", "delta": 20}

        resolution = parse_resolution(text)
        if resolution and any(word in compact for word in ["解像度", "resolution"]):
            width, height = resolution
            return {"action": "resolution", "width": width, "height": height}
        if any(word in compact for word in ["解像度を下げ", "解像度下げ", "resolutiondown"]):
            return {"action": "resolution_delta", "direction": "down"}
        if any(word in compact for word in ["解像度を上げ", "解像度上げ", "resolutionup"]):
            return {"action": "resolution_delta", "direction": "up"}

        if any(word in compact for word in ["上下反転", "逆さ", "landscape_flipped"]):
            return {"action": "orientation", "orientation": "landscape_flipped"}
        if any(word in compact for word in ["縦反転", "portrait_flipped"]):
            return {"action": "orientation", "orientation": "portrait_flipped"}
        if any(word in compact for word in ["横向き", "横", "landscape"]):
            return {"action": "orientation", "orientation": "landscape"}
        if any(word in compact for word in ["縦向き", "縦", "portrait"]):
            return {"action": "orientation", "orientation": "portrait"}

        return None

    def _search_query(self, text: str) -> str | None:
        normalized = text.strip()
        patterns = [
            r"^検索して\s*(?P<query>.+)$",
            r"^検索して\s*(?P<query>.+?)\s*まとめて$",
            r"^(?P<query>.+?)について検索して$",
            r"^(?P<query>.+?)について検索して\s*まとめて$",
            r"^(?P<query>.+?)を検索して$",
            r"^(?P<query>.+?)を検索して\s*まとめて$",
            r"^(?P<query>.+?)で検索して$",
            r"^(?P<query>.+?)で検索して\s*まとめて$",
            r"^(?P<query>.+?)検索して$",
            r"^(?P<query>.+?)検索して\s*まとめて$",
            r"^search\s+(?P<query>.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                query = self._clean_query(match.group("query"))
                return query or None
        return None

    def _research_query(self, text: str) -> str | None:
        normalized = text.strip()
        patterns = [
            r"^検索して\s*(?P<query>.+?)\s*(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)について\s*(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)を\s*(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)について(?:ウェブサイト|webサイト|サイト)をいくつか検索し(?:て)?、?その内容を(?:最後に)?(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)について(?:ウェブサイト|webサイト|サイト)をいくつか調べ(?:て)?、?その内容を(?:最後に)?(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)について調べて$",
            r"^(?P<query>.+?)について調べて\s*(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)を調べて$",
            r"^(?P<query>.+?)を調べて\s*(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)調べて$",
            r"^(?P<query>.+?)調べて\s*(?:まとめて|要約して|要約)$",
            r"^(?P<query>.+?)について検索して\s*まとめて$",
            r"^(?P<query>.+?)について検索して\s*(?:要約して|要約)$",
            r"^(?P<query>.+?)を検索して\s*まとめて$",
            r"^(?P<query>.+?)を検索して\s*(?:要約して|要約)$",
            r"^(?P<query>.+?)検索して\s*まとめて$",
            r"^(?P<query>.+?)検索して\s*(?:要約して|要約)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                query = self._clean_query(match.group("query"))
                return query or None
        return None

    def _clean_query(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query).strip(" 　")
        cleaned = re.sub(r"(について|とは)$", "", cleaned).strip(" 　")
        return cleaned

    def _run_google_search(self, query: str) -> AgentResponse:
        provider_name, result = self._search_with_fallback(query)
        return AgentResponse(
            text=self._direct_tool_text(f"`{query}` を{provider_name}で検索しました。", result),
            steps=1,
        )

    def _run_search_and_summarize(self, query: str) -> AgentResponse:
        return self._run_google_top_results_summary(query)

    def _run_google_top_results_summary(self, query: str) -> AgentResponse:
        self._trace("検索要約フロー 1/4: Googleでキーワードを検索します。")
        provider_name, search_result = self._search_google_first(query)
        if not search_result.ok:
            return AgentResponse(text=search_result.content, steps=1)

        self._trace(f"検索要約フロー 2/4: 検索結果の上位{SEARCH_SUMMARY_LINK_LIMIT}件を取得します。")
        links = self._top_result_links(limit=SEARCH_SUMMARY_LINK_LIMIT)
        if not links:
            text_result = self._run_tool("browser", {"action": "text", "selector": "body", "timeout_ms": 10000})
            self._trace_tool_result(text_result)
            if not text_result.ok:
                return AgentResponse(text=self._direct_tool_text(f"`{query}` を{provider_name}で検索しました。", search_result), steps=2)

            summary = self._summarize_search_text(query, text_result.content)
            return AgentResponse(
                text=(
                    f"`{query}` を{provider_name}で検索しましたが、上位リンクを取得できませんでした。\n\n"
                    f"{summary}\n\n"
                    f"{search_result.content}"
                ),
                steps=2,
            )

        self._trace(f"検索要約フロー 3/4: 上位{len(links)}件のリンク先に移動して本文を読み込みます。")
        page_summaries = self._inspect_top_links(query, links)
        link_lines = "\n".join(f"{index}. {item['title']}\n   {item['url']}" for index, item in enumerate(links, start=1))

        self._trace("検索要約フロー 4/4: リンク先の内容をLLMでまとめます。")
        summary = self._combine_page_summaries(query, page_summaries)
        return AgentResponse(
            text=(
                f"`{query}` を{provider_name}で検索し、上位{len(links)}件のリンク先内容を確認しました。\n\n"
                f"{summary}\n\n"
                f"確認したリンク:\n{link_lines}\n\n"
                f"{search_result.content}"
            ),
            steps=4,
        )

    def _search_google_first(self, query: str) -> tuple[str, ToolResult]:
        google_result = self._run_search_flow(
            provider_name="Google",
            home_url="https://www.google.com/",
            query=query,
        )
        if google_result.ok and not self._is_google_blocked(google_result.content):
            return "Google", google_result

        self._trace("Googleの検索結果表示が完了しなかったため、DuckDuckGoで再検索します。")
        duckduckgo_result = self._run_search_flow(
            provider_name="DuckDuckGo",
            home_url="https://duckduckgo.com/",
            query=query,
        )
        return "DuckDuckGo", duckduckgo_result

    def _run_autonomous_research(self, query: str) -> AgentResponse:
        active_query = query
        provider_name = "検索エンジン"
        search_result = ToolResult("")
        all_links: list[dict[str, str]] = []
        page_summaries: list[dict[str, str]] = []
        visited_urls: set[str] = set()
        cycles_run = 0

        for cycle in range(1, RESEARCH_MAX_CYCLES + 1):
            cycles_run = cycle
            self._trace(f"自律調査サイクル {cycle}/{RESEARCH_MAX_CYCLES}: 計画")
            plan = self._plan_research_cycle(query, active_query, page_summaries)
            self._trace(f"計画: {plan}")

            self._trace(f"自律調査サイクル {cycle}/{RESEARCH_MAX_CYCLES}: 実行 - `{active_query}` を検索します。")
            provider_name, search_result = self._search_with_fallback(active_query)
            if not search_result.ok:
                self._trace("観察: 検索に失敗しました。")
                if cycle == RESEARCH_MAX_CYCLES:
                    return AgentResponse(text=search_result.content, steps=cycle)
                active_query = self._replan_search_query(query, active_query, page_summaries, "検索に失敗した")
                continue

            self._trace("観察: 検索結果ページからリンクを抽出します。")
            links = self._top_result_links(limit=SEARCH_SUMMARY_LINK_LIMIT)
            new_links = [link for link in links if link["url"] not in visited_urls]
            all_links.extend(link for link in new_links if link["url"] not in {item["url"] for item in all_links})

            if not new_links:
                self._trace("観察: 新しいリンクを取得できませんでした。")
            else:
                self._trace(f"調査: 新しいリンク {len(new_links)} 件を開いて本文を確認します。")
                inspected = self._inspect_top_links(query, new_links)
                page_summaries.extend(inspected)
                visited_urls.update(item["url"] for item in new_links)

            useful_count = len([item for item in page_summaries if self._is_useful_page_summary(item)])
            self._trace(f"観察: 有用なページ要約 {useful_count} 件 / 目標 {RESEARCH_MIN_USEFUL_PAGES} 件")
            if useful_count >= RESEARCH_MIN_USEFUL_PAGES:
                self._trace("観察: 最終まとめに十分な情報が集まりました。")
                break

            if cycle < RESEARCH_MAX_CYCLES:
                self._trace("再計画: 情報が不足しているため検索語を見直します。")
                active_query = self._replan_search_query(query, active_query, page_summaries, "有用なページが不足している")

        if not page_summaries and search_result.ok:
            text_result = self._run_tool("browser", {"action": "text", "selector": "body", "timeout_ms": 10000})
            self._trace_tool_result(text_result)
            if not text_result.ok:
                return AgentResponse(text=self._direct_tool_text(f"`{query}` を{provider_name}で検索しました。", search_result), steps=1)

            summary = self._summarize_search_text(query, text_result.content)
            return AgentResponse(
                text=(
                    f"`{query}` を{provider_name}で検索しましたが、リンク先の十分な調査ができませんでした。\n\n"
                    f"{summary}\n\n"
                    f"{search_result.content}"
                ),
                steps=cycles_run,
            )

        link_lines = "\n".join(f"{index}. {item['title']}\n   {item['url']}" for index, item in enumerate(all_links, start=1))
        summary = self._combine_page_summaries(query, page_summaries)
        return AgentResponse(
            text=(
                f"`{query}` を{provider_name}で検索し、自律的に計画・実行・観察・調査・再計画を行いました。\n\n"
                f"{summary}\n\n"
                f"確認したリンク:\n{link_lines}\n\n"
                f"{search_result.content}"
            ),
            steps=cycles_run,
        )

    def _plan_research_cycle(self, query: str, active_query: str, page_summaries: list[dict[str, str]]) -> str:
        useful_count = len([item for item in page_summaries if self._is_useful_page_summary(item)])
        if useful_count == 0:
            return f"`{active_query}` で検索し、上位リンクから公式情報・概要・直近情報を集める。"
        return f"不足している観点を補うため `{active_query}` で再検索し、有用ページを {RESEARCH_MIN_USEFUL_PAGES} 件以上に増やす。"

    def _replan_search_query(
        self,
        query: str,
        active_query: str,
        page_summaries: list[dict[str, str]],
        reason: str,
    ) -> str:
        prompt = (
            f"元の調査テーマ: {query}\n"
            f"現在の検索語: {active_query}\n"
            f"再計画理由: {reason}\n\n"
            "次に検索すべき日本語または英語の検索語を1行だけ返してください。"
            "公式情報、会社概要、製品、事業、ニュースなど調査に役立つ語を足してください。"
            "説明文や引用符は不要です。\n\n"
            "既に得た要約:\n"
            + "\n".join(f"- {item['summary']}" for item in page_summaries[-5:])
        )
        fallback = f"{query} 公式 会社概要 製品 事業 最新"
        candidate = self._summarize_with_llm(prompt, fallback=fallback)
        candidate = re.sub(r"^[`\"']|[`\"']$", "", candidate).strip()
        candidate = candidate.splitlines()[0].strip() if candidate else fallback
        if not candidate or candidate == active_query:
            return fallback
        return candidate

    def _is_useful_page_summary(self, item: dict[str, str]) -> bool:
        summary = item.get("summary", "")
        if summary.startswith(("ページを開けませんでした", "本文を取得できませんでした")):
            return False
        if "主要情報を確認できませんでした" in summary:
            return False
        if "要約できるテキストが少なすぎました" in summary:
            return False
        return len(summary) >= 30

    def _top_result_links(self, limit: int) -> list[dict[str, str]]:
        self._trace_tool_call("browser", {"action": "links", "limit": limit})
        links_result = self._run_tool("browser", {"action": "links", "limit": limit})
        self._trace_tool_result(links_result)
        if not links_result.ok:
            return []
        try:
            parsed = json.loads(links_result.content)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        links: list[dict[str, str]] = []
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(item.get("url"), str):
                links.append({"title": item["title"], "url": item["url"]})
        return links[:limit]

    def _inspect_top_links(self, query: str, links: list[dict[str, str]]) -> list[dict[str, str]]:
        page_summaries: list[dict[str, str]] = []
        for index, link in enumerate(links, start=1):
            self._trace(f"上位リンク {index}/{len(links)} を確認します: {self._trim_for_trace(link['title'])}")
            goto_args = {"action": "goto", "url": link["url"], "timeout_ms": 20000}
            self._trace_tool_call("browser", goto_args)
            goto_result = self._run_tool("browser", goto_args)
            self._trace_tool_result(goto_result)
            if not goto_result.ok:
                page_summaries.append({**link, "summary": f"ページを開けませんでした: {goto_result.content}"})
                continue

            wait_args = {"action": "wait", "load_state": "domcontentloaded", "text_min_length": 80, "timeout_ms": 15000}
            self._trace_tool_call("browser", wait_args)
            wait_result = self._run_tool("browser", wait_args)
            self._trace_tool_result(wait_result)

            text_args = {
                "action": "text",
                "selector": "body",
                "timeout_ms": 30000,
                "scroll_to_bottom": True,
                "max_chars": 0,
            }
            self._trace_tool_call("browser", text_args)
            text_result = self._run_tool("browser", text_args)
            self._trace_tool_result(text_result)
            if not text_result.ok:
                page_summaries.append({**link, "summary": f"本文を取得できませんでした: {text_result.content}"})
                continue

            page_summaries.append({**link, "content": text_result.content})
        return page_summaries

    def _search_with_fallback(self, query: str) -> tuple[str, ToolResult]:
        google_result = self._run_search_flow(
            provider_name="Google",
            home_url="https://www.google.com/",
            query=query,
        )
        if google_result.ok and not self._is_google_blocked(google_result.content):
            return "Google", google_result

        self._trace("Googleの検索結果表示が完了しなかったため、DuckDuckGoで再検索します。")
        duckduckgo_result = self._run_search_flow(
            provider_name="DuckDuckGo",
            home_url="https://duckduckgo.com/",
            query=query,
        )
        return "DuckDuckGo", duckduckgo_result

    def _run_search_flow(self, provider_name: str, home_url: str, query: str) -> ToolResult:
        search_selector = "textarea[name='q'], input[name='q']"

        self._trace_tool_call("browser", {"action": "goto", "url": home_url})
        goto_result = self._run_tool("browser", {"action": "goto", "url": home_url})
        self._trace_tool_result(goto_result)
        if not goto_result.ok:
            return goto_result

        self._trace_tool_call("browser", {"action": "type", "selector": search_selector, "text": query})
        type_result = self._run_tool(
            "browser",
            {"action": "type", "selector": search_selector, "text": query},
        )
        self._trace_tool_result(type_result)
        if not type_result.ok:
            return type_result

        self._trace_tool_call("browser", {"action": "press", "selector": search_selector, "key": "Enter"})
        press_result = self._run_tool(
            "browser",
            {"action": "press", "selector": search_selector, "key": "Enter"},
        )
        self._trace_tool_result(press_result)
        if not press_result.ok:
            return press_result

        wait_args = {"action": "wait", "load_state": "load", "text_min_length": 20, "timeout_ms": 20000}
        self._trace_tool_call("browser", wait_args)
        wait_result = self._run_tool("browser", wait_args)
        self._trace_tool_result(wait_result)
        if not wait_result.ok:
            return ToolResult(
                f"{provider_name}の検索結果ページへの遷移は始まりましたが、描画完了を確認できませんでした。\n{wait_result.content}",
                ok=False,
            )

        title_result = self._run_tool("browser", {"action": "title"})
        self._trace_tool_result(title_result)
        return title_result

    def _is_google_blocked(self, content: str) -> bool:
        lowered = content.lower()
        return "google.com/sorry/" in lowered or "/sorry/index" in lowered or "unusual traffic" in lowered

    def _summarize_search_text(self, query: str, content: str) -> str:
        ignored = {
            "DuckDuckGo",
            "すべて",
            "画像",
            "動画",
            "ニュース",
            "ショッピング",
            "地図",
            "検索",
            "設定",
            "プライバシー",
            "利用規約",
            "広告",
            "保護されています",
            "画像をさらに表示",
            "これは役に立ちましたか？",
        }
        ignored_prefixes = (
            "DuckDuckGo",
            "セーフサーチ",
            "検索設定",
            "メニューを開く",
            "日本",
            "全期間",
        )
        lines: list[str] = []
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 8 or line in ignored:
                continue
            if line == query or any(line.startswith(prefix) for prefix in ignored_prefixes):
                continue
            if line.startswith(("http://", "https://")):
                continue
            if re.fullmatch(r"[\w.-]+\.[a-z]{2,}.*", line, flags=re.IGNORECASE):
                continue
            if line not in lines:
                lines.append(line)
            if len(lines) >= 8:
                break

        if not lines:
            return "検索結果の本文を取得しましたが、要約できるテキストが少なすぎました。"

        bullets = "\n".join(f"- {line}" for line in lines[:5])
        return f"検索結果から見える `{query}` の要点候補:\n{bullets}"

    def _combine_page_summaries(self, query: str, page_summaries: list[dict[str, str]]) -> str:
        if not page_summaries:
            return f"`{query}` について、上位リンクから十分な情報を取得できませんでした。"

        successful = [item for item in page_summaries if item.get("content")]
        source_items = successful or page_summaries

        summary = self._overall_summary_text_with_llm(query, source_items)
        return f"`{query}` について複数サイトを確認したまとめ（{SEARCH_SUMMARY_TARGET_CHARS}文字程度）:\n{summary}"

    def _overall_summary_text_with_llm(self, query: str, source_items: list[dict[str, str]]) -> str:
        per_page_chars = max(1000, LLM_SUMMARY_SOURCE_CHARS // max(1, len(source_items)))
        source_text = "\n".join(
            f"--- ページ {index}: {item.get('title', '')} ---\n{self._copied_page_content_for_llm(item, per_page_chars)}"
            for index, item in enumerate(source_items, start=1)
        )
        prompt = (
            f"検索テーマ: {query}\n\n"
            f"以下は複数のウェブサイトからコピーした本文です。"
            f"全体として重要な点だけを、日本語で{SEARCH_SUMMARY_TARGET_CHARS}文字程度、最大180文字以内の1段落にまとめてください。\n"
            "ルール:\n"
            "- 各ページの個別要約ではなく、コピーされた本文全体を材料にして統合する\n"
            "- コピー本文から短いタイトル・ナビゲーション・共有ボタン文言を拾って並べない\n"
            "- 複数サイトに共通する内容や全体像を統合する\n"
            "- 特定の1サイトや1記事の細部だけに偏らない\n"
            "- 見出し、記事タイトル、引用文をつなげるだけにしない\n"
            "- 主語と述語のある新しい説明文として書く\n"
            "- 会社・製品・事業・直近の焦点など、検索テーマの説明としてまとまる内容にする\n"
            "- アクセシビリティ、Cookie、広告、別記事、株価表示だけの情報は無視する\n"
            "- TOP、SHARE、徹底解説、料金、サイト名、URL、出典名、絵文字、箇条書き、前置きは書かない\n"
            "- 本文にない推測は入れない\n\n"
            f"{source_text[:LLM_SUMMARY_SOURCE_CHARS]}"
        )
        summary = self._summarize_with_llm(
            prompt,
            fallback=self._overall_summary_text(query, [self._copied_page_content_for_llm(item, per_page_chars) for item in source_items]),
        )
        return summary

    def _copied_page_content_for_llm(self, item: dict[str, str], max_chars: int) -> str:
        content = item.get("content")
        if content:
            copied = self._copy_body_text_for_summary(content)
            if len(copied) <= max_chars:
                return copied
            return copied[:max_chars].rstrip() + "\n...<page content truncated>"
        return item.get("summary", "")

    def _copy_body_text_for_summary(self, content: str) -> str:
        copied_lines: list[str] = []
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not self._is_summary_source_line(line):
                continue
            if line not in copied_lines:
                copied_lines.append(line)
        if copied_lines:
            return "\n".join(copied_lines)
        return re.sub(r"\s+", " ", content).strip()

    def _is_summary_source_line(self, line: str) -> bool:
        if len(line) < 35 or len(line) > 500:
            return False
        blocked_exact = {
            "TOP",
            "SHARE",
            "MENU",
            "CONTACT",
            "お問い合わせ",
            "サイトマップ",
        }
        if line.upper() in blocked_exact:
            return False
        blocked_contains = (
            "TOP AI",
            "SHARE",
            "お役立ち情報",
            "徹底解説",
            "料金",
            "関連記事",
            "こちらもおすすめ",
            "続きを読む",
            "アクセシビリティ",
            "Cookie",
            "プライバシー",
            "ログイン",
            "無料相談",
            "資料請求",
        )
        if any(part in line for part in blocked_contains):
            return False
        if re.search(r"https?://|www\.", line):
            return False
        sentence_marks = line.count("。") + line.count("です") + line.count("ます")
        if sentence_marks == 0 and len(line) < 90:
            return False
        return True

    def _summarize_with_llm(self, prompt: str, fallback: str) -> str:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "あなたはウェブ調査結果を日本語で簡潔に要約するアシスタントです。",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            options: dict[str, Any] = {}
            if self.num_ctx is not None:
                options["num_ctx"] = self.num_ctx
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if options:
                kwargs["extra_body"] = {"options": options}

            completion = self.client.chat.completions.create(**kwargs)
            self.token_usage.add(completion.usage)
            if self.token_usage.has_values():
                self._trace(self.token_usage.render())
            content = completion.choices[0].message.content or ""
            cleaned = re.sub(r"\s+", " ", content).strip()
            if cleaned and not self._looks_non_japanese(cleaned):
                return self._clean_llm_summary(cleaned)
        except Exception as exc:
            self._trace(f"LLM要約に失敗しました: {exc}")
        return fallback

    def _clean_llm_summary(self, content: str) -> str:
        cleaned = content.strip().strip('"`')
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"[\U0001F300-\U0001FAFF]", "", cleaned)
        cleaned = cleaned.strip()
        cleaned = re.sub(r"^(上級者向け|初心者向け|ポイント|結論|TOP|SHARE)\s*[:：]?\s*", "", cleaned)
        cleaned = re.sub(r"^(要約|まとめ)\s*[:：]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned)
        cleaned = re.sub(r"\b(?:Hugging Face|Google|GitHub|Qiita|Zenn)\s*(?:ブログ|Blog)?\s*[:：]\s*", "", cleaned)
        cleaned = re.sub(r"\b(?:TOP|SHARE)\b", "", cleaned)
        cleaned = cleaned.replace("徹底解説", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" 、。:：")
        if cleaned and not cleaned.endswith("。"):
            cleaned += "。"
        if len(cleaned) <= SEARCH_SUMMARY_MAX_CHARS:
            return cleaned
        return cleaned[: SEARCH_SUMMARY_MAX_CHARS - 1].rstrip("、。 ") + "。"

    def _summary_needs_rewrite(self, summary: str) -> bool:
        if re.search(r"https?://|www\.", summary):
            return True
        if re.search(r"[\U0001F300-\U0001FAFF]", summary):
            return True
        if any(marker in summary for marker in ["ブログ:", "ブログ：", "上級者向け", "初心者向け"]):
            return True
        return len(summary) > SEARCH_SUMMARY_MAX_CHARS

    def _overall_summary_text(self, query: str, summaries: list[str]) -> str:
        fragments: list[str] = []
        for summary in summaries:
            for fragment in re.split(r"\s*/\s*|。|\. ", summary):
                cleaned = re.sub(r"\s+", " ", fragment).strip(" -:。")
                if len(cleaned) < 12 or cleaned in fragments:
                    continue
                fragments.append(cleaned)
                if len(fragments) >= 5:
                    break
            if len(fragments) >= 5:
                break

        if not fragments:
            return f"`{query}` について、取得できたページ本文から明確な要点を抽出できませんでした。"

        text = ""
        for fragment in fragments:
            candidate = fragment.rstrip("。") + "。"
            if not text:
                text = candidate
            elif len(text) + len(candidate) <= SEARCH_SUMMARY_MAX_CHARS:
                text += candidate
            if len(text) >= SEARCH_SUMMARY_TARGET_CHARS:
                break

        if len(text) > SEARCH_SUMMARY_MAX_CHARS:
            return text[: SEARCH_SUMMARY_TARGET_CHARS].rstrip("、。 ") + "。"
        return text

    def _important_lines(self, content: str) -> list[str]:
        ignored_exact = {
            "DuckDuckGo",
            "Google",
            "検索",
            "メニュー",
            "ログイン",
            "登録",
            "広告",
            "同意する",
            "拒否する",
            "Cookie",
            "Cookies",
            "プライバシー",
            "利用規約",
            "お問い合わせ",
            "サイトマップ",
        }
        ignored_prefixes = (
            "http://",
            "https://",
            "©",
            "Copyright",
            "All rights reserved",
            "JavaScript",
            "このサイトではCookie",
        )
        ignored_contains = (
            "アクセシビリティ",
            "Accessibility",
            "プライバシー",
            "Cookie",
            "広告",
            "関連記事",
            "こちらもおすすめ",
            "続きを読む",
            "シェア",
            "信頼される",
            "アクセスできます",
            "防災",
            "地震",
        )
        lines: list[str] = []
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 20 or len(line) > 260:
                continue
            if line in ignored_exact or any(line.startswith(prefix) for prefix in ignored_prefixes):
                continue
            if any(part in line for part in ignored_contains):
                continue
            if re.search(r"【[^】]*】", line) and not any(marker in line for marker in ["会社", "企業", "半導体", "製品", "事業"]):
                continue
            if re.fullmatch(r"[\W\d_]+", line):
                continue
            if line not in lines:
                lines.append(line)
        return lines

    def _prioritize_query_lines(self, query: str, lines: list[str]) -> list[str]:
        query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+", query)}
        if query.upper() == "AMD":
            query_terms.update({"amd", "advanced", "micro", "devices", "radeon", "ryzen", "epyc"})
        if query.lower() in {"intel", "インテル"}:
            query_terms.update({"intel", "インテル", "core", "xeon"})

        scored: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            lowered = line.lower()
            score = sum(3 for term in query_terms if term and term in lowered)
            if any(marker in line for marker in ["会社", "企業", "半導体", "製品", "事業", "AI", "CPU", "GPU", "データセンター"]):
                score += 2
            if any(marker in line for marker in ["ニュース", "発表", "公式", "概要"]):
                score += 1
            scored.append((score, -index, line))

        relevant = [line for score, _, line in sorted(scored, reverse=True) if score > 0]
        if len(relevant) >= 20:
            return relevant[:120]
        return (relevant + [line for line in lines if line not in relevant])[:120]

    def _known_site_url(self, normalized: str) -> str | None:
        for name, url in KNOWN_SITES.items():
            if name in normalized:
                return url
        return None

    def _is_browser_open_request(self, normalized: str) -> bool:
        return any(word in normalized for word in ["open", "browser", "browse", "ブラウザ", "開"])

    def _looks_non_japanese(self, content: str) -> bool:
        stripped = content.strip()
        if not stripped:
            return False
        thai = re.search(r"[\u0e00-\u0e7f]", stripped) is not None
        hangul = re.search(r"[\uac00-\ud7af]", stripped) is not None
        arabic = re.search(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]", stripped) is not None
        hebrew = re.search(r"[\u0590-\u05ff]", stripped) is not None
        cyrillic = re.search(r"[\u0400-\u04ff]", stripped) is not None
        devanagari = re.search(r"[\u0900-\u097f]", stripped) is not None
        japanese = re.search(r"[\u3040-\u30ff\u3400-\u9fff]", stripped) is not None
        english_meta = any(
            marker in stripped
            for marker in [
                "The user has provided",
                "Final Approach",
                "Self-Correction",
                "I was unable to process",
                "Could you please provide",
                "正在翻译中",
                "请提供英文文本",
                "請提供英文文本",
            ]
        )
        ascii_letters = len(re.findall(r"[A-Za-z]", stripped))
        japanese_chars = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", stripped))
        mostly_english = ascii_letters > 120 and ascii_letters > japanese_chars * 3
        corrupted = bool(re.search(r"([|\[\]{};:,])(?:\s*\1){4,}", stripped)) or bool(re.search(r"(?:\|\s*\[\s*){3,}", stripped))
        corrupted = corrupted or stripped.count(")") > 40 or stripped.count("）") > 40
        chinese_like = (
            len(re.findall(r"[的这那为们个种通用同意]", stripped)) > 20
            and len(re.findall(r"[\u3040-\u30ff]", stripped)) == 0
        )
        unexpected_script = thai or hangul or arabic or hebrew or cyrillic or devanagari
        return (unexpected_script and not japanese) or english_meta or mostly_english or corrupted or chinese_like

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
            "このエージェントには3つのモードがあります。\n\n"
            "- chatモード: `/mode chat`。ツールや定型処理を使わず、入力をLLMへそのまま渡します\n"
            "- askモード: `/mode ask`。読み取り・検索・確認はできますが、ファイル削除や書き込みを制限します\n"
            "- agentモード: `/mode agent`。ブラウザ、ターミナル、ディスプレイ、サンドボックスを制限なしで操作します\n\n"
            "ask/agentモードでできる主な操作です。\n\n"
            "- 四則演算: `1+2`、`(10 - 4) / 3`、`3×4`、`1たす2` などを直接計算\n"
            "- コード生成: `PythonでCSVを読むコードを生成して`、`codegen: fizzbuzz in JavaScript` など\n"
            "- ブラウザ操作: `open browser`、`open https://example.com`、ページタイトル取得、クリック、入力、スクリーンショットなど\n"
            "- ターミナル操作: `terminal: Get-Location` のようにローカルコマンドを実行\n"
            "- ディスプレイ操作: `画面を暗くして`、`解像度を下げて`、`ディスプレイを複製して` など\n"
            "- サンドボックス操作: `sandbox: python --version`、`サンドボックスをリセットして` など\n"
            "- 拡張: `src/openhands_agent/tools/` に新しい `Tool` を追加して操作を増やせます\n\n"
            f"現在は{self._mode_label()}です。"
        )

    def _direct_tool_text(self, success_text: str, result: ToolResult) -> str:
        if result.ok:
            return f"{success_text}\n{result.content}"
        return result.content

    def _run_tool(self, name: str, raw_arguments: str | dict[str, Any]) -> ToolResult:
        return self.mcp_tools.run(name, raw_arguments)

    def _mcp_restriction_reason(self, name: str, raw_arguments: str | dict[str, Any]) -> str | None:
        if self.mode != ASK_MODE:
            return None
        return self._ask_restriction_reason(name, raw_arguments)

    def _ask_restriction_reason(self, name: str, raw_arguments: str | dict[str, Any]) -> str | None:
        arguments = self._tool_arguments_for_check(raw_arguments)
        if arguments is None:
            return None

        if name == "browser" and str(arguments.get("action", "")).lower() == "screenshot":
            return "スクリーンショット保存はファイル書き込みです。"

        if name == "sandbox":
            action = str(arguments.get("action", "")).lower()
            if action in {"reset", "write", "delete"}:
                return f"sandbox action={action} はサンドボックス内のファイルを変更します。"
            if action == "run":
                command = str(arguments.get("command", ""))
                return self._unsafe_shell_command_reason(command)

        if name == "terminal_run":
            command = str(arguments.get("command", ""))
            return self._unsafe_shell_command_reason(command)

        return None

    def _tool_arguments_for_check(self, raw_arguments: str | dict[str, Any]) -> dict[str, Any] | None:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _unsafe_shell_command_reason(self, command: str) -> str | None:
        normalized = re.sub(r"\s+", " ", command).strip().lower()
        if not normalized:
            return None
        unsafe_patterns = [
            (r"(^|[^<])>{1,2}", "リダイレクトはファイル書き込みの可能性があります。"),
            (r"\b(remove-item|rm|del|erase|rmdir|rd|ri)\b", "削除コマンドが含まれています。"),
            (r"\b(new-item|ni|set-content|add-content|out-file|copy-item|cp|copy)\b", "ファイル作成または書き込みコマンドが含まれています。"),
            (r"\b(move-item|mv|move|rename-item|ren|mkdir|md|touch)\b", "ファイルやディレクトリを変更するコマンドが含まれています。"),
            (r"\b(chmod|icacls|attrib)\b", "ファイル権限や属性を変更するコマンドが含まれています。"),
            (r"\bgit\s+(add|commit|rm|mv|reset|clean|checkout|switch|restore|pull|merge|rebase|apply|stash)\b", "Git操作が作業ツリーや履歴を書き換える可能性があります。"),
            (r"\b(write_text|write_bytes|writelines|truncate)\b", "プログラム内の書き込み処理が含まれています。"),
            (r"\bopen\s*\([^)]*['\"][wa+]", "プログラム内のファイル書き込みモードが含まれています。"),
        ]
        for pattern, reason in unsafe_patterns:
            if re.search(pattern, normalized):
                return reason
        return None

    def _non_japanese_fallback_text(self, user_input: str = "") -> str:
        return (
            "すみません、モデルの応答が日本語から外れました。"
            "`open https://...`、`terminal: Get-Location`、`codegen: tetris game`、"
            "`ディスプレイを拡張して`、`明るさを70にして` のように指示してください。"
        )

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
