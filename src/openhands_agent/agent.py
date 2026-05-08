from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from openai import APIConnectionError, NotFoundError, OpenAI

from .tools.base import ToolRegistry, ToolResult
from .tools.display import parse_resolution


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


SEARCH_SUMMARY_LINK_LIMIT = 10
SEARCH_SUMMARY_TARGET_CHARS = 150
SEARCH_SUMMARY_MAX_CHARS = 180
LLM_SUMMARY_SOURCE_CHARS = 24000
RESEARCH_MAX_CYCLES = 3
RESEARCH_MIN_USEFUL_PAGES = 3


@dataclass
class AgentResponse:
    text: str
    steps: int
    token_usage: "TokenUsage | None" = None


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
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_input: str) -> AgentResponse:
        self.token_usage = TokenUsage()
        self._trace(f"入力を受け取りました: {self._trim_for_trace(user_input)}")
        direct_response = self._run_direct_command(user_input)
        if direct_response is not None:
            self._trace("直接操作として処理しました。")
            direct_response.token_usage = self.token_usage
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
                    text=(
                        "すみません、モデルの応答が日本語から外れました。"
                        "`open https://...`、`terminal: Get-Location`、`ディスプレイを拡張して`、`明るさを70にして` のように指示してください。"
                    ),
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
            result = self.tools.run("browser", {"action": "goto", "url": url})
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text(f"{url} を開きました。", result), steps=1)

        display_args = self._display_command(text, normalized)
        if display_args is not None:
            self._trace_tool_call("display", display_args)
            result = self.tools.run("display", display_args)
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text("ディスプレイ設定を操作しました。", result), steps=1)

        sandbox_args = self._sandbox_command(text, normalized)
        if sandbox_args is not None:
            self._trace_tool_call("sandbox", sandbox_args)
            result = self.tools.run("sandbox", sandbox_args)
            self._trace_tool_result(result)
            return AgentResponse(text=self._direct_tool_text("サンドボックスを操作しました。", result), steps=1)

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

    def _sandbox_command(self, text: str, normalized: str) -> dict[str, Any] | None:
        compact = re.sub(r"\s+", "", normalized)
        if "sandbox:" in normalized:
            command = text.split(":", 1)[1].strip()
            if not command:
                return {"action": "info"}
            return {"action": "run", "command": command}
        if "サンドボックス" not in compact and "sandbox" not in normalized:
            return None
        if any(word in compact for word in ["初期化", "リセット", "reset"]):
            return {"action": "reset"}
        if any(word in compact for word in ["一覧", "list"]):
            return {"action": "list"}
        if any(word in compact for word in ["情報", "場所", "info"]):
            return {"action": "info"}
        return {"action": "info"}

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
            text_result = self.tools.run("browser", {"action": "text", "selector": "body", "timeout_ms": 10000})
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
            text_result = self.tools.run("browser", {"action": "text", "selector": "body", "timeout_ms": 10000})
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
        links_result = self.tools.run("browser", {"action": "links", "limit": limit})
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
            goto_result = self.tools.run("browser", goto_args)
            self._trace_tool_result(goto_result)
            if not goto_result.ok:
                page_summaries.append({**link, "summary": f"ページを開けませんでした: {goto_result.content}"})
                continue

            wait_args = {"action": "wait", "load_state": "domcontentloaded", "text_min_length": 80, "timeout_ms": 15000}
            self._trace_tool_call("browser", wait_args)
            wait_result = self.tools.run("browser", wait_args)
            self._trace_tool_result(wait_result)

            text_args = {
                "action": "text",
                "selector": "body",
                "timeout_ms": 30000,
                "scroll_to_bottom": True,
                "max_chars": 0,
            }
            self._trace_tool_call("browser", text_args)
            text_result = self.tools.run("browser", text_args)
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
        goto_result = self.tools.run("browser", {"action": "goto", "url": home_url})
        self._trace_tool_result(goto_result)
        if not goto_result.ok:
            return goto_result

        self._trace_tool_call("browser", {"action": "type", "selector": search_selector, "text": query})
        type_result = self.tools.run(
            "browser",
            {"action": "type", "selector": search_selector, "text": query},
        )
        self._trace_tool_result(type_result)
        if not type_result.ok:
            return type_result

        self._trace_tool_call("browser", {"action": "press", "selector": search_selector, "key": "Enter"})
        press_result = self.tools.run(
            "browser",
            {"action": "press", "selector": search_selector, "key": "Enter"},
        )
        self._trace_tool_result(press_result)
        if not press_result.ok:
            return press_result

        wait_args = {"action": "wait", "load_state": "load", "text_min_length": 20, "timeout_ms": 20000}
        self._trace_tool_call("browser", wait_args)
        wait_result = self.tools.run("browser", wait_args)
        self._trace_tool_result(wait_result)
        if not wait_result.ok:
            return ToolResult(
                f"{provider_name}の検索結果ページへの遷移は始まりましたが、描画完了を確認できませんでした。\n{wait_result.content}",
                ok=False,
            )

        title_result = self.tools.run("browser", {"action": "title"})
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
        return ((thai or hangul) and not japanese) or english_meta or mostly_english or corrupted

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
