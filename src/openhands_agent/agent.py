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
            r"^(?P<query>.+?)について調べて$",
            r"^(?P<query>.+?)を調べて$",
            r"^(?P<query>.+?)調べて$",
            r"^(?P<query>.+?)について検索して\s*まとめて$",
            r"^(?P<query>.+?)を検索して\s*まとめて$",
            r"^(?P<query>.+?)検索して\s*まとめて$",
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
        provider_name, result = self._search_with_fallback(query)
        if not result.ok:
            return AgentResponse(text=result.content, steps=1)

        links = self._top_result_links(limit=3)
        if not links:
            text_result = self.tools.run("browser", {"action": "text", "selector": "body", "timeout_ms": 10000})
            self._trace_tool_result(text_result)
            if not text_result.ok:
                return AgentResponse(text=self._direct_tool_text(f"`{query}` を{provider_name}で検索しました。", result), steps=1)

            summary = self._summarize_search_text(query, text_result.content)
            return AgentResponse(
                text=(
                    f"`{query}` を{provider_name}で検索し、結果ページを表示しました。\n\n"
                    f"{summary}\n\n"
                    f"{result.content}"
                ),
                steps=1,
            )

        page_summaries = self._inspect_top_links(query, links)
        link_lines = "\n".join(f"{index}. {item['title']}\n   {item['url']}" for index, item in enumerate(links, start=1))
        summary = self._combine_page_summaries(query, page_summaries)
        return AgentResponse(
            text=(
                f"`{query}` を{provider_name}で検索し、上位3件を確認しました。\n\n"
                f"{summary}\n\n"
                f"確認したリンク:\n{link_lines}\n\n"
                f"{result.content}"
            ),
            steps=1,
        )

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

            text_result = self.tools.run("browser", {"action": "text", "selector": "body", "timeout_ms": 10000})
            self._trace_tool_result(text_result)
            if not text_result.ok:
                page_summaries.append({**link, "summary": f"本文を取得できませんでした: {text_result.content}"})
                continue

            page_summaries.append({**link, "summary": self._summarize_page_text(query, text_result.content)})
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

    def _summarize_page_text(self, query: str, content: str) -> str:
        lines = self._important_lines(content)
        if not lines:
            return "本文を取得しましたが、要約できるテキストが少なすぎました。"

        query_lower = query.lower()
        scored: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            score = 0
            if query_lower and query_lower in line.lower():
                score += 4
            if any(marker in line for marker in ["とは", "会社", "企業", "製品", "発表", "事業", "概要"]):
                score += 2
            score += max(0, 3 - index // 3)
            scored.append((score, line))

        selected: list[str] = []
        for _, line in sorted(scored, key=lambda item: item[0], reverse=True):
            if line not in selected:
                selected.append(line)
            if len(selected) >= 3:
                break
        return " / ".join(selected)

    def _combine_page_summaries(self, query: str, page_summaries: list[dict[str, str]]) -> str:
        if not page_summaries:
            return f"`{query}` について、上位リンクから十分な情報を取得できませんでした。"

        lines = [f"`{query}` について上位リンクから確認した要点:"]
        for index, item in enumerate(page_summaries, start=1):
            lines.append(f"- {index}件目: {item['summary']}")
        return "\n".join(lines)

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
        lines: list[str] = []
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 20 or len(line) > 260:
                continue
            if line in ignored_exact or any(line.startswith(prefix) for prefix in ignored_prefixes):
                continue
            if re.fullmatch(r"[\W\d_]+", line):
                continue
            if line not in lines:
                lines.append(line)
            if len(lines) >= 40:
                break
        return lines

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
