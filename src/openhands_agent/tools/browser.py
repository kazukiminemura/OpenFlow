from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from .base import JsonDict, Tool, ToolResult


class BrowserTool(Tool):
    name = "browser"
    description = "Control a Chromium browser: navigate, click, type, extract text, run JavaScript, and take screenshots."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["goto", "click", "type", "press", "wait", "text", "links", "title", "screenshot", "evaluate"],
            },
            "url": {"type": "string", "description": "URL for goto."},
            "selector": {"type": "string", "description": "CSS selector for click, type, press, or text."},
            "text": {"type": "string", "description": "Text to type."},
            "key": {"type": "string", "description": "Keyboard key for press, such as Enter."},
            "load_state": {
                "type": "string",
                "enum": ["domcontentloaded", "load", "networkidle"],
                "description": "Optional Playwright load state for wait.",
            },
            "text_min_length": {
                "type": "integer",
                "description": "Optional minimum body text length to wait for.",
                "default": 0,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum text characters to return for text/evaluate. Use 0 for no trimming.",
                "default": 8000,
            },
            "scroll_to_bottom": {
                "type": "boolean",
                "description": "Scroll to the bottom before extracting text, useful for lazy-loaded pages.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of links to return for links.",
                "default": 10,
            },
            "script": {"type": "string", "description": "JavaScript expression for evaluate."},
            "path": {"type": "string", "description": "Optional screenshot output path."},
            "timeout_ms": {"type": "integer", "default": 10000, "minimum": 1000, "maximum": 60000},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workdir: Path,
        headless: bool,
        light_mode: bool = True,
        block_resources: set[str] | None = None,
        viewport_width: int = 1024,
        viewport_height: int = 720,
    ) -> None:
        self.workdir = workdir
        self.headless = headless
        self.light_mode = light_mode
        self.block_resources = block_resources or set()
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def run(self, arguments: JsonDict) -> ToolResult:
        page = self._ensure_page()
        timeout = int(arguments.get("timeout_ms", 10000))
        action = str(arguments["action"])

        if action == "goto":
            url = str(arguments["url"])
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            status = response.status if response else "no response"
            return ToolResult(f"Navigated to {page.url}\nstatus: {status}\ntitle: {page.title()}")

        if action == "click":
            selector = str(arguments["selector"])
            page.locator(selector).click(timeout=timeout)
            return ToolResult(f"Clicked {selector}\nurl: {page.url}")

        if action == "type":
            selector = str(arguments["selector"])
            text = str(arguments["text"])
            page.locator(selector).fill(text, timeout=timeout)
            return ToolResult(f"Typed into {selector}")

        if action == "press":
            selector = str(arguments.get("selector") or "body")
            key = str(arguments["key"])
            page.locator(selector).press(key, timeout=timeout)
            return ToolResult(f"Pressed {key} on {selector}")

        if action == "wait":
            load_state = arguments.get("load_state")
            if load_state:
                page.wait_for_load_state(str(load_state), timeout=timeout)

            selector = arguments.get("selector")
            if selector:
                page.locator(str(selector)).first.wait_for(state="visible", timeout=timeout)

            text_min_length = int(arguments.get("text_min_length", 0))
            if text_min_length > 0:
                page.wait_for_function(
                    "(minLength) => document.body && document.body.innerText.trim().length >= minLength",
                    arg=text_min_length,
                    timeout=timeout,
                )

            return ToolResult(f"Wait completed\nurl: {page.url}\ntitle: {page.title()}")

        if action == "text":
            selector = str(arguments.get("selector") or "body")
            if bool(arguments.get("scroll_to_bottom", False)):
                self._scroll_to_bottom(page, timeout=timeout)
            content = page.locator(selector).inner_text(timeout=timeout)
            max_chars = int(arguments.get("max_chars", 8000))
            return ToolResult(self._trim(content, limit=max_chars))

        if action == "links":
            limit = int(arguments.get("limit", 10))
            links = self._extract_links(page, limit=limit)
            return ToolResult(json.dumps(links, ensure_ascii=False, indent=2))

        if action == "title":
            return ToolResult(f"title: {page.title()}\nurl: {page.url}")

        if action == "screenshot":
            path = self._screenshot_path(arguments.get("path"))
            page.screenshot(path=str(path), full_page=True, timeout=timeout)
            return ToolResult(f"Screenshot saved: {path}")

        if action == "evaluate":
            script = str(arguments["script"])
            value: Any = page.evaluate(script)
            max_chars = int(arguments.get("max_chars", 8000))
            return ToolResult(self._trim(repr(value), limit=max_chars))

        return ToolResult(f"Unsupported browser action: {action}", ok=False)

    def _ensure_page(self) -> Page:
        if self._page is not None and not self._page.is_closed():
            return self._page
        self._page = None
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        if self._browser is None or not self._browser.is_connected():
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=self._launch_args(),
            )
        self._page = self._browser.new_page(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            reduced_motion="reduce",
        )
        if self.light_mode and self.block_resources:
            self._page.route("**/*", self._route_request)
        return self._page

    def _launch_args(self) -> list[str]:
        if not self.light_mode:
            return []
        return [
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-features=Translate,MediaRouter",
            "--disable-renderer-backgrounding",
            "--disable-sync",
            "--metrics-recording-only",
            "--mute-audio",
            "--no-first-run",
        ]

    def _route_request(self, route: Any) -> None:
        if route.request.resource_type in self.block_resources:
            route.abort()
            return
        route.continue_()

    def _extract_links(self, page: Page, limit: int) -> list[dict[str, str]]:
        raw_links: list[dict[str, str]] = page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href]')).map((link) => ({
                title: (link.innerText || link.getAttribute('aria-label') || '').trim(),
                url: link.href
            }))"""
        )
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        seen_domains: set[str] = set()

        for link in raw_links:
            title = self._clean_text(link.get("title", ""))
            url = str(link.get("url", ""))
            if not self._is_result_link(title, url):
                continue

            normalized_url = url.split("#", 1)[0]
            domain = urlparse(normalized_url).netloc.lower()
            if normalized_url in seen_urls or domain in seen_domains:
                continue

            results.append({"title": title, "url": normalized_url})
            seen_urls.add(normalized_url)
            seen_domains.add(domain)
            if len(results) >= limit:
                break

        return results

    def _is_result_link(self, title: str, url: str) -> bool:
        if len(title) < 4:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        host = parsed.netloc.lower()
        blocked_hosts = {
            "duckduckgo.com",
            "www.google.com",
            "google.com",
            "accounts.google.com",
            "support.google.com",
            "policies.google.com",
            "www.youtube.com",
            "youtube.com",
            "m.youtube.com",
        }
        if host in blocked_hosts:
            return False
        blocked_title_parts = [
            "画像",
            "動画",
            "ニュース",
            "ショッピング",
            "検索設定",
            "プライバシー",
            "利用規約",
            "フィードバック",
        ]
        return not any(part in title for part in blocked_title_parts)

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def _screenshot_path(self, value: object) -> Path:
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.workdir / path
        else:
            path = self.workdir / "agent-screenshot.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _trim(self, text: str, limit: int = 8000) -> str:
        if limit <= 0:
            return text
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...<trimmed>"

    def _scroll_to_bottom(self, page: Page, timeout: int) -> None:
        page.evaluate(
            """async (timeoutMs) => {
                const deadline = Date.now() + timeoutMs;
                let previousHeight = -1;
                while (Date.now() < deadline) {
                    const height = document.body ? document.body.scrollHeight : 0;
                    window.scrollTo(0, height);
                    await new Promise((resolve) => setTimeout(resolve, 250));
                    if (height === previousHeight) {
                        break;
                    }
                    previousHeight = height;
                }
                window.scrollTo(0, 0);
            }""",
            timeout,
        )

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._playwright = None
        self._page = None
