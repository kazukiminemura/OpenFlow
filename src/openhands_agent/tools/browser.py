from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

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
                "enum": ["goto", "click", "type", "press", "text", "title", "screenshot", "evaluate"],
            },
            "url": {"type": "string", "description": "URL for goto."},
            "selector": {"type": "string", "description": "CSS selector for click, type, press, or text."},
            "text": {"type": "string", "description": "Text to type."},
            "key": {"type": "string", "description": "Keyboard key for press, such as Enter."},
            "script": {"type": "string", "description": "JavaScript expression for evaluate."},
            "path": {"type": "string", "description": "Optional screenshot output path."},
            "timeout_ms": {"type": "integer", "default": 10000, "minimum": 1000, "maximum": 60000},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, workdir: Path, headless: bool) -> None:
        self.workdir = workdir
        self.headless = headless
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

        if action == "text":
            selector = str(arguments.get("selector") or "body")
            content = page.locator(selector).inner_text(timeout=timeout)
            return ToolResult(self._trim(content))

        if action == "title":
            return ToolResult(f"title: {page.title()}\nurl: {page.url}")

        if action == "screenshot":
            path = self._screenshot_path(arguments.get("path"))
            page.screenshot(path=str(path), full_page=True, timeout=timeout)
            return ToolResult(f"Screenshot saved: {path}")

        if action == "evaluate":
            script = str(arguments["script"])
            value: Any = page.evaluate(script)
            return ToolResult(self._trim(repr(value)))

        return ToolResult(f"Unsupported browser action: {action}", ok=False)

    def _ensure_page(self) -> Page:
        if self._page is not None:
            return self._page
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page(viewport={"width": 1366, "height": 900})
        return self._page

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
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...<trimmed>"

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._playwright = None
        self._page = None
