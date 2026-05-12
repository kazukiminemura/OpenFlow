from __future__ import annotations

import fnmatch
import re
from difflib import SequenceMatcher
from pathlib import Path

from .base import JsonDict, Tool, ToolResult


DEFAULT_EXCLUDED_DIRS = {
    ".agent_sandbox",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
}


class LocalSearchTool(Tool):
    name = "local_search"
    description = "Search local files under the configured workdir by file name and/or text content."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for in file names and/or file contents.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file path to search, relative to the configured workdir.",
                "default": ".",
            },
            "mode": {
                "type": "string",
                "enum": ["all", "name", "content"],
                "description": "Search file names, file contents, or both.",
                "default": "all",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines/files to return.",
                "default": 50,
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir.resolve()

    def run(self, arguments: JsonDict) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult("検索語を指定してください。", ok=False)

        mode = str(arguments.get("mode", "all")).lower()
        if mode not in {"all", "name", "content"}:
            return ToolResult(f"Unsupported local_search mode: {mode}", ok=False)

        max_results = min(200, max(1, int(arguments.get("max_results", 50))))
        root = self._resolve(arguments.get("path") or ".")
        if not root.exists():
            return ToolResult(f"Path does not exist: {self._display_path(root)}", ok=False)

        matches: list[str] = []
        for file_path in self._iter_files(root):
            if mode in {"all", "name"} and self._matches_name(file_path, query):
                matches.append(f"name: {self._display_path(file_path)}")
                if len(matches) >= max_results:
                    break
            if mode in {"all", "content"}:
                matches.extend(self._content_matches(file_path, query, max_results - len(matches)))
                if len(matches) >= max_results:
                    break

        if not matches:
            fuzzy_matches = self._fuzzy_matches(root, query, mode, max_results)
            if fuzzy_matches:
                header = f"`{query}` に完全一致するローカルファイルは見つかりませんでした。近い候補:"
                return ToolResult(header + "\n" + "\n".join(fuzzy_matches))
            return ToolResult(f"`{query}` に一致するローカルファイルは見つかりませんでした。")

        truncated = " (上限に達したため一部のみ表示)" if len(matches) >= max_results else ""
        header = f"`{query}` のローカル検索結果{truncated}:"
        return ToolResult(header + "\n" + "\n".join(matches))

    def _resolve(self, value: object) -> Path:
        raw = str(value or ".")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workdir / path
        resolved = path.resolve()
        if resolved != self.workdir and self.workdir not in resolved.parents:
            raise ValueError(f"Local search path escapes workdir: {raw}")
        return resolved

    def _iter_files(self, root: Path):
        if root.is_file():
            yield root
            return

        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in DEFAULT_EXCLUDED_DIRS for part in path.relative_to(self.workdir).parts):
                continue
            yield path

    def _matches_name(self, path: Path, query: str) -> bool:
        lowered_query = query.lower()
        name = path.name.lower()
        relative = str(path.relative_to(self.workdir)).replace("\\", "/").lower()
        return lowered_query in name or lowered_query in relative or fnmatch.fnmatch(name, lowered_query)

    def _content_matches(self, path: Path, query: str, limit: int) -> list[str]:
        if limit <= 0 or self._looks_binary(path):
            return []

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
        matches: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not pattern.search(line):
                continue
            snippet = re.sub(r"\s+", " ", line).strip()
            if len(snippet) > 160:
                snippet = snippet[:157].rstrip() + "..."
            matches.append(f"content: {self._display_path(path)}:{line_number}: {snippet}")
            if len(matches) >= limit:
                break
        return matches

    def _fuzzy_matches(self, root: Path, query: str, mode: str, limit: int) -> list[str]:
        scored_matches: list[tuple[float, str]] = []
        for file_path in self._iter_files(root):
            if mode in {"all", "name"}:
                name_match = self._fuzzy_name_match(file_path, query)
                if name_match is not None:
                    scored_matches.append(name_match)
            if mode in {"all", "content"}:
                scored_matches.extend(self._fuzzy_content_matches(file_path, query, limit))

        deduped: dict[str, float] = {}
        for score, line in scored_matches:
            deduped[line] = max(score, deduped.get(line, 0.0))

        ordered = sorted(deduped.items(), key=lambda item: item[1], reverse=True)
        return [line for line, _score in ordered[:limit]]

    def _fuzzy_name_match(self, path: Path, query: str) -> tuple[float, str] | None:
        display_path = self._display_path(path)
        candidates = [path.name, path.stem, display_path]
        candidates.extend(part for part in re.split(r"[/\\_.\-\s]+", display_path) if part)
        best = self._best_fuzzy_candidate(query, candidates)
        if best is None:
            return None
        score, candidate = best
        return score, f"near name: {display_path} (matched `{candidate}`)"

    def _fuzzy_content_matches(self, path: Path, query: str, limit: int) -> list[tuple[float, str]]:
        if limit <= 0 or self._looks_binary(path):
            return []

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        matches: list[tuple[float, str]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            tokens = re.findall(r"[A-Za-z0-9_./\\-]{4,}", line)
            best = self._best_fuzzy_candidate(query, tokens)
            if best is None:
                continue
            score, candidate = best
            snippet = re.sub(r"\s+", " ", line).strip()
            if len(snippet) > 160:
                snippet = snippet[:157].rstrip() + "..."
            matches.append((score, f"near content: {self._display_path(path)}:{line_number}: matched `{candidate}` in {snippet}"))
            if len(matches) >= limit:
                break
        return matches

    def _best_fuzzy_candidate(self, query: str, candidates: list[str]) -> tuple[float, str] | None:
        normalized_query = query.lower()
        best_score = 0.0
        best_candidate = ""
        for candidate in candidates:
            normalized_candidate = candidate.lower()
            if not self._similar_length(normalized_query, normalized_candidate):
                continue
            score = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_score < 0.82:
            return None
        return best_score, best_candidate

    def _similar_length(self, query: str, candidate: str) -> bool:
        if not query or not candidate:
            return False
        allowed_delta = max(2, int(len(query) * 0.35))
        return abs(len(query) - len(candidate)) <= allowed_delta

    def _looks_binary(self, path: Path) -> bool:
        try:
            chunk = path.read_bytes()[:1024]
        except OSError:
            return True
        return b"\0" in chunk

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workdir)).replace("\\", "/")
        except ValueError:
            return str(path)
