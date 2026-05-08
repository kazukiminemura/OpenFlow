from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from .base import JsonDict, Tool, ToolResult


class SandboxTool(Tool):
    name = "sandbox"
    description = "Create and use an isolated workspace for coding experiments and tool behavior checks."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["info", "reset", "write", "read", "list", "run", "delete"],
                "description": "Sandbox action.",
            },
            "path": {
                "type": "string",
                "description": "Relative file or directory path inside the sandbox.",
                "default": ".",
            },
            "content": {
                "type": "string",
                "description": "File content for action=write.",
            },
            "command": {
                "type": "string",
                "description": "Command to run inside the sandbox for action=run.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum runtime before the command is stopped.",
                "default": 30,
                "minimum": 1,
                "maximum": 300,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, workdir: Path, sandbox_dir: str = ".agent_sandbox") -> None:
        self.workdir = workdir.resolve()
        self.root = (self.workdir / sandbox_dir).resolve()

    def run(self, arguments: JsonDict) -> ToolResult:
        action = str(arguments["action"])
        self.root.mkdir(parents=True, exist_ok=True)

        if action == "info":
            return ToolResult(json.dumps({"root": str(self.root)}, ensure_ascii=False, indent=2))
        if action == "reset":
            self._reset()
            return ToolResult(f"Sandbox reset: {self.root}")
        if action == "write":
            path = self._resolve(arguments.get("path") or "sandbox.txt")
            content = str(arguments.get("content", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(f"Wrote {path.relative_to(self.root)} ({len(content)} chars)")
        if action == "read":
            path = self._resolve(arguments.get("path") or ".")
            if not path.is_file():
                return ToolResult(f"Not a file: {path.relative_to(self.root)}", ok=False)
            return ToolResult(path.read_text(encoding="utf-8", errors="replace"))
        if action == "delete":
            path = self._resolve(arguments.get("path") or ".")
            if path == self.root:
                return ToolResult("Refusing to delete the sandbox root. Use action=reset instead.", ok=False)
            if not path.exists():
                return ToolResult(f"Path does not exist: {path.relative_to(self.root)}", ok=False)
            relative_path = path.relative_to(self.root)
            if path.is_dir():
                shutil.rmtree(path)
                return ToolResult(f"Deleted directory {relative_path}")
            path.unlink()
            return ToolResult(f"Deleted file {relative_path}")
        if action == "list":
            path = self._resolve(arguments.get("path") or ".")
            if not path.exists():
                return ToolResult(f"Path does not exist: {path.relative_to(self.root)}", ok=False)
            lines = self._list(path)
            return ToolResult("\n".join(lines) or "<empty>")
        if action == "run":
            command = str(arguments["command"])
            timeout = int(arguments.get("timeout_seconds", 30))
            return self._run_command(command, timeout)

        return ToolResult(f"Unsupported sandbox action: {action}", ok=False)

    def _reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, value: object) -> Path:
        raw = str(value or ".")
        path = (self.root / raw).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Sandbox path escapes root: {raw}")
        return path

    def _list(self, path: Path) -> list[str]:
        if path.is_file():
            return [str(path.relative_to(self.root))]
        return [
            str(item.relative_to(self.root)) + ("/" if item.is_dir() else "")
            for item in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        ]

    def _run_command(self, command: str, timeout: int) -> ToolResult:
        if platform.system() == "Windows":
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        else:
            completed = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        output = [
            f"sandbox: {self.root}",
            f"exit_code: {completed.returncode}",
            "stdout:",
            self._clean_output(completed.stdout),
            "stderr:",
            self._clean_output(completed.stderr),
        ]
        return ToolResult("\n".join(output), ok=completed.returncode == 0)

    def _clean_output(self, value: str | None) -> str:
        if value is None:
            return "<empty>"
        return value.strip() or "<empty>"
