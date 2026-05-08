from __future__ import annotations

import subprocess
import platform
from pathlib import Path

from .base import JsonDict, Tool, ToolResult


class TerminalTool(Tool):
    name = "terminal_run"
    description = "Run a non-interactive shell command in the local terminal and return stdout, stderr, and exit code."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Command to run. Use non-interactive commands only.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum runtime before the command is stopped.",
                "default": 30,
                "minimum": 1,
                "maximum": 300,
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory, relative to the configured agent workdir.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def run(self, arguments: JsonDict) -> ToolResult:
        command = str(arguments["command"])
        timeout = int(arguments.get("timeout_seconds", 30))
        cwd = self._resolve_cwd(arguments.get("cwd"))

        if platform.system() == "Windows":
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        else:
            completed = subprocess.run(
                command,
                cwd=cwd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        output = [
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

    def _resolve_cwd(self, value: object) -> Path:
        if not value:
            return self.workdir
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.workdir / path
        return path.resolve()
