from __future__ import annotations

import ast
import re
import unicodedata
from dataclasses import dataclass


class ArithmeticEvaluator:
    def try_format_result(self, text: str) -> str | None:
        expression = self._extract_expression(text)
        if expression is None:
            return None
        try:
            result = self._eval_expression(expression)
        except ZeroDivisionError:
            return "0で割ることはできません。"
        except ValueError:
            return None
        return f"{expression} = {self._format_result(result)}"

    def _extract_expression(self, text: str) -> str | None:
        normalized = unicodedata.normalize("NFKC", text).lower()
        replacements = {
            "足す": "+",
            "たす": "+",
            "プラス": "+",
            "引く": "-",
            "ひく": "-",
            "マイナス": "-",
            "掛ける": "*",
            "かける": "*",
            "かけて": "*",
            "×": "*",
            "割る": "/",
            "わる": "/",
            "÷": "/",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)
        normalized = re.sub(r"(?<=\d)\s*[xｘ]\s*(?=\d)", "*", normalized)

        candidates = [match.group(0).strip() for match in re.finditer(r"[\d+\-*/().\s]+", normalized)]
        candidates = [candidate for candidate in candidates if self._looks_like_arithmetic(candidate)]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _looks_like_arithmetic(self, expression: str) -> bool:
        expression = expression.strip()
        if len(expression) < 3:
            return False
        if not re.search(r"\d", expression):
            return False
        if not re.search(r"[+\-*/]", expression):
            return False
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", expression):
            return False
        return True

    def _eval_expression(self, expression: str) -> float:
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("invalid arithmetic expression") from exc
        return self._eval_node(parsed.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("unsupported arithmetic expression")

    def _format_result(self, result: float) -> str:
        if result.is_integer():
            return str(int(result))
        return format(result, ".12g")


class CodeGenerationDetector:
    def request_from(self, text: str) -> str | None:
        stripped = text.strip()
        normalized = unicodedata.normalize("NFKC", stripped).lower()
        compact = re.sub(r"\s+", "", normalized)

        explicit_prefixes = (
            "codegen:",
            "codegen：",
            "generate code:",
            "generate code：",
            "コード生成:",
            "コード生成：",
        )
        for prefix in explicit_prefixes:
            if normalized.startswith(prefix):
                request = stripped[len(prefix) :].strip()
                return request or stripped

        normalized_markers = (
            "build a",
            "build ",
            "creat ",
            "create a",
            "create ",
            "generate code",
            "make a",
            "make ",
            "write code",
            "write a function",
            "create a script",
            "sample code",
        )
        compact_markers = (
            "コード生成",
            "コードを生成",
            "コードを書",
            "コード作成",
            "サンプルコード",
            "プログラムを書",
            "アプリを作",
            "ゲームを作",
            "テトリスゲームを作",
            "関数を書",
            "スクリプトを書",
        )
        if any(marker in normalized for marker in normalized_markers):
            return stripped
        if any(marker in compact for marker in compact_markers):
            return stripped
        return None


@dataclass(frozen=True)
class SandboxCommandParser:
    def parse(self, text: str, normalized: str) -> dict[str, object] | None:
        compact = re.sub(r"\s+", "", normalized)
        if "sandbox:" in normalized:
            command = text.split(":", 1)[1].strip()
            if not command:
                return {"action": "info"}
            return {"action": "run", "command": command}
        delete_match = re.match(r"^(?:delete|remove|rm|削除して|削除)\s+(?P<path>.+)$", text.strip(), flags=re.IGNORECASE)
        if delete_match:
            path = self._relative_path(delete_match.group("path"))
            if path is not None:
                return {"action": "delete", "path": path}
        if "サンドボックス" not in compact and "sandbox" not in normalized:
            return None
        sandbox_path = self._path_from_text(text)
        if any(word in compact for word in ["削除", "delete", "remove", "rm"]):
            return {"action": "delete", "path": sandbox_path or "."}
        if any(word in compact for word in ["初期化", "リセット", "reset"]):
            return {"action": "reset"}
        if any(word in compact for word in ["一覧", "list"]):
            return {"action": "list", "path": sandbox_path or "."}
        if any(word in compact for word in ["情報", "場所", "info"]):
            return {"action": "info"}
        return {"action": "info"}

    def _relative_path(self, raw_path: str) -> str | None:
        cleaned = raw_path.strip().strip('"\'')
        cleaned = cleaned.replace("\\", "/")
        if not cleaned:
            return None
        marker = ".agent_sandbox/"
        if cleaned == ".agent_sandbox":
            return "."
        if marker in cleaned:
            return cleaned.split(marker, 1)[1].strip("/") or "."
        if cleaned.startswith("agent_sandbox/"):
            return cleaned.removeprefix("agent_sandbox/").strip("/") or "."
        if cleaned.startswith("/"):
            return None
        return cleaned

    def _path_from_text(self, text: str) -> str | None:
        match = re.search(r"(?:\.agent_sandbox[/\\])?(?P<path>generated[/\\][^\s]+|[\w.-]+(?:[/\\][\w.-]+)+)", text)
        if not match:
            return None
        return self._relative_path(match.group(0))
