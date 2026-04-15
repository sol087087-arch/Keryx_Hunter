# keryx/tools/ast_analyzer.py
# ASTAnalyzerTool — real Python static analysis via the built-in `ast` module.
#
# Replaces the mock codeql_query with concrete findings:
#   - subprocess calls with shell=True
#   - user-controlled args without "--" separator (git option injection)
#   - missing input validation before subprocess
#   - path traversal: user input flows into open() / Path() without sanitization
#   - hardcoded secrets (API keys, passwords in assignments)
#
# Registered as action name "codeql_query" so existing agent prompts work unchanged.

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .Toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.ast_analyzer")


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    rule:        str
    severity:    str          # HIGH / MEDIUM / LOW
    line:        int
    col:         int
    message:     str
    snippet:     str = ""

    def __str__(self) -> str:
        loc = f"line {self.line}"
        return f"[{self.severity}] {self.rule} @ {loc}: {self.message}"


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class _VulnVisitor(ast.NodeVisitor):
    """Single-pass AST visitor collecting security findings."""

    def __init__(self, source_lines: list[str]) -> None:
        self.findings: list[Finding] = []
        self._lines = source_lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", None)
        if line and 1 <= line <= len(self._lines):
            return self._lines[line - 1].rstrip()
        return ""

    def _add(self, rule: str, severity: str, node: ast.AST, message: str) -> None:
        self.findings.append(Finding(
            rule=rule,
            severity=severity,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            message=message,
            snippet=self._snippet(node),
        ))

    @staticmethod
    def _is_name(node: ast.expr, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    @staticmethod
    def _get_keyword_value(call: ast.Call, name: str) -> ast.expr | None:
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        return None

    @staticmethod
    def _is_true(node: ast.expr | None) -> bool:
        if node is None:
            return False
        return isinstance(node, ast.Constant) and node.value is True

    # ------------------------------------------------------------------
    # R1 — subprocess with shell=True
    # ------------------------------------------------------------------

    _SUBPROCESS_FUNCS = {
        "run", "call", "check_call", "check_output", "Popen",
        "create_subprocess_shell",
    }

    def _check_subprocess(self, node: ast.Call) -> None:
        func = node.func
        func_name: str | None = None

        if isinstance(func, ast.Attribute) and func.attr in self._SUBPROCESS_FUNCS:
            func_name = func.attr
        elif isinstance(func, ast.Name) and func.id in self._SUBPROCESS_FUNCS:
            func_name = func.id

        if func_name is None:
            return

        shell_val = self._get_keyword_value(node, "shell")
        if self._is_true(shell_val):
            self._add(
                "SUBPROCESS_SHELL_TRUE", "HIGH", node,
                f"{func_name}(..., shell=True) — command string is interpreted by the shell; "
                "any unsanitized input enables command injection.",
            )

    # ------------------------------------------------------------------
    # R2 — subprocess command list without "--" separator
    # ------------------------------------------------------------------

    def _check_missing_dashdash(self, node: ast.Call) -> None:
        """
        Detects patterns like:
            cmd.extend(["--author", author])   # author is a variable → git option injection
        if the list doesn't include a bare "--" sentinel.
        """
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ("extend", "append")):
            return
        if not node.args:
            return
        arg = node.args[0]

        # extend(["--flag", variable])  — second element is a Name (not a Constant)
        if isinstance(arg, ast.List) and len(arg.elts) == 2:
            flag, value = arg.elts
            if (
                isinstance(flag, ast.Constant)
                and isinstance(flag.value, str)
                and flag.value.startswith("--")
                and isinstance(value, ast.Name)
            ):
                self._add(
                    "GIT_OPTION_INJECTION", "HIGH", node,
                    f'cmd.extend(["{flag.value}", {value.id}]) — user-controlled value '
                    f'"{value.id}" is passed as the argument to "{flag.value}" without '
                    f'"--" separator; a value like "--upload-pack=x" becomes a git flag.',
                )

        # append(variable)  where variable is not a constant
        if isinstance(arg, ast.Name):
            self._add(
                "UNSANITIZED_SUBPROCESS_ARG", "MEDIUM", node,
                f'cmd.append({arg.id}) — variable "{arg.id}" appended to subprocess '
                f"command list without validation; may allow path traversal or option injection.",
            )

    # ------------------------------------------------------------------
    # R3 — open() / Path.read_text() with user-controlled path
    # ------------------------------------------------------------------

    _OPEN_FUNCS = {"open"}

    def _check_open(self, node: ast.Call) -> None:
        func = node.func
        is_open = (
            (isinstance(func, ast.Name) and func.id in self._OPEN_FUNCS)
            or (isinstance(func, ast.Attribute) and func.attr in ("read_text", "read_bytes", "open"))
        )
        if not is_open:
            return
        if not node.args:
            return
        path_arg = node.args[0]
        # Flag if the path comes from a function parameter (Name), not a literal
        if isinstance(path_arg, ast.Name):
            self._add(
                "OPEN_USER_PATH", "MEDIUM", node,
                f'open({path_arg.id}) — path comes from variable "{path_arg.id}"; '
                f"if caller-controlled, path traversal is possible.",
            )

    # ------------------------------------------------------------------
    # R4 — hardcoded secrets
    # ------------------------------------------------------------------

    _SECRET_KEYWORDS = ("password", "passwd", "secret", "api_key", "apikey", "token", "private_key")

    def _check_hardcoded_secret(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name_lower = target.id.lower()
            if not any(kw in name_lower for kw in self._SECRET_KEYWORDS):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                val = node.value.value
                if len(val) > 4:   # skip empty/placeholder
                    self._add(
                        "HARDCODED_SECRET", "HIGH", node,
                        f'"{target.id}" assigned a hardcoded string value — '
                        f"credentials should come from environment variables.",
                    )

    # ------------------------------------------------------------------
    # R5 — asyncio.create_subprocess_exec without "--" in cmd
    # ------------------------------------------------------------------

    def _check_create_subprocess_exec(self, node: ast.Call) -> None:
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "create_subprocess_exec"):
            return
        # Check if any arg is the result of a list that has user values appended
        # Heuristic: if called with *cmd where cmd is a variable, flag it
        for arg in node.args:
            if isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
                self._add(
                    "SUBPROCESS_EXEC_STARRED", "MEDIUM", node,
                    f"create_subprocess_exec(*{arg.value.id}) — command list unpacked from "
                    f'variable "{arg.value.id}"; if user input flows into this list without '
                    f'"--" sentinel, option injection is possible.',
                )

    # ------------------------------------------------------------------
    # R6 — json.loads() / json.load() on a non-literal (LLM output sink)
    # ------------------------------------------------------------------

    def _check_llm_output_sink(self, node: ast.Call) -> None:
        """
        Flags json.loads(x) / json.load(x) where x is not a string literal.
        These are deserialization sinks that may process LLM output or other
        external input without schema validation.
        """
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in ("loads", "load")
            and isinstance(func.value, ast.Name)
            and func.value.id == "json"
        ):
            return
        if not node.args:
            return
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant):
            return  # literal string — safe
        arg_repr = getattr(first_arg, "id", None) or ast.unparse(first_arg)
        self._add(
            "LLM_OUTPUT_SINK", "MEDIUM", node,
            f"json.{func.attr}({arg_repr}) — deserializes a non-literal value; "
            "if the source is LLM output or external input, validate schema before parsing.",
        )

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        self._check_subprocess(node)
        self._check_missing_dashdash(node)
        self._check_open(node)
        self._check_create_subprocess_exec(node)
        self._check_llm_output_sink(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_hardcoded_secret(node)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class ASTAnalyzerTool(BaseTool):
    """
    Python AST-based static analyzer.
    Registered as 'codeql_query' so agent prompts need no changes.
    """

    name = "codeql_query"
    description = (
        "Run static analysis on a Python file to find real security issues: "
        "subprocess injection, missing argument sanitization, path traversal, "
        "hardcoded secrets. Returns concrete findings with line numbers."
    )

    def __init__(self, allowed_root: str | None = None) -> None:
        super().__init__()
        self.allowed_root = Path(allowed_root).resolve() if allowed_root else None

    async def execute(
        self,
        action_input: dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        file_path = (
            action_input.get("file_path")
            or action_input.get("path")
            or action_input.get("target")
        )
        if not file_path:
            return ToolResult(
                success=False,
                output="Missing 'path' in action_input",
                error="path_missing",
            )

        path = Path(file_path).resolve()

        if self.allowed_root:
            try:
                path.relative_to(self.allowed_root)
            except ValueError:
                return ToolResult(
                    success=False,
                    output="Access denied: path outside allowed root",
                    error="path_traversal_blocked",
                )

        if not path.exists() or not path.is_file():
            return ToolResult(
                success=False,
                output=f"File not found: {file_path}",
                error="file_not_found",
            )

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(success=False, output=f"Read error: {exc}", error="read_error")

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return ToolResult(
                success=False,
                output=f"Syntax error — cannot parse: {exc}",
                error="syntax_error",
            )

        source_lines = source.splitlines()
        visitor = _VulnVisitor(source_lines)
        visitor.visit(tree)
        findings = visitor.findings

        if not findings:
            output = f"[AST] No findings in {path.name} ({len(source_lines)} lines analyzed)."
        else:
            lines = [
                f"[AST] {len(findings)} finding(s) in {path.name} "
                f"({len(source_lines)} lines analyzed):\n"
            ]
            for i, f in enumerate(findings, 1):
                lines.append(f"  [{i}] {f}")
                if f.snippet:
                    lines.append(f"       code: {f.snippet}")
            output = "\n".join(lines)

        return ToolResult(
            success=True,
            output=output,
            data={
                "file":          str(path),
                "total_lines":   len(source_lines),
                "findings_count": len(findings),
                "findings": [
                    {
                        "rule":     f.rule,
                        "severity": f.severity,
                        "line":     f.line,
                        "message":  f.message,
                        "snippet":  f.snippet,
                    }
                    for f in findings
                ],
            },
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_ast_analyzer_tool(allowed_root: str | None = None) -> ASTAnalyzerTool:
    return ASTAnalyzerTool(allowed_root=allowed_root)
