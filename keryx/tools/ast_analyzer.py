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
    source_type: str = "unknown"
    # source_type values:
    #   "param"      — vulnerable arg is a direct function parameter (highest risk:
    #                  caller controls it with no intermediate transformation)
    #   "local"      — a local variable (possibly already validated upstream)
    #   "expression" — a complex expression (f-string, BinOp, subscript, call);
    #                  tainted if any sub-Name is a param
    #   "constant"   — hardcoded literal (should not reach here; used as guard)
    #   "unknown"    — could not determine (e.g. no enclosing function found)

    def __str__(self) -> str:
        loc = f"line {self.line}"
        return f"[{self.severity}] {self.rule} @ {loc}: {self.message}"


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

def _is_re_escape_call(node: ast.expr) -> bool:
    """Return True if *node* is a call to re.escape(...) or re.escape(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # re.escape(x)
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "escape"
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
    ):
        return True
    return False


def _fstring_all_caps_names(node: ast.expr) -> bool:
    """Return True if *node* is an f-string whose only non-literal parts are
    ALL_CAPS Name references (module-level constants by Python convention).

    e.g. f"^{PREFIX}{EVENT}.*" — both PREFIX and EVENT are constants → safe.
    """
    if not isinstance(node, ast.JoinedStr):
        return False
    for value in node.values:
        if isinstance(value, ast.Constant):
            continue  # literal text part
        if isinstance(value, ast.FormattedValue):
            inner = value.value
            if isinstance(inner, ast.Name) and inner.id.isupper():
                continue  # ALL_CAPS name — treat as constant
            return False
        return False
    return True


class _VulnVisitor(ast.NodeVisitor):
    """Single-pass AST visitor collecting security findings."""

    def __init__(self, source_lines: list[str], file_path: str = "") -> None:
        self.findings: list[Finding] = []
        self._lines = source_lines
        self._file_path = file_path.lower()
        # Parameter tracking — updated by visit_FunctionDef/visit_AsyncFunctionDef.
        self._param_names: set[str] = set()
        self._param_stack: list[set[str]] = []
        # Import tracking for SSTI detection.
        # _name_to_module:  local name → originating module
        #   "from jinja2 import Template" → {"Template": "jinja2"}
        #   "from jinja2 import Template as T" → {"T": "jinja2"}
        # _ssti_aliases:    local names that ARE known SSTI template classes (HIGH)
        #   "from jinja2 import Template as T" → {"T"}
        # _module_aliases:  local module alias → canonical module name
        #   "import jinja2 as j" → {"j": "jinja2"}
        self._name_to_module: dict[str, str] = {}
        self._ssti_aliases:   set[str]       = set()
        self._module_aliases: dict[str, str] = {}
        # Names known to hold re.escape()-d values — populated by visit_Assign.
        self._re_escaped_names: set[str] = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", None)
        if line and 1 <= line <= len(self._lines):
            return self._lines[line - 1].rstrip()
        return ""

    def _source_type(self, arg: ast.expr) -> str:
        """Classify where a vulnerable argument originates.

        Returns one of:
          "param"      — directly from a function parameter (unvalidated by default)
          "local"      — a local variable (may have been validated upstream)
          "expression" — a compound expression; "param" if any sub-name is a param
          "constant"   — a literal (should have been caught by the constant guard)
          "unknown"    — not inside a function or type unrecognised
        """
        if isinstance(arg, ast.Constant):
            return "constant"
        # Not inside any function — caller is at module scope.
        if not self._param_stack and not self._param_names:
            return "unknown"
        if isinstance(arg, ast.Name):
            return "param" if arg.id in self._param_names else "local"
        # Compound: f-string, BinOp, Subscript, Call, etc.
        # Tainted if any direct-Name sub-node is a function parameter.
        if any(
            isinstance(n, ast.Name) and n.id in self._param_names
            for n in ast.walk(arg)
            if isinstance(n, ast.Name)
        ):
            return "param"
        return "expression"

    def _add(
        self,
        rule:     str,
        severity: str,
        node:     ast.AST,
        message:  str,
        *,
        vuln_arg: ast.expr | None = None,
    ) -> None:
        src = self._source_type(vuln_arg) if vuln_arg is not None else "unknown"
        self.findings.append(Finding(
            rule=rule,
            severity=severity,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            message=message,
            snippet=self._snippet(node),
            source_type=src,
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

    # Receiver names that suggest a subprocess command list
    _CMD_LIST_NAMES = frozenset({
        "cmd", "command", "args", "argv",
        "proc_args", "cmd_args", "command_args", "shell_cmd",
    })

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
            cmd_arg = node.args[0] if node.args else None
            self._add(
                "SUBPROCESS_SHELL_TRUE", "HIGH", node,
                f"{func_name}(..., shell=True) — command string is interpreted by the shell; "
                "any unsanitized input enables command injection.",
                vuln_arg=cmd_arg,
            )

    # ------------------------------------------------------------------
    # R2 — subprocess command list without "--" separator
    # ------------------------------------------------------------------

    @staticmethod
    def _fstring_variable_flag(node: ast.expr) -> str | None:
        """Return the variable name if *node* is f"--{variable}=..." (flag NAME is a variable).

        Safe  — flag name is a constant:  f"--author={x}"
                 AST: JoinedStr([Constant("--author="), FormattedValue(Name("x"))])
                 → first Constant does NOT equal exactly "--" → returns None.

        Unsafe — flag name is a variable: f"--{flag}={x}"
                 AST: JoinedStr([Constant("--"), FormattedValue(Name("flag")), ...])
                 → first Constant IS exactly "--" → returns "flag".
        """
        if not isinstance(node, ast.JoinedStr) or len(node.values) < 2:
            return None
        first = node.values[0]
        if not (isinstance(first, ast.Constant) and first.value == "--"):
            return None
        second = node.values[1]
        if isinstance(second, ast.FormattedValue) and isinstance(second.value, ast.Name):
            return second.value.id
        return None

    def _check_missing_dashdash(self, node: ast.Call) -> None:
        """
        R2a: cmd.extend(["--flag", variable])  → GIT_OPTION_INJECTION
             flag value passed as separate token; may be parsed as a new git flag.

        R2b: cmd.append(f"--{flag_var}=...")   → GIT_FLAG_NAME_INJECTION
             flag NAME is user-controlled; attacker picks which git option to set.
             (Distinct from R2a: here the flag itself, not just its value, is variable.)

        NOT flagged (safe patterns):
             ["--", path]           — bare "--" is the end-of-options separator
             f"--author={author}"   — flag name is a hardcoded constant
        """
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ("extend", "append")):
            return
        if not node.args:
            return
        arg = node.args[0]

        # R2a: extend(["--flag", variable])
        # Note: ["--", variable] is the SAFE end-of-options separator — skip it.
        if isinstance(arg, ast.List) and len(arg.elts) == 2:
            flag, value = arg.elts
            if (
                isinstance(flag, ast.Constant)
                and isinstance(flag.value, str)
                and flag.value.startswith("--")
                and flag.value != "--"   # bare "--" is the safe separator, not a flag
                and isinstance(value, ast.Name)
            ):
                self._add(
                    "GIT_OPTION_INJECTION", "HIGH", node,
                    f'cmd.extend(["{flag.value}", {value.id}]) — user-controlled value '
                    f'"{value.id}" is passed as the argument to "{flag.value}" without '
                    f'"--" separator; a value like "--upload-pack=x" becomes a git flag.',
                    vuln_arg=value,
                )

        # R2b: append(f"--{flag_var}=...") or extend([f"--{flag_var}=..."])
        # Flag NAME is a variable — attacker controls which git option is set.
        recv = func.value
        recv_name = recv.id if isinstance(recv, ast.Name) else "cmd"

        if isinstance(arg, ast.List):
            # extend([f"--{flag}=...", ...])
            for elt in arg.elts:
                flag_var = self._fstring_variable_flag(elt)
                if flag_var:
                    self._add(
                        "GIT_FLAG_NAME_INJECTION", "HIGH", node,
                        f'{recv_name}.extend([f"--{{{flag_var}}}=..."]) — '
                        f'the flag name "{flag_var}" is user-controlled; '
                        f"attacker can inject any git option.",
                        vuln_arg=elt,
                    )
        else:
            # append(f"--{flag}=...")
            flag_var = self._fstring_variable_flag(arg)
            if flag_var:
                self._add(
                    "GIT_FLAG_NAME_INJECTION", "HIGH", node,
                    f'{recv_name}.append(f"--{{{flag_var}}}=...") — '
                    f'the flag name "{flag_var}" is user-controlled; '
                    f"attacker can inject any git option.",
                    vuln_arg=arg,
                )

        # R2a-append: append(variable) where receiver looks like a subprocess cmd list
        if isinstance(arg, ast.Name):
            if not (isinstance(recv, ast.Name) and recv.id in self._CMD_LIST_NAMES):
                return
            self._add(
                "UNSANITIZED_SUBPROCESS_ARG", "MEDIUM", node,
                f'{recv.id}.append({arg.id}) — variable "{arg.id}" appended to '
                f'subprocess command list "{recv.id}" without validation; '
                f"may allow path traversal or option injection.",
                vuln_arg=arg,
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
                vuln_arg=path_arg,
            )

    # ------------------------------------------------------------------
    # R4 — hardcoded secrets
    # ------------------------------------------------------------------

    _SECRET_KEYWORDS = ("password", "passwd", "secret", "api_key", "apikey", "token", "private_key")

    # Directory names (path components) that indicate test/fixture code.
    _TEST_DIR_NAMES = frozenset({
        "tests", "test", "mocks", "mock", "fixtures", "fixture",
        "examples", "example", "samples", "sample", "demos", "demo",
        "stubs", "stub", "fakes", "fake",
    })

    # Value prefixes/substrings that look like test credentials, not real secrets.
    _PLACEHOLDER_PATTERNS = (
        "dummy", "placeholder", "changeme", "example", "sample",
        "your_", "insert_", "replace_", "todo", "fixme",
        "xxxxxxxxx", "aaaaaa", "123456",
    )

    def _is_test_context(self, val: str) -> bool:
        """Return True if the file path looks like test/fixture code.

        Checks:
        - Any path component (directory name) matches a known test dir name
        - File basename starts with "test_" or ends with "_test.py"
        """
        import os
        parts = self._file_path.replace("\\", "/").split("/")
        basename = parts[-1] if parts else ""
        # File name heuristics
        if basename.startswith("test_") or basename.endswith("_test.py"):
            return True
        if basename == "conftest.py":
            return True
        # Directory component heuristics
        return any(p in self._TEST_DIR_NAMES for p in parts[:-1])

    def _is_placeholder_value(self, val: str) -> bool:
        """Return True if the value looks like a test/placeholder credential."""
        v = val.lower()
        return any(p in v for p in self._PLACEHOLDER_PATTERNS)

    def _check_hardcoded_secret(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name_lower = target.id.lower()
            if not any(kw in name_lower for kw in self._SECRET_KEYWORDS):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                val = node.value.value
                if len(val) <= 4:
                    continue  # empty / trivially short
                if self._is_test_context(val):
                    continue  # test/fixture file — downgrade to noise
                if self._is_placeholder_value(val):
                    continue  # obvious placeholder value
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
                    vuln_arg=arg.value,
                )

    # ------------------------------------------------------------------
    # R6 — json.loads() / json.load() on externally-sourced input
    # ------------------------------------------------------------------

    def _check_unsafe_deserialization(self, node: ast.Call) -> None:
        """Flags json.loads/json.load where the argument is clearly external input.

        Fires when the first argument is:
          - A function parameter (caller-controlled)
          - An attribute access (request.body, response.text, obj.data, …)
          - A call expression (request.read(), recv(), input(), …)

        Skips local variables that are NOT function parameters — those are almost
        always assigned from internal sources and produce excessive noise.
        Skips string / bytes literals (always safe).
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
            return  # literal — safe
        # Plain local variable that is NOT a known function parameter → skip
        if isinstance(first_arg, ast.Name) and first_arg.id not in self._param_names:
            return
        arg_repr = getattr(first_arg, "id", None) or ast.unparse(first_arg)
        self._add(
            "UNSAFE_DESERIALIZATION", "MEDIUM", node,
            f"json.{func.attr}({arg_repr}) — deserializes external input without "
            "schema validation; an attacker can supply unexpected types or deeply "
            "nested structures. Validate with a strict schema (Pydantic / jsonschema) "
            "before deserializing.",
            vuln_arg=first_arg,
        )

    # ------------------------------------------------------------------
    # R7 — eval() / exec() with non-constant argument
    # ------------------------------------------------------------------

    _EVAL_EXEC_FUNCS = {"eval", "exec"}

    def _check_eval_exec(self, node: ast.Call) -> None:
        func = node.func
        if not (isinstance(func, ast.Name) and func.id in self._EVAL_EXEC_FUNCS):
            return
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant):
            return  # hardcoded string — safe
        arg_repr = getattr(arg, "id", None) or ast.unparse(arg)
        self._add(
            "UNSAFE_EVAL_EXEC", "HIGH", node,
            f"{func.id}({arg_repr}) — executes a non-literal expression; "
            "if caller-controlled, this is arbitrary code execution.",
            vuln_arg=arg,
        )

    # ------------------------------------------------------------------
    # R8 — pickle.loads() / pickle.load() with non-constant argument
    # ------------------------------------------------------------------

    def _check_unsafe_pickle(self, node: ast.Call) -> None:
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in ("loads", "load")
            and isinstance(func.value, ast.Name)
            and func.value.id == "pickle"
        ):
            return
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant):
            return
        arg_repr = getattr(arg, "id", None) or ast.unparse(arg)
        self._add(
            "UNSAFE_PICKLE", "HIGH", node,
            f"pickle.{func.attr}({arg_repr}) — deserializes arbitrary Python objects; "
            "pickle executes code during deserialization, making any untrusted "
            "source exploitable regardless of surrounding try/except.",
            vuln_arg=arg,
        )

    # ------------------------------------------------------------------
    # R9 — yaml.load() without SafeLoader / yaml.full_load()
    # ------------------------------------------------------------------

    @staticmethod
    def _has_safe_loader(call: ast.Call) -> bool:
        """Return True if the call has Loader=yaml.SafeLoader keyword."""
        for kw in call.keywords:
            if kw.arg != "Loader":
                continue
            v = kw.value
            # yaml.SafeLoader  (Attribute)
            if isinstance(v, ast.Attribute) and v.attr == "SafeLoader":
                return True
            # SafeLoader  (bare Name import)
            if isinstance(v, ast.Name) and v.id == "SafeLoader":
                return True
        return False

    def _check_unsafe_yaml(self, node: ast.Call) -> None:
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "yaml"
        ):
            return

        if func.attr == "safe_load":
            return  # safe by design

        if func.attr == "load":
            if self._has_safe_loader(node):
                return  # explicitly safe
            arg_repr = (
                getattr(node.args[0], "id", None) or ast.unparse(node.args[0])
            ) if node.args else "?"
            yaml_arg = node.args[0] if node.args else None
            self._add(
                "UNSAFE_YAML_LOAD", "HIGH", node,
                f"yaml.load({arg_repr}) without Loader=yaml.SafeLoader — "
                "can deserialize and execute arbitrary Python objects via YAML tags. "
                "Use yaml.safe_load() or pass Loader=yaml.SafeLoader explicitly.",
                vuln_arg=yaml_arg,
            )

        elif func.attr == "full_load":
            arg_repr = (
                getattr(node.args[0], "id", None) or ast.unparse(node.args[0])
            ) if node.args else "?"
            yaml_arg = node.args[0] if node.args else None
            self._add(
                "UNSAFE_YAML_LOAD", "MEDIUM", node,
                f"yaml.full_load({arg_repr}) — resolves Python-specific YAML tags; "
                "safer than yaml.load() but not equivalent to yaml.safe_load(). "
                "Prefer yaml.safe_load() for untrusted input.",
                vuln_arg=yaml_arg,
            )

    # ------------------------------------------------------------------
    # R10 — os.system() / os.popen() with non-constant argument
    # ------------------------------------------------------------------

    _OS_SHELL_FUNCS = {"system", "popen"}

    def _check_os_shell(self, node: ast.Call) -> None:
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in self._OS_SHELL_FUNCS
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            return
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant):
            return  # hardcoded command — safe
        arg_repr = getattr(arg, "id", None) or ast.unparse(arg)
        self._add(
            "OS_SHELL_INJECTION", "HIGH", node,
            f"os.{func.attr}({arg_repr}) — passes a non-literal string to the shell; "
            "any unsanitized input enables command injection. "
            "Use subprocess.run([...]) with a list instead.",
            vuln_arg=arg,
        )

    # ------------------------------------------------------------------
    # R11 — requests.<method>() with a non-constant URL (SSRF)
    # ------------------------------------------------------------------

    _REQUESTS_METHODS = frozenset({
        "get", "post", "put", "delete", "patch", "head", "options", "request",
    })

    def _check_ssrf(self, node: ast.Call) -> None:
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in self._REQUESTS_METHODS
            and isinstance(func.value, ast.Name)
            and func.value.id == "requests"
        ):
            return
        if not node.args:
            return
        # requests.request("GET", url) — URL is the second positional arg
        url_idx = 1 if func.attr == "request" else 0
        if len(node.args) <= url_idx:
            return
        url_arg = node.args[url_idx]
        if isinstance(url_arg, ast.Constant):
            return  # hardcoded URL — not SSRF
        arg_repr = getattr(url_arg, "id", None) or ast.unparse(url_arg)
        self._add(
            "SSRF", "HIGH", node,
            f"requests.{func.attr}({arg_repr}) — URL argument is not a constant; "
            "if caller-controlled this enables Server-Side Request Forgery (SSRF). "
            "Validate URL against an allowlist or use a URL parser with schema enforcement.",
            vuln_arg=url_arg,
        )

    # ------------------------------------------------------------------
    # R12 — Template(non_const) / env.from_string(non_const)  (SSTI)
    # ------------------------------------------------------------------

    # Known template engines that support code-execution expressions (e.g. {{7*7}})
    _SSTI_MODULES = {"jinja2", "mako", "chameleon", "django.template"}
    # from_string() callers that are genuine template environments.
    # Matched against the bare Name id OR any substring of ast.unparse(receiver).
    _SSTI_ENV_NAMES = {"env", "environment", "jinja_env", "jinja2_env", "template_env"}
    # Attribute-access patterns: if the unparse of the receiver ends with one of
    # these suffixes it is treated as a template environment (e.g. self.env, self._env).
    _SSTI_ENV_SUFFIXES = (".env", "._env", ".environment", ".jinja_env", ".jinja2_env")

    def _check_template_injection(self, node: ast.Call) -> None:
        """Detect Server-Side Template Injection (SSTI) sources.

        Two patterns:
        A. Template(non_constant) or <module>.Template(non_constant):
           HIGH only for known dangerous template engines (jinja2, mako, chameleon,
           django.template).  Bare Template(...) with unknown import → MEDIUM.
        B. <env>.from_string(non_constant):
           HIGH only when the receiver is a known Jinja2 Environment or a variable
           whose name suggests a template env.  All other .from_string() callers
           (e.g. LlamaGrammar.from_string) are skipped — not a template engine.

        Rendering with user *data* (template.render(name=user_input)) is NOT
        flagged here because the template string itself is fixed and safe.
        """
        func = node.func
        if not node.args:
            return
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant):
            return  # hardcoded template string — safe

        arg_repr = getattr(first_arg, "id", None) or ast.unparse(first_arg)

        # Pattern A — bare name: Template(non_const) or any aliased SSTI class
        if isinstance(func, ast.Name) and (
            func.id == "Template" or func.id in self._ssti_aliases
        ):
            origin = self._name_to_module.get(func.id, "")
            is_dangerous = (
                func.id in self._ssti_aliases
                or any(engine in origin for engine in self._SSTI_MODULES)
            )
            if is_dangerous:
                src_note = f" (imported from {origin!r})" if origin else ""
                self._add(
                    "TEMPLATE_INJECTION", "HIGH", node,
                    f"{func.id}({arg_repr}){src_note} — template string is not a constant; "
                    "caller-controlled input enables SSTI (arbitrary code execution) via "
                    "expressions such as {{{{7*7}}}} or {{{{''.__class__.__mro__}}}}. "
                    "Never pass user input as a template string.",
                    vuln_arg=first_arg,
                )
            else:
                # Unknown or safe import (e.g. string.Template) → MEDIUM
                self._add(
                    "TEMPLATE_INJECTION", "MEDIUM", node,
                    f"Template({arg_repr}) — template string is not a constant; "
                    "if this is jinja2.Template or mako.template.Template and the string "
                    "is caller-controlled, SSTI (arbitrary code execution) is possible. "
                    "Verify the import and pass user input only as render() context.",
                    vuln_arg=first_arg,
                )
            return

        # Pattern A — qualified: <module>.Template(non_const)
        if isinstance(func, ast.Attribute) and func.attr == "Template":
            raw_mod = ast.unparse(func.value)
            # Resolve module alias: "j.Template" where j = jinja2 → jinja2.Template
            mod_name = getattr(func.value, "id", None) or ""
            canonical = self._module_aliases.get(mod_name, raw_mod)
            if any(engine in canonical for engine in self._SSTI_MODULES):
                self._add(
                    "TEMPLATE_INJECTION", "HIGH", node,
                    f"{raw_mod}.Template({arg_repr}) — template string is not a constant; "
                    "caller-controlled input enables SSTI (arbitrary code execution) via "
                    "expressions such as {{{{7*7}}}} or {{{{''.__class__.__mro__}}}}. "
                    "Never pass user input as a template string.",
                    vuln_arg=first_arg,
                )
            else:
                self._add(
                    "TEMPLATE_INJECTION", "MEDIUM", node,
                    f"{raw_mod}.Template({arg_repr}) — template string is not a constant; "
                    "if this module supports code-execution expressions, SSTI is possible.",
                    vuln_arg=first_arg,
                )
            return

        # Pattern B: <env>.from_string(non_const) — only known template envs
        if isinstance(func, ast.Attribute) and func.attr == "from_string":
            env_repr = ast.unparse(func.value)
            env_name = getattr(func.value, "id", "") or ""
            is_known_env = (
                any(engine in env_repr for engine in self._SSTI_MODULES)
                or env_name.lower() in self._SSTI_ENV_NAMES
                or any(env_repr.endswith(sfx) for sfx in self._SSTI_ENV_SUFFIXES)
            )
            if is_known_env:
                self._add(
                    "TEMPLATE_INJECTION", "HIGH", node,
                    f"{env_repr}.from_string({arg_repr}) — template string is not a constant; "
                    "caller-controlled input enables SSTI. "
                    "Never pass user input as a template string; use render() context only.",
                    vuln_arg=first_arg,
                )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # R13 — re.<func>(non_constant_pattern)  (ReDoS)
    # ------------------------------------------------------------------

    _RE_FUNCS = frozenset({
        "compile", "match", "search", "fullmatch",
        "findall", "finditer", "sub", "subn", "split",
    })

    def _check_regex_dos(self, node: ast.Call) -> None:
        """Detect ReDoS: re.* called with a user-controlled pattern (first arg).

        The *pattern* is always the first positional argument to every re.*
        function.  If it is not a string literal, an attacker can supply a
        catastrophically backtracking pattern (e.g. ``(a+)+$``) that causes
        exponential CPU usage proportional to input length.

        Note: this flags the *pattern* source, not the *string* being matched.
        ``re.match(r"fixed", user_string)`` is safe and is NOT flagged.
        """
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in self._RE_FUNCS
            and isinstance(func.value, ast.Name)
            and func.value.id == "re"
        ):
            return
        if not node.args:
            return
        pattern_arg = node.args[0]
        if isinstance(pattern_arg, ast.Constant):
            return  # hardcoded pattern — safe
        if _is_re_escape_call(pattern_arg):
            return  # inline re.escape(x) — sanitized
        if isinstance(pattern_arg, ast.Name) and pattern_arg.id in self._re_escaped_names:
            return  # pre-assigned from re.escape() — sanitized
        if _fstring_all_caps_names(pattern_arg):
            return  # f-string built from module-level constants (ALL_CAPS) — safe
        arg_repr = getattr(pattern_arg, "id", None) or ast.unparse(pattern_arg)
        self._add(
            "REGEX_DOS", "HIGH", node,
            f"re.{func.attr}({arg_repr}) — the regex pattern is not a constant; "
            "if caller-controlled an attacker can supply a catastrophically "
            "backtracking expression (ReDoS) causing exponential CPU usage. "
            "Compile patterns from constants only; validate or reject untrusted "
            "input before using it as a regular expression.",
            vuln_arg=pattern_arg,
        )

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track parameter names so _source_type() can classify vulnerable args."""
        all_args = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        )
        if node.args.vararg:
            all_args = all_args + [node.args.vararg]
        if node.args.kwarg:
            all_args = all_args + [node.args.kwarg]
        params = {a.arg for a in all_args}
        self._param_stack.append(self._param_names)
        self._param_names = params
        self.generic_visit(node)
        self._param_names = self._param_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:
        """Track 'import X as Y' module aliases for SSTI detection."""
        for alias in node.names:
            if alias.asname:
                self._module_aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track 'from X import Y [as Z]' for SSTI detection.

        Populates _name_to_module and _ssti_aliases so that aliased
        imports like 'from jinja2 import Template as T' are caught.
        """
        if node.module:
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                self._name_to_module[local_name] = node.module
                # Mark as confirmed SSTI class if it's Template from a dangerous engine
                if alias.name == "Template" and any(
                    engine in node.module for engine in self._SSTI_MODULES
                ):
                    self._ssti_aliases.add(local_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_subprocess(node)
        self._check_missing_dashdash(node)
        self._check_open(node)
        self._check_create_subprocess_exec(node)
        self._check_unsafe_deserialization(node)
        self._check_eval_exec(node)
        self._check_unsafe_pickle(node)
        self._check_unsafe_yaml(node)
        self._check_os_shell(node)
        self._check_ssrf(node)
        self._check_template_injection(node)
        self._check_regex_dos(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_hardcoded_secret(node)
        # Track `name = re.escape(x)` so _check_regex_dos can skip it.
        if _is_re_escape_call(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._re_escaped_names.add(target.id)
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
        visitor = _VulnVisitor(source_lines, file_path=str(path))
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
                        "rule":        f.rule,
                        "severity":    f.severity,
                        "line":        f.line,
                        "message":     f.message,
                        "snippet":     f.snippet,
                        "source_type": f.source_type,
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
