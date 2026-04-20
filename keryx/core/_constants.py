# keryx/core/_constants.py
"""Module-level constants for KeryxAgent — pure data, no logic."""
from __future__ import annotations

# GBNF grammar — tested against llama.cpp grammar validator
OUTPUT_GRAMMAR = r'''
root ::= "{" ws kv-thought "," ws kv-action "," ws kv-input "," ws kv-conf ws "}"
kv-thought ::= "\"thought\"" ws ":" ws string
kv-action ::= "\"action\"" ws ":" ws string
kv-input ::= "\"action_input\"" ws ":" ws object
kv-conf ::= "\"confidence\"" ws ":" ws number
value ::= string | number | object | array | "true" | "false" | "null"
object ::= "{" ws ( string ws ":" ws value ( "," ws string ws ":" ws value )* )? ws "}"
array ::= "[" ws ( value ( "," ws value )* )? ws "]"
string ::= "\"" ( [^"\\] | "\\" . )* "\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
ws ::= [ \t\n\r]*
'''

VALID_ACTIONS: frozenset[str] = frozenset([
    "codeql_query",
    "gdb_analyze",
    "fuzzer_run",
    "git_blame",
    "injection_verifier",
    "read_file",
    "rag_search",
    "FINISH",
    "NO_ACTION",
])

# Tools whose output can confirm a real vulnerability (source check for confirmation logic)
VULN_CONFIRMING_TOOLS: frozenset[str] = frozenset([
    "gdb_analyze",
    "fuzzer_run",
    "codeql_query",
    "injection_verifier",
])

ESCALATION_COOLDOWN:  int   = 5
MAX_HISTORY_STEPS:    int   = 12
TOOL_TIMEOUT_SECONDS: float = 30.0

# Data-driven payload table for strict-mode auto-verification.
# Maps AST finding rule → injection payload.  None = static-only (no verifier call).
PAYLOAD_MAP: dict[str, str | None] = {
    "GIT_OPTION_INJECTION":        "--upload-pack=test",
    "SUBPROCESS_SHELL_TRUE":       "$(id)",
    "UNSANITIZED_SUBPROCESS_ARG":  "../../etc/passwd",
    "OPEN_USER_PATH":              "../../etc/passwd",
    "SUBPROCESS_EXEC_STARRED":     "../../etc/passwd",
    "HARDCODED_SECRET":            None,   # static-only, nothing to inject
    "SSRF":                        "http://169.254.169.254/latest/meta-data/",
    "TEMPLATE_INJECTION":          "{{7*7}}",
    "REGEX_DOS":                   "(a+)+$",
}

# Maps Python variable names (from AST) → likely action_input keys.
VARNAME_TO_FIELD: dict[str, str] = {
    "file_path":     "file",
    "filepath":      "file",
    "path":          "path",
    "author":        "author",
    "grep":          "grep",
    "since":         "since",
    "until":         "until",
    "template_str":  "template",
    "tpl":           "template",
    "user_tpl":      "template",
    "tmpl":          "template",
    "template":      "template",
    "pattern":       "pattern",
    "user_pattern":  "pattern",
    "regex":         "pattern",
    "pat":           "pattern",
    "regexp":        "pattern",
}
