# keryx/core/sarif.py
"""SARIF 2.1.0 serialiser for Keryx Hunter results.

Produces output compatible with GitHub Advanced Security / Code Scanning.
One SARIF run per hunt session; one result per confirmed vulnerability.

Usage (from project_hunt.py):
    from keryx.core.sarif import to_sarif
    sarif = to_sarif(hunt_results_dicts, repo_root=str(_REPO_ROOT))
    Path(path).write_text(json.dumps(sarif, indent=2))
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SCHEMA   = "https://json.schemastore.org/sarif-2.1.0.json"
_SARIF_VER = "2.1.0"
_TOOL_NAME = "Keryx Hunter"
_TOOL_VER  = "0.2.0"
_TOOL_URI  = "https://github.com/KeryxHunter/keryx"

# ---------------------------------------------------------------------------
# Rule metadata — short description + recommended fix surfaced in GitHub UI
# ---------------------------------------------------------------------------

_RULE_META: dict[str, tuple[str, str, str]] = {
    # rule_id → (camelCaseName, shortDescription, fix)
    "SUBPROCESS_SHELL_TRUE": (
        "SubprocessShellTrue",
        "subprocess.run() called with shell=True and a non-constant string argument.",
        "Pass a list instead of a string and remove shell=True.",
    ),
    "GIT_OPTION_INJECTION": (
        "GitOptionInjection",
        "User-controlled value injected unsanitised into a git argument list.",
        "Use --flag=value form; add ['--', path] before path arguments.",
    ),
    "GIT_FLAG_NAME_INJECTION": (
        "GitFlagNameInjection",
        "Variable flag name injected into a git argument list.",
        "Never build flag names from user input; use an allow-list.",
    ),
    "OPEN_USER_PATH": (
        "OpenUserPath",
        "open() / Path.read_text() called with a user-controlled path.",
        "Resolve and validate path against an allowed root before open().",
    ),
    "HARDCODED_SECRET": (
        "HardcodedSecret",
        "A secret, password, or API key appears to be hard-coded in source.",
        "Move secret to env var or secrets manager; rotate immediately.",
    ),
    "SUBPROCESS_EXEC_STARRED": (
        "SubprocessExecStarred",
        "asyncio.create_subprocess_exec called with a starred (*args) argument.",
        "Expand *args before the call and validate each element.",
    ),
    "UNSANITIZED_SUBPROCESS_ARG": (
        "UnsanitizedSubprocessArg",
        "Unsanitised user-controlled argument appended to a subprocess command.",
        "Validate/escape arg or use --flag=value; never pass raw user input.",
    ),
    "LLM_OUTPUT_SINK": (
        "LlmOutputSink",
        "json.loads() / json.load() deserialises a non-literal value (possible LLM output sink).",
        "Validate with strict JSON schema / Pydantic before parsing; never eval.",
    ),
    "UNSAFE_EVAL_EXEC": (
        "UnsafeEvalExec",
        "eval() / exec() called with a non-constant argument.",
        "Replace eval/exec with ast.literal_eval or a safe parser.",
    ),
    "UNSAFE_PICKLE": (
        "UnsafePickle",
        "pickle.loads() / pickle.load() deserialises an untrusted value.",
        "Replace pickle with json/msgpack; if pickle is required, sign payloads with hmac.",
    ),
    "UNSAFE_YAML_LOAD": (
        "UnsafeYamlLoad",
        "yaml.load() called without SafeLoader, enabling arbitrary code execution.",
        "Use yaml.safe_load() or pass Loader=yaml.SafeLoader explicitly.",
    ),
    "OS_SHELL_INJECTION": (
        "OsShellInjection",
        "os.system() / os.popen() called with a non-constant argument.",
        "Use subprocess.run([...], ...) with a list instead.",
    ),
    "SSRF": (
        "ServerSideRequestForgery",
        "requests.<method>() called with a non-constant URL argument (possible SSRF).",
        "Validate URL against an allowlist or enforce schema/host with a URL parser.",
    ),
    "TEMPLATE_INJECTION": (
        "TemplateInjection",
        "Template engine instantiated with a non-constant string — possible SSTI.",
        "Never pass user input as a template string; use it only as context data in render().",
    ),
    "REGEX_DOS": (
        "RegexDenialOfService",
        "re.* called with a non-constant pattern — possible ReDoS.",
        "Compile patterns from constants only; never use untrusted input as a regex pattern.",
    ),
}

_DEFAULT_META = ("UnknownRule", "Security finding detected by Keryx Hunter.", "Review and remediate.")

# Confirmed findings are always high severity in SARIF terms.
_LEVEL_MAP: dict[str, str] = {
    "HIGH":   "error",
    "MEDIUM": "warning",
    "LOW":    "note",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_line(observation: str) -> int | None:
    """Extract the first `@ line N` line number from a codeql observation string."""
    m = re.search(r'@\s*line\s+(\d+)', observation)
    return int(m.group(1)) if m else None


def _extract_severity(observation: str) -> str:
    """Extract [HIGH/MEDIUM/LOW] tag from observation; default HIGH."""
    m = re.search(r'\[(HIGH|MEDIUM|LOW)\]', observation)
    return m.group(1) if m else "HIGH"


def _artifact_uri(file_str: str, repo_root: str) -> tuple[str, str | None]:
    """Return (uri, uriBaseId).

    For paths inside the repo root, returns a relative uri + '%SRCROOT%' base.
    For external paths, returns an absolute file:// uri with no base.

    Handles three cases:
    1. Already-relative string (produced by project_hunt._rel()) → SRCROOT-relative.
    2. Absolute path inside repo_root → compute relative path → SRCROOT-relative.
    3. Absolute path outside repo_root → absolute file:// URI.
    """
    p = Path(file_str)

    # Case 1: already a relative path — treat as repo-relative directly.
    if not p.is_absolute():
        return p.as_posix(), "%SRCROOT%"

    # Case 2: absolute path — try to make it repo-relative.
    if repo_root:
        try:
            rel = p.relative_to(repo_root)
            return rel.as_posix(), "%SRCROOT%"
        except ValueError:
            pass

    # Case 3: absolute path outside the repo (or no repo_root given).
    return p.as_uri(), None


def _rule_entry(rule_id: str) -> dict[str, Any]:
    name, short_desc, fix = _RULE_META.get(rule_id, _DEFAULT_META)
    entry: dict[str, Any] = {
        "id":   rule_id,
        "name": name,
        "shortDescription": {"text": short_desc},
        "fullDescription":  {"text": f"{short_desc} {fix}"},
        "help": {"text": fix, "markdown": f"**Recommended fix:** {fix}"},
        "properties": {"tags": ["security"]},
    }
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_sarif(
    hunt_results: list[dict[str, Any]],
    *,
    repo_root: str = "",
) -> dict[str, Any]:
    """Convert Keryx Hunter hunt results to a SARIF 2.1.0 document.

    Parameters
    ----------
    hunt_results:
        List of hunt-result dicts as produced by ``write_output_json`` /
        the ``hunt_results`` key in the JSON report.  Each dict must have:
        ``file``, ``confirmed`` (list of vuln dicts with ``rule``,
        ``observation``, ``confidence_tag`` fields).
    repo_root:
        Absolute path to the repository root.  Used to compute relative URIs
        for files inside the repo.
    """
    seen_rules:  dict[str, dict] = {}
    results:     list[dict]      = []

    for hunt in hunt_results:
        file_str = hunt.get("file", "")
        for vuln in hunt.get("confirmed", []):
            rule_id     = vuln.get("rule") or vuln.get("rules", ["UNKNOWN"])[0]
            observation = vuln.get("observation", "")
            sev         = _extract_severity(observation)
            line_no     = _extract_line(observation)
            tag         = vuln.get("confidence_tag", "AST-only")
            verified    = vuln.get("verified", False)

            # Collect unique rules for the driver.rules array.
            if rule_id not in seen_rules:
                seen_rules[rule_id] = _rule_entry(rule_id)

            uri, base_id = _artifact_uri(file_str, repo_root)
            loc_obj: dict[str, Any] = {"uri": uri}
            if base_id:
                loc_obj["uriBaseId"] = base_id

            region: dict[str, int] = {}
            if line_no is not None:
                region["startLine"] = line_no

            location: dict[str, Any] = {
                "physicalLocation": {
                    "artifactLocation": loc_obj,
                    **({"region": region} if region else {}),
                }
            }

            _, short_desc, _ = _RULE_META.get(rule_id, _DEFAULT_META)
            message_text = (
                f"{short_desc} "
                f"[confidence_tag={tag}, verified={verified}]"
            )

            result: dict[str, Any] = {
                "ruleId":    rule_id,
                "level":     _LEVEL_MAP.get(sev, "error"),
                "message":   {"text": message_text},
                "locations": [location],
                "properties": {
                    "confidence_tag": tag,
                    "verified":       verified,
                },
            }

            # Attach fuzz PoC evidence when available.
            fuzz = vuln.get("fuzz_proof")
            if fuzz and fuzz.get("reproduced"):
                result["properties"]["fuzz_reproduced"]      = True
                result["properties"]["fuzz_confidence_boost"] = fuzz.get("confidence_boost", 0.0)

            results.append(result)

    sarif: dict[str, Any] = {
        "$schema": _SCHEMA,
        "version": _SARIF_VER,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name":            _TOOL_NAME,
                        "version":         _TOOL_VER,
                        "informationUri":  _TOOL_URI,
                        "rules":           list(seen_rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return sarif
