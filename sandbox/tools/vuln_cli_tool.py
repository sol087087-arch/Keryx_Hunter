# sandbox/tools/vuln_cli_tool.py
"""VulnCliTool — tool wrapper for sandbox/targets/vuln_cli.py.

Registered as "vuln_cli" so resolve_target_tool() matches it when the
hunt target is vuln_cli.py (stem == tool name).

Simulates the vulnerable behaviour: the author field is accepted without
sanitization and echoed back in the output.  No real subprocess is spawned —
this keeps the sandbox deterministic and safe.

Why no real subprocess?
    If we actually ran `git log --author <payload>`, git itself would reject
    unknown flags (e.g. --upload-pack=) and return a non-zero exit code,
    which InjectionVerifier would classify as REJECTED — a false negative.
    The simulation skips the OS layer and demonstrates what a *successfully
    injected* git call would look like: the payload passes through silently.
"""
from __future__ import annotations

from typing import Any

from keryx.tools.Toolbox import BaseTool, ToolResult


class VulnCliTool(BaseTool):
    """
    Simulates a vulnerable git-log wrapper.

    InjectionVerifier probes this tool by injecting a dangerous payload into
    the "author" field.  Because the tool echoes the value without validation,
    the output contains no rejection markers ("error", "denied", "blocked") →
    InjectionVerifier classifies it as ACCEPTED → VULN_CONFIRMED fires.
    """

    name = "vuln_cli"
    description = (
        "Fetch recent git-log entries optionally filtered by author. "
        "Wraps sandbox/targets/vuln_cli.py — intentionally vulnerable to "
        "git option injection via the 'author' field."
    )

    async def execute(
        self,
        action_input: dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        author = action_input.get("author", action_input.get("file", ""))
        repo   = action_input.get("repo",   ".")
        since  = action_input.get("since",  "")

        # Simulate vulnerable behaviour: author is embedded in the response
        # without any leading-dash check or "--" separator.
        lines = [
            f"repo    : {repo}",
            f"author  : {author}",    # unsanitized — payload passes through
        ]
        if since:
            lines.append(f"since   : {since}")
        lines += [
            "---",
            "a1b2c3d Implement fetch_commits()",
            "e4f5a6b Add argparse CLI",
            "c7d8e9f Initial commit",
        ]

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"author": author, "repo": repo, "since": since},
        )


def create_vuln_cli_tool() -> VulnCliTool:
    return VulnCliTool()
