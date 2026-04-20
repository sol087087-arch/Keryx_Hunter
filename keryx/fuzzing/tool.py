"""FuzzerTool — Toolbox wrapper for targeted PoC reproduction."""
from __future__ import annotations

import dataclasses
import json

from keryx.tools.Toolbox import BaseTool, ToolResult

from .poc import reproduce, PoCResult


class FuzzerTool(BaseTool):
    """Attempt to reproduce a confirmed finding via a sandboxed PoC harness.

    Accepts a finding dict (must contain at least ``rule``).  Returns a
    ToolResult whose ``data`` field carries the full PoCResult.
    """

    name            = "fuzzer"
    description     = (
        "Run a targeted proof-of-concept harness for a confirmed AST finding "
        "inside a resource-limited sandbox subprocess.  Returns reproduced=True "
        "when the exploit marker appears in output, confirming real exploitability."
    )
    default_timeout = 30.0   # generous wall-clock limit for the tool itself

    def __init__(self, sandbox_timeout: int = 10) -> None:
        super().__init__()
        self._sandbox_timeout = sandbox_timeout

    async def execute(
        self,
        action_input: dict,
        context=None,
    ) -> ToolResult:
        finding = action_input.get("finding")
        if not isinstance(finding, dict) or "rule" not in finding:
            return ToolResult(
                success=False,
                output="'finding' must be a dict with at least a 'rule' key.",
                error="invalid_input",
            )

        timeout = int(action_input.get("timeout", self._sandbox_timeout))
        poc: PoCResult = reproduce(finding, timeout=timeout)

        status_line = (
            f"[FuzzerTool] rule={poc.rule}  "
            f"reproduced={poc.reproduced}  "
            f"elapsed={poc.elapsed_s:.2f}s"
        )
        if poc.error:
            status_line += f"  error={poc.error!r}"

        detail_lines = [status_line]
        if poc.reproduced:
            detail_lines.append(f"  marker found in output — exploit confirmed")
            detail_lines.append(f"  stdout: {poc.stdout[:200]!r}")
        elif poc.stdout or poc.stderr:
            detail_lines.append(f"  stdout: {poc.stdout[:100]!r}")
            detail_lines.append(f"  stderr: {poc.stderr[:100]!r}")

        return ToolResult(
            success=True,
            output="\n".join(detail_lines),
            data=dataclasses.asdict(poc),
        )


def create_fuzzer_tool(sandbox_timeout: int = 10) -> FuzzerTool:
    return FuzzerTool(sandbox_timeout=sandbox_timeout)
