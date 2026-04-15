# keryx/tools/injection_verifier.py
# InjectionVerifier — probes another registered tool with injected payloads to
# verify whether it correctly rejects, accepts, or crashes on attacker-controlled input.
#
# Interface:
#   action_input = {
#       "target_tool":        "git_blame",          # tool name in the ToolBox registry
#       "inject_field":       "author",              # key in the tool's action_input to inject into
#       "payload":            "--upload-pack=test",  # the attacker-controlled value
#       "expected_behavior":  "reject",              # "reject" | "accept" | "any"
#   }
#
# Returns a ToolResult whose output describes:
#   - Whether the target tool accepted/rejected the payload
#   - Whether VULN_CONFIRMED should be raised (tool accepted a clearly-dangerous payload)
#   - Raw output snippet from the target tool for evidence

from __future__ import annotations

import logging
from typing import Any

from .Toolbox import BaseTool, ToolBox, ToolResult

logger = logging.getLogger("keryx.tools.injection_verifier")

# Payloads that a secure tool MUST reject — acceptance is a confirmed vulnerability
_DANGEROUS_MARKERS = (
    "--upload-pack=",
    "--exec=",
    "--receive-pack=",
    "$(", "`",                # shell command substitution
    ";",                      # command chain
    "&&", "||",               # logic chain
    "../../../",              # path traversal
    "\x00",                   # null byte injection
)


class InjectionVerifier(BaseTool):
    """
    Meta-tool that exercises another tool with injected payloads.

    Registered as 'injection_verifier'. The ToolBox must be provided at
    construction time so the tool can dispatch to the target at runtime.
    """

    name        = "injection_verifier"
    description = (
        "Probe a registered tool with an attacker-controlled payload injected "
        "into a specific input field.  Reports whether the tool correctly "
        "rejects, silently accepts, or crashes on the payload.  "
        "Confirmed injection → output contains VULN_CONFIRMED."
    )

    def __init__(self, toolbox: ToolBox) -> None:
        super().__init__()
        self._toolbox = toolbox

    async def execute(
        self,
        action_input: dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        target_tool       = action_input.get("target_tool", "")
        inject_field      = action_input.get("inject_field", "")
        payload           = action_input.get("payload", "")
        expected_behavior = action_input.get("expected_behavior", "any").lower()

        # ── Validate inputs ───────────────────────────────────────────────
        if not target_tool:
            return ToolResult(success=False, output="Missing 'target_tool'", error="missing_field")
        if not inject_field:
            return ToolResult(success=False, output="Missing 'inject_field'", error="missing_field")
        if payload == "":
            return ToolResult(success=False, output="Missing 'payload'", error="missing_field")

        tool = self._toolbox.get_tool(target_tool)
        if tool is None:
            available = self._toolbox.list_tools()
            return ToolResult(
                success=False,
                output=f"Target tool {target_tool!r} not found. Available: {available}",
                error="tool_not_found",
            )

        # ── Build injected action_input ───────────────────────────────────
        # Use a minimal but plausible base input for each known tool; fall back
        # to just the injected field so the tool can still respond.
        base_inputs: dict[str, dict[str, Any]] = {
            "git_blame":    {"file": "/dev/null"},
            "read_file":    {"path": "/dev/null"},
            "codeql_query": {"path": "/dev/null"},
        }
        injected_input = dict(base_inputs.get(target_tool, {}))
        # base_overrides lets callers substitute a real file path instead of /dev/null
        # so the target tool can actually execute (important for git_blame).
        base_overrides = action_input.get("base_overrides", {})
        if isinstance(base_overrides, dict):
            injected_input.update(base_overrides)
        injected_input[inject_field] = payload

        logger.info(
            f"[InjectionVerifier] Probing {target_tool!r} | "
            f"field={inject_field!r} | payload={payload!r}"
        )

        # ── Execute target tool with the injected payload ─────────────────
        result = await self._toolbox.execute_async(
            target_tool, injected_input, context
        )

        # ── Classify the outcome ──────────────────────────────────────────
        raw_output    = result.output or ""
        tool_rejected = not result.success or any(
            marker in raw_output.lower()
            for marker in ("error", "denied", "blocked", "not found", "invalid")
        )

        is_dangerous_payload = any(m in payload for m in _DANGEROUS_MARKERS)
        vuln_confirmed       = is_dangerous_payload and not tool_rejected

        # Build the report
        verdict = "REJECTED" if tool_rejected else "ACCEPTED"

        lines = [
            f"[InjectionVerifier] target={target_tool!r} field={inject_field!r}",
            f"  payload         : {payload!r}",
            f"  expected        : {expected_behavior.upper()}",
            f"  tool_success    : {result.success}",
            f"  verdict         : {verdict}",
        ]

        # Expected-vs-actual mismatch warnings
        if expected_behavior == "reject" and not tool_rejected:
            lines.append(
                "  [WARN] Tool accepted a payload it should have rejected — "
                "likely vulnerable to injection."
            )
        elif expected_behavior == "accept" and tool_rejected:
            lines.append(
                "  [WARN] Tool rejected a payload that should be valid — "
                "possible over-sanitization."
            )

        if vuln_confirmed:
            lines.append(
                "  VULN_CONFIRMED — dangerous payload was NOT rejected; "
                "input reaches the tool without sanitization."
            )

        lines.append(f"\n  --- target tool output (first 800 chars) ---")
        lines.append(raw_output[:800])

        output = "\n".join(lines)

        return ToolResult(
            success=True,
            output=output,
            data={
                "target_tool":       target_tool,
                "inject_field":      inject_field,
                "payload":           payload,
                "expected_behavior": expected_behavior,
                "tool_rejected":     tool_rejected,
                "vuln_confirmed":    vuln_confirmed,
                "target_success":    result.success,
            },
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_injection_verifier(toolbox: ToolBox) -> InjectionVerifier:
    return InjectionVerifier(toolbox=toolbox)
