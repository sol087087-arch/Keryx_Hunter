# keryx/core/_parser.py
"""Pure JSON → AgentStep parsing helpers.

No agent state, no context, no side effects.
All functions are safe to call and test in isolation.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentStep

from ._constants import VALID_ACTIONS


def extract_json(raw: str) -> str | None:
    """Extract the first complete JSON object from raw model output.

    Strips:
    - HTML-like injection attempts (e.g. <script>, </tool_call>)
    - C0/C1 control characters that can smuggle payloads past JSON parsers
    - Markdown code fences (```json ... ```)

    Returns the JSON string, or None if no object was found.
    """
    raw = re.sub(r'<[^>]{0,200}>', '', raw)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and start < end:
        return raw[start:end + 1]
    return None


def fallback_parse(raw: str) -> "AgentStep":
    """Last-resort parse: scan for a known action name in free text.

    Returns an AgentStep with confidence=0.3 if a known action is found,
    or a NO_ACTION step with confidence=0.1 otherwise.
    """
    from .agent import AgentStep   # local import avoids circular dependency

    raw_lower = raw.lower()
    for action in VALID_ACTIONS:
        if action.lower() in raw_lower and action != "NO_ACTION":
            return AgentStep(
                thought="fallback_parse_inferred",
                action=action,
                action_input={},
                confidence=0.3,
            )
    return AgentStep(
        thought="fallback_parse_failed",
        action="NO_ACTION",
        action_input={},
        confidence=0.1,
    )


def parse_response(raw: str) -> tuple["AgentStep", bool]:
    """Parse raw model output into an AgentStep.

    Returns (step, had_parse_error).
    The boolean flag lets callers decide whether to record the error
    (e.g. context.increment_parse_errors()) without coupling the parser
    to any context object.
    """
    from .agent import AgentStep   # local import avoids circular dependency

    json_str = extract_json(raw)
    if not json_str:
        print(f"[WARN] Could not extract JSON from response (len={len(raw)})")
        return fallback_parse(raw), True

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict) or "action" not in data:
            print("[WARN] JSON missing required 'action' key — treating as parse error")
            return fallback_parse(raw), True
        return AgentStep(
            thought=      str(data.get("thought", "")),
            action=       str(data.get("action", "NO_ACTION")),
            action_input= dict(data.get("action_input", {})),
            confidence=   float(data.get("confidence", 0.5)),
        ), False
    except Exception as exc:
        print(f"[WARN] JSON parse failed: {exc.__class__.__name__}: {exc}")
        return fallback_parse(raw), True
