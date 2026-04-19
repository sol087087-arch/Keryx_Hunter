# keryx/fuzzing/llm_harness.py
"""LLM-based PoC harness generator.

Provides ``make_llm_generate_fn(model)`` which returns a callable compatible
with ``harness.generate(..., llm_generate_fn=fn)``.  Used as a fallback when
no static template exists for a rule (e.g. LLM_OUTPUT_SINK) or when the user
passes ``--fuzz-llm`` to request AI-generated harnesses for every finding.

The generated script must:
  1. Be self-contained (stdlib only).
  2. Reconstruct the vulnerable pattern inline.
  3. Write the *marker* string to stdout when exploitation succeeds.
  4. Complete within the sandbox timeout.
"""
from __future__ import annotations

import re as _re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from keryx.models.interface import ModelInterface

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a security researcher writing sandboxed Python proof-of-concept scripts.

Rules:
- Output ONLY valid Python code. No markdown fences, no commentary outside the script.
- The script must be completely self-contained: stdlib only, no pip-installable packages.
- Reconstruct the vulnerable code pattern inline — never import the original target module.
- Write the exact marker string to sys.stdout when the exploit succeeds.
- The script must finish within the sandbox CPU/time limit (no infinite loops, no sleep > 1 s).
- Destructive operations (rm -rf, DROP TABLE, etc.) are forbidden.
"""

_USER = """\
Security finding that needs a proof-of-concept:
  rule        : {rule}
  observation : {obs}
  marker      : {marker!r}

Write a self-contained Python script that:
1. Reconstructs the minimal vulnerable pattern from the observation above.
2. Executes a crafted payload that proves exploitability.
3. Calls sys.stdout.write(marker + "\\n") exactly once when exploitation is confirmed.

Output the script only — no explanation, no markdown.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = _re.compile(r"^```(?:python)?\s*\n?(.*?)\n?```\s*$", _re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences, if present."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def _looks_like_python(script: str) -> bool:
    """Cheap heuristic: must contain the marker reference and be parseable."""
    import ast
    try:
        ast.parse(script)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_llm_generate_fn(model: "ModelInterface") -> Callable[[dict, str], str | None]:
    """Return an ``llm_generate_fn`` compatible with ``harness.generate()``.

    Parameters
    ----------
    model:
        Any ``ModelInterface`` implementation.  The function calls
        ``model.generate(prompt)`` synchronously.

    Returns
    -------
    Callable[[dict, str], str | None]
        Accepts ``(finding, marker)`` and returns a Python script string, or
        ``None`` if generation fails or the output is not valid Python.
    """
    def _generate(finding: dict, marker: str) -> str | None:
        rule = finding.get("rule", "UNKNOWN")
        obs  = (finding.get("observation") or finding.get("message") or "")[:600]
        prompt = _SYSTEM + "\n\n" + _USER.format(rule=rule, obs=obs, marker=marker)

        try:
            raw = model.generate(prompt)
        except Exception:
            return None

        script = _strip_fences(raw)

        # Sanity: must embed the marker and be valid Python.
        if marker not in script:
            return None
        if not _looks_like_python(script):
            return None

        return script

    return _generate
