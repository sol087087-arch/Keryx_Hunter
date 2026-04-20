"""PoCResult dataclass and the reproduce() orchestrator."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

from . import harness, sandbox

_CONFIDENCE_BOOST = 0.35  # added to agent confidence on successful reproduction


@dataclass
class PoCResult:
    rule: str
    reproduced: bool
    payload: str          # the injected string (marker or crafted payload)
    marker: str           # unique sentinel expected in output
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    confidence_boost: float = 0.0
    elapsed_s: float = 0.0
    error: str | None = None
    harness_script: str = field(default="", repr=False)  # full script, for debugging
    # How strongly the exploit was confirmed:
    #   "reachable" — code path reached, no anomalous behaviour observed
    #   "triggered" — anomalous behaviour (timeout / non-zero exit) without marker
    #   "exploited" — exploit marker found in stdout/stderr
    verification_level: Literal["reachable", "triggered", "exploited"] = "reachable"


def reproduce(
    finding: dict,
    *,
    timeout: int = 10,
    llm_generate_fn: Callable[[dict, str], str] | None = None,
) -> PoCResult:
    """Generate and execute a harness for *finding* in a sandboxed subprocess.

    Returns a PoCResult with ``reproduced=True`` when the exploit marker
    appears in stdout/stderr, confirming real exploitability.
    """
    rule   = finding.get("rule", "UNKNOWN")
    marker = harness.MARKER_PREFIX + uuid.uuid4().hex[:8].upper()

    script = harness.generate(finding, marker, llm_generate_fn=llm_generate_fn)
    if script is None:
        return PoCResult(
            rule=rule,
            reproduced=False,
            payload="",
            marker=marker,
            error=f"No harness available for rule '{rule}'; "
                  "add a template or pass llm_generate_fn.",
        )

    result     = sandbox.run(script, timeout=timeout)
    combined   = result.stdout + result.stderr
    reproduced = marker in combined

    # Harness may self-report level via KERYX_LEVEL=<value> in stdout/stderr.
    # This lets rules like REGEX_DOS set an honest level (triggered/reachable)
    # while still emitting the marker so reproduced=True.
    keryx_level: str | None = None
    for line in combined.splitlines():
        if line.startswith("KERYX_LEVEL="):
            keryx_level = line.split("=", 1)[1].strip()
            break

    if keryx_level in ("exploited", "triggered", "reachable"):
        level = keryx_level
    elif reproduced:
        level = "exploited"
    elif result.timed_out or (result.exit_code is not None and result.exit_code != 0):
        level = "triggered"
    else:
        level = "reachable"

    return PoCResult(
        rule               = rule,
        reproduced         = reproduced,
        payload            = marker,
        marker             = marker,
        stdout             = result.stdout[:2000],
        stderr             = result.stderr[:1000],
        exit_code          = result.exit_code,
        confidence_boost   = _CONFIDENCE_BOOST if reproduced else 0.0,
        elapsed_s          = result.elapsed_s,
        harness_script     = script,
        verification_level = level,
    )
