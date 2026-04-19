# keryx/core/_loop_guards.py
"""Pure, stateless stop-condition helpers for KeryxAgent.

No class, no self — fully testable with plain AgentStep objects and primitives.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentStep

_VULN_SIGNALS: tuple[str, ...] = (
    "VULN_CONFIRMED",
    "AddressSanitizer",
    "heap-use-after-free",
    "stack-buffer-overflow",
    "SEGFAULT",
    "CRASH",
    "UAF",
    # AST analyzer signals
    "[HIGH]",
    "GIT_OPTION_INJECTION",
    "SUBPROCESS_SHELL_TRUE",
    "HARDCODED_SECRET",
)


def observation_confirms_vuln(observation: str) -> bool:
    """True if the observation contains any recognised vulnerability signal."""
    return any(s in observation for s in _VULN_SIGNALS)


def check_exit_clean(
    action: "AgentStep",
    consecutive_clean: int,
    max_clean: int,
) -> tuple[bool, int]:
    """Decide whether to early-exit after a clean codeql scan.

    Returns (should_exit, new_consecutive_count).
    Pure — the caller owns the counter and updates it from the returned value.

    A clean scan (no HIGH findings) increments the counter.
    Any HIGH finding resets it to zero.
    When the counter reaches max_clean the loop should stop.
    """
    if action.action != "codeql_query":
        return False, consecutive_clean
    if observation_confirms_vuln(action.observation):
        return False, 0          # HIGH found — reset counter
    new = consecutive_clean + 1
    return new >= max_clean, new
