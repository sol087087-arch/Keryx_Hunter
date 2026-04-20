# keryx/core/_escalation.py
# Stateless escalation helpers — pure functions, no side effects.
# Loop state (escalation_level, attempts_at_level) lives in _run_hunt().

from __future__ import annotations

import logging
from typing import Any

from .shared_context import SharedContext

logger = logging.getLogger("keryx.orchestrator.escalation")

_THRESHOLD_MAP: dict[int, float] = {1: 0.65, 2: 0.60, 3: 0.55, 4: 0.50}
_DEFAULT_THRESHOLD = 0.60


def get_max_escalation(is_airgapped: bool) -> int:
    """Maximum escalation level allowed in the current network mode."""
    return 2 if is_airgapped else 4


def get_threshold(escalation_level: int) -> float:
    """Confidence threshold decreases as escalation level rises."""
    return _THRESHOLD_MAP.get(escalation_level, _DEFAULT_THRESHOLD)


def next_state(
    current_level: int,
    current_attempts: int,
    max_attempts: int,
) -> tuple[int, int]:
    """
    Return (new_level, new_attempts) after one failed attempt.

    If attempts reaches max_attempts the level is incremented and
    attempts reset to 0. The caller detects escalation by comparing
    new_level != current_level.
    """
    new_attempts = current_attempts + 1
    if new_attempts >= max_attempts:
        return current_level + 1, 0
    return current_level, new_attempts


def should_escalate(
    result: dict[str, Any],
    context: SharedContext,
    current_level: int,
    max_level: int,
) -> bool:
    """
    True when the result is unsatisfactory and there are levels left.

    Conditions (any one triggers escalation):
    - average_confidence < 0.5
    - no hypotheses generated yet
    - parse_errors > 5
    """
    if len(result.get("confirmed_vulns", [])) >= 1:
        return False
    low_confidence = result.get("average_confidence", 0.0) < 0.5
    no_progress = len(context.hypotheses) == 0
    too_many_errors = context.parse_errors > 5
    return (low_confidence or no_progress or too_many_errors) and current_level < max_level
