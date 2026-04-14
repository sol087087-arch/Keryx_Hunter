# keryx/core/_escalation.py
# Stateless escalation helpers — predicates and thresholds only.
# Loop state (escalation_level, attempts_at_level) lives in _run_hunt().

from __future__ import annotations

import logging
from typing import Any

import psutil

from .shared_context import SharedContext

logger = logging.getLogger("keryx.orchestrator.escalation")

_THRESHOLD_MAP: dict[int, float] = {1: 0.65, 2: 0.60, 3: 0.55, 4: 0.50}
_DEFAULT_THRESHOLD = 0.60
_MEMORY_FLOOR_BYTES = 512 * 1024 * 1024  # 512 MB


async def check_resources() -> bool:
    """Return False if available RAM is below the safety floor."""
    try:
        if psutil.virtual_memory().available < _MEMORY_FLOOR_BYTES:
            logger.warning("Low memory — stopping escalation")
            return False
    except Exception:
        pass
    return True


def should_escalate_further(
    result: dict[str, Any],
    current_level: int,
    max_escalation: int,
    context: SharedContext,
) -> bool:
    """
    True when the current result is unsatisfactory and there are levels left.
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
    return (low_confidence or no_progress or too_many_errors) and current_level < max_escalation


def get_dynamic_threshold(escalation_level: int) -> float:
    """Confidence threshold decreases as escalation level rises."""
    return _THRESHOLD_MAP.get(escalation_level, _DEFAULT_THRESHOLD)
