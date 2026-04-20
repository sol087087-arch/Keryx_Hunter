# keryx/core/_resources.py
# System resource checks — separate from escalation logic.

from __future__ import annotations

import logging

import psutil

logger = logging.getLogger("keryx.orchestrator.resources")

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
