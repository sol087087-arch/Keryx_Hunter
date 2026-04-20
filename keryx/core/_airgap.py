# keryx/core/_airgap.py
# AirGap detection — parallel TCP probe with explicit-mode override.

from __future__ import annotations

import asyncio
import logging
import os
import socket

logger = logging.getLogger("keryx.orchestrator.airgap")

_PROBES: list[tuple[str, int]] = [
    ("1.1.1.1", 53),        # Cloudflare DNS
    ("8.8.8.8", 53),        # Google DNS
    ("208.67.222.222", 53), # OpenDNS
]


async def detect_airgap(explicit_mode: str) -> bool:
    """
    Return True if the environment is air-gapped.

    Decision order:
    1. explicit_mode == "airgapped" → always True (no probe needed)
    2. HTTP_PROXY / HTTPS_PROXY set → assume network reachable
    3. Parallel TCP probe to 3 well-known DNS servers → any success → False
    4. All probes fail → True
    """
    if explicit_mode == "airgapped":
        return True

    if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
        logger.debug("Proxy env var detected — assuming network available")
        return False

    try:
        async with asyncio.timeout(3.0):
            results = await asyncio.gather(*[_probe(h, p) for h, p in _PROBES])
            if any(results):
                return False
    except TimeoutError:
        pass

    logger.info("Network unreachable — air-gapped mode active")
    return True


async def _probe(host: str, port: int) -> bool:
    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: socket.create_connection((host, port), timeout=1.5),
            ),
            timeout=2.0,
        )
        return True
    except Exception:
        return False
