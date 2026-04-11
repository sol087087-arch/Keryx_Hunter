#!/usr/bin/env python3
"""Debug script: lazy loading correctness + hotspot smoke test.

Usage:
    python debug_lazy.py [path] ["since string"]

Examples:
    python debug_lazy.py .
    python debug_lazy.py ./src "2 weeks ago"
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make the package importable without installation when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("debug_lazy")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


async def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    since  = sys.argv[2] if len(sys.argv) > 2 else "3 months ago"

    # ── 1. Import speed ───────────────────────────────────────────────────
    logger.info("=== Fast import test ===")
    t0 = time.perf_counter()
    import keryx
    elapsed = time.perf_counter() - t0
    logger.info("Import time: %.3fs (target <0.5s)", elapsed)
    logger.info("Version: %s  tuple: %s", keryx.__version__, keryx.__version_tuple__)

    # ── 2. Eager symbols (schemas + toolbox) ─────────────────────────────
    logger.info("=== Eager symbol test ===")
    # FIX: use toolbox.ToolResult (dataclass), not schemas.ToolResult.
    # keryx.ToolResult is imported from toolbox in __init__.py.
    tr = keryx.ToolResult(success=True, output="smoke test")
    assert tr.success, "ToolResult broken"
    logger.info("ToolResult OK: success=%s", tr.success)

    risk = keryx.RiskLevel.HIGH
    logger.info("RiskLevel OK: %s", risk.value)

    # ── 3. Lazy tool loading ──────────────────────────────────────────────
    logger.info("=== Lazy tool load test ===")
    tool = keryx.create_git_blame_tool(timeout_seconds=60.0)
    logger.info("GitBlameTool: name=%s available=%s", tool.name, tool.is_available())

    if not tool.is_available():
        logger.warning("git not found — skipping hotspot test")
        return 0

    # ── 4. Hotspot smoke test ─────────────────────────────────────────────
    logger.info("=== Hotspot smoke test | path=%s since='%s' ===", target, since)
    result = await tool.execute({
        "command":     "hotspots",
        "path":        target,
        "n":           50,
        "since":       since,
        "extensions":  [".py", ".toml", ".c", ".cpp", ".h"],
        "min_changes": 1,
    })

    # result is toolbox.ToolResult (dataclass) — consistent with rest of pipeline
    logger.info("Success: %s  error: %s", result.success, result.error)

    if not result.success:
        logger.error("Tool failed: %s", result.output)
        return 1

    hotspots = result.data.get("hotspots", [])
    logger.info("Hotspots found: %d", len(hotspots))
    for h in hotspots[:3]:
        logger.info(
            "  %-40s risk=%-8s score=%5.1f churn=%d authors=%d",
            h["file"], h["risk_level"], h["complexity_score"],
            h["changes"], h["authors"],
        )

    # ── 5. Metrics ────────────────────────────────────────────────────────
    logger.info("=== Metrics ===")
    m = tool.get_metrics()
    logger.info(
        "calls=%d successes=%d rate=%.1f%% avg_time=%.0fms",
        m["calls"], m["successes"],
        m["success_rate"] * 100, m["avg_time_ms"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
