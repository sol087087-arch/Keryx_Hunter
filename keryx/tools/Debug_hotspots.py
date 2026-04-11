#!/usr/bin/env python3
"""Debug script for GitBlameTool with optional Rich output.

Usage:
    python debug_hotspots.py [path] [--since "2 weeks ago"]
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Make the package importable without installation when run from repo root.
sys.path.insert(0, str(Path(__file__).parent))

from keryx.tools.git_blame import create_git_blame_tool

# FIX: logger must be defined before use — was missing entirely.
logger = logging.getLogger("debug_hotspots")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


async def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    since  = sys.argv[2] if len(sys.argv) > 2 else "3 months ago"

    tool = create_git_blame_tool(timeout_seconds=60.0)

    if not tool.is_available():
        logger.error("Git not found in PATH.")
        return 1

    logger.info("Analyzing hotspots in: %s (since: %s)", target, since)

    result = await tool.execute({
        "command":     "hotspots",
        "path":        target,
        "n":           50,
        "since":       since,
        "extensions":  [".py", ".toml", ".c", ".cpp", ".h"],
        "min_changes": 1,
    })

    if not result.success:
        logger.error("Tool failed: %s — %s", result.error, result.output)
        return 1

    hotspots = result.data.get("hotspots", [])
    logger.info("Found %d hotspots", len(hotspots))
    for h in hotspots[:3]:
        logger.info(
            "  %s | risk=%s | score=%.1f | churn=%d | authors=%d",
            h["file"], h["risk_level"], h["complexity_score"],
            h["changes"], h["authors"],
        )

    metrics = tool.get_metrics()
    logger.info(
        "Metrics: calls=%d successes=%d rate=%.1f%% avg=%.0fms",
        metrics["calls"], metrics["successes"],
        metrics["success_rate"] * 100, metrics["avg_time_ms"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
