# keryx/core/_checkpoint.py
# CheckpointManager — atomic JSON checkpoint read/write for KeryxOrchestrator.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .shared_context import SharedContext

logger = logging.getLogger("keryx.orchestrator.checkpoint")

_DEFAULT_DIR = Path(".keryx_checkpoints")


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path = _DEFAULT_DIR) -> None:
        self._dir = checkpoint_dir

    def checkpoint_path(self, target_path: str) -> Path:
        key = hashlib.sha256(target_path.encode()).hexdigest()[:12]
        return self._dir / f"orchestrator_{key}.json"

    async def save(
        self,
        target_path: str,
        context: SharedContext,
        metrics: dict[str, Any],
    ) -> None:
        self._dir.mkdir(exist_ok=True)
        path = self.checkpoint_path(target_path)
        state = {"context": context.to_dict(), "metrics": metrics}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_atomic, path, state)

    def _write_atomic(self, path: Path, data: dict[str, Any]) -> None:
        """Write to .tmp then rename — safe against partial writes."""
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.rename(path)
            logger.debug("Checkpoint saved: %s", path.name)
        except Exception as exc:
            logger.error("Atomic checkpoint write failed: %s", exc)
            tmp.unlink(missing_ok=True)

    def load(self, target_path: str) -> SharedContext | None:
        path = self.checkpoint_path(target_path)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ctx = SharedContext.from_dict(data["context"])
            logger.info("Resumed from checkpoint (%s)", path.name)
            return ctx
        except Exception as exc:
            logger.warning("Checkpoint load failed (%s) — starting fresh", exc)
            return None
