# keryx/advisors/manager.py
# AdvisorManager — registration, priority dispatch, circuit breaker, parallel execution.
# FIX 8: this is the SINGLE AdvisorManager. base.py no longer defines one.
# Sovereign, production-hardened, deadlock-free sync bridge.

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .base import AdvisorResponse, BaseAdvisor

logger = logging.getLogger("keryx.advisors.manager")


# ---------------------------------------------------------------------------
# Error classification for circuit breaker
# ---------------------------------------------------------------------------
_SYSTEM_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
    MemoryError,
)


def _is_system_error(exc: BaseException) -> bool:
    """True if exception indicates infrastructure failure, not bad model output."""
    return isinstance(exc, _SYSTEM_ERRORS)


# ---------------------------------------------------------------------------
# Circuit breaker state
# ---------------------------------------------------------------------------
@dataclass
class AdvisorHealth:
    """Per-advisor circuit breaker state."""
    consecutive_system_failures: int = 0
    last_failure_time: float = 0.0
    total_calls: int = 0
    total_latency_ms: float = 0.0
    is_open: bool = False
    _open_threshold: int = 5
    _recovery_seconds: float = 60.0

    def record_success(self, latency_ms: float) -> None:
        self.consecutive_system_failures = 0
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.is_open = False

    def record_failure(self, exc: BaseException | None = None) -> None:
        self.total_calls += 1
        if exc is None or _is_system_error(exc):
            self.consecutive_system_failures += 1
            self.last_failure_time = time.time()
            if self.consecutive_system_failures >= self._open_threshold:
                self.is_open = True
                logger.warning(
                    f"[CircuitBreaker] Opened after "
                    f"{self.consecutive_system_failures} consecutive system failures"
                )

    def can_attempt(self) -> bool:
        if not self.is_open:
            return True
        # Half-open: allow one retry after recovery window
        if time.time() - self.last_failure_time > self._recovery_seconds:
            self.is_open = False
            self.consecutive_system_failures = 0
            logger.info("[CircuitBreaker] Half-open: allowing retry")
            return True
        return False

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_calls, 1)


# ---------------------------------------------------------------------------
# Call metrics
# ---------------------------------------------------------------------------
@dataclass
class CallMetrics:
    advisor_name: str
    timestamp: float
    latency_ms: float
    success: bool
    triggered: bool


# ---------------------------------------------------------------------------
# AdvisorManager
# ---------------------------------------------------------------------------
class AdvisorManager:
    """
    Central registry and dispatcher for all advisors.
    Features:
    - Priority-based dispatch
    - Circuit breaker (system errors only)
    - Parallel execution support
    - Deadlock-free synchronous bridge
    """
    def __init__(
        self,
        default_advisor_name: str | None = None,
        max_concurrent_advisor_calls: int | None = None,
        enable_circuit_breaker: bool = True,
    ):
        self._advisors: dict[str, BaseAdvisor] = {}
        self._health: dict[str, AdvisorHealth] = {}
        self._default: str | None = default_advisor_name
        self._lock = threading.RLock()
        self._enable_cb = enable_circuit_breaker
        self._priorities: dict[str, int] = {}

        workers = max_concurrent_advisor_calls or min(os.cpu_count() or 2, 4)
        if workers > 8:
            logger.warning(
                f"[AdvisorManager] {workers} workers requested — "
                "large counts cause OOM with local LLM advisors. Consider ≤4."
            )

        # Dedicated background loop for sync bridge
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()

        # Metrics
        self._call_history: deque[CallMetrics] = deque(maxlen=1000)
        self._total_calls: int = 0
        self._fallback_count: int = 0

        logger.info(
            f"[AdvisorManager] Initialized | "
            f"workers={workers} | circuit_breaker={enable_circuit_breaker}"
        )

    # ------------------------------------------------------------------
    # calls_made property
    # ------------------------------------------------------------------
    @property
    def calls_made(self) -> int:
        """Compatible with orchestrator.py"""
        return self._total_calls

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> AdvisorManager:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.shutdown()
        return False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, advisor: BaseAdvisor, priority: int = 50) -> None:
        with self._lock:
            self._advisors[advisor.name] = advisor
            self._health[advisor.name] = AdvisorHealth()
            self._priorities[advisor.name] = priority
            if self._default is None:
                self._default = advisor.name
            logger.info(f"[AdvisorManager] Registered: {advisor.name!r} (priority={priority})")

    def register_many(self, advisors: list[tuple[BaseAdvisor, int]]) -> None:
        for advisor, priority in advisors:
            self.register(advisor, priority)

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name not in self._advisors:
                return False
            del self._advisors[name]
            del self._health[name]
            self._priorities.pop(name, None)
            if self._default == name:
                remaining = list(self._advisors.keys())
                self._default = remaining[0] if remaining else None
            logger.info(f"[AdvisorManager] Unregistered: {name!r}")
            return True

    def set_default(self, name: str) -> None:
        if name not in self._advisors:
            raise KeyError(f"Advisor {name!r} not registered")
        self._default = name

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def has_advisor(self, name: str) -> bool:
        return name in self._advisors

    def get_advisor(self, name: str) -> BaseAdvisor | None:
        return self._advisors.get(name)

    def list_advisors(self) -> list[str]:
        return sorted(self._advisors.keys(), key=lambda n: self._priorities.get(n, 50))

    # ------------------------------------------------------------------
    # Core async dispatch
    # ------------------------------------------------------------------
    async def get_advice_async(
        self,
        context: Any,
        advisor_name: str | None = None,
        fallback_chain: list[str] | None = None,
        _visited: set[str] | None = None,
    ) -> AdvisorResponse:
        _visited = _visited or set()
        name = advisor_name or self._default

        if not name or name not in self._advisors:
            return self._error_response(f"Advisor not found: {name!r}")

        if name in _visited:
            return self._error_response(f"Circular fallback detected at: {name!r}")

        _visited.add(name)
        advisor = self._advisors[name]
        health = self._health[name]

        if self._enable_cb and not health.can_attempt():
            logger.warning(f"[AdvisorManager] Circuit open for {name!r} — trying fallback")
            return await self._try_fallback(context, fallback_chain, _visited, f"circuit_open:{name}")

        if not advisor.can_advise():
            return await self._try_fallback(context, fallback_chain, _visited, f"limit_reached:{name}")

        # Trigger check
        try:
            triggered = await asyncio.wait_for(
                advisor.should_trigger_async(context), timeout=5.0
            )
        except TimeoutError:
            logger.warning(f"[AdvisorManager] Trigger check timeout: {name!r}")
            triggered = False

        if not triggered:
            return AdvisorResponse(metadata={"advisor": name, "triggered": False})

        # Execute
        start = time.time()
        system_exc: BaseException | None = None

        try:
            logger.info(f"[AdvisorManager] Executing: {name!r}")
            response = await asyncio.wait_for(
                advisor.advise_with_tracking(context),
                timeout=60.0,
            )
            ms = (time.time() - start) * 1000
            health.record_success(ms)
            self._record_call(name, ms, success=True, triggered=True)
            self._total_calls += 1
            return response

        except TimeoutError as exc:
            system_exc = exc
            logger.error(f"[AdvisorManager] {name!r} timed out")
        except _SYSTEM_ERRORS as exc:
            system_exc = exc
            logger.error(f"[AdvisorManager] {name!r} system error: {exc}")
        except Exception as exc:
            # Business-logic failure
            logger.warning(f"[AdvisorManager] {name!r} business error: {exc}")
            ms = (time.time() - start) * 1000
            health.record_failure(exc=None)
            self._record_call(name, ms, success=False, triggered=True)
            self._total_calls += 1
            return self._error_response(str(exc))

        # System failure path
        ms = (time.time() - start) * 1000
        health.record_failure(exc=system_exc)
        self._record_call(name, ms, success=False, triggered=True)
        self._total_calls += 1
        return await self._try_fallback(
            context, fallback_chain, _visited, str(system_exc) if system_exc else "unknown"
        )

    async def _try_fallback(
        self,
        context: Any,
        chain: list[str] | None,
        visited: set[str],
        reason: str,
    ) -> AdvisorResponse:
        if not chain:
            return self._error_response(reason)

        self._fallback_count += 1
        next_name = chain[0]
        remaining = chain[1:] or None
        logger.info(f"[AdvisorManager] Fallback → {next_name!r} (reason: {reason})")
        return await self.get_advice_async(context, next_name, remaining, visited)

    # ------------------------------------------------------------------
    # Priority-based selection
    # ------------------------------------------------------------------
    async def get_advice_priority(
        self,
        context: Any,
        advisor_names: list[str] | None = None,
    ) -> AdvisorResponse:
        names = advisor_names or list(self._advisors.keys())
        sorted_names = sorted(names, key=lambda n: self._priorities.get(n, 50))

        for name in sorted_names:
            response = await self.get_advice_async(context, name)
            if not response.is_empty():
                return response

        return self._error_response("No advisors triggered")

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    async def get_advice_parallel(
        self,
        context: Any,
        advisor_names: list[str],
        require_all: bool = False,
        timeout: float = 30.0,
    ) -> list[AdvisorResponse]:
        valid = [n for n in advisor_names if n in self._advisors]
        if not valid:
            return []

        tasks = [
            asyncio.create_task(
                asyncio.wait_for(self.get_advice_async(context, name), timeout=timeout)
            )
            for name in valid
        ]

        try:
            raw = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(f"[AdvisorManager] Parallel timeout after {timeout}s")
            for t in tasks:
                if not t.done():
                    t.cancel()
            raw = []

        responses = [r for r in raw if isinstance(r, AdvisorResponse)]
        logger.info(f"[AdvisorManager] Parallel: {len(responses)}/{len(valid)} succeeded")
        return responses

    # ------------------------------------------------------------------
    # Synchronous bridge
    # ------------------------------------------------------------------
    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._sync_loop and not self._sync_loop.is_closed():
                return self._sync_loop

            self._loop_ready.clear()
            loop = asyncio.new_event_loop()
            self._sync_loop = loop

            def _run() -> None:
                asyncio.set_event_loop(loop)
                self._loop_ready.set()
                try:
                    loop.run_forever()
                except Exception as exc:  # pragma: no cover
                    logger.error(f"[AdvisorManager] Sync loop crashed: {exc}")  # pragma: no cover

            self._sync_thread = threading.Thread(target=_run, daemon=True, name="advisor-loop")
            self._sync_thread.start()

            if not self._loop_ready.wait(timeout=3.0):  # pragma: no cover
                raise RuntimeError("[AdvisorManager] Sync event loop failed to start")  # pragma: no cover

            return loop

    def get_advice(
        self,
        context: Any,
        advisor_name: str | None = None,
        fallback_chain: list[str] | None = None,
    ) -> AdvisorResponse:
        try:
            asyncio.get_running_loop()
            logger.warning(
                "[AdvisorManager] get_advice() called from async context. "
                "Use get_advice_async() instead."
            )
        except RuntimeError:
            pass  # correct case

        loop = self._ensure_sync_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.get_advice_async(context, advisor_name, fallback_chain),
            loop,
        )
        try:
            return future.result(timeout=90.0)
        except Exception as exc:
            logger.error(f"[AdvisorManager] Sync dispatch failed: {exc}")
            return self._error_response(str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def can_advise(self, advisor_name: str | None = None) -> bool:
        if advisor_name:
            adv = self._advisors.get(advisor_name)
            health = self._health.get(advisor_name)
            if not adv or not health:
                return False
            return adv.can_advise() and (not self._enable_cb or health.can_attempt())

        return any(self.can_advise(n) for n in self._advisors)

    def _record_call(
        self, name: str, latency_ms: float, success: bool, triggered: bool
    ) -> None:
        self._call_history.append(CallMetrics(
            advisor_name=name,
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
            triggered=triggered,
        ))

    def get_metrics(self) -> dict[str, Any]:
        now = time.time()
        recent = [c for c in self._call_history if now - c.timestamp < 300]
        return {
            "total_calls": self._total_calls,
            "fallback_count": self._fallback_count,
            "recent_5m": len(recent),
            "advisor_health": {
                name: {
                    "system_failures": h.consecutive_system_failures,
                    "circuit_open": h.is_open,
                    "avg_latency_ms": round(h.avg_latency_ms, 1),
                    "total_calls": h.total_calls,
                }
                for name, h in self._health.items()
            },
            "latency_histogram": self._latency_histogram(recent),
        }

    @staticmethod
    def _latency_histogram(calls: list[CallMetrics]) -> dict[str, int]:
        buckets: dict[str, int] = {"<100ms": 0, "100-500ms": 0, "0.5-1s": 0, "1-5s": 0, ">5s": 0}
        for c in calls:
            if c.latency_ms < 100:
                buckets["<100ms"] += 1
            elif c.latency_ms < 500:
                buckets["100-500ms"] += 1
            elif c.latency_ms < 1000:
                buckets["0.5-1s"] += 1
            elif c.latency_ms < 5000:
                buckets["1-5s"] += 1
            else:
                buckets[">5s"] += 1
        return buckets

    def get_advisor_health(self, name: str) -> AdvisorHealth | None:
        return self._health.get(name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_all(self) -> None:
        with self._lock:
            for advisor in self._advisors.values():
                advisor.reset()
            for health in self._health.values():
                health.consecutive_system_failures = 0
                health.is_open = False
            self._call_history.clear()
            self._total_calls = 0
            self._fallback_count = 0
            logger.debug("[AdvisorManager] Reset complete")

    def shutdown(self, wait: bool = True) -> None:
        logger.info("[AdvisorManager] Shutting down...")
        if self._sync_loop and not self._sync_loop.is_closed():
            self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0 if wait else 0.5)
        try:
            if self._sync_loop:
                self._sync_loop.close()
        except Exception:
            pass
        logger.info("[AdvisorManager] Done")

    def _error_response(self, error: str) -> AdvisorResponse:
        return AdvisorResponse(
            metadata={"error": error, "timestamp": time.time()}
        )

    def __repr__(self) -> str:
        healthy = sum(1 for h in self._health.values() if not h.is_open)
        return (
            f"AdvisorManager("
            f"advisors={len(self._advisors)}, "
            f"healthy={healthy}, "
            f"calls={self._total_calls})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_advisor_manager(
    advisors: list[tuple[BaseAdvisor, int]] | None = None,
    default: str | None = None,
    **kwargs: Any,
) -> AdvisorManager:
    manager = AdvisorManager(default_advisor_name=default, **kwargs)
    if advisors:
        manager.register_many(advisors)
    return manager
