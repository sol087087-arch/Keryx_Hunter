# keryx/advisors/cascade.py
# AdvancedCascadeAdvisor — production-hardened chain-of-responsibility advisor.
#
# Features vs simple CascadeAdvisor in base.py:
#   - Per-advisor time budget: min(max_per_advisor_seconds, remaining / N_enabled)
#   - Overall cascade_timeout with graceful budget exhaustion
#   - Error classification: TRANSIENT / RECOVERABLE / CRITICAL
#   - Auto-disabling: after N consecutive timeouts, or on first CRITICAL error
#   - Escalation awareness: skips local-only advisors when confidence is low
#   - AdvisorState: per-advisor tracking (failures, timeouts, timing, disabled flag)
#   - Full reset() restoring all AdvisorState to zero
#
# Factory:
#   create_cascade_advisor(advisors, mode="advanced") → BaseAdvisor
#   mode="simple" returns the plain CascadeAdvisor from base.py
#   AdvisorManager is fully agnostic — it receives a BaseAdvisor, nothing more.

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .base import AdvisorResponse, BaseAdvisor

logger = logging.getLogger("keryx.advisors.cascade")

_MIN_REMAINING_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
class ErrorSeverity(Enum):
    TRANSIENT   = auto()   # rate-limit, 503, connection reset — skip, no penalty
    RECOVERABLE = auto()   # bad output, partial failure — increment failure counter
    CRITICAL    = auto()   # auth failure, model gone — disable advisor for session


# ---------------------------------------------------------------------------
# Per-advisor runtime state
# ---------------------------------------------------------------------------
@dataclass
class AdvisorState:
    """Mutable runtime state for one advisor slot in the chain."""
    advisor:                BaseAdvisor
    disabled:               bool               = False
    consecutive_failures:   int                = 0
    consecutive_timeouts:   int                = 0
    total_time_ms:          float              = 0.0
    last_error:             str | None         = None
    last_error_severity:    ErrorSeverity | None = None


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class CascadeConfig:
    """
    All tuning knobs for AdvancedCascadeAdvisor in one place.
    Pass a custom instance to create_cascade_advisor() or directly to the class.
    """
    cascade_timeout:                    float = 60.0
    max_per_advisor_seconds:            float = 30.0
    critical_confidence_threshold:      float = 0.3
    disable_on_critical:                bool  = True
    disable_after_consecutive_timeouts: int   = 2
    max_calls_per_session:              int   = 5


# ---------------------------------------------------------------------------
# AdvancedCascadeAdvisor
# ---------------------------------------------------------------------------
class AdvancedCascadeAdvisor(BaseAdvisor):
    """
    Chain-of-responsibility advisor with strict time budgets and error handling.

    Chain execution per call to advise():
      For each enabled advisor (in registration order):
        1. Skip if disabled or local-only during escalation
        2. Check overall budget — break if < _MIN_REMAINING_SECONDS remaining
        3. Compute per-advisor budget: min(max_per_advisor_seconds, remaining / N_enabled)
        4. Check trigger (async-safe)
        5. Execute with asyncio.wait_for(per_budget)
        6. Non-empty response → return immediately (_enrich_response)
        7. TimeoutError → increment consecutive_timeouts; disable after N
        8. Exception → classify; disable if CRITICAL
      Cascade exhausted → _empty_response with full chain_status
    """
    name = "advanced-cascade"

    def __init__(
        self,
        advisors: list[BaseAdvisor],
        name: str = "advanced-cascade",
        config: CascadeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if not advisors:
            raise ValueError("AdvancedCascadeAdvisor requires at least one advisor")
        for adv in advisors:
            if adv.name == name:
                raise ValueError(
                    f"Circular reference: cascade '{name}' contains itself"
                )

        cfg = config or CascadeConfig()
        super().__init__(max_calls_per_session=cfg.max_calls_per_session, **kwargs)
        self.name  = name
        self.config = cfg
        self._chain: list[AdvisorState] = [AdvisorState(a) for a in advisors]

        logger.info(
            f"[{self.name}] chain=[{', '.join(s.advisor.name for s in self._chain)}] | "
            f"timeout={cfg.cascade_timeout}s | "
            f"critical_threshold={cfg.critical_confidence_threshold}"
        )

    # ------------------------------------------------------------------
    # Trigger checks
    # ------------------------------------------------------------------
    def should_trigger(self, context: Any) -> bool:
        """Sync trigger: True if any enabled advisor would trigger."""
        for state in self._chain:
            if state.disabled:
                continue
            try:
                result = state.advisor.should_trigger(context)
                if inspect.iscoroutine(result):
                    result.close()   # avoid "coroutine never awaited"
                    return True      # conservative: assume yes
                if result:
                    return True
            except Exception as exc:
                logger.debug(
                    f"[{self.name}] Trigger error '{state.advisor.name}': {exc}"
                )
        return False

    async def should_trigger_async(self, context: Any) -> bool:
        """Async trigger: awaits should_trigger_async on sub-advisors if available."""
        for state in self._chain:
            if state.disabled:
                continue
            try:
                method = getattr(state.advisor, "should_trigger_async", None)
                if method is not None and inspect.iscoroutinefunction(method):
                    triggered = await method(context)
                else:
                    result   = state.advisor.should_trigger(context)
                    triggered = await result if inspect.iscoroutine(result) else bool(result)
                if triggered:
                    return True
            except Exception as exc:
                logger.debug(
                    f"[{self.name}] Async trigger error '{state.advisor.name}': {exc}"
                )
        return False

    # ------------------------------------------------------------------
    # Core cascade
    # ------------------------------------------------------------------
    async def advise(self, context: Any) -> AdvisorResponse:
        start  = time.monotonic()
        path:   list[str]        = []
        timing: dict[str, float] = {}
        errors: dict[str, str]   = {}

        # Escalation awareness — low confidence → skip local-only advisors
        conf = getattr(context, "get_last_confidence", lambda: 1.0)()
        escalating = (
            conf is not None
            and conf < self.config.critical_confidence_threshold
        )
        if escalating:
            logger.info(
                f"[{self.name}] Low confidence ({conf:.2f}) — "
                "escalating past local-only advisors"
            )

        for i, state in enumerate(self._chain):
            advisor = state.advisor

            if state.disabled:
                logger.debug(f"[{self.name}] Skipping '{advisor.name}' — disabled")
                continue

            if escalating and not getattr(advisor, "requires_network", False):
                logger.debug(
                    f"[{self.name}] Escalation: skipping local '{advisor.name}'"
                )
                continue

            # Overall budget
            elapsed   = time.monotonic() - start
            remaining = self.config.cascade_timeout - elapsed
            if remaining <= _MIN_REMAINING_SECONDS:
                logger.warning(
                    f"[{self.name}] Budget exhausted before '{advisor.name}'"
                )
                break

            # Per-advisor budget
            enabled_ahead = sum(1 for s in self._chain[i:] if not s.disabled)
            per_budget    = min(
                self.config.max_per_advisor_seconds,
                remaining / max(enabled_ahead, 1),
            )
            logger.info(
                f"[{self.name}] [{i + 1}/{len(self._chain)}] '{advisor.name}' | "
                f"budget={per_budget:.1f}s"
            )

            adv_start = time.monotonic()
            try:
                triggered = await advisor.should_trigger_async(context)
                if not triggered:
                    logger.debug(f"[{self.name}] '{advisor.name}' did not trigger")
                    continue

                # Call advise() directly so exceptions propagate to our
                # classifier. advise_with_tracking() would swallow them.
                response = await asyncio.wait_for(
                    advisor.advise(context),
                    timeout=per_budget,
                )
                adv_ms = (time.monotonic() - adv_start) * 1000
                state.total_time_ms        += adv_ms
                state.consecutive_failures  = 0
                state.consecutive_timeouts  = 0
                timing[advisor.name] = adv_ms
                path.append(advisor.name)

                if not response.is_empty():
                    logger.info(
                        f"[{self.name}] Success from '{advisor.name}' ({adv_ms:.0f}ms)"
                    )
                    return self._enrich_response(response, path, timing, errors)

                logger.debug(f"[{self.name}] '{advisor.name}' empty — trying next")

            except TimeoutError:
                adv_ms = (time.monotonic() - adv_start) * 1000
                state.total_time_ms       += adv_ms
                state.consecutive_timeouts += 1
                state.consecutive_failures += 1
                errors[advisor.name] = f"timeout_{adv_ms / 1000:.1f}s"
                logger.warning(f"[{self.name}] '{advisor.name}' timed out")

                if (
                    state.consecutive_timeouts
                    >= self.config.disable_after_consecutive_timeouts
                ):
                    state.disabled = True
                    logger.error(
                        f"[{self.name}] Disabled '{advisor.name}' — "
                        f"{state.consecutive_timeouts} consecutive timeouts"
                    )

            except Exception as exc:
                adv_ms   = (time.monotonic() - adv_start) * 1000
                severity = self._classify_error(exc)
                state.total_time_ms       += adv_ms
                state.consecutive_failures += 1
                state.last_error           = str(exc)
                state.last_error_severity  = severity
                errors[advisor.name] = f"{severity.name}:{str(exc)[:60]}"
                logger.error(
                    f"[{self.name}] '{advisor.name}' {severity.name}: {exc}"
                )

                if severity == ErrorSeverity.CRITICAL and self.config.disable_on_critical:
                    state.disabled = True
                    logger.error(
                        f"[{self.name}] Disabled '{advisor.name}' for session"
                    )

        total_ms = (time.monotonic() - start) * 1000
        logger.info(f"[{self.name}] Cascade exhausted ({total_ms:.0f}ms)")
        budget_exhausted = total_ms >= self.config.cascade_timeout * 1000
        return self._empty_response(path, timing, errors, budget_exhausted)

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_error(exc: Exception) -> ErrorSeverity:
        s = str(exc).lower()
        if any(k in s for k in (
            "auth", "api key", "unauthorized", "401", "403",
            "not found", "model unavailable", "connection refused",
        )):
            return ErrorSeverity.CRITICAL
        if any(k in s for k in (
            "timeout", "rate limit", "429", "503",
            "connection reset", "temporarily", "congestion",
        )):
            return ErrorSeverity.TRANSIENT
        return ErrorSeverity.RECOVERABLE

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------
    @staticmethod
    def _enrich_response(
        response: AdvisorResponse,
        path:     list[str],
        timing:   dict[str, float],
        errors:   dict[str, str],
    ) -> AdvisorResponse:
        from dataclasses import replace
        new_meta = dict(response.metadata or {})
        new_meta["cascade"] = {
            "successful_advisor": path[-1] if path else None,
            "path":       path,
            "timing_ms":  {k: int(v) for k, v in timing.items()},
            "failures":   errors,
        }
        return replace(response, metadata=new_meta)

    def _empty_response(
        self,
        path:             list[str],
        timing:           dict[str, float],
        errors:           dict[str, str],
        budget_exhausted: bool,
    ) -> AdvisorResponse:
        return AdvisorResponse(
            metadata={
                "advisor": self.name,
                "cascade": {
                    "path":             path,
                    "attempted":        len(path),
                    "failed":           len(errors),
                    "timing_ms":        {k: int(v) for k, v in timing.items()},
                    "errors":           errors,
                    "budget_exhausted": budget_exhausted,
                    "chain_status": [
                        {
                            "name":     s.advisor.name,
                            "disabled": s.disabled,
                            "failures": s.consecutive_failures,
                            "timeouts": s.consecutive_timeouts,
                            "time_ms":  int(s.total_time_ms),
                        }
                        for s in self._chain
                    ],
                },
            }
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset call counter (via super) and restore all AdvisorState to zero."""
        super().reset()
        for state in self._chain:
            state.disabled              = False
            state.consecutive_failures  = 0
            state.consecutive_timeouts  = 0
            state.total_time_ms         = 0.0
            state.last_error            = None
            state.last_error_severity   = None
            try:
                state.advisor.reset()
            except Exception as exc:
                logger.warning(
                    f"[{self.name}] Failed to reset '{state.advisor.name}': {exc}"
                )

    def get_metrics(self) -> dict[str, Any]:
        return {
            **super().get_metrics(),
            "chain_length":      len(self._chain),
            "active":            sum(1 for s in self._chain if not s.disabled),
            "disabled_advisors": [s.advisor.name for s in self._chain if s.disabled],
            "cascade_timeout":   self.config.cascade_timeout,
            "child_metrics": {
                s.advisor.name: s.advisor.get_metrics() for s in self._chain
            },
        }

    def __repr__(self) -> str:
        active = [s.advisor.name for s in self._chain if not s.disabled]
        return (
            f"AdvancedCascadeAdvisor(name={self.name!r}, "
            f"active={active}, "
            f"budget={self.config.cascade_timeout}s)"
        )


# ---------------------------------------------------------------------------
# Factory — single entry point for all cascade advisor creation
# ---------------------------------------------------------------------------
def create_cascade_advisor(
    advisors: list[BaseAdvisor],
    mode:     str                = "advanced",
    name:     str | None         = None,
    config:   CascadeConfig | None = None,
    **kwargs: Any,
) -> BaseAdvisor:
    """
    Create a cascade advisor.

    Args:
        advisors:  Ordered list of advisors to chain.
        mode:      "advanced" (default) — full feature set.
                   "simple"            — plain CascadeAdvisor from base.py.
        name:      Optional name override.
        config:    CascadeConfig for the advanced version (ignored in simple mode).
        **kwargs:  Forwarded to the constructor.

    Returns:
        BaseAdvisor — callers stay agnostic about the concrete type.
    """
    if mode == "simple":
        from .base import CascadeAdvisor
        return CascadeAdvisor(
            advisors, name=name or "cascade-advisor", **kwargs
        )

    return AdvancedCascadeAdvisor(
        advisors,
        name=name or "advanced-cascade",
        config=config or CascadeConfig(),
        **kwargs,
    )
