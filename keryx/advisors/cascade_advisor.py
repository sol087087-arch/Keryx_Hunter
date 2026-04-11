# keryx/advisors/cascade_advisor.py
# CascadeAdvisor — hardened chain-of-responsibility with time budgets,
# error classification, and escalation awareness.
# 
# This is the CANONICAL CascadeAdvisor. base.py should import from here:
# from .cascade_advisor import CascadeAdvisor
# 
# Requires Python 3.11+
from **future** import annotations
import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from .base import BaseAdvisor, AdvisorResponse
logger = logging.getLogger("keryx.advisors.cascade")
_MAX_PER_ADVISOR_SECONDS = 30.0
_MIN_REMAINING_SECONDS = 0.5
class ErrorSeverity(Enum):
    TRANSIENT = auto() # Retryable: timeout, rate limit, 503
    RECOVERABLE = auto() # Bad output, JSON parse error
    CRITICAL = auto() # Auth failure, model unloaded, 401/403
@dataclass
class AdvisorState:
    """Runtime state for one advisor in the chain."""
    advisor: BaseAdvisor
    disabled: bool = False
    consecutive_failures: int = 0
    total_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_severity: Optional[ErrorSeverity] = None
class CascadeAdvisor(BaseAdvisor):
    """
    Chain-of-responsibility advisor with strict time budgets and error classification.
    Chain execution:
        For each enabled advisor in order:
            1. Check trigger (async-safe)
            2. Allocate time budget: min(_MAX_PER_ADVISOR_SECONDS, remaining/N)
            3. Execute with asyncio.wait_for
            4. On non-empty response → return immediately
            5. On CRITICAL error → disable advisor for rest of session (documented)
            6. On TRANSIENT → skip, no penalty
            7. On RECOVERABLE → increment failure counter
    Escalation:
        When confidence < critical_confidence_threshold, skip advisors with
        requires_network=False (local-only) and go straight to cloud/specialist.
    """
    name = "cascade-advisor"
    def **init**(
        self,
        advisors: List[BaseAdvisor],
        name: str = "cascade-advisor",
        max_calls_per_session: int = 5,
        cascade_timeout: float = 60.0,
        critical_confidence_threshold: float = 0.3,
        disable_on_critical: bool = True,
        **kwargs,
    ):
        if not advisors:
            raise ValueError("CascadeAdvisor requires at least one advisor")
        for adv in advisors:
            if adv.name == name:
                raise ValueError(f"Circular reference: cascade '{name}' contains itself")
        super().**init**(max_calls_per_session=max_calls_per_session, **kwargs)
        self.name = name
        self.cascade_timeout = cascade_timeout
        self.critical_confidence_threshold = critical_confidence_threshold
        self.disable_on_critical = disable_on_critical
        self._chain: List[AdvisorState] = [AdvisorState(a) for a in advisors]
        logger.info(
            f"[{self.name}] chain=[{', '.join(a.advisor.name for a in self._chain)}] | "
            f"timeout={cascade_timeout}s | critical_threshold={critical_confidence_threshold}"
        )
    # ------------------------------------------------------------------
    # Trigger — must implement @abstractmethod from BaseAdvisor
    # ------------------------------------------------------------------
    def should_trigger(self, context: Any) -> bool:
        """Sync trigger: True if any enabled advisor would trigger."""
        for state in self._chain:
            if state.disabled:
                continue
            try:
                result = state.advisor.should_trigger(context)
                if inspect.iscoroutine(result):
                    result.close()
                    return True # conservative
                if result:
                    return True
            except Exception as exc:
                logger.debug(f"[{self.name}] Trigger check error '{state.advisor.name}': {exc}")
        return False
    async def should_trigger_async(self, context: Any) -> bool:
        """Async trigger: awaits each advisor's should_trigger_async."""
        for state in self._chain:
            if state.disabled:
                continue
            try:
                method = getattr(state.advisor, 'should_trigger_async', None)
                if method is not None and asyncio.iscoroutinefunction(method):
                    triggered = await method(context)
                else:
                    result = state.advisor.should_trigger(context)
                    triggered = await result if inspect.iscoroutine(result) else bool(result)
                if triggered:
                    return True
            except Exception as exc:
                logger.debug(f"[{self.name}] Async trigger error '{state.advisor.name}': {exc}")
        return False
    # ------------------------------------------------------------------
    # Core cascade
    # ------------------------------------------------------------------
    async def advise(self, context: Any) -> AdvisorResponse:
        start = time.monotonic()
        path: List[str] = []
        timing: Dict[str, float] = {}
        errors: Dict[str, str] = {}
        conf = getattr(context, 'get_last_confidence', lambda: 1.0)()
        critical = (conf is not None) and (conf < self.critical_confidence_threshold)
        if critical:
            logger.info(
                f"[{self.name}] Critical confidence ({conf:.2f}) — "
                "skipping local-only advisors"
            )
        for i, state in enumerate(self._chain):
            advisor = state.advisor
            if state.disabled:
                logger.debug(f"[{self.name}] Skipping '{advisor.name}' — disabled")
                continue
            # Escalation: skip local advisors when confidence is critically low.
            # Uses requires_network flag (False = local-only).
            if critical and not advisor.requires_network:
                logger.debug(f"[{self.name}] Escalating past local '{advisor.name}'")
                continue
            elapsed = time.monotonic() - start
            remaining = self.cascade_timeout - elapsed
            if remaining <= _MIN_REMAINING_SECONDS:
                logger.warning(f"[{self.name}] Budget exhausted before '{advisor.name}'")
                break
            enabled_remaining = sum(1 for s in self._chain[i:] if not s.disabled)
            per_budget = min(
                _MAX_PER_ADVISOR_SECONDS,
                remaining / max(enabled_remaining, 1),
            )
            logger.info(
                f"[{self.name}] [{i + 1}/{len(self._chain)}] '{advisor.name}' | "
                f"budget={per_budget:.1f}s"
            )
            adv_start = time.monotonic()
            try:
                triggered = await state.advisor.should_trigger_async(context)
                if not triggered:
                    logger.debug(f"[{self.name}] '{advisor.name}' did not trigger")
                    continue
                response = await asyncio.wait_for(
                    advisor.advise_with_tracking(context),
                    timeout=per_budget,
                )
                adv_ms = (time.monotonic() - adv_start) * 1000
                state.total_time_ms += adv_ms
                timing[advisor.name] = adv_ms
                path.append(advisor.name)
                if not response.is_empty():
                    logger.info(f"[{self.name}] Success from '{advisor.name}' ({adv_ms:.0f}ms)")
                    return self.*enrich_response(response, path, timing, errors)
                logger.debug(f"[{self.name}] '{advisor.name}' returned empty — next")
            except asyncio.TimeoutError:
                adv_ms = (time.monotonic() - adv_start) * 1000
                state.total_time_ms += adv_ms
                state.consecutive_failures += 1
                errors[advisor.name] = f"timeout*{adv_ms / 1000:.1f}s"
                logger.warning(f"[{self.name}] '{advisor.name}' timed out")
                if state.consecutive_failures >= 2:
                    state.disabled = True
                    logger.error(f"[{self.name}] Disabled '{advisor.name}' — repeated timeouts")
            except Exception as exc:
                adv_ms = (time.monotonic() - adv_start) * 1000
                state.total_time_ms += adv_ms
                severity = self._classify_error(exc)
                state.last_error = str(exc)
                state.last_error_severity = severity
                errors[advisor.name] = f"{severity.name}:{str(exc)[:60]}"
                logger.error(f"[{self.name}] '{advisor.name}' {severity.name}: {exc}")
                if severity == ErrorSeverity.CRITICAL and self.disable_on_critical:
                    # Disabled for the rest of this session.
                    # Call reset() to re-enable between hunts.
                    state.disabled = True
                    logger.error(
                        f"[{self.name}] Disabled '{advisor.name}' for session. "
                        "Call reset() to re-enable."
                    )
                elif severity != ErrorSeverity.TRANSIENT:
                    state.consecutive_failures += 1
        total_ms = (time.monotonic() - start) * 1000
        logger.info(f"[{self.name}] Cascade exhausted ({total_ms:.0f}ms)")
        budget_exhausted = total_ms >= self.cascade_timeout * 1000
        return self._empty_response(path, timing, errors, budget_exhausted)
    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_error(exc: Exception) -> ErrorSeverity:
        s = str(exc).lower()
        if any(k in s for k in ('auth', 'api key', 'unauthorized', '401', '403',
                                 'not found', 'model unavailable', 'connection refused')):
            return ErrorSeverity.CRITICAL
        if any(k in s for k in ('timeout', 'rate limit', '429', '503',
                                 'connection reset', 'temporarily', 'congestion')):
            return ErrorSeverity.TRANSIENT
        return ErrorSeverity.RECOVERABLE
    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------
    @staticmethod
    def _enrich_response(
        response: AdvisorResponse,
        path: List[str],
        timing: Dict[str, float],
        errors: Dict[str, str],
    ) -> AdvisorResponse:
        """Return a new AdvisorResponse — never mutate the original."""
        from dataclasses import replace
        new_meta = dict(response.metadata or {})
        new_meta["cascade"] = {
            "successful_advisor": path[-1] if path else None,
            "path": path,
            "timing_ms": {k: int(v) for k, v in timing.items()},
            "failures": errors,
        }
        return replace(response, metadata=new_meta)
    def _empty_response(
        self,
        path: List[str],
        timing: Dict[str, float],
        errors: Dict[str, str],
        budget_exhausted: bool,
    ) -> AdvisorResponse:
        return AdvisorResponse(
            metadata={
                "advisor": self.name,
                "cascade": {
                    "path": path,
                    "attempted": len(path),
                    "failed": len(errors),
                    "timing_ms": {k: int(v) for k, v in timing.items()},
                    "errors": errors,
                    "budget_exhausted": budget_exhausted,
                    "chain_status": [
                        {
                            "name": s.advisor.name,
                            "disabled": s.disabled,
                            "failures": s.consecutive_failures,
                            "time_ms": int(s.total_time_ms),
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
        """Re-enable all advisors and clear failure counters."""
        super().reset()
        for state in self._chain:
            state.disabled = False
            state.consecutive_failures = 0
            state.total_time_ms = 0.0
            state.last_error = None
            state.last_error_severity = None
            try:
                state.advisor.reset()
            except Exception as exc:
                logger.warning(f"[{self.name}] Failed to reset '{state.advisor.name}': {exc}")
    def get_chain_health(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.advisor.name,
                "disabled": s.disabled,
                "failures": s.consecutive_failures,
                "last_error": s.last_error,
                "last_error_severity": s.last_error_severity.name if s.last_error_severity else None,
                "time_ms": int(s.total_time_ms),
                "requires_network": s.advisor.requires_network,
            }
            for s in self._chain
        ]
    def get_metrics(self) -> Dict[str, Any]:
        return {
            **super().get_metrics(),
            "chain_length": len(self._chain),
            "active": sum(1 for s in self._chain if not s.disabled),
            "disabled": [s.advisor.name for s in self._chain if s.disabled],
            "cascade_timeout": self.cascade_timeout,
            "child_metrics": {s.advisor.name: s.advisor.get_metrics() for s in self._chain},
            "health": self.get_chain_health(),
        }
    def **repr**(self) -> str:
        active = [s.advisor.name for s in self._chain if not s.disabled]
        return (
            f"CascadeAdvisor(name={self.name!r}, "
            f"active={active}, "
            f"budget={self.cascade_timeout}s)"
        )
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_cascade_advisor(
    advisors: List[BaseAdvisor],
    name: str = "cascade-advisor",
    cascade_timeout: float = 60.0,
    critical_confidence_threshold: float = 0.3,
    **kwargs,
) -> CascadeAdvisor:
    return CascadeAdvisor(
        advisors=advisors,
        name=name,
        cascade_timeout=cascade_timeout,
        critical_confidence_threshold=critical_confidence_threshold,
        **kwargs,
    )
