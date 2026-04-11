# keryx/advisors/base.py
# BaseAdvisor interface, AdvisorResponse, AdvisorManager, CascadeAdvisor.
# All components that agent.py, router.py, and orchestrator.py depend on.
# Sovereign, air-gapped, async-aware.
from **future** import annotations
import asyncio
import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger("keryx.advisors")
# ---------------------------------------------------------------------------
# AdvisorResponse — the object every advisor returns
# ---------------------------------------------------------------------------
@dataclass
class AdvisorResponse:
    """
    Structured advice from an Advisor to the Executor.
    agent.py reads:
      - strategic_direction → appended to prompt as ADVISOR GUIDANCE
      - adjust_confidence_threshold → changes agent.confidence_threshold
      - suggested_hypotheses → added to context for prioritization
      - blacklist_hypotheses → marked as false positives, skipped
      - metadata → stored in context, logged, reported
    All fields optional — an advisor can return partial guidance.
    """
    strategic_direction: str = ""
    adjust_confidence_threshold: Optional[float] = None
    suggested_hypotheses: List[str] = field(default_factory=list)
    blacklist_hypotheses: List[str] = field(default_factory=list)
    suggested_tools: List[str] = field(default_factory=list)
    priority_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    def is_empty(self) -> bool:
        """
        True if the response carries no actionable content.
        Used by CascadeAdvisor to decide whether to try the next advisor.
        """
        return (
            not self.strategic_direction.strip()
            and not self.suggested_hypotheses
            and not self.blacklist_hypotheses
            and not self.suggested_tools
            and not self.priority_files
            and self.adjust_confidence_threshold is None
        )
    def **repr**(self) -> str:
        return (
            f"AdvisorResponse("
            f"direction={self.strategic_direction[:60]!r}, "
            f"confirmed={len(self.suggested_hypotheses)}, "
            f"blacklisted={len(self.blacklist_hypotheses)}, "
            f"threshold={self.adjust_confidence_threshold})"
        )
# ---------------------------------------------------------------------------
# BaseAdvisor
# ---------------------------------------------------------------------------
class BaseAdvisor(ABC):
    """
    Abstract base for all KeryxHunter advisors.
    Lifecycle:
        1. AdvisorManager.register(advisor) — register at startup
        2. agent.py calls advisor_manager.can_advise() before escalating
        3. agent.py calls advisor_manager.get_advice(context) — sync, via run_in_executor
        4. AdvisorManager dispatches to the right advisor's .advise(context)
        5. agent.py calls context.add_advisor_advice(response)
        6. agent.py calls _apply_advisor_advice() to act on the response
    Required:
        - should_trigger(context) → bool
        - advise(context) → AdvisorResponse (async)
    Optional:
        - reset()
        - get_metrics()
    """
    # Subclasses must set this
    name: str = "base-advisor"
    requires_network: bool = False # used by router for air-gapped filtering
    def **init**(
        self,
        max_calls_per_session: int = 5,
        cooldown_steps: int = 5,
        **kwargs, # absorb extra kwargs from subclass chains
    ):
        self.max_calls_per_session = max_calls_per_session
        self.cooldown_steps = cooldown_steps
        self.calls_made: int = 0
        self._last_call_step: int = -cooldown_steps # allow first call immediately
        self._total_time_ms: float = 0.0
        self._errors: int = 0
    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def should_trigger(self, context: Any) -> bool:
        """
        Synchronous trigger check. Called before every potential escalation.
        Keep this cheap — it runs in the hot loop.
        """
    @abstractmethod
    async def advise(self, context: Any) -> AdvisorResponse:
        """
        Async advice generation. May call LLMs, tools, etc.
        Must return AdvisorResponse — never raise (handle internally).
        """
    # ------------------------------------------------------------------
    # Public methods used by agent.py and AdvisorManager
    # ------------------------------------------------------------------
    def can_advise(self) -> bool:
        """Check if this advisor has budget remaining for this session."""
        return self.calls_made < self.max_calls_per_session
    async def advise_with_tracking(self, context: Any) -> AdvisorResponse:
        """
        Wrapper around advise() that tracks calls, timing, and errors.
        Called by AdvisorManager and CascadeAdvisor.
        """
        if not self.can_advise():
            logger.warning(f"[{self.name}] Call limit reached ({self.calls_made}/{self.max_calls_per_session})")
            return AdvisorResponse(
                strategic_direction="",
                metadata={"advisor": self.name, "error": "call_limit_reached"},
            )
        start = time.time()
        try:
            response = await self.advise(context)
            self.calls_made += 1
            self._total_time_ms += (time.time() - start) * 1000
            logger.debug(f"[{self.name}] Call #{self.calls_made} | {self._total_time_ms:.0f}ms total")
            return response
        except Exception as exc:
            self._errors += 1
            self.calls_made += 1
            logger.error(f"[{self.name}] advise() raised: {exc}", exc_info=True)
            return AdvisorResponse(
                strategic_direction="",
                metadata={"advisor": self.name, "error": str(exc)},
            )
    async def should_trigger_async(self, context: Any) -> bool:
        """
        Async-safe trigger check.
        If should_trigger() returns a coroutine (subclass went async), awaits it.
        Otherwise wraps sync result.
        """
        result = self.should_trigger(context)
        if inspect.iscoroutine(result):
            return await result
        return bool(result)
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-session state. Called between hunts."""
        self.calls_made = 0
        self._last_call_step = -self.cooldown_steps
        self._errors = 0
        self._total_time_ms = 0.0
    def get_metrics(self) -> Dict[str, Any]:
        avg_ms = self._total_time_ms / self.calls_made if self.calls_made > 0 else 0.0
        return {
            "name": self.name,
            "calls_made": self.calls_made,
            "max_calls_per_session": self.max_calls_per_session,
            "errors": self._errors,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": avg_ms,
            "requires_network": self.requires_network,
        }
    def **repr**(self) -> str:
        return f"{self.**class**.**name**}(name={self.name!r}, calls={self.calls_made}/{self.max_calls_per_session})"
# ---------------------------------------------------------------------------
# AdvisorManager — registry used by agent.py and router.py
# ---------------------------------------------------------------------------
class AdvisorManager:
    """
    Registry and dispatcher for all advisors.
    agent.py uses:
        manager.can_advise()
        manager.get_advice(context) ← sync, called via run_in_executor
    router.py uses:
        manager.has_advisor(name)
        manager.get_advisor(name)
    orchestrator.py reads:
        manager.calls_made
    """
    def **init**(self, default_advisor_name: Optional[str] = None):
        self._advisors: Dict[str, BaseAdvisor] = {}
        self._default: Optional[str] = default_advisor_name
        self.calls_made: int = 0
    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, advisor: BaseAdvisor) -> None:
        self._advisors[advisor.name] = advisor
        logger.info(f"[AdvisorManager] Registered: {advisor.name}")
        if self._default is None:
            self._default = advisor.name
    def set_default(self, name: str) -> None:
        if name not in self._advisors:
            raise KeyError(f"Advisor '{name}' not registered")
        self._default = name
    # ------------------------------------------------------------------
    # Lookup (used by router.py)
    # ------------------------------------------------------------------
    def has_advisor(self, name: str) -> bool:
        return name in self._advisors
    def get_advisor(self, name: str) -> Optional[BaseAdvisor]:
        """Return advisor by name, or None if not registered."""
        return self._advisors.get(name)
    # ------------------------------------------------------------------
    # Dispatch (used by agent.py via run_in_executor → sync wrapper)
    # ------------------------------------------------------------------
    def can_advise(self) -> bool:
        """True if ANY registered advisor can still provide advice."""
        return any(a.can_advise() for a in self._advisors.values())
    def get_advice(self, context: Any, advisor_name: Optional[str] = None) -> AdvisorResponse:
        """
        Synchronous advice dispatch.
        agent.py calls this inside run_in_executor:
            advice = await loop.run_in_executor(None, manager.get_advice, context)
        Internally runs the async advise_with_tracking() in a new event loop.
        This is intentional: agent.py controls its own event loop and the
        advisor runs in a thread pool, so creating a fresh loop here is correct.
        """
        name = advisor_name or self._default
        if not name or name not in self._advisors:
            logger.warning(f"[AdvisorManager] No advisor found: {name!r}")
            return AdvisorResponse(strategic_direction="", metadata={"error": "no_advisor"})
        advisor = self._advisors[name]
        if not advisor.can_advise():
            logger.warning(f"[AdvisorManager] Advisor '{name}' at call limit")
            return AdvisorResponse(strategic_direction="", metadata={"advisor": name, "error": "limit"})
        # Run async advise in a fresh event loop (we're in a thread pool)
        try:
            loop = asyncio.new_event_loop()
            try:
                response = loop.run_until_complete(advisor.advise_with_tracking(context))
            finally:
                loop.close()
            self.calls_made += 1
            return response
        except Exception as exc:
            logger.error(f"[AdvisorManager] get_advice failed: {exc}", exc_info=True)
            return AdvisorResponse(strategic_direction="", metadata={"error": str(exc)})
    # ------------------------------------------------------------------
    # Bulk ops
    # ------------------------------------------------------------------
    def reset_all(self) -> None:
        """Reset all advisors between hunts."""
        for advisor in self._advisors.values():
            advisor.reset()
        self.calls_made = 0
    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            "total_calls": self.calls_made,
            "advisors": {name: a.get_metrics() for name, a in self._advisors.items()},
        }
    def **repr**(self) -> str:
        names = list(self._advisors.keys())
        return f"AdvisorManager(advisors={names}, default={self._default!r}, calls={self.calls_made})"
# ---------------------------------------------------------------------------
# CascadeAdvisor — chains advisors, tries in order until one responds
# ---------------------------------------------------------------------------
class CascadeAdvisor(BaseAdvisor):
    """
    Chains multiple advisors in priority order.
    Tries each advisor; uses the first non-empty response.
    If all pass or return empty — returns empty AdvisorResponse.
    Design:
    - Does NOT persist _current_index between calls.
      Each call starts from advisor[0] unless a previous response was non-empty.
      Rationale: conditions change between steps; skipping advisors permanently
      means missing a cheaper advisor that became relevant again.
    - Uses should_trigger_async() for correct coroutine handling.
    - advise_with_tracking() provides call counting and error capture.
    """
    name = "cascade-advisor"
    def **init**(
        self,
        advisors: List[BaseAdvisor],
        name: str = "cascade-advisor",
        max_calls_per_session: int = 5,
        **kwargs,
    ):
        if not advisors:
            raise ValueError("CascadeAdvisor requires at least one advisor in chain")
        super().**init**(max_calls_per_session=max_calls_per_session, **kwargs)
        self.name = name
        self._advisors = advisors
        logger.info(
            f"[{self.name}] Initialized | chain=[{', '.join(a.name for a in advisors)}]"
        )
    def should_trigger(self, context: Any) -> bool:
        """
        Trigger if ANY advisor in the chain says yes.
        FIX: Previous version used hasattr(result, '**await**') to detect coroutines.
        inspect.iscoroutine() is the correct check. But should_trigger() must be sync
        per the BaseAdvisor contract — we run async checks in advise() instead.
        Here we conservatively return True if any sync trigger fires.
        """
        for advisor in self._advisors:
            result = advisor.should_trigger(context)
            # Guard: subclass may have accidentally made should_trigger async
            if inspect.iscoroutine(result):
                result.close() # prevent "coroutine never awaited" warning
                return True # conservative: assume yes, let advise() filter
            if result:
                return True
        return False
    async def advise(self, context: Any) -> AdvisorResponse:
        """
        Try advisors in order. Return first non-empty response.
        FIX: _current_index removed — each call starts fresh.
        FIX: advise_with_tracking() and should_trigger_async() now defined in BaseAdvisor.
        FIX: response.is_empty() now defined on AdvisorResponse.
        """
        for advisor in self._advisors:
            if not advisor.can_advise():
                logger.debug(f"[{self.name}] Skipping '{advisor.name}' — at call limit")
                continue
            try:
                triggered = await advisor.should_trigger_async(context)
            except Exception as exc:
                logger.warning(f"[{self.name}] should_trigger_async failed for '{advisor.name}': {exc}")
                triggered = False
            if not triggered:
                logger.debug(f"[{self.name}] '{advisor.name}' did not trigger")
                continue
            logger.info(f"[{self.name}] Trying advisor: {advisor.name}")
            response = await advisor.advise_with_tracking(context)
            if not response.is_empty():
                logger.info(f"[{self.name}] Got non-empty response from '{advisor.name}'")
                return response
            logger.debug(f"[{self.name}] '{advisor.name}' returned empty — trying next")
        logger.info(f"[{self.name}] No advisor produced advice")
        return AdvisorResponse(
            strategic_direction="",
            metadata={"cascade": "no_advisor_triggered", "chain": [a.name for a in self._advisors]},
        )
    def reset(self) -> None:
        """Reset cascade and all child advisors."""
        super().reset()
        for advisor in self._advisors:
            advisor.reset()
    def get_metrics(self) -> Dict[str, Any]:
        return {
            **super().get_metrics(),
            "chain": [a.name for a in self._advisors],
            "child_metrics": {a.name: a.get_metrics() for a in self._advisors},
        }
# ---------------------------------------------------------------------------
# RuleBasedAdvisor — simple threshold-based advisor, useful as fallback
# ---------------------------------------------------------------------------
class RuleBasedAdvisor(BaseAdvisor):
    """
    Deterministic advisor: fires when confidence drops below threshold
    or parse errors accumulate. No LLM required — zero cost, always available.
    Use as the first advisor in a CascadeAdvisor chain.
    """
    name = "rule-based-advisor"
    requires_network = False
    def **init**(
        self,
        confidence_threshold: float = 0.5,
        max_parse_errors: int = 3,
        no_progress_after_steps: int = 15,
        **kwargs,
    ):
        super().**init**(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.max_parse_errors = max_parse_errors
        self.no_progress_after_steps = no_progress_after_steps
    def should_trigger(self, context: Any) -> bool:
        last_conf = getattr(context, 'get_last_confidence', lambda: None)()
        low_conf = last_conf is not None and last_conf < self.confidence_threshold
        errors = getattr(context, 'parse_errors', 0) >= self.max_parse_errors
        steps = getattr(context, 'steps_taken', 0)
        hyps = getattr(context, 'hypotheses', [])
        stalled = steps > self.no_progress_after_steps and not hyps
        return low_conf or errors or stalled
    async def advise(self, context: Any) -> AdvisorResponse:
        last_conf = getattr(context, 'get_last_confidence', lambda: None)()
        parse_errs = getattr(context, 'parse_errors', 0)
        steps = getattr(context, 'steps_taken', 0)
        hyps = getattr(context, 'hypotheses', [])
        parts: List[str] = []
        adj_threshold = None
        if last_conf is not None and last_conf < self.confidence_threshold:
            parts.append(
                f"Confidence is low ({last_conf:.2f}). "
                "Broaden search: try adjacent modules and IPC boundaries."
            )
            adj_threshold = max(0.4, self.confidence_threshold - 0.1)
        if parse_errs >= self.max_parse_errors:
            parts.append(
                f"Model produced {parse_errs} parse errors. "
                "Simplify the prompt or switch to a smaller grammar."
            )
        if steps > self.no_progress_after_steps and not hyps:
            parts.append(
                "No hypotheses after many steps. "
                "Focus on memory management code: allocators, GC roots, IPC serialization."
            )
        return AdvisorResponse(
            strategic_direction=" ".join(parts) if parts else "Rules check passed.",
            adjust_confidence_threshold=adj_threshold,
            metadata={"advisor": self.name, "rules_fired": len(parts)},
        )
