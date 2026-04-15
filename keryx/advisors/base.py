# keryx/advisors/base.py
# BaseAdvisor, AdvisorResponse, CascadeAdvisor, RuleBasedAdvisor.
# AdvisorManager lives in manager.py (production dispatch engine).
# FIX 8: removed duplicate AdvisorManager from this file.

from __future__ import annotations

import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("keryx.advisors")


# ---------------------------------------------------------------------------
# AdvisorResponse
# ---------------------------------------------------------------------------
@dataclass
class AdvisorResponse:
    """
    Structured advice from an Advisor to the Executor.
    agent.py reads:
      strategic_direction → injected into next prompt
      adjust_confidence_threshold → changes agent.confidence_threshold
      suggested_hypotheses → prioritized in context
      blacklist_hypotheses → marked as false positive, skipped
      suggested_tools → executor tries these tools next
      priority_files → executor focuses on these files
    All fields optional — an advisor returns only what it knows.
    """
    strategic_direction: str = ""
    adjust_confidence_threshold: float | None = None
    suggested_hypotheses: list[str] = field(default_factory=list)
    blacklist_hypotheses: list[str] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)
    priority_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict compatible with AdvisorAdvice.from_dict()."""
        return {
            "strategy":                    self.strategic_direction,
            "strategic_direction":         self.strategic_direction,
            "adjust_confidence_threshold": self.adjust_confidence_threshold,
            "suggested_hypotheses":        self.suggested_hypotheses,
            "blacklist_hypotheses":        self.blacklist_hypotheses,
            "suggested_tools":             self.suggested_tools,
            "priority_files":              self.priority_files,
            **self.metadata,
        }

    def is_empty(self) -> bool:
        """
        True if no actionable content.
        CascadeAdvisor uses this to decide whether to try the next advisor.
        """
        return (
            not self.strategic_direction.strip()
            and not self.suggested_hypotheses
            and not self.blacklist_hypotheses
            and not self.suggested_tools
            and not self.priority_files
            and self.adjust_confidence_threshold is None
        )

    def __repr__(self) -> str:
        return (
            f"AdvisorResponse("
            f"direction={self.strategic_direction[:60]!r}, "
            f"suggested={len(self.suggested_hypotheses)}, "
            f"blacklisted={len(self.blacklist_hypotheses)}, "
            f"threshold={self.adjust_confidence_threshold})"
        )


# ---------------------------------------------------------------------------
# BaseAdvisor
# ---------------------------------------------------------------------------
class BaseAdvisor(ABC):
    """
    Abstract base for all KeryxHunter advisors.
    Required interface:
        should_trigger(context) → bool — cheap sync check
        advise(context) → AdvisorResponse — async, may call LLMs
    Provided by base:
        can_advise() → bool
        advise_with_tracking(context) → AdvisorResponse
        should_trigger_async(context) → bool
        reset()
        get_metrics() → dict
    Class attributes subclasses must set:
        name: str = "my-advisor"
        requires_network: bool = False (True for cloud advisors)
    """
    name: str = "base-advisor"
    requires_network: bool = False

    def __init__(
        self,
        max_calls_per_session: int = 5,
        cooldown_steps: int = 5,
        **kwargs: Any,
    ):
        self.max_calls_per_session = max_calls_per_session
        self.cooldown_steps = cooldown_steps
        self.calls_made: int = 0
        self._total_time_ms: float = 0.0
        self._errors: int = 0

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    @abstractmethod
    def should_trigger(self, context: Any) -> bool:
        """Cheap sync trigger check. Runs in the hot loop — keep it fast."""

    @abstractmethod
    async def advise(self, context: Any) -> AdvisorResponse:
        """
        Async advice generation. May call LLMs, tools, etc.
        Must not raise — catch internally and return AdvisorResponse with error metadata.
        """

    # ------------------------------------------------------------------
    # Provided methods
    # ------------------------------------------------------------------
    def can_advise(self) -> bool:
        return self.calls_made < self.max_calls_per_session

    async def advise_with_tracking(self, context: Any) -> AdvisorResponse:
        """Wraps advise() with call counting, timing, and exception capture."""
        if not self.can_advise():
            logger.warning(
                f"[{self.name}] Call limit reached ({self.calls_made}/{self.max_calls_per_session})"
            )
            return AdvisorResponse(metadata={"advisor": self.name, "error": "call_limit_reached"})

        start = time.time()
        try:
            response = await self.advise(context)
            self.calls_made += 1
            self._total_time_ms += (time.time() - start) * 1000
            return response
        except Exception as exc:
            self._errors += 1
            self.calls_made += 1
            self._total_time_ms += (time.time() - start) * 1000
            logger.error(f"[{self.name}] advise() raised: {exc}", exc_info=True)
            return AdvisorResponse(metadata={"advisor": self.name, "error": str(exc)})

    async def should_trigger_async(self, context: Any) -> bool:
        """
        Async-safe trigger check.
        Uses inspect.iscoroutine() — correct detection vs hasattr(__await__).
        """
        result = self.should_trigger(context)
        if inspect.iscoroutine(result):
            return await result
        return bool(result)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.calls_made = 0
        self._errors = 0
        self._total_time_ms = 0.0

    def get_metrics(self) -> dict[str, Any]:
        avg = self._total_time_ms / self.calls_made if self.calls_made else 0.0
        return {
            "name": self.name,
            "calls_made": self.calls_made,
            "max_calls_per_session": self.max_calls_per_session,
            "errors": self._errors,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_time_ms": round(avg, 1),
            "requires_network": self.requires_network,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"calls={self.calls_made}/{self.max_calls_per_session})"
        )


# ---------------------------------------------------------------------------
# CascadeAdvisor
# ---------------------------------------------------------------------------
class CascadeAdvisor(BaseAdvisor):
    """
    Chains advisors in priority order.
    Tries each; returns the first non-empty response.
    Fixes vs original:
    - No _current_index: starts fresh each call (conditions change between steps)
    - advise_with_tracking() and should_trigger_async() provided by BaseAdvisor
    - response.is_empty() provided by AdvisorResponse
    - inspect.iscoroutine() instead of hasattr(__await__)
    """
    name = "cascade-advisor"

    def __init__(
        self,
        advisors: list[BaseAdvisor],
        name: str = "cascade-advisor",
        max_calls_per_session: int = 5,
        **kwargs: Any,
    ):
        if not advisors:
            raise ValueError("CascadeAdvisor requires at least one advisor in chain")
        super().__init__(max_calls_per_session=max_calls_per_session, **kwargs)
        self.name = name
        self._advisors = advisors
        logger.info(
            f"[{self.name}] chain=[{', '.join(a.name for a in advisors)}]"
        )

    def should_trigger(self, context: Any) -> bool:
        """Trigger if any sync advisor in chain says yes."""
        for advisor in self._advisors:
            result = advisor.should_trigger(context)
            if inspect.iscoroutine(result):
                result.close()  # avoid "never awaited" warning
                return True  # conservative: assume yes
            if result:
                return True
        return False

    async def advise(self, context: Any) -> AdvisorResponse:
        """Try advisors in order; return first non-empty response."""
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
                continue

            logger.info(f"[{self.name}] Trying: {advisor.name}")
            response = await advisor.advise_with_tracking(context)

            if not response.is_empty():
                logger.info(f"[{self.name}] Got advice from '{advisor.name}'")
                return response

            logger.debug(f"[{self.name}] '{advisor.name}' returned empty — next")

        return AdvisorResponse(
            metadata={"cascade": "no_advisor_triggered",
                      "chain": [a.name for a in self._advisors]},
        )

    def reset(self) -> None:
        super().reset()
        for advisor in self._advisors:
            advisor.reset()

    def get_metrics(self) -> dict[str, Any]:
        return {
            **super().get_metrics(),
            "chain": [a.name for a in self._advisors],
            "child_metrics": {a.name: a.get_metrics() for a in self._advisors},
        }


# ---------------------------------------------------------------------------
# RuleBasedAdvisor — zero cost, no LLM, use first in cascade
# ---------------------------------------------------------------------------
class RuleBasedAdvisor(BaseAdvisor):
    """
    Deterministic advisor based on context thresholds.
    No LLM required — always available, zero cost.
    Recommended as advisor[0] in every CascadeAdvisor chain.
    """
    name = "rule-based-advisor"
    requires_network = False

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_parse_errors: int = 3,
        no_progress_after_steps: int = 15,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.max_parse_errors = max_parse_errors
        self.no_progress_after_steps = no_progress_after_steps

    @staticmethod
    def _has_unconfirmed_high(context: Any) -> bool:
        """True if recent steps contain codeql HIGH findings but no vuln is confirmed yet."""
        if getattr(context, "confirmed_vulns", []):
            return False
        steps = list(getattr(context, "steps", []))[-3:]
        return any(
            "[HIGH]" in str(s.get("observation", ""))
            for s in steps
        )

    def should_trigger(self, context: Any) -> bool:
        conf = getattr(context, 'get_last_confidence', lambda: None)()
        errors = getattr(context, 'parse_errors', 0)
        steps = getattr(context, 'steps_taken', 0)
        hyps = getattr(context, 'hypotheses', [])

        return (
            (conf is not None and conf < self.confidence_threshold)
            or errors >= self.max_parse_errors
            or (steps > self.no_progress_after_steps and not hyps)
            or self._has_unconfirmed_high(context)   # always advise on unverified HIGH
        )

    async def advise(self, context: Any) -> AdvisorResponse:
        conf = getattr(context, 'get_last_confidence', lambda: None)()
        errors = getattr(context, 'parse_errors', 0)
        steps = getattr(context, 'steps_taken', 0)
        hyps = getattr(context, 'hypotheses', [])

        parts: list[str] = []
        adj = None
        suggested_tools: list[str] = []

        if self._has_unconfirmed_high(context):
            parts.append(
                "codeql_query found HIGH security findings. "
                "You must run injection_verifier on the vulnerable field before finishing. "
                "Example: inject 'author' with payload '--upload-pack=test' and "
                "expected_behavior 'reject'. "
                "If the target tool is not git_blame, use the actual tool name and "
                "the field name from the GIT_OPTION_INJECTION finding."
            )
            suggested_tools.append("injection_verifier")

        if conf is not None and conf < self.confidence_threshold:
            parts.append(
                f"Confidence low ({conf:.2f}). "
                "Focus on subprocess/shell call boundaries and parameter sanitization. "
                "Run codeql_query on the full target file to surface all findings at once."
            )
            adj = max(0.4, self.confidence_threshold - 0.1)

        if errors >= self.max_parse_errors:
            parts.append(
                f"Model produced {errors} parse errors. "
                "Use simpler action_input — fewer optional fields, plain strings only."
            )

        if steps > self.no_progress_after_steps and not hyps:
            parts.append(
                "No hypotheses after many steps. "
                "Pivot: run codeql_query to enumerate all security findings, "
                "then use git_blame to inspect commit history for recent security-relevant changes."
            )

        return AdvisorResponse(
            strategic_direction=" ".join(parts) if parts else "Rules check passed.",
            adjust_confidence_threshold=adj,
            suggested_tools=suggested_tools,
            metadata={"advisor": self.name, "rules_fired": len(parts)},
        )
