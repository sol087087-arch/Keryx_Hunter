# keryx/core/shared_context.py
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Evidence:
    """A piece of evidence supporting a vulnerability hypothesis."""
    location:    str                         # e.g. "js/src/vm/JSObject.cpp:1847"
    description: str
    tags:        list[str]          = field(default_factory=list)
    timestamp:   float              = field(default_factory=time.time)
    metadata:    dict[str, Any]     = field(default_factory=dict)


@dataclass
class AdvisorAdvice:
    """
    Structured advice from the Advisor layer.
    Stored as a typed dataclass so agent.py can access fields as attributes
    without AttributeError (previously stored as bare Dict).
    """
    strategy:                    str            = ""
    strategic_direction:         str            = ""
    adjust_confidence_threshold: float | None = None
    raw:                         dict[str, Any]  = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdvisorAdvice:
        return cls(
            strategy=                    d.get("strategy", ""),
            strategic_direction=         d.get("strategic_direction", ""),
            adjust_confidence_threshold= d.get("adjust_confidence_threshold"),
            raw=                         d,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy":                    self.strategy,
            "strategic_direction":         self.strategic_direction,
            "adjust_confidence_threshold": self.adjust_confidence_threshold,
        }


@dataclass
class SharedContext:
    """
    Single source of truth for the entire hunt.
    Used by Executor, Advisor, and Orchestrator.

    All methods expected by agent.py are implemented here — no stubs.
    """
    target_path: str
    capability:  str = "deep_reasoning"
    mode:        str = "hybrid"

    # Orchestrator escalation state — serialised so resume works correctly.
    escalation_level: int = 1
    # routing_plan is a RoutingPlan dataclass; not serialised (too complex).
    routing_plan: Any | None = field(default=None, repr=False)

    # Core state
    # FIX 7: steps_taken is managed ONLY by add_step() — agent.py must not
    # increment it directly.  The field is still public for checkpoint restore.
    steps_taken:     int                    = 0
    steps:           list[dict[str, Any]]   = field(default_factory=list)
    hypotheses:      set[str]               = field(default_factory=set)
    blacklist:       set[str]               = field(default_factory=set)
    confirmed_vulns: list[dict[str, Any]]   = field(default_factory=list)
    scanned_files:   list[str]              = field(default_factory=list)
    parse_errors:    int                    = 0

    # Advisor state
    advisor_advice:         list[AdvisorAdvice]      = field(default_factory=list)
    pending_advisor_advice: AdvisorAdvice | None  = None
    _advisor_guidance:      str                       = field(default="", init=False)

    # Critique log
    _critiques: list[str] = field(default_factory=list, init=False)

    # Evidence
    evidence: list[Evidence] = field(default_factory=list)

    # Runtime metrics (not serialised — reset on resume is fine)
    _start_time: float = field(default_factory=time.time, init=False)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, action: Any, observation: str) -> None:
        """
        Record a completed step.
        FIX 7: this is the ONLY place steps_taken is incremented.
        agent.py must not do `self.context.steps_taken += 1` separately.

        FIX SERIALIZATION: AgentStep dataclass objects are not JSON-serialisable.
        Convert to a plain dict here so that to_dict() / checkpoint write never
        hits 'TypeError: Object of type AgentStep is not JSON serializable'.
        Readers (get_recent_history, get_last_confidence) use dict key access.
        """
        if hasattr(action, "action"):
            action_record: Any = {
                "name":       getattr(action, "action", str(action)),
                "input":      getattr(action, "action_input", {}),
                "thought":    getattr(action, "thought", ""),
                "confidence": getattr(action, "confidence", 0.0),
            }
        else:
            action_record = str(action)
        self.steps.append({
            "timestamp":   time.time(),
            "action":      action_record,
            "observation": observation[:1000],
        })
        self.steps_taken += 1

    def get_last_step(self) -> dict[str, Any] | None:
        """FIX 2: was missing — called by agent._self_critique_async()."""
        return self.steps[-1] if self.steps else None

    def get_recent_history(self, n: int = 5) -> str:
        """
        FIX 10: was a stub returning a fixed string.
        Now returns a real formatted summary of the last n steps.
        """
        recent = self.steps[-n:]
        if not recent:
            return "No steps taken yet."
        lines: list[str] = []
        for i, s in enumerate(recent, 1):
            action      = s.get("action", {})
            # action is now always a dict (stored by add_step) or str fallback
            if isinstance(action, dict):
                action_name = action.get("name", str(action))
                confidence  = action.get("confidence")
            else:
                action_name = str(action)
                confidence  = None
            observation = s.get("observation", "")
            conf_str    = f" | conf={confidence:.2f}" if confidence is not None else ""
            lines.append(
                f"  [{i}] {action_name}{conf_str}\n"
                f"      obs: {observation[:200]}{'...' if len(observation) > 200 else ''}"
            )
        return "\n".join(lines)

    def get_last_confidence(self) -> float | None:
        """
        FIX 9: was a stub returning None — escalation logic never fired.
        Now reads confidence from the last recorded step dict.
        """
        last = self.get_last_step()
        if last is None:
            return None
        action = last.get("action")
        if isinstance(action, dict):
            return action.get("confidence")
        return None

    def trim_history(self, keep: int = 6) -> None:
        """
        FIX 6: was missing — called by agent when prompt exceeds max_prompt_chars.
        Keeps the most recent `keep` steps; older ones are dropped from RAM
        but steps_taken is preserved so step numbering stays correct.
        """
        if len(self.steps) > keep:
            self.steps = self.steps[-keep:]

    # ------------------------------------------------------------------
    # Hypothesis management
    # ------------------------------------------------------------------

    def add_hypothesis(self, text: str) -> None:
        if text and text not in self.blacklist:
            self.hypotheses.add(text)

    def blacklist_hypothesis(self, text: str) -> None:
        self.blacklist.add(text)
        self.hypotheses.discard(text)

    def get_deduplicated_hypotheses(self) -> str:
        return "\n".join(sorted(self.hypotheses)) if self.hypotheses else "No hypotheses yet."

    # ------------------------------------------------------------------
    # Parse error tracking
    # ------------------------------------------------------------------

    def increment_parse_errors(self) -> None:
        """FIX 1: was missing — called 3 times in agent.py."""
        self.parse_errors += 1

    def set_escalation_level(self, level: int) -> None:
        """Called by KeryxOrchestrator when progressive escalation advances."""
        self.escalation_level = level

    # ------------------------------------------------------------------
    # Evidence and critique
    # ------------------------------------------------------------------

    def add_evidence(
        self,
        location:    str,
        description: str,
        tags:        list[str] | None = None,
    ) -> None:
        self.evidence.append(Evidence(location, description, tags or []))

    def request_additional_evidence(self, action_name: str) -> None:
        """
        FIX 3: was missing — called by agent._apply_critique().
        Records that additional evidence is needed for this action path.
        Simple implementation: adds a tagged evidence placeholder.
        """
        self.evidence.append(Evidence(
            location=    action_name,
            description= "Additional evidence requested by self-critique.",
            tags=        ["needs_evidence"],
        ))

    def add_critique(self, critique: str) -> None:
        """FIX 4: was missing — called by agent._apply_critique()."""
        self._critiques.append(critique)

    # ------------------------------------------------------------------
    # Advisor interface
    # ------------------------------------------------------------------

    def add_advisor_advice(self, advice: Any) -> None:
        """
        FIX 8: normalise incoming advice to AdvisorAdvice dataclass so
        agent.py can access advice.strategy etc. as attributes, not dict keys.
        Accepts AdvisorAdvice, dict, or any object with a to_dict() method.
        """
        if isinstance(advice, AdvisorAdvice):
            typed = advice
        elif isinstance(advice, dict):
            typed = AdvisorAdvice.from_dict(advice)
        elif hasattr(advice, "to_dict"):
            typed = AdvisorAdvice.from_dict(advice.to_dict())
        else:
            typed = AdvisorAdvice(strategy=str(advice))

        self.advisor_advice.append(typed)
        self.pending_advisor_advice = typed

    def get_pending_advisor_advice(self) -> AdvisorAdvice | None:
        return self.pending_advisor_advice

    def has_pending_advisor_advice(self) -> bool:
        return self.pending_advisor_advice is not None

    def clear_pending_advisor_advice(self) -> None:
        self.pending_advisor_advice = None

    def set_advisor_guidance(self, guidance: str) -> None:
        """FIX 5a: was missing — called by agent._apply_advisor_advice()."""
        self._advisor_guidance = guidance or ""

    def get_advisor_guidance(self) -> str:
        """FIX 5b: was missing — called by agent._build_prompt_with_guidance()."""
        return self._advisor_guidance

    # ------------------------------------------------------------------
    # Summary / metrics
    # ------------------------------------------------------------------

    def build_summary(self) -> str:
        runtime = round(time.time() - self._start_time, 1)
        return (
            f"Target:                  {self.target_path}\n"
            f"Steps taken:             {self.steps_taken}\n"
            f"Active hypotheses:       {len(self.hypotheses)}\n"
            f"Confirmed vulns:         {len(self.confirmed_vulns)}\n"
            f"Evidence collected:      {len(self.evidence)}\n"
            f"Parse errors:            {self.parse_errors}\n"
            f"Runtime:                 {runtime}s"
        )

    def get_metrics(self) -> dict[str, Any]:
        return {
            "runtime_sec":        round(time.time() - self._start_time, 2),
            "steps_count":        len(self.steps),
            "evidence_count":     len(self.evidence),
            "hypotheses_active":  len(self.hypotheses),
            "confirmed_vulns":    len(self.confirmed_vulns),
            "parse_errors":       self.parse_errors,
        }

    # ------------------------------------------------------------------
    # JSON serialisation (replaces pickle — no arbitrary code execution)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_path":       self.target_path,
            "capability":        self.capability,
            "mode":              self.mode,
            "escalation_level":  self.escalation_level,
            "steps_taken":       self.steps_taken,
            "steps":          self.steps,
            "hypotheses":     list(self.hypotheses),
            "blacklist":      list(self.blacklist),
            "confirmed_vulns":self.confirmed_vulns,
            "scanned_files":  self.scanned_files,
            "parse_errors":   self.parse_errors,
            "advisor_advice": [a.to_dict() for a in self.advisor_advice],
            "advisor_guidance":self._advisor_guidance,
            "critiques":      self._critiques,
            "evidence":       [asdict(e) for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedContext:
        ctx = cls(
            target_path=      data.get("target_path", ""),
            capability=       data.get("capability", "deep_reasoning"),
            mode=             data.get("mode", "hybrid"),
            escalation_level= data.get("escalation_level", 1),
        )
        ctx.steps_taken     = data.get("steps_taken", 0)
        ctx.steps           = data.get("steps", [])
        ctx.hypotheses      = set(data.get("hypotheses", []))
        ctx.blacklist       = set(data.get("blacklist", []))
        ctx.confirmed_vulns = data.get("confirmed_vulns", [])
        ctx.scanned_files   = data.get("scanned_files", [])
        ctx.parse_errors    = data.get("parse_errors", 0)
        ctx.advisor_advice  = [
            AdvisorAdvice.from_dict(a) for a in data.get("advisor_advice", [])
        ]
        ctx._advisor_guidance = data.get("advisor_guidance", "")
        ctx._critiques        = data.get("critiques", [])
        ctx.evidence          = [Evidence(**e) for e in data.get("evidence", [])]
        return ctx


__all__ = ["SharedContext", "Evidence", "AdvisorAdvice"]
