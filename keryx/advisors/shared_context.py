# keryx/core/shared_context.py
# SharedContext — hardened single source of truth.
# Anti-loop protection, cost escalation, evidence memory, observation truncation.

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("keryx.context")


# ---------------------------------------------------------------------------
# ContextStep
# ---------------------------------------------------------------------------
@dataclass
class ContextStep:
    step_num: int
    action: str
    action_input: dict[str, Any]
    observation: str
    thought: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    needs_evidence: str | None = None
    is_essential: bool = False
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_num": self.step_num,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "thought": self.thought,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "needs_evidence": self.needs_evidence,
            "is_essential": self.is_essential,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContextStep:
        return cls(
            step_num=d["step_num"],
            action=d["action"],
            action_input=d.get("action_input", {}),
            observation=d.get("observation", ""),
            thought=d.get("thought", ""),
            confidence=d.get("confidence", 0.0),
            timestamp=d.get("timestamp", 0.0),
            needs_evidence=d.get("needs_evidence"),
            is_essential=d.get("is_essential", False),
            error_count=d.get("error_count", 0),
        )

    def to_text(self, verbose: bool = True) -> str:
        if verbose:
            parts = [
                f"Step {self.step_num} | {self.action} | conf={self.confidence:.2f}",
                f" Thought: {self.thought[:120]}",
                f" Observation: {self.observation[:200]}",
            ]
            if self.needs_evidence:
                parts.append(f" [NEEDS_EVIDENCE: {self.needs_evidence}]")
            if self.error_count > 0:
                parts.append(f" [ERRORS: {self.error_count}]")
            return "\n".join(parts)
        return f"Step {self.step_num}: {self.action} → {self.thought[:60]}..."


# ---------------------------------------------------------------------------
# SharedContext
# ---------------------------------------------------------------------------
_DEFAULT_MAX_OBSERVATION = 5000
_DEFAULT_MAX_HISTORY = 100
_DEFAULT_MAX_CRITIQUES = 50
_DEFAULT_LOOP_THRESHOLD = 3
_DEFAULT_COST_THRESHOLD = 2.0


class SharedContext:
    """
    Hardened shared hunt state.
    Step counter ownership: steps_taken is incremented ONLY inside add_step().
    Advisor advice lifecycle uses explicit consume/peek API.
    Stupid loop detection is per-action-signature.
    """
    def __init__(
        self,
        target_path: str,
        capability: str = "deep_reasoning",
        mode: str = "hybrid",
        routing_plan: Any = None,
        max_observation_size: int = _DEFAULT_MAX_OBSERVATION,
        max_history_steps: int = _DEFAULT_MAX_HISTORY,
        max_critiques: int = _DEFAULT_MAX_CRITIQUES,
        loop_threshold: int = _DEFAULT_LOOP_THRESHOLD,
        cost_escalation_usd: float = _DEFAULT_COST_THRESHOLD,
    ):
        self.target_path = target_path
        self.capability = capability
        self.mode = mode
        self.routing_plan = routing_plan
        self.created_at = time.time()

        # Thresholds
        self._max_obs = max_observation_size
        self._max_history = max_history_steps
        self._max_critiques = max_critiques
        self._loop_threshold = loop_threshold
        self._cost_threshold = cost_escalation_usd

        # Scan state
        self.steps_taken: int = 0
        self.parse_errors: int = 0
        self.scanned_files: set[str] = set()
        self.hypotheses: list[str] = []
        self.confirmed_vulns: list[dict] = []
        self.blacklisted_hypotheses: list[str] = []
        self.failed_tools: list[str] = []
        self.escalation_level: int = 1

        # Cost tracking
        self.session_cost_usd: float = 0.0
        self._cost_by_file: dict[str, float] = defaultdict(float)

        # History
        self._steps: deque = deque(maxlen=max_history_steps)
        self._critiques: deque = deque(maxlen=max_critiques)

        # Per-action failure counters
        self._action_failure_counts: dict[str, int] = defaultdict(int)
        self._stupid_loop_detected: bool = False

        # Evidence memory
        self._evidence_map: dict[str, str] = {}
        self._evidence_index: dict[str, list[str]] = defaultdict(list)

        # Advisor state
        self._pending_advisor_advice: Any | None = None
        self._advisor_guidance: str = ""

        # Deduplication sets
        self._hypothesis_set: set[str] = set()
        self._blacklist_set: set[str] = set()

        logger.info(f"[SharedContext] Initialized | target={target_path}")

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------
    def add_step(self, action: Any, observation: str) -> None:
        if isinstance(action, dict):
            name = action.get("action", "unknown")
            inp = action.get("action_input", {})
            thought = action.get("thought", "")
            confidence = float(action.get("confidence", 0.0))
        else:
            name = getattr(action, "action", str(action))
            inp = getattr(action, "action_input", {})
            thought = getattr(action, "thought", "")
            confidence = float(getattr(action, "confidence", 0.0))

        # Truncate oversized observations
        if len(observation) > self._max_obs:
            observation = observation[:self._max_obs] + "... [TRUNCATED]"

        is_essential = any(s in observation for s in (
            "VULN_CONFIRMED", "CRITICAL_PATH", "heap-use-after-free",
            "AddressSanitizer", "stack-buffer-overflow",
        ))

        # Per-signature failure counter
        sig = f"{name}:{json.dumps(inp, sort_keys=True, default=str)}"
        is_error = any(kw in observation.lower() for kw in ("error", "fail", "timeout", "crash"))

        if is_error:
            self._action_failure_counts[sig] += 1
            error_count = self._action_failure_counts[sig]
            if error_count >= self._loop_threshold:
                self._stupid_loop_detected = True
                logger.warning(
                    f"[SharedContext] Stupid loop: '{name}' failed "
                    f"{error_count}× — consider escalation"
                )
        else:
            self._action_failure_counts.pop(sig, None)
            error_count = 0

        step = ContextStep(
            step_num=self.steps_taken,
            action=name,
            action_input=inp,
            observation=observation,
            thought=thought,
            confidence=confidence,
            is_essential=is_essential,
            error_count=error_count,
        )

        self._steps.append(step)
        self.steps_taken += 1

        if name == "read_file":
            fp = inp.get("file_path") or inp.get("path", "")
            if fp:
                self.scanned_files.add(fp)

    def detect_stupid_loop(self) -> bool:
        return self._stupid_loop_detected

    def reset_stupid_loop_flag(self) -> None:
        self._stupid_loop_detected = False
        self._action_failure_counts.clear()

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------
    def add_cost(self, cost_usd: float, file_path: str | None = None) -> None:
        self.session_cost_usd += cost_usd
        target = file_path
        if target is None and self._steps:
            last = self._steps[-1]
            if last.action == "read_file":
                target = last.action_input.get("file_path", "")
        if target:
            self._cost_by_file[target] += cost_usd

    def should_escalate_by_cost(
        self, threshold: float | None = None
    ) -> tuple[bool, str]:
        thr = threshold or self._cost_threshold
        confirmed_observations = {
            v.get("observation", "")[:50] for v in self.confirmed_vulns
        }
        for fp, cost in self._cost_by_file.items():
            if cost <= thr:
                continue
            covered = any(fp in obs for obs in confirmed_observations)
            if not covered:
                return True, f"Spent ${cost:.2f} on {fp!r} with no confirmed vuln"

        if self.session_cost_usd > 10.0 and not self.confirmed_vulns:
            return True, f"Total ${self.session_cost_usd:.2f} with zero confirmed vulns"

        return False, ""

    def get_expensive_files(self, min_cost: float = 1.0) -> list[tuple[str, float]]:
        return sorted(
            [(f, c) for f, c in self._cost_by_file.items() if c >= min_cost],
            key=lambda x: x[1], reverse=True,
        )

    # ------------------------------------------------------------------
    # Evidence memory
    # ------------------------------------------------------------------
    def add_evidence(
        self,
        pointer: str,
        observation_fragment: str,
        keywords: list[str] | None = None,
    ) -> None:
        self._evidence_map[pointer] = observation_fragment
        kws = keywords or [w for w in observation_fragment.lower().split()[:3] if len(w) > 3]
        for kw in kws:
            self._evidence_index[kw].append(pointer)

    def get_evidence(self, pointer: str) -> str | None:
        return self._evidence_map.get(pointer)

    def search_evidence(self, keyword: str) -> list[str]:
        return self._evidence_index.get(keyword.lower(), [])

    def get_evidence_summary(self) -> dict[str, int]:
        return {
            "total_fragments": len(self._evidence_map),
            "indexed_keywords": len(self._evidence_index),
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def get_recent_history(self, n: int = 6, compressed: bool = True) -> str:
        if not self._steps:
            return "No history yet."
        steps = list(self._steps)[-n:]
        parts = []
        for i, step in enumerate(steps):
            verbose = (not compressed) or (i >= len(steps) - 2)
            parts.append(step.to_text(verbose=verbose))
        return "\n\n".join(parts)

    def get_last_step(self) -> str | None:
        return self._steps[-1].to_text() if self._steps else None

    def get_last_confidence(self) -> float | None:
        return self._steps[-1].confidence if self._steps else None

    def add_critique(self, critique: str) -> None:
        if critique:
            self._critiques.append(critique[:500])

    def trim_history(self, keep_last: int = 6) -> None:
        if len(self._steps) <= keep_last:
            return
        all_steps = list(self._steps)
        recent = all_steps[-keep_last:]
        recent_nums = {s.step_num for s in recent}
        essential = [s for s in all_steps if s.is_essential and s.step_num not in recent_nums]
        budget = keep_last - len(recent)
        extra = essential[-budget:] if budget > 0 else []
        combined = sorted(extra + recent, key=lambda s: s.step_num)
        self._steps = deque(combined, maxlen=self._max_history)
        logger.debug(f"[SharedContext] Trimmed history → {len(self._steps)} steps")

    # ------------------------------------------------------------------
    # Hypothesis management
    # ------------------------------------------------------------------
    def add_hypothesis(self, hyp: str) -> None:
        if hyp and hyp not in self._hypothesis_set and hyp not in self._blacklist_set:
            self.hypotheses.append(hyp)
            self._hypothesis_set.add(hyp)

    def blacklist_hypothesis(self, hyp: str) -> None:
        if not hyp:
            return
        if hyp not in self._blacklist_set:
            self.blacklisted_hypotheses.append(hyp)
            self._blacklist_set.add(hyp)
            if hyp in self.hypotheses:
                self.hypotheses.remove(hyp)
            self._hypothesis_set.discard(hyp)

    def get_deduplicated_hypotheses(self) -> str:
        active = [h for h in self.hypotheses if h not in self._blacklist_set]
        if not active:
            return "None" if not self.blacklisted_hypotheses else "None (all blacklisted)"
        seen: set[str] = set()
        unique = []
        for h in active:
            if h not in seen:
                unique.append(h)
                seen.add(h)
        return "\n".join(f"- {h[:200]}" for h in unique[:10])

    def request_additional_evidence(self, action: str) -> None:
        if self._steps:
            self._steps[-1].needs_evidence = action

    # ------------------------------------------------------------------
    # Error / escalation tracking
    # ------------------------------------------------------------------
    def increment_parse_errors(self) -> None:
        self.parse_errors += 1

    def reset_parse_errors(self) -> None:
        self.parse_errors = 0

    def reset_escalation_counter(self) -> None:
        self.parse_errors = 0

    def set_escalation_level(self, level: int) -> None:
        self.escalation_level = max(1, min(level, 4))

    def add_confirmed_vuln(self, vuln: dict[str, Any]) -> None:
        v = dict(vuln)
        v.setdefault("timestamp", time.time())
        self.confirmed_vulns.append(v)
        if self._steps:
            self._steps[-1].is_essential = True

    def add_failed_tool(self, tool_name: str) -> None:
        if tool_name not in self.failed_tools:
            self.failed_tools.append(tool_name)

    # ------------------------------------------------------------------
    # Advisor state
    # ------------------------------------------------------------------
    def add_advisor_advice(self, advice: Any) -> None:
        self._pending_advisor_advice = advice

    def has_pending_advisor_advice(self) -> bool:
        return self._pending_advisor_advice is not None

    def peek_advisor_advice(self) -> Any | None:
        return self._pending_advisor_advice

    def consume_advisor_advice(self) -> Any | None:
        advice = self._pending_advisor_advice
        self._pending_advisor_advice = None
        return advice

    def get_pending_advisor_advice(self) -> Any | None:
        return self.consume_advisor_advice()

    def clear_pending_advisor_advice(self) -> None:
        self._pending_advisor_advice = None

    def set_advisor_guidance(self, guidance: str) -> None:
        self._advisor_guidance = guidance[:1000]

    def get_advisor_guidance(self) -> str:
        return self._advisor_guidance

    # ------------------------------------------------------------------
    # Knowledge graph fragment
    # ------------------------------------------------------------------
    def get_knowledge_graph_fragment(self) -> dict[str, Any]:
        escalate_by_cost, cost_reason = self.should_escalate_by_cost()
        return {
            "files_touched": list(self.scanned_files),
            "active_hypotheses": self.hypotheses[-5:],
            "dead_ends_count": len(self.blacklisted_hypotheses),
            "evidence_fragments": len(self._evidence_map),
            "stupid_loop_flag": self._stupid_loop_detected,
            "cost_trigger": escalate_by_cost,
            "cost_reason": cost_reason,
        }

    # ------------------------------------------------------------------
    # Metrics and summary
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict[str, Any]:
        return {
            "steps_taken": self.steps_taken,
            "files_scanned": len(self.scanned_files),
            "hypotheses": len(self.hypotheses),
            "confirmed_vulns": len(self.confirmed_vulns),
            "blacklisted": len(self.blacklisted_hypotheses),
            "parse_errors": self.parse_errors,
            "escalation_level": self.escalation_level,
            "session_cost_usd": self.session_cost_usd,
            "stupid_loop": self._stupid_loop_detected,
            "evidence_fragments": len(self._evidence_map),
            "has_pending_advice": self.has_pending_advisor_advice(),
            "history_length": len(self._steps),
        }

    def build_summary(self) -> str:
        duration = time.time() - self.created_at
        lines = [
            "=" * 50,
            "KERYX HUNTER SUMMARY",
            "=" * 50,
            f"Target: {self.target_path}",
            f"Capability: {self.capability}",
            f"Mode: {self.mode}",
            f"Duration: {duration:.1f}s",
            f"Steps: {self.steps_taken}",
            f"Files: {len(self.scanned_files)}",
            f"Hypotheses: {len(self.hypotheses)}",
            f"Confirmed: {len(self.confirmed_vulns)}",
            f"Cost: ${self.session_cost_usd:.4f}",
            f"Loop flag: {'YES' if self._stupid_loop_detected else 'No'}",
        ]
        if self.confirmed_vulns:
            lines.append("\n--- CONFIRMED ---")
            for i, v in enumerate(self.confirmed_vulns, 1):
                conf = v.get("confidence", 0)
                lines.append(f" {i}. {v.get('action', '?')} | conf={conf:.2f}")

        expensive = self.get_expensive_files(min_cost=0.5)
        if expensive:
            lines.append("\n--- EXPENSIVE FILES ---")
            for fp, cost in expensive[:3]:
                lines.append(f" ${cost:.2f} {fp}")

        lines.append("=" * 50)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        routing_dict = None
        if self.routing_plan and hasattr(self.routing_plan, 'capability'):
            routing_dict = {
                "capability": getattr(self.routing_plan, 'capability', None),
                "advisor_name": getattr(self.routing_plan, 'advisor_name', None),
                "budget_usd": getattr(self.routing_plan, 'budget_usd', None),
            }
        return {
            "version": 2,
            "target_path": self.target_path,
            "capability": self.capability,
            "mode": self.mode,
            "routing_plan": routing_dict,
            "created_at": self.created_at,
            "steps_taken": self.steps_taken,
            "parse_errors": self.parse_errors,
            "scanned_files": list(self.scanned_files),
            "hypotheses": self.hypotheses,
            "confirmed_vulns": self.confirmed_vulns,
            "blacklisted_hypotheses": self.blacklisted_hypotheses,
            "failed_tools": self.failed_tools,
            "escalation_level": self.escalation_level,
            "session_cost_usd": self.session_cost_usd,
            "cost_by_file": dict(self._cost_by_file),
            "steps": [s.to_dict() for s in self._steps],
            "critiques": list(self._critiques),
            "advisor_guidance": self._advisor_guidance,
            "evidence_map": self._evidence_map,
            "stupid_loop_detected": self._stupid_loop_detected,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SharedContext:
        ctx = cls(
            target_path=d["target_path"],
            capability=d.get("capability", "deep_reasoning"),
            mode=d.get("mode", "hybrid"),
            routing_plan=d.get("routing_plan"),
        )
        ctx.created_at = d.get("created_at", time.time())
        ctx.steps_taken = d.get("steps_taken", 0)
        ctx.parse_errors = d.get("parse_errors", 0)
        ctx.scanned_files = set(d.get("scanned_files", []))
        ctx.hypotheses = d.get("hypotheses", [])
        ctx._hypothesis_set = set(ctx.hypotheses)
        ctx.confirmed_vulns = d.get("confirmed_vulns", [])
        ctx.blacklisted_hypotheses = d.get("blacklisted_hypotheses", [])
        ctx._blacklist_set = set(ctx.blacklisted_hypotheses)
        ctx.failed_tools = d.get("failed_tools", [])
        ctx.escalation_level = d.get("escalation_level", 1)
        ctx.session_cost_usd = d.get("session_cost_usd", 0.0)
        ctx._advisor_guidance = d.get("advisor_guidance", "")
        ctx._stupid_loop_detected = d.get("stupid_loop_detected", False)
        ctx._cost_by_file = defaultdict(float, d.get("cost_by_file", {}))

        ctx._steps = deque(maxlen=ctx._max_history)
        for s in d.get("steps", []):
            ctx._steps.append(ContextStep.from_dict(s))

        ctx._critiques = deque(
            d.get("critiques", [])[-ctx._max_critiques:],
            maxlen=ctx._max_critiques,
        )
        ctx._evidence_map = d.get("evidence_map", {})
        ctx._evidence_index = defaultdict(list)
        for pointer, text in ctx._evidence_map.items():
            for kw in text.lower().split()[:3]:
                if len(kw) > 3:
                    ctx._evidence_index[kw].append(pointer)

        return ctx

    def get_checkpoint_hash(self) -> str:
        data = f"{self.target_path}:{self.steps_taken}:{len(self.confirmed_vulns)}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return (
            f"SharedContext("
            f"target={self.target_path!r}, "
            f"steps={self.steps_taken}, "
            f"cost=${self.session_cost_usd:.2f}, "
            f"loop={self._stupid_loop_detected})"
        )
