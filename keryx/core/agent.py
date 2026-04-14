# keryx/core/agent.py
# Core ReAct agent for KeryxHunter - Executor + Advisor escalation.
# Sovereign, air-gapped, robust.

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..advisors.manager import AdvisorManager
from ..core.shared_context import SharedContext
from ..models.interface import ModelInterface
from ..tools.Toolbox import ToolBox


@dataclass
class AgentStep:
    """Single step in the ReAct loop."""
    thought:      str
    action:       str
    action_input: dict[str, Any]
    observation:  str   = ""
    confidence:   float = 0.0
    timestamp:    float = field(default_factory=time.time)


@dataclass
class BudgetController:
    """Track and enforce API/model budget for cloud models."""
    max_cost_usd: float = 10.0
    current_cost: float = 0.0
    max_calls:    int   = 500
    calls_made:   int   = 0

    def can_proceed(self, model: ModelInterface) -> bool:
        if self.calls_made >= self.max_calls:
            return False
        return self.current_cost + self._estimate_call_cost(model) <= self.max_cost_usd

    def record_call(
        self,
        model:      ModelInterface,
        tokens_in:  int = 1000,
        tokens_out: int = 500,
    ) -> None:
        cost_in  = model.cost_per_1k_input_tokens  or 0.0
        cost_out = model.cost_per_1k_output_tokens or 0.0
        self.current_cost += (tokens_in / 1000) * cost_in + (tokens_out / 1000) * cost_out
        self.calls_made   += 1

    def _estimate_call_cost(self, model: ModelInterface) -> float:
        cost_in  = model.cost_per_1k_input_tokens  or 0.0
        cost_out = model.cost_per_1k_output_tokens or 0.0
        return (2.0 * cost_in) + (1.0 * cost_out)

    @property
    def remaining_budget(self) -> float:
        return self.max_cost_usd - self.current_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cost_usd":      self.max_cost_usd,
            "current_cost":      self.current_cost,
            "calls_made":        self.calls_made,
            "remaining_budget":  self.remaining_budget,
        }


# ---------------------------------------------------------------------------
# GBNF grammar — tested against llama.cpp grammar validator
# ---------------------------------------------------------------------------
_OUTPUT_GRAMMAR = r'''
root ::= "{" ws kv-thought "," ws kv-action "," ws kv-input "," ws kv-conf ws "}"
kv-thought ::= "\"thought\"" ws ":" ws string
kv-action ::= "\"action\"" ws ":" ws string
kv-input ::= "\"action_input\"" ws ":" ws object
kv-conf ::= "\"confidence\"" ws ":" ws number
value ::= string | number | object | array | "true" | "false" | "null"
object ::= "{" ws ( string ws ":" ws value ( "," ws string ws ":" ws value )* )? ws "}"
array ::= "[" ws ( value ( "," ws value )* )? ws "]"
string ::= "\"" ( [^"\\] | "\\" . )* "\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
ws ::= [ \t\n\r]*
'''

_VALID_ACTIONS = frozenset([
    "codeql_query",
    "gdb_analyze",
    "fuzzer_run",
    "git_blame",
    "read_file",
    "rag_search",
    "FINISH",
    "NO_ACTION",
])

# Tools whose output can confirm a real vulnerability (source check for FIX 7)
_VULN_CONFIRMING_TOOLS = frozenset(["gdb_analyze", "fuzzer_run"])

_ESCALATION_COOLDOWN    = 5
_MAX_HISTORY_STEPS      = 12
_TOOL_TIMEOUT_SECONDS   = 30.0


class KeryxAgent:
    """
    Main Executor agent.
    Runs the ReAct loop, calls tools, self-critiques, escalates to Advisor.
    Supports checkpoint/resume via JSON (not pickle).
    """

    def __init__(
        self,
        executor_model:       ModelInterface,
        advisor_manager:      AdvisorManager,
        toolbox:              ToolBox,
        max_steps:            int            = 50,
        confidence_threshold: float          = 0.65,
        max_prompt_chars:     int            = 32_000,
        checkpoint_dir:       Path | None = None,
        generate_timeout:     float          = 90.0,
        budget_usd:           float | None= None,
        enforce_airgapped:    bool           = False,
    ) -> None:
        self.executor             = executor_model
        self.advisor_manager      = advisor_manager
        self.tools                = toolbox
        self.max_steps            = max_steps
        self.confidence_threshold = confidence_threshold
        self.max_prompt_chars     = max_prompt_chars
        self.generate_timeout     = generate_timeout
        self.checkpoint_dir       = checkpoint_dir or Path(".keryx_checkpoints")
        self.enforce_airgapped    = enforce_airgapped
        self._shared_context: SharedContext | None = None

        self.budget = BudgetController(max_cost_usd=budget_usd) if budget_usd else None

        self._steps_since_escalation:  int = _ESCALATION_COOLDOWN
        self._consecutive_failures:    int = 0
        self._max_consecutive_failures:int = 3

    # ------------------------------------------------------------------
    # Context property
    # ------------------------------------------------------------------

    @property
    def context(self) -> SharedContext:
        """Non-optional view of the shared context.

        Valid only after run() initialises it.  Raises AssertionError with a
        clear message if accessed before run() is called.
        """
        assert self._shared_context is not None, (
            "KeryxAgent.context accessed before run() — call run() first."
        )
        return self._shared_context

    @context.setter
    def context(self, value: SharedContext) -> None:
        """Allow external assignment (e.g. test fixtures) via the same name."""
        self._shared_context = value

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        target_path: str,
        capability:  str  = "deep_reasoning",
        resume:      bool = True,
        context: SharedContext | None = None,
    ) -> dict[str, Any]:
        """Main hunt loop.

        context: pre-built SharedContext injected by KeryxOrchestrator so that
                 accumulated state (steps, hypotheses, escalation level) is preserved
                 across escalation attempts.  When provided, resume is ignored.
        """
        if context is not None:
            self._shared_context = context
        else:
            self._shared_context = self._load_checkpoint(target_path) if resume else None
        if self._shared_context is None:
            self._shared_context = SharedContext(target_path=target_path, capability=capability)

        print(
            f"[Agent] Hunt starting | target={target_path} | capability={capability} "
            f"| step={self.context.steps_taken}/{self.max_steps}"
        )
        if self.budget:
            mode = "airgapped" if self.executor.is_local else "cloud"
            print(f"[Agent] Budget: ${self.budget.max_cost_usd} | Mode: {mode}")

        # Enforce air-gapped constraint: refuse to run if a network model was
        # selected despite the airgapped flag.  Checked once before the loop
        # rather than per-step to surface the misconfiguration immediately.
        if self.enforce_airgapped and getattr(self.executor, "requires_network", False):
            print(
                "[Agent] AIRGAP VIOLATION: executor requires network but "
                "enforce_airgapped=True — aborting hunt."
            )
            return {
                "status": "failed",
                "reason": "airgap_violation",
                "target": self.context.target_path,
                "steps_taken": self.context.steps_taken,
                "confirmed_vulns": [],
                "hypotheses": list(self.context.hypotheses),
                "average_confidence": 0.0,
            }

        while self.context.steps_taken < self.max_steps:

            # ── Budget ────────────────────────────────────────────────────
            if self.budget and not self.budget.can_proceed(self.executor):
                print(
                    f"[Agent] Budget exhausted "
                    f"(${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd}) — finishing."
                )
                break

            # ── Health ────────────────────────────────────────────────────
            if not await self._check_model_health():
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    print("[Agent] Model unhealthy — cannot continue.")
                    break
                continue

            step = self.context.steps_taken + 1
            print(f"[Step {step}/{self.max_steps}]")

            # ── Prompt ───────────────────────────────────────────────────
            if self.context.has_pending_advisor_advice():
                prompt = self._build_prompt_with_guidance()
                print("[Agent] Including Advisor guidance in prompt")
            else:
                prompt = self._build_prompt()

            # FIX 5: trim_history() now exists — uncommented
            if len(prompt) > self.max_prompt_chars:
                self.context.trim_history()
                prompt = self._build_prompt()

            # ── Generate ─────────────────────────────────────────────────
            action = await self._safe_generate(prompt)
            self._steps_since_escalation += 1

            if self.budget and action:
                self.budget.record_call(
                    self.executor,
                    tokens_in=len(prompt) // 4,
                    tokens_out=500,
                )

            if action is None:
                self._consecutive_failures += 1
                action = AgentStep(
                    thought="emergency_fallback_model_unresponsive",
                    action="NO_ACTION",
                    action_input={},
                    confidence=0.1,
                )
            else:
                self._consecutive_failures = 0

            # ── Validate action name ──────────────────────────────────────
            if action.action not in _VALID_ACTIONS:
                print(f"[WARN] Unknown action '{action.action}' — treating as NO_ACTION")
                action.action = "NO_ACTION"

            # ── Execute tool ──────────────────────────────────────────────
            if action.action not in ("NO_ACTION", "FINISH"):
                try:
                    observation = await asyncio.wait_for(
                        self._execute_tool_async(action.action, action.action_input),
                        timeout=_TOOL_TIMEOUT_SECONDS,
                    )
                    action.observation = observation
                    self.context.add_step(action, observation)
                    print(f"[Tool] {action.action} completed in {time.time() - action.timestamp:.2f}s")

                    critique = await self._self_critique_async()
                    self._apply_critique(critique, action)

                except TimeoutError:
                    msg = f"Tool '{action.action}' timed out after {_TOOL_TIMEOUT_SECONDS}s"
                    print(f"[ERROR] {msg}")
                    action.observation = msg
                    action.confidence  *= 0.5
                    self.context.add_step(action, msg)

                except Exception as exc:
                    msg = f"Tool '{action.action}' failed: {exc}"
                    print(f"[ERROR] {msg}")
                    action.observation = msg
                    action.confidence  *= 0.7
                    self.context.add_step(action, msg)
            else:
                self.context.add_step(action, "No action taken")

            # ── Apply advisor advice ──────────────────────────────────────
            if self.context.has_pending_advisor_advice():
                self._apply_advisor_advice()

            # ── Escalate ─────────────────────────────────────────────────
            await self._maybe_escalate()

            # ── Early stop — vuln confirmed ───────────────────────────────
            # FIX 7: only confirm if observation comes from a tool that can
            # actually crash the target, not from read_file text matching.
            if (
                action.confidence > 0.9
                and action.action in _VULN_CONFIRMING_TOOLS
                and self._observation_confirms_vuln(action.observation)
            ):
                print("[Agent] High-confidence vulnerability confirmed — stopping early.")
                self.context.confirmed_vulns.append({
                    "step":        step,
                    "action":      action.action,
                    "observation": action.observation[:500],
                    "confidence":  action.confidence,
                })
                break

            if action.action == "FINISH":
                print("[Agent] Executor signalled FINISH.")
                break

            # ── Checkpoint every 5 steps ──────────────────────────────────
            if step % 5 == 0:
                self._save_checkpoint(target_path)

        return self._generate_final_report()

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    async def _check_model_health(self) -> bool:
        try:
            result = await asyncio.wait_for(
                self._generate_async("ping", max_tokens=1),
                timeout=5.0,
            )
            return bool(result)
        except Exception:
            return False

    async def _safe_generate(self, prompt: str) -> AgentStep | None:
        for attempt in range(2):
            try:
                raw = await asyncio.wait_for(
                    self._generate_async(prompt),
                    timeout=self.generate_timeout,
                )
            except TimeoutError:
                print(f"[WARN] Generation timeout (attempt {attempt + 1})")
                # FIX 4: count generation timeouts as parse errors
                self.context.increment_parse_errors()
                if attempt == 1:
                    return None
                continue

            action = self._parse_response(raw)
            if action.action != "NO_ACTION" or action.thought:
                return action
        return None

    async def _generate_async(self, prompt: str, max_tokens: int = 1000) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.executor.generate(
                prompt, grammar=_OUTPUT_GRAMMAR, max_tokens=max_tokens
            ),
        )

    async def _execute_tool_async(self, action: str, action_input: dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.tools.execute, action, action_input
        )
        # ToolBox.execute() returns ToolResult; extract text so add_step() gets a str.
        if hasattr(result, "output"):
            return result.output if result.output is not None else str(result)
        return str(result)

    async def _self_critique_async(self) -> str:
        prompt = (
            "You are reviewing your last analysis step.\n"
            "Is the code path reachable? Any false-positive risk? "
            "Is there a simpler explanation? "
            "Output your critique in 2-3 sentences.\n\n"
            f"Step: {self.context.get_recent_history(1)}"
        )
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.executor.generate(prompt, max_tokens=350),
                ),
                timeout=15.0,
            )
        except TimeoutError:
            return "Critique timed out — proceeding with original hypothesis."

    def _apply_critique(self, critique: str, action: AgentStep) -> None:
        critique_lower = critique.lower()

        if "false positive" in critique_lower or "not reachable" in critique_lower:
            hyp = action.action_input.get("hypothesis", "")
            self.context.blacklist_hypothesis(hyp)
            print("[Critique] Hypothesis blacklisted as false positive")
            action.confidence *= 0.3

        elif "needs more evidence" in critique_lower or "insufficient" in critique_lower:
            self.context.request_additional_evidence(action.action)
            print("[Critique] Additional evidence requested")

        elif "confirmed" in critique_lower or "likely" in critique_lower:
            action.confidence = min(action.confidence + 0.1, 0.95)
            print(f"[Critique] Confidence boosted to {action.confidence:.2f}")

        self.context.add_critique(critique)

    def _apply_advisor_advice(self) -> None:
        """
        FIX 1+2+3: advice is now AdvisorAdvice (dataclass), not a dict.
        Access fields as attributes. Save strategic_direction via
        set_advisor_guidance() so _build_prompt_with_guidance() picks it up.
        """
        advice = self.context.get_pending_advisor_advice()
        if not advice:
            return

        print(f"[Advisor] Applying advice: {advice.strategy[:100]}")

        # Persist strategic direction for future prompt injections
        if advice.strategic_direction:
            self.context.set_advisor_guidance(advice.strategic_direction)

        # Adjust confidence threshold if recommended
        if advice.adjust_confidence_threshold is not None:
            self.confidence_threshold = max(
                0.3, min(0.9, advice.adjust_confidence_threshold)
            )
            print(f"[Advisor] Confidence threshold → {self.confidence_threshold}")

        self.context.clear_pending_advisor_advice()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _extract_json(self, raw: str) -> str | None:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:].strip()
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            return raw[start:end + 1]
        return None

    def _fallback_parse(self, raw: str) -> AgentStep:
        raw_lower = raw.lower()
        for action in _VALID_ACTIONS:
            if action.lower() in raw_lower and action != "NO_ACTION":
                return AgentStep(
                    thought="fallback_parse_inferred",
                    action=action,
                    action_input={},
                    confidence=0.3,
                )
        return AgentStep(
            thought="fallback_parse_failed",
            action="NO_ACTION",
            action_input={},
            confidence=0.1,
        )

    def _parse_response(self, raw: str) -> AgentStep:
        json_str = self._extract_json(raw)
        if not json_str:
            print(f"[WARN] Could not extract JSON from response (len={len(raw)})")
            # FIX 4: increment so _should_escalate() escalation-by-errors fires
            self.context.increment_parse_errors()
            return self._fallback_parse(raw)

        try:
            data = json.loads(json_str)
            return AgentStep(
                thought=      str(data.get("thought", "")),
                action=       str(data.get("action", "NO_ACTION")),
                action_input= dict(data.get("action_input", {})),
                confidence=   float(data.get("confidence", 0.5)),
            )
        except Exception as exc:
            print(f"[WARN] JSON parse failed: {exc.__class__.__name__}: {exc}")
            self.context.increment_parse_errors()
            return self._fallback_parse(raw)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        ctx    = self.context
        recent = ctx.get_recent_history(_MAX_HISTORY_STEPS)

        # FIX 5: budget line built separately — ternary mid-concatenation
        # causes the remaining string literals to be silently dropped.
        budget_line = (
            f"Budget used : ${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd:.2f}\n"
            if self.budget else ""
        )

        return (
            "You are Keryx Executor — a precise, local vulnerability hunter.\n"
            f"Target       : {ctx.target_path}\n"
            f"Scanned      : {len(ctx.scanned_files)} files\n"
            f"Hypotheses   : {len(ctx.hypotheses)}\n"
            f"Confirmed    : {len(ctx.confirmed_vulns)}\n"
            f"Parse errors : {ctx.parse_errors}\n"
            f"Advisor calls: {self.advisor_manager.calls_made}\n"
            f"{budget_line}"
            "\nRecent history (last few steps):\n"
            f"{recent}\n\n"
            "Known hypotheses (do NOT repeat these):\n"
            f"{ctx.get_deduplicated_hypotheses()}\n\n"
            "Available actions: " + ", ".join(sorted(_VALID_ACTIONS)) + "\n\n"
            "Think step by step. Output ONLY valid JSON matching the grammar."
        )

    def _build_prompt_with_guidance(self) -> str:
        """
        FIX 2: use get_advisor_guidance() — returns a clean string set by
        _apply_advisor_advice() via set_advisor_guidance().
        Previously called get_pending_advisor_advice() which returns an
        AdvisorAdvice dataclass repr, not useful for the LLM.
        """
        base     = self._build_prompt()
        guidance = self.context.get_advisor_guidance()
        if guidance:
            base += f"\n\n=== ADVISOR GUIDANCE ===\n{guidance}\n========================\n"
        return base

    def _simplify_prompt(self, prompt: str) -> str:
        lines = prompt.split("\n")
        essential = [
            line for line in lines
            if line.strip() and not line.startswith("Recent history")
        ]
        return "\n".join(essential[:50])

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    async def _maybe_escalate(self, force: bool = False) -> None:
        if not force and not self._should_escalate():
            return
        if not self.advisor_manager.can_advise():
            print("[Agent] Advisor limit reached — continuing without advice.")
            return

        print("[Agent] Escalating to Advisor...")
        loop   = asyncio.get_running_loop()
        advice = await loop.run_in_executor(
            None, self.advisor_manager.get_advice, self.context
        )
        self.context.add_advisor_advice(advice)
        self._steps_since_escalation = 0

    def _should_escalate(self) -> bool:
        if self._steps_since_escalation < _ESCALATION_COOLDOWN:
            return False

        last_conf        = self.context.get_last_confidence()
        no_progress      = (
            len(self.context.hypotheses) == 0
            and self.context.steps_taken > 15
        )
        too_many_errors  = self.context.parse_errors > 3
        low_confidence   = last_conf is not None and last_conf < self.confidence_threshold

        return low_confidence or no_progress or too_many_errors

    # ------------------------------------------------------------------
    # Checkpoint — JSON only, no pickle
    # ------------------------------------------------------------------

    def _checkpoint_path(self, target_path: str) -> Path:
        key = hashlib.sha256(target_path.encode()).hexdigest()[:12]
        return self.checkpoint_dir / f"checkpoint_{key}.json"

    def _save_checkpoint(self, target_path: str) -> None:
        """FIX 6: JSON serialisation via SharedContext.to_dict() — no pickle."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(target_path)
        try:
            payload = {
                "context":                self.context.to_dict(),
                "budget":                 self.budget.to_dict() if self.budget else None,
                "steps_since_escalation": self._steps_since_escalation,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Checkpoint save failed: {exc}")

    def _load_checkpoint(self, target_path: str) -> SharedContext | None:
        """FIX 6: load from JSON — no arbitrary code execution."""
        path = self._checkpoint_path(target_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ctx = SharedContext.from_dict(payload["context"])
            print(f"[Agent] Resumed from checkpoint at step {ctx.steps_taken}")
            if payload.get("budget") and self.budget:
                b = payload["budget"]
                self.budget.current_cost = b.get("current_cost", 0.0)
                self.budget.calls_made   = b.get("calls_made", 0)
            self._steps_since_escalation = payload.get(
                "steps_since_escalation", _ESCALATION_COOLDOWN
            )
            return ctx
        except Exception as exc:
            print(f"[WARN] Checkpoint load failed ({exc}) — starting fresh")
            return None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _observation_confirms_vuln(observation: str) -> bool:
        """
        FIX 7: called only when action.action in _VULN_CONFIRMING_TOOLS,
        so source-check is enforced at the call site, not here.
        Signals that constitute a real crash/sanitizer finding.
        """
        signals = (
            "VULN_CONFIRMED",
            "AddressSanitizer",
            "heap-use-after-free",
            "stack-buffer-overflow",
            "SEGFAULT",
            "CRASH",
            "UAF",
        )
        return any(s in observation for s in signals)

    def _generate_final_report(self) -> dict[str, Any]:
        return {
            "status":           "completed",
            "target":           self.context.target_path,
            "steps_taken":      self.context.steps_taken,
            "max_steps":        self.max_steps,
            "confirmed_vulns":  self.context.confirmed_vulns,
            "hypotheses_count": len(self.context.hypotheses),
            "advisor_calls":    self.advisor_manager.calls_made,
            "parse_errors":     self.context.parse_errors,
            "budget":           self.budget.to_dict() if self.budget else None,
            "model": {
                "name":     self.executor.model_name,
                "is_local": self.executor.is_local,
            },
            "summary": self.context.build_summary(),
        }
