# keryx/core/agent.py
# Core ReAct agent for KeryxHunter - Executor + Advisor escalation.
# Sovereign, air-gapped, robust.

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..advisors.manager import AdvisorManager
from ..core.shared_context import SharedContext
from ..models.interface import ModelInterface
from ..tools.Toolbox import ToolBox
from ._budget import BudgetController
from ._constants import (
    ESCALATION_COOLDOWN,
    MAX_HISTORY_STEPS,
    OUTPUT_GRAMMAR,
    TOOL_TIMEOUT_SECONDS,
    VALID_ACTIONS,
    VULN_CONFIRMING_TOOLS,
)
from ._loop_guards import check_exit_clean, observation_confirms_vuln
from ._parser import extract_json, fallback_parse, parse_response
from .verification_pipeline import VerificationPipeline

# Re-export so existing callers (`from keryx.core.agent import BudgetController`) still work.
__all__ = ["KeryxAgent", "AgentStep", "BudgetController"]

# Patchable module-level aliases kept for backward compatibility.
# Tests that do `agent_mod._TOOL_TIMEOUT_SECONDS = 0.05` or
# `from keryx.core.agent import _ESCALATION_COOLDOWN` still work because
# these names live in this module's __dict__ and are looked up as globals
# at call time — not bound at import time.
_ESCALATION_COOLDOWN:  int   = ESCALATION_COOLDOWN
_TOOL_TIMEOUT_SECONDS: float = TOOL_TIMEOUT_SECONDS


@dataclass
class AgentStep:
    """Single step in the ReAct loop."""
    thought:      str
    action:       str
    action_input: dict[str, Any]
    observation:  str   = ""
    confidence:   float = 0.0
    timestamp:    float = field(default_factory=time.time)


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
        checkpoint_dir:       Path | None    = None,
        generate_timeout:     float          = 90.0,
        budget_usd:           float | None   = None,
        enforce_airgapped:    bool           = False,
        critique_model:       ModelInterface | None = None,
        verification_mode:    Literal["none", "strict", "flexible"] = "flexible",
        max_clean_scans_before_exit: int = 1,
    ) -> None:
        self.executor             = executor_model
        # critique_executor handles health checks, self-critique, and other
        # lightweight calls.  Falls back to executor_model if not provided.
        self.critique_executor    = critique_model or executor_model
        self.advisor_manager      = advisor_manager
        self.tools                = toolbox
        self.max_steps            = max_steps
        self.confidence_threshold = confidence_threshold
        self.max_prompt_chars     = max_prompt_chars
        self.generate_timeout     = generate_timeout
        self.checkpoint_dir       = checkpoint_dir or Path(".keryx_checkpoints")
        self.enforce_airgapped    = enforce_airgapped
        self._shared_context: SharedContext | None = None
        # Verification mode controls how codeql HIGH findings are handled:
        #   "none"     — immediate static confirm, no injection_verifier call
        #   "strict"   — auto-verify via injection_verifier (CI/CD pipelines)
        #   "flexible" — agent decides; 2-step window then static fallback
        self._verification_mode: Literal["none", "strict", "flexible"] = verification_mode

        # Step at which codeql first reported a HIGH finding (flexible mode window).
        self._codeql_high_found_step: int = -100

        self.budget = BudgetController(max_cost_usd=budget_usd) if budget_usd else None

        self._steps_since_escalation:       int = _ESCALATION_COOLDOWN
        self._consecutive_failures:         int = 0
        self._max_consecutive_failures:     int = 3
        self._steps_since_new_hypothesis:   int = 0   # stagnation tracker
        self._consecutive_low_confidence:   int = 0   # low-conf streak
        self._codeql_unconfirmed:           bool = False  # AST found HIGH but not confirmed
        self._codeql_high_observation:      str  = ""     # observation snapshot from triggering step
        # Set to True when critique explicitly declares the codeql HIGH finding a
        # false positive.  Blocks the flexible static fallback from confirming it.
        self._codeql_high_rejected:         bool = False
        # Early-exit: break after this many consecutive clean codeql scans (no HIGH findings).
        # 1 = exit on first clean scan (fast mode); set higher to let agent probe more paths.
        self._consecutive_clean_scans:      int  = 0
        self._max_clean_scans_before_exit:  int  = max_clean_scans_before_exit
        # One-shot hint from a previous model pass (Phase 1c escalation).
        # Injected into the first prompt only; cleared after that step.
        self._initial_hint: str | None = None
        # Flexible mode: track whether the agent has read the file source at least
        # once.  The 2-step verification window does NOT start until this is True,
        # so the LLM always sees the code before confirming.
        self._file_read_done: bool = False
        # Snapshot of the codeql HIGH observation before read_file has been called.
        # Cleared and promoted to _codeql_high_found_step once read_file completes.
        self._pending_high_observation: str = ""

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
        target_path:  str,
        capability:   str              = "deep_reasoning",
        resume:       bool             = True,
        context:      SharedContext | None = None,
        context_hint: str | None       = None,
    ) -> dict[str, Any]:
        """Main hunt loop.

        context:      pre-built SharedContext injected by KeryxOrchestrator so that
                      accumulated state (steps, hypotheses, escalation level) is
                      preserved across escalation attempts.  When provided, resume
                      is ignored.
        context_hint: optional one-line note from a previous model pass, prepended
                      to the FIRST step prompt only (Phase 1c escalation hint).
                      Cleared after the first generate call so it does not repeat.
        """
        if context is not None:
            self._shared_context = context
        else:
            self._shared_context = self._load_checkpoint(target_path) if resume else None
        if self._shared_context is None:
            self._shared_context = SharedContext(target_path=target_path, capability=capability)

        if context_hint:
            self._initial_hint = context_hint
            print(f"[Agent] Received hint from previous model: {context_hint[:200]}"
                  f"{'...' if len(context_hint) > 200 else ''}")

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

            # ── Pre-step guard (budget + health) ─────────────────────────
            precond = await self._precondition_check()
            if precond == "break":    break
            if precond == "continue": continue

            step = self.context.steps_taken + 1
            print(f"[Step {step}/{self.max_steps}]")

            # ── Prompt ───────────────────────────────────────────────────
            if self.context.has_pending_advisor_advice():
                prompt = self._build_prompt_with_guidance()
                print("[Agent] Including Advisor guidance in prompt")
            else:
                prompt = self._build_prompt()

            if len(prompt) > self.max_prompt_chars:
                self.context.trim_history()
                prompt = self._build_prompt()

            # ── Generate ─────────────────────────────────────────────────
            action = await self._safe_generate(prompt)
            self._initial_hint = None   # one-shot: clear after first generate
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
            if action.action not in VALID_ACTIONS:
                print(f"[WARN] Unknown action '{action.action}' — treating as NO_ACTION")
                action.action = "NO_ACTION"

            # ── Execute tool + critique + per-step bookkeeping ────────────
            hyp    = action.action_input.get("hypothesis", "")
            action = await self._execute_and_critique(action)

            # ── Early exit: clean codeql scan ─────────────────────────────
            if self._should_exit_clean(action):
                break

            # ── Apply advisor advice ──────────────────────────────────────
            if self.context.has_pending_advisor_advice():
                self._apply_advisor_advice()

            # ── Escalate ─────────────────────────────────────────────────
            await self._maybe_escalate()

            # ── Vuln confirmation stop conditions ─────────────────────────
            if await self._should_stop_on_confirmation(action, step, hyp):
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
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.critique_executor.generate("ping", max_tokens=10),
                ),
                timeout=30.0,
            )
            return isinstance(result, str)
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
                prompt, grammar=OUTPUT_GRAMMAR, max_tokens=max_tokens
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

    async def _self_critique_async(self, hypothesis: str = "", observation: str = "") -> str:
        hyp_line = f"Hypothesis under review: {hypothesis}\n" if hypothesis else ""
        obs_line = f"Evidence gathered:\n{observation[:6000]}\n\n" if observation else ""
        target   = self.context.target_path
        import os as _os
        basename = _os.path.basename(target)
        is_test  = (
            basename.startswith("test_")
            or basename.endswith("_test.py")
            or "/test" in target.replace("\\", "/")
            or "/mock" in target.replace("\\", "/")
        )
        test_note = (
            f"NOTE: The file being analyzed ({basename}) is a TEST FILE. "
            "Findings in test/fixture code are usually false positives — "
            "test files exist in a controlled environment and are not attack surfaces.\n\n"
            if is_test else ""
        )
        prompt = (
            "You are reviewing a specific vulnerability hypothesis about analyzed source code.\n"
            "The 'read_file' / 'git_blame' calls are just tools for gathering evidence — "
            "do NOT judge those as vulnerabilities themselves.\n\n"
            f"{test_note}"
            f"Target file: {target}\n"
            f"{hyp_line}"
            f"{obs_line}"
            "Answer concisely:\n"
            "1. Does the evidence actually support this hypothesis? (YES / PARTIALLY / NO)\n"
            "2. Is the vulnerable code path reachable from untrusted external input "
            "(NOT in test/fixture/mock code)?\n"
            "3. Your verdict: 'GENUINE CONCERN', 'NEEDS MORE EVIDENCE', or "
            "'THIS IS A FALSE POSITIVE' (use this if the code is test-only, "
            "the value is a placeholder, or the path is clearly not reachable from user input).\n"
        )
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.critique_executor.generate(prompt, max_tokens=250),
                ),
                timeout=15.0,
            )
        except TimeoutError:
            return "Critique timed out — proceeding with original hypothesis."

    def _apply_critique(self, critique: str, action: AgentStep) -> None:
        critique_lower = critique.lower()

        # injection_verifier REJECTED ≠ code is safe.
        # The static AST finding stands; dynamic test only adds confidence, never removes it.
        if action.action == "injection_verifier":
            if "VULN_CONFIRMED" in action.observation:
                action.confidence = min(action.confidence + 0.2, 0.95)
                print("[Critique] injection_verifier confirmed — confidence boosted")
            else:
                print("[Critique] injection_verifier rejected — static finding preserved")
            self.context.add_critique(critique)
            return

        # Only blacklist on explicit false-positive declaration, not incidental mentions
        is_false_positive = (
            "this is a false positive" in critique_lower
            or "not reachable" in critique_lower
            or "unreachable" in critique_lower
        )
        if is_false_positive:
            hyp = action.action_input.get("hypothesis", "")
            self.context.blacklist_hypothesis(hyp)
            print("[Critique] Hypothesis blacklisted as false positive")
            action.confidence *= 0.3
            # If we're inside or right at the codeql HIGH verification window,
            # mark the finding itself as critique-rejected so the static fallback
            # does not confirm it.
            if self._codeql_high_found_step >= 0 or self._pending_high_observation:
                self._codeql_high_rejected = True
                print("[Critique] codeql HIGH marked as critique-rejected — will not auto-confirm")

        elif "needs more evidence" in critique_lower or "insufficient" in critique_lower:
            self.context.request_additional_evidence(action.action)
            print("[Critique] Additional evidence requested")

        elif "confirmed" in critique_lower or "likely" in critique_lower or "genuine" in critique_lower:
            action.confidence = min(action.confidence + 0.1, 0.95)
            print(f"[Critique] Confidence boosted to {action.confidence:.2f}")

        self.context.add_critique(critique)

    # ------------------------------------------------------------------
    # Step execution — tool call + critique + per-step bookkeeping
    # ------------------------------------------------------------------

    async def _execute_and_critique(self, action: AgentStep) -> AgentStep:
        """Run the tool, record hypothesis + evidence, run self-critique,
        update stagnation/confidence trackers.  Returns the filled AgentStep.

        All per-step side effects are scoped here so run() stays a clean loop.
        """
        prev_hyp_count = len(self.context.hypotheses)
        hyp = action.action_input.get("hypothesis", "")

        if action.action not in ("NO_ACTION", "FINISH"):
            try:
                obs = await asyncio.wait_for(
                    self._execute_tool_async(action.action, action.action_input),
                    timeout=_TOOL_TIMEOUT_SECONDS,
                )
                action.observation = obs
                self.context.add_step(action, obs)
                print(f"[Tool] {action.action} completed in {time.time() - action.timestamp:.2f}s")
                if action.action == "read_file":
                    self._file_read_done = True
                    # Promote any parked codeql HIGH into the active window now
                    # that the agent has read the source.
                    if (
                        self._verification_mode == "flexible"
                        and self._pending_high_observation
                        and self._codeql_high_found_step < 0
                    ):
                        self._codeql_high_found_step  = self.context.steps_taken
                        self._codeql_high_observation = self._pending_high_observation
                        self._pending_high_observation = ""
                        print(
                            "[Agent] flexible mode — read_file done, "
                            "verification window now open (2 steps)."
                        )

                if hyp:
                    self.context.add_hypothesis(hyp)
                    if len(obs) > 50:
                        loc = (
                            action.action_input.get("path")
                            or action.action_input.get("file")
                            or action.action
                        )
                        self.context.add_evidence(location=str(loc), description=hyp[:200])

                critique = await self._self_critique_async(hypothesis=hyp, observation=obs)
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

        # Stagnation tracker: reset when a new hypothesis was added this step
        if len(self.context.hypotheses) > prev_hyp_count:
            self._steps_since_new_hypothesis = 0
        else:
            self._steps_since_new_hypothesis += 1

        # Confidence streak tracker
        if action.confidence >= 0.3:
            self._consecutive_low_confidence = 0
        else:
            self._consecutive_low_confidence += 1

        # Flag unconfirmed codeql HIGH so advisor triggers on next chance
        if action.action == "codeql_query" and observation_confirms_vuln(action.observation):
            self._codeql_unconfirmed = True

        return action

    # ------------------------------------------------------------------
    # Loop control helpers — each stop condition in one testable place
    # ------------------------------------------------------------------

    async def _precondition_check(self) -> Literal["proceed", "break", "continue"]:
        """Pre-step gate: budget and model health.

        Returns:
            "proceed"  — all clear, run the step
            "break"    — terminate the hunt loop
            "continue" — skip this iteration (health flap, try again next step)
        """
        if self.budget and not self.budget.can_proceed(self.executor):
            print(
                f"[Agent] Budget exhausted "
                f"(${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd}) — finishing."
            )
            return "break"
        if not await self._check_model_health():
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                print("[Agent] Model unhealthy — cannot continue.")
                return "break"
            return "continue"
        return "proceed"

    def _should_exit_clean(self, action: AgentStep) -> bool:
        """Returns True if the loop should stop due to a clean codeql scan.

        Delegates to the pure check_exit_clean() and updates the counter.
        Called BEFORE advisor advice so we don't waste an LLM call on a clean file.
        """
        should_exit, self._consecutive_clean_scans = check_exit_clean(
            action, self._consecutive_clean_scans, self._max_clean_scans_before_exit
        )
        if should_exit:
            print(
                f"[Agent] codeql returned no findings "
                f"({self._consecutive_clean_scans}x clean) — early exit."
            )
        return should_exit

    @staticmethod
    def _extract_rules(observation: str) -> list[str]:
        """Return deduplicated rule names found in an AST/codeql observation.

        Parses patterns like ``[HIGH] SUBPROCESS_SHELL_TRUE @ line 5`` and
        ``[MEDIUM] UNSAFE_DESERIALIZATION @ line 12``.  Order is preserved (first
        occurrence wins), duplicates removed.
        """
        seen: dict[str, None] = {}
        for m in re.finditer(r"\[(?:HIGH|MEDIUM|LOW)\]\s+(\w+)", observation):
            seen.setdefault(m.group(1), None)
        return list(seen)

    async def _should_stop_on_confirmation(
        self, action: AgentStep, step: int, hyp: str
    ) -> bool:
        """Post-advisor termination gate: vuln confirmed, static fallback, FINISH.

        Returns True if the loop should stop.
        Side effects: appends to context.confirmed_vulns, resets verification fields.
        """
        ast_found = (
            action.action == "codeql_query"
            and observation_confirms_vuln(action.observation)
        )
        # injection_verifier uses a lower confidence bar — VULN_CONFIRMED in output
        # is already an objective signal, no need for model confidence >= 0.85.
        dynamic_confirmed = (
            (
                action.confidence >= 0.85
                and action.action in VULN_CONFIRMING_TOOLS - {"codeql_query", "injection_verifier"}
                and observation_confirms_vuln(action.observation)
            )
            or (
                action.action == "injection_verifier"
                and observation_confirms_vuln(action.observation)
            )
        )

        # ── Three-mode confirmation dispatch ──────────────────────────
        if ast_found:
            mode = self._verification_mode

            if mode == "none":
                # Immediate static confirm — no verifier, no window.
                self._codeql_unconfirmed = False
                print("[Agent] none mode — static confirm (AST-only).")
                _rules = self._extract_rules(action.observation)
                self.context.confirmed_vulns.append({
                    "step":           step,
                    "action":         action.action,
                    "observation":    action.observation[:500],
                    "confidence":     action.confidence,
                    "hypothesis":     hyp,
                    "verified":       False,
                    "confidence_tag": "AST-only",
                    "rules":          _rules,
                    "rule":           _rules[0] if _rules else "UNKNOWN",
                })
                return True

            elif mode == "strict":
                # Auto-verify — injection_verifier for internal targets,
                # fuzzer PoC for external targets (target_tool not in toolbox).
                iv_ok, verify_method = await self._auto_verify_injection(action.observation)
                tag = f"AST+{verify_method}" if iv_ok else "AST-only"
                self._codeql_unconfirmed = False
                print(f"[Agent] strict mode — confirmed ({tag}).")
                _rules = self._extract_rules(action.observation)
                self.context.confirmed_vulns.append({
                    "step":           step,
                    "action":         action.action,
                    "observation":    action.observation[:500],
                    "confidence":     action.confidence,
                    "hypothesis":     hyp,
                    "verified":       iv_ok,
                    "confidence_tag": tag,
                    "rules":          _rules,
                    "rule":           _rules[0] if _rules else "UNKNOWN",
                })
                return True

            else:  # "flexible"
                # Give agent 2 steps to call injection_verifier itself.
                # Window only starts after the agent has read the file — ensures
                # the LLM sees source context before confirming.
                if self._codeql_high_found_step < 0:
                    if not self._file_read_done:
                        # Park the observation; window will start once read_file fires.
                        self._pending_high_observation = action.observation[:800]
                        print(
                            "[Agent] flexible mode — codeql HIGH found, "
                            "waiting for read_file before starting verification window."
                        )
                    else:
                        self._codeql_high_found_step = self.context.steps_taken
                        self._codeql_high_observation = action.observation[:800]
                        print(
                            "[Agent] flexible mode — codeql HIGH found. "
                            "Agent has 2 steps to run injection_verifier."
                        )

        # flexible static fallback: window elapsed → confirm on AST alone
        # Blocked if critique already declared the finding a false positive.
        if (
            self._verification_mode == "flexible"
            and self._codeql_unconfirmed
            and self._codeql_high_found_step >= 0
            and (self.context.steps_taken - self._codeql_high_found_step) >= 2
        ):
            if self._codeql_high_rejected:
                print("[Agent] Verification window elapsed — NOT confirming (critique rejected).")
                self._codeql_unconfirmed       = False
                self._codeql_high_found_step   = -100
                self._codeql_high_observation  = ""
                self._codeql_high_rejected     = False
                return False
            print("[Agent] Verification window elapsed — confirming (AST-only).")
            self._codeql_unconfirmed = False
            _rules = self._extract_rules(self._codeql_high_observation)
            self.context.confirmed_vulns.append({
                "step":           self._codeql_high_found_step,
                "action":         "codeql_query",
                "observation":    self._codeql_high_observation,
                "confidence":     action.confidence,
                "hypothesis":     hyp,
                "verified":       False,
                "confidence_tag": "AST-only",
                "rules":          _rules,
                "rule":           _rules[0] if _rules else "UNKNOWN",
            })
            return True

        # dynamic confirm: injection_verifier returned VULN_CONFIRMED
        if dynamic_confirmed:
            self._codeql_unconfirmed      = False
            self._codeql_high_found_step  = -100
            self._codeql_high_observation = ""
            self._codeql_high_rejected    = False
            print("[Agent] Dynamically confirmed via injection_verifier — stopping early.")
            _rules = self._extract_rules(action.observation)
            self.context.confirmed_vulns.append({
                "step":           step,
                "action":         action.action,
                "observation":    action.observation[:500],
                "confidence":     action.confidence,
                "hypothesis":     hyp,
                "verified":       True,
                "confidence_tag": "AST+injection_verifier",
                "rules":          _rules,
                "rule":           _rules[0] if _rules else "UNKNOWN",
            })
            return True

        if action.action == "FINISH":
            print("[Agent] Executor signalled FINISH.")
            return True

        return False

    def _apply_advisor_advice(self) -> None:
        advice = self.context.get_pending_advisor_advice()
        if not advice:
            return

        direction = advice.strategic_direction or advice.strategy
        print(f"[Advisor] Applying advice: {direction[:120]}")

        if direction:
            self.context.set_advisor_guidance(direction)

        if advice.adjust_confidence_threshold is not None:
            self.confidence_threshold = max(
                0.3, min(0.9, advice.adjust_confidence_threshold)
            )
            print(f"[Advisor] Confidence threshold → {self.confidence_threshold:.2f}")

        for hyp in advice.raw.get("suggested_hypotheses", []):
            if hyp:
                self.context.add_hypothesis(hyp)
                print(f"[Advisor] Suggested hypothesis: {hyp[:80]}")

        for hyp in advice.raw.get("blacklist_hypotheses", []):
            if hyp:
                self.context.blacklist_hypothesis(hyp)
                print(f"[Advisor] Blacklisted: {hyp[:80]}")

        self.context.clear_pending_advisor_advice()

    # ------------------------------------------------------------------
    # Auto-verification — thin wrappers; logic lives in VerificationPipeline
    # ------------------------------------------------------------------

    def _resolve_target_tool(self) -> str | None:
        return VerificationPipeline(self.tools, self.context).resolve_target_tool()

    @staticmethod
    def _extract_findings_for_verification(
        codeql_observation: str,
    ) -> list[tuple[str, str, str]]:
        return VerificationPipeline.extract_findings(codeql_observation)

    async def _auto_verify_injection(self, codeql_observation: str) -> tuple[bool, str]:
        """Return (confirmed, method_tag) from VerificationPipeline.auto_verify()."""
        return await VerificationPipeline(self.tools, self.context).auto_verify(codeql_observation)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Parsing — thin wrappers; pure logic lives in _parser.py
    # ------------------------------------------------------------------

    def _extract_json(self, raw: str) -> str | None:
        return extract_json(raw)

    def _fallback_parse(self, raw: str) -> AgentStep:
        return fallback_parse(raw)

    def _parse_response(self, raw: str) -> AgentStep:
        step, had_error = parse_response(raw)
        if had_error:
            self.context.increment_parse_errors()
        return step

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_urgent_verifier_note(self) -> str:
        """Injected into the prompt when codeql HIGH is unconfirmed and time is running out.
        Only active in flexible mode — strict/none handle verification programmatically."""
        if self._verification_mode != "flexible":
            return ""
        if not self._codeql_unconfirmed or self._codeql_high_found_step < 0:
            return ""
        steps_left = 2 - (self.context.steps_taken - self._codeql_high_found_step)
        if steps_left <= 0:
            return ""
        target = self.context.target_path
        return (
            f"\n\n=== URGENT — {steps_left} step(s) left to verify ===\n"
            "codeql_query found HIGH findings. You MUST call injection_verifier NOW.\n"
            "Use the field name from the GIT_OPTION_INJECTION finding "
            "(e.g. 'author', 'grep', 'since') with payload '--upload-pack=test' "
            "and expected_behavior 'reject'.\n"
            f"CRITICAL: include base_overrides to run on the real file:\n"
            f'  "base_overrides": {{"file": {target!r}}}\n'
            "After injection_verifier, use FINISH.\n"
            "=========================================\n"
        )

    def _build_prompt(self) -> str:
        ctx    = self.context
        recent = ctx.get_recent_history(MAX_HISTORY_STEPS)

        budget_line = (
            f"Budget used : ${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd:.2f}\n"
            if self.budget else ""
        )

        try:
            import os
            target_lines = sum(1 for _ in open(ctx.target_path, errors="replace"))
        except Exception:
            target_lines = "unknown"

        hint_block = (
            f"[HINT FROM PREVIOUS MODEL]\n{self._initial_hint}\n\n"
            if self._initial_hint else ""
        )
        return (
            "You are Keryx Executor — a precise vulnerability hunter analyzing Python code.\n"
            f"{hint_block}"
            f"Target file  : {ctx.target_path}  ({target_lines} lines total)\n"
            f"Hypotheses   : {len(ctx.hypotheses)}\n"
            f"Confirmed    : {len(ctx.confirmed_vulns)}\n"
            f"{budget_line}"
            "\nRecent history (last few steps):\n"
            f"{recent}\n\n"
            "Known hypotheses (do NOT repeat these):\n"
            f"{ctx.get_deduplicated_hypotheses()}\n\n"
            "Available actions: " + ", ".join(sorted(VALID_ACTIONS)) + "\n\n"
            "Output ONLY a JSON object in this exact format:\n"
            '{"thought": "your reasoning", "action": "read_file", '
            '"action_input": {"path": "/full/path", "start_line": 1, "end_line": 100, '
            '"hypothesis": "one-line vuln hypothesis"}, "confidence": 0.7}\n\n'
            "Rules:\n"
            f"- Target file is {target_lines} lines — use start_line/end_line to read specific sections\n"
            "- For read_file: 'path' key required; use start_line/end_line (1-based) to navigate\n"
            "- For git_blame: 'file' key required\n"
            "- For codeql_query: 'path' key required — runs real AST static analysis, returns ALL findings at once\n"
            "- For injection_verifier: 'target_tool', 'inject_field', 'payload', 'expected_behavior' required.\n"
            f"  Also include 'base_overrides': {{'file': {self.context.target_path!r}}} so the target\n"
            "  tool runs on the real file instead of a placeholder.\n"
            "  Example: {\"action\": \"injection_verifier\", \"action_input\": {\"target_tool\": \"git_blame\",\n"
            f"    \"inject_field\": \"author\", \"payload\": \"--upload-pack=test\", \"expected_behavior\": \"reject\",\n"
            f"    \"base_overrides\": {{\"file\": {self.context.target_path!r}}}}}\n"
            "- STRATEGY:\n"
            "  1. Run codeql_query on the full target file FIRST — returns all findings in one call.\n"
            "  2. MANDATORY (flexible mode): call read_file on the lines around EVERY HIGH finding.\n"
            "     Before confirming, explicitly answer in your 'thought':\n"
            "       (a) Is this production code or a test fixture / mock / example?\n"
            "           Signs of test code: file path contains 'test', 'mock', 'fixture', 'example',\n"
            "           'sample', 'demo'; variable names like MOCK_*, FAKE_*, TEST_*; values like\n"
            "           'CATABC...', 'dummy', 'placeholder', 'changeme', 'example_key'.\n"
            "       (b) Is the vulnerable value actually reachable from untrusted input?\n"
            "     If the finding is in test code OR the value is an obvious placeholder → do NOT confirm.\n"
            "  3. MANDATORY: if codeql found GIT_OPTION_INJECTION, SUBPROCESS_SHELL_TRUE, or\n"
            "     UNSANITIZED_SUBPROCESS_ARG — you MUST call injection_verifier before finishing.\n"
            "     Use the exact field and a relevant payload (e.g. '--upload-pack=test' for git flags,\n"
            "     '../../../etc/passwd' for path fields). Set expected_behavior to 'reject'.\n"
            "     Always include base_overrides with the target file path (see example above).\n"
            "  4. Only use FINISH after you have read the relevant code section.\n"
            "  Do NOT confirm based on codeql output alone — always read the code first.\n"
            "- Always include a 'hypothesis' describing what vulnerability you suspect\n"
            "- Read DIFFERENT sections each step — do not re-read the same lines\n"
            "- When done investigating, use FINISH action\n"
            + self._build_urgent_verifier_note()
        )

    def _build_prompt_with_guidance(self) -> str:
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

        stagnation      = self._steps_since_new_hypothesis >= 5
        low_conf_streak = self._consecutive_low_confidence >= 3
        level_elevated  = self.context.escalation_level >= 2
        unconfirmed_codeql = self._codeql_unconfirmed

        return stagnation or low_conf_streak or level_elevated or unconfirmed_codeql

    # ------------------------------------------------------------------
    # Checkpoint — JSON only, no pickle
    # ------------------------------------------------------------------

    def _checkpoint_path(self, target_path: str) -> Path:
        key = hashlib.sha256(target_path.encode()).hexdigest()[:12]
        return self.checkpoint_dir / f"checkpoint_{key}.json"

    def _save_checkpoint(self, target_path: str) -> None:
        """JSON serialisation via SharedContext.to_dict() — no pickle."""
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
        """Load from JSON — no arbitrary code execution."""
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
        """Delegates to the module-level pure function in _loop_guards.
        Kept for backward-compatibility with existing callers and tests.
        """
        return observation_confirms_vuln(observation)

    def _generate_final_report(self) -> dict[str, Any]:
        from dataclasses import asdict
        clusters = self.context.cluster_hypotheses()
        return {
            "status":              "completed",
            "target":              self.context.target_path,
            "steps_taken":         self.context.steps_taken,
            "max_steps":           self.max_steps,
            "confirmed_vulns":     self.context.confirmed_vulns,
            "hypotheses":          list(self.context.hypotheses),
            "hypotheses_count":    len(self.context.hypotheses),
            "hypothesis_clusters": clusters,
            "blacklisted":         list(self.context.blacklist),
            "evidence":            [asdict(e) for e in self.context.evidence],
            "critiques":           list(self.context._critiques),
            "advisor_calls":       self.advisor_manager.calls_made,
            "parse_errors":        self.context.parse_errors,
            "budget":              self.budget.to_dict() if self.budget else None,
            "model": {
                "name":     self.executor.model_name,
                "is_local": self.executor.is_local,
            },
            "summary": self.context.build_summary(),
        }
