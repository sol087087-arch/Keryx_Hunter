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
from typing import Any, Literal

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
    "injection_verifier",
    "read_file",
    "rag_search",
    "FINISH",
    "NO_ACTION",
])

# Tools whose output can confirm a real vulnerability (source check for FIX 7)
_VULN_CONFIRMING_TOOLS = frozenset(["gdb_analyze", "fuzzer_run", "codeql_query", "injection_verifier"])

_ESCALATION_COOLDOWN    = 5
_MAX_HISTORY_STEPS      = 12
_TOOL_TIMEOUT_SECONDS   = 30.0

# Data-driven payload table for strict-mode auto-verification.
# Maps AST finding rule → injection payload.  None = static-only (no verifier call).
_PAYLOAD_MAP: dict[str, str | None] = {
    "GIT_OPTION_INJECTION":        "--upload-pack=test",
    "SUBPROCESS_SHELL_TRUE":       "$(id)",
    "UNSANITIZED_SUBPROCESS_ARG":  "../../etc/passwd",
    "OPEN_USER_PATH":              "../../etc/passwd",
    "SUBPROCESS_EXEC_STARRED":     "../../etc/passwd",
    "HARDCODED_SECRET":            None,   # static-only, nothing to inject
}

# Maps Python variable names (from AST) → likely action_input keys.
# Needed because tool internals use `action_input.get("file")` but the
# local variable inside the tool is often named `file_path`.
_VARNAME_TO_FIELD: dict[str, str] = {
    "file_path": "file",
    "filepath":  "file",
    "path":      "path",
    "author":    "author",
    "grep":      "grep",
    "since":     "since",
    "until":     "until",
}


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
        self._steps_since_new_hypothesis:   int = 0   # P4: stagnation tracker
        self._consecutive_low_confidence:   int = 0   # P4: low-conf streak
        self._codeql_unconfirmed:           bool = False  # P4: AST found HIGH but not confirmed
        self._codeql_high_observation:     str  = ""    # observation snapshot from the triggering codeql step

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
            _prev_hyp_count = len(self.context.hypotheses)   # P4: stagnation baseline
            if action.action not in ("NO_ACTION", "FINISH"):
                try:
                    observation = await asyncio.wait_for(
                        self._execute_tool_async(action.action, action.action_input),
                        timeout=_TOOL_TIMEOUT_SECONDS,
                    )
                    action.observation = observation
                    self.context.add_step(action, observation)
                    print(f"[Tool] {action.action} completed in {time.time() - action.timestamp:.2f}s")

                    # Register hypothesis before critique so it can be blacklisted
                    hyp = action.action_input.get("hypothesis", "")
                    if hyp:
                        self.context.add_hypothesis(hyp)
                        # Record non-trivial observations as evidence
                        if len(observation) > 50:
                            loc = action.action_input.get("path") or action.action_input.get("file") or action.action
                            self.context.add_evidence(location=str(loc), description=hyp[:200])

                    critique = await self._self_critique_async(hypothesis=hyp, observation=observation)
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

            # ── P4: update advisor-trigger state ─────────────────────────
            # Stagnation: reset counter if any new hypothesis was added this step
            if len(self.context.hypotheses) > _prev_hyp_count:
                self._steps_since_new_hypothesis = 0
            else:
                self._steps_since_new_hypothesis += 1

            # Confidence streak: reset on any step with confidence >= 0.3
            if action.confidence >= 0.3:
                self._consecutive_low_confidence = 0
            else:
                self._consecutive_low_confidence += 1

            # Unconfirmed codeql HIGH: flag so advisor triggers on next chance
            if (
                action.action == "codeql_query"
                and self._observation_confirms_vuln(action.observation)
            ):
                self._codeql_unconfirmed = True

            # ── Apply advisor advice ──────────────────────────────────────
            if self.context.has_pending_advisor_advice():
                self._apply_advisor_advice()

            # ── Escalate ─────────────────────────────────────────────────
            await self._maybe_escalate()

            # ── Early stop — vuln confirmed ───────────────────────────────
            # codeql_query (AST analyzer) is objective on HIGH findings.
            # After AST confirms, auto-run injection_verifier (0 extra LLM tokens)
            # to attempt dynamic proof before declaring the vuln confirmed.
            # Other vuln-confirming tools (fuzzer, gdb, injection_verifier) require
            # high confidence + crash/confirmed signal.
            ast_found = (
                action.action == "codeql_query"
                and self._observation_confirms_vuln(action.observation)
            )
            # injection_verifier uses a lower confidence bar — VULN_CONFIRMED in output
            # is already an objective signal, no need for model confidence >= 0.85.
            dynamic_confirmed = (
                (
                    action.confidence >= 0.85
                    and action.action in _VULN_CONFIRMING_TOOLS - {"codeql_query", "injection_verifier"}
                    and self._observation_confirms_vuln(action.observation)
                )
                or (
                    action.action == "injection_verifier"
                    and self._observation_confirms_vuln(action.observation)
                )
            )

            # ── Three-mode confirmation dispatch ──────────────────────────
            if ast_found:
                mode = self._verification_mode

                if mode == "none":
                    # Immediate static confirm — no verifier, no window.
                    # Designed for fast batch scans where speed > precision.
                    self._codeql_unconfirmed = False
                    print("[Agent] none mode — static confirm (AST-only).")
                    self.context.confirmed_vulns.append({
                        "step":           step,
                        "action":         action.action,
                        "observation":    action.observation[:500],
                        "confidence":     action.confidence,
                        "hypothesis":     hyp,
                        "verified":       False,
                        "confidence_tag": "AST-only",
                    })
                    break

                elif mode == "strict":
                    # Auto-verify via injection_verifier — no extra LLM step.
                    # Designed for CI/CD: deterministic, reproducible.
                    iv_ok = await self._auto_verify_injection(action.observation)
                    tag = "AST+injection_verifier" if iv_ok else "AST-only"
                    self._codeql_unconfirmed = False
                    print(f"[Agent] strict mode — confirmed ({tag}).")
                    self.context.confirmed_vulns.append({
                        "step":           step,
                        "action":         action.action,
                        "observation":    action.observation[:500],
                        "confidence":     action.confidence,
                        "hypothesis":     hyp,
                        "verified":       iv_ok,
                        "confidence_tag": tag,
                    })
                    break

                else:  # "flexible"
                    # Give agent 2 steps to call injection_verifier itself.
                    # After the window, fall through to static_fallback below.
                    if self._codeql_high_found_step < 0:
                        self._codeql_high_found_step = self.context.steps_taken
                        self._codeql_high_observation = action.observation[:800]
                        print(
                            "[Agent] flexible mode — codeql HIGH found. "
                            "Agent has 2 steps to run injection_verifier."
                        )

            # flexible static fallback: window elapsed → confirm on AST alone
            static_fallback = (
                self._verification_mode == "flexible"
                and self._codeql_unconfirmed
                and self._codeql_high_found_step >= 0
                and (self.context.steps_taken - self._codeql_high_found_step) >= 2
            )
            if static_fallback:
                print("[Agent] Verification window elapsed — confirming (AST-only).")
                self._codeql_unconfirmed = False
                self.context.confirmed_vulns.append({
                    "step":           self._codeql_high_found_step,
                    "action":         "codeql_query",
                    "observation":    self._codeql_high_observation,
                    "confidence":     action.confidence,
                    "hypothesis":     hyp,
                    "verified":       False,
                    "confidence_tag": "AST-only",
                })
                break

            # dynamic confirm: injection_verifier returned VULN_CONFIRMED
            if dynamic_confirmed:
                self._codeql_unconfirmed = False
                self._codeql_high_found_step = -100
                self._codeql_high_observation = ""
                print("[Agent] Dynamically confirmed via injection_verifier — stopping early.")
                self.context.confirmed_vulns.append({
                    "step":           step,
                    "action":         action.action,
                    "observation":    action.observation[:500],
                    "confidence":     action.confidence,
                    "hypothesis":     hyp,
                    "verified":       True,
                    "confidence_tag": "AST+injection_verifier",
                })
                break

            if action.action == "FINISH":
                print("[Agent] Executor signalled FINISH.")
                break

            # ── Checkpoint every 5 steps ──────────────────────────────────
            # Saves after the step completes (steps_taken already incremented).
            # On resume, the agent continues from this step count.
            # Checkpoint format: JSON only — no pickle, no arbitrary execution.
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

    async def _self_critique_async(self, hypothesis: str = "", observation: str = "") -> str:
        hyp_line = f"Hypothesis under review: {hypothesis}\n" if hypothesis else ""
        obs_line = f"Evidence gathered:\n{observation[:6000]}\n\n" if observation else ""
        prompt = (
            "You are reviewing a specific vulnerability hypothesis about analyzed source code.\n"
            "The 'read_file' / 'git_blame' calls are just tools for gathering evidence — "
            "do NOT judge those as vulnerabilities themselves.\n\n"
            f"{hyp_line}"
            f"{obs_line}"
            "Answer concisely:\n"
            "1. Does the evidence actually support this hypothesis? (YES / PARTIALLY / NO)\n"
            "2. Is the vulnerable code path reachable in normal operation?\n"
            "3. Your verdict: 'GENUINE CONCERN', 'NEEDS MORE EVIDENCE', or "
            "'THIS IS A FALSE POSITIVE' (only if the code clearly does not have this issue).\n"
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
        # Only allow confidence boost here — blacklisting is not appropriate.
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

        elif "needs more evidence" in critique_lower or "insufficient" in critique_lower:
            self.context.request_additional_evidence(action.action)
            print("[Critique] Additional evidence requested")

        elif "confirmed" in critique_lower or "likely" in critique_lower or "genuine" in critique_lower:
            action.confidence = min(action.confidence + 0.1, 0.95)
            print(f"[Critique] Confidence boosted to {action.confidence:.2f}")

        self.context.add_critique(critique)

    def _apply_advisor_advice(self) -> None:
        advice = self.context.get_pending_advisor_advice()
        if not advice:
            return

        direction = advice.strategic_direction or advice.strategy
        print(f"[Advisor] Applying advice: {direction[:120]}")

        # Strategic direction → injected into next prompt
        if direction:
            self.context.set_advisor_guidance(direction)

        # Confidence threshold adjustment
        if advice.adjust_confidence_threshold is not None:
            self.confidence_threshold = max(
                0.3, min(0.9, advice.adjust_confidence_threshold)
            )
            print(f"[Advisor] Confidence threshold → {self.confidence_threshold:.2f}")

        # Suggested hypotheses (from AdvisorResponse rich fields, stored in raw)
        for hyp in advice.raw.get("suggested_hypotheses", []):
            if hyp:
                self.context.add_hypothesis(hyp)
                print(f"[Advisor] Suggested hypothesis: {hyp[:80]}")

        # Advisor-driven blacklist
        for hyp in advice.raw.get("blacklist_hypotheses", []):
            if hyp:
                self.context.blacklist_hypothesis(hyp)
                print(f"[Advisor] Blacklisted: {hyp[:80]}")

        self.context.clear_pending_advisor_advice()

    # ------------------------------------------------------------------
    # Auto-verification (strict mode — no LLM call, pure tool pipeline)
    # ------------------------------------------------------------------

    def _resolve_target_tool(self) -> str | None:
        """
        Infer which registered tool corresponds to the file being analyzed.

        Strategy B+A:
        1. (A) Exact stem match: git_blame.py → "git_blame" — check in toolbox.
        2. (A normalized) Underscore/hyphen variants.
        3. (B) Partial match: any registered tool name contained in the stem.
        4. Warn and return None if nothing found.
        """
        available = set(self.tools.list_tools())
        stem = Path(self.context.target_path).stem          # "git_blame"

        # A — exact
        if stem in available:
            return stem
        # A — normalized (hyphens → underscores)
        normalized = stem.replace("-", "_").replace(" ", "_")
        if normalized in available:
            return normalized
        # B — partial: "git_blame_wrapper" contains "git_blame"
        for tool in available:
            if tool in stem or stem in tool:
                return tool

        print(
            f"[WARN] target_tool not resolved for {stem!r} — "
            "skipping dynamic verification"
        )
        return None

    @staticmethod
    def _extract_findings_for_verification(
        codeql_observation: str,
    ) -> list[tuple[str, str, str]]:
        """
        Parse HIGH findings from codeql output.

        Returns list of (rule, inject_field, payload) tuples ready for
        injection_verifier.  Skips rules with no payload (e.g. HARDCODED_SECRET).
        """
        import re

        # Match lines like: [HIGH] RULE_NAME @ line N: <context snippet>
        high_lines = re.findall(
            r'\[HIGH\] (\w+) @.*?:\s*(.+?)(?:\n|$)', codeql_observation
        )

        results: list[tuple[str, str, str]] = []
        seen_rules: set[str] = set()

        for rule, context_text in high_lines:
            if rule in seen_rules:
                continue          # one attempt per rule is enough
            payload = _PAYLOAD_MAP.get(rule)
            if payload is None:
                continue          # HARDCODED_SECRET etc. — static-only

            # Extract the injectable field name from the context snippet
            inject_field: str | None = None

            if rule == "GIT_OPTION_INJECTION":
                m = re.search(r'cmd\.extend\(\["--[\w-]+",\s*(\w+)\]', context_text)
                if m:
                    inject_field = _VARNAME_TO_FIELD.get(m.group(1), m.group(1))

            elif rule in ("UNSANITIZED_SUBPROCESS_ARG", "SUBPROCESS_EXEC_STARRED"):
                m = re.search(r'cmd\.(?:append|exec)\((\w+)\)', context_text)
                if m:
                    inject_field = _VARNAME_TO_FIELD.get(m.group(1), m.group(1))

            elif rule == "OPEN_USER_PATH":
                m = re.search(r'open\((\w+)\)', context_text)
                if m:
                    inject_field = _VARNAME_TO_FIELD.get(m.group(1), m.group(1))

            elif rule == "SUBPROCESS_SHELL_TRUE":
                inject_field = "file"   # best-effort for shell=True calls

            if inject_field:
                results.append((rule, inject_field, payload))
                seen_rules.add(rule)

        return results

    async def _auto_verify_injection(self, codeql_observation: str) -> bool:
        """
        Strict-mode auto-verification: resolve target tool, extract HIGH
        findings, probe each via injection_verifier — zero extra LLM tokens.

        Returns True if ANY finding is dynamically confirmed (VULN_CONFIRMED).
        """
        target_tool = self._resolve_target_tool()
        if target_tool is None:
            return False

        findings = self._extract_findings_for_verification(codeql_observation)
        if not findings:
            print("[AutoVerify] No verifiable HIGH findings — static-only confirmation.")
            return False

        for rule, inject_field, payload in findings:
            print(
                f"[AutoVerify] {target_tool!r} | rule={rule} | "
                f"field={inject_field!r} | payload={payload!r}"
            )
            try:
                result = await asyncio.wait_for(
                    self.tools.execute_async(
                        "injection_verifier",
                        {
                            "target_tool":       target_tool,
                            "inject_field":      inject_field,
                            "payload":           payload,
                            "expected_behavior": "reject",
                            "base_overrides":    {"file": self.context.target_path},
                        },
                    ),
                    timeout=20.0,
                )
            except Exception as exc:
                print(f"[AutoVerify] Verifier call failed ({rule}): {exc}")
                continue

            observation = result.output if hasattr(result, "output") else str(result)
            self.context.add_step(
                AgentStep(
                    thought=f"auto_verify_{rule}",
                    action="injection_verifier",
                    action_input={
                        "target_tool": target_tool,
                        "inject_field": inject_field,
                        "payload": payload,
                    },
                    confidence=0.9 if "VULN_CONFIRMED" in observation else 0.35,
                ),
                observation,
            )
            self.context.add_evidence(
                location=f"injection_verifier:{target_tool}.{inject_field}",
                description=f"strict-mode auto-verification of {rule}",
            )

            if "VULN_CONFIRMED" in observation:
                print(f"[AutoVerify] CONFIRMED — {target_tool}.{inject_field}")
                return True
            print(f"[AutoVerify] REJECTED — {target_tool}.{inject_field}")

        return False

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
        recent = ctx.get_recent_history(_MAX_HISTORY_STEPS)

        # FIX 5: budget line built separately — ternary mid-concatenation
        # causes the remaining string literals to be silently dropped.
        budget_line = (
            f"Budget used : ${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd:.2f}\n"
            if self.budget else ""
        )

        try:
            import os
            target_lines = sum(1 for _ in open(ctx.target_path, errors="replace"))
        except Exception:
            target_lines = "unknown"

        return (
            "You are Keryx Executor — a precise vulnerability hunter analyzing Python code.\n"
            f"Target file  : {ctx.target_path}  ({target_lines} lines total)\n"
            f"Hypotheses   : {len(ctx.hypotheses)}\n"
            f"Confirmed    : {len(ctx.confirmed_vulns)}\n"
            f"{budget_line}"
            "\nRecent history (last few steps):\n"
            f"{recent}\n\n"
            "Known hypotheses (do NOT repeat these):\n"
            f"{ctx.get_deduplicated_hypotheses()}\n\n"
            "Available actions: " + ", ".join(sorted(_VALID_ACTIONS)) + "\n\n"
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
            "  2. Use read_file only to inspect specific lines cited in codeql output.\n"
            "  3. MANDATORY: if codeql found GIT_OPTION_INJECTION, SUBPROCESS_SHELL_TRUE, or\n"
            "     UNSANITIZED_SUBPROCESS_ARG — you MUST call injection_verifier before finishing.\n"
            "     Use the exact field and a relevant payload (e.g. '--upload-pack=test' for git flags,\n"
            "     '../../../etc/passwd' for path fields). Set expected_behavior to 'reject'.\n"
            "     Always include base_overrides with the target file path (see example above).\n"
            "  4. Only use FINISH after injection_verifier has run on at least one HIGH finding.\n"
            "  Do NOT call read_file multiple times before codeql_query.\n"
            "- Always include a 'hypothesis' describing what vulnerability you suspect\n"
            "- Read DIFFERENT sections each step — do not re-read the same lines\n"
            "- When done investigating, use FINISH action\n"
            + self._build_urgent_verifier_note()
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

        # P4: four specific trigger conditions (fire on ANY one)
        # 1. Stagnation — no new hypothesis added in the last 5 steps
        stagnation = self._steps_since_new_hypothesis >= 5

        # 2. Confidence streak — confidence < 0.3 for 3 consecutive steps
        low_conf_streak = self._consecutive_low_confidence >= 3

        # 3. Escalation level already elevated by the orchestrator
        level_elevated = self.context.escalation_level >= 2

        # 4. AST analyzer found HIGH findings but none have been confirmed yet
        unconfirmed_codeql = self._codeql_unconfirmed

        return stagnation or low_conf_streak or level_elevated or unconfirmed_codeql

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
            # AST analyzer signals
            "[HIGH]",
            "GIT_OPTION_INJECTION",
            "SUBPROCESS_SHELL_TRUE",
            "HARDCODED_SECRET",
        )
        return any(s in observation for s in signals)

    def _generate_final_report(self) -> dict[str, Any]:
        from dataclasses import asdict
        clusters = self.context.cluster_hypotheses()
        return {
            "status":            "completed",
            "target":            self.context.target_path,
            "steps_taken":       self.context.steps_taken,
            "max_steps":         self.max_steps,
            "confirmed_vulns":   self.context.confirmed_vulns,
            "hypotheses":        list(self.context.hypotheses),
            "hypotheses_count":  len(self.context.hypotheses),
            "hypothesis_clusters": clusters,
            "blacklisted":       list(self.context.blacklist),
            "evidence":          [asdict(e) for e in self.context.evidence],
            "critiques":         list(self.context._critiques),
            "advisor_calls":     self.advisor_manager.calls_made,
            "parse_errors":      self.context.parse_errors,
            "budget":            self.budget.to_dict() if self.budget else None,
            "model": {
                "name":     self.executor.model_name,
                "is_local": self.executor.is_local,
            },
            "summary": self.context.build_summary(),
        }
