# Core ReAct agent for KeryxHunter - Executor + Advisor escalation
# Sovereign, air-gapped, robust.
# CRITICAL FIXES: Advisor influence, tool timeouts, budget control, context windowing
import json
import asyncio
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from threading import Lock
from ..models.interface import ModelInterface
from ..tools.toolbox import ToolBox
from ..advisors.manager import AdvisorManager
from ..shared_context import SharedContext
@dataclass
class AgentStep:
    """Single step in the ReAct loop."""
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
@dataclass
class BudgetController:
    """Track and enforce API/model budget for cloud models."""
    max_cost_usd: float = 10.0
    current_cost: float = 0.0
    max_calls: int = 500
    calls_made: int = 0
   
    def can_proceed(self, model: ModelInterface) -> bool:
        """Check if we can make another call within budget."""
        if self.calls_made >= self.max_calls:
            return False
       
        estimated_cost = self._estimate_call_cost(model)
        return self.current_cost + estimated_cost <= self.max_cost_usd
   
    def record_call(self, model: ModelInterface, tokens_in: int = 1000, tokens_out: int = 500) -> None:
        """Record a model call for budget tracking."""
        cost_per_1k_in = model.cost_per_1k_input_tokens or 0.0
        cost_per_1k_out = model.cost_per_1k_output_tokens or 0.0
        call_cost = (tokens_in / 1000) * cost_per_1k_in + (tokens_out / 1000) * cost_per_1k_out
        self.current_cost += call_cost
        self.calls_made += 1
   
    def _estimate_call_cost(self, model: ModelInterface) -> float:
        """Estimate cost of next call (conservative)."""
        cost_in = model.cost_per_1k_input_tokens or 0.0
        cost_out = model.cost_per_1k_output_tokens or 0.0
        # Assume 2k input, 1k output as conservative estimate
        return (2.0 * cost_in) + (1.0 * cost_out)
   
    @property
    def remaining_budget(self) -> float:
        return self.max_cost_usd - self.current_cost
   
    def to_dict(self) -> Dict:
        return {
            "max_cost_usd": self.max_cost_usd,
            "current_cost": self.current_cost,
            "calls_made": self.calls_made,
            "remaining_budget": self.remaining_budget
        }
# ---------------------------------------------------------------------------
# GBNF grammar — complete, tested against llama.cpp grammar validator
# ---------------------------------------------------------------------------
_OUTPUT_GRAMMAR = r"""
root ::= "{" ws kv-thought "," ws kv-action "," ws kv-input "," ws kv-conf ws "}"
kv-thought ::= ""thought"" ws ":" ws string
kv-action ::= ""action"" ws ":" ws string
kv-input ::= ""action_input"" ws ":" ws object
kv-conf ::= ""confidence"" ws ":" ws number
value ::= string | number | object | array | "true" | "false" | "null"
object ::= "{" ws ( string ws ":" ws value ( "," ws string ws ":" ws value )* )? ws "}"
array ::= "[" ws ( value ( "," ws value )* )? ws "]"
string ::= """ ( [^\"\x00-\x1f] | "\" ( ["\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* """
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [+-]? [0-9]+ )?
ws ::= [ \t\n\r]*
"""
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
# How many steps must pass before Advisor can be called again.
_ESCALATION_COOLDOWN = 5
# Maximum history steps before summarization
_MAX_HISTORY_STEPS = 12
# Tool execution timeout in seconds
_TOOL_TIMEOUT_SECONDS = 30.0
class KeryxAgent:
    """
    Main Executor agent.
    Runs the ReAct loop, calls tools, self-critiques, escalates to Advisor.
    Supports checkpoint/resume so a crash at step 40 doesn't mean starting over.
    """
    def **init**(
        self,
        executor_model: ModelInterface,
        advisor_manager: AdvisorManager,
        toolbox: ToolBox,
        max_steps: int = 50,
        confidence_threshold: float = 0.65,
        max_prompt_chars: int = 32_000, # Reduced from 48k for local models
        checkpoint_dir: Optional[Path] = None,
        generate_timeout: float = 90.0,
        budget_usd: Optional[float] = None,
    ):
        self.executor = executor_model
        self.advisor_manager = advisor_manager
        self.tools = toolbox
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.max_prompt_chars = max_prompt_chars
        self.generate_timeout = generate_timeout
        self.checkpoint_dir = checkpoint_dir or Path(".keryx_checkpoints")
        self.context: Optional[SharedContext] = None
       
        # Budget controller (None = unlimited)
        self.budget = BudgetController(max_cost_usd=budget_usd) if budget_usd else None
       
        # Steps since last escalation — starts at cooldown so first check is allowed.
        self._steps_since_escalation: int = _ESCALATION_COOLDOWN
       
        # Health tracking
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = 3
    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        target_path: str,
        capability: str = "deep_reasoning",
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Main hunt loop.
        Args:
            target_path: Path to codebase to analyze.
            capability: Capability profile (see capabilities.yaml).
            resume: If True and a checkpoint exists, resume from it.
        """
        self.context = self._load_checkpoint(target_path) if resume else None
        if self.context is None:
            self.context = SharedContext(target_path=target_path, capability=capability)
        print(f"[Agent] Hunt starting | target={target_path} | capability={capability} "
              f"| step={self.context.steps_taken}/{self.max_steps}")
       
        if self.budget:
            print(f"[Agent] Budget: ${self.budget.max_cost_usd} | Mode: {'airgapped' if self.executor.is_local else 'cloud'}")
        while self.context.steps_taken < self.max_steps:
            # ── 0. Budget check ─────────────────────────────────────────
            if self.budget and not self.budget.can_proceed(self.executor):
                print(f"[Agent] Budget exhausted (${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd}) — finishing.")
                break
           
            # ── 0.5 Health check ─────────────────────────────────────────
            if not await self._check_model_health():
                if self._consecutive_failures >= self._max_consecutive_failures:
                    print("[Agent] Model unhealthy — cannot continue.")
                    break
                continue
           
            step = self.context.steps_taken + 1
            print(f"[Step {step}/{self.max_steps}]")
            # ── 1. Build prompt with advisor guidance if available ──────
            if self.context.has_pending_advisor_advice():
                prompt = self._build_prompt_with_guidance()
                print("[Agent] Including Advisor guidance in prompt")
            else:
                prompt = self._build_prompt()
           
            # Trim if over limit
            if len(prompt) > self.max_prompt_chars:
                self.context.trim_history()
                prompt = self._build_prompt()
            # ── 2. Generate with timeout ──────────────────────────────────
            action = await self._safe_generate(prompt)
            self.context.steps_taken += 1
            self._steps_since_escalation += 1
           
            # Track budget if applicable
            if self.budget and action:
                self.budget.record_call(self.executor, tokens_in=len(prompt)//4, tokens_out=500)
            if action is None:
                self._consecutive_failures += 1
                # Emergency fallback action
                action = AgentStep(
                    thought="emergency_fallback_model_unresponsive",
                    action="NO_ACTION",
                    action_input={},
                    confidence=0.1,
                    timestamp=time.time()
                )
            else:
                self._consecutive_failures = 0
            # ── 3. Validate action name ───────────────────────────────────
            if action.action not in _VALID_ACTIONS:
                print(f"[WARN] Unknown action '{action.action}' — treating as NO_ACTION")
                action.action = "NO_ACTION"
            # ── 4. Execute tool with timeout ───────────────────────────────
            if action.action not in ("NO_ACTION", "FINISH"):
                try:
                    observation = await asyncio.wait_for(
                        self._execute_tool_async(action.action, action.action_input),
                        timeout=_TOOL_TIMEOUT_SECONDS
                    )
                    action.observation = observation
                    self.context.add_step(action, observation)
                   
                    print(f"[Tool] {action.action} completed in {time.time() - action.timestamp:.2f}s")
                   
                    # Self-critique after real work only (and apply it!)
                    critique = await self._self_critique_async()
                    self._apply_critique(critique, action)
                   
                except asyncio.TimeoutError:
                    error_msg = f"Tool '{action.action}' timed out after {_TOOL_TIMEOUT_SECONDS}s"
                    print(f"[ERROR] {error_msg}")
                    action.observation = error_msg
                    action.confidence *= 0.5 # Reduce confidence on tool failure
                    self.context.add_step(action, error_msg)
                   
                except Exception as e:
                    error_msg = f"Tool '{action.action}' failed: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    action.observation = error_msg
                    action.confidence *= 0.7
                    self.context.add_step(action, error_msg)
            else:
                self.context.add_step(action, "No action taken")
            # ── 5. Apply pending advisor advice (not just store!) ──────────
            if self.context.has_pending_advisor_advice():
                self._apply_advisor_advice()
            # ── 6. Escalate if needed ─────────────────────────────────────
            await self._maybe_escalate()
            # ── 7. Early stop conditions ───────────────────────────────────
            if action.confidence > 0.9 and self._observation_confirms_vuln(action.observation):
                print("[Agent] High-confidence vulnerability confirmed — stopping early.")
                self.context.confirmed_vulns.append({
                    "step": step,
                    "action": action.action,
                    "observation": action.observation[:500],
                    "confidence": action.confidence
                })
                break
            if action.action == "FINISH":
                print("[Agent] Executor signalled FINISH.")
                break
            # ── 8. Checkpoint ─────────────────────────────────────────────
            if step % 5 == 0: # Checkpoint every 5 steps
                self._save_checkpoint(target_path)
        return self._generate_final_report()
    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    async def _check_model_health(self) -> bool:
        """Check if the model is responsive (critical for local GPU models)."""
        try:
            result = await asyncio.wait_for(
                self._generate_async("ping", max_tokens=1),
                timeout=5.0
            )
            return bool(result)
        except Exception:
            return False
    async def _safe_generate(self, prompt: str) -> Optional[AgentStep]:
        """Generate with timeout and parse. Returns None on unrecoverable failure."""
        for attempt in range(2): # one retry on parse failure
            try:
                raw = await asyncio.wait_for(
                    self._generate_async(prompt),
                    timeout=self.generate_timeout,
                )
            except asyncio.TimeoutError:
                print(f"[WARN] Generation timeout (attempt {attempt + 1})")
                self.context.increment_parse_errors()
                if attempt == 1:
                    # Last attempt - simplify prompt
                    simplified = self._simplify_prompt(prompt)
                    try:
                        raw = await asyncio.wait_for(
                            self._generate_async(simplified),
                            timeout=self.generate_timeout * 1.5,
                        )
                    except asyncio.TimeoutError:
                        return None
                else:
                    continue
            action = self._parse_response(raw)
            if action.action != "NO_ACTION" or action.thought:
                return action
        return None # both attempts failed
    async def _generate_async(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Run synchronous executor.generate() in a thread pool so it
        doesn't block the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.executor.generate(prompt, grammar=_OUTPUT_GRAMMAR, max_tokens=max_tokens)
        )
    async def _execute_tool_async(self, action: str, action_input: Dict) -> str:
        """Execute tool with timeout support."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.tools.execute, action, action_input
        )
    async def _self_critique_async(self) -> str:
        """Async self-critique so it doesn't block the loop."""
        last_step = self.context.get_last_step()
        if not last_step:
            return ""
        prompt = (
            "You are reviewing your last analysis step.\n"
            "Is the code path reachable? Any false-positive risk? "
            "Is there a simpler explanation? "
            "Output your critique in 2-3 sentences.\n\n"
            f"Step: {last_step}"
        )
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.executor.generate(prompt, max_tokens=350)),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            return "Critique timed out — proceeding with original hypothesis."
    def _apply_critique(self, critique: str, action: AgentStep) -> None:
        """Apply critique to modify context — CRITICAL FIX!"""
        critique_lower = critique.lower()
       
        if "false positive" in critique_lower or "not reachable" in critique_lower:
            # Blacklist this hypothesis
            self.context.blacklist_hypothesis(action.action_input.get("hypothesis", ""))
            print("[Critique] Hypothesis blacklisted as false positive")
            action.confidence *= 0.3
           
        elif "needs more evidence" in critique_lower or "insufficient" in critique_lower:
            # Request additional evidence on next step
            self.context.request_additional_evidence(action.action)
            print("[Critique] Additional evidence requested")
           
        elif "confirmed" in critique_lower or "likely" in critique_lower:
            # Increase confidence
            action.confidence = min(action.confidence + 0.1, 0.95)
            print(f"[Critique] Confidence boosted to {action.confidence:.2f}")
       
        self.context.add_critique(critique)
    def _apply_advisor_advice(self) -> None:
        """Apply advisor advice to modify execution strategy — CRITICAL FIX!"""
        advice = self.context.get_pending_advisor_advice()
        if not advice:
            return
       
        print(f"[Advisor] Applying advice: {advice.strategy[:100] if advice.strategy else 'None'}")
       
        # Modify confidence threshold based on advisor recommendation
        if advice.adjust_confidence_threshold:
            self.confidence_threshold = max(0.3, min(0.9, advice.adjust_confidence_threshold))
            print(f"[Advisor] Confidence threshold adjusted to {self.confidence_threshold}")
       
        # Add strategic direction to context for future prompts
        self.context.set_advisor_guidance(advice.strategic_direction)
       
        # Clear pending advice
        self.context.clear_pending_advisor_advice()
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify prompt when model is struggling (fallback mode)."""
        lines = prompt.split('\n')
        # Keep only essential sections
        essential = []
        for line in lines:
            if any(key in line for key in ['Target', 'Available actions', 'Output ONLY valid JSON']):
                essential.append(line)
            elif line.strip() and not line.startswith('Recent history'):
                essential.append(line)
        return '\n'.join(essential[:50]) # Limit to 50 lines
    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _extract_json(self, raw: str) -> Optional[str]:
        """Extract JSON from model response (handles markdown, extra text)."""
        raw = raw.strip()
       
        # Strip markdown code fences
        if raw.startswith(""):             parts = raw.split("")
if len(parts) >= 2:
raw = parts[1]
if raw.startswith("json"):
raw = raw[4:]
# Find first { and last }
start = raw.find('{')
end = raw.rfind('}')
if start != -1 and end != -1 and start < end:
return raw[start:end+1]
return None
def _fallback_parse(self, raw: str) -> AgentStep:
"""Fallback parsing when JSON extraction fails."""
raw_lower = raw.lower()
# Try to infer action from keywords
for action in _VALID_ACTIONS:
if action.lower() in raw_lower and action != "NO_ACTION":
return AgentStep(
thought="fallback_parse_inferred",
action=action,
action_input={},
confidence=0.3,
timestamp=time.time()
)
return AgentStep(
thought="fallback_parse_failed",
action="NO_ACTION",
action_input={},
confidence=0.1,
timestamp=time.time()
)
def _parse_response(self, raw: str) -> AgentStep:
"""Parse JSON response. Logs failures, uses fallback."""
json_str = self._extract_json(raw)
if not json_str:
print(f"[WARN] Could not extract JSON from response (len={len(raw)})")
self.context.increment_parse_errors()
return self._fallback_parse(raw)
try:
data = json.loads(json_str)
return AgentStep(
thought=str(data.get("thought", "")),
action=str(data.get("action", "NO_ACTION")),
action_input=dict(data.get("action_input", {})),
confidence=float(data.get("confidence", 0.5)),
timestamp=time.time()
)
except (json.JSONDecodeError, TypeError, ValueError) as exc:
print(f"[WARN] JSON parse failed ({exc.**class**.**name**}: {exc})")
self.context.increment_parse_errors()
return self._fallback_parse(raw)
# ------------------------------------------------------------------
# Prompt building
# ------------------------------------------------------------------
def _build_prompt(self) -> str:
ctx = self.context
recent = ctx.get_recent_history(_MAX_HISTORY_STEPS)
return (
"You are Keryx Executor — a precise, local vulnerability hunter.\n"
f"Target : {ctx.target_path}\n"
f"Scanned : {len(ctx.scanned_files)} files\n"
f"Hypotheses : {len(ctx.hypotheses)}\n"
f"Confirmed : {len(ctx.confirmed_vulns)}\n"
f"Parse errors: {ctx.parse_errors}\n"
f"Advisor calls: {self.advisor_manager.calls_made}\n"
f"Budget used : ${self.budget.current_cost:.4f} / ${self.budget.max_cost_usd:.2f}\n\n" if self.budget else ""
"Recent history (last few steps):\n"
f"{recent}\n\n"
"Known hypotheses (do NOT repeat these):\n"
f"{ctx.get_deduplicated_hypotheses()}\n\n"
"Available actions: " + ", ".join(sorted(_VALID_ACTIONS)) + "\n\n"
"Think step by step. Output ONLY valid JSON matching the grammar."
)
def _build_prompt_with_guidance(self) -> str:
"""Include advisor guidance in prompt."""
base = self._build_prompt()
guidance = self.context.get_advisor_guidance()
if guidance:
base += f"\n\n=== ADVISOR GUIDANCE ===\n{guidance}\n========================\n"
return base
# ------------------------------------------------------------------
# Escalation
# ------------------------------------------------------------------
async def _maybe_escalate(self, force: bool = False) -> None:
"""Escalate to Advisor if conditions are met."""
if not force and not self._should_escalate():
return
if not self.advisor_manager.can_advise():
print("[Agent] Advisor limit reached — continuing without advice.")
return
print("[Agent] Escalating to Advisor...")
loop = asyncio.get_event_loop()
advice = await loop.run_in_executor(
None, self.advisor_manager.get_advice, self.context
)
self.context.add_advisor_advice(advice)
self._steps_since_escalation = 0 # reset cooldown
def _should_escalate(self) -> bool:
"""
Escalate when:

Confidence is low
No progress after many steps
Too many parse errors
Cooldown has passed (prevent hammering Advisor every step)
"""
if self._steps_since_escalation < _ESCALATION_COOLDOWN:
return False
last_conf = self.context.get_last_confidence()
no_progress = (
len(self.context.hypotheses) == 0
and self.context.steps_taken > 15
)
too_many_errors = self.context.parse_errors > 3
low_confidence = last_conf is not None and last_conf < self.confidence_threshold
return low_confidence or no_progress or too_many_errors

# ------------------------------------------------------------------
# Checkpoint / resume
# ------------------------------------------------------------------
def *checkpoint_path(self, target_path: str) -> Path:
key = hashlib.sha256(target_path.encode()).hexdigest()[:12]
return self.checkpoint_dir / f"checkpoint*{key}.pkl"
def _save_checkpoint(self, target_path: str) -> None:
self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
path = self._checkpoint_path(target_path)
try:
with open(path, "wb") as f:
pickle.dump({
"context": self.context,
"budget": self.budget,
"steps_since_escalation": self._steps_since_escalation,
}, f)
except Exception as exc:
print(f"[WARN] Checkpoint save failed: {exc}")
def _load_checkpoint(self, target_path: str) -> Optional[SharedContext]:
path = self._checkpoint_path(target_path)
if not path.exists():
return None
try:
with open(path, "rb") as f:
data = pickle.load(f)
ctx = data.get("context")
if ctx:
print(f"[Agent] Resumed from checkpoint at step {ctx.steps_taken}")
# Restore budget and state if needed
if data.get("budget") and self.budget:
self.budget = data["budget"]
self._steps_since_escalation = data.get("steps_since_escalation", _ESCALATION_COOLDOWN)
return ctx
except Exception as exc:
print(f"[WARN] Checkpoint load failed ({exc}) — starting fresh")
return None
# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------
@staticmethod
def _observation_confirms_vuln(observation: str) -> bool:
"""Check if tool output contains a confirmed vulnerability signal."""
signals = ("VULN_CONFIRMED", "AddressSanitizer", "heap-use-after-free",
"stack-buffer-overflow", "SEGFAULT", "CRASH", "UAF")
return any(s in observation for s in signals)
def _generate_final_report(self) -> Dict[str, Any]:
"""Generate comprehensive final report."""
return {
"status": "completed",
"target": self.context.target_path,
"steps_taken": self.context.steps_taken,
"max_steps": self.max_steps,
"confirmed_vulns": self.context.confirmed_vulns,
"hypotheses_count": len(self.context.hypotheses),
"advisor_calls": self.advisor_manager.calls_made,
"parse_errors": self.context.parse_errors,
"budget": self.budget.to_dict() if self.budget else None,
"model": {
"name": self.executor.model_name,
"is_local": self.executor.is_local,
},
"summary": self.context.build_summary(),
}
