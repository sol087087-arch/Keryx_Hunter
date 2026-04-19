#!/usr/bin/env python3
"""self_hunt.py — end-to-end system validation.

Runs KeryxAgent against keryx/advisors/manager.py (a complex, real target).
Uses:
  - ScriptedModel:       deterministic ReAct JSON, no real LLM needed
  - create_default_toolbox: real ASTAnalyzer + ReadFile + InjectionVerifier
  - RuleBasedAdvisor:    zero-cost advisor, no LLM needed

Exit codes:
  0 — hunt completed (vuln confirmed OR clean exit)
  1 — unexpected failure
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make sure the repo root is on sys.path when run as a script.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from keryx.advisors.base import RuleBasedAdvisor
from keryx.advisors.manager import create_advisor_manager
from keryx.core.agent import KeryxAgent
from keryx.models.interface import (
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelInterface,
)
from keryx.tools.Toolbox import create_default_toolbox

TARGET = str(_REPO_ROOT / "keryx" / "advisors" / "manager.py")


# ---------------------------------------------------------------------------
# ScriptedModel
# ---------------------------------------------------------------------------

class ScriptedModel(ModelInterface):
    """
    Deterministic model that produces realistic ReAct JSON steps.

    Step selection is based on the presence of key markers in the prompt:
      - No observation yet   → read_file the target
      - After read_file obs  → codeql_query the target
      - After codeql obs     → FINISH (or injection_verifier if HIGH found)
    """

    def __init__(self, target_path: str) -> None:
        self.model_name = "scripted-model-v1"
        self._target = target_path
        self._call_count = 0

    # ── sync generate (async delegates here) ─────────────────────────────

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._call_count += 1

        # Health-check ping — agent calls generate("ping", max_tokens=10)
        if prompt.strip() == "ping":
            return "pong"

        # Critique prompt — return a useful verdict without JSON
        if "vulnerability hypothesis" in prompt or "Does the evidence" in prompt:
            return "GENUINE CONCERN — evidence supports the hypothesis."

        step = self._infer_step(prompt)
        raw = json.dumps(step, indent=2)
        print(f"[ScriptedModel] call={self._call_count} action={step['action']!r}")
        return raw

    def _infer_step(self, prompt: str) -> dict:
        """Choose the next action based on what is already in the prompt.

        The agent's get_recent_history() formats history as:
            [N] action_name | conf=X.XX
                obs: <observation text>
        So we look for those markers to detect which steps have already run.
        """
        # Did an injection_verifier step already run?
        has_injection_obs = ("] injection_verifier" in prompt and "obs:" in prompt)

        # Did a codeql_query step already run? (AST analyzer outputs "[AST]")
        has_codeql_obs = ("] codeql_query" in prompt and "obs:" in prompt) or "[AST]" in prompt

        # Did a read_file step already run?
        has_read_obs = "] read_file" in prompt and "obs:" in prompt

        # Are there HIGH findings visible in the prompt (from codeql output)?
        has_high = "[HIGH]" in prompt

        if has_injection_obs or (has_codeql_obs and not has_high):
            # Clean scan or injection verification done → finish
            verdict = "No HIGH findings — target appears clean." if not has_high else "Injection verification complete."
            return {
                "thought": f"Static analysis complete. {verdict}",
                "action": "FINISH",
                "action_input": {},
                "confidence": 0.88,
            }

        if has_codeql_obs and has_high:
            # HIGH findings visible but not yet verified → run injection_verifier
            return {
                "thought": (
                    "codeql_query found HIGH severity findings. "
                    "I will use injection_verifier to confirm whether the target tool "
                    "actually accepts the dangerous payload."
                ),
                "action": "injection_verifier",
                "action_input": {
                    "target_tool":       "codeql_query",
                    "inject_field":      "path",
                    "payload":           "../../../etc/passwd",
                    "expected_behavior": "reject",
                    "base_overrides":    {"path": self._target},
                },
                "confidence": 0.78,
            }

        if has_read_obs:
            # File content is in the prompt → run AST analysis
            return {
                "thought": (
                    "I have read the target file. "
                    "Now I will run static analysis to identify security findings."
                ),
                "action": "codeql_query",
                "action_input": {"path": self._target},
                "confidence": 0.80,
            }

        # First step: read the file to understand its structure
        return {
            "thought": (
                "I will start by reading the target file to understand its structure "
                "before running security analysis."
            ),
            "action": "read_file",
            "action_input": {"path": self._target},
            "confidence": 0.75,
        }

    # ── async path ────────────────────────────────────────────────────────

    async def generate_async(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens)

    def generate_result(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        text = self.generate(prompt, config)
        return GenerationResult(
            text=text,
            tokens_input=len(prompt) // 4,
            tokens_output=len(text) // 4,
            duration_ms=5.0,
            finish_reason="stop",
            metadata={"source": "scripted"},
        )

    def generate_stream(self, prompt, config=None):
        yield self.generate(prompt, config)

    def generate_with_tools(self, prompt, tools, config=None):
        return self.generate(prompt, config)

    # ── stubs ─────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def get_context_length(self) -> int:
        return 32_768

    def is_healthy(self) -> bool:
        return True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        return CostEstimate(0.0, 0.0, 0.0)

    def get_usage_cost(self) -> CostEstimate:
        return CostEstimate(0.0, 0.0, 0.0)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_context_length=32_768,
            supports_tool_calling=False,
            supports_grammar=True,
            supports_batching=False,
            supports_streaming=False,
            supports_speculative=False,
            supports_min_p=False,
            requires_gpu=False,
            is_local=True,
            is_quantized=False,
        )

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_agent(target: str) -> KeryxAgent:
    toolbox = create_default_toolbox(allowed_root=str(_REPO_ROOT))

    advisor  = RuleBasedAdvisor(confidence_threshold=0.5)
    manager  = create_advisor_manager(advisors=[(advisor, 10)])

    model = ScriptedModel(target_path=target)

    return KeryxAgent(
        executor_model=model,
        advisor_manager=manager,
        toolbox=toolbox,
        max_steps=10,
        confidence_threshold=0.65,
        verification_mode="strict",
        max_clean_scans_before_exit=1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print(f"[SelfHunt] target : {TARGET}")
    print(f"[SelfHunt] repo   : {_REPO_ROOT}")
    print("=" * 60)

    agent = build_agent(TARGET)
    result = await agent.run(target_path=TARGET, resume=False)

    print("\n" + "=" * 60)
    print("[SelfHunt] RESULT")
    print("=" * 60)
    print(f"  status          : {result.get('status')}")
    print(f"  steps_taken     : {result.get('steps_taken')}")
    print(f"  confirmed_vulns : {result.get('confirmed_vulns')}")
    print(f"  hypotheses      : {result.get('hypotheses')}")
    print(f"  avg_confidence  : {result.get('average_confidence', 0):.2f}")

    vulns = result.get("confirmed_vulns", [])
    if vulns:
        print(f"\n[SelfHunt] CONFIRMED: {len(vulns)} vulnerability(ies) found.")
    else:
        print("\n[SelfHunt] No vulnerabilities confirmed (clean or early-exit).")

    status = result.get("status", "")
    if status not in ("completed", "finished", "max_steps_reached", "early_exit_clean",
                      "budget_exhausted", "health_failure", "failed"):
        print(f"[SelfHunt] Unexpected status {status!r}")
        sys.exit(1)

    print("[SelfHunt] Done.")


if __name__ == "__main__":
    asyncio.run(main())
