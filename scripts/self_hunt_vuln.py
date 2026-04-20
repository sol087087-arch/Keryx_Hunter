#!/usr/bin/env python3
"""self_hunt_vuln.py — end-to-end hunt against a known-vulnerable sandbox target.

Target  : sandbox/targets/vuln_cli.py
          Contains GIT_OPTION_INJECTION (HIGH) — author passed to git without "--".

Usage:
    python scripts/self_hunt_vuln.py [strict|flexible|none] [--scripted]

    strict   (default) — auto_verify fires after codeql HIGH; 0 extra LLM steps.
    flexible           — agent gets a 2-step window to call injection_verifier itself.
    none               — static AST confirm only, no injection_verifier.
    --scripted         — force ScriptedModel even when an API key is available.

Model selection:
    Reads ANTHROPIC_API_KEY from the environment, or falls back to keryx/.env.
    Uses claude-haiku-4-5-20251001 by default (cheap, fast).
    Falls back to ScriptedModel if no key is found.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from keryx.advisors.base import RuleBasedAdvisor
from keryx.advisors.manager import create_advisor_manager
from keryx.core.agent import KeryxAgent
from keryx.models.interface import ModelInterface
from keryx.tools.Toolbox import create_default_toolbox
from sandbox.tools.vuln_cli_tool import create_vuln_cli_tool

TARGET      = str(_REPO_ROOT / "sandbox" / "targets" / "vuln_cli.py")
HAIKU_ID    = "claude-haiku-4-5-20251001"
BUDGET_USD  = 0.50     # hard cap per run


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    env_file = _REPO_ROOT / "keryx" / ".env"
    if env_file.exists():
        key = env_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


# ---------------------------------------------------------------------------
# ScriptedModel (deterministic fallback — zero API cost)
# ---------------------------------------------------------------------------

import json
from collections.abc import Iterator
from typing import Any

from keryx.models.interface import (
    CostEstimate, GenerationConfig, GenerationResult,
    ModelCapabilities, ToolCall, ToolDefinition,
)


class ScriptedModel(ModelInterface):
    """Deterministic fallback: read_file → codeql_query → FINISH."""

    model_name = "scripted-fallback"

    def __init__(self, target: str) -> None:
        self._target    = target
        self.main_calls = 0

    def generate(self, prompt: str, config: GenerationConfig | None = None,
                 *, grammar: str | None = None, max_tokens: int | None = None) -> str:
        if prompt.strip() == "ping":
            return "pong"
        if "vulnerability hypothesis" in prompt or "Does the evidence" in prompt:
            return "GENUINE CONCERN — static analysis finding is valid."
        self.main_calls += 1
        step = self._next_step(prompt)
        print(f"  [ScriptedModel] call={self.main_calls}  action={step['action']!r}")
        return json.dumps(step)

    def _next_step(self, prompt: str) -> dict:
        if ("] codeql_query" in prompt and "obs:" in prompt) or "[AST]" in prompt:
            return {"thought": "Analysis done.", "action": "FINISH",
                    "action_input": {}, "confidence": 0.88}
        if "] read_file" in prompt and "obs:" in prompt:
            return {"thought": "Read target. Running AST analysis.",
                    "action": "codeql_query",
                    "action_input": {"path": self._target}, "confidence": 0.82}
        return {"thought": "Starting: read target first.",
                "action": "read_file",
                "action_input": {"path": self._target}, "confidence": 0.75}

    async def generate_async(self, prompt: str, config=None,
                             *, grammar=None, max_tokens=None) -> str:
        return self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens)

    def generate_result(self, prompt, config=None) -> GenerationResult:
        return GenerationResult(text=self.generate(prompt, config),
                                tokens_input=10, tokens_output=10,
                                duration_ms=1.0, finish_reason="stop")

    def generate_stream(self, prompt, config=None) -> Iterator[str]:
        yield self.generate(prompt, config)

    def generate_with_tools(self, prompt, tools, config=None) -> str | ToolCall:
        return self.generate(prompt, config)

    def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def get_context_length(self) -> int:
        return 32_768

    def is_healthy(self) -> bool:
        return True

    def estimate_cost(self, i, o) -> CostEstimate:
        return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

    def get_usage_cost(self) -> CostEstimate:
        return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_context_length=32_768, supports_tool_calling=False,
            supports_grammar=True, supports_batching=False, supports_streaming=False,
            supports_speculative=False, supports_min_p=False, requires_gpu=False,
            is_local=True, is_quantized=False,
        )

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_agent(
    mode:     str,
    scripted: bool = False,
) -> tuple[KeryxAgent, ModelInterface]:
    toolbox = create_default_toolbox(allowed_root=str(_REPO_ROOT))
    toolbox.register(create_vuln_cli_tool())

    advisor = RuleBasedAdvisor(confidence_threshold=0.5)
    manager = create_advisor_manager(advisors=[(advisor, 10)])

    model: ModelInterface
    if scripted:
        model = ScriptedModel(target=TARGET)
        print("[VulnHunt] model : ScriptedModel (deterministic)")
    else:
        api_key = _load_api_key()
        if api_key:
            from keryx.models.remote import create_anthropic_model
            model = create_anthropic_model(
                api_key=api_key,
                model=HAIKU_ID,
                max_budget_usd=BUDGET_USD,
            )
            print(f"[VulnHunt] model : {HAIKU_ID}  (budget cap ${BUDGET_USD})")
        else:
            model = ScriptedModel(target=TARGET)
            print("[VulnHunt] model : ScriptedModel (no API key found — fallback)")

    agent = KeryxAgent(
        executor_model=model,
        advisor_manager=manager,
        toolbox=toolbox,
        max_steps=15,
        confidence_threshold=0.60,
        verification_mode=mode,
        max_clean_scans_before_exit=1,
    )
    return agent, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(mode: str, scripted: bool) -> None:
    print("=" * 62)
    print(f"[VulnHunt] target            : {TARGET}")
    print(f"[VulnHunt] verification_mode : {mode}")

    agent, model = build_agent(mode, scripted=scripted)
    print("=" * 62)

    t0     = time.perf_counter()
    result = await agent.run(target_path=TARGET, resume=False)
    elapsed = time.perf_counter() - t0

    vulns  = result.get("confirmed_vulns", [])
    steps  = result.get("steps_taken", 0)
    status = result.get("status", "?")

    # Cost / call metrics — works for both model types
    cost_info = model.get_usage_cost()
    total_cost = cost_info.get("total_cost_usd", 0.0)
    api_calls  = getattr(model, "_call_count", None) or getattr(model, "main_calls", "n/a")

    print()
    print("=" * 62)
    print("[VulnHunt] RESULT")
    print("=" * 62)
    print(f"  status            : {status}")
    print(f"  steps_taken       : {steps}")
    print(f"  api_calls         : {api_calls}")
    print(f"  cost_usd          : ${total_cost:.5f}")
    print(f"  elapsed_s         : {elapsed:.2f}")
    print(f"  confirmed_vulns   : {len(vulns)}")

    for i, v in enumerate(vulns, 1):
        print(f"\n  [{i}] action         : {v.get('action')}")
        print(f"       confidence_tag : {v.get('confidence_tag')}")
        print(f"       verified       : {v.get('verified')}")
        obs = v.get("observation", "")
        print(f"       observation    :\n    {obs[:300]}")

    if vulns:
        print(f"\n[VulnHunt] CONFIRMED {len(vulns)} vulnerability(ies).")
    else:
        print("\n[VulnHunt] No vulnerabilities confirmed.")

    if mode == "strict" and not vulns:
        print("[VulnHunt] WARNING: strict mode expected VULN_CONFIRMED.")
        sys.exit(1)


if __name__ == "__main__":
    args     = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags    = [a for a in sys.argv[1:] if a.startswith("--")]
    mode     = args[0] if args else "strict"
    scripted = "--scripted" in flags

    if mode not in ("strict", "flexible", "none"):
        print(f"Usage: {sys.argv[0]} [strict|flexible|none] [--scripted]")
        sys.exit(1)

    asyncio.run(main(mode, scripted=scripted))
