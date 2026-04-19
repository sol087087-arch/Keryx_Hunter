"""Integration tests: clean-exit loop guard.

Verifies that KeryxAgent exits after `max_clean_scans_before_exit` consecutive
clean codeql scans, consuming exactly N main-loop LLM calls and no more.

These tests sit above test_loop_guards.py (pure function) and below a full
end-to-end hunt — they exercise the agent's *use* of check_exit_clean():
the loop break, the step counter, and the model-call economy.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from keryx.advisors.base import AdvisorResponse, BaseAdvisor
from keryx.advisors.manager import create_advisor_manager
from keryx.core.agent import KeryxAgent
from keryx.models.interface import (
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelInterface,
    ToolCall,
    ToolDefinition,
)
from keryx.tools.Toolbox import BaseTool, ToolResult, create_toolbox


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CLEAN_OBS = "[AST] No findings in clean.py (50 lines analyzed)."
_HIGH_OBS  = (
    '[HIGH] GIT_OPTION_INJECTION @ line 10: '
    'cmd.extend(["--author", author])\n'
)
_TARGET = "/tmp/clean_exit_test.py"


# ---------------------------------------------------------------------------
# Test doubles — tools
# ---------------------------------------------------------------------------

class _CleanCodeqlTool(BaseTool):
    """codeql_query that always returns zero findings."""
    name = "codeql_query"

    async def execute(self, action_input: Any, context: Any = None) -> ToolResult:
        return ToolResult(success=True, output=_CLEAN_OBS)


class _HighThenCleanTool(BaseTool):
    """Returns a HIGH finding on the first call, clean on every subsequent call."""
    name = "codeql_query"

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    async def execute(self, action_input: Any, context: Any = None) -> ToolResult:
        self._calls += 1
        if self._calls == 1:
            return ToolResult(success=True, output=_HIGH_OBS)
        return ToolResult(success=True, output=_CLEAN_OBS)


# ---------------------------------------------------------------------------
# Test doubles — advisor
# ---------------------------------------------------------------------------

class _SilentAdvisor(BaseAdvisor):
    """Never triggers — eliminates advisor noise from step counts."""
    name = "silent-advisor"

    def should_trigger(self, context: Any) -> bool:
        return False

    async def advise(self, context: Any) -> AdvisorResponse:
        return AdvisorResponse()


# ---------------------------------------------------------------------------
# Test doubles — model
# ---------------------------------------------------------------------------

class _CodeqlModel(ModelInterface):
    """Always requests codeql_query.

    Handles two non-ReAct call types transparently:
      - health-check ping  → "pong"
      - self-critique       → "NEEDS MORE EVIDENCE"

    All other calls are counted as main ReAct calls.
    """

    model_name = "scripted-codeql-model"

    def __init__(self) -> None:
        self.main_calls: int = 0

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if prompt.strip() == "ping":
            return "pong"
        if "vulnerability hypothesis" in prompt or "Does the evidence" in prompt:
            return "NEEDS MORE EVIDENCE"
        self.main_calls += 1
        return json.dumps({
            "thought": f"running static analysis (main_call={self.main_calls})",
            "action": "codeql_query",
            "action_input": {"path": _TARGET},
            "confidence": 0.80,
        })

    def generate_result(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        text = self.generate(prompt, config)
        return GenerationResult(text=text, tokens_input=10, tokens_output=10,
                                duration_ms=1.0, finish_reason="stop")

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        yield self.generate(prompt, config)

    def generate_with_tools(self, prompt: str, tools: list[ToolDefinition],
                            config: GenerationConfig | None = None) -> str | ToolCall:
        return self.generate(prompt, config)

    def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def get_context_length(self) -> int:
        return 32_768

    def is_healthy(self) -> bool:
        return True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
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
# Factory helper
# ---------------------------------------------------------------------------

def _make_agent(
    codeql_tool: BaseTool,
    max_clean: int,
    max_steps: int = 20,
) -> tuple[KeryxAgent, _CodeqlModel]:
    model   = _CodeqlModel()
    toolbox = create_toolbox(tools=[codeql_tool])
    manager = create_advisor_manager(advisors=[(_SilentAdvisor(), 5)])
    agent   = KeryxAgent(
        executor_model=model,
        advisor_manager=manager,
        toolbox=toolbox,
        max_steps=max_steps,
        verification_mode="none",      # tests loop guard only, not verification path
        max_clean_scans_before_exit=max_clean,
    )
    return agent, model


# ---------------------------------------------------------------------------
# TestCleanExit — core exit mechanics
# ---------------------------------------------------------------------------

class TestCleanExit:
    """Verify the loop breaks at exactly N consecutive clean codeql scans."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 3])
    async def test_exits_after_n_consecutive_clean_scans(self, n: int) -> None:
        agent, model = _make_agent(_CleanCodeqlTool(), max_clean=n)
        result = await agent.run(_TARGET, resume=False)

        assert result["steps_taken"] == n, (
            f"expected {n} steps for max_clean={n}, got {result['steps_taken']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 3])
    async def test_no_extra_llm_calls_after_exit(self, n: int) -> None:
        """Model is called exactly N times for main ReAct steps — no token waste."""
        agent, model = _make_agent(_CleanCodeqlTool(), max_clean=n)
        await agent.run(_TARGET, resume=False)

        assert model.main_calls == n, (
            f"expected {n} main LLM calls, got {model.main_calls}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 3])
    async def test_confirmed_vulns_empty_on_clean_exit(self, n: int) -> None:
        agent, model = _make_agent(_CleanCodeqlTool(), max_clean=n)
        result = await agent.run(_TARGET, resume=False)

        assert result["confirmed_vulns"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 3])
    async def test_counter_equals_n_at_exit(self, n: int) -> None:
        """Internal consecutive_clean_scans counter matches the exit threshold."""
        agent, model = _make_agent(_CleanCodeqlTool(), max_clean=n)
        await agent.run(_TARGET, resume=False)

        assert agent._consecutive_clean_scans == n

    @pytest.mark.asyncio
    async def test_max_steps_not_the_limiting_factor(self) -> None:
        """max_steps is large; the clean-exit path, not the budget, causes the break."""
        agent, model = _make_agent(_CleanCodeqlTool(), max_clean=1, max_steps=50)
        result = await agent.run(_TARGET, resume=False)

        # Loop broke via clean-exit, well before max_steps
        assert result["steps_taken"] == 1
        assert result["steps_taken"] < 50


# ---------------------------------------------------------------------------
# TestHighThenClean — counter reset when vuln is detected
# ---------------------------------------------------------------------------

class TestHighThenClean:
    """Verify a HIGH finding resets the clean counter, deferring early exit.

    Uses verification_mode='flexible' so the HIGH finding does NOT stop the
    loop immediately (as 'none' would).  In flexible mode the agent gets a
    2-step window to call injection_verifier itself; the clean-exit guard fires
    first (after the next clean scan), so confirmed_vulns stays empty.
    """

    def _agent_flexible(self, max_clean: int) -> tuple[KeryxAgent, _CodeqlModel]:
        model   = _CodeqlModel()
        toolbox = create_toolbox(tools=[_HighThenCleanTool()])
        manager = create_advisor_manager(advisors=[(_SilentAdvisor(), 5)])
        agent   = KeryxAgent(
            executor_model=model,
            advisor_manager=manager,
            toolbox=toolbox,
            max_steps=20,
            verification_mode="flexible",   # HIGH records finding but doesn't stop loop
            max_clean_scans_before_exit=max_clean,
        )
        return agent, model

    @pytest.mark.asyncio
    async def test_high_resets_counter_exit_deferred_to_next_clean(self) -> None:
        """
        Step 1: HIGH finding  → counter reset to 0, loop continues (flexible mode).
        Step 2: clean finding → counter becomes 1 ≥ max_clean(1), clean-exit fires.
        Total steps: 2.
        """
        agent, model = self._agent_flexible(max_clean=1)
        result = await agent.run(_TARGET, resume=False)

        assert result["steps_taken"] == 2

    @pytest.mark.asyncio
    async def test_counter_is_one_after_high_then_clean(self) -> None:
        """Counter at exit is 1, not 2 — the HIGH scan correctly reset it to 0."""
        agent, model = self._agent_flexible(max_clean=1)
        await agent.run(_TARGET, resume=False)

        assert agent._consecutive_clean_scans == 1

    @pytest.mark.asyncio
    async def test_clean_exit_wins_before_flexible_fallback(self) -> None:
        """Clean-exit fires (at step 2) before the flexible 2-step window elapses.

        Consequence: confirmed_vulns is empty — the clean scan pre-empts static
        fallback confirmation.  This is intentional: a clean follow-up scan
        means the first HIGH was likely a false positive.
        """
        agent, model = self._agent_flexible(max_clean=1)
        result = await agent.run(_TARGET, resume=False)

        assert result["confirmed_vulns"] == []

    @pytest.mark.asyncio
    async def test_model_called_twice_for_high_then_clean(self) -> None:
        """Two codeql steps → exactly two main LLM calls, no extras."""
        agent, model = self._agent_flexible(max_clean=1)
        await agent.run(_TARGET, resume=False)

        assert model.main_calls == 2
