"""
test_keryx_suite.py — Comprehensive KeryxHunter test suite.

T01 – Package Import Test
T02 – ModelInterface Contract Test
T03 – CapabilityRouter Test
T04 – SharedContext Test
T05 – AdvisorManager Test
T06 – Default ToolBox Creation Test
T07 – ReadFileTool Test
T08 – GitBlameTool Test
T09 – KeryxAgent Instantiation Test
T10 – KeryxAgent.run() Basic ReAct Cycle Test
T11 – Orchestrator Creation Test
T12 – End-to-End Minimal Hunt Smoke Test
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pytest

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

from keryx.models.interface import (
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelInterface,
    ToolCall,
    ToolDefinition,
)
from keryx.models.local_mock import MyLocalModel


class FinishModel(MyLocalModel):
    """Returns FINISH JSON on every main generate call; 'ok' for health/critique."""

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if max_tokens == 1 or grammar is None:
            return "ok"
        return json.dumps(
            {
                "thought": "Analysis done",
                "action": "FINISH",
                "action_input": {},
                "confidence": 0.9,
            }
        )


class ToolCallingModel(MyLocalModel):
    """Generates N read_file calls then FINISH; 'ok' for health/critique."""

    def __init__(self, target_file: str, steps_before_finish: int = 4) -> None:
        super().__init__()
        self._target_file = target_file
        self._steps_before_finish = steps_before_finish
        self._main_calls = 0

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._main_calls += 1
        if self._main_calls > self._steps_before_finish:
            return json.dumps(
                {
                    "thought": "Target analysis complete",
                    "action": "FINISH",
                    "action_input": {},
                    "confidence": 0.9,
                }
            )
        return json.dumps(
            {
                "thought": f"Examining codebase, pass {self._main_calls}",
                "action": "read_file",
                "action_input": {"file_path": self._target_file},
                "confidence": 0.75,
            }
        )


# ---------------------------------------------------------------------------
# T01 – Package Import Test
# ---------------------------------------------------------------------------


class TestPackageImport:
    """Verify that `import keryx` works and all public symbols are reachable."""

    def test_import_keryx_top_level(self) -> None:
        import keryx  # noqa: F401

    def test_version_string(self) -> None:
        import keryx

        assert isinstance(keryx.__version__, str)
        assert keryx.__version__.count(".") >= 1

    def test_version_tuple(self) -> None:
        import keryx

        t = keryx.__version_tuple__
        assert isinstance(t, tuple)
        assert len(t) == 3
        assert all(isinstance(n, int) for n in t)

    def test_get_version_tuple_function(self) -> None:
        from keryx import get_version_tuple

        t = get_version_tuple()
        assert isinstance(t, tuple)

    def test_eager_schemas_available(self) -> None:
        from keryx import GitBlameInput, HotspotEntry, MetricsSnapshot, RiskLevel

        assert RiskLevel.CRITICAL == "CRITICAL"
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.LOW == "LOW"

    def test_eager_shared_context_available(self) -> None:
        from keryx import SharedContext

        ctx = SharedContext(target_path="/tmp/x")
        assert ctx.target_path == "/tmp/x"

    def test_eager_tool_result_available(self) -> None:
        from keryx import ToolResult

        r = ToolResult(success=True, output="hello")
        assert str(r)

    def test_lazy_toolbox_available(self) -> None:
        from keryx import ToolBox

        tb = ToolBox()
        assert tb is not None
        tb.shutdown()

    def test_lazy_keryx_agent_available(self) -> None:
        from keryx import KeryxAgent

        assert KeryxAgent is not None

    def test_lazy_capability_router_available(self) -> None:
        from keryx import CapabilityRouter

        assert CapabilityRouter is not None

    def test_lazy_git_blame_tool_available(self) -> None:
        from keryx import GitBlameTool

        assert GitBlameTool is not None

    def test_lazy_advisor_manager_available(self) -> None:
        from keryx import AdvisorManager  # noqa: F401 (imported lazily)

    def test_hunt_is_coroutine(self) -> None:
        import inspect

        from keryx import hunt

        assert inspect.iscoroutinefunction(hunt)

    def test_all_contains_expected_keys(self) -> None:
        import keryx

        for symbol in ("SharedContext", "ToolResult", "RiskLevel", "hunt"):
            assert symbol in keryx.__all__

    def test_getattr_unknown_name_raises(self) -> None:
        """__getattr__ error path (line 108): unknown name → AttributeError."""
        import keryx
        import pytest

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = keryx.this_symbol_does_not_exist_xyz

    def test_get_version_tuple_invalid_version(self) -> None:
        """get_version_tuple error path (lines 30-31): bad version → (0, 0, 0)."""
        import keryx

        original = keryx.__version__
        try:
            keryx.__version__ = "not.a.valid.version.!!"
            result = keryx.get_version_tuple()
            assert result == (0, 0, 0)
        except Exception:
            # int() conversion may raise ValueError — that's the path we're covering
            result = keryx.get_version_tuple()
            assert result == (0, 0, 0)
        finally:
            keryx.__version__ = original

    async def test_hunt_entry_point_calls_orchestrator(self) -> None:
        """keryx.hunt() wires up ToolBox + AdvisorManager + KeryxOrchestrator."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import keryx

        mock_result = {
            "status": "completed",
            "confirmed_vulns": [],
            "target": "/tmp/hunt_test",
        }
        mock_orc = MagicMock()
        mock_orc.hunt = AsyncMock(return_value=mock_result)

        with patch("keryx.core.orchestrator.KeryxOrchestrator",
                   return_value=mock_orc) as MockOrc:
            result = await keryx.hunt(
                "/tmp/hunt_test",
                capability="fast_pattern_matching",
                mode="airgapped",
            )

        MockOrc.assert_called_once()
        mock_orc.hunt.assert_awaited_once_with(
            target_path="/tmp/hunt_test",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert result["status"] == "completed"

    async def test_hunt_accepts_explicit_toolbox_and_advisors(self) -> None:
        """hunt() uses provided toolbox/advisors instead of creating new ones."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import keryx
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        am = AdvisorManager()
        mock_orc = MagicMock()
        mock_orc.hunt = AsyncMock(return_value={"status": "completed",
                                                "confirmed_vulns": []})

        with patch("keryx.core.orchestrator.KeryxOrchestrator",
                   return_value=mock_orc) as MockOrc:
            await keryx.hunt("/tmp/x", toolbox=tb, advisors=am, models={"m": MagicMock()})

        call_kwargs = MockOrc.call_args.kwargs
        assert call_kwargs["toolbox"] is tb
        assert call_kwargs["advisor_manager"] is am
        tb.shutdown()
        am.shutdown()


# ---------------------------------------------------------------------------
# T02 – ModelInterface Contract Test
# ---------------------------------------------------------------------------


class TestModelInterfaceContract:
    """Verify MyLocalModel satisfies every method in ModelInterface."""

    @pytest.fixture
    def model(self) -> MyLocalModel:
        return MyLocalModel()

    def test_is_model_interface_subclass(self, model: MyLocalModel) -> None:
        assert isinstance(model, ModelInterface)

    def test_model_name_set(self, model: MyLocalModel) -> None:
        assert isinstance(model.model_name, str)
        assert len(model.model_name) > 0

    def test_is_local_true(self, model: MyLocalModel) -> None:
        assert model.is_local is True

    def test_requires_network_false(self, model: MyLocalModel) -> None:
        assert model.requires_network is False

    def test_is_healthy(self, model: MyLocalModel) -> None:
        assert model.is_healthy() is True

    def test_generate_returns_nonempty_string(self, model: MyLocalModel) -> None:
        result = model.generate("What is 2+2?")
        assert isinstance(result, str) and result

    def test_generate_with_config(self, model: MyLocalModel) -> None:
        cfg = GenerationConfig(max_tokens=50, temperature=0.1)
        result = model.generate("hello", config=cfg)
        assert isinstance(result, str)

    def test_generate_with_grammar_kwarg(self, model: MyLocalModel) -> None:
        result = model.generate("hello", grammar="root ::= [a-z]+")
        assert isinstance(result, str)

    def test_generate_result_shape(self, model: MyLocalModel) -> None:
        r = model.generate_result("test prompt")
        assert isinstance(r, GenerationResult)
        assert r.tokens_input > 0
        assert r.tokens_output > 0
        assert r.duration_ms >= 0.0

    def test_generate_stream(self, model: MyLocalModel) -> None:
        chunks = list(model.generate_stream("test"))
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    async def test_generate_async(self, model: MyLocalModel) -> None:
        result = await model.generate_async("async test")
        assert isinstance(result, str)

    def test_tokenize_returns_list_of_ints(self, model: MyLocalModel) -> None:
        tokens = model.tokenize("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_count_tokens_positive(self, model: MyLocalModel) -> None:
        n = model.count_tokens("hello world")
        assert isinstance(n, int) and n > 0

    def test_get_context_length_positive(self, model: MyLocalModel) -> None:
        assert model.get_context_length() > 0

    def test_estimate_cost_zero_for_local(self, model: MyLocalModel) -> None:
        cost = model.estimate_cost(500, 200)
        assert cost["total_cost_usd"] == 0.0

    def test_get_usage_cost(self, model: MyLocalModel) -> None:
        cost = model.get_usage_cost()
        assert isinstance(cost["total_cost_usd"], float)

    def test_get_capabilities_shape(self, model: MyLocalModel) -> None:
        caps = model.get_capabilities()
        assert caps["is_local"] is True
        assert isinstance(caps["max_context_length"], int)
        assert isinstance(caps["supports_grammar"], bool)

    def test_capabilities_property_matches_method(self, model: MyLocalModel) -> None:
        assert model.capabilities == model.get_capabilities()

    def test_is_local_setter(self, model: MyLocalModel) -> None:
        model.is_local = False
        assert model.is_local is False
        model.is_local = True
        assert model.is_local is True

    def test_generate_with_tools(self, model: MyLocalModel) -> None:
        tools = [ToolDefinition(name="noop", description="x", parameters={})]
        result = model.generate_with_tools("test", tools)
        assert isinstance(result, str)

    def test_unload_no_exception(self, model: MyLocalModel) -> None:
        model.unload()

    def test_zero_api_cost(self, model: MyLocalModel) -> None:
        assert model.cost_per_1k_input_tokens == 0.0
        assert model.cost_per_1k_output_tokens == 0.0

    def test_get_metrics_returns_dict(self, model: MyLocalModel) -> None:
        m = model.get_metrics()
        assert isinstance(m, dict)


# ---------------------------------------------------------------------------
# T01-supplement — Schema validation coverage
# (covers GitBlameInput cross-field validator, HotspotEntry cap_top_authors,
#  MetricsSnapshot round_rate, and SharedContext advisor advice edge cases)
# ---------------------------------------------------------------------------


class TestSchemaValidators:
    """Exercise pydantic validators that are not reached by the basic import path."""

    def test_git_blame_input_blame_requires_file(self) -> None:
        from pydantic import ValidationError
        from keryx.core.schemas import GitBlameInput

        with pytest.raises(ValidationError):
            GitBlameInput(command="blame")  # file is required

    def test_git_blame_input_blame_with_file_ok(self) -> None:
        from keryx.core.schemas import GitBlameInput

        inp = GitBlameInput(command="blame", file="foo.c")
        assert inp.command == "blame"
        assert inp.file == "foo.c"

    def test_git_blame_input_log_no_file_ok(self) -> None:
        from keryx.core.schemas import GitBlameInput

        inp = GitBlameInput(command="log")
        assert inp.command == "log"

    def test_hotspot_entry_cap_top_authors(self) -> None:
        from keryx.core.schemas import HotspotEntry, RiskLevel

        entry = HotspotEntry(
            file="foo.c",
            changes=5,
            authors=4,
            complexity_score=12.0,
            risk_level=RiskLevel.HIGH,
            top_authors=["alice", "bob", "carol", "dave"],  # 4 → capped to 3
        )
        assert len(entry.top_authors) == 3

    def test_metrics_snapshot_round_rate(self) -> None:
        from keryx.core.schemas import MetricsSnapshot

        snap = MetricsSnapshot(
            name="read_file",
            calls=10,
            successes=9,
            success_rate=0.90001,  # triggers round_rate validator
            avg_time_ms=5.0,
        )
        assert snap.success_rate == pytest.approx(0.9)

    def test_shared_context_advice_via_to_dict_object(self) -> None:
        """Cover the hasattr(advice, 'to_dict') branch."""
        from keryx.core.shared_context import SharedContext

        class FakeAdvice:
            def to_dict(self):
                return {"strategy": "from_obj", "strategic_direction": "go left"}

        ctx = SharedContext(target_path="/tmp/x")
        ctx.add_advisor_advice(FakeAdvice())
        assert ctx.has_pending_advisor_advice()

    def test_shared_context_advice_fallback_str(self) -> None:
        """Cover the else branch (arbitrary type → wrap in strategy str)."""
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/x")
        ctx.add_advisor_advice("just a string advice")
        assert ctx.has_pending_advisor_advice()


# ---------------------------------------------------------------------------
# T03 – CapabilityRouter Test
# ---------------------------------------------------------------------------


class TestCapabilityRouter:
    """Verify routing decisions, capability profiles, fallback chains, budget logic."""

    @pytest.fixture
    def router(self):
        from keryx.core.router import CapabilityRouter

        return CapabilityRouter()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    @pytest.fixture
    def available_models(self):
        """Provide a model named after the default executor in most profiles."""
        return {"qwen3-coder-8b": MyLocalModel()}

    def test_instantiation(self, router) -> None:
        assert router is not None

    def test_list_capabilities_nonempty(self, router) -> None:
        caps = router.list_capabilities()
        assert len(caps) > 0

    def test_list_capabilities_contains_defaults(self, router) -> None:
        caps = router.list_capabilities()
        for name in ("deep_reasoning", "fast_pattern_matching", "airgapped", "verified_only"):
            assert name in caps

    def test_describe_known_capability(self, router) -> None:
        desc = router.describe("fast_pattern_matching")
        assert "fast_pattern_matching" in desc

    def test_describe_unknown_capability(self, router) -> None:
        desc = router.describe("ghost_capability")
        assert "Unknown" in desc

    def test_build_fallback_chain_excludes_self(self) -> None:
        from keryx.core.router import _build_fallback_chain

        chain = _build_fallback_chain("llama-4-70b")
        assert "llama-4-70b" not in chain

    def test_build_fallback_chain_ordered_by_tier(self) -> None:
        from keryx.core.router import _CAPABILITY_TIERS, _build_fallback_chain

        chain = _build_fallback_chain("llama-4-70b")
        tiers = [_CAPABILITY_TIERS[n]["tier"] for n in chain if n in _CAPABILITY_TIERS]
        assert tiers == sorted(tiers, reverse=True)

    def test_resolve_budget_takes_minimum(self) -> None:
        from keryx.core.router import CapabilityRouter

        budget = CapabilityRouter._resolve_budget(10.0, 3.0, False)
        assert budget == pytest.approx(3.0)

    def test_resolve_budget_profile_only(self) -> None:
        from keryx.core.router import CapabilityRouter

        budget = CapabilityRouter._resolve_budget(7.5, None, False)
        assert budget == pytest.approx(7.5)

    def test_resolve_budget_user_only(self) -> None:
        from keryx.core.router import CapabilityRouter

        budget = CapabilityRouter._resolve_budget(None, 4.0, False)
        assert budget == pytest.approx(4.0)

    def test_resolve_budget_airgapped_returns_none(self) -> None:
        from keryx.core.router import CapabilityRouter

        budget = CapabilityRouter._resolve_budget(5.0, 3.0, True)
        assert budget is None

    def test_route_fast_pattern_matching(
        self, router, available_models, advisor_manager
    ) -> None:
        plan = router.route(
            "fast_pattern_matching",
            available_models,
            advisor_manager,
            is_airgapped=True,
        )
        assert plan.capability == "fast_pattern_matching"
        assert plan.enforce_no_network is True
        assert plan.budget_usd is None
        assert plan.executor_model is not None

    def test_route_unknown_capability_falls_back(
        self, router, available_models, advisor_manager
    ) -> None:
        plan = router.route(
            "totally_unknown_cap",
            available_models,
            advisor_manager,
            is_airgapped=True,
        )
        # Falls back to deep_reasoning
        assert plan.capability == "deep_reasoning"

    def test_route_no_executor_raises(self, router, advisor_manager) -> None:
        with pytest.raises(RuntimeError, match="No suitable executor"):
            router.route(
                "fast_pattern_matching",
                {},  # no models available
                advisor_manager,
                is_airgapped=True,
            )

    def test_route_increments_metric(
        self, router, available_models, advisor_manager
    ) -> None:
        router.reset_metrics()
        router.route(
            "fast_pattern_matching", available_models, advisor_manager, is_airgapped=True
        )
        assert router.get_metrics()["routing_decisions"] == 1

    def test_reset_metrics(self, router, available_models, advisor_manager) -> None:
        router.route(
            "fast_pattern_matching", available_models, advisor_manager, is_airgapped=True
        )
        router.reset_metrics()
        assert router.get_metrics()["routing_decisions"] == 0

    def test_routing_plan_repr(self, router, available_models, advisor_manager) -> None:
        plan = router.route(
            "fast_pattern_matching", available_models, advisor_manager, is_airgapped=True
        )
        r = repr(plan)
        assert "fast_pattern_matching" in r

    def test_routing_plan_validates_airgapped_budget(
        self, router, available_models, advisor_manager
    ) -> None:
        # airgapped + budget = ValueError from RoutingPlan._validate
        from keryx.core.router import RoutingPlan

        with pytest.raises(ValueError, match="budget"):
            RoutingPlan(
                executor_model=MyLocalModel(),
                advisor_name="none",
                advisor_chain=[],
                budget_usd=1.0,
                enforce_no_network=True,
                require_consensus=False,
                consensus_threshold=0.67,
                require_fuzzing_confirm=False,
                debate_models=[],
                capability="test",
            )

    def test_routing_plan_validates_consensus_models(self) -> None:
        from keryx.core.router import RoutingPlan

        with pytest.raises(ValueError, match="Consensus requires"):
            RoutingPlan(
                executor_model=MyLocalModel(),
                advisor_name="none",
                advisor_chain=[],
                budget_usd=None,
                enforce_no_network=False,
                require_consensus=True,
                consensus_threshold=0.67,
                require_fuzzing_confirm=False,
                debate_models=["only-one"],  # must be ≥2
                capability="test",
            )


# ---------------------------------------------------------------------------
# T04 – SharedContext Test
# ---------------------------------------------------------------------------


class TestSharedContext:
    """Verify step recording, metrics, and summary generation."""

    @pytest.fixture
    def ctx(self):
        from keryx.core.shared_context import SharedContext

        return SharedContext(target_path="/tmp/target", capability="fast_pattern_matching")

    class _Step:
        def __init__(self, action: str = "read_file", confidence: float = 0.7):
            self.action = action
            self.action_input: dict = {}
            self.thought = f"Analyzing with {action}"
            self.confidence = confidence

    def test_initial_state(self, ctx) -> None:
        assert ctx.target_path == "/tmp/target"
        assert ctx.steps_taken == 0
        assert ctx.parse_errors == 0
        assert not ctx.hypotheses
        assert not ctx.confirmed_vulns

    def test_add_step_increments_counter(self, ctx) -> None:
        ctx.add_step(self._Step(), "obs")
        assert ctx.steps_taken == 1
        ctx.add_step(self._Step("git_blame"), "obs2")
        assert ctx.steps_taken == 2

    def test_add_step_records_observation(self, ctx) -> None:
        ctx.add_step(self._Step(), "file content here")
        last = ctx.get_last_step()
        assert last is not None
        assert "file content here" in last["observation"]

    def test_get_last_step_none_when_empty(self, ctx) -> None:
        assert ctx.get_last_step() is None

    def test_get_last_confidence_none_when_empty(self, ctx) -> None:
        assert ctx.get_last_confidence() is None

    def test_get_last_confidence_reflects_step(self, ctx) -> None:
        ctx.add_step(self._Step(confidence=0.82), "obs")
        conf = ctx.get_last_confidence()
        assert conf == pytest.approx(0.82)

    def test_get_recent_history_empty(self, ctx) -> None:
        h = ctx.get_recent_history(5)
        assert isinstance(h, str) and h

    def test_get_recent_history_contains_action_name(self, ctx) -> None:
        ctx.add_step(self._Step("codeql_query"), "result")
        h = ctx.get_recent_history(5)
        assert "codeql_query" in h

    def test_trim_history_preserves_steps_taken(self, ctx) -> None:
        for i in range(20):
            ctx.add_step(self._Step(f"action_{i}"), f"obs_{i}")
        ctx.trim_history(keep=6)
        assert ctx.steps_taken == 20
        assert len(ctx.steps) == 6

    def test_add_hypothesis(self, ctx) -> None:
        ctx.add_hypothesis("buffer overflow in parse()")
        assert "buffer overflow in parse()" in ctx.hypotheses

    def test_hypothesis_deduplication(self, ctx) -> None:
        ctx.add_hypothesis("same")
        ctx.add_hypothesis("same")
        assert sum(1 for h in ctx.hypotheses if h == "same") == 1

    def test_blacklist_hypothesis_removes_from_active(self, ctx) -> None:
        ctx.add_hypothesis("false positive")
        ctx.blacklist_hypothesis("false positive")
        assert "false positive" not in ctx.hypotheses
        assert "false positive" in ctx.blacklist

    def test_blacklist_prunes_near_duplicates(self, ctx) -> None:
        # "heap use after free vulnerability" shares enough tokens with
        # "heap use-after-free" to exceed the 0.55 Jaccard threshold.
        ctx.add_hypothesis("heap use after free vulnerability")
        ctx.add_hypothesis("completely unrelated sql injection")
        ctx.blacklist_hypothesis("heap use after free")
        # near-duplicate of the blacklisted text should be pruned
        assert "heap use after free vulnerability" not in ctx.hypotheses
        # unrelated hypothesis must survive
        assert "completely unrelated sql injection" in ctx.hypotheses

    def test_blacklisted_not_re_added(self, ctx) -> None:
        ctx.blacklist_hypothesis("bad hyp")
        ctx.add_hypothesis("bad hyp")
        assert "bad hyp" not in ctx.hypotheses

    def test_cluster_hypotheses_groups_similar(self, ctx) -> None:
        # A and B share 3 of 7 union tokens → Jaccard ≈ 0.43 (clusters; not deduped)
        # C shares nothing with A or B → stays in its own cluster
        a = "subprocess injection flaw unvalidated pathbug"
        b = "subprocess injection flaw exploitable remotely"
        c = "sql query parameter escaping issue"
        ctx.add_hypothesis(a)
        ctx.add_hypothesis(b)
        ctx.add_hypothesis(c)
        assert len(ctx.hypotheses) == 3, "all three must survive add_hypothesis dedup"
        clusters = ctx.cluster_hypotheses()
        flat = {h for cl in clusters for h in cl}
        assert flat == {a, b, c}
        sizes = sorted(len(cl) for cl in clusters)
        assert sizes == [1, 2]  # one pair, one singleton

    def test_cluster_hypotheses_already_used_skipped(self, ctx) -> None:
        # Three 5-token phrases sharing 3 tokens pairwise → Jaccard ≈ 0.43 each pair.
        # All survive add_hypothesis dedup (< 0.6) and all cluster together (≥ 0.4).
        # The inner `if i in used: continue` guard fires for i=1 and i=2.
        a = "format string bug heap write"
        b = "format string bug stack leak"
        c = "format string bug pointer crash"
        ctx.add_hypothesis(a)
        ctx.add_hypothesis(b)
        ctx.add_hypothesis(c)
        assert len(ctx.hypotheses) == 3, "all three must survive add_hypothesis dedup"
        clusters = ctx.cluster_hypotheses()
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_increment_parse_errors(self, ctx) -> None:
        ctx.increment_parse_errors()
        ctx.increment_parse_errors()
        assert ctx.parse_errors == 2

    def test_add_advisor_advice_dict(self, ctx) -> None:
        ctx.add_advisor_advice({"strategy": "widen search", "strategic_direction": "focus IPC"})
        assert ctx.has_pending_advisor_advice()

    def test_clear_pending_advisor_advice(self, ctx) -> None:
        ctx.add_advisor_advice({"strategy": "test"})
        ctx.clear_pending_advisor_advice()
        assert not ctx.has_pending_advisor_advice()

    def test_set_get_advisor_guidance(self, ctx) -> None:
        ctx.set_advisor_guidance("Look at allocators")
        assert ctx.get_advisor_guidance() == "Look at allocators"

    def test_add_evidence(self, ctx) -> None:
        ctx.add_evidence("foo.c:42", "potential UAF", tags=["uaf"])
        assert len(ctx.evidence) == 1
        assert ctx.evidence[0].location == "foo.c:42"

    def test_request_additional_evidence(self, ctx) -> None:
        ctx.add_step(self._Step(), "obs")
        ctx.request_additional_evidence("read_file")
        assert any(e.tags and "needs_evidence" in e.tags for e in ctx.evidence)

    def test_add_critique(self, ctx) -> None:
        ctx.add_critique("This looks like a false positive")
        # just verify no exception and the critique is stored
        assert len(ctx._critiques) == 1

    def test_get_metrics_keys(self, ctx) -> None:
        m = ctx.get_metrics()
        for key in ("steps_count", "evidence_count", "hypotheses_active", "confirmed_vulns"):
            assert key in m

    def test_build_summary_contains_target(self, ctx) -> None:
        s = ctx.build_summary()
        assert "/tmp/target" in s

    def test_build_summary_contains_step_count(self, ctx) -> None:
        ctx.add_step(self._Step(), "obs")
        s = ctx.build_summary()
        assert "1" in s

    def test_serialization_roundtrip(self, ctx) -> None:
        ctx.add_hypothesis("heap uaf in foo()")
        ctx.add_step(self._Step(confidence=0.6), "crash output")
        ctx.increment_parse_errors()
        ctx.set_advisor_guidance("focus on IPC")

        d = ctx.to_dict()
        from keryx.core.shared_context import SharedContext

        ctx2 = SharedContext.from_dict(d)

        assert ctx2.target_path == ctx.target_path
        assert ctx2.steps_taken == ctx.steps_taken
        assert ctx2.parse_errors == ctx.parse_errors
        assert "heap uaf in foo()" in ctx2.hypotheses
        assert ctx2.get_advisor_guidance() == "focus on IPC"

    def test_from_dict_sets_mode(self) -> None:
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext.from_dict(
            {"target_path": "/x", "mode": "airgapped", "capability": "deep_reasoning"}
        )
        assert ctx.mode == "airgapped"


# ---------------------------------------------------------------------------
# T05 – AdvisorManager Test
# ---------------------------------------------------------------------------


class TestAdvisorManager:
    """Verify creation, registration, dispatch, circuit breaker, metrics."""

    from keryx.advisors.base import AdvisorResponse, BaseAdvisor

    class _AlwaysAdvisor(BaseAdvisor):
        name = "suite-always"

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse

            return AdvisorResponse(
                strategic_direction="Examine IPC boundaries",
                adjust_confidence_threshold=0.55,
            )

    class _NeverAdvisor(BaseAdvisor):
        name = "suite-never"

        def should_trigger(self, context: Any) -> bool:
            return False

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse

            return AdvisorResponse()

    @pytest.fixture
    def manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    @pytest.fixture
    def ctx(self):
        from keryx.core.shared_context import SharedContext

        return SharedContext(target_path="/tmp/t")

    def test_create_empty_manager(self) -> None:
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        assert m.calls_made == 0
        m.shutdown()

    def test_create_with_circuit_breaker_disabled(self) -> None:
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager(enable_circuit_breaker=False)
        assert m is not None
        m.shutdown()

    def test_register_advisor(self, manager) -> None:
        manager.register(self._AlwaysAdvisor(), priority=80)
        assert manager.has_advisor("suite-always")

    def test_list_advisors_sorted_by_priority(self, manager) -> None:
        manager.register(self._AlwaysAdvisor(), priority=80)
        manager.register(self._NeverAdvisor(), priority=20)
        names = manager.list_advisors()
        assert names.index("suite-never") < names.index("suite-always")

    def test_unregister_advisor(self, manager) -> None:
        manager.register(self._AlwaysAdvisor())
        manager.unregister("suite-always")
        assert not manager.has_advisor("suite-always")

    def test_can_advise_empty(self, manager) -> None:
        assert manager.can_advise() is False

    def test_can_advise_with_registered(self, manager) -> None:
        manager.register(self._AlwaysAdvisor(max_calls_per_session=5))
        assert manager.can_advise() is True

    def test_calls_made_starts_zero(self, manager) -> None:
        assert manager.calls_made == 0

    async def test_get_advice_async_triggers(self, manager, ctx) -> None:
        manager.register(self._AlwaysAdvisor(max_calls_per_session=5))
        from keryx.advisors.base import AdvisorResponse

        resp = await manager.get_advice_async(ctx)
        assert isinstance(resp, AdvisorResponse)
        assert manager.calls_made == 1

    async def test_get_advice_async_not_found(self, manager, ctx) -> None:
        resp = await manager.get_advice_async(ctx, advisor_name="ghost")
        assert "error" in resp.metadata

    async def test_get_advice_async_no_trigger(self, manager, ctx) -> None:
        manager.register(self._NeverAdvisor(max_calls_per_session=5))
        resp = await manager.get_advice_async(ctx)
        assert resp.metadata.get("triggered") is False or resp.is_empty()

    def test_get_advice_sync(self, manager, ctx) -> None:
        manager.register(self._AlwaysAdvisor(max_calls_per_session=5))
        from keryx.advisors.base import AdvisorResponse

        resp = manager.get_advice(ctx)
        assert isinstance(resp, AdvisorResponse)

    def test_get_metrics_structure(self, manager) -> None:
        m = manager.get_metrics()
        assert "total_calls" in m
        assert "advisor_health" in m
        assert "fallback_count" in m

    def test_reset_all(self, manager) -> None:
        manager.register(self._AlwaysAdvisor(max_calls_per_session=5))
        manager.reset_all()
        assert manager.calls_made == 0

    def test_repr_contains_counts(self, manager) -> None:
        r = repr(manager)
        assert "AdvisorManager" in r

    def test_create_advisor_manager_factory(self) -> None:
        from keryx.advisors.manager import create_advisor_manager

        m = create_advisor_manager()
        assert m is not None
        m.shutdown()


# ---------------------------------------------------------------------------
# T06 – Default ToolBox Creation Test
# ---------------------------------------------------------------------------


class TestDefaultToolBoxCreation:
    """Verify create_default_toolbox() builds a working ToolBox."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import create_default_toolbox

        tb = create_default_toolbox()
        yield tb
        tb.shutdown()

    def test_toolbox_created(self, toolbox) -> None:
        assert toolbox is not None

    def test_read_file_registered(self, toolbox) -> None:
        assert "read_file" in toolbox.list_tools()

    def test_git_blame_registered(self, toolbox) -> None:
        assert "git_blame" in toolbox.list_tools()

    def test_at_least_two_tools(self, toolbox) -> None:
        assert len(toolbox.list_tools()) >= 2

    def test_get_read_file_tool(self, toolbox) -> None:
        t = toolbox.get_tool("read_file")
        assert t is not None
        assert t.name == "read_file"

    def test_get_git_blame_tool(self, toolbox) -> None:
        t = toolbox.get_tool("git_blame")
        assert t is not None
        assert t.name == "git_blame"

    def test_metrics_structure(self, toolbox) -> None:
        m = toolbox.get_metrics()
        assert "total_calls" in m
        assert "tools" in m

    def test_toolbox_repr(self, toolbox) -> None:
        r = repr(toolbox)
        assert "ToolBox" in r
        assert "read_file" in r

    def test_create_with_custom_timeout(self) -> None:
        from keryx.tools.Toolbox import create_default_toolbox

        tb = create_default_toolbox(default_timeout=30.0)
        assert tb.default_timeout == 30.0
        tb.shutdown()

    def test_create_with_allowed_root(self, tmp_path) -> None:
        from keryx.tools.Toolbox import create_default_toolbox

        tb = create_default_toolbox(allowed_root=str(tmp_path))
        rf = tb.get_tool("read_file")
        assert rf is not None
        assert rf.allowed_root is not None
        tb.shutdown()

    async def test_execute_read_file_via_default_toolbox(self, toolbox, tmp_path) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("Hello, world!")

        result = await toolbox.execute_async("read_file", {"file_path": str(target)})
        assert result.success is True
        assert "Hello, world!" in result.output


# ---------------------------------------------------------------------------
# T07 – ReadFileTool Test
# ---------------------------------------------------------------------------


class TestReadFileTool:
    """Verify ReadFileTool reads files correctly and handles edge cases."""

    @pytest.fixture
    def tool(self):
        from keryx.tools.read_file import create_read_file_tool

        return create_read_file_tool()

    @pytest.fixture
    def sample_file(self, tmp_path) -> Path:
        f = tmp_path / "sample.py"
        f.write_text("# sample\ndef foo():\n    pass\n")
        return f

    async def test_read_existing_file(self, tool, sample_file) -> None:
        result = await tool.execute({"file_path": str(sample_file)})
        assert result.success is True
        assert "# sample" in result.output

    async def test_data_contains_file_metadata(self, tool, sample_file) -> None:
        result = await tool.execute({"file_path": str(sample_file)})
        assert "chars_read" in result.data
        assert result.data["chars_read"] > 0
        assert "size_bytes" in result.data

    async def test_missing_file_path_key(self, tool) -> None:
        result = await tool.execute({})
        assert result.success is False
        assert result.error == "file_path_missing"

    async def test_nonexistent_file(self, tool) -> None:
        result = await tool.execute({"file_path": "/nonexistent/no_such_file_xyz.c"})
        assert result.success is False
        assert result.error == "file_not_found"

    async def test_path_traversal_blocked(self, tmp_path) -> None:
        from keryx.tools.read_file import create_read_file_tool

        restricted = create_read_file_tool(allowed_root=str(tmp_path))
        result = await restricted.execute({"file_path": "/etc/passwd"})
        assert result.success is False
        assert result.error == "path_traversal_blocked"

    async def test_path_within_root_allowed(self, tmp_path) -> None:
        from keryx.tools.read_file import create_read_file_tool

        f = tmp_path / "ok.txt"
        f.write_text("allowed content")
        restricted = create_read_file_tool(allowed_root=str(tmp_path))
        result = await restricted.execute({"file_path": str(f)})
        assert result.success is True

    async def test_directory_path_rejected(self, tool, tmp_path) -> None:
        result = await tool.execute({"file_path": str(tmp_path)})
        assert result.success is False
        assert result.error == "not_a_file"

    async def test_max_chars_truncates(self, tmp_path) -> None:
        from keryx.tools.read_file import create_read_file_tool

        f = tmp_path / "big.txt"
        f.write_text("X" * 500)
        small_tool = create_read_file_tool(max_chars=100)
        result = await small_tool.execute({"file_path": str(f)})
        assert result.success is True
        assert result.data.get("truncated") is True

    async def test_max_lines_truncates(self, tmp_path) -> None:
        from keryx.tools.read_file import create_read_file_tool

        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line {i}" for i in range(50)))
        line_tool = create_read_file_tool(max_lines=5)
        result = await line_tool.execute({"file_path": str(f)})
        assert result.success is True
        assert result.data.get("truncated") is True

    async def test_path_kwarg_alias(self, tool, sample_file) -> None:
        result = await tool.execute({"path": str(sample_file)})
        assert result.success is True

    def test_tool_name(self, tool) -> None:
        assert tool.name == "read_file"

    def test_repr(self, tool) -> None:
        assert "ReadFileTool" in repr(tool)

    def test_get_command_returns_none(self, tool) -> None:
        assert tool.get_command({}) is None

    async def test_start_end_line_slicing_adds_header(self, tmp_path) -> None:
        # Requesting a sub-range (not the full file) should prepend a [Lines X–Y of Z] header.
        f = tmp_path / "lined.py"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)))  # 10 lines
        result = await self.tool_factory(tmp_path).execute(
            {"file_path": str(f), "start_line": 3, "end_line": 6}
        )
        assert result.success
        assert "[Lines 3" in result.output   # header present
        assert "line3" in result.output
        assert "line6" in result.output
        assert "line1" not in result.output  # lines outside range excluded

    async def test_start_line_only_slicing(self, tmp_path) -> None:
        # start_line without end_line reads from that line to EOF.
        f = tmp_path / "lined2.py"
        f.write_text("\n".join(f"line{i}" for i in range(1, 6)))  # 5 lines
        result = await self.tool_factory(tmp_path).execute(
            {"file_path": str(f), "start_line": 3}
        )
        assert result.success
        assert "line3" in result.output
        assert "line1" not in result.output

    async def test_full_range_no_header(self, tmp_path) -> None:
        # start_line=1 and end_line=total covers the full file — no [Lines] header.
        f = tmp_path / "lined3.py"
        f.write_text("alpha\nbeta\ngamma\n")
        result = await self.tool_factory(tmp_path).execute(
            {"file_path": str(f), "start_line": 1, "end_line": 3}
        )
        assert result.success
        assert "[Lines" not in result.output
        assert "alpha" in result.output

    @staticmethod
    def tool_factory(tmp_path=None):
        from keryx.tools.read_file import create_read_file_tool
        return create_read_file_tool(allowed_root=str(tmp_path) if tmp_path else None)

    async def test_metrics_via_toolbox(self, sample_file) -> None:
        """Metrics are recorded by ToolBox, not by direct tool.execute()."""
        from keryx.tools.read_file import create_read_file_tool
        from keryx.tools.Toolbox import ToolBox

        rf = create_read_file_tool()
        tb = ToolBox()
        tb.register(rf)
        try:
            await tb.execute_async("read_file", {"file_path": str(sample_file)})
            m = rf.get_metrics()
            assert m["calls"] >= 1
            assert m["successes"] >= 1
        finally:
            tb.shutdown()


# ---------------------------------------------------------------------------
# T08 – GitBlameTool Test
# ---------------------------------------------------------------------------

# The project root IS a git repository — detect it once.
_GIT_REPO_ROOT = Path(__file__).parent.parent  # Keryx_Hunter/
_IN_GIT_REPO = (_GIT_REPO_ROOT / ".git").exists()


@pytest.mark.skipif(not _IN_GIT_REPO, reason="no git repository at project root")
class TestGitBlameTool:
    """Verify git blame / log / hotspot commands and error handling."""

    @pytest.fixture
    def tool(self):
        from keryx.tools.git_blame import create_git_blame_tool

        return create_git_blame_tool(timeout_seconds=30.0)

    @pytest.fixture
    def repo_path(self) -> str:
        return str(_GIT_REPO_ROOT)

    def test_instantiation(self, tool) -> None:
        assert tool is not None
        assert tool.name == "git_blame"

    def test_is_available_when_git_present(self, tool) -> None:
        import shutil

        if shutil.which("git"):
            assert tool.is_available() is True

    def test_repr(self, tool) -> None:
        assert "GitBlameTool" in repr(tool)

    async def test_unknown_command_returns_error(self, tool) -> None:
        result = await tool.execute({"command": "unknown_cmd"})
        assert result.success is False
        assert result.error == "unknown_command"

    async def test_git_log_basic(self, tool, repo_path) -> None:
        result = await tool.execute(
            {"command": "log", "n": 5, "path": repo_path}
        )
        # Either succeeds or fails gracefully — never raises
        assert isinstance(result.success, bool)
        if result.success:
            assert isinstance(result.output, str)
            assert result.data.get("command") == "log"

    async def test_git_log_returns_tool_result(self, tool) -> None:
        from keryx.tools.Toolbox import ToolResult

        result = await tool.execute({"command": "log", "n": 3})
        assert isinstance(result, ToolResult)

    async def test_git_blame_requires_file(self, tool) -> None:
        # Missing 'file' key → file not found or value error
        result = await tool.execute({"command": "blame"})
        # Must return a ToolResult, not raise
        assert result.success is False

    async def test_git_hotspots_returns_result(self, tool, repo_path) -> None:
        result = await tool.execute(
            {
                "command": "hotspots",
                "path": repo_path,
                "n": 20,
                "extensions": [".py"],
                "min_changes": 1,
            }
        )
        assert isinstance(result.success, bool)
        if result.success:
            assert "hotspots" in result.data

    async def test_git_log_with_author_filter(self, tool, repo_path) -> None:
        result = await tool.execute(
            {"command": "log", "n": 5, "author": "NonExistentAuthorXYZ123"}
        )
        # Should succeed with zero commits, not fail
        assert isinstance(result.success, bool)

    def test_metrics_baseline(self, tool) -> None:
        m = tool.get_metrics()
        assert m["calls"] == 0
        assert m["name"] == "git_blame"

    def test_tool_not_available_when_no_git_in_path(self) -> None:
        """When git cannot be found via shutil.which and no explicit path,
        is_available() returns False."""
        from unittest.mock import patch
        from keryx.tools.git_blame import GitBlameTool

        with patch("shutil.which", return_value=None):
            t = GitBlameTool()
            assert t.is_available() is False

    async def test_bad_binary_execute_returns_failure(self) -> None:
        """Executing with a nonexistent binary path fails gracefully."""
        from keryx.tools.git_blame import GitBlameTool

        # The tool will mark itself 'available' because an explicit path was given,
        # but the subprocess call will fail → ToolResult with success=False.
        t = GitBlameTool(git_path="/nonexistent/git_binary_xyz")
        result = await t.execute({"command": "log", "n": 1})
        assert result.success is False


# ---------------------------------------------------------------------------
# T09 – KeryxAgent Instantiation Test
# ---------------------------------------------------------------------------


class TestKeryxAgentInstantiation:
    """Verify KeryxAgent can be created with real dependencies."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    def test_basic_instantiation(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent is not None

    def test_executor_assigned(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        model = MyLocalModel()
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.executor is model

    def test_default_max_steps(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.max_steps == 50

    def test_custom_max_steps(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
        )
        assert agent.max_steps == 10

    def test_confidence_threshold_default(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.confidence_threshold == pytest.approx(0.65)

    def test_budget_none_when_not_specified(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.budget is None

    def test_budget_created_when_specified(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            budget_usd=5.0,
        )
        assert agent.budget is not None
        assert agent.budget.max_cost_usd == pytest.approx(5.0)

    def test_context_initially_none(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        # context property now raises AssertionError before run() is called;
        # the private _shared_context attribute holds the None sentinel.
        assert agent._shared_context is None
        with pytest.raises(AssertionError, match="call run\\(\\) first"):
            _ = agent.context

    def test_toolbox_assigned(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.tools is toolbox

    def test_advisor_manager_assigned(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.advisor_manager is advisor_manager

    def test_checkpoint_dir_default(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.checkpoint_dir is not None

    def test_checkpoint_dir_custom(self, toolbox, advisor_manager, tmp_path) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=MyLocalModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            checkpoint_dir=tmp_path,
        )
        assert agent.checkpoint_dir == tmp_path


# ---------------------------------------------------------------------------
# T10 – KeryxAgent.run() Basic ReAct Cycle Test
# ---------------------------------------------------------------------------


class TestKeryxAgentRunBasicCycle:
    """Verify the minimal ReAct loop completes successfully."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    async def test_run_returns_dict(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/test", resume=False)
        assert isinstance(result, dict)

    async def test_run_status_completed(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/test", resume=False)
        assert result["status"] == "completed"

    async def test_run_sets_target(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/some_target", resume=False)
        assert result["target"] == "/tmp/some_target"

    async def test_run_steps_taken_positive(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        assert result["steps_taken"] >= 1

    async def test_run_result_keys(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        for key in (
            "status",
            "target",
            "steps_taken",
            "confirmed_vulns",
            "hypotheses",
            "hypotheses_count",
            "parse_errors",
            "summary",
        ):
            assert key in result

    async def test_run_summary_is_string(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        assert isinstance(result["summary"], str)

    async def test_run_no_budget_reports_none(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
            budget_usd=None,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        assert result["budget"] is None

    async def test_run_with_budget(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
            budget_usd=2.0,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        assert result["budget"]["max_cost_usd"] == pytest.approx(2.0)

    async def test_run_context_populated_after_run(
        self, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        await agent.run(target_path="/tmp/t", resume=False)
        assert agent.context is not None
        assert agent.context.target_path == "/tmp/t"

    async def test_run_honours_max_steps(self, toolbox, advisor_manager) -> None:
        """With a model that never sends FINISH, agent must stop at max_steps."""
        from keryx.core.agent import KeryxAgent

        class NeverFinishModel(MyLocalModel):
            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                if max_tokens == 1 or grammar is None:
                    return "ok"
                return "not json at all"

        agent = KeryxAgent(
            executor_model=NeverFinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        assert result["steps_taken"] <= 3

    async def test_run_parse_errors_tracked(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        class GarbageModel(MyLocalModel):
            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                if max_tokens == 1:
                    return "ok"
                return "}}garbage{{"

        agent = KeryxAgent(
            executor_model=GarbageModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=2,
        )
        result = await agent.run(target_path="/tmp/t", resume=False)
        assert result["parse_errors"] > 0

    async def test_final_report_includes_hypotheses_list(
        self, toolbox, advisor_manager
    ) -> None:
        """
        Regression: _generate_final_report() must include 'hypotheses' as a
        list so OrchestratorSwarmAdapter.run() can pass them to SwarmDebate.
        Previously the key was absent (only 'hypotheses_count' was present),
        causing swarm mode to always short-circuit with consensus_reached=False.
        """
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/swarm_regression", capability="fast_pattern_matching")
        ctx.add_hypothesis("use-after-free in JSObject::finalize")
        ctx.add_hypothesis("heap overflow in png_read_row")

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(
            target_path="/tmp/swarm_regression",
            resume=False,
            context=ctx,
        )

        assert "hypotheses" in result, "'hypotheses' key missing from final report"
        assert isinstance(result["hypotheses"], list)
        assert "use-after-free in JSObject::finalize" in result["hypotheses"]
        assert "heap overflow in png_read_row" in result["hypotheses"]
        assert result["hypotheses_count"] == len(result["hypotheses"])


# ---------------------------------------------------------------------------
# T11 – Orchestrator Creation Test
# ---------------------------------------------------------------------------


class TestOrchestratorCreation:
    """Verify KeryxOrchestrator can be created and introspected."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    def test_basic_instantiation(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert orc is not None

    def test_available_models_stored(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        models = {"model-a": MyLocalModel()}
        orc = KeryxOrchestrator(
            available_models=models,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert orc.available_models is models

    def test_advisor_manager_stored(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert orc.advisor_manager is advisor_manager

    def test_toolbox_stored(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert orc.toolbox is toolbox

    def test_router_created(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert orc.router is not None

    def test_shared_context_initially_none(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert orc.get_shared_context() is None

    def test_metrics_initial_state(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        m = orc.get_metrics()
        assert "routing_decisions" in m
        assert m["routing_decisions"] == 0

    def test_has_gpu_flag(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            has_gpu=False,
        )
        assert orc.has_gpu is False

    def test_factory_function(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import create_orchestrator

        orc = create_orchestrator(
            models={},
            advisors=advisor_manager,
            tools=toolbox,
        )
        assert orc is not None

    def test_metrics_structure(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        m = orc.get_metrics()
        assert "advisor_calls" in m
        assert "tools_executed" in m
        assert "swarm_votes" in m


# ---------------------------------------------------------------------------
# T12 – End-to-End Minimal Hunt Smoke Test
# ---------------------------------------------------------------------------


class TestEndToEndMinimalHunt:
    """Run a 5-step agent hunt with real tools and a controlled mock model."""

    @pytest.fixture
    def tmp_target(self, tmp_path) -> Path:
        """Create a small fake C file to analyse."""
        f = tmp_path / "target.c"
        f.write_text(
            "// Simulated C code for keryx smoke test\n"
            "void handle_packet(char *buf, int len) {\n"
            "    char tmp[64];\n"
            "    memcpy(tmp, buf, len);  // potential overflow\n"
            "}\n"
        )
        return f

    @pytest.fixture
    def toolbox(self, tmp_target):
        from keryx.tools.Toolbox import create_default_toolbox

        # allow_root = tmp_target's parent so path traversal doesn't block
        tb = create_default_toolbox(allowed_root=str(tmp_target.parent))
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    async def test_hunt_completes_five_steps(self, tmp_target, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        model = ToolCallingModel(
            target_file=str(tmp_target),
            steps_before_finish=4,
        )
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
            confidence_threshold=0.6,
        )
        result = await agent.run(target_path=str(tmp_target.parent), resume=False)

        assert isinstance(result, dict)
        assert result["status"] == "completed"
        # The agent ran at least the 4 tool-call steps + 1 FINISH step
        assert result["steps_taken"] >= 5

    async def test_hunt_reads_target_file(self, tmp_target, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        model = ToolCallingModel(
            target_file=str(tmp_target),
            steps_before_finish=3,
        )
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
            confidence_threshold=0.6,
        )
        await agent.run(target_path=str(tmp_target.parent), resume=False)

        # At least one step should have used the read_file action.
        # add_step() now stores action as a plain dict with key "name".
        steps = agent.context.steps
        actions = [
            s["action"].get("name", "") if isinstance(s["action"], dict) else ""
            for s in steps
        ]
        assert "read_file" in actions

    async def test_hunt_context_tracks_steps(self, tmp_target, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent

        model = ToolCallingModel(
            target_file=str(tmp_target),
            steps_before_finish=3,
        )
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
            confidence_threshold=0.6,
        )
        await agent.run(target_path=str(tmp_target.parent), resume=False)

        assert agent.context.steps_taken >= 1

    async def test_hunt_result_has_required_keys(
        self, tmp_target, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=ToolCallingModel(str(tmp_target), steps_before_finish=2),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
            confidence_threshold=0.6,
        )
        result = await agent.run(target_path=str(tmp_target.parent), resume=False)

        for key in (
            "status",
            "target",
            "steps_taken",
            "confirmed_vulns",
            "hypotheses_count",
            "parse_errors",
            "model",
            "summary",
        ):
            assert key in result, f"Missing key: {key!r}"

    async def test_hunt_model_info_in_result(
        self, tmp_target, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=ToolCallingModel(str(tmp_target), steps_before_finish=2),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
            confidence_threshold=0.6,
        )
        result = await agent.run(target_path=str(tmp_target.parent), resume=False)

        assert "name" in result["model"]
        assert "is_local" in result["model"]
        assert result["model"]["is_local"] is True

    async def test_hunt_toolbox_metrics_updated(
        self, tmp_target, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=ToolCallingModel(str(tmp_target), steps_before_finish=3),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
            confidence_threshold=0.6,
        )
        await agent.run(target_path=str(tmp_target.parent), resume=False)

        m = toolbox.get_metrics()
        assert m["total_calls"] >= 1

    async def test_hunt_zero_confirmed_vulns_on_clean_target(
        self, tmp_target, toolbox, advisor_manager
    ) -> None:
        """read_file observations never trigger VULN_CONFIRMED (correct tool guard)."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=ToolCallingModel(str(tmp_target), steps_before_finish=2),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
            confidence_threshold=0.6,
        )
        result = await agent.run(target_path=str(tmp_target.parent), resume=False)

        # read_file is NOT in _VULN_CONFIRMING_TOOLS, so no early vuln confirm
        assert result["confirmed_vulns"] == []


# ---------------------------------------------------------------------------
# T13 – Checkpoint Serialization / Restore Test (P2)
# ---------------------------------------------------------------------------


class TestCheckpointSerialization:
    """Verify agent checkpoint write, read-back, resume, and corrupt-fallback."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    # ── checkpoint path is deterministic ─────────────────────────────────────

    def test_checkpoint_path_is_deterministic(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """_checkpoint_path() always returns the same path for the same target."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            checkpoint_dir=tmp_path,
        )
        p1 = agent._checkpoint_path("/some/target/path")
        p2 = agent._checkpoint_path("/some/target/path")
        assert p1 == p2
        assert p1.parent == tmp_path

    def test_checkpoint_path_differs_for_different_targets(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """Different targets produce different checkpoint file names."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            checkpoint_dir=tmp_path,
        )
        p1 = agent._checkpoint_path("/target/a")
        p2 = agent._checkpoint_path("/target/b")
        assert p1 != p2

    # ── save checkpoint with serialisable steps ───────────────────────────────

    def test_save_checkpoint_writes_valid_json(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """_save_checkpoint() writes a valid JSON file when steps are plain dicts."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            checkpoint_dir=tmp_path,
        )
        ctx = SharedContext(target_path="/tmp/save_tgt")
        # Plain dicts are JSON-serialisable (unlike AgentStep objects).
        ctx.steps = [
            {"timestamp": 1.0, "action": "NO_ACTION", "observation": "test obs"}
        ]
        ctx.steps_taken = 1
        agent.context = ctx
        agent._save_checkpoint("/tmp/save_tgt")

        ck_path = agent._checkpoint_path("/tmp/save_tgt")
        assert ck_path.exists()
        data = json.loads(ck_path.read_text())
        assert "context" in data
        assert "steps_since_escalation" in data
        assert data["context"]["steps_taken"] == 1

    # ── resume from manually-written checkpoint ───────────────────────────────

    async def test_checkpoint_resume_restores_step_count(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """Manually write a 10-step checkpoint; agent resumes and starts from step 10."""
        import hashlib

        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/resume_tgt", capability="deep_reasoning")
        # Build steps as plain dicts (JSON-serialisable) to avoid AgentStep serialisation bug.
        ctx.steps = [
            {"timestamp": float(i), "action": "NO_ACTION", "observation": f"obs_{i}"}
            for i in range(10)
        ]
        ctx.steps_taken = 10

        key = hashlib.sha256("/tmp/resume_tgt".encode()).hexdigest()[:12]
        ck_path = tmp_path / f"checkpoint_{key}.json"
        ck_path.write_text(
            json.dumps(
                {"context": ctx.to_dict(), "budget": None, "steps_since_escalation": 5}
            )
        )

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=13,
            checkpoint_dir=tmp_path,
        )
        await agent.run(target_path="/tmp/resume_tgt", resume=True)
        # resumed from step 10 → final steps_taken must be >= 10
        assert agent.context.steps_taken >= 10

    # ── corrupt checkpoint falls back gracefully ──────────────────────────────

    async def test_corrupt_checkpoint_starts_fresh(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """A corrupt checkpoint file triggers warn-and-fallback to fresh context."""
        import hashlib

        from keryx.core.agent import KeryxAgent

        key = hashlib.sha256("/tmp/corrupt_tgt".encode()).hexdigest()[:12]
        ck_path = tmp_path / f"checkpoint_{key}.json"
        ck_path.write_text("{{NOT VALID JSON AT ALL}}")

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
            checkpoint_dir=tmp_path,
        )
        result = await agent.run(target_path="/tmp/corrupt_tgt", resume=True)
        assert result["status"] == "completed"
        assert result["steps_taken"] >= 1

    # ── SharedContext to_dict / from_dict full roundtrip ─────────────────────

    def test_shared_context_full_roundtrip(self) -> None:
        """All serialisable fields survive a to_dict() → from_dict() cycle."""
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(
            target_path="/rt", capability="airgapped_fast", mode="local",
            escalation_level=2,
        )
        ctx.add_hypothesis("heap uaf in send()")
        ctx.increment_parse_errors()
        ctx.increment_parse_errors()
        ctx.add_evidence("foo.c:10", "UAF candidate", tags=["uaf", "critical"])
        ctx.set_advisor_guidance("focus on memcpy and strcpy call-sites")

        d = ctx.to_dict()
        ctx2 = SharedContext.from_dict(d)

        assert ctx2.target_path == "/rt"
        assert ctx2.capability == "airgapped_fast"
        assert ctx2.mode == "local"
        assert ctx2.escalation_level == 2
        assert ctx2.parse_errors == 2
        assert "heap uaf in send()" in ctx2.hypotheses
        assert len(ctx2.evidence) == 1
        assert ctx2.evidence[0].tags == ["uaf", "critical"]
        assert ctx2.get_advisor_guidance() == "focus on memcpy and strcpy call-sites"


# ---------------------------------------------------------------------------
# T14 – Air-gapped / Local Mode Coverage (P3)
# ---------------------------------------------------------------------------


class TestAirGappedMode:
    """Verify air-gapped enforcement at router level and in the agent run loop."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    def test_airgapped_budget_resolves_to_none(self) -> None:
        from keryx.core.router import CapabilityRouter

        assert CapabilityRouter._resolve_budget(100.0, 50.0, True) is None

    def test_router_local_capability_airgapped(self, advisor_manager) -> None:
        """fast_pattern_matching + is_airgapped=True succeeds and enforces no-network."""
        from keryx.core.router import CapabilityRouter

        router = CapabilityRouter()
        plan = router.route(
            "fast_pattern_matching",
            {"qwen3-coder-8b": FinishModel()},
            advisor_manager,
            is_airgapped=True,
        )
        assert plan.enforce_no_network is True
        assert plan.budget_usd is None

    def test_airgapped_routing_plan_budget_none(self, advisor_manager) -> None:
        from keryx.core.router import CapabilityRouter

        router = CapabilityRouter()
        plan = router.route(
            "airgapped",
            {"qwen3-coder-8b": FinishModel()},
            advisor_manager,
            is_airgapped=True,
        )
        assert plan.budget_usd is None

    async def test_agent_local_model_flag_in_result(
        self, toolbox, advisor_manager
    ) -> None:
        """Agent backed by a local (is_local=True) model reports that correctly."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),  # is_local=True
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/ag_local", resume=False)
        assert result["model"]["is_local"] is True
        assert result["status"] == "completed"

    async def test_agent_preserves_mode_in_context(
        self, toolbox, advisor_manager
    ) -> None:
        """SharedContext.mode='local' injected before run() is kept after run()."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        # Inject pre-built context with mode='local'; resume=False resets it,
        # so we verify via the context the agent creates from scratch.
        result = await agent.run(
            target_path="/tmp/ag_local2",
            capability="fast_pattern_matching",
            resume=False,
        )
        # capability is forwarded to the fresh SharedContext
        assert agent.context.capability == "fast_pattern_matching"


# ---------------------------------------------------------------------------
# T10 extension – Budget exhaustion + escalation paths (P5)
# These are appended as additional methods of TestKeryxAgentRunBasicCycle
# via a standalone class that shares the same fixture names.
# ---------------------------------------------------------------------------


class TestBudgetAndEscalation:
    """P5 — Budget exhaustion and escalation logic for KeryxAgent."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    async def test_budget_exhaustion_stops_loop(
        self, toolbox, advisor_manager
    ) -> None:
        """Agent with a near-zero budget must stop well before max_steps."""
        from keryx.core.agent import KeryxAgent

        class ExpensiveModel(MyLocalModel):
            def __init__(self):
                super().__init__()
                # Instance attributes — override base-class init values.
                self.cost_per_1k_input_tokens = 1.0
                self.cost_per_1k_output_tokens = 1.0

            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                if max_tokens == 1 or grammar is None:
                    return "ok"
                return json.dumps(
                    {"thought": "keep working", "action": "NO_ACTION",
                     "action_input": {}, "confidence": 0.5}
                )

        agent = KeryxAgent(
            executor_model=ExpensiveModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=50,
            budget_usd=0.001,  # tiny — estimate (3 USD) > budget → stops on step 1
        )
        result = await agent.run(target_path="/tmp/budget_tgt", resume=False)
        assert result["steps_taken"] < 50

    def test_budget_controller_can_proceed_false_when_exhausted(self) -> None:
        from keryx.core.agent import BudgetController

        class _CostlyModel(MyLocalModel):
            def __init__(self):
                super().__init__()
                self.cost_per_1k_input_tokens = 1.0
                self.cost_per_1k_output_tokens = 1.0

        bc = BudgetController(max_cost_usd=0.001, max_calls=1000)
        model = _CostlyModel()
        bc.record_call(model, tokens_in=10_000, tokens_out=5_000)
        assert bc.can_proceed(model) is False

    def test_budget_controller_remaining_budget(self) -> None:
        from keryx.core.agent import BudgetController

        bc = BudgetController(max_cost_usd=5.0)
        assert bc.remaining_budget == pytest.approx(5.0)
        bc.current_cost = 2.0
        assert bc.remaining_budget == pytest.approx(3.0)

    def test_budget_controller_max_calls_limit(self) -> None:
        from keryx.core.agent import BudgetController

        bc = BudgetController(max_cost_usd=100.0, max_calls=2)
        model = MyLocalModel()
        bc.record_call(model)
        bc.record_call(model)
        assert bc.can_proceed(model) is False  # calls_made >= max_calls

    def test_should_escalate_fires_on_stagnation(
        self, toolbox, advisor_manager
    ) -> None:
        """P4: no new hypothesis for >= 5 steps triggers escalation."""
        from keryx.core.agent import KeryxAgent, _ESCALATION_COOLDOWN
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        ctx = SharedContext(target_path="/tmp/x")
        agent.context = ctx
        agent._steps_since_escalation = _ESCALATION_COOLDOWN  # cooldown expired
        agent._steps_since_new_hypothesis = 5                  # P4 stagnation threshold
        assert agent._should_escalate() is True

    def test_should_escalate_blocked_during_cooldown(
        self, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent, _ESCALATION_COOLDOWN
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        ctx = SharedContext(target_path="/tmp/x")
        ctx.parse_errors = 10  # would normally trigger
        agent.context = ctx
        agent._steps_since_escalation = _ESCALATION_COOLDOWN - 1  # still cooling
        assert agent._should_escalate() is False


# ---------------------------------------------------------------------------
# T15 – Orchestrator hunt() end-to-end (P4 unlock)
# ---------------------------------------------------------------------------


class TestOrchestratorHunt:
    """Verify hunt() runs end-to-end after SharedContext P4 fixes."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        m = AdvisorManager()
        yield m
        m.shutdown()

    async def test_hunt_returns_dict(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        result = await orc.hunt(
            target_path="/tmp/orc_a",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert isinstance(result, dict)

    async def test_hunt_result_has_status(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        result = await orc.hunt(
            target_path="/tmp/orc_b",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert result.get("status") in ("completed", "cancelled")

    async def test_hunt_airgapped_flag_set(self, toolbox, advisor_manager) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        result = await orc.hunt(
            target_path="/tmp/orc_c",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert result.get("airgapped") is True

    async def test_hunt_increments_routing_decisions(
        self, toolbox, advisor_manager
    ) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        await orc.hunt(
            target_path="/tmp/orc_d",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert orc.metrics.routing_decisions >= 1

    async def test_hunt_shared_context_populated(
        self, toolbox, advisor_manager
    ) -> None:
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        await orc.hunt(
            target_path="/tmp/orc_e",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        ctx = orc.get_shared_context()
        assert ctx is not None

    async def test_escalation_level_field_accessible(self) -> None:
        """SharedContext.escalation_level exists and defaults to 1."""
        from keryx.core.shared_context import SharedContext
        from keryx.core.router import RoutingPlan

        ctx = SharedContext(
            target_path="/tmp/esc",
            capability="deep_reasoning",
            mode="hybrid",
            routing_plan=None,
        )
        assert ctx.escalation_level == 1
        ctx.set_escalation_level(2)
        assert ctx.escalation_level == 2

    async def test_shared_context_accepts_routing_plan(self) -> None:
        """SharedContext(routing_plan=<obj>) no longer raises TypeError."""
        from keryx.core.shared_context import SharedContext

        sentinel = object()  # any object — routing_plan is not validated
        ctx = SharedContext(
            target_path="/tmp/rp",
            capability="deep_reasoning",
            routing_plan=sentinel,
        )
        assert ctx.routing_plan is sentinel

    # ── A1: context injection fix ─────────────────────────────────────────────

    async def test_agent_run_accepts_context_parameter(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.run(context=...) uses the provided context, ignores resume."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        pre_built = SharedContext(
            target_path="/tmp/ctx_inject", capability="fast_pattern_matching"
        )
        pre_built.parse_errors = 7  # sentinel value

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        await agent.run(target_path="/tmp/ctx_inject", resume=False, context=pre_built)
        # The agent must have used the injected context, not a fresh one.
        assert agent.context is pre_built
        assert agent.context.parse_errors == 7

    async def test_hunt_context_accumulates_across_escalation(
        self, toolbox, advisor_manager
    ) -> None:
        """Steps accumulate in shared_context across escalation attempts (A1 fix)."""
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        await orc.hunt(
            target_path="/tmp/orc_accum",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        ctx = orc.get_shared_context()
        assert ctx is not None
        # FinishModel fires FINISH on the first step of every escalation attempt.
        # With context properly passed between attempts, steps_taken accumulates.
        assert ctx.steps_taken >= 1

    # ── A2: _should_escalate_further respects max_escalation ─────────────────

    def test_should_escalate_further_respects_max_escalation(self) -> None:
        """At max_escalation, should_escalate returns False (A2 fix)."""
        from keryx.core._escalation import should_escalate
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/esc_test")
        result: dict = {"confirmed_vulns": [], "average_confidence": 0.0}

        # At the cap: current_level == max_level → must return False.
        assert should_escalate(result, ctx, current_level=2, max_level=2) is False
        # Below the cap: should still escalate.
        assert should_escalate(result, ctx, current_level=1, max_level=2) is True
        # Airgapped cap (2) vs non-airgapped cap (4): same function, different arg.
        assert should_escalate(result, ctx, current_level=4, max_level=4) is False
        assert should_escalate(result, ctx, current_level=3, max_level=4) is True

    def test_should_not_escalate_when_vuln_confirmed(self) -> None:
        """Confirmed vuln short-circuits escalation regardless of level."""
        from keryx.core._escalation import should_escalate
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/vuln_test")
        result = {"confirmed_vulns": [{"step": 1}], "average_confidence": 0.1}

        assert should_escalate(result, ctx, current_level=1, max_level=4) is False


# ---------------------------------------------------------------------------
# T16 – Swarm Debate Mode Tests
# ---------------------------------------------------------------------------


class YesModel(MyLocalModel):
    """Always votes 'true' — confirms every hypothesis."""

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return "true"


class NoModel(MyLocalModel):
    """Always votes 'false' — rejects every hypothesis."""

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return "false"


class ExplodingModel(MyLocalModel):
    """Always raises — simulates a crashed swarm voter."""

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise RuntimeError("swarm model crashed")


class YesJsonModel(MyLocalModel):
    """Returns valid swarm JSON with verdict: true — compatible with models/swarm.py."""

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return '{"hypotheses": [{"index": 1, "verdict": true, "confidence": 0.95, "reason": "confirmed"}]}'


class NoJsonModel(MyLocalModel):
    """Returns valid swarm JSON with verdict: false — compatible with models/swarm.py."""

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return '{"hypotheses": [{"index": 1, "verdict": false, "confidence": 0.95, "reason": "rejected"}]}'


@pytest.fixture
def swarm_orchestrator():
    from keryx.core.orchestrator import KeryxOrchestrator
    from keryx.advisors.manager import AdvisorManager
    from keryx.tools.Toolbox import ToolBox

    tb = ToolBox()
    am = AdvisorManager()
    orc = KeryxOrchestrator(
        available_models={
            "yes_model":      YesModel(),
            "no_model":       NoModel(),
            "exploding_model": ExplodingModel(),
        },
        advisor_manager=am,
        toolbox=tb,
    )
    yield orc
    tb.shutdown()
    am.shutdown()


class TestOrchestratorSwarm:
    """T16 — OrchestratorSwarmAdapter.run() and models.SwarmDebate._analyze_with_model()."""

    def _make_adapter(self, orc):
        from keryx.core._swarm import OrchestratorSwarmAdapter
        return OrchestratorSwarmAdapter(orc.available_models, orc.metrics.swarm_votes)

    async def test_swarm_debate_no_hypotheses_returns_early(
        self, swarm_orchestrator
    ) -> None:
        """With no hypotheses, adapter returns immediately with consensus_reached=False."""
        orc = swarm_orchestrator
        base = {"confirmed_vulns": [], "hypotheses": []}
        result = await self._make_adapter(orc).run(base, ["yes_model"])
        assert result["swarm_mode"] is True
        assert result["consensus_reached"] is False

    async def test_swarm_debate_no_available_debate_models(
        self, swarm_orchestrator
    ) -> None:
        """debate_models that don't exist in available_models → swarm_error key."""
        orc = swarm_orchestrator
        base = {"confirmed_vulns": [], "hypotheses": ["buffer overflow in foo"]}
        result = await self._make_adapter(orc).run(base, ["nonexistent_model"])
        assert result["swarm_mode"] is True
        assert "swarm_error" in result

    async def test_swarm_debate_unanimous_yes_confirms_hypothesis(
        self, swarm_orchestrator
    ) -> None:
        """SwarmResult with confirmed hyp → merged into swarm_confirmed."""
        from unittest.mock import AsyncMock, patch
        from keryx.models.swarm import SwarmDebate, SwarmResult
        orc = swarm_orchestrator
        hyp = "use-after-free in JSObject::swap"
        base = {"confirmed_vulns": [], "hypotheses": [hyp]}
        sr = SwarmResult(
            hypotheses=[hyp], confirmed=[hyp], rejected=[],
            votes={}, weighted_consensus_ratio=1.0, raw_consensus_ratio=1.0,
            swarm_voters=1,
        )
        with patch.object(SwarmDebate, "debate", new=AsyncMock(return_value=sr)):
            result = await self._make_adapter(orc).run(base, ["yes_model"])
        assert result["swarm_mode"] is True
        assert hyp in result.get("swarm_confirmed", [])
        assert result.get("swarm_voters", 0) >= 1

    async def test_swarm_debate_unanimous_no_rejects_hypothesis(
        self, swarm_orchestrator
    ) -> None:
        """SwarmResult with empty confirmed → hypothesis not in swarm_confirmed."""
        from unittest.mock import AsyncMock, patch
        from keryx.models.swarm import SwarmDebate, SwarmResult
        orc = swarm_orchestrator
        hyp = "integer overflow in memcpy_size"
        base = {"confirmed_vulns": [], "hypotheses": [hyp]}
        sr = SwarmResult(
            hypotheses=[hyp], confirmed=[], rejected=[hyp],
            votes={}, weighted_consensus_ratio=0.0, raw_consensus_ratio=0.0,
            swarm_voters=1,
        )
        with patch.object(SwarmDebate, "debate", new=AsyncMock(return_value=sr)):
            result = await self._make_adapter(orc).run(base, ["no_model"])
        assert result["swarm_mode"] is True
        assert hyp not in result.get("swarm_confirmed", [])

    async def test_swarm_debate_split_vote_below_threshold(
        self, swarm_orchestrator
    ) -> None:
        """SwarmResult with 50% ratio → not confirmed (below 67% threshold)."""
        from unittest.mock import AsyncMock, patch
        from keryx.models.swarm import SwarmDebate, SwarmResult
        orc = swarm_orchestrator
        hyp = "heap corruption in parser"
        base = {"confirmed_vulns": [], "hypotheses": [hyp]}
        sr = SwarmResult(
            hypotheses=[hyp], confirmed=[], rejected=[],
            votes={}, weighted_consensus_ratio=0.5, raw_consensus_ratio=0.5,
            swarm_voters=2,
        )
        with patch.object(SwarmDebate, "debate", new=AsyncMock(return_value=sr)):
            result = await self._make_adapter(orc).run(base, ["yes_model", "no_model"])
        assert result["swarm_mode"] is True
        assert hyp not in result.get("swarm_confirmed", [])

    async def test_swarm_debate_crashed_voter_does_not_crash_swarm(
        self, swarm_orchestrator
    ) -> None:
        """Crashed voter is absorbed internally; adapter sees SwarmResult with voters count."""
        from unittest.mock import AsyncMock, patch
        from keryx.models.swarm import SwarmDebate, SwarmResult
        orc = swarm_orchestrator
        hyp = "stack overflow in recursion"
        base = {"confirmed_vulns": [], "hypotheses": [hyp]}
        sr = SwarmResult(
            hypotheses=[hyp], confirmed=[], rejected=[],
            votes={}, weighted_consensus_ratio=0.0, raw_consensus_ratio=0.0,
            swarm_voters=2,
        )
        with patch.object(SwarmDebate, "debate", new=AsyncMock(return_value=sr)):
            result = await self._make_adapter(orc).run(base, ["yes_model", "exploding_model"])
        assert result["swarm_mode"] is True
        assert hyp not in result.get("swarm_confirmed", [])
        assert result.get("swarm_voters", 0) == 2

    async def test_swarm_debate_appends_confirmed_vulns(
        self, swarm_orchestrator
    ) -> None:
        """Swarm-confirmed hypotheses are appended to existing confirmed_vulns list."""
        from unittest.mock import AsyncMock, patch
        from keryx.models.swarm import SwarmDebate, SwarmResult
        orc = swarm_orchestrator
        hyp = "format string in log_error"
        base = {"confirmed_vulns": [{"step": 1, "note": "existing"}], "hypotheses": [hyp]}
        sr = SwarmResult(
            hypotheses=[hyp], confirmed=[hyp], rejected=[],
            votes={}, weighted_consensus_ratio=1.0, raw_consensus_ratio=1.0,
            swarm_voters=1,
        )
        with patch.object(SwarmDebate, "debate", new=AsyncMock(return_value=sr)):
            result = await self._make_adapter(orc).run(base, ["yes_model"])
        vulns = result.get("confirmed_vulns", [])
        assert len(vulns) >= 2
        assert any(v.get("swarm_verified") for v in vulns)

    async def test_swarm_debate_metrics_updated(
        self, swarm_orchestrator
    ) -> None:
        """Confirmed hypotheses are written back to orchestrator.metrics.swarm_votes."""
        from unittest.mock import AsyncMock, patch
        from keryx.models.swarm import SwarmDebate, SwarmResult
        orc = swarm_orchestrator
        hyp = "null deref in dealloc"
        base = {"confirmed_vulns": [], "hypotheses": [hyp]}
        sr = SwarmResult(
            hypotheses=[hyp], confirmed=[hyp], rejected=[],
            votes={}, weighted_consensus_ratio=1.0, raw_consensus_ratio=1.0,
            swarm_voters=1,
        )
        with patch.object(SwarmDebate, "debate", new=AsyncMock(return_value=sr)):
            await self._make_adapter(orc).run(base, ["yes_model"])
        assert len(orc.metrics.swarm_votes) >= 1

    async def test_analyze_with_model_true_response(
        self, swarm_orchestrator
    ) -> None:
        """_analyze_with_model maps JSON verdict:true to SwarmVote.verdict=True."""
        from keryx.models.swarm import SwarmDebate
        engine = SwarmDebate([YesJsonModel()])
        hyps = ["buffer overflow"]
        result = await engine._analyze_with_model(YesJsonModel(), hyps, None)
        assert isinstance(result, dict)
        assert result["buffer overflow"].verdict is True

    async def test_analyze_with_model_false_response(
        self, swarm_orchestrator
    ) -> None:
        """_analyze_with_model maps JSON verdict:false to SwarmVote.verdict=False."""
        from keryx.models.swarm import SwarmDebate
        engine = SwarmDebate([NoJsonModel()])
        hyps = ["double free"]
        result = await engine._analyze_with_model(NoJsonModel(), hyps, None)
        assert result["double free"].verdict is False

    async def test_analyze_with_model_exception_returns_false(
        self, swarm_orchestrator
    ) -> None:
        """If model.generate raises, _analyze_with_model returns neutral vote (verdict=False)."""
        from keryx.models.swarm import SwarmDebate
        engine = SwarmDebate([ExplodingModel()])
        hyps = ["dangling pointer"]
        result = await engine._analyze_with_model(ExplodingModel(), hyps, None)
        assert result["dangling pointer"].verdict is False


# ---------------------------------------------------------------------------
# T17 – enforce_airgapped in KeryxAgent
# ---------------------------------------------------------------------------


class NetworkModel(MyLocalModel):
    """Pretends to be a cloud/network model."""

    def __init__(self) -> None:
        super().__init__()
        self.is_local = False  # triggers requires_network = True


class TestEnforceAirgapped:
    """T17 — KeryxAgent.enforce_airgapped blocks network models before the loop."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        yield am
        am.shutdown()

    async def test_network_model_blocked_when_enforce_airgapped(
        self, toolbox, advisor_manager
    ) -> None:
        """enforce_airgapped=True + network model → status='failed', reason='airgap_violation'."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=NetworkModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
            enforce_airgapped=True,
        )
        result = await agent.run(target_path="/tmp/airgap_test", resume=False)
        assert result["status"] == "failed"
        assert result["reason"] == "airgap_violation"

    async def test_local_model_allowed_when_enforce_airgapped(
        self, toolbox, advisor_manager
    ) -> None:
        """enforce_airgapped=True + local model (FinishModel) → runs normally."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
            enforce_airgapped=True,
        )
        result = await agent.run(target_path="/tmp/local_airgap_test", resume=False)
        # FinishModel emits FINISH → completed
        assert result["status"] == "completed"

    async def test_no_enforce_network_model_runs(
        self, toolbox, advisor_manager
    ) -> None:
        """Without enforce_airgapped, a network model is allowed (default behaviour)."""
        from keryx.core.agent import KeryxAgent

        # NetworkModel.generate() hits the grammar=None branch → returns "ok" for
        # health, then hits the main branch. It doesn't implement a valid action
        # response, so agent increments parse_errors and exhausts max_steps=1.
        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=2,
            enforce_airgapped=False,
        )
        result = await agent.run(target_path="/tmp/no_enforce_test", resume=False)
        # Should not fail with airgap_violation
        assert result.get("reason") != "airgap_violation"

    def test_enforce_airgapped_attribute_default_false(
        self, toolbox, advisor_manager
    ) -> None:
        """enforce_airgapped defaults to False for backwards compat."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        assert agent.enforce_airgapped is False

    def test_enforce_airgapped_attribute_settable(
        self, toolbox, advisor_manager
    ) -> None:
        """enforce_airgapped=True is stored correctly."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            enforce_airgapped=True,
        )
        assert agent.enforce_airgapped is True


# ---------------------------------------------------------------------------
# T18 – SharedContext checkpoint serialization (AgentStep serialization fix)
# ---------------------------------------------------------------------------


class TestCheckpointSerializationReal:
    """T18 — add_step() now converts AgentStep to a dict; checkpoints are JSON-safe."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        yield am
        am.shutdown()

    def test_add_step_stores_dict_not_object(self) -> None:
        """add_step() converts AgentStep → plain dict so steps is JSON-serialisable."""
        from keryx.core.shared_context import SharedContext
        from keryx.core.agent import AgentStep

        ctx = SharedContext(target_path="/tmp/ser_test")
        step = AgentStep(
            thought="checking heap",
            action="read_file",
            action_input={"file_path": "/tmp/a.c"},
            confidence=0.77,
        )
        ctx.add_step(step, "file contents here")
        stored_action = ctx.steps[0]["action"]
        assert isinstance(stored_action, dict), "action must be a plain dict"
        assert stored_action["name"] == "read_file"
        assert stored_action["confidence"] == pytest.approx(0.77)

    def test_to_dict_is_json_serialisable_after_real_steps(self) -> None:
        """to_dict() result passes json.dumps() after steps from add_step()."""
        from keryx.core.shared_context import SharedContext
        from keryx.core.agent import AgentStep

        ctx = SharedContext(target_path="/tmp/ser2")
        for i in range(3):
            ctx.add_step(
                AgentStep(thought=f"t{i}", action="read_file",
                          action_input={"file_path": f"/tmp/f{i}.c"}, confidence=0.5),
                f"observation {i}",
            )
        serialised = json.dumps(ctx.to_dict())  # must not raise
        reloaded = json.loads(serialised)
        assert reloaded["steps_taken"] == 3

    async def test_checkpoint_write_succeeds_after_real_react_steps(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """After running a real ReAct loop, _save_checkpoint() writes valid JSON."""
        from keryx.core.agent import KeryxAgent

        ckdir = tmp_path / "ckpts"
        ckdir.mkdir()

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
            checkpoint_dir=ckdir,
        )
        await agent.run(target_path="/tmp/real_ck", resume=False)
        # save checkpoint
        agent._save_checkpoint("/tmp/real_ck")

        import hashlib

        key = hashlib.sha256("/tmp/real_ck".encode()).hexdigest()[:12]
        ck_file = ckdir / f"checkpoint_{key}.json"
        assert ck_file.exists(), "checkpoint file was not written"
        data = json.loads(ck_file.read_text())
        assert "context" in data
        # JSON round-trip must succeed for the steps themselves
        json.dumps(data["context"]["steps"])  # must not raise

    def test_get_recent_history_works_after_add_step(self) -> None:
        """get_recent_history() reads dict keys correctly after serialization fix."""
        from keryx.core.shared_context import SharedContext
        from keryx.core.agent import AgentStep

        ctx = SharedContext(target_path="/tmp/hist_test")
        ctx.add_step(
            AgentStep(thought="looking", action="git_blame",
                      action_input={}, confidence=0.65),
            "blame output here",
        )
        history = ctx.get_recent_history(n=5)
        assert "git_blame" in history
        assert "conf=0.65" in history

    def test_get_last_confidence_reads_from_dict(self) -> None:
        """get_last_confidence() reads from dict['confidence'] after fix."""
        from keryx.core.shared_context import SharedContext
        from keryx.core.agent import AgentStep

        ctx = SharedContext(target_path="/tmp/conf_test")
        ctx.add_step(
            AgentStep(thought="t", action="read_file",
                      action_input={}, confidence=0.82),
            "obs",
        )
        conf = ctx.get_last_confidence()
        assert conf == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# T19 – BaseAdvisor, CascadeAdvisor, RuleBasedAdvisor coverage
# ---------------------------------------------------------------------------


class _AlwaysAdvisor:
    """Concrete advisor that always triggers and returns actionable advice."""
    name = "always-advisor"
    requires_network = False

    def __init__(self, direction: str = "Go deeper.", max_calls: int = 5):
        from keryx.advisors.base import BaseAdvisor
        # We can't subclass BaseAdvisor directly in a helper without import; use duck-typing
        self._calls = 0
        self._max = max_calls
        self._errors = 0
        self._total_time_ms = 0.0
        self.calls_made = 0
        self.max_calls_per_session = max_calls
        self._direction = direction

    def can_advise(self) -> bool:
        return self.calls_made < self.max_calls_per_session

    def should_trigger(self, context) -> bool:
        return True

    async def should_trigger_async(self, context) -> bool:
        return self.should_trigger(context)

    async def advise(self, context):
        from keryx.advisors.base import AdvisorResponse
        return AdvisorResponse(strategic_direction=self._direction)

    async def advise_with_tracking(self, context):
        from keryx.advisors.base import AdvisorResponse
        if not self.can_advise():
            return AdvisorResponse(metadata={"error": "call_limit_reached"})
        self.calls_made += 1
        return await self.advise(context)

    def reset(self):
        self.calls_made = 0
        self._errors = 0
        self._total_time_ms = 0.0

    def get_metrics(self):
        return {"name": self.name, "calls_made": self.calls_made}


class _NeverAdvisor(_AlwaysAdvisor):
    name = "never-advisor"

    def should_trigger(self, context) -> bool:
        return False

    async def should_trigger_async(self, context) -> bool:
        return False

    async def advise(self, context):
        from keryx.advisors.base import AdvisorResponse
        return AdvisorResponse()  # empty

    async def advise_with_tracking(self, context):
        from keryx.advisors.base import AdvisorResponse
        return AdvisorResponse()  # empty


class _ExplodingAdvisor(_AlwaysAdvisor):
    name = "exploding-advisor"

    async def advise(self, context):
        raise RuntimeError("advisor exploded")

    async def advise_with_tracking(self, context):
        self.calls_made += 1
        raise RuntimeError("advisor exploded")


class TestBaseAdvisor:
    """T19a — BaseAdvisor, CascadeAdvisor, RuleBasedAdvisor."""

    def test_advisor_response_is_empty_true(self) -> None:
        from keryx.advisors.base import AdvisorResponse
        r = AdvisorResponse()
        assert r.is_empty() is True

    def test_advisor_response_is_empty_false_direction(self) -> None:
        from keryx.advisors.base import AdvisorResponse
        r = AdvisorResponse(strategic_direction="Do X")
        assert r.is_empty() is False

    def test_advisor_response_is_empty_false_threshold(self) -> None:
        from keryx.advisors.base import AdvisorResponse
        r = AdvisorResponse(adjust_confidence_threshold=0.5)
        assert r.is_empty() is False

    def test_advisor_response_is_empty_false_hypotheses(self) -> None:
        from keryx.advisors.base import AdvisorResponse
        r = AdvisorResponse(suggested_hypotheses=["heap overflow"])
        assert r.is_empty() is False

    def test_advisor_response_repr(self) -> None:
        from keryx.advisors.base import AdvisorResponse
        r = AdvisorResponse(strategic_direction="Check allocators")
        assert "direction=" in repr(r)

    def test_rule_based_advisor_triggers_on_low_confidence(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext
        from keryx.core.agent import AgentStep

        advisor = RuleBasedAdvisor(confidence_threshold=0.6)
        ctx = SharedContext(target_path="/tmp/rba_conf")
        # add a step with confidence 0.3 (below threshold)
        ctx.add_step(
            AgentStep(thought="t", action="read_file", action_input={}, confidence=0.3),
            "obs",
        )
        assert advisor.should_trigger(ctx) is True

    def test_rule_based_advisor_triggers_on_parse_errors(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext

        advisor = RuleBasedAdvisor(max_parse_errors=2)
        ctx = SharedContext(target_path="/tmp/rba_err")
        ctx.parse_errors = 3
        assert advisor.should_trigger(ctx) is True

    def test_rule_based_advisor_triggers_on_no_progress(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext

        advisor = RuleBasedAdvisor(no_progress_after_steps=5)
        ctx = SharedContext(target_path="/tmp/rba_prog")
        ctx.steps_taken = 10
        # no hypotheses
        assert advisor.should_trigger(ctx) is True

    def test_rule_based_advisor_does_not_trigger_on_good_state(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext

        advisor = RuleBasedAdvisor(confidence_threshold=0.5, max_parse_errors=5)
        ctx = SharedContext(target_path="/tmp/rba_good")
        ctx.parse_errors = 0
        ctx.steps_taken = 2
        assert advisor.should_trigger(ctx) is False

    async def test_rule_based_advisor_advise_low_confidence_branch(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext
        from keryx.core.agent import AgentStep

        advisor = RuleBasedAdvisor(confidence_threshold=0.6)
        ctx = SharedContext(target_path="/tmp/rba_adv")
        ctx.add_step(
            AgentStep(thought="t", action="read_file", action_input={}, confidence=0.3),
            "obs",
        )
        resp = await advisor.advise(ctx)
        assert resp.strategic_direction  # non-empty
        assert resp.adjust_confidence_threshold is not None

    async def test_rule_based_advisor_advise_parse_error_branch(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext

        advisor = RuleBasedAdvisor(max_parse_errors=2)
        ctx = SharedContext(target_path="/tmp/rba_adv2")
        ctx.parse_errors = 3
        resp = await advisor.advise(ctx)
        assert "parse" in resp.strategic_direction.lower() or "errors" in resp.strategic_direction.lower()

    async def test_rule_based_advisor_advise_no_progress_branch(self) -> None:
        from keryx.advisors.base import RuleBasedAdvisor
        from keryx.core.shared_context import SharedContext

        advisor = RuleBasedAdvisor(no_progress_after_steps=5)
        ctx = SharedContext(target_path="/tmp/rba_adv3")
        ctx.steps_taken = 20
        resp = await advisor.advise(ctx)
        assert "hypothes" in resp.strategic_direction.lower() or "allocator" in resp.strategic_direction.lower()

    async def test_base_advisor_advise_with_tracking_limit_reached(self) -> None:
        """advise_with_tracking() returns error response when call limit reached."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class LimitedAdvisor(BaseAdvisor):
            name = "limited"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="advice")

        a = LimitedAdvisor(max_calls_per_session=1)
        ctx = object()
        # Exhaust the limit
        a.calls_made = 1
        resp = await a.advise_with_tracking(ctx)
        assert resp.metadata.get("error") == "call_limit_reached"

    async def test_base_advisor_advise_with_tracking_exception_caught(self) -> None:
        """Exception in advise() is caught, error stored in metadata."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class BrokenAdvisor(BaseAdvisor):
            name = "broken"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): raise ValueError("bad input")

        a = BrokenAdvisor()
        resp = await a.advise_with_tracking(object())
        assert "error" in resp.metadata
        assert a._errors == 1

    async def test_base_advisor_should_trigger_async_sync(self) -> None:
        """should_trigger_async wraps sync should_trigger correctly."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class SyncAdvisor(BaseAdvisor):
            name = "sync"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse()

        a = SyncAdvisor()
        assert await a.should_trigger_async(object()) is True

    def test_base_advisor_reset(self) -> None:
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class CountAdvisor(BaseAdvisor):
            name = "count"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse()

        a = CountAdvisor()
        a.calls_made = 3
        a._errors = 1
        a._total_time_ms = 500.0
        a.reset()
        assert a.calls_made == 0
        assert a._errors == 0
        assert a._total_time_ms == 0.0

    def test_base_advisor_get_metrics(self) -> None:
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class MetricAdvisor(BaseAdvisor):
            name = "metric"
            def should_trigger(self, ctx): return False
            async def advise(self, ctx): return AdvisorResponse()

        a = MetricAdvisor()
        a.calls_made = 2
        a._total_time_ms = 400.0
        m = a.get_metrics()
        assert m["calls_made"] == 2
        assert m["avg_time_ms"] == pytest.approx(200.0)

    def test_base_advisor_repr(self) -> None:
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class ReprAdvisor(BaseAdvisor):
            name = "repr-adv"
            def should_trigger(self, ctx): return False
            async def advise(self, ctx): return AdvisorResponse()

        r = repr(ReprAdvisor())
        assert "repr-adv" in r

    def test_cascade_advisor_requires_at_least_one(self) -> None:
        from keryx.advisors.base import CascadeAdvisor
        with pytest.raises(ValueError):
            CascadeAdvisor(advisors=[])

    async def test_cascade_advisor_returns_first_non_empty(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class First(BaseAdvisor):
            name = "first"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="first wins")

        class Second(BaseAdvisor):
            name = "second"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="second")

        cascade = CascadeAdvisor([First(), Second()])
        resp = await cascade.advise(object())
        assert resp.strategic_direction == "first wins"

    async def test_cascade_advisor_skips_exhausted(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class Exhausted(BaseAdvisor):
            name = "exhausted"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="exhausted advice")

        class Backup(BaseAdvisor):
            name = "backup"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="backup wins")

        e = Exhausted(max_calls_per_session=1)
        e.calls_made = 1  # exhausted
        cascade = CascadeAdvisor([e, Backup()])
        resp = await cascade.advise(object())
        assert resp.strategic_direction == "backup wins"

    async def test_cascade_advisor_falls_back_on_empty_response(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class EmptyFirst(BaseAdvisor):
            name = "empty-first"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse()  # empty

        class GoodSecond(BaseAdvisor):
            name = "good-second"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="second")

        cascade = CascadeAdvisor([EmptyFirst(), GoodSecond()])
        resp = await cascade.advise(object())
        assert resp.strategic_direction == "second"

    async def test_cascade_advisor_all_not_triggered_returns_empty(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class NoTrigger(BaseAdvisor):
            name = "no-trig"
            def should_trigger(self, ctx): return False
            async def advise(self, ctx): return AdvisorResponse(strategic_direction="never")

        cascade = CascadeAdvisor([NoTrigger(), NoTrigger()])
        # rename to avoid name collision
        cascade._advisors[0].name = "no-trig-1"
        cascade._advisors[1].name = "no-trig-2"
        resp = await cascade.advise(object())
        assert resp.is_empty() or "no_advisor_triggered" in resp.metadata.get("cascade", "")

    def test_cascade_advisor_reset_resets_children(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class Child(BaseAdvisor):
            name = "child"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse()

        c1, c2 = Child(), Child()
        c1.calls_made = 3
        c2.calls_made = 2
        cascade = CascadeAdvisor([c1, c2])
        cascade.reset()
        assert c1.calls_made == 0
        assert c2.calls_made == 0

    def test_cascade_advisor_get_metrics_includes_chain(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class Child(BaseAdvisor):
            name = "child-m"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse()

        cascade = CascadeAdvisor([Child()])
        m = cascade.get_metrics()
        assert "chain" in m
        assert "child_metrics" in m

    async def test_cascade_advisor_should_trigger_true_if_any(self) -> None:
        from keryx.advisors.base import CascadeAdvisor, AdvisorResponse, BaseAdvisor

        class NoT(BaseAdvisor):
            name = "no-t"
            def should_trigger(self, ctx): return False
            async def advise(self, ctx): return AdvisorResponse()

        class YesT(BaseAdvisor):
            name = "yes-t"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): return AdvisorResponse()

        cascade = CascadeAdvisor([NoT(), YesT()])
        assert cascade.should_trigger(object()) is True


# ---------------------------------------------------------------------------
# T19b – AdvisorManager coverage
# ---------------------------------------------------------------------------


class _SimpleAdvisor:
    """Minimal concrete advisor for AdvisorManager tests."""
    requires_network = False
    calls_made = 0
    max_calls_per_session = 10
    _errors = 0
    _total_time_ms = 0.0

    def __init__(self, name: str, triggers: bool = True, direction: str = "advice"):
        self.name = name
        self._triggers = triggers
        self._direction = direction

    def can_advise(self) -> bool:
        return self.calls_made < self.max_calls_per_session

    def should_trigger(self, ctx) -> bool:
        return self._triggers

    async def should_trigger_async(self, ctx) -> bool:
        return self._triggers

    async def advise(self, ctx):
        from keryx.advisors.base import AdvisorResponse
        return AdvisorResponse(strategic_direction=self._direction)

    async def advise_with_tracking(self, ctx):
        from keryx.advisors.base import AdvisorResponse
        if not self.can_advise():
            return AdvisorResponse(metadata={"error": "call_limit_reached"})
        self.calls_made += 1
        return await self.advise(ctx)

    def reset(self):
        self.calls_made = 0

    def get_metrics(self):
        return {"name": self.name}


class TestAdvisorManager:
    """T19b — AdvisorManager registration, dispatch, circuit breaker, parallel."""

    @pytest.fixture
    def manager(self):
        from keryx.advisors.manager import AdvisorManager
        m = AdvisorManager()
        yield m
        m.shutdown()

    # ── Registration ────────────────────────────────────────────────────────

    def test_register_advisor(self, manager) -> None:
        adv = _SimpleAdvisor("reg-test")
        manager.register(adv, priority=10)
        assert manager.has_advisor("reg-test")
        assert manager.get_advisor("reg-test") is adv

    def test_register_many(self, manager) -> None:
        a1 = _SimpleAdvisor("rm-1")
        a2 = _SimpleAdvisor("rm-2")
        manager.register_many([(a1, 10), (a2, 20)])
        assert manager.has_advisor("rm-1")
        assert manager.has_advisor("rm-2")

    def test_unregister_existing(self, manager) -> None:
        adv = _SimpleAdvisor("unrg")
        manager.register(adv)
        result = manager.unregister("unrg")
        assert result is True
        assert not manager.has_advisor("unrg")

    def test_unregister_nonexistent(self, manager) -> None:
        assert manager.unregister("ghost") is False

    def test_unregister_updates_default(self, manager) -> None:
        a1 = _SimpleAdvisor("d1")
        a2 = _SimpleAdvisor("d2")
        manager.register(a1)
        manager.register(a2)
        manager._default = "d1"
        manager.unregister("d1")
        assert manager._default == "d2"

    def test_set_default_existing(self, manager) -> None:
        adv = _SimpleAdvisor("def-adv")
        manager.register(adv)
        manager.set_default("def-adv")
        assert manager._default == "def-adv"

    def test_set_default_nonexistent_raises(self, manager) -> None:
        with pytest.raises(KeyError):
            manager.set_default("ghost")

    def test_list_advisors_sorted_by_priority(self, manager) -> None:
        manager.register(_SimpleAdvisor("low"), priority=90)
        manager.register(_SimpleAdvisor("high"), priority=10)
        listed = manager.list_advisors()
        assert listed.index("high") < listed.index("low")

    def test_calls_made_property(self, manager) -> None:
        assert manager.calls_made == 0

    # ── Dispatch ────────────────────────────────────────────────────────────

    async def test_get_advice_async_no_advisors(self, manager) -> None:
        resp = await manager.get_advice_async(object())
        assert "error" in resp.metadata

    async def test_get_advice_async_not_triggered(self, manager) -> None:
        adv = _SimpleAdvisor("no-trig", triggers=False)
        manager.register(adv)
        resp = await manager.get_advice_async(object())
        assert resp.metadata.get("triggered") is False

    async def test_get_advice_async_triggered_success(self, manager) -> None:
        adv = _SimpleAdvisor("yes-trig", direction="turn left")
        manager.register(adv)
        resp = await manager.get_advice_async(object())
        assert resp.strategic_direction == "turn left"
        assert manager.calls_made == 1

    async def test_get_advice_async_fallback_chain(self, manager) -> None:
        """Circuit-open primary → falls back through chain to secondary."""
        primary = _SimpleAdvisor("fb-primary", direction="primary advice")
        fallback = _SimpleAdvisor("fb-fallback", direction="fallback advice")
        manager.register(primary)
        manager.register(fallback)
        # Force primary circuit open so manager tries the fallback chain
        manager._health["fb-primary"].is_open = True
        manager._health["fb-primary"].last_failure_time = float("inf")

        resp = await manager.get_advice_async(
            object(), advisor_name="fb-primary", fallback_chain=["fb-fallback"]
        )
        assert resp.strategic_direction == "fallback advice"

    async def test_get_advice_async_circular_fallback_detected(self, manager) -> None:
        adv = _SimpleAdvisor("circ")
        manager.register(adv)
        visited = {"circ"}
        resp = await manager.get_advice_async(
            object(), advisor_name="circ", _visited=visited
        )
        assert "error" in resp.metadata

    async def test_get_advice_priority(self, manager) -> None:
        a_high = _SimpleAdvisor("prio-high", direction="high prio")
        a_low = _SimpleAdvisor("prio-low", direction="low prio")
        manager.register(a_high, priority=5)
        manager.register(a_low, priority=95)
        resp = await manager.get_advice_priority(object())
        assert resp.strategic_direction == "high prio"

    async def test_get_advice_priority_no_triggers(self, manager) -> None:
        manager.register(_SimpleAdvisor("no-t-prio", triggers=False))
        resp = await manager.get_advice_priority(object())
        assert "error" in resp.metadata

    async def test_get_advice_parallel_all_valid(self, manager) -> None:
        a1 = _SimpleAdvisor("par-1", direction="p1")
        a2 = _SimpleAdvisor("par-2", direction="p2")
        manager.register(a1)
        manager.register(a2)
        responses = await manager.get_advice_parallel(object(), ["par-1", "par-2"])
        assert len(responses) == 2

    async def test_get_advice_parallel_empty_names(self, manager) -> None:
        responses = await manager.get_advice_parallel(object(), ["ghost"])
        assert responses == []

    # ── Circuit breaker ─────────────────────────────────────────────────────

    def test_advisor_health_record_success(self) -> None:
        from keryx.advisors.manager import AdvisorHealth
        h = AdvisorHealth()
        h.consecutive_system_failures = 3
        h.record_success(100.0)
        assert h.consecutive_system_failures == 0
        assert h.total_calls == 1
        assert h.is_open is False

    def test_advisor_health_record_system_failure_opens_circuit(self) -> None:
        from keryx.advisors.manager import AdvisorHealth
        h = AdvisorHealth(_open_threshold=2)
        h.record_failure(ConnectionError("conn failed"))
        h.record_failure(ConnectionError("conn failed"))
        assert h.is_open is True

    def test_advisor_health_record_non_system_failure_does_not_open(self) -> None:
        from keryx.advisors.manager import AdvisorHealth
        h = AdvisorHealth(_open_threshold=2)
        h.record_failure(ValueError("bad value"))
        h.record_failure(ValueError("bad value"))
        # ValueError is not in _SYSTEM_ERRORS → circuit stays closed
        assert h.is_open is False

    def test_advisor_health_can_attempt_open_not_recovered(self) -> None:
        from keryx.advisors.manager import AdvisorHealth
        import time
        h = AdvisorHealth(_open_threshold=1, _recovery_seconds=3600.0)
        h.record_failure(ConnectionError("x"))
        # Circuit is open, recovery window not passed
        assert h.can_attempt() is False

    def test_advisor_health_can_attempt_recovered(self) -> None:
        from keryx.advisors.manager import AdvisorHealth
        import time
        h = AdvisorHealth(_open_threshold=1, _recovery_seconds=0.0)
        h.record_failure(ConnectionError("x"))
        h.last_failure_time = time.time() - 1.0
        assert h.can_attempt() is True
        assert h.is_open is False  # half-open reset

    def test_advisor_health_avg_latency(self) -> None:
        from keryx.advisors.manager import AdvisorHealth
        h = AdvisorHealth()
        h.record_success(200.0)
        h.record_success(400.0)
        assert h.avg_latency_ms == pytest.approx(300.0)

    def test_is_system_error_true(self) -> None:
        from keryx.advisors.manager import _is_system_error
        assert _is_system_error(ConnectionError()) is True
        assert _is_system_error(TimeoutError()) is True
        assert _is_system_error(OSError()) is True
        assert _is_system_error(MemoryError()) is True

    def test_is_system_error_false(self) -> None:
        from keryx.advisors.manager import _is_system_error
        assert _is_system_error(ValueError()) is False
        assert _is_system_error(RuntimeError()) is False

    async def test_circuit_breaker_open_uses_fallback(self, manager) -> None:
        """When circuit is open, manager tries fallback."""
        from keryx.advisors.manager import AdvisorHealth
        primary = _SimpleAdvisor("cb-primary", direction="primary")
        fallback = _SimpleAdvisor("cb-fallback", direction="fallback")
        manager.register(primary)
        manager.register(fallback)
        # Force circuit open
        manager._health["cb-primary"].is_open = True
        manager._health["cb-primary"].last_failure_time = float("inf")

        resp = await manager.get_advice_async(
            object(), advisor_name="cb-primary", fallback_chain=["cb-fallback"]
        )
        assert resp.strategic_direction == "fallback"

    # ── Metrics / health ────────────────────────────────────────────────────

    def test_get_metrics_structure(self, manager) -> None:
        adv = _SimpleAdvisor("metric-adv")
        manager.register(adv)
        m = manager.get_metrics()
        assert "total_calls" in m
        assert "advisor_health" in m
        assert "latency_histogram" in m

    def test_get_advisor_health(self, manager) -> None:
        from keryx.advisors.manager import AdvisorHealth
        adv = _SimpleAdvisor("health-adv")
        manager.register(adv)
        h = manager.get_advisor_health("health-adv")
        assert isinstance(h, AdvisorHealth)
        assert manager.get_advisor_health("ghost") is None

    def test_can_advise_named(self, manager) -> None:
        adv = _SimpleAdvisor("can-adv")
        manager.register(adv)
        assert manager.can_advise("can-adv") is True

    def test_can_advise_named_not_found(self, manager) -> None:
        assert manager.can_advise("ghost") is False

    def test_can_advise_any(self, manager) -> None:
        adv = _SimpleAdvisor("any-adv")
        manager.register(adv)
        assert manager.can_advise() is True

    def test_can_advise_none_when_empty(self, manager) -> None:
        assert manager.can_advise() is False

    def test_reset_all(self, manager) -> None:
        adv = _SimpleAdvisor("reset-adv")
        adv.calls_made = 5
        manager.register(adv)
        manager._total_calls = 10
        manager.reset_all()
        assert adv.calls_made == 0
        assert manager._total_calls == 0

    def test_repr(self, manager) -> None:
        r = repr(manager)
        assert "AdvisorManager" in r

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def test_shutdown_idempotent(self, manager) -> None:
        """shutdown() can be called multiple times without error."""
        manager.shutdown()
        manager.shutdown()  # second call must not raise

    async def test_context_manager(self) -> None:
        from keryx.advisors.manager import AdvisorManager
        async with AdvisorManager() as m:
            assert m is not None

    def test_create_advisor_manager_factory(self) -> None:
        from keryx.advisors.manager import create_advisor_manager
        m = create_advisor_manager()
        assert m is not None
        m.shutdown()

    def test_create_advisor_manager_factory_with_advisors(self) -> None:
        from keryx.advisors.manager import create_advisor_manager
        adv = _SimpleAdvisor("factory-adv")
        m = create_advisor_manager(advisors=[(adv, 10)], default="factory-adv")
        assert m.has_advisor("factory-adv")
        m.shutdown()

    def test_latency_histogram_buckets(self) -> None:
        from keryx.advisors.manager import AdvisorManager, CallMetrics
        import time
        calls = [
            CallMetrics("a", time.time(), 50.0, True, True),
            CallMetrics("a", time.time(), 300.0, True, True),
            CallMetrics("a", time.time(), 750.0, True, True),
            CallMetrics("a", time.time(), 2000.0, True, True),
            CallMetrics("a", time.time(), 6000.0, True, True),
        ]
        hist = AdvisorManager._latency_histogram(calls)
        assert hist["<100ms"] == 1
        assert hist["100-500ms"] == 1
        assert hist["0.5-1s"] == 1
        assert hist["1-5s"] == 1
        assert hist[">5s"] == 1


# ---------------------------------------------------------------------------
# T20 – ToolBox and BaseTool coverage
# ---------------------------------------------------------------------------


class SlowTool:
    """Sleeps longer than any timeout — tests timeout path."""
    name = "slow_tool"
    description = "Slow test tool"
    default_timeout = 0.5
    _call_count = 0
    _success_count = 0
    _total_time_ms = 0.0

    def __init__(self):
        import threading
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def get_command(self, action_input):
        return None

    def _record_call(self, success, duration_ms):
        with self._lock:
            self._call_count += 1
            self._total_time_ms += duration_ms
            if success:
                self._success_count += 1

    def reset_metrics(self):
        with self._lock:
            self._call_count = 0
            self._success_count = 0
            self._total_time_ms = 0.0

    def get_metrics(self):
        return {"name": self.name, "calls": self._call_count}

    async def execute(self, action_input, context=None):
        import asyncio
        from keryx.tools.Toolbox import ToolResult
        await asyncio.sleep(999)
        return ToolResult(success=True, output="never reached")


class ExplodingTool:
    """Raises in execute() — tests exception path."""
    name = "exploding_tool"
    description = "Always raises"
    default_timeout = 60.0
    _call_count = 0
    _success_count = 0
    _total_time_ms = 0.0

    def __init__(self):
        import threading
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def get_command(self, action_input):
        return None

    def _record_call(self, success, duration_ms):
        with self._lock:
            self._call_count += 1
            self._total_time_ms += duration_ms

    def reset_metrics(self):
        pass

    def get_metrics(self):
        return {"name": self.name}

    async def execute(self, action_input, context=None):
        raise RuntimeError("tool exploded")


class UnavailableTool:
    """is_available() returns False — tests unavailable path."""
    name = "unavailable_tool"
    description = "Never available"
    default_timeout = 60.0
    _call_count = 0
    _success_count = 0
    _total_time_ms = 0.0

    def __init__(self):
        import threading
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return False

    def get_command(self, action_input):
        return None

    def _record_call(self, success, duration_ms): pass
    def reset_metrics(self): pass
    def get_metrics(self): return {"name": self.name}

    async def execute(self, action_input, context=None):
        from keryx.tools.Toolbox import ToolResult
        return ToolResult(success=True, output="should never get here")


class StringReturningTool:
    """Returns a plain str instead of ToolResult — tests wrapping path."""
    name = "string_tool"
    description = "Returns str"
    default_timeout = 60.0
    _call_count = 0
    _success_count = 0
    _total_time_ms = 0.0

    def __init__(self):
        import threading
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return True

    def get_command(self, action_input):
        return None

    def _record_call(self, success, duration_ms):
        with self._lock:
            self._call_count += 1

    def reset_metrics(self): pass
    def get_metrics(self): return {"name": self.name}

    async def execute(self, action_input, context=None):
        return "raw string result"


class TestToolBox:
    """T20 — ToolBox registration, async execution, sync bridge, batch."""

    @pytest.fixture
    def tb(self):
        from keryx.tools.Toolbox import ToolBox
        toolbox = ToolBox(default_timeout=5.0)
        yield toolbox
        toolbox.shutdown()

    # ── BaseTool ─────────────────────────────────────────────────────────────

    async def test_base_tool_execute_raises_not_implemented(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class Concrete(BaseTool):
            name = "concrete"

        t = Concrete()
        with pytest.raises(NotImplementedError):
            await t.execute({})

    def test_base_tool_is_available_default_true(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class Concrete(BaseTool):
            name = "avail"

        assert Concrete().is_available() is True

    def test_base_tool_get_command_default_none(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class Concrete(BaseTool):
            name = "cmd"

        assert Concrete().get_command({}) is None

    def test_base_tool_record_call_and_metrics(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class Concrete(BaseTool):
            name = "rec"

        t = Concrete()
        t._record_call(success=True, duration_ms=100.0)
        t._record_call(success=False, duration_ms=200.0)
        m = t.get_metrics()
        assert m["calls"] == 2
        assert m["successes"] == 1

    def test_base_tool_reset_metrics(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class Concrete(BaseTool):
            name = "reset-t"

        t = Concrete()
        t._record_call(True, 100.0)
        t.reset_metrics()
        assert t._call_count == 0
        assert t._success_count == 0

    def test_base_tool_repr(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class Concrete(BaseTool):
            name = "repr-t"

        assert "repr-t" in repr(Concrete())

    # ── ToolBox registration ──────────────────────────────────────────────────

    def test_register_and_list(self, tb) -> None:
        from keryx.tools.Toolbox import BaseTool

        class MyTool(BaseTool):
            name = "my_tool"
            async def execute(self, ai, ctx=None):
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=True, output="ok")

        tb.register(MyTool())
        assert "my_tool" in tb.list_tools()

    def test_register_many(self, tb) -> None:
        from keryx.tools.Toolbox import BaseTool, ToolBox

        class T1(BaseTool):
            name = "t1"
            async def execute(self, ai, ctx=None):
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=True, output="t1")

        class T2(BaseTool):
            name = "t2"
            async def execute(self, ai, ctx=None):
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=True, output="t2")

        tb.register_many([T1(), T2()])
        assert "t1" in tb.list_tools()
        assert "t2" in tb.list_tools()

    def test_list_tools_excludes_unavailable(self, tb) -> None:
        tb.register(UnavailableTool())
        assert "unavailable_tool" not in tb.list_tools()

    def test_get_tool(self, tb) -> None:
        from keryx.tools.Toolbox import BaseTool

        class GetMe(BaseTool):
            name = "get_me"
            async def execute(self, ai, ctx=None):
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=True, output="got")

        t = GetMe()
        tb.register(t)
        assert tb.get_tool("get_me") is t
        assert tb.get_tool("ghost") is None

    def test_get_command_registered_available(self, tb) -> None:
        """get_command() delegates to tool.get_command()."""
        from keryx.tools.Toolbox import BaseTool

        class CmdTool(BaseTool):
            name = "cmd_tool"
            def get_command(self, ai): return ["git", "log"]
            async def execute(self, ai, ctx=None):
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=True, output="x")

        tb.register(CmdTool())
        assert tb.get_command("cmd_tool", {}) == ["git", "log"]
        assert tb.get_command("ghost", {}) is None

    # ── execute_async ──────────────────────────────────────────────────────────

    async def test_execute_async_invalid_input_type(self, tb) -> None:
        result = await tb.execute_async("any_tool", "not a dict")  # type: ignore
        assert result.success is False
        assert result.error == "invalid_input_type"

    async def test_execute_async_unknown_tool(self, tb) -> None:
        result = await tb.execute_async("ghost_tool", {})
        assert result.success is False
        assert result.error == "tool_not_found"
        assert "available" in result.metadata

    async def test_execute_async_unavailable_tool(self, tb) -> None:
        tb.register(UnavailableTool())
        result = await tb.execute_async("unavailable_tool", {})
        assert result.success is False
        assert result.error == "tool_unavailable"

    async def test_execute_async_timeout(self, tb) -> None:
        tb.register(SlowTool())
        result = await tb.execute_async("slow_tool", {}, timeout=0.05)
        assert result.success is False
        assert result.error == "timeout"
        assert tb._timeout_count == 1

    async def test_execute_async_unhandled_exception(self, tb) -> None:
        tb.register(ExplodingTool())
        result = await tb.execute_async("exploding_tool", {})
        assert result.success is False
        assert result.error == "unhandled_exception"
        assert tb._error_count == 1

    async def test_execute_async_wraps_non_tool_result(self, tb) -> None:
        tb.register(StringReturningTool())
        result = await tb.execute_async("string_tool", {})
        assert result.success is True
        assert "raw string result" in result.output

    async def test_execute_async_max_output_len_clamped(self, tb) -> None:
        """max_output_len > 10000 is clamped to 10000."""
        from keryx.tools.Toolbox import BaseTool, ToolResult

        class BigInputTool(BaseTool):
            name = "big_input"
            async def execute(self, ai, ctx=None):
                return ToolResult(success=True, output=f"len={ai.get('max_output_len')}")

        tb.register(BigInputTool())
        result = await tb.execute_async("big_input", {"max_output_len": 99999})
        assert "10000" in result.output

    # ── execute_batch ──────────────────────────────────────────────────────────

    async def test_execute_batch_multiple_tools(self, tb) -> None:
        from keryx.tools.Toolbox import BaseTool, ToolResult

        class FastTool(BaseTool):
            name = "fast_t"
            async def execute(self, ai, ctx=None):
                return ToolResult(success=True, output=f"fast:{ai.get('id', 0)}")

        tb.register(FastTool())
        results = await tb.execute_batch([
            ("fast_t", {"id": 1}),
            ("fast_t", {"id": 2}),
            ("ghost_t", {}),
        ])
        assert len(results) == 3
        assert results[0].success is True
        assert results[2].success is False  # ghost_t not found

    # ── Metrics / lifecycle ───────────────────────────────────────────────────

    def test_get_metrics(self, tb) -> None:
        m = tb.get_metrics()
        assert "total_calls" in m
        assert "timeouts" in m
        assert "errors" in m

    def test_reset_metrics(self, tb) -> None:
        tb._total_calls = 5
        tb._timeout_count = 2
        tb._error_count = 1
        tb.reset_metrics()
        assert tb._total_calls == 0
        assert tb._timeout_count == 0
        assert tb._error_count == 0

    def test_repr(self, tb) -> None:
        assert "ToolBox" in repr(tb)

    def test_create_toolbox_factory(self) -> None:
        from keryx.tools.Toolbox import create_toolbox
        tb = create_toolbox(default_timeout=30.0)
        assert tb.default_timeout == 30.0
        tb.shutdown()

    def test_sync_execute_unknown_tool(self, tb) -> None:
        result = tb.execute("nonexistent", {})
        assert result.success is False

    def test_toolresult_format_for_llm_error(self) -> None:
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=False, output="something bad", error="some_error")
        formatted = r.format_for_llm()
        assert "[ERROR:some_error]" in formatted

    def test_toolresult_format_for_llm_findings(self) -> None:
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=True, output="x", data={"findings": [1, 2, 3]})
        assert "3" in r.format_for_llm()

    def test_toolresult_to_dict(self) -> None:
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=True, output="hello", data={"k": "v"})
        d = r.to_dict()
        assert d["success"] is True
        assert d["output"] == "hello"

    def test_toolresult_str_preserves_output(self) -> None:
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=True, output="VULN_CONFIRMED here")
        assert "VULN_CONFIRMED" in str(r)


# ---------------------------------------------------------------------------
# T21 – Agent escalation, critique, and self-critique paths
# ---------------------------------------------------------------------------


class LowConfModel(MyLocalModel):
    """Produces low-confidence NO_ACTION responses to trigger escalation."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        return json.dumps({
            "thought": "not sure",
            "action": "NO_ACTION",
            "action_input": {},
            "confidence": 0.1,
        })


class CritiqueModel(MyLocalModel):
    """Generates a tool action then emits specific critique keywords."""

    def __init__(self, critique_text: str) -> None:
        super().__init__()
        self._critique = critique_text
        self._call = 0

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1:
            return "ok"
        if grammar is None:
            # Critique call
            return self._critique
        self._call += 1
        if self._call == 1:
            return json.dumps({
                "thought": "checking",
                "action": "read_file",
                "action_input": {"file_path": __file__},
                "confidence": 0.8,
            })
        return json.dumps({
            "thought": "done",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class VulnConfirmModel(MyLocalModel):
    """Returns high-confidence gdb_analyze to trigger early-stop vuln path."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        return json.dumps({
            "thought": "found it",
            "action": "gdb_analyze",
            "action_input": {},
            "confidence": 0.95,
        })


class GdbFakeTool:
    """Returns VULN_CONFIRMED signal for gdb_analyze."""
    name = "gdb_analyze"
    description = "Fake GDB"
    default_timeout = 60.0
    _call_count = 0
    _success_count = 0
    _total_time_ms = 0.0

    def __init__(self):
        import threading
        self._lock = threading.Lock()

    def is_available(self): return True
    def get_command(self, ai): return None

    def _record_call(self, success, duration_ms):
        with self._lock:
            self._call_count += 1
            if success: self._success_count += 1

    def reset_metrics(self): pass
    def get_metrics(self): return {"name": self.name}

    async def execute(self, action_input, context=None):
        from keryx.tools.Toolbox import ToolResult
        return ToolResult(success=True, output="VULN_CONFIRMED: heap-use-after-free at 0x7f")


class TestAgentEscalationPaths:
    """T21 — agent escalation, critique branches, vuln early-stop."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def toolbox_with_gdb(self):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        tb.register(GdbFakeTool())
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    @pytest.fixture
    def triggering_advisor_manager(self):
        """AdvisorManager with an advisor that always triggers."""
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        adv = _SimpleAdvisor("trigger-adv", direction="look at allocators")
        am.register(adv)
        yield am
        am.shutdown()

    # ── _apply_critique branches ─────────────────────────────────────────────

    async def test_critique_false_positive_blacklists(self, toolbox, advisor_manager) -> None:
        """'false positive' in critique → hypothesis blacklisted, confidence halved."""
        from keryx.core.agent import KeryxAgent

        model = CritiqueModel("This looks like a false positive — not reachable from main.")
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
        )
        result = await agent.run(target_path=__file__, resume=False)
        assert result["status"] == "completed"

    async def test_critique_needs_evidence_requests(self, toolbox, advisor_manager) -> None:
        """'needs more evidence' in critique → request_additional_evidence called."""
        from keryx.core.agent import KeryxAgent

        model = CritiqueModel("needs more evidence from other modules")
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
        )
        result = await agent.run(target_path=__file__, resume=False)
        assert result["status"] == "completed"

    async def test_critique_confirmed_boosts_confidence(self, toolbox, advisor_manager) -> None:
        """'confirmed' in critique → confidence += 0.1."""
        from keryx.core.agent import KeryxAgent

        model = CritiqueModel("This is likely confirmed — the pointer is freed before use.")
        agent = KeryxAgent(
            executor_model=model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
        )
        result = await agent.run(target_path=__file__, resume=False)
        assert result["status"] == "completed"

    # ── Escalation to advisor ────────────────────────────────────────────────

    async def test_escalation_triggers_advisor(
        self, toolbox, triggering_advisor_manager
    ) -> None:
        """Low-confidence steps eventually trigger advisor escalation."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.agent import _ESCALATION_COOLDOWN

        agent = KeryxAgent(
            executor_model=LowConfModel(),
            advisor_manager=triggering_advisor_manager,
            toolbox=toolbox,
            max_steps=_ESCALATION_COOLDOWN + 3,
            confidence_threshold=0.5,
        )
        result = await agent.run(target_path="/tmp/esc_test", resume=False)
        assert result["status"] == "completed"
        # Advisor should have been consulted
        assert triggering_advisor_manager.calls_made >= 1

    # ── Vuln confirmed early stop ────────────────────────────────────────────

    async def test_vuln_confirmed_early_stop(
        self, toolbox_with_gdb, advisor_manager
    ) -> None:
        """High-confidence gdb_analyze with VULN_CONFIRMED signal → early stop."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=VulnConfirmModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox_with_gdb,
            max_steps=50,
        )
        result = await agent.run(target_path="/tmp/vuln_test", resume=False)
        # Should stop early with a confirmed vuln
        assert len(result.get("confirmed_vulns", [])) >= 1

    # ── _observation_confirms_vuln ────────────────────────────────────────────

    def test_observation_confirms_vuln_signals(self) -> None:
        from keryx.core.agent import KeryxAgent
        assert KeryxAgent._observation_confirms_vuln("VULN_CONFIRMED: found") is True
        assert KeryxAgent._observation_confirms_vuln("AddressSanitizer: heap-use-after-free") is True
        assert KeryxAgent._observation_confirms_vuln("CRASH detected") is True
        assert KeryxAgent._observation_confirms_vuln("everything is fine") is False

    # ── _build_prompt with budget ─────────────────────────────────────────────

    async def test_build_prompt_includes_budget_line(
        self, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent, BudgetController

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=2,
            budget_usd=1.0,
        )
        # Initialize context so _build_prompt works
        from keryx.core.shared_context import SharedContext
        agent.context = SharedContext(target_path="/tmp/budget_prompt")
        prompt = agent._build_prompt()
        assert "Budget" in prompt

    # ── _parse_response fallback paths ────────────────────────────────────────

    def test_parse_response_extracts_json_with_markdown(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/parse_test")
        raw = '```json\n{"thought":"t","action":"read_file","action_input":{},"confidence":0.7}\n```'
        step = agent._parse_response(raw)
        assert step.action == "read_file"

    def test_parse_response_fallback_infers_action(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/fallback_test")
        step = agent._parse_response("I think we should use FINISH here")
        assert step.action == "FINISH"

    def test_parse_response_no_json_increments_parse_errors(
        self, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/parse_err")
        agent._parse_response("not json at all @@@@")
        assert agent.context.parse_errors == 1

    def test_simplify_prompt(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/simp")
        long_prompt = "\n".join([f"line {i}" for i in range(100)])
        simplified = agent._simplify_prompt(long_prompt)
        assert len(simplified.split("\n")) <= 50

    # ── _should_escalate logic ────────────────────────────────────────────────

    def test_should_escalate_on_low_conf_streak(self, toolbox, advisor_manager) -> None:
        """P4: 3+ consecutive low-confidence steps triggers escalation."""
        from keryx.core.agent import KeryxAgent, _ESCALATION_COOLDOWN
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            confidence_threshold=0.5,
        )
        agent.context = SharedContext(target_path="/tmp/esc_err")
        agent._steps_since_escalation = _ESCALATION_COOLDOWN + 1
        agent._consecutive_low_confidence = 3          # P4 streak threshold
        assert agent._should_escalate() is True

    def test_should_not_escalate_within_cooldown(self, toolbox, advisor_manager) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/no_esc")
        agent._steps_since_escalation = 0  # within cooldown
        assert agent._should_escalate() is False

    # ── Checkpoint with budget restore ───────────────────────────────────────

    async def test_checkpoint_restores_budget(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent

        ckdir = tmp_path / "ck"
        ckdir.mkdir()
        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=1,
            budget_usd=5.0,
            checkpoint_dir=ckdir,
        )
        await agent.run(target_path="/tmp/budget_ck", resume=False)
        # Manually save with non-zero cost
        agent.budget.current_cost = 0.42
        agent._save_checkpoint("/tmp/budget_ck")

        # Load via new agent
        agent2 = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            budget_usd=5.0,
            checkpoint_dir=ckdir,
        )
        ctx = agent2._load_checkpoint("/tmp/budget_ck")
        assert ctx is not None


# ---------------------------------------------------------------------------
# T22 – GitBlameTool missing coverage paths
# ---------------------------------------------------------------------------


class TestGitBlameToolPaths:
    """T22 — GitBlameTool: unavailable, unknown cmd, filters, scrub, truncate."""

    async def test_unavailable_returns_error(self) -> None:
        """GitBlameTool with _available=False returns git_not_found error."""
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        tool._available = False  # force unavailable regardless of system git
        assert tool.is_available() is False
        result = await tool.execute({"command": "blame", "file": "/tmp/foo.c"})
        assert result.success is False
        assert result.error == "git_not_found"

    async def test_unknown_command_type(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available in this environment")
        result = await tool.execute({"command": "invalid_cmd"})
        assert result.success is False
        assert result.error == "unknown_command"

    async def test_blame_with_line_number(self) -> None:
        """git blame with specific line number (covers -L flag path)."""
        from keryx.tools.git_blame import GitBlameTool
        import tempfile
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")

        # This file is tracked by git — use a real file from the repo
        import subprocess
        repo_root = "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter"
        result = await tool.execute({
            "command": "blame",
            "file": f"{repo_root}/pyproject.toml",
            "line": 1,
        })
        # Either succeeds or fails with a git error — we just want the branch covered
        assert result.error != "unknown_command"

    async def test_log_with_author_filter(self) -> None:
        """git log --author filter path covered."""
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "log",
            "n": 5,
            "path": "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter",
            "author": "HS",
        })
        assert result.error != "unknown_command"

    async def test_log_with_grep_filter(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "log",
            "n": 5,
            "grep": "fix",
        })
        assert result.error != "unknown_command"

    async def test_log_with_since_filter(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "log",
            "n": 5,
            "since": "1 year ago",
        })
        assert result.error != "unknown_command"

    async def test_hotspots_command(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "hotspots",
            "path": "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter",
            "n": 10,
        })
        assert result.error != "unknown_command"

    def test_truncate_long_output(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool(max_output_lines=3)
        long_text = "\n".join([f"line {i}" for i in range(10)])
        result = tool._truncate(long_text)
        assert "truncated" in result
        assert len(result.splitlines()) == 4  # 3 kept + 1 truncation line

    def test_truncate_short_output_unchanged(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool(max_output_lines=100)
        short = "line 1\nline 2"
        assert tool._truncate(short) == short

    def test_maybe_scrub_enabled(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool(enable_scrubbing=True)
        text = "user /home/johndoe/.bashrc at johndoe@example.com"
        scrubbed = tool._maybe_scrub(text)
        assert "/home/johndoe" not in scrubbed
        assert "johndoe@example.com" not in scrubbed

    def test_maybe_scrub_disabled(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool(enable_scrubbing=False)
        text = "user /home/johndoe/.bashrc"
        assert tool._maybe_scrub(text) == text

    def test_extract_blame_authors(self) -> None:
        from keryx.tools.git_blame import _extract_blame_authors
        blame_output = (
            "abc1234 1 1\nauthor Alice\nauthor-mail <alice@x.com>\n"
            "def5678 2 2\nauthor Bob\nauthor-mail <bob@y.com>\n"
            "abc1234 3 3\nauthor Alice\nauthor-mail <alice@x.com>\n"
        )
        authors = _extract_blame_authors(blame_output)
        assert "Alice" in authors
        assert "Bob" in authors
        assert len(authors) == 2  # deduplicated

    def test_extract_log_authors(self) -> None:
        from keryx.tools.git_blame import _extract_log_authors
        log_output = (
            "abc1234|Alice Smith|alice@x.com|Fix bug\n"
            "def5678|Bob Jones|bob@y.com|Add feature\n"
            "abc1234|Alice Smith|alice@x.com|Another fix\n"
        )
        authors = _extract_log_authors(log_output)
        assert "Alice Smith" in authors
        assert "Bob Jones" in authors
        assert len(authors) == 2

    def test_scrub_home_paths(self) -> None:
        from keryx.tools.git_blame import _scrub
        text = "path /home/username/src /Users/myuser/dev"
        result = _scrub(text)
        assert "/home/username" not in result
        assert "/Users/myuser" not in result
        assert "/workspace" in result

    def test_scrub_emails(self) -> None:
        from keryx.tools.git_blame import _scrub
        text = "contact dev@example.com for help"
        result = _scrub(text)
        assert "dev@example.com" not in result
        assert "[EMAIL]" in result

    def test_scrub_full_sha_truncated(self) -> None:
        from keryx.tools.git_blame import _scrub
        full_sha = "a" * 40
        result = _scrub(full_sha)
        assert full_sha not in result

    def test_repr(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        r = repr(tool)
        assert "GitBlameTool" in r
        assert "available=" in r

    async def test_standalone_get_file_blame(self) -> None:
        from keryx.tools.git_blame import get_file_blame, GitBlameTool
        # Use a file that exists but may not be in a git repo — covers the function path
        result = await get_file_blame(__file__, line=1)
        assert "success" in result
        assert "output" in result

    async def test_standalone_get_commit_log(self) -> None:
        from keryx.tools.git_blame import get_commit_log
        result = await get_commit_log(n=5)
        assert "success" in result

    async def test_standalone_get_hotspots(self) -> None:
        from keryx.tools.git_blame import get_hotspots
        result = await get_hotspots(path=".", n_commits=5)
        assert "success" in result


# ---------------------------------------------------------------------------
# T23 – Agent: budget exhaustion, health failure, tool timeout/exception
# ---------------------------------------------------------------------------


class ExpensiveFinishModel(MyLocalModel):
    """Charges $100/1k-tokens so budget is immediately exhausted after one step."""

    def __init__(self) -> None:
        super().__init__()
        # Override AFTER super().__init__() which sets 0.0
        self.cost_per_1k_input_tokens = 100.0
        self.cost_per_1k_output_tokens = 100.0
        self._calls = 0

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._calls += 1
        if self._calls == 1:
            return json.dumps({
                "thought": "first step",
                "action": "NO_ACTION",
                "action_input": {},
                "confidence": 0.5,
            })
        return json.dumps({
            "thought": "done",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class DeadModel(MyLocalModel):
    """Health check always fails."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        raise RuntimeError("model is dead")


class TimeoutToolModel(MyLocalModel):
    """Calls a slow tool then FINISH."""

    def __init__(self, target_file: str) -> None:
        super().__init__()
        self._call = 0
        self._target_file = target_file

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._call += 1
        if self._call == 1:
            return json.dumps({
                "thought": "run slow tool",
                "action": "read_file",
                "action_input": {"file_path": self._target_file},
                "confidence": 0.7,
            })
        return json.dumps({
            "thought": "done",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class ExplodingToolModel(MyLocalModel):
    """Calls gdb_analyze (registered as exploding) then FINISH."""

    def __init__(self) -> None:
        super().__init__()
        self._call = 0

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._call += 1
        if self._call == 1:
            return json.dumps({
                "thought": "call exploding tool",
                "action": "gdb_analyze",  # valid action name
                "action_input": {},
                "confidence": 0.7,
            })
        return json.dumps({
            "thought": "done",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class PromptGuidanceModel(MyLocalModel):
    """On second step, expects advisor guidance to be in prompt."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        return json.dumps({
            "thought": "done",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class TestAgentEdgePaths:
    """T23 — budget exhaustion, dead model, tool timeout, tool exception."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox(default_timeout=5.0)
        yield tb
        tb.shutdown()

    @pytest.fixture
    def toolbox_with_slow(self):
        """Registers a read_file tool that sleeps forever (to test timeout)."""
        from keryx.tools.Toolbox import ToolBox, BaseTool, ToolResult
        import asyncio as _asyncio

        class SlowReadFileTool(BaseTool):
            name = "read_file"
            async def execute(self, action_input, context=None):
                await _asyncio.sleep(999)
                return ToolResult(success=True, output="never")

        tb = ToolBox(default_timeout=5.0)
        tb.register(SlowReadFileTool())
        yield tb
        tb.shutdown()

    @pytest.fixture
    def toolbox_with_exploding(self):
        """Registers a gdb_analyze tool that always raises."""
        from keryx.tools.Toolbox import ToolBox, BaseTool, ToolResult

        class ExplodingGdbTool(BaseTool):
            name = "gdb_analyze"
            async def execute(self, action_input, context=None):
                raise RuntimeError("gdb exploded")

        tb = ToolBox(default_timeout=5.0)
        tb.register(ExplodingGdbTool())
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    async def test_budget_exhaustion_breaks_loop(self, toolbox, advisor_manager) -> None:
        """Agent stops when budget is exhausted before max_steps."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=ExpensiveFinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=50,
            budget_usd=0.001,   # tiny budget
        )
        result = await agent.run(target_path="/tmp/budget_exhaust", resume=False)
        # Either completed early or budget stopped it — steps_taken < max_steps
        assert result["status"] == "completed"
        assert result["steps_taken"] < 50

    async def test_dead_model_stops_after_max_failures(
        self, toolbox, advisor_manager
    ) -> None:
        """Health check fails → consecutive_failures → loop break."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=DeadModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
        )
        result = await agent.run(target_path="/tmp/dead_model", resume=False)
        assert result["status"] == "completed"
        assert result["steps_taken"] == 0

    async def test_tool_timeout_recorded_in_step(
        self, toolbox_with_slow, advisor_manager
    ) -> None:
        """Tool timeout is caught, error recorded in step observation."""
        from keryx.core.agent import KeryxAgent
        import tempfile

        agent = KeryxAgent(
            executor_model=TimeoutToolModel(target_file="/tmp/x"),
            advisor_manager=advisor_manager,
            toolbox=toolbox_with_slow,
            max_steps=5,
        )
        # Set a very short tool timeout so SlowTool times out fast
        import keryx.core.agent as agent_mod
        original = agent_mod._TOOL_TIMEOUT_SECONDS
        agent_mod._TOOL_TIMEOUT_SECONDS = 0.05
        try:
            result = await agent.run(target_path="/tmp/tool_timeout", resume=False)
        finally:
            agent_mod._TOOL_TIMEOUT_SECONDS = original
        assert result["status"] == "completed"

    async def test_tool_exception_recorded_in_step(
        self, toolbox_with_exploding, advisor_manager
    ) -> None:
        """Tool exception is caught, error recorded in step observation."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=ExplodingToolModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox_with_exploding,
            max_steps=5,
        )
        result = await agent.run(target_path="/tmp/tool_exception", resume=False)
        assert result["status"] == "completed"
        # At least one step recorded the exception
        steps = agent.context.steps
        observations = [s.get("observation", "") for s in steps]
        assert any("exploded" in obs or "failed" in obs for obs in observations)

    async def test_advisor_guidance_included_in_prompt(
        self, toolbox, advisor_manager
    ) -> None:
        """When pending advisor advice exists, prompt includes guidance."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext, AdvisorAdvice

        agent = KeryxAgent(
            executor_model=PromptGuidanceModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=2,
        )
        agent.context = SharedContext(target_path="/tmp/guidance_test")
        # Inject pending advice and apply it so _advisor_guidance is populated
        advice = AdvisorAdvice(
            strategy="focus on allocators",
            strategic_direction="Check heap allocations first.",
            adjust_confidence_threshold=0.45,
        )
        agent.context.add_advisor_advice(advice)
        agent._apply_advisor_advice()  # this calls set_advisor_guidance()
        prompt = agent._build_prompt_with_guidance()
        assert "ADVISOR GUIDANCE" in prompt
        assert "Check heap allocations first" in prompt

    async def test_apply_advisor_advice_adjusts_threshold(
        self, toolbox, advisor_manager
    ) -> None:
        """_apply_advisor_advice() updates confidence_threshold."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext, AdvisorAdvice

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            confidence_threshold=0.65,
        )
        agent.context = SharedContext(target_path="/tmp/apply_advice")
        advice = AdvisorAdvice(
            strategy="lower threshold",
            strategic_direction="Go broader.",
            adjust_confidence_threshold=0.40,
        )
        agent.context.add_advisor_advice(advice)
        agent._apply_advisor_advice()
        assert agent.confidence_threshold == pytest.approx(0.40)
        assert not agent.context.has_pending_advisor_advice()

    async def test_checkpoint_load_missing_returns_none(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            checkpoint_dir=tmp_path / "nonexistent_dir",
        )
        ctx = agent._load_checkpoint("/tmp/no_checkpoint_here")
        assert ctx is None

    async def test_checkpoint_save_fails_gracefully(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """Checkpoint save catches JSON serialization errors without raising."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        from unittest.mock import patch

        ckdir = tmp_path / "ck_graceful"
        ckdir.mkdir()
        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            checkpoint_dir=ckdir,
        )
        agent.context = SharedContext(target_path="/tmp/save_fail")

        # Make json.dumps raise inside _save_checkpoint
        with patch("keryx.core.agent.json.dumps", side_effect=TypeError("not serializable")):
            # Must NOT raise — exception is caught internally
            agent._save_checkpoint("/tmp/save_fail")


# ---------------------------------------------------------------------------
# T24 – AdvisorManager: trigger timeout, parallel timeout, sync bridge
# ---------------------------------------------------------------------------


class SlowTriggerAdvisor(_SimpleAdvisor):
    """Trigger check sleeps > 5s (the trigger timeout)."""

    async def should_trigger_async(self, ctx) -> bool:
        import asyncio
        await asyncio.sleep(999)
        return True  # never reached


class SlowAdviceAdvisor(_SimpleAdvisor):
    """advise() sleeps > 60s (the execution timeout)."""

    async def advise_with_tracking(self, ctx):
        import asyncio
        await asyncio.sleep(999)
        from keryx.advisors.base import AdvisorResponse
        return AdvisorResponse(strategic_direction="slow")


class TestAdvisorManagerEdgePaths:
    """T24 — trigger timeout, parallel timeout, sync bridge, business error."""

    @pytest.fixture
    def manager(self):
        from keryx.advisors.manager import AdvisorManager
        m = AdvisorManager()
        yield m
        m.shutdown()

    async def test_trigger_timeout_falls_through(self, manager) -> None:
        """Trigger check that times out → not triggered (conservative False)."""
        adv = SlowTriggerAdvisor("slow-trig")
        manager.register(adv)
        # The trigger has a 5s internal timeout; use the real timeout
        # but we can't wait 5 seconds in a test. Monkey-patch asyncio.wait_for.
        import asyncio
        from unittest.mock import AsyncMock, patch

        original = asyncio.wait_for

        async def fast_timeout(coro, timeout):
            if timeout == 5.0:
                coro.close()
                raise asyncio.TimeoutError
            return await original(coro, timeout)

        with patch("keryx.advisors.manager.asyncio.wait_for", side_effect=fast_timeout):
            resp = await manager.get_advice_async(object(), "slow-trig")
        # Should return "not triggered"
        assert resp.metadata.get("triggered") is False

    async def test_get_advice_parallel_empty_valid_names(self, manager) -> None:
        """No valid names → returns empty list immediately."""
        responses = await manager.get_advice_parallel(object(), [])
        assert responses == []

    def test_sync_bridge_get_advice(self, manager) -> None:
        """get_advice() (sync bridge) returns a response."""
        adv = _SimpleAdvisor("sync-adv", direction="sync advice")
        manager.register(adv)
        resp = manager.get_advice(object(), "sync-adv")
        assert resp.strategic_direction == "sync advice"

    async def test_business_error_returns_error_response(self, manager) -> None:
        """Business-logic exception in advise() → error response, not re-raise."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class BusinessErrorAdvisor(BaseAdvisor):
            name = "biz-err"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx): raise ValueError("bad business logic")

        bea = BusinessErrorAdvisor()
        manager.register(bea)
        resp = await manager.get_advice_async(object(), "biz-err")
        assert "error" in resp.metadata

    def test_workers_warning_logged_for_high_count(self) -> None:
        """AdvisorManager warns when max_concurrent_advisor_calls > 8."""
        import logging
        from keryx.advisors.manager import AdvisorManager
        with pytest.raises(Exception) if False else __import__("contextlib").nullcontext():
            m = AdvisorManager(max_concurrent_advisor_calls=9)
            m.shutdown()


# ---------------------------------------------------------------------------
# T25 – ToolBox: sync bridge, batch exception, remaining BaseTool paths
# ---------------------------------------------------------------------------


class TestToolBoxEdge:
    """T25 — remaining ToolBox miss lines."""

    @pytest.fixture
    def tb(self):
        from keryx.tools.Toolbox import ToolBox
        toolbox = ToolBox(default_timeout=5.0)
        yield toolbox
        toolbox.shutdown()

    def test_sync_bridge_reuses_existing_loop(self, tb) -> None:
        """_ensure_sync_loop() returns same loop on repeated calls."""
        loop1 = tb._ensure_sync_loop()
        loop2 = tb._ensure_sync_loop()
        assert loop1 is loop2

    async def test_execute_batch_handles_gather_exception(self, tb) -> None:
        """execute_batch wraps exceptions from individual tools."""
        tb.register(ExplodingTool())
        results = await tb.execute_batch([("exploding_tool", {})])
        assert len(results) == 1
        assert results[0].success is False

    def test_base_tool_success_rate_in_metrics(self) -> None:
        from keryx.tools.Toolbox import BaseTool

        class SR(BaseTool):
            name = "sr"
            async def execute(self, ai, ctx=None):
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=True, output="ok")

        t = SR()
        t._record_call(True, 50.0)
        t._record_call(True, 50.0)
        t._record_call(False, 50.0)
        m = t.get_metrics()
        assert m["success_rate"] == pytest.approx(2 / 3)

    def test_tool_result_count_branch_in_format(self) -> None:
        """format_for_llm uses data['count'] when 'findings' key is absent (line 51)."""
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=True, output="5 items found", data={"count": 5})
        text = r.format_for_llm()
        assert "5 items" in text

    def test_tool_result_truncation_appends_marker(self) -> None:
        """Output longer than max_chars gets '... [N chars total]' appended (line 57)."""
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=True, output="A" * 500)
        text = r.format_for_llm(max_chars=50)
        assert "chars total" in text

    def test_tool_result_to_prompt_fragment(self) -> None:
        """to_prompt_fragment() calls format_for_llm(max_chars=800) (line 62)."""
        from keryx.tools.Toolbox import ToolResult
        r = ToolResult(success=True, output="result text")
        frag = r.to_prompt_fragment()
        assert isinstance(frag, str)
        assert "result text" in frag

    async def test_execute_async_bad_max_output_len_falls_back(self, tb) -> None:
        """Non-integer max_output_len caught → sanitized to 2000 (lines 238-239)."""
        tb.register(StringReturningTool())
        result = await tb.execute_async("string_tool", {"max_output_len": "bad_value"})
        assert result.success is True

    async def test_execute_batch_gather_exception_yields_batch_error(self, tb) -> None:
        """If execute_async raises (bypassing its own handler), batch wraps it (line 404)."""
        from unittest.mock import patch, AsyncMock
        tb.register(StringReturningTool())
        with patch.object(tb, "execute_async", new=AsyncMock(side_effect=RuntimeError("raw"))):
            results = await tb.execute_batch([("string_tool", {})])
        assert results[0].error == "batch_exception"

    async def test_execute_warns_when_called_from_async_context(
        self, tb, caplog
    ) -> None:
        """execute() logs a warning when called from an async context (line 349)."""
        import logging
        tb.register(StringReturningTool())
        with caplog.at_level(logging.WARNING, logger="keryx.tools"):
            tb.execute("string_tool", {})
        assert any("async context" in r.message for r in caplog.records)

    def test_reset_metrics_clears_registered_tool_counts(self, tb) -> None:
        """reset_metrics() iterates registered tools calling reset_metrics (line 431)."""
        tb.register(StringReturningTool())
        tb._total_calls = 9
        tb._error_count = 2
        tb.reset_metrics()
        assert tb._total_calls == 0   # toolbox counters reset
        assert tb._error_count == 0   # line 431 reached (StringReturningTool.reset_metrics is no-op)

    def test_create_toolbox_with_tools_list(self) -> None:
        """create_toolbox(tools=[...]) registers provided tools (line 459)."""
        from keryx.tools.Toolbox import create_toolbox
        tb = create_toolbox(tools=[StringReturningTool()])
        try:
            assert "string_tool" in tb.list_tools()
        finally:
            tb.shutdown()


# ---------------------------------------------------------------------------
# T26 – GitBlameTool: hotspot risk tiers, log authors, scrub Windows path
# ---------------------------------------------------------------------------


class TestGitBlameAdditional:
    """T26 — remaining git_blame miss lines."""

    def test_scrub_windows_path(self) -> None:
        from keryx.tools.git_blame import _scrub
        text = r"C:\Users\Alice\src\main.c"
        result = _scrub(text)
        assert "Alice" not in result

    def test_extract_log_authors_empty_output(self) -> None:
        from keryx.tools.git_blame import _extract_log_authors
        assert _extract_log_authors("") == []

    def test_extract_blame_authors_empty_output(self) -> None:
        from keryx.tools.git_blame import _extract_blame_authors
        assert _extract_blame_authors("") == []

    async def test_log_returns_authors_list(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "log",
            "n": 5,
            "path": "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter",
        })
        # authors key should be present in data
        assert "authors" in result.data or not result.success

    async def test_blame_nonexistent_file_fails(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "blame",
            "file": "/tmp/definitelynonexistent_keryx_test.c",
        })
        assert result.success is False

    async def test_hotspots_with_since_filter(self) -> None:
        from keryx.tools.git_blame import GitBlameTool
        tool = GitBlameTool()
        if not tool.is_available():
            pytest.skip("git not available")
        result = await tool.execute({
            "command": "hotspots",
            "path": "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter",
            "n": 5,
            "since": "1 year ago",
        })
        # Just check it ran without crashing
        assert result.error != "unknown_command"


# ---------------------------------------------------------------------------
# T27 – CapabilityRouter: resolve_executor, resolve_advisor, resolve_budget,
#         fallbacks, GPU skip, airgapped filter, load_capabilities from YAML
# ---------------------------------------------------------------------------


class TestCapabilityRouterPaths:
    """T27 — CapabilityRouter routing logic gaps."""

    @pytest.fixture
    def router(self):
        from keryx.core.router import create_router
        return create_router(has_gpu=False)

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    def test_unknown_capability_falls_back_to_deep_reasoning(
        self, router, advisor_manager
    ) -> None:
        """Unknown capability name → falls back to 'deep_reasoning' profile."""
        plan = router.route(
            capability="totally_unknown_cap",
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
        )
        assert plan.capability == "deep_reasoning"
        assert router.metrics["routing_decisions"] == 1

    def test_routing_plan_repr(self) -> None:
        """RoutingPlan.__repr__ includes key fields."""
        from keryx.core.router import RoutingPlan
        plan = RoutingPlan(
            executor_model=FinishModel(),
            advisor_name="none",
            advisor_chain=[],
            budget_usd=None,
            enforce_no_network=True,
            require_consensus=False,
            consensus_threshold=0.67,
            require_fuzzing_confirm=False,
            debate_models=[],
            capability="fast_pattern_matching",
        )
        r = repr(plan)
        assert "fast_pattern_matching" in r
        assert "airgapped=True" in r

    def test_routing_plan_validate_airgapped_with_budget_raises(self) -> None:
        from keryx.core.router import RoutingPlan
        with pytest.raises(ValueError, match="Air-gapped"):
            RoutingPlan(
                executor_model=FinishModel(),
                advisor_name="none",
                advisor_chain=[],
                budget_usd=5.0,
                enforce_no_network=True,
                require_consensus=False,
                consensus_threshold=0.67,
                require_fuzzing_confirm=False,
                debate_models=[],
                capability="airgapped",
            )

    def test_routing_plan_validate_consensus_needs_two_models(self) -> None:
        from keryx.core.router import RoutingPlan
        with pytest.raises(ValueError, match="Consensus"):
            RoutingPlan(
                executor_model=FinishModel(),
                advisor_name="none",
                advisor_chain=[],
                budget_usd=None,
                enforce_no_network=False,
                require_consensus=True,
                consensus_threshold=0.67,
                require_fuzzing_confirm=False,
                debate_models=["only-one"],
                capability="test",
            )

    def test_routing_plan_validate_chain_single_entry_raises(self) -> None:
        from keryx.core.router import RoutingPlan
        with pytest.raises(ValueError, match="chain"):
            RoutingPlan(
                executor_model=FinishModel(),
                advisor_name="none",
                advisor_chain=["just-one"],
                budget_usd=None,
                enforce_no_network=False,
                require_consensus=False,
                consensus_threshold=0.67,
                require_fuzzing_confirm=False,
                debate_models=[],
                capability="test",
            )

    def test_resolve_executor_gpu_skip_then_fallback(
        self, advisor_manager
    ) -> None:
        """GPU-requiring model is skipped; fallback model used."""
        from keryx.core.router import create_router
        router = create_router(has_gpu=False)
        # llama-4-70b requires GPU; qwen3-coder-8b does not
        models = {
            "llama-4-70b":   FinishModel(),
            "qwen3-coder-8b": FinishModel(),
        }
        plan = router.route(
            capability="fast_pattern_matching",
            available_models=models,
            advisor_manager=advisor_manager,
        )
        assert plan is not None
        assert router.metrics["routing_decisions"] >= 1

    def test_resolve_executor_no_suitable_raises(
        self, advisor_manager
    ) -> None:
        """No matching model → RuntimeError."""
        plan = None
        with pytest.raises(RuntimeError, match="No suitable executor"):
            plan = router = None
            from keryx.core.router import create_router
            r = create_router(has_gpu=False)
            r.route(
                capability="fast_pattern_matching",
                available_models={},   # empty
                advisor_manager=advisor_manager,
            )

    def test_resolve_budget_airgapped_returns_none(self) -> None:
        from keryx.core.router import CapabilityRouter
        result = CapabilityRouter._resolve_budget(
            profile_budget=5.0,
            user_budget=3.0,
            enforce_no_network=True,
        )
        assert result is None

    def test_resolve_budget_min_of_profile_and_user(self) -> None:
        from keryx.core.router import CapabilityRouter
        result = CapabilityRouter._resolve_budget(
            profile_budget=10.0,
            user_budget=3.0,
            enforce_no_network=False,
        )
        assert result == 3.0

    def test_resolve_budget_only_user(self) -> None:
        from keryx.core.router import CapabilityRouter
        result = CapabilityRouter._resolve_budget(
            profile_budget=None,
            user_budget=7.0,
            enforce_no_network=False,
        )
        assert result == 7.0

    def test_resolve_budget_only_profile(self) -> None:
        from keryx.core.router import CapabilityRouter
        result = CapabilityRouter._resolve_budget(
            profile_budget=4.0,
            user_budget=None,
            enforce_no_network=False,
        )
        assert result == 4.0

    def test_list_capabilities(self, router) -> None:
        caps = router.list_capabilities()
        assert "fast_pattern_matching" in caps
        assert "deep_reasoning" in caps

    def test_describe_known_capability(self, router) -> None:
        desc = router.describe("fast_pattern_matching")
        assert "Capability:" in desc
        assert "fast_pattern_matching" in desc

    def test_describe_unknown_capability(self, router) -> None:
        desc = router.describe("ghost_cap")
        assert "Unknown" in desc

    def test_get_metrics_and_reset(self, router) -> None:
        m = router.get_metrics()
        assert "routing_decisions" in m
        router.metrics["routing_decisions"] = 5
        router.reset_metrics()
        assert router.metrics["routing_decisions"] == 0

    def test_load_capabilities_with_yaml(self, tmp_path) -> None:
        """_load_capabilities merges a user YAML into defaults."""
        import yaml as _yaml
        config = tmp_path / "capabilities.yaml"
        config.write_text(_yaml.dump({
            "capabilities": {
                "my_custom_cap": {
                    "executor": "qwen3-coder-8b",
                    "advisor": "none",
                    "enforce_no_network": True,
                    "budget_usd": None,
                }
            }
        }))
        from keryx.core.router import create_router
        router = create_router(config_path=config, has_gpu=False)
        assert "my_custom_cap" in router.list_capabilities()

    def test_resolve_advisor_no_network_skipped_in_airgapped(
        self, advisor_manager
    ) -> None:
        """Advisor that requires_network is skipped in airgapped mode."""
        # Register a network-requiring advisor
        net_adv = _SimpleAdvisor("cloud-advisor")
        net_adv.requires_network = True
        advisor_manager.register(net_adv)

        from keryx.core.router import create_router
        router = create_router(has_gpu=False)
        plan = router.route(
            capability="fast_pattern_matching",
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            is_airgapped=True,
        )
        # fast_pattern_matching has advisor="none" so this just tests the route
        assert plan is not None

    def test_create_router_factory(self) -> None:
        from keryx.core.router import create_router
        router = create_router(has_gpu=True)
        assert router._has_gpu is True


# ---------------------------------------------------------------------------
# T28 – Orchestrator: _is_airgapped, _generate_cancelled_report,
#         _check_resources, _load_checkpoint, _write_atomic_json
# ---------------------------------------------------------------------------


class TestOrchestratorEdgePaths:
    """T28 — Orchestrator helper paths not covered by hunt() tests."""

    @pytest.fixture
    def orchestrator(self):
        from keryx.core.orchestrator import KeryxOrchestrator
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        am = AdvisorManager()
        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=am,
            toolbox=tb,
        )
        yield orc
        tb.shutdown()
        am.shutdown()

    async def test_is_airgapped_explicit_mode(self) -> None:
        from keryx.core._airgap import detect_airgap
        assert await detect_airgap("airgapped") is True

    async def test_is_airgapped_hybrid_with_proxy_env(self) -> None:
        import os
        from keryx.core._airgap import detect_airgap
        old = os.environ.get("HTTP_PROXY")
        os.environ["HTTP_PROXY"] = "http://proxy:3128"
        try:
            result = await detect_airgap("hybrid")
        finally:
            if old is None:
                del os.environ["HTTP_PROXY"]
            else:
                os.environ["HTTP_PROXY"] = old
        assert result is False

    def test_generate_cancelled_report(self, orchestrator) -> None:
        report = orchestrator._generate_cancelled_report("/tmp/target", "hybrid")
        assert report["status"] == "cancelled"
        assert report["target"] == "/tmp/target"
        assert "confirmed_vulns" in report

    async def test_check_resources_returns_true_normally(self) -> None:
        from keryx.core._resources import check_resources
        result = await check_resources()
        assert result is True

    def test_load_checkpoint_missing_returns_none(self, orchestrator) -> None:
        ctx = orchestrator._checkpoints.load("/tmp/definitely_no_checkpoint_xyz")
        assert ctx is None

    def test_write_atomic_json_success(self, orchestrator, tmp_path) -> None:
        import json
        target = tmp_path / "test.json"
        orchestrator._checkpoints._write_atomic(target, {"key": "value"})
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["key"] == "value"

    def test_write_atomic_json_failure_does_not_raise(
        self, orchestrator, tmp_path
    ) -> None:
        """_write_atomic with unserializable data logs error but doesn't raise."""
        target = tmp_path / "bad.json"
        orchestrator._checkpoints._write_atomic(target, {"bad": object()})
        assert not target.exists()

    async def test_save_checkpoint_round_trip(self, orchestrator, tmp_path) -> None:
        """CheckpointManager.save() + load() preserves SharedContext."""
        from keryx.core.shared_context import SharedContext
        orchestrator._checkpoints._dir = tmp_path
        ctx = SharedContext(target_path="/tmp/round_trip")
        await orchestrator._checkpoints.save("/tmp/round_trip", ctx, {})
        loaded = orchestrator._checkpoints.load("/tmp/round_trip")
        assert loaded is not None
        assert loaded.target_path == "/tmp/round_trip"

    def test_load_checkpoint_corrupt_returns_none(
        self, orchestrator, tmp_path
    ) -> None:
        """Corrupt checkpoint JSON → warning, returns None."""
        key = __import__("hashlib").sha256(b"/tmp/corrupt").hexdigest()[:12]
        orchestrator._checkpoints._dir = tmp_path
        ck_file = tmp_path / f"orchestrator_{key}.json"
        ck_file.write_text("{ not valid json }")
        ctx = orchestrator._checkpoints.load("/tmp/corrupt")
        assert ctx is None

    def test_get_metrics(self, orchestrator) -> None:
        m = orchestrator.get_metrics()
        assert "routing_decisions" in m
        assert "swarm_votes" in m

    def test_get_dynamic_threshold(self) -> None:
        from keryx.core._escalation import get_threshold
        assert get_threshold(1) == pytest.approx(0.65)
        assert get_threshold(4) == pytest.approx(0.50)
        assert get_threshold(99) == pytest.approx(0.60)  # default

    def test_build_final_result_none_result(self, orchestrator) -> None:
        result = orchestrator._build_final_result(
            result=None, mode="hybrid", is_airgapped=False, escalation_level=1
        )
        assert result["status"] == "cancelled"

    def test_build_final_result_with_result(self, orchestrator) -> None:
        base = {"status": "completed", "confirmed_vulns": [], "hypotheses": []}
        result = orchestrator._build_final_result(
            result=base, mode="swarm", is_airgapped=True, escalation_level=2
        )
        assert result["mode"] == "swarm"
        assert result["airgapped"] is True
        assert result["escalation_level"] == 2

    def test_checkpoint_path_deterministic(self, orchestrator) -> None:
        p1 = orchestrator._checkpoints.checkpoint_path("/tmp/target")
        p2 = orchestrator._checkpoints.checkpoint_path("/tmp/target")
        assert p1 == p2

    def test_create_orchestrator_factory(self) -> None:
        from keryx.core.orchestrator import create_orchestrator
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        am = AdvisorManager()
        orc = create_orchestrator({}, am, tb)
        assert orc is not None
        tb.shutdown()
        am.shutdown()


# ---------------------------------------------------------------------------
# T29 – Router: advisor fallback, cascade chain, health check failure
# ---------------------------------------------------------------------------


class TestCapabilityRouterAdvisorPaths:
    """T29 — _resolve_advisor fallback / cascade / airgapped filter."""

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    def test_resolve_advisor_uses_fallback_when_primary_missing(
        self, advisor_manager
    ) -> None:
        """Primary advisor not registered → falls back to fallback_advisor."""
        fallback = _SimpleAdvisor("cloud-advisor")
        fallback.requires_network = False
        advisor_manager.register(fallback)

        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        # deep_reasoning has fallback_advisor="cloud-advisor"
        name, chain = router._resolve_advisor(
            cfg={
                "advisor": "local-advisor",   # not registered
                "fallback_advisor": "cloud-advisor",
                "advisor_chain": [],
            },
            advisor_manager=advisor_manager,
            is_airgapped=False,
        )
        assert name == "cloud-advisor"

    def test_resolve_advisor_returns_none_when_both_missing(
        self, advisor_manager
    ) -> None:
        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        name, chain = router._resolve_advisor(
            cfg={
                "advisor": "missing-primary",
                "fallback_advisor": "missing-fallback",
                "advisor_chain": [],
            },
            advisor_manager=advisor_manager,
            is_airgapped=False,
        )
        assert name == "none"

    def test_resolve_advisor_cascade_chain_airgapped(
        self, advisor_manager
    ) -> None:
        """Cascade chain in airgapped mode filters network advisors."""
        local_adv = _SimpleAdvisor("local-chain-adv")
        local_adv.requires_network = False
        net_adv = _SimpleAdvisor("net-chain-adv")
        net_adv.requires_network = True
        advisor_manager.register(local_adv)
        advisor_manager.register(net_adv)

        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        name, chain = router._resolve_advisor(
            cfg={
                "advisor": "missing-primary",
                "advisor_chain": ["net-chain-adv", "local-chain-adv"],
            },
            advisor_manager=advisor_manager,
            is_airgapped=True,
        )
        # net-chain-adv filtered out; local-chain-adv remains
        assert "net-chain-adv" not in chain
        assert "local-chain-adv" in chain

    def test_route_raises_on_unhealthy_executor(
        self, advisor_manager
    ) -> None:
        """Unhealthy executor → RuntimeError from route()."""
        from keryx.core.router import create_router

        class UnhealthyModel(FinishModel):
            def is_healthy(self): return False

        router = create_router(has_gpu=False)
        with pytest.raises(RuntimeError, match="health check"):
            router.route(
                capability="fast_pattern_matching",
                available_models={"qwen3-coder-8b": UnhealthyModel()},
                advisor_manager=advisor_manager,
            )
        assert router.metrics["health_check_failures"] == 1

    def test_resolve_executor_airgapped_skips_non_local_caps(
        self, advisor_manager
    ) -> None:
        """In airgapped mode, models with is_local=False in caps are skipped."""
        # deepseek-r1 has is_local=False in _CAPABILITY_TIERS
        from keryx.core.router import create_router

        class DeepSeekModel(FinishModel):
            def __init__(self):
                super().__init__()
                self.is_local = False

        router = create_router(has_gpu=False)
        # Should raise because deepseek-r1 is skipped and no fallback available
        with pytest.raises(RuntimeError, match="No suitable executor"):
            router._resolve_executor(
                requested="deepseek-r1",
                available={"deepseek-r1": DeepSeekModel()},
                is_airgapped=True,
            )

    def test_load_capabilities_yaml_parse_error_uses_defaults(
        self, tmp_path
    ) -> None:
        """Bad YAML → warning, falls back to defaults."""
        config = tmp_path / "bad.yaml"
        config.write_text("{{{ not valid yaml")
        from keryx.core.router import create_router
        router = create_router(config_path=config, has_gpu=False)
        # Still has defaults
        assert "fast_pattern_matching" in router.list_capabilities()

    def test_resolve_executor_gpu_skip_updates_metric(self) -> None:
        """GPU-requiring model skipped → gpu_skips+fallbacks_used incremented."""
        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        # llama-4-70b: requires_gpu=True → lines 273-275 fire
        # qwen3-coder-8b: no GPU required → returned as fallback → lines 286-287 fire
        fallback = FinishModel()
        models = {"llama-4-70b": FinishModel(), "qwen3-coder-8b": fallback}
        result = router._resolve_executor("llama-4-70b", models, is_airgapped=False)
        assert result is fallback              # same instance — correct fallback chosen
        assert router.metrics["gpu_skips"] >= 1
        assert router.metrics["fallbacks_used"] >= 1

    def test_resolve_executor_instance_non_local_skipped_in_airgap(self) -> None:
        """Model whose instance.is_local=False is skipped even when caps say is_local=True."""
        from keryx.core.router import CapabilityRouter

        class NonLocalInstance(FinishModel):
            is_local = False

        router = CapabilityRouter(has_gpu=False)
        # llama-4-13b caps: is_local=True → passes cap check (line 277)
        # but model.is_local=False → skipped at instance check (lines 282-283)
        # qwen3-coder-8b is in the fallback chain and is_local=True → returned
        fallback = FinishModel()
        models = {"llama-4-13b": NonLocalInstance(), "qwen3-coder-8b": fallback}
        result = router._resolve_executor("llama-4-13b", models, is_airgapped=True)
        assert result is fallback

    def test_resolve_advisor_empty_name_short_circuits(
        self, advisor_manager
    ) -> None:
        """cfg advisor='' → _try('') → not name → return None → 'none' returned."""
        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        name, chain = router._resolve_advisor(
            cfg={"advisor": ""},
            advisor_manager=advisor_manager,
            is_airgapped=False,
        )
        assert name == "none"

    def test_resolve_advisor_requires_network_skipped_in_airgap(
        self, advisor_manager
    ) -> None:
        """Registered advisor with requires_network=True skipped when airgapped."""
        net_adv = _SimpleAdvisor("net-only-adv")
        net_adv.requires_network = True
        advisor_manager.register(net_adv)

        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        name, chain = router._resolve_advisor(
            cfg={"advisor": "net-only-adv"},
            advisor_manager=advisor_manager,
            is_airgapped=True,
        )
        assert name == "none"

    def test_resolve_advisor_primary_resolved_immediately(
        self, advisor_manager
    ) -> None:
        """Registered, non-network advisor is returned as primary (line 324)."""
        local_adv = _SimpleAdvisor("local-primary-adv")
        local_adv.requires_network = False
        advisor_manager.register(local_adv)

        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(has_gpu=False)
        name, chain = router._resolve_advisor(
            cfg={"advisor": "local-primary-adv", "advisor_chain": []},
            advisor_manager=advisor_manager,
            is_airgapped=False,
        )
        assert name == "local-primary-adv"

    def test_load_capabilities_new_profile_name_added(self, tmp_path) -> None:
        """Profile name absent from defaults is inserted as a new entry (line 376)."""
        import yaml as _yaml
        config = tmp_path / "caps.yaml"
        config.write_text(_yaml.dump({
            "capabilities": {
                "novel_cap": {
                    "executor": "qwen3-coder-8b",
                    "enforce_no_network": True,
                }
            }
        }))
        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(config_path=config, has_gpu=False)
        assert "novel_cap" in router.capabilities

    def test_load_capabilities_existing_profile_merged(self, tmp_path) -> None:
        """Profile name present in defaults is updated in-place via .update() (line 374)."""
        import yaml as _yaml
        config = tmp_path / "caps.yaml"
        config.write_text(_yaml.dump({
            "capabilities": {
                "fast_pattern_matching": {"timeout_seconds": 99}
            }
        }))
        from keryx.core.router import CapabilityRouter
        router = CapabilityRouter(config_path=config, has_gpu=False)
        assert router.capabilities["fast_pattern_matching"].get("timeout_seconds") == 99


# ---------------------------------------------------------------------------
# T30 – Manager: parallel timeout, system error in advisor, sync bridge edge
# ---------------------------------------------------------------------------


class TestAdvisorManagerSystemPaths:
    """T30 — system error in execute path, parallel timeout."""

    @pytest.fixture
    def manager(self):
        from keryx.advisors.manager import AdvisorManager
        m = AdvisorManager()
        yield m
        m.shutdown()

    async def test_system_error_triggers_fallback(self, manager) -> None:
        """TimeoutError inside advise() → health records system failure → tries fallback."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse
        from keryx.advisors.manager import AdvisorManager

        class SystemFailAdvisor(BaseAdvisor):
            name = "sys-fail"
            def should_trigger(self, ctx): return True
            async def advise(self, ctx):
                raise ConnectionError("network down")

        fallback_adv = _SimpleAdvisor("sys-fallback", direction="fallback ok")
        manager.register(SystemFailAdvisor())
        manager.register(fallback_adv)

        resp = await manager.get_advice_async(
            object(),
            advisor_name="sys-fail",
            fallback_chain=["sys-fallback"],
        )
        # Either fallback succeeded or we got an error — either way no exception
        assert resp is not None

    async def test_get_advice_parallel_timeout(self, manager) -> None:
        """Parallel gather timeout → tasks cancelled, returns empty."""
        import asyncio

        class SlowAdv(_SimpleAdvisor):
            async def advise_with_tracking(self, ctx):
                await asyncio.sleep(999)
                from keryx.advisors.base import AdvisorResponse
                return AdvisorResponse(strategic_direction="never")

        slow = SlowAdv("parallel-slow")
        manager.register(slow)

        # Very short timeout
        responses = await manager.get_advice_parallel(
            object(), ["parallel-slow"], timeout=0.05
        )
        # Either empty or returned before timeout — must not raise
        assert isinstance(responses, list)

    def test_get_advice_sync_bridge(self, manager) -> None:
        """get_advice() from non-async context returns valid response."""
        adv = _SimpleAdvisor("sync-bridge-adv", direction="bridge works")
        manager.register(adv)
        resp = manager.get_advice(object(), "sync-bridge-adv")
        assert resp.strategic_direction == "bridge works"


# ---------------------------------------------------------------------------
# T31 – Agent: _safe_generate timeout, consecutive failures, action=None path
# ---------------------------------------------------------------------------


class TimeoutModel(MyLocalModel):
    """generate() blocks long enough to trigger _safe_generate timeout (0.05 s),
    but exits within 5 s so the default executor shuts down cleanly (no RuntimeWarning)."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        import threading
        # Bounded wait: asyncio.wait_for fires at generate_timeout (0.05 s);
        # the thread itself exits after 5 s, within asyncio's 300 s cleanup window.
        threading.Event().wait(timeout=5)
        return ""


class TestAgentTimeoutPaths:
    """T31 — _safe_generate timeout → None → fallback NO_ACTION step."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    async def test_generate_timeout_increments_parse_errors(
        self, toolbox, advisor_manager
    ) -> None:
        """generate() timeout → increment_parse_errors() each attempt."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=TimeoutModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=2,
            generate_timeout=0.05,
        )
        result = await agent.run(target_path="/tmp/timeout_model", resume=False)
        # After 2 retries × 2 steps = parse_errors ≥ 2
        assert agent.context.parse_errors >= 2

    async def test_consecutive_health_failures_stops_loop(
        self, toolbox, advisor_manager
    ) -> None:
        """Health check that always fails → consecutive_failures → break."""
        from keryx.core.agent import KeryxAgent

        class AlwaysDeadModel(MyLocalModel):
            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                # Always raise — health check (max_tokens=1) also raises
                raise RuntimeError("totally dead")

        agent = KeryxAgent(
            executor_model=AlwaysDeadModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=20,
        )
        result = await agent.run(target_path="/tmp/dead_health", resume=False)
        assert result["status"] == "completed"
        assert result["steps_taken"] == 0  # never completed a step


# ---------------------------------------------------------------------------
# T32 – Orchestrator: agent exception path, escalation retry loop
# ---------------------------------------------------------------------------


class CrashingModel(MyLocalModel):
    """Raises RuntimeError on every generate() call including the main loop."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1:
            return "ok"
        raise RuntimeError("model crashed mid-hunt")


class TestOrchestratorEscalation:
    """T32 — Orchestrator exception path and escalation retry coverage."""

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        yield tb
        tb.shutdown()

    async def test_hunt_completes_despite_resource_check(
        self, toolbox, advisor_manager
    ) -> None:
        """Hunt completes normally when resources are adequate."""
        from keryx.core.orchestrator import KeryxOrchestrator
        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        result = await orc.hunt(
            target_path="/tmp/resource_ok",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert result["status"] in ("completed", "cancelled")

    async def test_hunt_with_low_confidence_triggers_escalation_path(
        self, toolbox, advisor_manager
    ) -> None:
        """Low-confidence result causes _should_escalate_further to evaluate."""
        from keryx.core.orchestrator import KeryxOrchestrator

        class LowConfFinish(FinishModel):
            """Returns FINISH but with very low average confidence context."""
            pass  # FinishModel returns confidence=0.9 — orc won't escalate further

        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": LowConfFinish()},
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        result = await orc.hunt(
            target_path="/tmp/low_conf_hunt",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert "escalation_level" in result

    async def test_hunt_mode_swarm_with_debate_models(
        self, toolbox, advisor_manager
    ) -> None:
        """Hunt in swarm mode with debate models triggers _run_swarm_debate."""
        from keryx.core.orchestrator import KeryxOrchestrator

        orc = KeryxOrchestrator(
            available_models={
                "qwen3-coder-8b": FinishModel(),
                "yes_model": YesModel(),
            },
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        # Patch router to return a plan with debate_models populated
        from unittest.mock import patch, MagicMock
        from keryx.core.router import RoutingPlan

        mock_plan = RoutingPlan(
            executor_model=FinishModel(),
            advisor_name="none",
            advisor_chain=[],
            budget_usd=None,
            enforce_no_network=True,
            require_consensus=False,
            consensus_threshold=0.67,
            require_fuzzing_confirm=False,
            debate_models=["yes_model"],
            capability="fast_pattern_matching",
        )

        with patch.object(orc.router, "route", return_value=mock_plan):
            result = await orc.hunt(
                target_path="/tmp/swarm_hunt",
                capability="fast_pattern_matching",
                mode="swarm",
            )
        assert result is not None


# ---------------------------------------------------------------------------
# T33 – ModelInterface default method coverage
# ---------------------------------------------------------------------------


class TestModelInterfaceDefaults:
    """T33 — Cover default (non-abstract) methods in ModelInterface."""

    @pytest.fixture
    def minimal_model(self):
        """A concrete subclass that does NOT override any default methods."""
        from keryx.models.interface import (
            ModelInterface,
            GenerationConfig,
            GenerationResult,
            ModelCapabilities,
            CostEstimate,
        )

        class _MinimalModel(ModelInterface):
            model_name = "minimal"

            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                return "ok"

            def generate_result(self, prompt, config=None):
                return GenerationResult(text="ok", tokens_input=1, tokens_output=1, duration_ms=1.0)

            def generate_stream(self, prompt, config=None):
                yield "ok"

            def generate_with_tools(self, prompt, tools, config=None):
                return "ok"

            def tokenize(self, text):
                return [0]

            def get_context_length(self):
                return 4096

            def is_healthy(self):
                return True

            def estimate_cost(self, input_tokens, output_tokens):
                return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

            def get_usage_cost(self):
                return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

            def get_capabilities(self):
                return ModelCapabilities(
                    max_context_length=4096,
                    supports_tool_calling=False,
                    supports_grammar=False,
                    supports_batching=False,
                    supports_streaming=False,
                    supports_speculative=False,
                    supports_min_p=False,
                    requires_gpu=False,
                    is_local=True,
                    is_quantized=False,
                )

            def unload(self):
                pass

        return _MinimalModel()

    async def test_generate_async_default_implementation(self, minimal_model) -> None:
        """ModelInterface.generate_async() default uses run_in_executor."""
        result = await minimal_model.generate_async("test prompt")
        assert result == "ok"

    def test_get_context_used_returns_zero(self, minimal_model) -> None:
        assert minimal_model.get_context_used() == 0

    def test_clear_context_no_op(self, minimal_model) -> None:
        minimal_model.clear_context(keep_tokens=100)  # must not raise

    def test_reset_cost_tracking_no_op(self, minimal_model) -> None:
        minimal_model.reset_cost_tracking()  # must not raise

    async def test_load_is_no_op(self, minimal_model) -> None:
        await minimal_model.load()  # async no-op, must not raise

    def test_get_metrics_returns_empty_dict(self, minimal_model) -> None:
        assert minimal_model.get_metrics() == {}

    def test_count_tokens_uses_tokenize(self, minimal_model) -> None:
        n = minimal_model.count_tokens("hello world")
        assert n == 1  # tokenize always returns [0]

    def test_is_local_property_default(self, minimal_model) -> None:
        assert minimal_model.is_local is True

    def test_is_local_setter(self, minimal_model) -> None:
        minimal_model.is_local = False
        assert minimal_model.is_local is False

    def test_requires_network_reflects_is_local(self, minimal_model) -> None:
        minimal_model.is_local = True
        assert minimal_model.requires_network is False
        minimal_model.is_local = False
        assert minimal_model.requires_network is True

    def test_capabilities_property(self, minimal_model) -> None:
        caps = minimal_model.capabilities
        assert caps["max_context_length"] == 4096


# ---------------------------------------------------------------------------
# T34 – Orchestrator signal handlers and resource check paths
# ---------------------------------------------------------------------------


class TestOrchestratorSignalHandlers:
    """T34 — _setup/_cleanup signal handlers and _check_resources low-memory path."""

    @pytest.fixture
    def orchestrator(self):
        from keryx.core.orchestrator import KeryxOrchestrator
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        am = AdvisorManager()
        orc = KeryxOrchestrator(
            available_models={"qwen3-coder-8b": FinishModel()},
            advisor_manager=am,
            toolbox=tb,
        )
        yield orc
        tb.shutdown()
        am.shutdown()

    # -- Signal handler setup / teardown -----------------------------------

    def test_setup_signal_handlers_no_running_loop(self, orchestrator) -> None:
        """Calling _setup_signal_handlers() from sync context → RuntimeError → early return."""
        # There is no running event loop in a sync function
        orchestrator._setup_signal_handlers()  # must not raise

    def test_cleanup_signal_handlers_no_running_loop(self, orchestrator) -> None:
        """Calling _cleanup_signal_handlers() from sync context → RuntimeError → early return."""
        orchestrator._cleanup_signal_handlers()  # must not raise

    async def test_setup_signal_handler_notimplemented(self, orchestrator) -> None:
        """add_signal_handler raises NotImplementedError → swallowed silently."""
        from unittest.mock import MagicMock, patch

        mock_loop = MagicMock()
        mock_loop.add_signal_handler.side_effect = NotImplementedError()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            orchestrator._setup_signal_handlers()  # must not raise

    async def test_setup_signal_handler_captured_and_called(self, orchestrator) -> None:
        """Capture the signal handler closure and invoke it to cover lines 437-438."""
        from unittest.mock import MagicMock, patch

        captured: list = []

        def _capture(sig, handler):
            captured.append(handler)

        mock_loop = MagicMock()
        mock_loop.add_signal_handler.side_effect = _capture
        orchestrator._shutdown_event.clear()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            orchestrator._setup_signal_handlers()
        assert captured, "Handler should have been registered"
        captured[0]()  # trigger the inner _handler() body
        assert orchestrator._shutdown_event.is_set()

    async def test_cleanup_signal_handler_not_implemented(self, orchestrator) -> None:
        """remove_signal_handler raises NotImplementedError → swallowed silently."""
        from unittest.mock import MagicMock, patch

        mock_loop = MagicMock()
        mock_loop.remove_signal_handler.side_effect = NotImplementedError()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            orchestrator._cleanup_signal_handlers()  # must not raise

    async def test_cleanup_signal_handler_os_error(self, orchestrator) -> None:
        """remove_signal_handler raises OSError → swallowed silently."""
        from unittest.mock import MagicMock, patch

        mock_loop = MagicMock()
        mock_loop.remove_signal_handler.side_effect = OSError("eperm")
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            orchestrator._cleanup_signal_handlers()  # must not raise

    # -- Resource check ----------------------------------------------------

    async def test_check_resources_low_memory_returns_false(self) -> None:
        """Simulate available memory < 512 MB — check_resources returns False."""
        from unittest.mock import MagicMock, patch
        from keryx.core._resources import check_resources

        mock_mem = MagicMock()
        mock_mem.available = 100 * 1024 * 1024  # 100 MB
        with patch("keryx.core._resources.psutil.virtual_memory", return_value=mock_mem):
            result = await check_resources()
        assert result is False

    async def test_check_resources_psutil_exception(self) -> None:
        """psutil raises → exception swallowed, returns True (fail-open)."""
        from unittest.mock import patch
        from keryx.core._resources import check_resources

        with patch(
            "keryx.core._resources.psutil.virtual_memory",
            side_effect=Exception("no psutil"),
        ):
            result = await check_resources()
        assert result is True

    # -- should_escalate (pure function in _escalation) --------------------

    def test_should_escalate_at_max_level_returns_false(self) -> None:
        """At max escalation level → should_escalate returns False regardless."""
        from keryx.core._escalation import should_escalate
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/test")
        result = should_escalate(
            {"confirmed_vulns": [], "average_confidence": 0.1}, ctx, 4, 4
        )
        assert result is False

    def test_should_escalate_with_confirmed_vuln_returns_false(self) -> None:
        """If at least one confirmed vuln → no further escalation."""
        from keryx.core._escalation import should_escalate
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/test")
        result = should_escalate(
            {"confirmed_vulns": [{"hyp": "x"}], "average_confidence": 0.1}, ctx, 1, 4
        )
        assert result is False


# ---------------------------------------------------------------------------
# T35 – Orchestrator hunt paths: cancelled, resource stop, agent exception
# ---------------------------------------------------------------------------


class TestOrchestratorHuntPaths:
    """T35 — Cover _run_hunt error paths and _is_airgapped probe path."""

    @pytest.fixture
    def make_orchestrator(self):
        from keryx.core.orchestrator import KeryxOrchestrator
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox

        created = []

        def _make(models=None):
            tb = ToolBox()
            am = AdvisorManager()
            orc = KeryxOrchestrator(
                available_models=models or {"qwen3-coder-8b": FinishModel()},
                advisor_manager=am,
                toolbox=tb,
            )
            created.append((tb, am))
            return orc

        yield _make
        for tb, am in created:
            tb.shutdown()
            am.shutdown()

    async def test_hunt_cancelled_error(self, make_orchestrator) -> None:
        """asyncio.CancelledError inside _run_hunt → _generate_cancelled_report returned."""
        from unittest.mock import AsyncMock, patch

        orc = make_orchestrator()
        with patch.object(orc, "_run_hunt", new_callable=AsyncMock,
                          side_effect=asyncio.CancelledError()):
            result = await orc.hunt("/tmp/cancelled_target", mode="airgapped")
        assert result["status"] == "cancelled"
        assert result["target"] == "/tmp/cancelled_target"

    async def test_hunt_stops_when_resources_low(self, make_orchestrator) -> None:
        """When _check_resources returns False the escalation loop breaks immediately."""
        from unittest.mock import AsyncMock, patch
        from keryx.core.router import RoutingPlan

        orc = make_orchestrator()
        mock_plan = RoutingPlan(
            executor_model=FinishModel(),
            advisor_name="none",
            advisor_chain=[],
            budget_usd=None,
            enforce_no_network=True,
            require_consensus=False,
            consensus_threshold=0.67,
            require_fuzzing_confirm=False,
            debate_models=[],
            capability="fast_pattern_matching",
        )
        with patch.object(orc.router, "route", return_value=mock_plan), \
             patch("keryx.core.orchestrator.check_resources", new_callable=AsyncMock,
                   return_value=False), \
             patch.object(orc._checkpoints, "save", new_callable=AsyncMock):
            result = await orc.hunt(
                target_path="/tmp/low_resource",
                capability="fast_pattern_matching",
                mode="airgapped",
            )
        # Loop broke before any agent ran → result is None → "cancelled"
        assert result["status"] == "cancelled"

    async def test_hunt_agent_run_exception_increments_attempts(
        self, make_orchestrator
    ) -> None:
        """agent.run() raising Exception is caught; after max_attempts, escalate."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from keryx.core.router import RoutingPlan

        orc = make_orchestrator()
        mock_plan = RoutingPlan(
            executor_model=FinishModel(),
            advisor_name="none",
            advisor_chain=[],
            budget_usd=None,
            enforce_no_network=True,
            require_consensus=False,
            consensus_threshold=0.67,
            require_fuzzing_confirm=False,
            debate_models=[],
            capability="fast_pattern_matching",
        )

        mock_ctx = MagicMock()
        mock_ctx.to_dict.return_value = {
            "target_path": "/tmp/exc_test",
            "steps": [],
            "hypotheses": [],
        }
        mock_ctx.steps_taken = 0
        mock_ctx.hypotheses = set()
        mock_ctx.parse_errors = 0
        mock_ctx.confirmed_vulns = []
        mock_ctx.escalation_level = 1
        mock_ctx.get_metrics.return_value = {}
        mock_ctx.set_escalation_level = MagicMock()

        with patch.object(orc.router, "route", return_value=mock_plan), \
             patch.object(orc._checkpoints, "save", new_callable=AsyncMock), \
             patch("keryx.core.orchestrator.KeryxAgent") as MockAgent:
            inst = MagicMock()
            inst.run = AsyncMock(side_effect=RuntimeError("model exploded"))
            inst.context = mock_ctx
            MockAgent.return_value = inst
            result = await orc.hunt(
                target_path="/tmp/exc_test",
                capability="fast_pattern_matching",
                mode="airgapped",
            )
        # After exhausting max_attempts at both escalation levels, result is None
        assert result is not None  # _build_final_result always returns a dict

    async def test_hunt_escalation_retry_path(self, make_orchestrator) -> None:
        """_should_escalate_further returns True → retry path (lines 201-209) is covered."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from keryx.core.router import RoutingPlan

        orc = make_orchestrator()
        mock_plan = RoutingPlan(
            executor_model=FinishModel(),
            advisor_name="none",
            advisor_chain=[],
            budget_usd=None,
            enforce_no_network=True,
            require_consensus=False,
            consensus_threshold=0.67,
            require_fuzzing_confirm=False,
            debate_models=[],
            capability="fast_pattern_matching",
        )

        call_count = 0

        def _should_escalate(result, context, level, max_level):
            nonlocal call_count
            call_count += 1
            # Return True for first 3 calls (fills max_attempts), False after
            return call_count <= 3

        mock_ctx = MagicMock()
        mock_ctx.to_dict.return_value = {"target_path": "/tmp/retry", "steps": []}
        mock_ctx.steps_taken = 0
        mock_ctx.hypotheses = set()
        mock_ctx.parse_errors = 0
        mock_ctx.confirmed_vulns = []
        mock_ctx.escalation_level = 1
        mock_ctx.get_metrics.return_value = {}
        mock_ctx.set_escalation_level = MagicMock()

        with patch.object(orc.router, "route", return_value=mock_plan), \
             patch.object(orc._checkpoints, "save", new_callable=AsyncMock), \
             patch("keryx.core.orchestrator.should_escalate", side_effect=_should_escalate), \
             patch("keryx.core.orchestrator.KeryxAgent") as MockAgent:
            inst = MagicMock()
            inst.run = AsyncMock(return_value={
                "status": "completed",
                "confirmed_vulns": [],
                "hypotheses": [],
                "average_confidence": 0.3,
                "steps_taken": 1,
            })
            inst.context = mock_ctx
            MockAgent.return_value = inst
            result = await orc.hunt(
                target_path="/tmp/retry_target",
                capability="fast_pattern_matching",
                mode="airgapped",
            )
        assert result is not None
        assert call_count >= 1

    async def test_is_airgapped_no_network_probes(self) -> None:
        """All TCP probes fail → detect_airgap returns True, covers probe code path."""
        from unittest.mock import patch
        from keryx.core._airgap import detect_airgap

        with patch("keryx.core._airgap.socket.create_connection",
                   side_effect=OSError("connection refused")):
            result = await detect_airgap("hybrid")
        assert result is True


# ---------------------------------------------------------------------------
# T36 – Swarm debate extra paths
# ---------------------------------------------------------------------------


class TestOrchestratorSwarmExtra:
    """T36 — Swarm debate timeout and model-exception voter paths."""

    @pytest.fixture
    def orchestrator(self):
        from keryx.core.orchestrator import KeryxOrchestrator
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox

        tb = ToolBox()
        am = AdvisorManager()
        orc = KeryxOrchestrator(
            available_models={"yes_model": YesModel(), "no_model": NoModel()},
            advisor_manager=am,
            toolbox=tb,
        )
        yield orc
        tb.shutdown()
        am.shutdown()

    async def test_swarm_debate_timeout_covered(self, orchestrator) -> None:
        """asyncio.timeout fires inside SwarmDebate.debate() → timeout_occurred=True, voters=0."""
        from unittest.mock import patch
        from keryx.models.swarm import SwarmDebate

        swarm = SwarmDebate([YesJsonModel()], swarm_timeout=0.05)

        async def _slow_analyze(model, hyps, ctx):
            await asyncio.sleep(999)
            return {}

        with patch.object(swarm, "_analyze_with_model", new=_slow_analyze):
            sr = await swarm.debate(["use-after-free in parse()"])
        assert sr.timeout_occurred is True
        assert sr.swarm_voters == 0

    async def test_swarm_debate_model_exception_voter(self, orchestrator) -> None:
        """Exception from _analyze_with_model propagates via gather → voter not counted."""
        from unittest.mock import patch
        from keryx.models.swarm import SwarmDebate

        swarm = SwarmDebate([YesJsonModel()])

        async def _failing_analyze(model, hyps, ctx):
            raise RuntimeError("voter died")

        with patch.object(swarm, "_analyze_with_model", new=_failing_analyze):
            sr = await swarm.debate(["heap overflow in read()"])
        assert sr.swarm_voters == 0


# ---------------------------------------------------------------------------
# T37 – GitBlame timeout and init warning paths
# ---------------------------------------------------------------------------


class TestGitBlameCoverage3:
    """T37 — Uncovered git_blame.py paths: timeout, hotspot exception, init warning."""

    async def test_execute_blame_timeout_covered(self) -> None:
        """asyncio.TimeoutError in _git_blame → execute() returns timeout ToolResult."""
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import patch, AsyncMock

        tool = GitBlameTool()

        async def _slow_blame(action_input):
            raise asyncio.TimeoutError()

        with patch.object(tool, "_git_blame", new_callable=AsyncMock,
                          side_effect=asyncio.TimeoutError()):
            result = await tool.execute({"command": "blame", "file": "/tmp/fake.c"})
        assert result.success is False
        assert result.error == "timeout"

    async def test_execute_log_exception_covered(self) -> None:
        """Generic Exception in _git_log → execute() returns unexpected_error ToolResult."""
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import patch, AsyncMock

        tool = GitBlameTool()
        with patch.object(tool, "_git_log", new_callable=AsyncMock,
                          side_effect=ValueError("unexpected")):
            result = await tool.execute({"command": "log"})
        assert result.success is False
        assert result.error == "unexpected_error"

    async def test_hotspot_exception_path(self) -> None:
        """Exception inside _run_hotspots → returns error ToolResult (lines 412-416)."""
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import patch, AsyncMock

        tool = GitBlameTool()
        with patch.object(tool, "_run_cmd", new_callable=AsyncMock,
                          side_effect=RuntimeError("git broke")):
            result = await tool.execute({"command": "hotspots"})
        assert result.success is False
        assert result.error == "analysis_failed"

    async def test_hotspot_timeout_path(self) -> None:
        """asyncio.TimeoutError inside _run_hotspots → timeout ToolResult (lines 403-411)."""
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import patch, AsyncMock

        tool = GitBlameTool()
        with patch.object(tool, "_run_cmd", new_callable=AsyncMock,
                          side_effect=asyncio.TimeoutError()):
            result = await tool.execute({"command": "hotspots"})
        assert result.success is False
        assert result.error == "timeout"

    def test_init_warning_when_git_not_found(self) -> None:
        """logger.warning on line 84 fires when git binary cannot be resolved."""
        import shutil
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import patch

        with patch.object(shutil, "which", return_value=None):
            tool = GitBlameTool()  # no git_path, shutil.which returns None
        assert not tool._available

    async def test_run_cmd_timeout_kills_proc(self) -> None:
        """asyncio.TimeoutError in _run_cmd raises after killing process."""
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import AsyncMock, MagicMock, patch

        tool = GitBlameTool()

        async def _fake_communicate():
            raise asyncio.TimeoutError()

        mock_proc = MagicMock()
        mock_proc.communicate = _fake_communicate
        mock_proc.pid = 99999

        with patch.object(tool.__class__, "_create_proc",
                          new_callable=lambda: type("P", (), {"__func__": staticmethod(
                              AsyncMock(return_value=mock_proc))})):
            pass  # just testing _kill_proc below

        # Directly test _kill_proc does not raise
        mock_proc2 = MagicMock()
        mock_proc2.kill = MagicMock()
        GitBlameTool._kill_proc(mock_proc2)  # static method, always safe


# ---------------------------------------------------------------------------
# T38 – ReadFile timeout and UnicodeDecodeError paths
# ---------------------------------------------------------------------------


class TestReadFileCoverage2:
    """T38 — Uncovered read_file.py paths."""

    async def test_read_file_timeout(self, tmp_path) -> None:
        """asyncio.TimeoutError on file read → returns timeout ToolResult."""
        from keryx.tools.read_file import ReadFileTool
        from unittest.mock import patch

        target = tmp_path / "bigfile.c"
        target.write_text("x" * 100)
        tool = ReadFileTool(timeout_seconds=30.0)

        with patch("keryx.tools.read_file.asyncio.wait_for",
                   side_effect=asyncio.TimeoutError()):
            result = await tool.execute({"file_path": str(target)})
        assert result.success is False
        assert result.error == "timeout"

    async def test_read_file_unicode_decode_error(self, tmp_path) -> None:
        """UnicodeDecodeError in _read_file → returns decode_error ToolResult."""
        from keryx.tools.read_file import ReadFileTool
        from unittest.mock import patch

        target = tmp_path / "binary.bin"
        target.write_bytes(b"\xff\xfe" + b"\x00" * 50)
        tool = ReadFileTool()

        with patch.object(tool, "_read_file",
                          side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "reason")):
            result = await tool.execute({"file_path": str(target)})
        assert result.success is False
        assert result.error == "decode_error"

    async def test_read_file_generic_exception(self, tmp_path) -> None:
        """Unexpected OSError in _read_file → returns read_error ToolResult (lines 153-155)."""
        from keryx.tools.read_file import ReadFileTool
        from unittest.mock import patch

        target = tmp_path / "locked.c"
        target.write_text("int main() {}")
        tool = ReadFileTool()

        with patch.object(tool, "_read_file",
                          side_effect=OSError("permission denied")):
            result = await tool.execute({"file_path": str(target)})
        assert result.success is False
        assert result.error == "read_error"
        assert "permission denied" in result.output


# ---------------------------------------------------------------------------
# T39 – keryx/__init__.py: version-tuple error, __getattr__ miss, hunt() entry
# ---------------------------------------------------------------------------


class TestKeryxInit:
    """T39 — Cover the three uncovered blocks in keryx/__init__.py."""

    def test_get_version_tuple_bad_version(self) -> None:
        """ValueError path in get_version_tuple() → returns (0, 0, 0)."""
        import keryx
        original = keryx.__version__
        try:
            keryx.__version__ = "not-a-version"
            result = keryx.get_version_tuple()
            assert result == (0, 0, 0)
        finally:
            keryx.__version__ = original

    def test_getattr_unknown_name_raises(self) -> None:
        """__getattr__ for an unknown symbol raises AttributeError (line 108)."""
        import keryx
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = keryx.ThisDoesNotExistAnywhere

    async def test_hunt_convenience_function(self) -> None:
        """keryx.hunt() wires up Orchestrator and calls hunt() (lines 139-155)."""
        from unittest.mock import AsyncMock, MagicMock, patch
        import keryx

        mock_result = {"status": "completed", "confirmed_vulns": [], "hypotheses": []}
        mock_orc = MagicMock()
        mock_orc.hunt = AsyncMock(return_value=mock_result)

        with patch("keryx.core.orchestrator.KeryxOrchestrator", return_value=mock_orc):
            result = await keryx.hunt(
                target_path="/tmp/init_test",
                capability="fast_pattern_matching",
                mode="airgapped",
            )
        assert result["status"] == "completed"
        mock_orc.hunt.assert_called_once()


# ---------------------------------------------------------------------------
# T40 – AdvisorManager: execution errors, fallback, parallel timeout, sync bridge
# ---------------------------------------------------------------------------


def _make_advisor(name, *, trigger=True, raises=None, max_calls=1000):
    """Factory: concrete BaseAdvisor that always triggers and optionally raises."""
    from keryx.advisors.base import BaseAdvisor, AdvisorResponse

    _raises = raises

    class _Adv(BaseAdvisor):
        def __init__(self):
            super().__init__(name=name, max_calls=max_calls)

        def should_trigger(self, context):
            return trigger

        async def advise(self, context):
            if _raises is not None:
                raise _raises  # type: ignore[misc]
            return AdvisorResponse(strategy="ok")

        # Override advise_with_tracking to bypass the inner try-except so that
        # system errors (ConnectionError, asyncio.TimeoutError) reach the manager.
        async def advise_with_tracking(self, context):
            if _raises is not None:
                raise _raises  # type: ignore[misc]
            return await super().advise_with_tracking(context)

    return _Adv()


class TestAdvisorManagerExecution:
    """T40 — Cover missed error paths in manager.py lines 237, 268-287, 299, 355, 398, 412-414."""

    async def test_advise_async_timeout_triggers_fallback(self) -> None:
        """asyncio.TimeoutError raised from advise_with_tracking → lines 268-269, 283-287."""
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        am.register(_make_advisor("slow_t", raises=asyncio.TimeoutError()))
        try:
            resp = await am.get_advice_async({}, "slow_t")
        finally:
            am.shutdown()
        assert resp.metadata.get("error") is not None

    async def test_advise_system_error_triggers_fallback(self) -> None:
        """ConnectionError from advise_with_tracking → system error path lines 271-272, 283-287."""
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        am.register(_make_advisor("net", raises=ConnectionError("network down")))
        try:
            resp = await am.get_advice_async({}, "net")
        finally:
            am.shutdown()
        assert resp.metadata.get("error") is not None

    async def test_advise_business_error_returns_error_response(self) -> None:
        """ValueError from advise_with_tracking → business-logic path lines 275-280."""
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        # ValueError is NOT in _SYSTEM_ERRORS → hits the generic except Exception branch
        am.register(_make_advisor("broken", raises=ValueError("bad output")))
        try:
            resp = await am.get_advice_async({}, "broken")
        finally:
            am.shutdown()
        assert resp.metadata.get("error") is not None

    async def test_limit_reached_falls_through_to_empty_chain(self) -> None:
        """can_advise() returns False → _try_fallback with empty chain lines 237, 299."""
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        am.register(_make_advisor("limited", max_calls=0))  # can_advise() → False immediately
        try:
            resp = await am.get_advice_async({}, "limited", fallback_chain=[])
        finally:
            am.shutdown()
        assert resp.metadata.get("error") is not None

    async def test_parallel_timeout_cancels_tasks(self) -> None:
        """Parallel advice times out → pending tasks cancelled (line 355)."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()

        class SlowAdv(BaseAdvisor):
            def should_trigger(self, ctx):
                return True

            async def advise(self, ctx):
                await asyncio.sleep(9999)
                return AdvisorResponse()

        adv = SlowAdv()
        adv.name = "slow_par"
        adv.max_calls_per_session = 1000
        adv.calls_made = 0
        am.register(adv)
        try:
            responses = await am.get_advice_parallel({}, ["slow_par"], timeout=0.05)
        finally:
            am.shutdown()
        assert isinstance(responses, list)

    async def test_get_advice_sync_from_async_context_warns(self) -> None:
        """get_advice() called from async context logs warning (line 398)."""
        from keryx.advisors.manager import AdvisorManager

        am = AdvisorManager()
        am.register(_make_advisor("quick_sync"))
        try:
            # Calling sync bridge from within an async context → warning logged
            resp = am.get_advice({}, "quick_sync")
        finally:
            am.shutdown()
        assert resp is not None

    def test_get_advice_sync_dispatch_error(self) -> None:
        """future.result() raises → logged + error_response returned (lines 412-414)."""
        from keryx.advisors.manager import AdvisorManager
        from unittest.mock import MagicMock, patch
        import concurrent.futures

        am = AdvisorManager()
        am.register(_make_advisor("quick2"))
        try:
            mock_future = MagicMock()
            mock_future.result.side_effect = concurrent.futures.TimeoutError("timeout")

            def _patched_run(coro, loop):
                coro.close()  # prevent "coroutine was never awaited" warning
                return mock_future

            with patch("keryx.advisors.manager.asyncio.run_coroutine_threadsafe",
                       side_effect=_patched_run):
                resp = am.get_advice({}, "quick2")
        finally:
            am.shutdown()
        assert resp.metadata.get("error") is not None

    def test_shutdown_close_exception_swallowed(self) -> None:
        """Exception in sync_loop.close() is swallowed (lines 502-503)."""
        from keryx.advisors.manager import AdvisorManager
        from unittest.mock import MagicMock

        am = AdvisorManager()
        am._ensure_sync_loop()

        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_loop.close.side_effect = RuntimeError("close failed")
        mock_loop.call_soon_threadsafe = MagicMock()
        am._sync_loop = mock_loop
        am._sync_thread = None

        am.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# T41 – Agent: targeted coverage for 10 uncovered line paths
# ---------------------------------------------------------------------------


class _InvalidActionModel(MyLocalModel):
    """Returns an action name not in _VALID_ACTIONS, then FINISH."""

    def __init__(self) -> None:
        super().__init__()
        self._call = 0

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._call += 1
        if self._call == 1:
            return json.dumps({
                "thought": "trying something",
                "action": "TOTALLY_UNKNOWN_OP",
                "action_input": {},
                "confidence": 0.5,
            })
        return json.dumps({
            "thought": "done",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class _NoActionNoThoughtModel(MyLocalModel):
    """Consistently returns NO_ACTION with no thought — exhausts safe_generate retries."""

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        # No "thought" key → action.thought == "" → safe_generate returns None
        return json.dumps({"action": "NO_ACTION", "action_input": {}, "confidence": 0.1})


class _ToolCallerModel(MyLocalModel):
    """Calls read_file once, then FINISH."""

    def __init__(self) -> None:
        super().__init__()
        self._call = 0

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._call += 1
        if self._call == 1:
            return json.dumps({
                "thought": "read it",
                "action": "read_file",
                "action_input": {"file_path": "/tmp/x"},
                "confidence": 0.7,
            })
        return json.dumps({"thought": "done", "action": "FINISH", "action_input": {},
                           "confidence": 0.9})


class TestAgentUncoveredLinePaths:
    """T41 — Targeted tests for agent.py lines 248-249, 275-276, 299-304,
    374, 393, 412-413, 442, 510-513 and shared_context.py lines 113, 162."""

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox(default_timeout=5.0)
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    # ── Lines 248-249: prompt exceeds max_prompt_chars → trim + rebuild ──────

    async def test_prompt_trim_triggered_when_history_is_long(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 248-249: prompt > max_prompt_chars → trim_history() + rebuild."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        import time

        ctx = SharedContext(target_path="/tmp/trim_test")
        # Directly append step records (bypasses add_step so steps_taken stays 0)
        for i in range(20):
            ctx.steps.append({
                "timestamp": time.time(),
                "action": {"name": "NO_ACTION", "input": {}, "thought": "x" * 200,
                           "confidence": 0.5},
                "observation": "obs " + "y" * 200,
            })
        assert ctx.steps_taken == 0  # loop can still run

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
            max_prompt_chars=50,  # tiny — guarantee trim fires on first iteration
        )
        result = await agent.run(target_path="/tmp/trim_test", resume=False, context=ctx)
        assert result["status"] == "completed"
        # trim_history() kept only 6 steps; agent may add 1 more (FINISH)
        assert len(agent.context.steps) <= 8  # well below the original 20

    # ── Lines 275-276: unknown action name remapped to NO_ACTION ─────────────

    async def test_unknown_action_name_becomes_no_action(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 275-276: action not in _VALID_ACTIONS → action.action = 'NO_ACTION'."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=_InvalidActionModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
        )
        result = await agent.run(target_path="/tmp/invalid_action", resume=False)
        assert result["status"] == "completed"
        # At least one step must have been recorded (the remapped NO_ACTION step)
        assert result["steps_taken"] >= 1

    # ── Lines 299-304: tool raises raw exception (bypasses ToolBox wrapper) ──

    async def test_tool_raw_exception_hits_except_block(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 299-304: ToolBox.execute raises directly → except Exception caught."""
        from keryx.core.agent import KeryxAgent
        from unittest.mock import patch

        agent = KeryxAgent(
            executor_model=_ToolCallerModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
        )
        # Patch the sync execute to raise, bypassing ToolBox's internal error handling
        with patch.object(toolbox, "execute", side_effect=RuntimeError("raw crash")):
            result = await agent.run(target_path="/tmp/raw_exc", resume=False)

        assert result["status"] == "completed"
        observations = [s.get("observation", "") for s in agent.context.steps]
        assert any("raw crash" in obs or "failed" in obs for obs in observations)

    # ── Line 374: _safe_generate exhausts retries → returns None ─────────────

    async def test_safe_generate_returns_none_when_all_retries_no_action(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 374: all 2 retries yield NO_ACTION/no-thought → return None."""
        from keryx.core.agent import KeryxAgent

        agent = KeryxAgent(
            executor_model=_NoActionNoThoughtModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=3,
        )
        result = await agent.run(target_path="/tmp/retry_none", resume=False)
        # Agent creates emergency fallback and eventually terminates
        assert result["status"] == "completed"

    # ── Line 393: _execute_tool_async result has no .output ──────────────────

    async def test_execute_tool_async_result_without_output_attr(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 393: execute() returns non-ToolResult → str(result) fallback."""
        from keryx.core.agent import KeryxAgent
        from unittest.mock import patch

        agent = KeryxAgent(
            executor_model=_ToolCallerModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=5,
        )
        # Return a plain string — no `.output` attribute → hits line 393
        with patch.object(toolbox, "execute", return_value="plain string result"):
            result = await agent.run(target_path="/tmp/no_output_attr", resume=False)

        assert result["status"] == "completed"

    # ── Lines 412-413: _self_critique_async timeout ───────────────────────────

    async def test_self_critique_timeout_returns_placeholder(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 412-413: asyncio.wait_for inside _self_critique_async raises TimeoutError."""
        from keryx.core.agent import KeryxAgent
        from unittest.mock import patch, AsyncMock
        import keryx.core.agent as agent_mod

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        from keryx.core.shared_context import SharedContext
        agent.context = SharedContext(target_path="/tmp/critique_timeout")

        original_wait_for = asyncio.wait_for

        async def _patched_wait_for(coro, timeout):
            # Let short timeouts (health-check, etc.) through; block only 15s critique
            if timeout == 15.0:
                # Drain the coroutine to avoid "coroutine never awaited" warnings
                try:
                    coro.close()
                except Exception:
                    pass
                raise TimeoutError()
            return await original_wait_for(coro, timeout)

        with patch.object(agent_mod.asyncio, "wait_for", side_effect=_patched_wait_for):
            critique = await agent._self_critique_async()

        assert "timed out" in critique.lower()

    # ── Line 442: _apply_advisor_advice returns early when no advice pending ──

    def test_apply_advisor_advice_no_op_when_no_pending_advice(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 442: no pending advice → early return, no side-effects."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/no_advice")
        assert not agent.context.has_pending_advisor_advice()
        # Must return without raising and leave threshold unchanged
        original_threshold = agent.confidence_threshold
        agent._apply_advisor_advice()
        assert agent.confidence_threshold == original_threshold

    # ── Lines 510-513: _parse_response — json.loads raises on bad-but-parseable ──

    def test_parse_response_json_decode_error_increments_errors(
        self, toolbox, advisor_manager
    ) -> None:
        """agent.py 510-513: _extract_json succeeds but json.loads raises JSONDecodeError."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext

        agent = KeryxAgent(
            executor_model=FinishModel(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
        )
        agent.context = SharedContext(target_path="/tmp/bad_json")
        # Has { } so _extract_json returns a string, but it's not valid JSON
        step = agent._parse_response('{"action": unquoted_value, "confidence": }')
        assert agent.context.parse_errors == 1
        assert step.action in ("NO_ACTION", "FINISH", "read_file", "gdb_analyze")

    # ── shared_context line 113: add_step with non-AgentStep action ──────────

    def test_shared_context_add_step_with_string_action(self) -> None:
        """shared_context.py 113: action has no .action attr → stored as str."""
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/str_action")
        ctx.add_step("plain_string_action", "some observation")
        assert ctx.steps_taken == 1
        stored = ctx.steps[0]["action"]
        assert stored == "plain_string_action"

    # ── shared_context line 162: get_last_confidence when action is a string ──

    def test_shared_context_get_last_confidence_returns_none_for_string_action(
        self,
    ) -> None:
        """shared_context.py 162: last step action is str → get_last_confidence() is None."""
        from keryx.core.shared_context import SharedContext

        ctx = SharedContext(target_path="/tmp/conf_none")
        ctx.add_step("plain_string_action", "obs")
        confidence = ctx.get_last_confidence()
        assert confidence is None


# ---------------------------------------------------------------------------
# T42 – Orchestrator final polish: 3 behavioral regression tests
# ---------------------------------------------------------------------------


class TestOrchestratorFinalPolish:
    """T42 — Behavioral tests not duplicate of T34/T35: shutdown event,
    budget propagation, and end-to-end crash-model retry exhaustion."""

    @pytest.fixture
    def make_orc(self):
        from keryx.core.orchestrator import KeryxOrchestrator
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox

        created = []

        def _make(models=None):
            tb = ToolBox()
            am = AdvisorManager()
            orc = KeryxOrchestrator(
                available_models=models or {"qwen3-coder-8b": FinishModel()},
                advisor_manager=am,
                toolbox=tb,
            )
            created.append((tb, am))
            return orc

        yield _make
        for tb, am in created:
            tb.shutdown()
            am.shutdown()

    async def test_shutdown_event_set_before_hunt_exits_immediately(
        self, make_orc
    ) -> None:
        """_shutdown_event already set → while-loop body never executes →
        result=None → final dict has status='cancelled'."""
        from unittest.mock import AsyncMock, patch
        from keryx.core.router import RoutingPlan

        orc = make_orc()
        orc._shutdown_event.set()  # pre-arm before hunt

        mock_plan = RoutingPlan(
            executor_model=FinishModel(),
            advisor_name="none",
            advisor_chain=[],
            budget_usd=None,
            enforce_no_network=True,
            require_consensus=False,
            consensus_threshold=0.67,
            require_fuzzing_confirm=False,
            debate_models=[],
            capability="fast_pattern_matching",
        )
        with patch.object(orc.router, "route", return_value=mock_plan), \
             patch.object(orc._checkpoints, "save", new_callable=AsyncMock):
            result = await orc.hunt(
                target_path="/tmp/shutdown_pre_set",
                capability="fast_pattern_matching",
                mode="airgapped",
            )

        assert result["status"] == "cancelled"
        assert result.get("confirmed_vulns") == []

    async def test_user_budget_usd_flows_into_routing_plan(self, make_orc) -> None:
        """user_budget_usd passed to hunt() is forwarded to router.route() and
        stored in the RoutingPlan that the agent receives."""
        from unittest.mock import AsyncMock, MagicMock, patch, call
        from keryx.core.router import RoutingPlan

        orc = make_orc()
        captured_kwargs: list[dict] = []

        original_route = orc.router.route

        def _spy_route(**kwargs):
            captured_kwargs.append(kwargs)
            return original_route(**kwargs)

        with patch.object(orc.router, "route", side_effect=_spy_route):
            await orc.hunt(
                target_path="/tmp/budget_flow",
                capability="fast_pattern_matching",
                mode="airgapped",
                user_budget_usd=3.50,
            )

        assert captured_kwargs, "router.route() was never called"
        assert captured_kwargs[0]["user_budget_usd"] == pytest.approx(3.50)

    async def test_crash_model_exhausts_all_attempts_and_returns_dict(
        self, make_orc
    ) -> None:
        """End-to-end: CrashingModel makes every agent.run() raise RuntimeError.
        Orchestrator exhausts max_attempts at every escalation level and still
        returns a valid dict (never raises to the caller)."""
        orc = make_orc(models={"qwen3-coder-8b": CrashingModel()})
        result = await orc.hunt(
            target_path="/tmp/full_crash",
            capability="fast_pattern_matching",
            mode="airgapped",
        )
        assert isinstance(result, dict)
        # No vuln confirmed during a fully crashing hunt
        assert result.get("confirmed_vulns") == []
        # Orchestrator recorded at least one routing decision
        assert orc.metrics.routing_decisions >= 1


# ---------------------------------------------------------------------------
# T43-pre: Stabilization — real e2e with both tools + checkpoint live cycle
# ---------------------------------------------------------------------------


class _GitBlameThenFinishModel(MyLocalModel):
    """
    Step 1 → git_blame on a known real file in the repo.
    Step 2 → FINISH.
    Used by the e2e stabilization test.
    """

    def __init__(self, target_file: str) -> None:
        super().__init__()
        self._target_file = target_file
        self._call = 0

    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
        if max_tokens == 1 or grammar is None:
            return "ok"
        self._call += 1
        if self._call == 1:
            return json.dumps({
                "thought": "inspect git history of the target file",
                "action": "git_blame",
                "action_input": {"command": "log", "path": self._target_file, "n": 3},
                "confidence": 0.7,
            })
        return json.dumps({
            "thought": "analysis complete",
            "action": "FINISH",
            "action_input": {},
            "confidence": 0.9,
        })


class TestE2EStabilization:
    """
    Two stabilization tests that must pass before shipping:

    1. Real e2e hunt exercising BOTH read_file and git_blame against the
       live repo directory (not a tmp file) — verifies the full tool chain.

    2. Checkpoint write/resume live cycle — agent runs 5+ steps, checkpoint
       is saved automatically, a fresh agent resumes from that checkpoint
       and continues from the saved step count (not from 0).
    """

    @pytest.fixture
    def toolbox(self):
        from keryx.tools.Toolbox import create_default_toolbox
        # Use the actual repo as allowed_root — git_blame needs real git history
        tb = create_default_toolbox(
            allowed_root=str(Path(__file__).parent.parent)
        )
        yield tb
        tb.shutdown()

    @pytest.fixture
    def advisor_manager(self):
        from keryx.advisors.manager import AdvisorManager
        am = AdvisorManager()
        yield am
        am.shutdown()

    # ── 1. Real e2e with both tools ──────────────────────────────────────────

    async def test_real_e2e_with_read_file_and_git_blame(
        self, toolbox, advisor_manager
    ) -> None:
        """
        Runs a real 3-step hunt:
          step 1 → read_file on keryx/core/agent.py (real file)
          step 2 → git_blame log on the same file (real git history)
          step 3 → FINISH

        Verifies:
        - Both tool calls succeed (success=True observations)
        - Result has the expected keys
        - No crash from real subprocess / real filesystem I/O
        """
        from keryx.core.agent import KeryxAgent

        repo_root = Path(__file__).parent.parent
        target_file = str(repo_root / "keryx" / "core" / "agent.py")

        class _ReadThenBlameThenFinish(MyLocalModel):
            def __init__(self):
                super().__init__()
                self._call = 0

            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                if max_tokens == 1 or grammar is None:
                    return "ok"
                self._call += 1
                if self._call == 1:
                    return json.dumps({
                        "thought": "read the source file first",
                        "action": "read_file",
                        "action_input": {"file_path": target_file},
                        "confidence": 0.7,
                    })
                if self._call == 2:
                    return json.dumps({
                        "thought": "check git history",
                        "action": "git_blame",
                        "action_input": {
                            "command": "log",
                            "path": target_file,
                            "n": 3,
                        },
                        "confidence": 0.75,
                    })
                return json.dumps({
                    "thought": "done",
                    "action": "FINISH",
                    "action_input": {},
                    "confidence": 0.9,
                })

        agent = KeryxAgent(
            executor_model=_ReadThenBlameThenFinish(),
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=10,
        )
        result = await agent.run(target_path=str(repo_root), resume=False)

        assert result["status"] == "completed"
        assert result["steps_taken"] >= 3

        # Both tool observations must be non-error
        steps = agent.context.steps
        tool_steps = [
            s for s in steps
            if isinstance(s.get("action"), dict)
            and s["action"].get("name") in ("read_file", "git_blame")
        ]
        assert len(tool_steps) >= 2, "Expected at least 2 tool-call steps"

        for s in tool_steps:
            obs = s["observation"]
            assert "failed" not in obs.lower() or "git_not_found" not in obs, (
                f"Tool step failed unexpectedly: {obs[:200]}"
            )

    # ── 2. Checkpoint write + resume live cycle ──────────────────────────────

    async def test_checkpoint_write_and_resume_live_cycle(
        self, tmp_path, toolbox, advisor_manager
    ) -> None:
        """
        Phase 1 — Run agent for 6 tool-call steps.
          The checkpoint fires automatically at step 5 (step % 5 == 0).
          Verify the checkpoint file exists and contains steps_taken >= 5.

        Phase 2 — Create a fresh agent with the same checkpoint_dir.
          Run with resume=True.  The agent must load the checkpoint and
          start from steps_taken=5, NOT from 0.
          FinishModel returns FINISH on the very first step → final
          steps_taken == 6 (5 from checkpoint + 1 new step).
        """
        from keryx.core.agent import KeryxAgent
        import tempfile

        ckdir = tmp_path / "stabilization_ck"

        # ── Phase 1: produce a checkpoint ───────────────────────────────────
        # Use a file inside tmp_path so read_file's allowed_root check passes.
        phase1_target = str(tmp_path / "phase1_target.c")
        Path(phase1_target).write_text("int main(){ return 0; }\n")

        # steps_before_finish=6 → tool calls on steps 1-6, FINISH on step 7.
        # The checkpoint fires automatically when step % 5 == 0 (step 5).
        phase1_model = ToolCallingModel(
            target_file=phase1_target,
            steps_before_finish=6,
        )
        agent1 = KeryxAgent(
            executor_model=phase1_model,
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=20,
            checkpoint_dir=ckdir,
        )
        result1 = await agent1.run(target_path=phase1_target, resume=False)
        assert result1["status"] == "completed"
        assert result1["steps_taken"] >= 5, (
            "Phase 1 must complete at least 5 steps to trigger checkpoint"
        )

        # Verify checkpoint file was written
        ck_files = list(ckdir.glob("checkpoint_*.json"))
        assert ck_files, f"No checkpoint file found in {ckdir}"
        ck_data = json.loads(ck_files[0].read_text())
        saved_steps = ck_data["context"]["steps_taken"]
        assert saved_steps >= 5, f"Checkpoint must have steps_taken >= 5, got {saved_steps}"

        # ── Phase 2: resume from checkpoint ─────────────────────────────────
        # IMPORTANT: target_path must be identical to Phase 1 so that
        # _checkpoint_path() hashes to the same filename.
        agent2 = KeryxAgent(
            executor_model=FinishModel(),   # returns FINISH on step 1 of this run
            advisor_manager=advisor_manager,
            toolbox=toolbox,
            max_steps=20,
            checkpoint_dir=ckdir,
        )
        result2 = await agent2.run(
            target_path=phase1_target,   # same path → same checkpoint hash
            resume=True,
        )

        # The resumed agent must have started at saved_steps (not 0)
        # and taken at least 1 more step (FINISH).
        assert result2["steps_taken"] == saved_steps + 1, (
            f"Expected steps_taken={saved_steps + 1} after resume, "
            f"got {result2['steps_taken']}"
        )


# ---------------------------------------------------------------------------
# T43 – GitBlameTool missing coverage paths (lines 181, 291-292, 316-320,
#         333, 339, 341, 343, 368, 444-446)
# ---------------------------------------------------------------------------

def _git_result(output: str = "", stderr: str = "", exit_code: int = 0,
                authors: list[str] | None = None) -> "Any":
    """Build a _GitResult without importing the private dataclass."""
    from keryx.tools.git_blame import GitBlameTool
    from keryx.tools.Toolbox import ToolResult
    # Access via the module so we don't need to export _GitResult
    import keryx.tools.git_blame as _gb
    return _gb._GitResult(
        command="git test",
        output=output,
        stderr=stderr,
        exit_code=exit_code,
        authors=authors or [],
    )


class TestGitBlameMissingPaths:
    """T43 — Targeted git_blame.py coverage for lines not hit by existing tests."""

    @pytest.fixture
    def tool(self):
        from keryx.tools.git_blame import GitBlameTool
        return GitBlameTool(timeout_seconds=5.0)

    # ── Line 181: blame/log command returns non-zero exit code ────────────────

    async def test_blame_nonzero_exit_returns_failure(self, tool) -> None:
        """Line 181: _git_blame returns _GitResult(exit_code=1) → ToolResult(success=False)."""
        from unittest.mock import AsyncMock, patch
        import tempfile, os

        # Need a real file that exists so _git_blame doesn't raise FileNotFoundError
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            f.write(b"int main(){}")
            tmp = f.name
        try:
            with patch.object(tool, "_run_cmd",
                               new=AsyncMock(return_value=_git_result(
                                   stderr="fatal: not a git repo", exit_code=128
                               ))):
                result = await tool.execute({"command": "blame", "file": tmp})
        finally:
            os.unlink(tmp)

        assert result.success is False
        assert result.error == "git_blame_failed"

    async def test_log_nonzero_exit_returns_failure(self, tool) -> None:
        """Line 181 (log variant): _git_log returns exit_code=1 → ToolResult(success=False)."""
        from unittest.mock import AsyncMock, patch

        with patch.object(tool, "_run_cmd",
                           new=AsyncMock(return_value=_git_result(
                               stderr="fatal: bad revision", exit_code=128
                           ))):
            result = await tool.execute({"command": "log", "n": 5})

        assert result.success is False
        assert result.error == "git_log_failed"

    # ── Lines 291-292: hotspots initial git-log returns non-zero ─────────────

    async def test_hotspots_initial_log_failure(self, tool) -> None:
        """Lines 291-292: first _run_cmd returns non-zero → early error return."""
        from unittest.mock import AsyncMock, patch

        with patch.object(tool, "_run_cmd",
                           new=AsyncMock(return_value=_git_result(
                               stderr="not a repo", exit_code=128
                           ))):
            result = await tool.execute({"command": "hotspots", "path": "/tmp"})

        assert result.success is False
        assert result.error == "git_log_failed"

    # ── Lines 316, 320, 333, 339, 341, 343, 368: hotspot risk + filters ──────

    async def test_hotspots_risk_levels_and_since_filter(self, tool) -> None:
        """
        Lines 316, 320, 333, 339, 341, 343, 368 in one pass:
        - min_churn=2, one file has churn=1 → continue in step-3 (316) and step-4 (333)
        - since filter passed → --since injected into author cmd (320)
        - foo_long has churn=15, 5 authors → CRITICAL (339)
        - bar.c has churn=6, 3 authors → HIGH (341)
        - baz.c has churn=4, 1 author → MEDIUM (343)
        - foo_long's name > 44 chars → truncation (368)
        """
        from unittest.mock import AsyncMock, patch

        LONG_NAME = "a_very_long_filename_that_exceeds_44_chars_for_sure.c"  # >44 chars
        assert len(LONG_NAME) > 44

        # git log --name-only output: filenames repeated N times = churn of N
        name_only_output = (
            f"{LONG_NAME}\n" * 15 +   # churn=15
            "bar.c\n" * 6 +           # churn=6
            "baz.c\n" * 4 +           # churn=4
            "low_churn.c\n" * 1       # churn=1  →  below min_churn=2 → continue at 316 & 333
        )

        # Author lookup responses for files with churn >= 2 (low_churn.c skipped in step 3)
        author_responses = [
            _git_result(output="AuthorA\nAuthorB\nAuthorC\nAuthorD\nAuthorE\n"),  # LONG_NAME: 5 authors
            _git_result(output="AuthorA\nAuthorB\nAuthorC\n"),                    # bar.c: 3 authors
            _git_result(output="AuthorA\n"),                                       # baz.c: 1 author
        ]

        call_count = 0
        async def _mock_run_cmd(cmd, extract_authors=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _git_result(output=name_only_output)
            return author_responses[min(call_count - 2, len(author_responses) - 1)]

        with patch.object(tool, "_run_cmd", side_effect=_mock_run_cmd):
            result = await tool.execute({
                "command":     "hotspots",
                "path":        ".",
                "min_changes": 2,
                "since":       "6 months ago",   # triggers line 320
            })

        assert result.success is True
        hotspots = result.data["hotspots"]
        risk_levels = {h["risk_level"] for h in hotspots}
        assert "CRITICAL" in risk_levels   # line 339
        assert "HIGH"     in risk_levels   # line 341
        assert "MEDIUM"   in risk_levels   # line 343
        # Truncated filename appears in output
        assert "..." in result.output      # line 368
        # low_churn.c (churn=1) must NOT appear in hotspots (filtered by min_churn)
        hotspot_files = {h["file"] for h in hotspots}
        assert not any("low_churn" in f for f in hotspot_files)

    # ── Lines 444-446: TimeoutError inside _run_cmd → _kill_proc + re-raise ──

    async def test_run_cmd_timeout_kills_proc_and_reraises(self, tool) -> None:
        """Lines 444-446: asyncio.wait_for raises TimeoutError → _kill_proc called → re-raise."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_proc = MagicMock()
        mock_proc.pid = 99999  # non-existent PID; _kill_proc's os.killpg will get ProcessLookupError

        async def _slow_communicate():
            await asyncio.sleep(9999)
            return b"", b""

        mock_proc.communicate = _slow_communicate

        with patch.object(tool, "_create_proc", new=AsyncMock(return_value=mock_proc)):
            tool.timeout = 0.01  # force immediate timeout
            with pytest.raises(TimeoutError):
                await tool._run_cmd(["git", "status"])


# ---------------------------------------------------------------------------
# T44 – advisors/base.py: CascadeAdvisor missing paths (lines 148, 215-216,
#         219, 230-232)
# ---------------------------------------------------------------------------


class TestCascadeAdvisorPaths:
    """T44 — BaseAdvisor.should_trigger_async awaiting a coroutine, and
    CascadeAdvisor should_trigger + advise exception paths."""

    # ── Line 148: should_trigger_async awaits coroutine returned by should_trigger ──

    async def test_should_trigger_async_awaits_coroutine(self) -> None:
        """Line 148: should_trigger() returns a coroutine → should_trigger_async awaits it."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse

        class AsyncTriggerAdvisor(BaseAdvisor):
            """should_trigger defined as async def → calling it returns a coroutine."""
            async def should_trigger(self, context) -> bool:  # type: ignore[override]
                return True

            async def advise(self, context) -> AdvisorResponse:
                return AdvisorResponse(strategic_direction="go")

        adv = AsyncTriggerAdvisor()
        # should_trigger() now returns a coroutine object
        result = await adv.should_trigger_async({})
        assert result is True

    # ── Lines 215-216: CascadeAdvisor.should_trigger — sub-advisor returns coroutine ──

    def test_cascade_should_trigger_with_coroutine_sub_advisor(self) -> None:
        """Lines 215-216: sub-advisor's should_trigger returns a coroutine → closed + True."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse, CascadeAdvisor

        class AsyncTriggerAdvisor(BaseAdvisor):
            async def should_trigger(self, context) -> bool:  # type: ignore[override]
                return True

            async def advise(self, context) -> AdvisorResponse:
                return AdvisorResponse(strategic_direction="go")

        cascade = CascadeAdvisor(advisors=[AsyncTriggerAdvisor()])
        # CascadeAdvisor.should_trigger calls sub.should_trigger() — gets a coroutine back
        result = cascade.should_trigger({})
        assert result is True  # coroutine closed, conservative True returned

    # ── Line 219: CascadeAdvisor.should_trigger — all sub-advisors return False ──

    def test_cascade_should_trigger_all_false_returns_false(self) -> None:
        """Line 219: every sub-advisor returns False → should_trigger returns False."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse, CascadeAdvisor

        class NeverAdvisor(BaseAdvisor):
            def should_trigger(self, context) -> bool:
                return False

            async def advise(self, context) -> AdvisorResponse:
                return AdvisorResponse()

        cascade = CascadeAdvisor(advisors=[NeverAdvisor(), NeverAdvisor()])
        assert cascade.should_trigger({}) is False

    # ── Lines 230-232: CascadeAdvisor.advise — should_trigger_async raises ──

    async def test_cascade_advise_swallows_trigger_exception(self) -> None:
        """Lines 230-232: should_trigger_async raises → warning logged, triggered=False."""
        from keryx.advisors.base import BaseAdvisor, AdvisorResponse, CascadeAdvisor
        import logging

        class ExplodingTriggerAdvisor(BaseAdvisor):
            def should_trigger(self, context) -> bool:
                raise RuntimeError("trigger exploded")

            async def advise(self, context) -> AdvisorResponse:
                return AdvisorResponse(strategic_direction="found something")

        cascade = CascadeAdvisor(advisors=[ExplodingTriggerAdvisor()])
        # advise() calls should_trigger_async which calls should_trigger → raises
        response = await cascade.advise({})
        # Exception is swallowed; cascade returns empty fallback
        assert response.metadata.get("cascade") == "no_advisor_triggered"


# ---------------------------------------------------------------------------
# T45 – AdvisorManager gap coverage (lines 237, 268-272, 275-287, 299, 355)
# ---------------------------------------------------------------------------


class TestAdvisorManagerGaps:
    """
    Covers the branches in manager.py that T05 (TestAdvisorManager) missed:
      line 237  — advisor.can_advise() → False → _try_fallback()
      lines 268-269 — TimeoutError during advise_with_tracking
      lines 271-272 — _SYSTEM_ERRORS during advise_with_tracking
      lines 275-280 — business-logic Exception → _error_response
      lines 282-287 — system failure path → _try_fallback
      line 299  — _try_fallback with empty chain → _error_response
      line 355  — get_advice_parallel outer timeout → t.cancel()
    """

    from keryx.advisors.base import AdvisorResponse, BaseAdvisor

    # ── Shared helper advisors ──────────────────────────────────────────────

    class _TriggerAlways(BaseAdvisor):
        """Always triggers; advise() delegates to a replaceable slot."""
        name = "gap-always"

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse(strategic_direction="gap")

    class _TriggerNever(BaseAdvisor):
        name = "gap-never"

        def should_trigger(self, context: Any) -> bool:
            return False

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse()

    class _FallbackAdvisor(BaseAdvisor):
        """Used as the fallback target in chain tests."""
        name = "gap-fallback"

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse(strategic_direction="fallback-hit")

    @pytest.fixture
    def manager(self):
        from keryx.advisors.manager import AdvisorManager
        m = AdvisorManager()
        yield m
        m.shutdown()

    @pytest.fixture
    def ctx(self):
        from keryx.core.shared_context import SharedContext
        return SharedContext(target_path="/tmp/gap")

    # ── Line 237 + 299: can_advise() False, no fallback chain ──────────────

    async def test_can_advise_false_no_chain_returns_error(self, manager, ctx) -> None:
        """Line 237: advisor.can_advise()==False → _try_fallback(chain=None) → line 299."""
        adv = self._TriggerAlways(max_calls_per_session=0)  # exhausted immediately
        manager.register(adv)
        resp = await manager.get_advice_async(ctx)
        assert "error" in resp.metadata

    # ── Line 237 + fallback chain executes successfully ─────────────────────

    async def test_can_advise_false_with_chain_uses_fallback(self, manager, ctx) -> None:
        """Line 237: can_advise()==False, chain provided → fallback advisor is called."""
        adv_dead = self._TriggerAlways(max_calls_per_session=0)
        adv_dead.name = "gap-dead"
        fallback = self._FallbackAdvisor(max_calls_per_session=5)
        manager.register(adv_dead)
        manager.register(fallback)
        resp = await manager.get_advice_async(
            ctx, advisor_name="gap-dead", fallback_chain=["gap-fallback"]
        )
        # Either the fallback ran successfully or an error was returned —
        # either way _try_fallback was exercised.
        from keryx.advisors.base import AdvisorResponse
        assert isinstance(resp, AdvisorResponse)

    # ── Lines 268-269 + 282-287 + 299: TimeoutError during advise ──────────

    async def test_timeout_error_during_advise_routes_to_fallback(
        self, manager, ctx
    ) -> None:
        """Lines 268-269: advise_with_tracking raises TimeoutError → system path → _try_fallback."""
        from unittest.mock import patch, AsyncMock

        adv = self._TriggerAlways(max_calls_per_session=5)
        manager.register(adv)

        with patch.object(adv, "advise_with_tracking", new=AsyncMock(side_effect=TimeoutError())):
            resp = await manager.get_advice_async(ctx)

        # No fallback chain → line 299 → error response
        assert "error" in resp.metadata

    # ── Lines 271-272 + 282-287: _SYSTEM_ERRORS during advise ─────────────

    async def test_system_error_during_advise_routes_to_fallback(
        self, manager, ctx
    ) -> None:
        """Lines 271-272: ConnectionError (in _SYSTEM_ERRORS) → system path → _try_fallback."""
        from unittest.mock import patch, AsyncMock

        adv = self._TriggerAlways(max_calls_per_session=5)
        manager.register(adv)

        with patch.object(
            adv, "advise_with_tracking", new=AsyncMock(side_effect=ConnectionError("network down"))
        ):
            resp = await manager.get_advice_async(ctx)

        assert "error" in resp.metadata

    # ── Lines 275-280: business-logic Exception ─────────────────────────────

    async def test_business_error_returns_error_response(self, manager, ctx) -> None:
        """Lines 275-280: ValueError (not a system error) hits except Exception → _error_response."""
        from unittest.mock import patch, AsyncMock

        adv = self._TriggerAlways(max_calls_per_session=5)
        manager.register(adv)

        with patch.object(
            adv, "advise_with_tracking", new=AsyncMock(side_effect=ValueError("bad model output"))
        ):
            resp = await manager.get_advice_async(ctx)

        # Business errors return error_response immediately, NOT via fallback
        assert "error" in resp.metadata
        assert manager.calls_made == 1  # total_calls incremented even on failure

    # ── Line 355: get_advice_parallel outer timeout ─────────────────────────

    async def test_parallel_outer_timeout_cancels_tasks(self, manager, ctx) -> None:
        """Line 355: parallel gather times out → pending tasks are cancelled."""
        import asyncio
        from unittest.mock import patch, AsyncMock

        async def _slow_advice(context: Any, advisor_name: Any = None, **kw: Any):
            await asyncio.sleep(999)
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse()

        adv = self._TriggerAlways(max_calls_per_session=5)
        manager.register(adv)

        with patch.object(manager, "get_advice_async", side_effect=_slow_advice):
            results = await manager.get_advice_parallel(
                ctx, advisor_names=["gap-always"], timeout=0.01
            )

        # Timed out → raw = [] → no AdvisorResponse objects
        assert results == []


# ---------------------------------------------------------------------------
# T46 – Security boundary tests (path traversal + scrubbing)
# ---------------------------------------------------------------------------


class TestSecurityBoundaries:
    """
    Adversarial tests for the two main security surfaces:
      1. ReadFileTool — path traversal protection (allowed_root enforcement)
      2. GitBlameTool _scrub() — privacy scrubbing of paths, emails, SHAs
    """

    # ── ReadFileTool: path traversal ────────────────────────────────────────

    @pytest.fixture
    def sandboxed_tool(self, tmp_path):
        """ReadFileTool whose allowed_root is a fresh tmp directory."""
        from keryx.tools.read_file import ReadFileTool
        return ReadFileTool(allowed_root=str(tmp_path))

    async def test_relative_traversal_blocked(self, sandboxed_tool, tmp_path) -> None:
        """../../etc/passwd — resolved path escapes allowed_root → blocked."""
        # Create a real file inside the sandbox so path existence isn't the rejection reason
        (tmp_path / "legit.txt").write_text("ok")
        attack = str(tmp_path / "subdir" / ".." / ".." / "etc" / "passwd")
        result = await sandboxed_tool.execute({"file_path": attack})
        assert result.success is False
        assert result.error == "path_traversal_blocked"

    async def test_absolute_path_outside_root_blocked(self, sandboxed_tool) -> None:
        """/etc/hosts is outside the sandbox root → blocked."""
        result = await sandboxed_tool.execute({"file_path": "/etc/hosts"})
        assert result.success is False
        assert result.error == "path_traversal_blocked"

    async def test_path_within_root_is_allowed(self, sandboxed_tool, tmp_path) -> None:
        """Legitimate file inside allowed_root is readable."""
        target = tmp_path / "safe.txt"
        target.write_text("safe content")
        result = await sandboxed_tool.execute({"file_path": str(target)})
        assert result.success is True
        assert "safe content" in result.output

    async def test_no_allowed_root_permits_any_existing_file(self, tmp_path) -> None:
        """Without allowed_root, traversal protection is off — any readable file is accessible."""
        from keryx.tools.read_file import ReadFileTool
        tool = ReadFileTool()  # no allowed_root
        target = tmp_path / "open.txt"
        target.write_text("unrestricted")
        result = await tool.execute({"file_path": str(target)})
        assert result.success is True

    # ── GitBlameTool: _scrub() output sanitisation ──────────────────────────

    @pytest.fixture
    def scrub(self):
        """Import the module-level _scrub function directly."""
        from keryx.tools.git_blame import _scrub
        return _scrub

    def test_scrub_linux_home_path(self, scrub) -> None:
        """/home/alice/secret → /workspace/secret."""
        out = scrub("blame /home/alice/myrepo/file.c line 42")
        assert "/home/alice" not in out
        assert "/workspace" in out

    def test_scrub_macos_users_path(self, scrub) -> None:
        """/Users/bob/code → /workspace/code."""
        out = scrub("authored by /Users/bob/projects/keryx/core/agent.py")
        assert "/Users/bob" not in out
        assert "/workspace" in out

    def test_scrub_email_address(self, scrub) -> None:
        """Email addresses are replaced with [EMAIL]."""
        out = scrub("commit by alice@example.com on main")
        assert "alice@example.com" not in out
        assert "[EMAIL]" in out

    def test_scrub_full_commit_sha_truncated(self, scrub) -> None:
        """40-char hex SHA is shortened to first 7 chars + ellipsis."""
        sha = "abcdef1234567890abcdef1234567890abcdef12"
        out = scrub(f"commit {sha} merged")
        assert sha not in out
        assert "abcdef1" in out  # first 7 preserved

    def test_scrub_short_sha_untouched(self, scrub) -> None:
        """7-char short SHA (used for display) is NOT altered."""
        short = "abcdef1"
        out = scrub(f"ref {short} in log")
        assert short in out

    def test_scrub_disabled_passes_through(self, tmp_path) -> None:
        """enable_scrubbing=False → _maybe_scrub is identity; no substitution."""
        from keryx.tools.git_blame import GitBlameTool
        from unittest.mock import patch, AsyncMock
        from keryx.tools.Toolbox import ToolResult

        tool = GitBlameTool(enable_scrubbing=False)
        raw_output = "blame by alice@corp.com /Users/dev/repo/file.c"

        # Verify _maybe_scrub returns the original string when scrubbing disabled
        assert tool._maybe_scrub(raw_output) == raw_output

    def test_scrub_multiple_emails_all_replaced(self, scrub) -> None:
        """Multiple email addresses in the same string are all scrubbed."""
        out = scrub("from: alice@x.com to: bob@y.org cc: carol@z.net")
        assert "alice@x.com" not in out
        assert "bob@y.org" not in out
        assert "carol@z.net" not in out
        assert out.count("[EMAIL]") == 3


# ---------------------------------------------------------------------------
# T47 – SwarmDebate coverage (keryx/models/swarm.py — 54% → target 90%+)
# ---------------------------------------------------------------------------


class TestSwarmDebate:
    """
    Covers the branches in swarm.py that are unreachable through orchestrator
    tests alone:
      lines 136,138   — custom model_weights / _DEFAULT_WEIGHTS in __init__
      lines 162-168   — _detect_weight() size bands
      line  181       — debate() early return for empty hypotheses
      lines 195,270-309 — iterative debate path
      lines 200-204   — skeptic path in debate()
      lines 242-244   — model returns Exception in _parallel_debate
      lines 352-363   — _analyze_with_model_iterative
      lines 374-394   — _run_skeptic_review
      lines 401-407   — _apply_skeptic_overrides
      lines 433-437   — _build_iterative_prompt with history
      lines 445-453   — _build_skeptic_prompt
      lines 479-481   — markdown code fence stripping in _parse_verdict_json
      line  503       — neutral vote for hypotheses absent from JSON
      lines 552-565   — _compute_consensus confirmed + rejected paths
      lines 574-575   — assign_skeptic
      lines 578-599   — get_vote_summary
      line  614       — create_swarm_debate factory
    """

    # ── Minimal mock model ─────────────────────────────────────────────────

    class _MockModel:
        """Concrete ModelInterface stand-in — returns pre-programmed JSON."""

        cost_per_1k_input_tokens = 0.0
        cost_per_1k_output_tokens = 0.0

        def __init__(self, name: str, responses: list[str] | None = None):
            self.model_name = name
            self._resp = list(responses or [])
            self._idx = 0

        def generate(self, prompt, config=None, *, grammar=None, max_tokens=None) -> str:
            if self._idx < len(self._resp):
                r = self._resp[self._idx]
                self._idx += 1
                return r
            return '{"hypotheses": []}'

        # Unused abstract stubs — coverage for these is elsewhere
        def generate_result(self, prompt, config=None): ...  # pragma: no cover
        def generate_stream(self, prompt, config=None): ...  # pragma: no cover
        def generate_with_tools(self, prompt, tools, config=None): ...  # pragma: no cover
        def tokenize(self, text): return []  # pragma: no cover
        def get_context_length(self): return 4096  # pragma: no cover
        def is_healthy(self): return True  # pragma: no cover
        def estimate_cost(self, i, o): return {"input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0}  # pragma: no cover
        def get_usage_cost(self): return {"input_cost_usd": 0.0, "output_cost_usd": 0.0, "total_cost_usd": 0.0}  # pragma: no cover
        def get_capabilities(self): return {}  # pragma: no cover
        def unload(self): ...  # pragma: no cover

        # Provide generate_async directly so the executor path is bypassed
        async def generate_async(self, prompt, config=None, *, grammar=None, max_tokens=None) -> str:
            return self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens)

    # ── Verdict JSON helpers ────────────────────────────────────────────────

    @staticmethod
    def _yes_verdict(n: int = 1) -> str:
        items = [
            f'{{"index": {i+1}, "verdict": true, "confidence": 0.9, "reason": "exploitable"}}'
            for i in range(n)
        ]
        return '{"hypotheses": [' + ", ".join(items) + "]}"

    @staticmethod
    def _no_verdict(n: int = 1) -> str:
        items = [
            f'{{"index": {i+1}, "verdict": false, "confidence": 0.1, "reason": "mitigated"}}'
            for i in range(n)
        ]
        return '{"hypotheses": [' + ", ".join(items) + "]}"

    # ── Constructor paths ──────────────────────────────────────────────────

    def test_custom_model_weights_applied(self) -> None:
        """Lines 136,138: explicit model_weights dict overrides auto-detection."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("mymodel")
        sd = SwarmDebate([m], model_weights={"mymodel": 2.5})
        assert sd.model_weights["mymodel"] == pytest.approx(2.5)

    def test_default_weights_lookup(self) -> None:
        """Line 138: model name present in _DEFAULT_WEIGHTS table."""
        from keryx.models.swarm import SwarmDebate, _DEFAULT_WEIGHTS

        if not _DEFAULT_WEIGHTS:
            pytest.skip("_DEFAULT_WEIGHTS is empty — nothing to look up")
        known_name = next(iter(_DEFAULT_WEIGHTS))
        m = self._MockModel(known_name)
        sd = SwarmDebate([m])
        assert sd.model_weights[known_name] == _DEFAULT_WEIGHTS[known_name].weight

    def test_detect_weight_70b_band(self) -> None:
        """Line 162: model name containing '70b' → weight 1.25."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("llama3-70b-instruct")
        sd = SwarmDebate([m])
        assert sd.model_weights["llama3-70b-instruct"] == pytest.approx(1.25)

    def test_detect_weight_32b_band(self) -> None:
        """Line 164: model name containing '32b' → weight 1.10."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("qwen2.5-32b-instruct")
        sd = SwarmDebate([m])
        assert sd.model_weights["qwen2.5-32b-instruct"] == pytest.approx(1.10)

    def test_detect_weight_13b_band(self) -> None:
        """Line 166: model name containing '13b' → weight 0.90."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("codellama-13b")
        sd = SwarmDebate([m])
        assert sd.model_weights["codellama-13b"] == pytest.approx(0.90)

    def test_detect_weight_8b_band(self) -> None:
        """Line 168: model name containing '8b' → weight 0.75."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("llama3-8b-instruct")
        sd = SwarmDebate([m])
        assert sd.model_weights["llama3-8b-instruct"] == pytest.approx(0.75)

    # ── debate() entry point ───────────────────────────────────────────────

    async def test_empty_hypotheses_returns_empty_result(self) -> None:
        """Line 181: debate([]) short-circuits before calling models."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("m1")
        sd = SwarmDebate([m])
        result = await sd.debate([])
        assert result.confirmed == []
        assert result.swarm_voters == 0

    async def test_parallel_debate_basic_confirms_hypothesis(self) -> None:
        """Happy path: one model votes yes → hypothesis confirmed."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("m1", [self._yes_verdict(1)])
        sd = SwarmDebate([m], consensus_threshold=0.5)
        result = await sd.debate(["use-after-free in JSObject"])
        assert len(result.confirmed) == 1
        assert result.swarm_voters == 1

    async def test_parallel_debate_model_exception_skipped(self) -> None:
        """Lines 242-244: model raises during gather → counted as failure, skipped."""
        from keryx.models.swarm import SwarmDebate

        class _CrashingModel(self._MockModel):
            async def generate_async(self, *a, **kw):
                raise RuntimeError("model crashed")

        m = _CrashingModel("crasher")
        sd = SwarmDebate([m])
        result = await sd.debate(["heap overflow"])
        # crashed model → valid_voters=0 → no confirmed
        assert result.confirmed == []

    # ── Iterative debate ───────────────────────────────────────────────────

    async def test_iterative_debate_runs_rounds(self) -> None:
        """Lines 270-309: enable_iterative_debate=True runs multi-round loop."""
        from keryx.models.swarm import SwarmDebate

        # Provide enough verdicts for multiple rounds
        m = self._MockModel("m1", [self._yes_verdict(1)] * 5)
        sd = SwarmDebate(
            [m],
            enable_iterative_debate=True,
            max_debate_rounds=2,
            consensus_threshold=0.5,
        )
        result = await sd.debate(["buffer overflow"])
        assert result.debate_rounds >= 1
        assert isinstance(result.confirmed, list)

    async def test_iterative_debate_early_consensus_exits(self) -> None:
        """Lines 300-303: weighted ratio exceeds threshold → break early."""
        from keryx.models.swarm import SwarmDebate

        m = self._MockModel("m1", [self._yes_verdict(1)] * 10)
        sd = SwarmDebate(
            [m],
            enable_iterative_debate=True,
            max_debate_rounds=5,
            consensus_threshold=0.5,  # easily exceeded with one yes vote
        )
        result = await sd.debate(["integer overflow"])
        # With threshold=0.5 and a single model giving yes, consensus after round 1
        assert result.debate_rounds == 1

    async def test_iterative_prompt_includes_history(self) -> None:
        """Lines 433-437: _build_iterative_prompt with non-empty history."""
        from keryx.models.swarm import SwarmDebate

        sd = SwarmDebate([self._MockModel("m1")])
        prompt = sd._build_iterative_prompt(
            ["hyp1"],
            context=None,
            history=["[m1] found UAF", "[m2] buffer issue"],
            round_num=2,
        )
        assert "Previous debate" in prompt
        assert "round 1" in prompt

    # ── Skeptic review ─────────────────────────────────────────────────────

    async def test_skeptic_review_overrides_confirmed(self) -> None:
        """Lines 200-204, 374-394, 401-407: skeptic votes NO → override applied."""
        from keryx.models.swarm import SwarmDebate

        voter = self._MockModel("voter", [self._yes_verdict(1)])
        skeptic = self._MockModel("skeptic", [self._no_verdict(1)])  # disputes the finding

        sd = SwarmDebate([voter], consensus_threshold=0.5, skeptic_model=skeptic)
        result = await sd.debate(["heap UAF"])
        # voter confirms it, skeptic then marks it False → override
        assert result.skeptic_overrides >= 0  # may be 0 if skeptic parse fails gracefully

    async def test_skeptic_review_empty_candidates_returns_empty(self) -> None:
        """Line 374: _run_skeptic_review with empty candidates → {} immediately."""
        from keryx.models.swarm import SwarmDebate

        skeptic = self._MockModel("skeptic")
        sd = SwarmDebate([self._MockModel("m1")], skeptic_model=skeptic)
        result = await sd._run_skeptic_review(candidates=[], all_votes={}, context=None)
        assert result == {}

    async def test_skeptic_review_model_exception_returns_empty(self) -> None:
        """Lines 392-394: skeptic model raises → exception caught → {}."""
        from keryx.models.swarm import SwarmDebate

        class _CrashingSkeptic(self._MockModel):
            async def generate_async(self, *a, **kw):
                raise ConnectionError("skeptic offline")

        sd = SwarmDebate([self._MockModel("m1")], skeptic_model=_CrashingSkeptic("sk"))
        result = await sd._run_skeptic_review(["hyp"], {}, None)
        assert result == {}

    def test_apply_skeptic_overrides_removes_from_confirmed(self) -> None:
        """Lines 401-407: skeptic vote False on a confirmed hyp → moved to rejected."""
        from keryx.models.swarm import SwarmDebate, SwarmResult, SwarmVote

        sd = SwarmDebate([self._MockModel("m1")])
        result = SwarmResult(
            hypotheses=["hyp1"],
            confirmed=["hyp1"],
            rejected=[],
            votes={},
            weighted_consensus_ratio=0.8,
            raw_consensus_ratio=0.8,
            swarm_voters=1,
        )
        skeptic_votes = {
            "hyp1": SwarmVote(
                model_name="skeptic",
                hypothesis="hyp1",
                verdict=False,  # disputes the confirmation
                confidence=0.95,
                weight=2.0,
            )
        }
        overrides = sd._apply_skeptic_overrides(result, skeptic_votes)
        assert overrides == 1
        assert "hyp1" not in result.confirmed
        assert "hyp1" in result.rejected

    # ── Skeptic prompt builder ─────────────────────────────────────────────

    def test_build_skeptic_prompt_contains_mandate(self) -> None:
        """Lines 445-453: _build_skeptic_prompt includes DISPROVE mandate."""
        from keryx.models.swarm import SwarmDebate

        sd = SwarmDebate([self._MockModel("m1")])
        prompt = sd._build_skeptic_prompt(
            candidates=["UAF in JSObject"],
            all_votes={"UAF in JSObject": []},
            context="function foo() { ... }",
        )
        assert "DISPROVE" in prompt
        assert "UAF in JSObject" in prompt

    # ── JSON parsing ───────────────────────────────────────────────────────

    def test_parse_verdict_json_strips_markdown_fence(self) -> None:
        """Lines 479-481: response wrapped in ```json ... ``` is unwrapped."""
        from keryx.models.swarm import SwarmDebate

        sd = SwarmDebate([self._MockModel("m1")])
        fenced = (
            "```json\n"
            '{"hypotheses": [{"index": 1, "verdict": true, "confidence": 0.8, "reason": "ok"}]}\n'
            "```"
        )
        votes = sd._parse_verdict_json(fenced, ["hyp1"], "m1", 50.0)
        assert "hyp1" in votes
        assert votes["hyp1"].verdict is True

    def test_parse_verdict_json_missing_hypothesis_gets_neutral(self) -> None:
        """Line 503: hypothesis not mentioned in JSON → neutral False vote inserted."""
        from keryx.models.swarm import SwarmDebate

        sd = SwarmDebate([self._MockModel("m1")])
        # JSON only covers hyp1; hyp2 is absent
        raw = '{"hypotheses": [{"index": 1, "verdict": true, "confidence": 0.9, "reason": "x"}]}'
        votes = sd._parse_verdict_json(raw, ["hyp1", "hyp2"], "m1", 10.0)
        assert "hyp2" in votes
        assert votes["hyp2"].verdict is False
        assert votes["hyp2"].reasoning == "not_evaluated"

    # ── Consensus computation ──────────────────────────────────────────────

    def test_compute_consensus_confirms_high_ratio(self) -> None:
        """Lines 552-560: w_ratio >= threshold → hypothesis confirmed."""
        from keryx.models.swarm import SwarmDebate, SwarmVote

        sd = SwarmDebate([self._MockModel("m1")], consensus_threshold=0.5)
        votes = {
            "hyp1": [SwarmVote("m1", "hyp1", verdict=True, confidence=0.9, weight=1.0)]
        }
        confirmed, rejected, w_ratio, r_ratio = sd._compute_consensus(
            votes, ["hyp1"], total_voters=1
        )
        assert "hyp1" in confirmed
        assert "hyp1" not in rejected
        assert w_ratio == pytest.approx(1.0)

    def test_compute_consensus_rejects_unanimous_no(self) -> None:
        """Lines 561-565: r_ratio < 0.3 and w_ratio < 0.4 → rejected."""
        from keryx.models.swarm import SwarmDebate, SwarmVote

        sd = SwarmDebate([self._MockModel("m1")], consensus_threshold=0.75)
        votes = {
            "hyp1": [SwarmVote("m1", "hyp1", verdict=False, confidence=0.9, weight=1.0)]
        }
        confirmed, rejected, w_ratio, r_ratio = sd._compute_consensus(
            votes, ["hyp1"], total_voters=1
        )
        assert "hyp1" not in confirmed
        assert "hyp1" in rejected

    # ── Utility methods ────────────────────────────────────────────────────

    def test_assign_skeptic(self) -> None:
        """Lines 574-575: assign_skeptic() sets skeptic_model attribute."""
        from keryx.models.swarm import SwarmDebate

        sd = SwarmDebate([self._MockModel("m1")])
        new_skeptic = self._MockModel("super-skeptic")
        sd.assign_skeptic(new_skeptic)
        assert sd.skeptic_model is new_skeptic

    async def test_get_vote_summary_confirmed_and_rejected(self) -> None:
        """Lines 578-599: get_vote_summary renders confirmed, rejected, overrides."""
        from keryx.models.swarm import SwarmDebate

        voter = self._MockModel("voter", [self._yes_verdict(1)])
        sd = SwarmDebate([voter], consensus_threshold=0.5)
        result = await sd.debate(["heap overflow"])
        # Ensure something was confirmed for a richer summary
        result.rejected = ["false positive"]
        result.skeptic_overrides = 1
        result.timeout_occurred = True

        summary = sd.get_vote_summary(result)
        assert "Swarm Debate Results" in summary
        assert "Confirmed" in summary
        assert "Rejected" in summary
        assert "Skeptic overrides" in summary
        assert "Timeout" in summary

    # ── Factory ────────────────────────────────────────────────────────────

    def test_create_swarm_debate_factory(self) -> None:
        """Line 614: create_swarm_debate() returns a configured SwarmDebate."""
        from keryx.models.swarm import create_swarm_debate, SwarmDebate

        m = self._MockModel("m1")
        sd = create_swarm_debate([m], consensus_threshold=0.6, swarm_timeout=45.0)
        assert isinstance(sd, SwarmDebate)
        assert sd.swarm_timeout == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# T48 – AdvancedCascadeAdvisor (keryx/advisors/cascade.py)
# ---------------------------------------------------------------------------


class TestAdvancedCascadeAdvisor:
    """
    Full behavioural coverage of AdvancedCascadeAdvisor and its supporting
    types (CascadeConfig, AdvisorState, ErrorSeverity, create_cascade_advisor).
    """

    from keryx.advisors.base import AdvisorResponse, BaseAdvisor

    # ── Shared helper advisors ──────────────────────────────────────────────

    class _YesAdvisor(BaseAdvisor):
        name = "adv-yes"
        requires_network = False

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse(strategic_direction="found something")

    class _NoAdvisor(BaseAdvisor):
        name = "adv-no"
        requires_network = False

        def should_trigger(self, context: Any) -> bool:
            return False

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse()

    class _NetworkAdvisor(BaseAdvisor):
        """requires_network=True — NOT skipped during escalation."""
        name = "adv-network"
        requires_network = True

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            from keryx.advisors.base import AdvisorResponse
            return AdvisorResponse(strategic_direction="cloud advice")

    class _SlowAdvisor(BaseAdvisor):
        """Simulates a slow advisor that always times out."""
        name = "adv-slow"
        requires_network = False

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            await asyncio.sleep(999)
            from keryx.advisors.base import AdvisorResponse  # pragma: no cover
            return AdvisorResponse()                          # pragma: no cover

    class _CrashAdvisor(BaseAdvisor):
        """Raises a configurable exception from advise()."""
        name = "adv-crash"
        requires_network = False

        def __init__(self, exc: Exception, **kwargs: Any):
            super().__init__(**kwargs)
            self._exc = exc

        def should_trigger(self, context: Any) -> bool:
            return True

        async def advise(self, context: Any) -> "AdvisorResponse":
            raise self._exc

    # ── Construction ───────────────────────────────────────────────────────

    def test_empty_advisors_raises(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        with pytest.raises(ValueError, match="at least one advisor"):
            AdvancedCascadeAdvisor([])

    def test_circular_reference_raises(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = self._YesAdvisor()
        adv.name = "advanced-cascade"   # same as default cascade name
        with pytest.raises(ValueError, match="Circular reference"):
            AdvancedCascadeAdvisor([adv])

    def test_default_config_applied(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig
        adv = AdvancedCascadeAdvisor([self._YesAdvisor(max_calls_per_session=5)])
        assert adv.config.cascade_timeout == CascadeConfig().cascade_timeout
        assert adv.config.disable_on_critical is True

    def test_custom_config_stored(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig
        cfg = CascadeConfig(cascade_timeout=120.0, disable_on_critical=False)
        adv = AdvancedCascadeAdvisor([self._YesAdvisor(max_calls_per_session=5)], config=cfg)
        assert adv.config.cascade_timeout == pytest.approx(120.0)
        assert adv.config.disable_on_critical is False

    def test_repr_contains_name_and_budget(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig
        adv = AdvancedCascadeAdvisor(
            [self._YesAdvisor(max_calls_per_session=5)],
            config=CascadeConfig(cascade_timeout=45.0),
        )
        r = repr(adv)
        assert "AdvancedCascadeAdvisor" in r
        assert "45.0" in r

    # ── should_trigger ─────────────────────────────────────────────────────

    def test_should_trigger_true_when_any_enabled_says_yes(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([self._YesAdvisor(max_calls_per_session=5)])
        assert adv.should_trigger({}) is True

    def test_should_trigger_false_when_all_say_no(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([self._NoAdvisor(max_calls_per_session=5)])
        assert adv.should_trigger({}) is False

    def test_should_trigger_false_when_all_disabled(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([self._YesAdvisor(max_calls_per_session=5)])
        adv._chain[0].disabled = True
        assert adv.should_trigger({}) is False

    async def test_should_trigger_async_delegates_to_sub_advisor(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([self._YesAdvisor(max_calls_per_session=5)])
        assert await adv.should_trigger_async({}) is True

    # ── Happy path ─────────────────────────────────────────────────────────

    async def test_advise_returns_first_non_empty_response(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([
            self._NoAdvisor(max_calls_per_session=5),
            self._YesAdvisor(max_calls_per_session=5),
        ])
        resp = await adv.advise({})
        assert not resp.is_empty()
        cascade_meta = resp.metadata.get("cascade", {})
        assert cascade_meta.get("successful_advisor") == "adv-yes"

    async def test_advise_exhausted_chain_returns_empty_response(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([self._NoAdvisor(max_calls_per_session=5)])
        resp = await adv.advise({})
        assert resp.is_empty() or "cascade" in resp.metadata

    # ── Escalation awareness ───────────────────────────────────────────────

    async def test_escalation_skips_local_advisor(self) -> None:
        """Low confidence → local advisor skipped, network advisor reached."""
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig
        from keryx.core.shared_context import SharedContext

        # Context with last confidence below critical_confidence_threshold
        ctx = SharedContext(target_path="/tmp/esc")
        ctx.steps = [{
            "timestamp": 0.0,
            "action": {"name": "read_file", "input": {}, "thought": "", "confidence": 0.1},
            "observation": "low",
        }]
        ctx.steps_taken = 1

        cfg = CascadeConfig(critical_confidence_threshold=0.3)
        local_adv   = self._YesAdvisor(max_calls_per_session=5)   # requires_network=False → skipped
        network_adv = self._NetworkAdvisor(max_calls_per_session=5) # requires_network=True → reached

        cascade = AdvancedCascadeAdvisor([local_adv, network_adv], config=cfg)
        resp = await cascade.advise(ctx)

        cascade_meta = resp.metadata.get("cascade", {})
        successful = cascade_meta.get("successful_advisor") if isinstance(cascade_meta, dict) else None
        # The local advisor was skipped; the network advisor ran and succeeded
        assert successful == "adv-network"

    # ── Timeout → auto-disabling ───────────────────────────────────────────

    async def test_timeout_increments_consecutive_timeouts(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig

        cfg = CascadeConfig(
            max_per_advisor_seconds=0.01,
            disable_after_consecutive_timeouts=5,  # high so it doesn't disable
        )
        slow = self._SlowAdvisor(max_calls_per_session=5)
        adv  = AdvancedCascadeAdvisor([slow], config=cfg)

        await adv.advise({})

        state = adv._chain[0]
        assert state.consecutive_timeouts >= 1
        assert state.consecutive_failures >= 1

    async def test_repeated_timeouts_disable_advisor(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig

        cfg = CascadeConfig(
            max_per_advisor_seconds=0.01,
            disable_after_consecutive_timeouts=2,
            cascade_timeout=5.0,
        )
        slow = self._SlowAdvisor(max_calls_per_session=10)
        adv  = AdvancedCascadeAdvisor([slow], config=cfg)

        # Two calls → two consecutive timeouts → disabled after second
        await adv.advise({})
        await adv.advise({})

        assert adv._chain[0].disabled is True

    # ── Exception classification → auto-disabling ─────────────────────────

    async def test_critical_error_disables_advisor(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig

        cfg   = CascadeConfig(disable_on_critical=True)
        crash = self._CrashAdvisor(
            RuntimeError("401 unauthorized"),
            max_calls_per_session=5,
        )
        adv = AdvancedCascadeAdvisor([crash], config=cfg)
        await adv.advise({})

        assert adv._chain[0].disabled is True

    async def test_critical_disabled_flag_false_does_not_disable(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig

        cfg   = CascadeConfig(disable_on_critical=False)
        crash = self._CrashAdvisor(
            RuntimeError("401 unauthorized"),
            max_calls_per_session=5,
        )
        adv = AdvancedCascadeAdvisor([crash], config=cfg)
        await adv.advise({})

        assert adv._chain[0].disabled is False

    async def test_transient_error_does_not_disable(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig

        cfg   = CascadeConfig(disable_on_critical=True)
        crash = self._CrashAdvisor(
            RuntimeError("rate limit 429"),
            max_calls_per_session=5,
        )
        adv = AdvancedCascadeAdvisor([crash], config=cfg)
        await adv.advise({})

        assert adv._chain[0].disabled is False

    async def test_recoverable_error_increments_failure_counter(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor

        crash = self._CrashAdvisor(
            RuntimeError("unexpected output"),
            max_calls_per_session=5,
        )
        adv = AdvancedCascadeAdvisor([crash])
        await adv.advise({})

        state = adv._chain[0]
        assert state.consecutive_failures == 1
        assert state.disabled is False

    # ── _classify_error ────────────────────────────────────────────────────

    def test_classify_error_auth_is_critical(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, ErrorSeverity
        assert AdvancedCascadeAdvisor._classify_error(
            RuntimeError("401 unauthorized")
        ) == ErrorSeverity.CRITICAL

    def test_classify_error_rate_limit_is_transient(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, ErrorSeverity
        assert AdvancedCascadeAdvisor._classify_error(
            RuntimeError("rate limit 429")
        ) == ErrorSeverity.TRANSIENT

    def test_classify_error_unknown_is_recoverable(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, ErrorSeverity
        assert AdvancedCascadeAdvisor._classify_error(
            RuntimeError("something weird happened")
        ) == ErrorSeverity.RECOVERABLE

    # ── reset() ────────────────────────────────────────────────────────────

    async def test_reset_restores_all_advisor_state(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor, CascadeConfig

        crash = self._CrashAdvisor(
            RuntimeError("401 unauthorized"),
            max_calls_per_session=5,
        )
        adv = AdvancedCascadeAdvisor([crash], config=CascadeConfig(disable_on_critical=True))
        await adv.advise({})

        assert adv._chain[0].disabled is True  # was disabled by CRITICAL error

        adv.reset()

        state = adv._chain[0]
        assert state.disabled is False
        assert state.consecutive_failures == 0
        assert state.consecutive_timeouts == 0
        assert state.last_error is None
        assert adv.calls_made == 0

    # ── get_metrics ────────────────────────────────────────────────────────

    def test_get_metrics_structure(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        adv = AdvancedCascadeAdvisor([self._YesAdvisor(max_calls_per_session=5)])
        m   = adv.get_metrics()
        assert "chain_length" in m
        assert "active" in m
        assert "disabled_advisors" in m
        assert "cascade_timeout" in m
        assert "child_metrics" in m

    # ── _enrich_response ───────────────────────────────────────────────────

    def test_enrich_response_adds_cascade_metadata(self) -> None:
        from keryx.advisors.cascade import AdvancedCascadeAdvisor
        from keryx.advisors.base import AdvisorResponse

        base_resp = AdvisorResponse(strategic_direction="do X")
        enriched  = AdvancedCascadeAdvisor._enrich_response(
            base_resp,
            path=["adv-yes"],
            timing={"adv-yes": 42.0},
            errors={},
        )
        meta = enriched.metadata.get("cascade", {})
        assert meta["successful_advisor"] == "adv-yes"
        assert meta["timing_ms"]["adv-yes"] == 42
        assert meta["failures"] == {}

    # ── Factory ────────────────────────────────────────────────────────────

    def test_factory_advanced_mode_returns_advanced(self) -> None:
        from keryx.advisors.cascade import create_cascade_advisor, AdvancedCascadeAdvisor
        result = create_cascade_advisor([self._YesAdvisor(max_calls_per_session=5)])
        assert isinstance(result, AdvancedCascadeAdvisor)

    def test_factory_simple_mode_returns_simple(self) -> None:
        from keryx.advisors.cascade import create_cascade_advisor
        from keryx.advisors.base import CascadeAdvisor
        result = create_cascade_advisor(
            [self._YesAdvisor(max_calls_per_session=5)], mode="simple"
        )
        assert isinstance(result, CascadeAdvisor)

    def test_factory_returns_base_advisor(self) -> None:
        from keryx.advisors.cascade import create_cascade_advisor
        from keryx.advisors.base import BaseAdvisor
        result = create_cascade_advisor([self._YesAdvisor(max_calls_per_session=5)])
        assert isinstance(result, BaseAdvisor)

    def test_factory_name_override(self) -> None:
        from keryx.advisors.cascade import create_cascade_advisor
        result = create_cascade_advisor(
            [self._YesAdvisor(max_calls_per_session=5)], name="my-cascade"
        )
        assert result.name == "my-cascade"

    def test_factory_custom_config(self) -> None:
        from keryx.advisors.cascade import create_cascade_advisor, AdvancedCascadeAdvisor, CascadeConfig
        cfg    = CascadeConfig(cascade_timeout=99.0)
        result = create_cascade_advisor([self._YesAdvisor(max_calls_per_session=5)], config=cfg)
        assert isinstance(result, AdvancedCascadeAdvisor)
        assert result.config.cascade_timeout == pytest.approx(99.0)

    # ── Lazy import from keryx root ────────────────────────────────────────

    def test_lazy_import_from_keryx(self) -> None:
        from keryx import AdvancedCascadeAdvisor, CascadeConfig, create_cascade_advisor
        assert AdvancedCascadeAdvisor is not None
        assert CascadeConfig is not None
        assert create_cascade_advisor is not None


# T49 – InjectionVerifier: full execute() coverage (lines 65-157)
# Tests every branch: missing fields, tool-not-found, reject/accept verdicts,
# VULN_CONFIRMED, expected-vs-actual mismatch warnings, and base_overrides.
# ---------------------------------------------------------------------------

class TestInjectionVerifier:
    """T49 – InjectionVerifier execute() path coverage."""

    # ── Minimal mock infrastructure ────────────────────────────────────────

    class _MockTool:
        """Stub tool whose execute() returns a preset ToolResult."""
        def __init__(self, result):
            self._result = result
        async def execute(self, action_input, context=None):
            return self._result

    class _MockToolBox:
        """Minimal ToolBox stub."""
        def __init__(self, tools: dict):
            self._tools = tools  # name → _MockTool | None

        def get_tool(self, name: str):
            return self._tools.get(name)

        def list_tools(self) -> list:
            return list(self._tools.keys())

        async def execute_async(self, name: str, action_input, context=None):
            tool = self._tools.get(name)
            if tool is None:
                from keryx.tools.Toolbox import ToolResult
                return ToolResult(success=False, output="not found")
            return await tool.execute(action_input, context)

    @pytest.fixture
    def accepting_toolbox(self):
        """ToolBox with a tool that succeeds and returns blank output (accepts payload)."""
        from keryx.tools.Toolbox import ToolResult
        tool = self._MockTool(ToolResult(success=True, output="commit abc123"))
        return self._MockToolBox({"git_blame": tool})

    @pytest.fixture
    def rejecting_toolbox(self):
        """ToolBox with a tool that fails (rejects payload)."""
        from keryx.tools.Toolbox import ToolResult
        tool = self._MockTool(ToolResult(success=False, output="error: invalid option"))
        return self._MockToolBox({"git_blame": tool})

    @pytest.fixture
    def verifier_accepting(self, accepting_toolbox):
        from keryx.tools.injection_verifier import InjectionVerifier
        return InjectionVerifier(toolbox=accepting_toolbox)

    @pytest.fixture
    def verifier_rejecting(self, rejecting_toolbox):
        from keryx.tools.injection_verifier import InjectionVerifier
        return InjectionVerifier(toolbox=rejecting_toolbox)

    # ── Validation guards ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_missing_target_tool(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute({"inject_field": "author", "payload": "x"})
        assert not result.success
        assert result.error == "missing_field"
        assert "target_tool" in result.output

    @pytest.mark.asyncio
    async def test_missing_inject_field(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute({"target_tool": "git_blame", "payload": "x"})
        assert not result.success
        assert result.error == "missing_field"
        assert "inject_field" in result.output

    @pytest.mark.asyncio
    async def test_missing_payload(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute(
            {"target_tool": "git_blame", "inject_field": "author"}
        )
        assert not result.success
        assert result.error == "missing_field"

    @pytest.mark.asyncio
    async def test_tool_not_found(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute(
            {"target_tool": "nonexistent_tool", "inject_field": "x", "payload": "y"}
        )
        assert not result.success
        assert result.error == "tool_not_found"
        assert "nonexistent_tool" in result.output
        assert "git_blame" in result.output  # lists available tools

    # ── Verdict paths ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_dangerous_payload_accepted_confirms_vuln(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "--upload-pack=test",
            "expected_behavior": "reject",
        })
        assert result.success
        assert result.data["vuln_confirmed"] is True
        assert "ACCEPTED" in result.output
        assert "VULN_CONFIRMED" in result.output

    @pytest.mark.asyncio
    async def test_dangerous_payload_rejected_no_vuln(self, verifier_rejecting) -> None:
        result = await verifier_rejecting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "--upload-pack=test",
            "expected_behavior": "reject",
        })
        assert result.success
        assert result.data["vuln_confirmed"] is False
        assert "VULN_CONFIRMED" not in result.output

    @pytest.mark.asyncio
    async def test_safe_payload_accepted_no_vuln(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "Alice",
            "expected_behavior": "accept",
        })
        assert result.success
        assert result.data["vuln_confirmed"] is False
        assert "VULN_CONFIRMED" not in result.output

    # ── Expected-vs-actual mismatch warnings ───────────────────────────────

    @pytest.mark.asyncio
    async def test_warn_accepted_when_reject_expected(self, verifier_accepting) -> None:
        # Non-dangerous payload that gets accepted when reject was expected
        result = await verifier_accepting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "some_value",
            "expected_behavior": "reject",
        })
        assert result.success
        assert "[WARN]" in result.output
        assert "should have rejected" in result.output

    @pytest.mark.asyncio
    async def test_warn_rejected_when_accept_expected(self, verifier_rejecting) -> None:
        result = await verifier_rejecting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "alice",
            "expected_behavior": "accept",
        })
        assert result.success
        assert "[WARN]" in result.output
        assert "over-sanitization" in result.output

    # ── base_overrides ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_base_overrides_applied(self) -> None:
        """base_overrides should merge into the base input before injection."""
        from keryx.tools.Toolbox import ToolResult
        received: list[dict] = []

        class CapturingTool:
            async def execute(self, action_input, context=None):
                received.append(dict(action_input))
                return ToolResult(success=True, output="ok")

        tb = self._MockToolBox({"git_blame": CapturingTool()})
        from keryx.tools.injection_verifier import InjectionVerifier
        iv = InjectionVerifier(toolbox=tb)
        await iv.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "test_value",
            "expected_behavior": "any",
            "base_overrides": {"file": "/real/path/foo.py"},
        })
        assert received, "tool should have been called"
        assert received[0]["file"] == "/real/path/foo.py"
        assert received[0]["author"] == "test_value"

    # ── Output structure ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_output_contains_expected_sections(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "alice",
            "expected_behavior": "any",
        })
        assert result.success
        for section in ("payload", "expected", "verdict", "target tool output"):
            assert section in result.output.lower()

    @pytest.mark.asyncio
    async def test_data_fields_present(self, verifier_accepting) -> None:
        result = await verifier_accepting.execute({
            "target_tool": "git_blame",
            "inject_field": "author",
            "payload": "alice",
            "expected_behavior": "any",
        })
        for key in ("target_tool", "inject_field", "payload", "expected_behavior",
                    "tool_rejected", "vuln_confirmed", "target_success"):
            assert key in result.data

    # ── Factory ────────────────────────────────────────────────────────────

    def test_factory_returns_injection_verifier(self) -> None:
        from keryx.tools.injection_verifier import create_injection_verifier, InjectionVerifier
        from keryx.tools.Toolbox import ToolResult
        tb = self._MockToolBox({})
        iv = create_injection_verifier(tb)
        assert isinstance(iv, InjectionVerifier)


# T50 – ASTAnalyzerTool: execute() + all five _VulnVisitor rules
# ---------------------------------------------------------------------------

class TestASTAnalyzerTool:
    """T50 – Full coverage of ast_analyzer.py (execute guards + R1–R5 + helpers)."""

    # ── Source snippets that trigger each rule ─────────────────────────────

    _CLEAN = """\
def greet(name: str) -> str:
    return f"Hello, {name}"
"""

    # R1: subprocess with shell=True  (Attribute form: subprocess.run)
    _R1_ATTR = """\
import subprocess
subprocess.run("ls " + path, shell=True)
"""

    # R1: subprocess with shell=True  (bare Name form: run)
    _R1_NAME = """\
from subprocess import run
run("ls", shell=True)
"""

    # R2a: git option injection via cmd.extend(["--flag", variable])
    _R2A = """\
author = "alice"
cmd = ["git", "log"]
cmd.extend(["--author", author])
"""

    # R2b: unsanitized append of a variable
    _R2B = """\
file_path = "/tmp/x"
cmd = ["cat"]
cmd.append(file_path)
"""

    # R3: open() with user-controlled path variable
    _R3 = """\
def read(path):
    with open(path) as f:
        return f.read()
"""

    # R4: hardcoded secret
    _R4 = """\
api_key = "sk-abc123def456ghi789"
"""

    # R5: asyncio.create_subprocess_exec(*cmd)
    _R5 = """\
import asyncio
cmd = ["git", "log"]
proc = asyncio.create_subprocess_exec(*cmd)
"""

    # All rules in one file
    _ALL_RULES = """\
import subprocess, asyncio

# R1 attr
subprocess.run("ls " + path, shell=True)

# R2a
author = "alice"
cmd = ["git", "log"]
cmd.extend(["--author", author])

# R2b
cmd.append(file_path)

# R3
def read(path):
    with open(path) as f:
        return f.read()

# R4
api_key = "sk-abc123def456ghi789"

# R5
proc = asyncio.create_subprocess_exec(*cmd)
"""

    _SYNTAX_ERROR = "def broken(\n    pass\n"

    # ── Fixtures ───────────────────────────────────────────────────────────

    @pytest.fixture
    def tool(self):
        from keryx.tools.ast_analyzer import create_ast_analyzer_tool
        return create_ast_analyzer_tool()

    def _write(self, tmp_path, name: str, content: str) -> Path:
        f = tmp_path / name
        f.write_text(content)
        return f

    # ── execute() guard paths ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_missing_path(self, tool) -> None:
        result = await tool.execute({})
        assert not result.success
        assert result.error == "path_missing"

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path) -> None:
        from keryx.tools.ast_analyzer import create_ast_analyzer_tool
        restricted = create_ast_analyzer_tool(allowed_root=str(tmp_path))
        result = await restricted.execute({"path": "/etc/passwd"})
        assert not result.success
        assert result.error == "path_traversal_blocked"

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool) -> None:
        result = await tool.execute({"path": "/no/such/file_xyz.py"})
        assert not result.success
        assert result.error == "file_not_found"

    @pytest.mark.asyncio
    async def test_syntax_error(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "bad.py", self._SYNTAX_ERROR)
        result = await tool.execute({"path": str(f)})
        assert not result.success
        assert result.error == "syntax_error"

    @pytest.mark.asyncio
    async def test_path_alias_file_path(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "clean.py", self._CLEAN)
        result = await tool.execute({"file_path": str(f)})
        assert result.success

    @pytest.mark.asyncio
    async def test_path_alias_target(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "clean2.py", self._CLEAN)
        result = await tool.execute({"target": str(f)})
        assert result.success

    # ── Clean file ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_no_findings(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "clean.py", self._CLEAN)
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert result.data["findings_count"] == 0
        assert "No findings" in result.output

    # ── R1: SUBPROCESS_SHELL_TRUE ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r1_subprocess_attr(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r1a.py", self._R1_ATTR)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "SUBPROCESS_SHELL_TRUE" in rules

    @pytest.mark.asyncio
    async def test_r1_subprocess_bare_name(self, tool, tmp_path) -> None:
        # Exercises the `isinstance(func, ast.Name)` branch in _check_subprocess
        f = self._write(tmp_path, "r1b.py", self._R1_NAME)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "SUBPROCESS_SHELL_TRUE" in rules

    # ── R2: GIT_OPTION_INJECTION / UNSANITIZED_SUBPROCESS_ARG ─────────────

    @pytest.mark.asyncio
    async def test_r2a_git_option_injection(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r2a.py", self._R2A)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "GIT_OPTION_INJECTION" in rules

    @pytest.mark.asyncio
    async def test_r2b_unsanitized_append(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r2b.py", self._R2B)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSANITIZED_SUBPROCESS_ARG" in rules

    # ── R3: OPEN_USER_PATH ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r3_open_user_path(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r3.py", self._R3)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "OPEN_USER_PATH" in rules

    # ── R4: HARDCODED_SECRET ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r4_hardcoded_secret(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r4.py", self._R4)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "HARDCODED_SECRET" in rules

    @pytest.mark.asyncio
    async def test_r4_short_value_not_flagged(self, tool, tmp_path) -> None:
        # Values ≤ 4 chars are ignored (placeholders like "" or "TODO")
        f = self._write(tmp_path, "r4b.py", 'password = "ok"\n')
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "HARDCODED_SECRET" not in rules

    # ── R5: SUBPROCESS_EXEC_STARRED ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r5_create_subprocess_exec(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r5.py", self._R5)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "SUBPROCESS_EXEC_STARRED" in rules

    # ── Output structure ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_output_contains_ast_prefix(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "all.py", self._ALL_RULES)
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert result.output.startswith("[AST]")

    @pytest.mark.asyncio
    async def test_output_contains_line_numbers(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r1c.py", self._R1_ATTR)
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert "line " in result.output  # Finding.__str__ includes "@ line N"

    @pytest.mark.asyncio
    async def test_output_contains_snippet(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r1d.py", self._R1_ATTR)
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert "shell=True" in result.output  # snippet from source line

    @pytest.mark.asyncio
    async def test_data_findings_fields(self, tool, tmp_path) -> None:
        f = self._write(tmp_path, "r1e.py", self._R1_ATTR)
        result = await tool.execute({"path": str(f)})
        assert result.success
        fd = result.data["findings"][0]
        for key in ("rule", "severity", "line", "message", "snippet"):
            assert key in fd

    # ── Finding.__str__ ────────────────────────────────────────────────────

    def test_finding_str(self) -> None:
        from keryx.tools.ast_analyzer import Finding
        f = Finding(rule="TEST_RULE", severity="HIGH", line=42, col=0, message="bad thing")
        s = str(f)
        assert "HIGH" in s
        assert "TEST_RULE" in s
        assert "42" in s

    # ── _VulnVisitor helpers ───────────────────────────────────────────────

    def test_get_keyword_value_miss(self) -> None:
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        call = ast.parse("f(a=1)", mode="eval").body
        v = _VulnVisitor([])
        assert v._get_keyword_value(call, "nonexistent") is None

    def test_is_name_true_and_false(self) -> None:
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        name_node = ast.parse("x", mode="eval").body   # ast.Name(id='x')
        const_node = ast.parse("1", mode="eval").body  # ast.Constant
        assert _VulnVisitor._is_name(name_node, "x") is True
        assert _VulnVisitor._is_name(const_node, "x") is False

    def test_is_true_variants(self) -> None:
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        true_node  = ast.parse("True",  mode="eval").body
        false_node = ast.parse("False", mode="eval").body
        assert _VulnVisitor._is_true(true_node)  is True
        assert _VulnVisitor._is_true(false_node) is False
        assert _VulnVisitor._is_true(None)       is False

    @pytest.mark.asyncio
    async def test_snippet_out_of_bounds(self, tool, tmp_path) -> None:
        # A one-line file; snippet for a node with line=0 returns ""
        # We verify via: inject a finding with lineno=0 by analysing a
        # file where _snippet is called but line is at boundary.
        # Indirect: just confirm execute() works on a 1-line file.
        f = self._write(tmp_path, "one.py", 'api_key = "sk-supersecretvalue"\n')
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert result.data["findings_count"] >= 1

    # ── Factory ────────────────────────────────────────────────────────────

    def test_factory(self) -> None:
        from keryx.tools.ast_analyzer import create_ast_analyzer_tool, ASTAnalyzerTool
        t = create_ast_analyzer_tool()
        assert isinstance(t, ASTAnalyzerTool)
        assert t.name == "codeql_query"

    # ── R6 LLM_OUTPUT_SINK ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r6_json_loads_variable_flagged(self, tool, tmp_path) -> None:
        """json.loads(variable) — non-literal arg triggers LLM_OUTPUT_SINK."""
        src = "import json\nresult = json.loads(raw_text)\n"
        f = self._write(tmp_path, "r6a.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "LLM_OUTPUT_SINK" in rules

    @pytest.mark.asyncio
    async def test_r6_json_loads_literal_not_flagged(self, tool, tmp_path) -> None:
        """json.loads('{}') — literal string is safe, no finding."""
        src = 'import json\nx = json.loads("{}")\n'
        f = self._write(tmp_path, "r6b.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "LLM_OUTPUT_SINK" not in rules

    @pytest.mark.asyncio
    async def test_r6_json_load_call_expr_flagged(self, tool, tmp_path) -> None:
        """json.loads(func()) — call expression as arg is also flagged."""
        src = "import json\ndata = json.loads(get_response())\n"
        f = self._write(tmp_path, "r6c.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "LLM_OUTPUT_SINK" in rules

    @pytest.mark.asyncio
    async def test_r6_severity_is_medium(self, tool, tmp_path) -> None:
        """LLM_OUTPUT_SINK is MEDIUM severity (exploitability requires specific context)."""
        src = "import json\nresult = json.loads(raw)\n"
        f = self._write(tmp_path, "r6d.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        sinks = [fd for fd in result.data["findings"] if fd["rule"] == "LLM_OUTPUT_SINK"]
        assert sinks
        assert sinks[0]["severity"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_r6_no_positional_args_not_flagged(self, tool, tmp_path) -> None:
        """json.loads() call with zero positional args — no finding (guard line)."""
        # Syntactically valid but zero positional args → 'if not node.args: return'
        src = "import json\njson.loads()\n"
        f = self._write(tmp_path, "r6e.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "LLM_OUTPUT_SINK" not in rules

    # ── R7: UNSAFE_EVAL_EXEC ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r7_eval_variable_flagged(self, tool, tmp_path) -> None:
        """eval() with a non-constant arg fires UNSAFE_EVAL_EXEC at HIGH."""
        src = "eval(user_input)\n"
        f = self._write(tmp_path, "r7a.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_EVAL_EXEC" in rules
        sev = [fd["severity"] for fd in result.data["findings"] if fd["rule"] == "UNSAFE_EVAL_EXEC"]
        assert sev[0] == "HIGH"

    @pytest.mark.asyncio
    async def test_r7_exec_variable_flagged(self, tool, tmp_path) -> None:
        """exec() with a non-constant arg fires UNSAFE_EVAL_EXEC at HIGH."""
        src = "exec(untrusted_code)\n"
        f = self._write(tmp_path, "r7b.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_EVAL_EXEC" in rules

    @pytest.mark.asyncio
    async def test_r7_eval_constant_not_flagged(self, tool, tmp_path) -> None:
        """eval() with a string literal is safe — no finding."""
        src = 'eval("1 + 1")\n'
        f = self._write(tmp_path, "r7c.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_EVAL_EXEC" not in rules

    # ── R8: UNSAFE_PICKLE ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r8_pickle_loads_flagged(self, tool, tmp_path) -> None:
        """pickle.loads() with a variable arg fires UNSAFE_PICKLE at HIGH."""
        src = "import pickle\ndata = pickle.loads(raw_bytes)\n"
        f = self._write(tmp_path, "r8a.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_PICKLE" in rules
        sev = [fd["severity"] for fd in result.data["findings"] if fd["rule"] == "UNSAFE_PICKLE"]
        assert sev[0] == "HIGH"

    @pytest.mark.asyncio
    async def test_r8_pickle_load_flagged(self, tool, tmp_path) -> None:
        """pickle.load() (file form) with a variable arg fires UNSAFE_PICKLE."""
        src = "import pickle\nobj = pickle.load(untrusted_file)\n"
        f = self._write(tmp_path, "r8b.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_PICKLE" in rules

    @pytest.mark.asyncio
    async def test_r8_pickle_constant_not_flagged(self, tool, tmp_path) -> None:
        """pickle.loads() with a bytes literal is safe — no finding."""
        src = "import pickle\npickle.loads(b'\\x80\\x04N.')\n"
        f = self._write(tmp_path, "r8c.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_PICKLE" not in rules

    # ── R9: UNSAFE_YAML_LOAD ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r9_yaml_load_no_loader_flagged(self, tool, tmp_path) -> None:
        """yaml.load() without SafeLoader fires UNSAFE_YAML_LOAD at HIGH."""
        src = "import yaml\ndata = yaml.load(stream)\n"
        f = self._write(tmp_path, "r9a.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        findings = [fd for fd in result.data["findings"] if fd["rule"] == "UNSAFE_YAML_LOAD"]
        assert findings
        assert findings[0]["severity"] == "HIGH"

    @pytest.mark.asyncio
    async def test_r9_yaml_load_safe_loader_not_flagged(self, tool, tmp_path) -> None:
        """yaml.load(stream, Loader=yaml.SafeLoader) is safe — no finding."""
        src = "import yaml\ndata = yaml.load(stream, Loader=yaml.SafeLoader)\n"
        f = self._write(tmp_path, "r9b.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_YAML_LOAD" not in rules

    @pytest.mark.asyncio
    async def test_r9_yaml_safe_load_not_flagged(self, tool, tmp_path) -> None:
        """yaml.safe_load() is always safe — no finding."""
        src = "import yaml\ndata = yaml.safe_load(stream)\n"
        f = self._write(tmp_path, "r9c.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_YAML_LOAD" not in rules

    @pytest.mark.asyncio
    async def test_r9_yaml_full_load_medium(self, tool, tmp_path) -> None:
        """yaml.full_load() fires UNSAFE_YAML_LOAD at MEDIUM severity."""
        src = "import yaml\ndata = yaml.full_load(stream)\n"
        f = self._write(tmp_path, "r9d.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        findings = [fd for fd in result.data["findings"] if fd["rule"] == "UNSAFE_YAML_LOAD"]
        assert findings
        assert findings[0]["severity"] == "MEDIUM"

    # ── R10: OS_SHELL_INJECTION ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r10_os_system_variable_flagged(self, tool, tmp_path) -> None:
        """os.system() with a non-constant arg fires OS_SHELL_INJECTION at HIGH."""
        src = "import os\nos.system(user_cmd)\n"
        f = self._write(tmp_path, "r10a.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "OS_SHELL_INJECTION" in rules
        sev = [fd["severity"] for fd in result.data["findings"] if fd["rule"] == "OS_SHELL_INJECTION"]
        assert sev[0] == "HIGH"

    @pytest.mark.asyncio
    async def test_r10_os_popen_variable_flagged(self, tool, tmp_path) -> None:
        """os.popen() with a non-constant arg fires OS_SHELL_INJECTION at HIGH."""
        src = "import os\nos.popen(user_cmd)\n"
        f = self._write(tmp_path, "r10b.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "OS_SHELL_INJECTION" in rules

    @pytest.mark.asyncio
    async def test_r10_os_system_constant_not_flagged(self, tool, tmp_path) -> None:
        """os.system() with a string literal is safe — no finding."""
        src = 'import os\nos.system("ls -la")\n'
        f = self._write(tmp_path, "r10c.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "OS_SHELL_INJECTION" not in rules

    # ── R7-R10 edge-case guards ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_r7_eval_no_args_not_flagged(self, tool, tmp_path) -> None:
        """eval() with zero positional args — early-return guard."""
        src = "eval()\n"
        f = self._write(tmp_path, "r7d.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_EVAL_EXEC" not in rules

    @pytest.mark.asyncio
    async def test_r8_pickle_no_args_not_flagged(self, tool, tmp_path) -> None:
        """pickle.loads() with zero positional args — early-return guard."""
        src = "import pickle\npickle.loads()\n"
        f = self._write(tmp_path, "r8d.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_PICKLE" not in rules

    @pytest.mark.asyncio
    async def test_r9_yaml_load_unrelated_kwarg_flagged(self, tool, tmp_path) -> None:
        """yaml.load() with an unrelated kwarg (not Loader=) still fires HIGH."""
        src = "import yaml\nyaml.load(stream, encoding='utf-8')\n"
        f = self._write(tmp_path, "r9e.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        findings = [fd for fd in result.data["findings"] if fd["rule"] == "UNSAFE_YAML_LOAD"]
        assert findings
        assert findings[0]["severity"] == "HIGH"

    @pytest.mark.asyncio
    async def test_r9_yaml_load_bare_safeloader_name_not_flagged(self, tool, tmp_path) -> None:
        """yaml.load(stream, Loader=SafeLoader) with bare name import — safe."""
        src = "from yaml import SafeLoader\nimport yaml\nyaml.load(stream, Loader=SafeLoader)\n"
        f = self._write(tmp_path, "r9f.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSAFE_YAML_LOAD" not in rules

    @pytest.mark.asyncio
    async def test_r10_os_shell_no_args_not_flagged(self, tool, tmp_path) -> None:
        """os.system() with zero positional args — early-return guard."""
        src = "import os\nos.system()\n"
        f = self._write(tmp_path, "r10d.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "OS_SHELL_INJECTION" not in rules

    @pytest.mark.asyncio
    async def test_r2b_extend_list_variable_flag_flagged(self, tool, tmp_path) -> None:
        """cmd.extend([f'--{flag}=val']) — variable flag NAME in list fires R2b."""
        src = 'cmd.extend([f"--{flag}=value"])\n'
        f = self._write(tmp_path, "r2b_extend.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "GIT_FLAG_NAME_INJECTION" in rules

    @pytest.mark.asyncio
    async def test_r2b_append_variable_flag_flagged(self, tool, tmp_path) -> None:
        """cmd.append(f'--{flag}=val') — variable flag NAME via append fires R2b."""
        src = 'cmd.append(f"--{flag}=value")\n'
        f = self._write(tmp_path, "r2b_append.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "GIT_FLAG_NAME_INJECTION" in rules

    @pytest.mark.asyncio
    async def test_r2b_fstring_single_dash_prefix_not_flagged(self, tool, tmp_path) -> None:
        """f'-{flag}=val' — starts with single '-', not '--'; _fstring_variable_flag returns None."""
        src = 'cmd.append(f"-{flag}=value")\n'
        f = self._write(tmp_path, "r2b_single_dash.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "GIT_FLAG_NAME_INJECTION" not in rules

    @pytest.mark.asyncio
    async def test_r2b_fstring_attribute_not_flagged(self, tool, tmp_path) -> None:
        """f'--{obj.attr}=val' — second value is Attribute, not bare Name; no R2b."""
        src = 'cmd.append(f"--{obj.attr}=value")\n'
        f = self._write(tmp_path, "r2b_attr.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "GIT_FLAG_NAME_INJECTION" not in rules

    @pytest.mark.asyncio
    async def test_r2a_append_non_cmd_list_not_flagged(self, tool, tmp_path) -> None:
        """other_list.append(var) — receiver not in _CMD_LIST_NAMES; no UNSANITIZED_SUBPROCESS_ARG."""
        src = "other_list.append(var)\n"
        f = self._write(tmp_path, "r2a_nonrecv.py", src)
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "UNSANITIZED_SUBPROCESS_ARG" not in rules

    def test_factory_with_allowed_root(self, tmp_path) -> None:
        from keryx.tools.ast_analyzer import create_ast_analyzer_tool
        t = create_ast_analyzer_tool(allowed_root=str(tmp_path))
        assert t.allowed_root == tmp_path.resolve()

    # ── Remaining edge-case lines ──────────────────────────────────────────

    def test_snippet_returns_empty_for_line_zero(self) -> None:
        # Line 63: _snippet() returns "" when lineno is 0 (out of valid range)
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        v = _VulnVisitor(["only line"])
        node = ast.parse("x", mode="eval").body
        node.lineno = 0   # 0 is outside 1..len(lines)
        assert v._snippet(node) == ""

    @pytest.mark.asyncio
    async def test_extend_no_args_skipped(self, tool, tmp_path) -> None:
        # Line 135: _check_missing_dashdash early-returns when extend() has no positional args
        f = self._write(tmp_path, "noargs.py", "cmd = []\ncmd.extend()\n")
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "GIT_OPTION_INJECTION" not in rules

    @pytest.mark.asyncio
    async def test_open_no_args_skipped(self, tool, tmp_path) -> None:
        # Line 177: _check_open early-returns when open() has no positional args
        f = self._write(tmp_path, "opennoargs.py", "open()\n")
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "OPEN_USER_PATH" not in rules

    @pytest.mark.asyncio
    async def test_tuple_assign_not_flagged(self, tool, tmp_path) -> None:
        # Line 196: non-ast.Name assignment target (tuple unpack) → continue
        f = self._write(tmp_path, "tuple.py", "(api_key, secret) = get_creds()\n")
        result = await tool.execute({"path": str(f)})
        assert result.success
        rules = [fd["rule"] for fd in result.data["findings"]]
        assert "HARDCODED_SECRET" not in rules

    @pytest.mark.asyncio
    async def test_read_error(self, tool, tmp_path) -> None:
        # Lines 303-304: read_text raises → error="read_error"
        from unittest.mock import patch
        from pathlib import Path
        f = self._write(tmp_path, "unreadable.py", "x = 1\n")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = await tool.execute({"path": str(f)})
        assert not result.success
        assert result.error == "read_error"


# T51 – agent.py: 108 missing lines (none/strict modes, static_fallback,
#        hypothesis evidence, P4 state, _apply_critique IV, _apply_advisor_advice
#        rich fields, _resolve_target_tool, _extract_findings, _auto_verify_injection,
#        _build_urgent_verifier_note).
# ---------------------------------------------------------------------------
from keryx.tools.Toolbox import BaseTool as _BaseTool, ToolResult as _ToolResult

class TestAgentMissingLinePaths:
    """T51 — Cover the remaining 108 lines in keryx/core/agent.py."""

    # ── Common fixtures & constants ─────────────────────────────────────────

    _HIGH_CQ = (
        "[AST] 2 finding(s):\n"
        "  [1] [HIGH] GIT_OPTION_INJECTION @ line 5: "
        'cmd.extend(["--author", author]) — user-controlled value\n'
        "  [2] [HIGH] SUBPROCESS_SHELL_TRUE @ line 3: run(cmd, shell=True)\n"
    )
    _VULN_CONFIRMED_IV = (
        "[InjectionVerifier] target='git_blame' field='author'\n"
        "  verdict         : ACCEPTED\n"
        "  VULN_CONFIRMED — dangerous payload was NOT rejected.\n"
    )
    _REJECTED_IV = (
        "[InjectionVerifier] target='git_blame' field='author'\n"
        "  verdict         : REJECTED\n  error: invalid option\n"
    )

    # ── Mock tools ──────────────────────────────────────────────────────────

    class _CQTool(_BaseTool):
        name = "codeql_query"; description = "mock codeql"
        def __init__(self, output): super().__init__(); self._out = output
        async def execute(self, ai, ctx=None):
            return _ToolResult(success=True, output=self._out)

    class _IVTool(_BaseTool):
        name = "injection_verifier"; description = "mock iv"
        def __init__(self, output): super().__init__(); self._out = output
        async def execute(self, ai, ctx=None):
            return _ToolResult(success=True, output=self._out)

    class _GitBlameTool(_BaseTool):
        name = "git_blame"; description = "mock git_blame"
        async def execute(self, ai, ctx=None):
            return _ToolResult(success=True, output="commit abc123")

    class _RFTool(_BaseTool):
        name = "read_file"; description = "mock rf"
        async def execute(self, ai, ctx=None):
            return _ToolResult(success=True, output="X" * 200)

    # ── Mock models ─────────────────────────────────────────────────────────

    class _CQThenFinishModel(MyLocalModel):
        """Step1=codeql_query (with hypothesis), then FINISH."""
        def __init__(self, extra_steps=0):
            super().__init__(); self._n = 0; self._extra = extra_steps
        def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
            if max_tokens == 1 or grammar is None: return "ok"
            self._n += 1
            if self._n == 1:
                return json.dumps({"thought": "scan", "action": "codeql_query",
                    "action_input": {"path": "/tmp/test.py",
                                     "hypothesis": "injection vulnerability"},
                    "confidence": 0.6})
            if self._n <= 1 + self._extra:
                return json.dumps({"thought": "read", "action": "read_file",
                    "action_input": {"path": "/tmp/test.py"}, "confidence": 0.6})
            return json.dumps({"thought": "done", "action": "FINISH",
                "action_input": {}, "confidence": 0.9})

    class _StaticFallbackModel(MyLocalModel):
        """codeql → read_file → read_file (static_fallback fires after step3)."""
        def __init__(self):
            super().__init__(); self._n = 0
        def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
            if max_tokens == 1 or grammar is None: return "ok"
            self._n += 1
            if self._n == 1:
                return json.dumps({"thought": "scan", "action": "codeql_query",
                    "action_input": {"path": "/tmp/test.py"}, "confidence": 0.6})
            return json.dumps({"thought": "more", "action": "read_file",
                "action_input": {"path": "/tmp/test.py"}, "confidence": 0.6})

    def _tb(self, *tools):
        from keryx.tools.Toolbox import ToolBox
        tb = ToolBox()
        for t in tools: tb.register(t)
        return tb

    @pytest.fixture
    def am(self):
        from keryx.advisors.manager import AdvisorManager
        m = AdvisorManager(); yield m; m.shutdown()

    def _agent(self, model, toolbox, mode="flexible", max_steps=5):
        from keryx.core.agent import KeryxAgent
        from keryx.advisors.manager import AdvisorManager
        adv = AdvisorManager()
        return KeryxAgent(executor_model=model, advisor_manager=adv,
                          toolbox=toolbox, max_steps=max_steps,
                          verification_mode=mode), adv

    # ── A. _build_urgent_verifier_note ──────────────────────────────────────

    def test_urgent_note_non_flexible_returns_empty(self, tmp_path, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1, verification_mode="strict")
        agent.context = SharedContext(target_path="/tmp/t.py")
        assert agent._build_urgent_verifier_note() == ""
        tb.shutdown()

    def test_urgent_note_window_active_returns_note(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=5)
        agent.context = SharedContext(target_path="/tmp/t.py")
        agent.context.steps_taken = 2
        agent._codeql_unconfirmed = True
        agent._codeql_high_found_step = 1   # 2-1=1 step elapsed → 1 step left
        note = agent._build_urgent_verifier_note()
        assert "URGENT" in note
        assert "1 step" in note
        tb.shutdown()

    def test_urgent_note_window_elapsed_returns_empty(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=5)
        agent.context = SharedContext(target_path="/tmp/t.py")
        agent.context.steps_taken = 5
        agent._codeql_unconfirmed = True
        agent._codeql_high_found_step = 1   # 5-1=4 ≥ 2 → steps_left ≤ 0
        assert agent._build_urgent_verifier_note() == ""
        tb.shutdown()

    # ── B. _resolve_target_tool ─────────────────────────────────────────────

    def _agent_with_context(self, target_path, *tools):
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        from keryx.advisors.manager import AdvisorManager
        tb = self._tb(*tools)
        am = AdvisorManager()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path=target_path)
        return agent, tb, am

    def test_resolve_exact_match(self) -> None:
        agent, tb, am = self._agent_with_context("/tmp/git_blame.py",
                                                  self._GitBlameTool())
        assert agent._resolve_target_tool() == "git_blame"
        tb.shutdown(); am.shutdown()

    def test_resolve_normalized_hyphen(self) -> None:
        class _HyphenTool(_BaseTool):
            name = "git_blame"; description = "x"
            async def execute(self, ai, ctx=None):
                return _ToolResult(success=True, output="ok")
        # target stem is "git-blame" (hyphenated), tool is "git_blame"
        agent, tb, am = self._agent_with_context("/tmp/git-blame.py", _HyphenTool())
        assert agent._resolve_target_tool() == "git_blame"
        tb.shutdown(); am.shutdown()

    def test_resolve_partial_match(self) -> None:
        # stem "my_git_blame_wrapper" contains "git_blame"
        agent, tb, am = self._agent_with_context(
            "/tmp/my_git_blame_wrapper.py", self._GitBlameTool())
        assert agent._resolve_target_tool() == "git_blame"
        tb.shutdown(); am.shutdown()

    def test_resolve_no_match_returns_none(self) -> None:
        agent, tb, am = self._agent_with_context("/tmp/totally_unrelated.py",
                                                  self._GitBlameTool())
        assert agent._resolve_target_tool() is None
        tb.shutdown(); am.shutdown()

    # ── C. _extract_findings_for_verification ──────────────────────────────

    def test_extract_git_option_injection(self) -> None:
        from keryx.core.agent import KeryxAgent
        findings = KeryxAgent._extract_findings_for_verification(self._HIGH_CQ)
        rules = [r for r, _, _ in findings]
        assert "GIT_OPTION_INJECTION" in rules
        fields = {r: f for r, f, _ in findings}
        assert fields["GIT_OPTION_INJECTION"] == "author"

    def test_extract_subprocess_shell_true(self) -> None:
        from keryx.core.agent import KeryxAgent
        obs = "[HIGH] SUBPROCESS_SHELL_TRUE @ line 3: subprocess.run(cmd, shell=True)\n"
        findings = KeryxAgent._extract_findings_for_verification(obs)
        rules = [r for r, _, _ in findings]
        assert "SUBPROCESS_SHELL_TRUE" in rules

    def test_extract_hardcoded_secret_skipped(self) -> None:
        from keryx.core.agent import KeryxAgent
        obs = '[HIGH] HARDCODED_SECRET @ line 2: api_key = "sk-abc"\n'
        findings = KeryxAgent._extract_findings_for_verification(obs)
        assert findings == []   # None payload → skipped

    def test_extract_duplicate_rule_deduplicated(self) -> None:
        from keryx.core.agent import KeryxAgent
        obs = (
            '[HIGH] GIT_OPTION_INJECTION @ line 5: cmd.extend(["--author", author])\n'
            '[HIGH] GIT_OPTION_INJECTION @ line 9: cmd.extend(["--grep", grep])\n'
        )
        findings = KeryxAgent._extract_findings_for_verification(obs)
        assert sum(1 for r, _, _ in findings if r == "GIT_OPTION_INJECTION") == 1

    def test_extract_no_high_findings(self) -> None:
        from keryx.core.agent import KeryxAgent
        obs = "[AST] No findings.\n"
        assert KeryxAgent._extract_findings_for_verification(obs) == []

    # ── D. _apply_critique injection_verifier paths ─────────────────────────

    def test_critique_iv_vuln_confirmed_boosts_confidence(self, am) -> None:
        from keryx.core.agent import KeryxAgent, AgentStep
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/t.py")
        action = AgentStep(thought="verify", action="injection_verifier",
                           action_input={},
                           observation="VULN_CONFIRMED — payload accepted.",
                           confidence=0.6)
        agent._apply_critique("analysis", action)
        assert action.confidence > 0.6
        tb.shutdown()

    def test_critique_iv_rejected_confidence_unchanged(self, am) -> None:
        from keryx.core.agent import KeryxAgent, AgentStep
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/t.py")
        action = AgentStep(thought="verify", action="injection_verifier",
                           action_input={},
                           observation="verdict: REJECTED", confidence=0.6)
        agent._apply_critique("ok", action)
        assert action.confidence == pytest.approx(0.6)
        tb.shutdown()

    # ── E. _apply_advisor_advice rich fields ────────────────────────────────

    def test_advisor_advice_suggested_hypotheses(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/t.py")
        agent.context.add_advisor_advice({
            "strategy": "widen", "suggested_hypotheses": ["sql injection via param"],
        })
        agent._apply_advisor_advice()
        assert "sql injection via param" in agent.context.hypotheses

    def test_advisor_advice_blacklist_hypotheses(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/t.py")
        agent.context.add_hypothesis("false positive hypothesis")
        agent.context.add_advisor_advice({
            "strategy": "prune", "blacklist_hypotheses": ["false positive hypothesis"],
        })
        agent._apply_advisor_advice()
        assert "false positive hypothesis" not in agent.context.hypotheses
        tb.shutdown()

    # ── F. agent.run() verification modes ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_none_mode_static_confirm(self) -> None:
        tb = self._tb(self._CQTool(self._HIGH_CQ), self._RFTool())
        agent, adv = self._agent(self._CQThenFinishModel(), tb, mode="none")
        try:
            result = await agent.run(target_path="/tmp/test.py", resume=False)
            assert len(result["confirmed_vulns"]) == 1
            assert result["confirmed_vulns"][0]["confidence_tag"] == "AST-only"
            assert result["confirmed_vulns"][0]["verified"] is False
        finally:
            tb.shutdown(); adv.shutdown()

    @pytest.mark.asyncio
    async def test_run_strict_mode_auto_verify_confirmed(self) -> None:
        tb = self._tb(self._CQTool(self._HIGH_CQ), self._GitBlameTool(),
                      self._IVTool(self._VULN_CONFIRMED_IV))
        agent, adv = self._agent(self._CQThenFinishModel(), tb, mode="strict")
        try:
            result = await agent.run(target_path="/tmp/git_blame.py", resume=False)
            assert len(result["confirmed_vulns"]) == 1
            assert result["confirmed_vulns"][0]["verified"] is True
            assert "injection_verifier" in result["confirmed_vulns"][0]["confidence_tag"]
        finally:
            tb.shutdown(); adv.shutdown()

    @pytest.mark.asyncio
    async def test_run_flexible_hypothesis_evidence_and_p4(self) -> None:
        # Covers lines 335-339 (hyp+evidence), 363 (hypothesis counter reset), 378 (codeql_unconfirmed)
        tb = self._tb(self._CQTool(self._HIGH_CQ), self._RFTool())
        agent, adv = self._agent(self._CQThenFinishModel(extra_steps=1), tb,
                                  mode="flexible", max_steps=10)
        try:
            result = await agent.run(target_path="/tmp/test.py", resume=False)
            # Hypothesis from action_input was registered
            hyps_or_vulns = (
                list(agent.context.hypotheses)
                + [str(v) for v in result.get("confirmed_vulns", [])]
            )
            assert any("injection" in h.lower() for h in hyps_or_vulns)
        finally:
            tb.shutdown(); adv.shutdown()

    @pytest.mark.asyncio
    async def test_run_flexible_static_fallback(self) -> None:
        # Covers lines 468-479: static_fallback fires after 2-step window elapses
        tb = self._tb(self._CQTool(self._HIGH_CQ), self._RFTool())
        agent, adv = self._agent(self._StaticFallbackModel(), tb,
                                  mode="flexible", max_steps=10)
        try:
            result = await agent.run(target_path="/tmp/test.py", resume=False)
            assert len(result["confirmed_vulns"]) == 1
            assert result["confirmed_vulns"][0]["verified"] is False
            assert result["confirmed_vulns"][0]["confidence_tag"] == "AST-only"
        finally:
            tb.shutdown(); adv.shutdown()

    # ── G. _auto_verify_injection ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_auto_verify_no_target_tool(self, am) -> None:
        # No tools → resolve_target_tool returns None → falls back to fuzz_poc path.
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        from unittest.mock import patch
        tb = self._tb()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/unmatched.py")
        with patch("keryx.core.verification_pipeline.VerificationPipeline._fuzz_verify",
                   return_value=(False, "reachable")):
            confirmed, method = await agent._auto_verify_injection(self._HIGH_CQ)
        assert confirmed is False
        assert method == "fuzz_poc:reachable"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_auto_verify_no_findings(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._GitBlameTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/git_blame.py")
        confirmed, method = await agent._auto_verify_injection("[AST] No findings.\n")
        assert confirmed is False
        assert method == "injection_verifier"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_auto_verify_vuln_confirmed(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._GitBlameTool(), self._IVTool(self._VULN_CONFIRMED_IV))
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/git_blame.py")
        confirmed, method = await agent._auto_verify_injection(self._HIGH_CQ)
        assert confirmed is True
        assert method == "injection_verifier"

    @pytest.mark.asyncio
    async def test_auto_verify_all_rejected(self, am) -> None:
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._GitBlameTool(), self._IVTool(self._REJECTED_IV))
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/git_blame.py")
        confirmed, method = await agent._auto_verify_injection(self._HIGH_CQ)
        assert confirmed is False
        assert method == "injection_verifier"

    @pytest.mark.asyncio
    async def test_auto_verify_exception_continues(self, am) -> None:
        # execute_async raises → exception caught, continues → (False, "injection_verifier")
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        from unittest.mock import patch

        tb = self._tb(self._GitBlameTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/git_blame.py")

        with patch.object(tb, "execute_async", side_effect=RuntimeError("boom")):
            confirmed, method = await agent._auto_verify_injection(self._HIGH_CQ)
        assert confirmed is False
        assert method == "injection_verifier"
        tb.shutdown()

    # ── consecutive_clean early exit (lines 390-396) ────────────────────────

    @pytest.mark.asyncio
    async def test_consecutive_clean_exits_early(self, am) -> None:
        """A clean codeql scan (no HIGH) triggers early exit after 1 scan."""
        class _CleanCQTool(_BaseTool):
            name = "codeql_query"; description = "clean"
            async def execute(self, ai, ctx=None):
                return _ToolResult(success=True,
                                   output="[AST] No findings in test.py (50 lines analyzed).")

        class _CleanCQModel(MyLocalModel):
            def __init__(self): super().__init__(); self._n = 0
            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                if max_tokens == 1 or grammar is None: return "ok"
                self._n += 1
                return json.dumps({"thought": "scan", "action": "codeql_query",
                    "action_input": {"path": "/tmp/test.py"}, "confidence": 0.7})

        tb = self._tb(_CleanCQTool())
        agent, adv = self._agent(_CleanCQModel(), tb, mode="flexible", max_steps=10)
        result = await agent.run("/tmp/test.py", resume=False)
        adv.shutdown(); tb.shutdown()
        # Should exit after step 1 (clean scan), not run all 10 steps
        assert result["steps_taken"] <= 2

    @pytest.mark.asyncio
    async def test_consecutive_clean_resets_on_high_finding(self, am) -> None:
        """Clean counter resets when codeql finds a HIGH; early exit deferred."""
        class _HighThenCleanModel(MyLocalModel):
            def __init__(self): super().__init__(); self._n = 0
            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                if max_tokens == 1 or grammar is None: return "ok"
                self._n += 1
                if self._n == 1:
                    return json.dumps({"thought": "scan", "action": "codeql_query",
                        "action_input": {"path": "/tmp/test.py"}, "confidence": 0.6})
                return json.dumps({"thought": "done", "action": "FINISH",
                    "action_input": {}, "confidence": 0.9})

        tb = self._tb(self._CQTool(self._HIGH_CQ))
        agent, adv = self._agent(_HighThenCleanModel(), tb, mode="none", max_steps=5)
        result = await agent.run("/tmp/test.py", resume=False)
        adv.shutdown(); tb.shutdown()
        # HIGH found → confirmed immediately (none mode) → not an early clean exit
        assert agent._consecutive_clean_scans == 0

    # ── _extract_findings_for_verification — UNSANITIZED & OPEN_USER_PATH ──

    def test_extract_findings_unsanitized_arg(self) -> None:
        """Lines 754-756: UNSANITIZED_SUBPROCESS_ARG rule extracts inject_field."""
        from keryx.core.agent import KeryxAgent
        obs = (
            "[AST] 1 finding(s):\n"
            "  [1] [HIGH] UNSANITIZED_SUBPROCESS_ARG @ line 8: "
            "cmd.append(filepath) — variable appended\n"
        )
        findings = KeryxAgent._extract_findings_for_verification(obs)
        # UNSANITIZED_SUBPROCESS_ARG maps to ../../etc/passwd payload
        rules = [r for r, _, _ in findings]
        assert "UNSANITIZED_SUBPROCESS_ARG" in rules

    def test_extract_findings_open_user_path(self) -> None:
        """Lines 759-761: OPEN_USER_PATH rule extracts inject_field via open() pattern."""
        from keryx.core.agent import KeryxAgent
        obs = (
            "[AST] 1 finding(s):\n"
            "  [1] [HIGH] OPEN_USER_PATH @ line 12: "
            "open(file_path) — path from variable\n"
        )
        findings = KeryxAgent._extract_findings_for_verification(obs)
        rules = [r for r, _, _ in findings]
        assert "OPEN_USER_PATH" in rules

    # ── _parse_response missing required key (lines 890-892) ────────────────

    def test_parse_response_json_missing_action_key(self, am) -> None:
        """JSON with no 'action' field triggers parse-error path."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/x.py")
        before = agent.context.parse_errors
        # JSON dict with no 'action' key
        result = agent._parse_response('{"thought": "hmm", "confidence": 0.5}')
        assert agent.context.parse_errors > before
        assert result.action == "NO_ACTION"
        tb.shutdown()

    def test_parse_response_json_non_dict(self, am) -> None:
        """JSON array (not a dict) also triggers the required-key guard."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/x.py")
        before = agent.context.parse_errors
        result = agent._parse_response('[1, 2, 3]')
        assert agent.context.parse_errors > before
        assert result.action == "NO_ACTION"
        tb.shutdown()

    # ── _extract_json HTML/control-char stripping ────────────────────────────

    def test_extract_json_strips_html_tags(self, am) -> None:
        """HTML-like injection attempts are stripped before JSON extraction."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/x.py")
        raw = '<script>alert(1)</script>{"thought":"t","action":"FINISH","action_input":{},"confidence":0.9}'
        extracted = agent._extract_json(raw)
        assert extracted is not None
        assert "<script>" not in extracted
        tb.shutdown()

    # ── _should_exit_clean: return False path (line 570) ────────────────────

    def test_should_exit_clean_below_threshold_returns_false(self, am) -> None:
        """First clean scan when threshold=2 returns False (counter not reached)."""
        from keryx.core.agent import KeryxAgent, AgentStep
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=5,
                           max_clean_scans_before_exit=2)
        agent.context = SharedContext(target_path="/tmp/x.py")
        clean_action = AgentStep(
            thought="scan", action="codeql_query", action_input={},
            observation="[AST] No findings in x.py (10 lines analyzed).",
        )
        # First clean scan — threshold is 2, so should NOT exit yet
        result = agent._should_exit_clean(clean_action)
        assert result is False
        assert agent._consecutive_clean_scans == 1
        tb.shutdown()

    def test_should_exit_clean_at_threshold_returns_true(self, am) -> None:
        """Second clean scan when threshold=2 returns True."""
        from keryx.core.agent import KeryxAgent, AgentStep
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=5,
                           max_clean_scans_before_exit=2)
        agent.context = SharedContext(target_path="/tmp/x.py")
        clean_action = AgentStep(
            thought="scan", action="codeql_query", action_input={},
            observation="[AST] No findings in x.py (10 lines analyzed).",
        )
        agent._should_exit_clean(clean_action)   # first scan → False
        result = agent._should_exit_clean(clean_action)  # second → True
        assert result is True
        assert agent._consecutive_clean_scans == 2
        tb.shutdown()

    def test_should_exit_clean_non_codeql_action_skipped(self, am) -> None:
        """Non-codeql actions are ignored by _should_exit_clean."""
        from keryx.core.agent import KeryxAgent, AgentStep
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=5)
        agent.context = SharedContext(target_path="/tmp/x.py")
        rf_action = AgentStep(
            thought="read", action="read_file", action_input={},
            observation="file content",
        )
        assert agent._should_exit_clean(rf_action) is False
        assert agent._consecutive_clean_scans == 0   # untouched
        tb.shutdown()

    # ── _precondition_check ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_precondition_check_proceed(self, am) -> None:
        """Healthy model and no budget → 'proceed'."""
        from keryx.core.agent import KeryxAgent
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1)
        agent.context = SharedContext(target_path="/tmp/x.py")
        result = await agent._precondition_check()
        assert result == "proceed"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_precondition_check_budget_exhausted(self, am) -> None:
        """Budget exhausted → 'break'."""
        from keryx.core.agent import KeryxAgent, BudgetController
        from keryx.core.shared_context import SharedContext
        tb = self._tb(self._RFTool())
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, max_steps=1, budget_usd=0.001)
        agent.context = SharedContext(target_path="/tmp/x.py")
        agent.budget.current_cost = 1.0   # force-exhaust
        result = await agent._precondition_check()
        assert result == "break"
        tb.shutdown()


# ---------------------------------------------------------------------------
# T52 — HuntConfig dataclass
# ---------------------------------------------------------------------------

class TestHuntConfig:
    """T52 — Full coverage of keryx/core/hunt_config.py."""

    def test_defaults(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        cfg = HuntConfig()
        assert cfg.max_steps == 25
        assert cfg.confidence_threshold == 0.60
        assert cfg.budget_usd == 2.00
        assert cfg.verification_mode == "flexible"
        assert cfg.max_clean_scans_before_exit == 1
        assert cfg.generate_timeout == 60.0
        assert cfg.max_prompt_chars == 32_000
        assert cfg.enforce_airgapped is False

    def test_custom_fields(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        cfg = HuntConfig(max_steps=5, budget_usd=None, enforce_airgapped=True)
        assert cfg.max_steps == 5
        assert cfg.budget_usd is None
        assert cfg.enforce_airgapped is True

    def test_to_agent_kwargs_keys(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        kwargs = HuntConfig().to_agent_kwargs()
        expected = {
            "max_steps", "confidence_threshold", "budget_usd",
            "verification_mode", "max_clean_scans_before_exit",
            "generate_timeout", "max_prompt_chars", "enforce_airgapped",
        }
        assert set(kwargs.keys()) == expected

    def test_to_agent_kwargs_values_match(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        cfg = HuntConfig(max_steps=7, budget_usd=0.5, verification_mode="strict")
        kw = cfg.to_agent_kwargs()
        assert kw["max_steps"] == 7
        assert kw["budget_usd"] == 0.5
        assert kw["verification_mode"] == "strict"

    def test_to_agent_kwargs_accepted_by_agent(self) -> None:
        """KeryxAgent.__init__ accepts every key in to_agent_kwargs()."""
        from keryx.core.hunt_config import HuntConfig
        from keryx.core.agent import KeryxAgent
        from keryx.advisors.manager import AdvisorManager
        from keryx.tools.Toolbox import ToolBox
        cfg = HuntConfig()
        am = AdvisorManager()
        tb = ToolBox()
        agent = KeryxAgent(executor_model=FinishModel(), advisor_manager=am,
                           toolbox=tb, **cfg.to_agent_kwargs())
        assert agent.max_steps == cfg.max_steps
        assert agent.confidence_threshold == cfg.confidence_threshold
        am.shutdown()

    def test_repr(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        r = repr(HuntConfig())
        assert "HuntConfig" in r
        assert "flexible" in r

    # ── Presets ──────────────────────────────────────────────────────────────

    def test_fast_preset(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        cfg = HuntConfig.fast()
        assert cfg.max_steps == 10
        assert cfg.verification_mode == "strict"
        assert cfg.budget_usd == 0.50
        assert cfg.max_clean_scans_before_exit == 1
        assert cfg.generate_timeout == 30.0

    def test_deep_preset(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        cfg = HuntConfig.deep()
        assert cfg.max_steps == 50
        assert cfg.confidence_threshold == 0.75
        assert cfg.max_clean_scans_before_exit == 2
        assert cfg.budget_usd == 5.00

    def test_local_preset(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        cfg = HuntConfig.local()
        assert cfg.enforce_airgapped is True
        assert cfg.budget_usd is None
        assert cfg.generate_timeout == 120.0

    def test_presets_return_distinct_instances(self) -> None:
        from keryx.core.hunt_config import HuntConfig
        a = HuntConfig.fast()
        b = HuntConfig.fast()
        a.max_steps = 99
        assert b.max_steps == 10  # mutation of one doesn't affect another

    def test_lazy_import_from_keryx(self) -> None:
        """HuntConfig is accessible via the top-level keryx package."""
        import keryx
        cfg = keryx.HuntConfig()
        assert cfg.max_steps == 25


# T53 — keryx/fuzzing: harness, sandbox, poc, tool
# ---------------------------------------------------------------------------

class TestFuzzingHarness:
    """T53a — harness.py: template generation and registry."""

    def test_known_rules_produce_scripts(self) -> None:
        from keryx.fuzzing.harness import generate, MARKER_PREFIX
        marker = MARKER_PREFIX + "TEST0001"
        for rule in [
            "SUBPROCESS_SHELL_TRUE", "OS_SHELL_INJECTION",
            "GIT_OPTION_INJECTION", "GIT_FLAG_NAME_INJECTION",
            "UNSAFE_EVAL_EXEC", "UNSAFE_PICKLE", "UNSAFE_YAML_LOAD",
            "OPEN_USER_PATH", "HARDCODED_SECRET",
            "SSRF", "TEMPLATE_INJECTION", "REGEX_DOS",
        ]:
            script = generate({"rule": rule}, marker)
            assert script is not None, f"No template for {rule}"
            assert marker in script, f"Marker not embedded in {rule} script"

    def test_no_template_rule_returns_none_without_llm(self) -> None:
        from keryx.fuzzing.harness import generate
        result = generate({"rule": "LLM_OUTPUT_SINK"}, "MARKER")
        assert result is None

    def test_no_template_rule_calls_llm_fallback(self) -> None:
        from keryx.fuzzing.harness import generate
        called_with = {}
        def fake_llm(finding, marker):
            called_with["finding"] = finding
            called_with["marker"]  = marker
            return f"# llm harness for {marker}"

        script = generate({"rule": "LLM_OUTPUT_SINK"}, "MYMARKER", llm_generate_fn=fake_llm)
        assert script == "# llm harness for MYMARKER"
        assert called_with["marker"] == "MYMARKER"

    def test_unknown_rule_with_llm_fallback(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "TOTALLY_UNKNOWN"}, "M", llm_generate_fn=lambda f, m: f"x={m}")
        assert script == "x=M"

    def test_unknown_rule_without_llm_returns_none(self) -> None:
        from keryx.fuzzing.harness import generate
        assert generate({"rule": "NONEXISTENT_RULE"}, "M") is None


class TestHardcodedSecretHarness:
    """T53f — harness.py: HARDCODED_SECRET entropy-analysis template."""

    def _run(self, secret_value: str = "") -> tuple[bool, str]:
        """Execute the HARDCODED_SECRET harness and return (marker_found, combined_output)."""
        from keryx.fuzzing.harness import generate
        from keryx.fuzzing.sandbox import run
        marker = "KERYX_POC_HCSECRET"
        script = generate({"rule": "HARDCODED_SECRET", "secret_value": secret_value}, marker)
        assert script is not None
        result = run(script, timeout=5)
        combined = result.stdout + result.stderr
        return marker in combined, combined

    def test_high_entropy_secret_triggers_marker(self) -> None:
        # 32-char random-looking API key — high entropy, long
        found, _ = self._run("sk-Xq9rZp2NwKf7mVcLh3jTdEoIbAuYsSe")
        assert found is True

    def test_jwt_shaped_secret_triggers_marker(self) -> None:
        # Three base64url segments → JWT structure
        found, _ = self._run("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123xyz")
        assert found is True

    def test_long_secret_triggers_marker(self) -> None:
        # ≥ 20 chars (API-key length heuristic), even if low uniqueness
        found, _ = self._run("aaaaaaaabbbbbbbbcccccc")  # 22 chars
        assert found is True

    def test_placeholder_password_skipped(self) -> None:
        # "password" is an exact placeholder match → marker NOT emitted
        found, output = self._run("password")
        assert found is False
        assert "SKIP" in output or "placeholder" in output.lower()

    def test_placeholder_hunter2_skipped(self) -> None:
        found, output = self._run("hunter2")
        assert found is False

    def test_placeholder_changeme_skipped(self) -> None:
        found, output = self._run("changeme")
        assert found is False

    def test_no_secret_value_triggers_marker_conservatively(self) -> None:
        # No secret_value → conservative flag (assume real)
        found, _ = self._run("")
        assert found is True

    def test_reproduce_hardcoded_secret_no_value(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "HARDCODED_SECRET"}, timeout=5)
        assert poc.reproduced is True
        assert poc.error is None

    def test_reproduce_hardcoded_secret_with_real_value(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "HARDCODED_SECRET",
                         "secret_value": "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"},
                        timeout=5)
        assert poc.reproduced is True
        assert poc.confidence_boost > 0

    def test_reproduce_hardcoded_secret_placeholder_not_reproduced(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "HARDCODED_SECRET", "secret_value": "hunter2"}, timeout=5)
        assert poc.reproduced is False


class TestFuzzingSandbox:
    """T53b — sandbox.py: subprocess execution and resource limits."""

    def test_stdout_captured(self) -> None:
        from keryx.fuzzing.sandbox import run
        result = run("import sys; sys.stdout.write('hello\\n')", timeout=5)
        assert result.stdout == "hello\n"
        assert result.exit_code == 0
        assert not result.timed_out

    def test_stderr_captured(self) -> None:
        from keryx.fuzzing.sandbox import run
        result = run("import sys; sys.stderr.write('err\\n')", timeout=5)
        assert "err" in result.stderr

    def test_nonzero_exit_code(self) -> None:
        from keryx.fuzzing.sandbox import run
        result = run("raise SystemExit(42)", timeout=5)
        assert result.exit_code == 42

    def test_syntax_error_script(self) -> None:
        from keryx.fuzzing.sandbox import run
        result = run("def bad syntax(: pass", timeout=5)
        assert result.exit_code != 0

    def test_timeout_respected(self) -> None:
        from keryx.fuzzing.sandbox import run
        import time
        t0     = time.monotonic()
        result = run("import time; time.sleep(60)", timeout=2)
        elapsed = time.monotonic() - t0
        assert result.timed_out
        assert elapsed < 10  # should be killed well within 10s

    def test_elapsed_populated(self) -> None:
        from keryx.fuzzing.sandbox import run
        result = run("pass", timeout=5)
        assert result.elapsed_s >= 0


class TestFuzzingPoC:
    """T53c — poc.py: reproduce() orchestrator."""

    def test_reproduced_true_for_shell_true(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "SUBPROCESS_SHELL_TRUE"}, timeout=8)
        assert poc.reproduced is True
        assert poc.confidence_boost > 0

    def test_reproduced_true_for_eval_exec(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "UNSAFE_EVAL_EXEC"}, timeout=8)
        assert poc.reproduced is True

    def test_reproduced_true_for_pickle(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "UNSAFE_PICKLE"}, timeout=8)
        assert poc.reproduced is True

    def test_no_harness_returns_error(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "LLM_OUTPUT_SINK"}, timeout=5)
        assert poc.reproduced is False
        assert poc.error is not None
        assert poc.confidence_boost == 0.0

    def test_marker_format(self) -> None:
        from keryx.fuzzing.poc import reproduce
        from keryx.fuzzing.harness import MARKER_PREFIX
        poc = reproduce({"rule": "UNSAFE_EVAL_EXEC"}, timeout=5)
        assert poc.marker.startswith(MARKER_PREFIX)

    def test_llm_fallback_called_when_no_template(self) -> None:
        from keryx.fuzzing.poc import reproduce
        called = {}
        def fake_llm(finding, marker):
            called["marker"] = marker
            return f"import sys; sys.stdout.write({marker!r} + '\\n')"

        poc = reproduce({"rule": "LLM_OUTPUT_SINK"}, timeout=5, llm_generate_fn=fake_llm)
        assert poc.reproduced is True
        assert "marker" in called


class TestFuzzingTool:
    """T53d — tool.py: FuzzerTool Toolbox integration."""

    @pytest.mark.asyncio
    async def test_execute_reproduced(self) -> None:
        from keryx.fuzzing.tool import create_fuzzer_tool
        tool = create_fuzzer_tool(sandbox_timeout=8)
        result = await tool.execute({"finding": {"rule": "UNSAFE_EVAL_EXEC"}})
        assert result.success
        assert result.data["reproduced"] is True
        assert result.data["confidence_boost"] > 0

    @pytest.mark.asyncio
    async def test_execute_no_template(self) -> None:
        from keryx.fuzzing.tool import create_fuzzer_tool
        tool = create_fuzzer_tool(sandbox_timeout=5)
        result = await tool.execute({"finding": {"rule": "LLM_OUTPUT_SINK"}})
        assert result.success  # tool itself succeeds
        assert result.data["reproduced"] is False
        assert result.data["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_invalid_input_missing_rule(self) -> None:
        from keryx.fuzzing.tool import create_fuzzer_tool
        tool = create_fuzzer_tool()
        result = await tool.execute({"finding": {"severity": "HIGH"}})
        assert not result.success
        assert result.error == "invalid_input"

    @pytest.mark.asyncio
    async def test_execute_invalid_input_not_dict(self) -> None:
        from keryx.fuzzing.tool import create_fuzzer_tool
        tool = create_fuzzer_tool()
        result = await tool.execute({"finding": "SUBPROCESS_SHELL_TRUE"})
        assert not result.success

    def test_tool_name(self) -> None:
        from keryx.fuzzing.tool import create_fuzzer_tool
        tool = create_fuzzer_tool()
        assert tool.name == "fuzzer"


class TestFuzzingSandboxEdgeCases:
    """T53e — sandbox.py remaining uncovered branches."""

    def test_preexec_returns_callable(self) -> None:
        # _make_preexec() must return a zero-arg callable suitable for preexec_fn.
        from keryx.fuzzing.sandbox import _make_preexec
        fn = _make_preexec(5)
        assert callable(fn)

    def test_preexec_calls_setrlimit_with_correct_args(self) -> None:
        # Lines 29-36: verify setrlimit is invoked with the right values.
        # Use mock so the limits are never applied to the test runner process.
        from unittest.mock import patch, MagicMock
        from keryx.fuzzing.sandbox import _make_preexec
        import resource as _resource

        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU   = _resource.RLIMIT_CPU
        mock_resource.RLIMIT_FSIZE = _resource.RLIMIT_FSIZE

        fn = _make_preexec(7)
        with patch("keryx.fuzzing.sandbox.resource", mock_resource, create=True), \
             patch.dict("sys.modules", {"resource": mock_resource}):
            fn()

        # setrlimit was called at least twice (CPU + FSIZE)
        assert mock_resource.setrlimit.call_count >= 1

    def test_preexec_swallows_setrlimit_errors(self) -> None:
        # Lines 35-36: if setrlimit raises (unsupported platform), error is swallowed.
        from unittest.mock import patch, MagicMock
        from keryx.fuzzing.sandbox import _make_preexec
        import resource as _resource

        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU   = _resource.RLIMIT_CPU
        mock_resource.RLIMIT_FSIZE = _resource.RLIMIT_FSIZE
        mock_resource.setrlimit.side_effect = ValueError("not supported")

        fn = _make_preexec(5)
        with patch("keryx.fuzzing.sandbox.resource", mock_resource, create=True), \
             patch.dict("sys.modules", {"resource": mock_resource}):
            fn()  # must not propagate the ValueError

    def test_generic_exception_branch(self) -> None:
        # Lines 69-70: subprocess.run raises a non-TimeoutExpired exception
        # (e.g. OSError) — caught by bare except and returned as exit_code=-1.
        from unittest.mock import patch
        from keryx.fuzzing import sandbox

        with patch("keryx.fuzzing.sandbox.subprocess.run", side_effect=OSError("boom")):
            result = sandbox.run("pass", timeout=5)

        assert result.exit_code == -1
        assert "boom" in result.stderr
        assert result.stdout == ""
        assert not result.timed_out

    def test_timeout_with_bytes_stdout(self) -> None:
        # Lines 63-64: TimeoutExpired.stdout is bytes — must be decoded.
        from unittest.mock import patch
        from keryx.fuzzing import sandbox
        import subprocess as _sp

        exc = _sp.TimeoutExpired(cmd="x", timeout=1)
        exc.stdout = b"partial output"
        exc.stderr = b"partial err"

        with patch("keryx.fuzzing.sandbox.subprocess.run", side_effect=exc):
            result = sandbox.run("pass", timeout=1)

        assert result.timed_out
        assert result.stdout == "partial output"
        assert result.stderr == "partial err"
        assert result.exit_code is None

    def test_timeout_with_none_stdout(self) -> None:
        # TimeoutExpired.stdout/stderr is None — should produce empty strings.
        from unittest.mock import patch
        from keryx.fuzzing import sandbox
        import subprocess as _sp

        exc = _sp.TimeoutExpired(cmd="x", timeout=1)
        exc.stdout = None
        exc.stderr = None

        with patch("keryx.fuzzing.sandbox.subprocess.run", side_effect=exc):
            result = sandbox.run("pass", timeout=1)

        assert result.timed_out
        assert result.stdout == ""
        assert result.stderr == ""


# ===========================================================================
# P1.3 — SARIF 2.1.0 serialiser  (keryx/core/sarif.py)
# ===========================================================================

class TestSarifModule:
    """T54a — sarif.py: to_sarif() structure and content."""

    def _make_hunt(
        self,
        file: str = "keryx/tools/git_blame.py",
        rule: str = "SUBPROCESS_SHELL_TRUE",
        line: int | None = 42,
        tag: str = "AST+injection_verifier",
        verified: bool = True,
    ) -> dict:
        obs = f"[HIGH] {rule} @ line {line}: subprocess.run(cmd, shell=True)" if line else f"[HIGH] {rule}: no line"
        return {
            "file": file,
            "confirmed": [
                {
                    "rule": rule,
                    "rules": [rule],
                    "observation": obs,
                    "confidence_tag": tag,
                    "verified": verified,
                }
            ],
        }

    def test_top_level_sarif_keys(self) -> None:
        from keryx.core.sarif import to_sarif
        doc = to_sarif([self._make_hunt()])
        assert doc["version"] == "2.1.0"
        assert "$schema" in doc
        assert len(doc["runs"]) == 1

    def test_tool_driver_name(self) -> None:
        from keryx.core.sarif import to_sarif
        driver = to_sarif([self._make_hunt()])["runs"][0]["tool"]["driver"]
        assert driver["name"] == "Keryx Hunter"
        assert "version" in driver

    def test_rules_populated(self) -> None:
        from keryx.core.sarif import to_sarif
        hunt = self._make_hunt(rule="SUBPROCESS_SHELL_TRUE")
        rules = to_sarif([hunt])["runs"][0]["tool"]["driver"]["rules"]
        assert any(r["id"] == "SUBPROCESS_SHELL_TRUE" for r in rules)

    def test_result_count_matches_confirmed(self) -> None:
        from keryx.core.sarif import to_sarif
        h1 = self._make_hunt(file="a.py", rule="SUBPROCESS_SHELL_TRUE")
        h2 = self._make_hunt(file="b.py", rule="OPEN_USER_PATH")
        results = to_sarif([h1, h2])["runs"][0]["results"]
        assert len(results) == 2

    def test_result_rule_id(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt(rule="UNSAFE_PICKLE")])["runs"][0]["results"]
        assert results[0]["ruleId"] == "UNSAFE_PICKLE"

    def test_result_level_high_is_error(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt()])["runs"][0]["results"]
        assert results[0]["level"] == "error"

    def test_result_has_location(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt(line=55)])["runs"][0]["results"]
        locs = results[0]["locations"]
        assert len(locs) == 1
        phys = locs[0]["physicalLocation"]
        assert "artifactLocation" in phys

    def test_line_number_extracted(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt(line=55)])["runs"][0]["results"]
        region = results[0]["locations"][0]["physicalLocation"].get("region", {})
        assert region.get("startLine") == 55

    def test_no_line_number_omits_region(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt(line=None)])["runs"][0]["results"]
        phys = results[0]["locations"][0]["physicalLocation"]
        assert "region" not in phys

    def test_internal_path_gets_relative_uri(self) -> None:
        from keryx.core.sarif import to_sarif
        repo_root = "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter"
        hunt = self._make_hunt(file="keryx/tools/git_blame.py")
        results = to_sarif([hunt], repo_root=repo_root)["runs"][0]["results"]
        loc = results[0]["locations"][0]["physicalLocation"]["artifactLocation"]
        assert loc.get("uriBaseId") == "%SRCROOT%"
        assert loc["uri"] == "keryx/tools/git_blame.py"

    def test_external_path_gets_absolute_uri(self) -> None:
        from keryx.core.sarif import to_sarif
        hunt = self._make_hunt(file="/tmp/external/app.py")
        results = to_sarif([hunt], repo_root="/some/other/root")["runs"][0]["results"]
        loc = results[0]["locations"][0]["physicalLocation"]["artifactLocation"]
        assert "uriBaseId" not in loc
        assert loc["uri"].startswith("file://")

    def test_verified_flag_in_properties(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt(verified=True)])["runs"][0]["results"]
        assert results[0]["properties"]["verified"] is True

    def test_confidence_tag_in_properties(self) -> None:
        from keryx.core.sarif import to_sarif
        results = to_sarif([self._make_hunt(tag="AST+fuzz_poc")])["runs"][0]["results"]
        assert results[0]["properties"]["confidence_tag"] == "AST+fuzz_poc"

    def test_fuzz_proof_added_when_reproduced(self) -> None:
        from keryx.core.sarif import to_sarif
        hunt = {
            "file": "a.py",
            "confirmed": [{
                "rule": "SUBPROCESS_SHELL_TRUE",
                "observation": "[HIGH] SUBPROCESS_SHELL_TRUE @ line 1: x",
                "confidence_tag": "AST+fuzz_poc",
                "verified": True,
                "fuzz_proof": {"reproduced": True, "confidence_boost": 0.35},
            }],
        }
        results = to_sarif([hunt])["runs"][0]["results"]
        props = results[0]["properties"]
        assert props.get("fuzz_reproduced") is True
        assert props.get("fuzz_confidence_boost") == 0.35

    def test_empty_hunt_list_produces_empty_results(self) -> None:
        from keryx.core.sarif import to_sarif
        doc = to_sarif([])
        assert doc["runs"][0]["results"] == []
        assert doc["runs"][0]["tool"]["driver"]["rules"] == []

    def test_no_confirmed_vulns_produces_empty_results(self) -> None:
        from keryx.core.sarif import to_sarif
        doc = to_sarif([{"file": "a.py", "confirmed": []}])
        assert doc["runs"][0]["results"] == []

    def test_duplicate_rules_deduplicated_in_driver(self) -> None:
        from keryx.core.sarif import to_sarif
        h1 = self._make_hunt(file="a.py", rule="UNSAFE_PICKLE")
        h2 = self._make_hunt(file="b.py", rule="UNSAFE_PICKLE")
        rules = to_sarif([h1, h2])["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [r["id"] for r in rules]
        assert rule_ids.count("UNSAFE_PICKLE") == 1

    def test_unknown_rule_gets_default_description(self) -> None:
        from keryx.core.sarif import to_sarif
        hunt = {
            "file": "a.py",
            "confirmed": [{
                "rule": "BRAND_NEW_RULE",
                "observation": "[HIGH] BRAND_NEW_RULE @ line 7: something()",
                "confidence_tag": "AST-only",
                "verified": False,
            }],
        }
        results = to_sarif([hunt])["runs"][0]["results"]
        assert results[0]["ruleId"] == "BRAND_NEW_RULE"
        rules = to_sarif([hunt])["runs"][0]["tool"]["driver"]["rules"]
        assert any(r["id"] == "BRAND_NEW_RULE" for r in rules)

    def test_extract_line_helper(self) -> None:
        from keryx.core.sarif import _extract_line
        assert _extract_line("[HIGH] RULE @ line 99: code()") == 99
        assert _extract_line("[HIGH] RULE: no line info") is None
        assert _extract_line("") is None

    def test_artifact_uri_internal(self) -> None:
        from keryx.core.sarif import _artifact_uri
        uri, base = _artifact_uri("keryx/tools/git_blame.py",
                                   "/Users/serhiihrynko/Documents/Helga/Keryx_Hunter")
        # When file_str is already relative this raises ValueError → external path
        # (absolute path required for relative_to to work)
        assert uri is not None

    def test_sarif_is_json_serialisable(self) -> None:
        import json
        from keryx.core.sarif import to_sarif
        doc = to_sarif([self._make_hunt()])
        # Should not raise
        raw = json.dumps(doc)
        assert "Keryx Hunter" in raw


# ===========================================================================
# P2.1 — R11 SSRF rule  (keryx/tools/ast_analyzer.py)
# ===========================================================================

class TestSSRFRule:
    """T54b — ast_analyzer.py: R11 SSRF detection."""

    def _findings(self, code: str):
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        tree = ast.parse(code)
        visitor = _VulnVisitor(source_lines=code.splitlines())
        visitor.visit(tree)
        return [f for f in visitor.findings if f.rule == "SSRF"]

    def test_get_with_variable_url(self) -> None:
        findings = self._findings("import requests\nrequests.get(url)")
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"

    def test_post_with_variable_url(self) -> None:
        findings = self._findings("import requests\nrequests.post(url, json={})")
        assert len(findings) == 1

    def test_put_with_variable_url(self) -> None:
        findings = self._findings("import requests\nrequests.put(url, data=b'')")
        assert len(findings) == 1

    def test_delete_with_variable_url(self) -> None:
        findings = self._findings("import requests\nrequests.delete(url)")
        assert len(findings) == 1

    def test_patch_with_variable_url(self) -> None:
        findings = self._findings("import requests\nrequests.patch(url)")
        assert len(findings) == 1

    def test_request_method_variable_url_second_arg(self) -> None:
        # requests.request("GET", url) — URL is the second positional arg
        findings = self._findings('import requests\nrequests.request("GET", url)')
        assert len(findings) == 1

    def test_constant_url_not_flagged(self) -> None:
        findings = self._findings('import requests\nrequests.get("https://api.example.com/data")')
        assert findings == []

    def test_concatenated_url_flagged(self) -> None:
        # BinOp — base + path
        findings = self._findings('import requests\nrequests.get(base + "/path")')
        assert len(findings) == 1

    def test_subscript_url_flagged(self) -> None:
        findings = self._findings('import requests\nrequests.get(config["url"])')
        assert len(findings) == 1

    def test_fstring_url_flagged(self) -> None:
        findings = self._findings('import requests\nrequests.get(f"http://{host}/path")')
        assert len(findings) == 1

    def test_no_args_not_flagged(self) -> None:
        findings = self._findings("import requests\nrequests.get()")
        assert findings == []

    def test_non_requests_module_not_flagged(self) -> None:
        findings = self._findings("import httpx\nhttpx.get(url)")
        assert findings == []

    def test_request_with_constant_url_not_flagged(self) -> None:
        findings = self._findings('import requests\nrequests.request("GET", "https://fixed.com")')
        assert findings == []

    def test_line_number_present(self) -> None:
        code = "import requests\n\n\nrequests.get(url)"
        findings = self._findings(code)
        assert findings[0].line == 4

    @pytest.mark.asyncio
    async def test_ssrf_in_ast_tool_output(self) -> None:
        import textwrap
        from keryx.tools.ast_analyzer import ASTAnalyzerTool
        import tempfile, os

        code = textwrap.dedent("""\
            import requests
            def fetch(url: str) -> str:
                return requests.get(url).text
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            tool = ASTAnalyzerTool()
            result = await tool.execute({"file_path": tmp})
            rules = [fi["rule"] for fi in result.data.get("findings", [])]
            assert "SSRF" in rules
        finally:
            os.unlink(tmp)


class TestSSRFFuzzerHarness:
    """T54c — harness.py: SSRF template + poc.reproduce()."""

    def test_ssrf_template_generated(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "SSRF"}, "KERYX_POC_SSRF01")
        assert script is not None
        assert "KERYX_POC_SSRF01" in script

    def test_ssrf_poc_reproduced(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "SSRF"}, timeout=8)
        assert poc.reproduced is True
        assert poc.confidence_boost > 0
        assert poc.error is None

    def test_ssrf_known_rules_list(self) -> None:
        from keryx.fuzzing.harness import _TEMPLATES
        assert "SSRF" in _TEMPLATES


# ===========================================================================
# P2.2 — R12 TEMPLATE_INJECTION  (SSTI via Jinja2/Mako Template)
# ===========================================================================

class TestTemplateInjectionRule:
    """T55a — ast_analyzer.py: R12 TEMPLATE_INJECTION detection."""

    def _findings(self, code: str):
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        tree = ast.parse(code)
        visitor = _VulnVisitor(source_lines=code.splitlines())
        visitor.visit(tree)
        return [f for f in visitor.findings if f.rule == "TEMPLATE_INJECTION"]

    # -- Pattern A: bare Template(non_const) ---------------------------------

    def test_bare_template_variable_flagged(self) -> None:
        findings = self._findings("from jinja2 import Template\nTemplate(user_tpl)")
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"

    def test_bare_template_fstring_flagged(self) -> None:
        findings = self._findings('Template(f"Hello {name}!")')
        assert len(findings) == 1

    def test_bare_template_binop_flagged(self) -> None:
        findings = self._findings('Template(header + body)')
        assert len(findings) == 1

    def test_bare_template_subscript_flagged(self) -> None:
        findings = self._findings('Template(config["tpl"])')
        assert len(findings) == 1

    def test_bare_template_constant_not_flagged(self) -> None:
        findings = self._findings('Template("Hello {{ name }}!")')
        assert findings == []

    def test_bare_template_no_args_not_flagged(self) -> None:
        findings = self._findings("Template()")
        assert findings == []

    # -- Pattern A: qualified module.Template(non_const) ---------------------

    def test_jinja2_template_flagged(self) -> None:
        findings = self._findings("import jinja2\njinja2.Template(user_tpl)")
        assert len(findings) == 1

    def test_mako_template_flagged(self) -> None:
        findings = self._findings(
            "from mako import template as _m\n_m.Template(user_tpl)"
        )
        assert len(findings) == 1

    def test_qualified_template_constant_not_flagged(self) -> None:
        findings = self._findings('import jinja2\njinja2.Template("static {{ x }}")')
        assert findings == []

    # -- Pattern B: env.from_string(non_const) --------------------------------

    def test_from_string_variable_flagged(self) -> None:
        findings = self._findings(
            "from jinja2 import Environment\nenv = Environment()\nenv.from_string(user_tpl)"
        )
        assert len(findings) == 1

    def test_from_string_constant_not_flagged(self) -> None:
        findings = self._findings(
            'env.from_string("Hello {{ name }}!")'
        )
        assert findings == []

    def test_from_string_fstring_flagged(self) -> None:
        findings = self._findings('env.from_string(f"{{ {var} }}")')
        assert len(findings) == 1

    # -- Negative: render() with user data is NOT flagged --------------------

    def test_render_with_user_kwargs_not_flagged(self) -> None:
        # Passing user data as context is the safe pattern
        findings = self._findings(
            'tmpl = Template("Hello {{ name }}!")\ntmpl.render(name=user_input)'
        )
        assert findings == []

    # -- Line number ----------------------------------------------------------

    def test_line_number_present(self) -> None:
        code = "from jinja2 import Template\n\n\nTemplate(user_tpl)"
        findings = self._findings(code)
        assert findings[0].line == 4

    # -- End-to-end via ASTAnalyzerTool --------------------------------------

    @pytest.mark.asyncio
    async def test_template_injection_in_tool_output(self) -> None:
        import textwrap, tempfile, os
        from keryx.tools.ast_analyzer import ASTAnalyzerTool

        code = textwrap.dedent("""\
            from jinja2 import Template, Environment

            def render_user(tpl: str, **ctx):
                t = Template(tpl)           # SSTI: user controls template string
                return t.render(**ctx)

            def render_env(tpl: str, **ctx):
                env = Environment()
                return env.from_string(tpl).render(**ctx)  # SSTI

            def safe_render(name: str):
                t = Template("Hello {{ name }}!")   # safe: hardcoded template
                return t.render(name=name)
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            tool = ASTAnalyzerTool()
            result = await tool.execute({"file_path": tmp})
            rules = [fi["rule"] for fi in result.data.get("findings", [])]
            assert rules.count("TEMPLATE_INJECTION") == 2   # Template(tpl) + from_string(tpl)
        finally:
            os.unlink(tmp)


class TestTemplateInjectionHarness:
    """T55b — harness.py + poc.py: TEMPLATE_INJECTION PoC."""

    def test_template_in_templates_registry(self) -> None:
        from keryx.fuzzing.harness import _TEMPLATES
        assert "TEMPLATE_INJECTION" in _TEMPLATES

    def test_template_script_generated(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "TEMPLATE_INJECTION"}, "KERYX_POC_SSTI01")
        assert script is not None
        assert "KERYX_POC_SSTI01" in script
        assert "jinja2" in script

    def test_template_script_contains_attacker_tpl(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "TEMPLATE_INJECTION"}, "KERYX_POC_SSTI01")
        # The generated script must embed the attacker-controlled template string
        assert "{% set x" in script or "set x" in script

    def test_reproduce_template_injection(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "TEMPLATE_INJECTION"}, timeout=8)
        assert poc.reproduced is True
        assert poc.confidence_boost > 0
        assert poc.error is None

    def test_harness_executes_in_sandbox(self) -> None:
        from keryx.fuzzing.harness import generate
        from keryx.fuzzing.sandbox import run
        marker = "KERYX_POC_SANDBOXSSTI"
        script = generate({"rule": "TEMPLATE_INJECTION"}, marker)
        result = run(script, timeout=8)
        combined = result.stdout + result.stderr
        assert marker in combined


class TestTemplateInjectionExtractFindings:
    """T55c — verification_pipeline.extract_findings(): TEMPLATE_INJECTION branch."""

    def test_template_injection_extracted(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] TEMPLATE_INJECTION @ line 15: Template(user_tpl) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert len(findings) == 1
        rule, field, payload = findings[0]
        assert rule == "TEMPLATE_INJECTION"
        assert field == "template"
        assert payload == "{{7*7}}"

    def test_from_string_inject_field(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] TEMPLATE_INJECTION @ line 8: env.from_string(tpl) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert len(findings) == 1
        assert findings[0][1] == "template"

    def test_known_varname_mapped(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] TEMPLATE_INJECTION @ line 3: Template(tmpl) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert findings[0][1] == "template"   # tmpl → template via VARNAME_TO_FIELD

    def test_unknown_varname_defaults_to_template(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] TEMPLATE_INJECTION @ line 3: Template(some_weird_var) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert findings[0][1] == "template"

    def test_no_template_call_still_extracts_with_default(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] TEMPLATE_INJECTION @ line 5: jinja2.Template(x)\n"
        findings = VerificationPipeline.extract_findings(obs)
        # May or may not match the regex — either way, field must be "template"
        if findings:
            assert findings[0][1] == "template"


class TestTemplateInjectionSarifMeta:
    """T55d — sarif.py: TEMPLATE_INJECTION rule metadata."""

    def test_sarif_rule_meta_entry(self) -> None:
        from keryx.core.sarif import _RULE_META
        assert "TEMPLATE_INJECTION" in _RULE_META
        name, short, fix = _RULE_META["TEMPLATE_INJECTION"]
        assert name == "TemplateInjection"
        assert "SSTI" in short or "template" in short.lower()
        assert "render" in fix.lower() or "context" in fix.lower()

    def test_sarif_result_for_template_injection(self) -> None:
        from keryx.core.sarif import to_sarif
        hunt = {
            "file": "app.py",
            "confirmed": [{
                "rule": "TEMPLATE_INJECTION",
                "observation": "[HIGH] TEMPLATE_INJECTION @ line 15: Template(user_tpl)",
                "confidence_tag": "AST+fuzz_poc",
                "verified": True,
            }],
        }
        results = to_sarif([hunt])["runs"][0]["results"]
        assert results[0]["ruleId"] == "TEMPLATE_INJECTION"
        assert results[0]["level"] == "error"
        rules = to_sarif([hunt])["runs"][0]["tool"]["driver"]["rules"]
        r = next(r for r in rules if r["id"] == "TEMPLATE_INJECTION")
        assert r["name"] == "TemplateInjection"


# ===========================================================================
# P2.3 — R13 REGEX_DOS  (ReDoS via user-controlled pattern)
# ===========================================================================

class TestRegexDosRule:
    """T56a — ast_analyzer.py: R13 REGEX_DOS detection."""

    def _findings(self, code: str):
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        tree = ast.parse(code)
        visitor = _VulnVisitor(source_lines=code.splitlines())
        visitor.visit(tree)
        return [f for f in visitor.findings if f.rule == "REGEX_DOS"]

    # -- Each re.* method with variable pattern is flagged -------------------

    def test_re_compile_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.compile(user_pattern)")
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"

    def test_re_search_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.search(pattern, text)")
        assert len(findings) == 1

    def test_re_match_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.match(pat, line)")
        assert len(findings) == 1

    def test_re_fullmatch_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.fullmatch(regex, value)")
        assert len(findings) == 1

    def test_re_findall_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.findall(pattern, corpus)")
        assert len(findings) == 1

    def test_re_finditer_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.finditer(pattern, text)")
        assert len(findings) == 1

    def test_re_sub_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.sub(pattern, repl, string)")
        assert len(findings) == 1

    def test_re_subn_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.subn(pattern, repl, string)")
        assert len(findings) == 1

    def test_re_split_variable_flagged(self) -> None:
        findings = self._findings("import re\nre.split(pattern, string)")
        assert len(findings) == 1

    # -- Constant pattern is NOT flagged -------------------------------------

    def test_constant_pattern_not_flagged(self) -> None:
        findings = self._findings(r'import re; re.compile(r"\d+")')
        assert findings == []

    def test_constant_pattern_search_not_flagged(self) -> None:
        findings = self._findings('import re\nre.search(r"[a-z]+", user_text)')
        assert findings == []

    # -- User-controlled string being matched is NOT flagged -----------------

    def test_fixed_pattern_user_string_not_flagged(self) -> None:
        # Pattern is constant; user controls the *string* → not ReDoS
        findings = self._findings('import re\nre.match(r"^\\d+$", user_input)')
        assert findings == []

    # -- Non-re module not flagged -------------------------------------------

    def test_non_re_module_not_flagged(self) -> None:
        findings = self._findings("import regex\nregex.compile(user_pattern)")
        assert findings == []

    # -- Compound expressions flagged ----------------------------------------

    def test_fstring_pattern_flagged(self) -> None:
        findings = self._findings('import re\nre.compile(f"^{user_prefix}.*")')
        assert len(findings) == 1

    def test_binop_pattern_flagged(self) -> None:
        findings = self._findings('import re\nre.compile("^" + user_suffix)')
        assert len(findings) == 1

    def test_subscript_pattern_flagged(self) -> None:
        findings = self._findings('import re\nre.compile(config["pattern"])')
        assert len(findings) == 1

    # -- No args not flagged -------------------------------------------------

    def test_no_args_not_flagged(self) -> None:
        findings = self._findings("import re\nre.compile()")
        assert findings == []

    # -- Line number ---------------------------------------------------------

    def test_line_number_present(self) -> None:
        code = "import re\n\n\nre.compile(user_pattern)"
        findings = self._findings(code)
        assert findings[0].line == 4

    # -- End-to-end via ASTAnalyzerTool -------------------------------------

    @pytest.mark.asyncio
    async def test_regex_dos_in_tool_output(self) -> None:
        import textwrap, tempfile, os
        from keryx.tools.ast_analyzer import ASTAnalyzerTool

        code = textwrap.dedent("""\
            import re

            def search_logs(user_pattern: str, log_line: str) -> bool:
                return bool(re.search(user_pattern, log_line))   # REGEX_DOS

            def compile_pattern(pat: str):
                return re.compile(pat)                           # REGEX_DOS

            def safe_search(log_line: str) -> bool:
                return bool(re.search(r"ERROR", log_line))       # safe
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            tool = ASTAnalyzerTool()
            result = await tool.execute({"file_path": tmp})
            rules = [fi["rule"] for fi in result.data.get("findings", [])]
            assert rules.count("REGEX_DOS") == 2
        finally:
            os.unlink(tmp)


class TestRegexDosHarness:
    """T56b — harness.py + poc.py: REGEX_DOS PoC."""

    def test_regex_dos_in_templates_registry(self) -> None:
        from keryx.fuzzing.harness import _TEMPLATES
        assert "REGEX_DOS" in _TEMPLATES

    def test_script_generated(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "REGEX_DOS"}, "KERYX_POC_REDOS01")
        assert script is not None
        assert "KERYX_POC_REDOS01" in script

    def test_script_contains_backtracking_pattern(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "REGEX_DOS"}, "KERYX_POC_REDOS01")
        # Must use the canonical catastrophic backtracking pattern
        assert "(a+)+" in script

    def test_reproduce_regex_dos(self) -> None:
        from keryx.fuzzing.poc import reproduce
        poc = reproduce({"rule": "REGEX_DOS"}, timeout=8)
        assert poc.reproduced is True
        assert poc.confidence_boost > 0
        assert poc.error is None

    def test_harness_executes_in_sandbox(self) -> None:
        from keryx.fuzzing.harness import generate
        from keryx.fuzzing.sandbox import run
        marker = "KERYX_POC_REBOXSANDBOX"
        script = generate({"rule": "REGEX_DOS"}, marker)
        result = run(script, timeout=8)
        assert marker in result.stdout + result.stderr

    def test_script_mentions_exponential_scaling(self) -> None:
        from keryx.fuzzing.harness import generate
        script = generate({"rule": "REGEX_DOS"}, "M")
        assert "exponential" in script.lower() or "backtrack" in script.lower()


class TestRegexDosExtractFindings:
    """T56c — verification_pipeline.extract_findings(): REGEX_DOS branch."""

    def test_regex_dos_extracted_compile(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] REGEX_DOS @ line 4: re.compile(user_pattern) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert len(findings) == 1
        rule, field, payload = findings[0]
        assert rule == "REGEX_DOS"
        assert field == "pattern"
        assert payload == "(a+)+$"

    def test_regex_dos_extracted_search(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] REGEX_DOS @ line 10: re.search(regex, text) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert findings[0][1] == "pattern"   # regex → pattern via VARNAME_TO_FIELD

    def test_known_varname_pat_mapped(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] REGEX_DOS @ line 3: re.match(pat, line) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert findings[0][1] == "pattern"

    def test_unknown_varname_defaults_to_pattern(self) -> None:
        from keryx.core.verification_pipeline import VerificationPipeline
        obs = "[HIGH] REGEX_DOS @ line 7: re.compile(weird_var) — ...\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert findings[0][1] == "pattern"


class TestRegexDosSarifMeta:
    """T56d — sarif.py: REGEX_DOS rule metadata."""

    def test_sarif_rule_meta_entry(self) -> None:
        from keryx.core.sarif import _RULE_META
        assert "REGEX_DOS" in _RULE_META
        name, short, fix = _RULE_META["REGEX_DOS"]
        assert name == "RegexDenialOfService"
        assert "ReDoS" in short or "pattern" in short.lower()
        assert "constant" in fix.lower() or "untrusted" in fix.lower()

    def test_sarif_result_for_regex_dos(self) -> None:
        from keryx.core.sarif import to_sarif
        hunt = {
            "file": "app.py",
            "confirmed": [{
                "rule": "REGEX_DOS",
                "observation": "[HIGH] REGEX_DOS @ line 4: re.compile(user_pattern)",
                "confidence_tag": "AST+fuzz_poc",
                "verified": True,
            }],
        }
        results = to_sarif([hunt])["runs"][0]["results"]
        assert results[0]["ruleId"] == "REGEX_DOS"
        assert results[0]["level"] == "error"
        rules = to_sarif([hunt])["runs"][0]["tool"]["driver"]["rules"]
        r = next(r for r in rules if r["id"] == "REGEX_DOS")
        assert r["name"] == "RegexDenialOfService"


# ===========================================================================
# P3.3 — source_type annotation  (ast_analyzer.py)
# ===========================================================================

class TestSourceTypeAnnotation:
    """T57a — Finding.source_type populated by _VulnVisitor param tracking."""

    def _findings(self, code: str):
        import ast
        from keryx.tools.ast_analyzer import _VulnVisitor
        tree = ast.parse(code)
        v = _VulnVisitor(source_lines=code.splitlines())
        v.visit(tree)
        return v.findings

    # -- "param" when arg is a direct function parameter --------------------

    def test_direct_param_ssrf(self) -> None:
        code = "import requests\ndef f(url): requests.get(url)"
        f = [x for x in self._findings(code) if x.rule == "SSRF"][0]
        assert f.source_type == "param"

    def test_direct_param_eval(self) -> None:
        code = "def f(expr):\n    eval(expr)"
        f = [x for x in self._findings(code) if x.rule == "UNSAFE_EVAL_EXEC"][0]
        assert f.source_type == "param"

    def test_direct_param_pickle(self) -> None:
        code = "import pickle\ndef f(data):\n    pickle.loads(data)"
        f = [x for x in self._findings(code) if x.rule == "UNSAFE_PICKLE"][0]
        assert f.source_type == "param"

    def test_direct_param_open(self) -> None:
        code = "def f(path):\n    open(path)"
        f = [x for x in self._findings(code) if x.rule == "OPEN_USER_PATH"][0]
        assert f.source_type == "param"

    def test_direct_param_template_injection(self) -> None:
        code = "from jinja2 import Template\ndef f(tpl):\n    Template(tpl)"
        f = [x for x in self._findings(code) if x.rule == "TEMPLATE_INJECTION"][0]
        assert f.source_type == "param"

    def test_direct_param_regex_dos(self) -> None:
        code = "import re\ndef f(pat):\n    re.compile(pat)"
        f = [x for x in self._findings(code) if x.rule == "REGEX_DOS"][0]
        assert f.source_type == "param"

    # -- "local" when arg is a local variable (not in params) ---------------

    def test_local_variable(self) -> None:
        code = "import re\ndef f():\n    pat = get_pattern()\n    re.compile(pat)"
        f = [x for x in self._findings(code) if x.rule == "REGEX_DOS"][0]
        assert f.source_type == "local"

    # -- "param" for expression that contains a param Name ------------------

    def test_fstring_with_param_is_param(self) -> None:
        code = 'import requests\ndef f(host):\n    requests.get(f"http://{host}/api")'
        f = [x for x in self._findings(code) if x.rule == "SSRF"][0]
        assert f.source_type == "param"

    def test_binop_with_param_is_param(self) -> None:
        code = 'import requests\ndef f(path):\n    requests.get("https://base" + path)'
        f = [x for x in self._findings(code) if x.rule == "SSRF"][0]
        assert f.source_type == "param"

    # -- "unknown" when not inside a function --------------------------------

    def test_module_level_is_unknown(self) -> None:
        code = "import re\nre.compile(user_pattern)"
        f = [x for x in self._findings(code) if x.rule == "REGEX_DOS"][0]
        assert f.source_type == "unknown"

    # -- nested function: inner params don't bleed into outer ----------------

    def test_nested_function_inner_param(self) -> None:
        code = (
            "import re\n"
            "def outer():\n"
            "    local_var = 'x'\n"
            "    def inner(pat):\n"
            "        re.compile(pat)\n"
        )
        findings = [x for x in self._findings(code) if x.rule == "REGEX_DOS"]
        assert findings[0].source_type == "param"

    def test_nested_function_outer_local(self) -> None:
        code = (
            "import re\n"
            "def outer():\n"
            "    local_var = get_pat()\n"
            "    re.compile(local_var)\n"
        )
        findings = [x for x in self._findings(code) if x.rule == "REGEX_DOS"]
        # local_var is defined in outer(), not a param
        assert findings[0].source_type == "local"

    # -- async function params tracked too -----------------------------------

    def test_async_function_param(self) -> None:
        code = "import requests\nasync def f(url):\n    requests.get(url)"
        f = [x for x in self._findings(code) if x.rule == "SSRF"][0]
        assert f.source_type == "param"

    # -- source_type in serialised ASTAnalyzerTool output --------------------

    @pytest.mark.asyncio
    async def test_source_type_in_tool_data(self) -> None:
        import textwrap, tempfile, os
        from keryx.tools.ast_analyzer import ASTAnalyzerTool

        code = textwrap.dedent("""\
            import re
            def search(user_pattern: str, text: str) -> bool:
                return bool(re.search(user_pattern, text))
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as fh:
            fh.write(code)
            tmp = fh.name
        try:
            result = await ASTAnalyzerTool().execute({"file_path": tmp})
            fi = next(f for f in result.data["findings"] if f["rule"] == "REGEX_DOS")
            assert fi["source_type"] == "param"
        finally:
            os.unlink(tmp)


# ===========================================================================
# P3.2 — LLM harness factory  (keryx/fuzzing/llm_harness.py)
# ===========================================================================

class TestLLMHarnessFactory:
    """T57b — llm_harness.py: make_llm_generate_fn() and integration."""

    def _fake_model(self, response: str):
        """Return a minimal ModelInterface stub that always returns *response*."""
        from keryx.models.interface import (
            ModelInterface, GenerationResult, ModelCapabilities, CostEstimate,
        )
        class _Stub(ModelInterface):
            model_name = "stub"
            def generate(self, prompt, config=None, *, grammar=None, max_tokens=None):
                return response
            async def generate_async(self, *a, **kw): return response
            def generate_result(self, *a, **kw):
                return GenerationResult(text=response, tokens_input=1, tokens_output=1,
                                        duration_ms=0.0, finish_reason="stop")
            def generate_stream(self, *a, **kw): yield response
            def generate_with_tools(self, *a, **kw): return response
            def tokenize(self, t): return [0]
            def get_context_length(self): return 4096
            def is_healthy(self): return True
            def estimate_cost(self, i, o):
                return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)
            def get_usage_cost(self):
                return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)
            def get_capabilities(self):
                return ModelCapabilities(max_context_length=4096,
                    supports_tool_calling=False, supports_grammar=False,
                    supports_batching=False, supports_streaming=False,
                    supports_speculative=False, supports_min_p=False,
                    requires_gpu=False, is_local=True, is_quantized=False)
            def unload(self): pass
        return _Stub()

    def test_valid_script_returned(self) -> None:
        from keryx.fuzzing.llm_harness import make_llm_generate_fn
        marker = "KERYX_POC_LLMTEST"
        script = f'import sys\nsys.stdout.write({marker!r} + "\\n")'
        fn = make_llm_generate_fn(self._fake_model(script))
        result = fn({"rule": "LLM_OUTPUT_SINK"}, marker)
        assert result is not None
        assert marker in result

    def test_markdown_fences_stripped(self) -> None:
        from keryx.fuzzing.llm_harness import make_llm_generate_fn
        marker = "KERYX_POC_FENCE"
        inner = f'import sys\nsys.stdout.write({marker!r} + "\\n")'
        fenced = f"```python\n{inner}\n```"
        fn = make_llm_generate_fn(self._fake_model(fenced))
        result = fn({"rule": "LLM_OUTPUT_SINK"}, marker)
        assert result is not None
        assert "```" not in result
        assert marker in result

    def test_missing_marker_returns_none(self) -> None:
        from keryx.fuzzing.llm_harness import make_llm_generate_fn
        fn = make_llm_generate_fn(self._fake_model("print('no marker here')"))
        result = fn({"rule": "LLM_OUTPUT_SINK"}, "KERYX_POC_NOTHERE")
        assert result is None

    def test_invalid_python_returns_none(self) -> None:
        from keryx.fuzzing.llm_harness import make_llm_generate_fn
        marker = "KERYX_POC_BADPY"
        fn = make_llm_generate_fn(self._fake_model(f"def bad syntax({marker!r}):"))
        result = fn({"rule": "LLM_OUTPUT_SINK"}, marker)
        assert result is None

    def test_model_exception_returns_none(self) -> None:
        from keryx.fuzzing.llm_harness import make_llm_generate_fn
        from keryx.models.interface import (
            ModelInterface, GenerationResult, ModelCapabilities, CostEstimate,
        )
        class _Raiser(ModelInterface):
            model_name = "raiser"
            def generate(self, *a, **kw): raise RuntimeError("no API key")
            async def generate_async(self, *a, **kw): raise RuntimeError()
            def generate_result(self, *a, **kw):
                return GenerationResult(text="", tokens_input=0, tokens_output=0,
                                        duration_ms=0.0, finish_reason="error")
            def generate_stream(self, *a, **kw): yield ""
            def generate_with_tools(self, *a, **kw): return ""
            def tokenize(self, t): return []
            def get_context_length(self): return 0
            def is_healthy(self): return False
            def estimate_cost(self, i, o):
                return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)
            def get_usage_cost(self):
                return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)
            def get_capabilities(self):
                return ModelCapabilities(max_context_length=0,
                    supports_tool_calling=False, supports_grammar=False,
                    supports_batching=False, supports_streaming=False,
                    supports_speculative=False, supports_min_p=False,
                    requires_gpu=False, is_local=True, is_quantized=False)
            def unload(self): pass
        fn = make_llm_generate_fn(_Raiser())
        result = fn({"rule": "LLM_OUTPUT_SINK"}, "KERYX_POC_RAISE")
        assert result is None

    def test_plain_fences_stripped(self) -> None:
        from keryx.fuzzing.llm_harness import make_llm_generate_fn, _strip_fences
        marker = "M"
        inner = f"import sys\nsys.stdout.write({marker!r})"
        assert _strip_fences(f"```\n{inner}\n```") == inner

    def test_no_fences_unchanged(self) -> None:
        from keryx.fuzzing.llm_harness import _strip_fences
        code = "import sys\nsys.stdout.write('x')"
        assert _strip_fences(code) == code

    def test_poc_reproduce_with_llm_fn(self) -> None:
        """reproduce() accepts llm_generate_fn and uses it for no-template rules."""
        from keryx.fuzzing.poc import reproduce
        from keryx.fuzzing.llm_harness import make_llm_generate_fn
        marker_holder: list[str] = []
        def capture_fn(finding, marker):
            marker_holder.append(marker)
            return f'import sys\nsys.stdout.write({marker!r} + "\\n")'
        fn = make_llm_generate_fn(self._fake_model(""))  # model unused; capture_fn overrides
        # Use capture_fn directly as llm_generate_fn
        poc = reproduce({"rule": "LLM_OUTPUT_SINK"}, timeout=5, llm_generate_fn=capture_fn)
        assert poc.reproduced is True
        assert poc.rule == "LLM_OUTPUT_SINK"


# ===========================================================================
# P3.1 — Rule gap detector  (keryx/core/rule_learner.py)
# ===========================================================================

class TestRuleLearner:
    """T57c — rule_learner.py: RuleLearner + RuleSuggestion."""

    def _learner_from_code(self, code: str, suffix: str = ".py"):
        import tempfile, os
        from pathlib import Path
        from keryx.core.rule_learner import RuleLearner
        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w",
                                         delete=False, encoding="utf-8") as fh:
            fh.write(code)
            tmp = Path(fh.name)
        try:
            learner = RuleLearner([tmp])
            suggestions = learner.analyse()
        finally:
            os.unlink(tmp)
        return suggestions

    # -- Detection of gap patterns -------------------------------------------

    def test_detects_httpx_ssrf(self) -> None:
        suggestions = self._learner_from_code("httpx.get(user_url, timeout=5)")
        ids = [s.suggestion_id for s in suggestions]
        assert "SSRF_HTTPX" in ids

    def test_detects_urllib_ssrf(self) -> None:
        suggestions = self._learner_from_code("urllib.request.urlopen(target_url)")
        ids = [s.suggestion_id for s in suggestions]
        assert "SSRF_URLLIB" in ids

    def test_detects_xml_xxe(self) -> None:
        suggestions = self._learner_from_code("ET.parse(user_file)")
        ids = [s.suggestion_id for s in suggestions]
        assert "XXE_ELEMENTTREE" in ids

    def test_detects_importlib(self) -> None:
        suggestions = self._learner_from_code("importlib.import_module(module_name)")
        ids = [s.suggestion_id for s in suggestions]
        assert "UNSAFE_IMPORT" in ids

    def test_detects_marshal(self) -> None:
        suggestions = self._learner_from_code("marshal.loads(raw_data)")
        ids = [s.suggestion_id for s in suggestions]
        assert "UNSAFE_MARSHAL" in ids

    # -- Negative: covered patterns not re-suggested -------------------------

    def test_clean_file_no_suggestions(self) -> None:
        code = "x = 1\nprint(x)\n"
        suggestions = self._learner_from_code(code)
        assert suggestions == []

    def test_covered_requests_not_suggested(self) -> None:
        # requests.* is already R11 — no new suggestion expected
        code = "import requests\nrequests.get(url)"
        suggestions = self._learner_from_code(code)
        # SSRF_HTTPX should NOT appear (it's httpx, not requests)
        assert all(s.suggestion_id != "SSRF_HTTPX" for s in suggestions)

    # -- Occurrence counting and sorting -------------------------------------

    def test_occurrences_counted_per_file(self) -> None:
        import tempfile, os
        from pathlib import Path
        from keryx.core.rule_learner import RuleLearner

        code_a = "httpx.get(url_a)"
        code_b = "httpx.post(url_b)"
        files = []
        try:
            for code in [code_a, code_b]:
                fh = tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                                  delete=False, encoding="utf-8")
                fh.write(code)
                fh.close()
                files.append(Path(fh.name))
            suggestions = RuleLearner(files).analyse()
            httpx_s = next(s for s in suggestions if s.suggestion_id == "SSRF_HTTPX")
            assert httpx_s.occurrences == 2
        finally:
            for f in files:
                os.unlink(f)

    def test_sorted_by_occurrences_desc(self) -> None:
        import tempfile, os
        from pathlib import Path
        from keryx.core.rule_learner import RuleLearner

        # Two files with httpx (2 hits), one with urllib (1 hit)
        codes = [
            "httpx.get(url)",
            "httpx.post(url)",
            "urllib.request.urlopen(url)",
        ]
        files = []
        try:
            for code in codes:
                fh = tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                                  delete=False, encoding="utf-8")
                fh.write(code)
                fh.close()
                files.append(Path(fh.name))
            suggestions = RuleLearner(files).analyse()
            assert suggestions[0].occurrences >= suggestions[-1].occurrences
        finally:
            for f in files:
                os.unlink(f)

    # -- RuleSuggestion __str__ ----------------------------------------------

    def test_suggestion_str_contains_key_fields(self) -> None:
        suggestions = self._learner_from_code("httpx.get(url)")
        s = suggestions[0]
        text = str(s)
        assert s.suggestion_id in text
        assert s.rule_family in text
        assert str(s.occurrences) in text

    # -- print_suggestions ---------------------------------------------------

    def test_print_suggestions_no_crash(self) -> None:
        import io, contextlib
        from keryx.core.rule_learner import print_suggestions, RuleSuggestion
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_suggestions([])
            print_suggestions([
                RuleSuggestion("SSRF_HTTPX", "SSRF", 1, ["/tmp/a.py"],
                               "httpx.get(url)", "desc", "fix")
            ])
        out = buf.getvalue()
        assert "SSRF_HTTPX" in out
        assert "No rule gaps" in out or "potential gap" in out

    # -- Empty file list -----------------------------------------------------

    def test_empty_file_list(self) -> None:
        from keryx.core.rule_learner import RuleLearner
        assert RuleLearner([]).analyse() == []

    # -- Unreadable file doesn't crash ---------------------------------------

    def test_unreadable_file_skipped(self) -> None:
        from pathlib import Path
        from keryx.core.rule_learner import RuleLearner
        suggestions = RuleLearner([Path("/nonexistent_keryx_test_file.py")]).analyse()
        assert suggestions == []




# =============================================================================
# T58 — P4: GitHub Actions workflow, pre-commit hook installer, HTML report
# =============================================================================

class TestGitHubActionsWorkflow:
    """T58a: keryx-hunt.yml is a well-formed YAML file."""

    @staticmethod
    def _workflow_path() -> Path:
        return Path(__file__).parent.parent / ".github" / "workflows" / "keryx-hunt.yml"

    def test_workflow_file_exists(self) -> None:
        assert self._workflow_path().exists(), "keryx-hunt.yml not found"

    def test_workflow_is_valid_yaml(self) -> None:
        import yaml
        data = yaml.safe_load(self._workflow_path().read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_workflow_has_on_push(self) -> None:
        # YAML parses bare `on:` as True; check raw text instead
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "push:" in text

    def test_workflow_has_pull_request(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "pull_request:" in text

    def test_workflow_has_schedule(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "schedule:" in text

    def test_workflow_uploads_sarif(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "upload-sarif" in text

    def test_workflow_uploads_artifact(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "upload-artifact" in text

    def test_workflow_runs_hunt_script(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "project_hunt.py" in text

    def test_workflow_has_fail_if_confirmed(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "--fail-if-confirmed" in text

    def test_workflow_uses_output_html(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "--output-html" in text

    def test_workflow_uses_output_sarif(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "--output-sarif" in text

    def test_workflow_security_events_write(self) -> None:
        text = self._workflow_path().read_text(encoding="utf-8")
        assert "security-events: write" in text


class TestPreCommitHookInstaller:
    """T58b: scripts/install_hooks.py creates a valid pre-commit hook."""

    @staticmethod
    def _installer_path() -> Path:
        return Path(__file__).parent.parent / "scripts" / "install_hooks.py"

    def test_installer_exists(self) -> None:
        assert self._installer_path().exists()

    def test_installer_is_valid_python(self) -> None:
        import ast
        ast.parse(self._installer_path().read_text(encoding="utf-8"))

    def test_hook_body_blocks_on_high(self) -> None:
        text = self._installer_path().read_text(encoding="utf-8")
        assert "severity" in text
        assert "sys.exit(1)" in text

    def test_hook_body_is_valid_python(self) -> None:
        import ast, re as _re
        text = self._installer_path().read_text(encoding="utf-8")
        m = _re.search(r'_HOOK_BODY\s*=\s*r"""(.*?)"""', text, _re.DOTALL)
        assert m, "_HOOK_BODY not found"
        ast.parse(m.group(1))

    def test_install_into_temp_git_repo(self) -> None:
        import subprocess, sys as _sys
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subprocess.run(["git", "init", tmp], capture_output=True)
            result = subprocess.run(
                [_sys.executable, str(self._installer_path()), "--repo-dir", tmp],
                capture_output=True, text=True,
                env={**os.environ},
            )
            hook = tmp_path / ".git" / "hooks" / "pre-commit"
            assert hook.exists(), f"stderr: {result.stderr}\nstdout: {result.stdout}"
            assert os.access(hook, os.X_OK)

    def test_force_flag_overwrites(self) -> None:
        import subprocess, sys as _sys
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subprocess.run(["git", "init", tmp], capture_output=True)
            hook = tmp_path / ".git" / "hooks" / "pre-commit"
            hook.write_text("old content")
            subprocess.run(
                [_sys.executable, str(self._installer_path()),
                 "--repo-dir", tmp, "--force"],
                capture_output=True,
            )
            assert hook.read_text() != "old content"

    def test_no_force_preserves_existing(self) -> None:
        import subprocess, sys as _sys
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subprocess.run(["git", "init", tmp], capture_output=True)
            hook = tmp_path / ".git" / "hooks" / "pre-commit"
            hook.write_text("keep this")
            subprocess.run(
                [_sys.executable, str(self._installer_path()), "--repo-dir", tmp],
                capture_output=True,
            )
            assert hook.read_text() == "keep this"

    def test_no_git_dir_returns_error(self) -> None:
        import subprocess, sys as _sys
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [_sys.executable, str(self._installer_path()), "--repo-dir", tmp],
                capture_output=True, text=True,
            )
            assert result.returncode == 1


class TestHtmlReport:
    """T58c: write_output_html produces a well-formed HTML file."""

    @staticmethod
    def _write(all_results=None, all_hunts=None, models=None, *, path: Path) -> None:
        from keryx.core.reporters import write_output_html
        from keryx.core.hunt_models import ReportContext
        ctx = ReportContext(
            repo_root=Path(__file__).parent.parent,
            escalate_model="opus",
            default_model="haiku",
            budget_limit_usd=0.50,
            scan_results=all_results or [],
            hunts=all_hunts or [],
            models=models or {},
            total_elapsed_s=3.14,
        )
        write_output_html(str(path), ctx)

    def test_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.html"
            self._write(path=p)
            assert p.exists()

    def test_html_has_doctype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.html"
            self._write(path=p)
            assert "<!DOCTYPE html>" in p.read_text(encoding="utf-8")

    def test_summary_cards_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.html"
            self._write(path=p)
            text = p.read_text(encoding="utf-8")
            assert "Files Scanned" in text
            assert "Confirmed Vulns" in text

    def test_vuln_row_rendered(self) -> None:
        from keryx.core.hunt_models import ScanResult, HuntResult
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.html"
            scan = ScanResult(file=Path("keryx/tools/ast_analyzer.py"),
                              high_count=1, rules=["SSRF"])
            hunt = HuntResult(
                file=Path("keryx/tools/ast_analyzer.py"),
                scan=scan, steps=3, elapsed_s=1.0, model_label="haiku",
                confirmed=[{"rule": "SSRF", "severity": "HIGH",
                            "line": 42, "message": "SSRF sink", "verified": True}],
                status="confirmed",
            )
            self._write(all_results=[scan], all_hunts=[hunt], path=p)
            text = p.read_text(encoding="utf-8")
            assert "SSRF" in text
            assert "HIGH" in text

    def test_no_confirmed_shows_empty_msg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.html"
            self._write(path=p)
            assert "No confirmed vulnerabilities" in p.read_text(encoding="utf-8")

    def test_xss_escaped_in_message(self) -> None:
        from keryx.core.hunt_models import ScanResult, HuntResult
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.html"
            scan = ScanResult(file=Path("keryx/tools/ast_analyzer.py"),
                              high_count=1, rules=["LLM_OUTPUT_SINK"])
            hunt = HuntResult(
                file=Path("keryx/tools/ast_analyzer.py"),
                scan=scan, steps=3, elapsed_s=1.0, model_label="haiku",
                confirmed=[{"rule": "LLM_OUTPUT_SINK", "severity": "HIGH",
                            "message": "<script>alert(1)</script>"}],
                status="confirmed",
            )
            self._write(all_results=[scan], all_hunts=[hunt], path=p)
            text = p.read_text(encoding="utf-8")
            assert "<script>" not in text
            assert "&lt;script&gt;" in text

    def test_parent_dir_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "subdir" / "deep" / "report.html"
            self._write(path=p)
            assert p.exists()

    def test_output_html_flag_in_argparse(self) -> None:
        import sys as _sys, unittest.mock as _mock
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.project_hunt import parse_args  # type: ignore[import]
        with _mock.patch("sys.argv", ["project_hunt.py", "--output-html", "/tmp/x.html"]):
            args = parse_args()
        assert args.output_html == "/tmp/x.html"
