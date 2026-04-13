"""KeryxAgent integration tests using MyLocalModel + stub toolbox."""
import asyncio
import pytest
from keryx.models.local_mock import MyLocalModel
from keryx.tools.Toolbox import ToolBox, BaseTool, ToolResult, create_toolbox
from keryx.advisors.manager import AdvisorManager
from keryx.advisors.base import BaseAdvisor, AdvisorResponse
from keryx.core.agent import KeryxAgent, AgentStep, BudgetController
from keryx.core.shared_context import SharedContext


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class FinishTool(BaseTool):
    """Returns FINISH-like output to end the agent loop quickly."""
    name = "read_file"

    async def execute(self, action_input, context=None):
        return ToolResult(success=True, output="file content: nothing suspicious")


class NoOpAdvisor(BaseAdvisor):
    name = "noop-advisor"

    def should_trigger(self, context):
        return False

    async def advise(self, context):
        return AdvisorResponse()


class FixedOutputModel(MyLocalModel):
    """Returns a valid FINISH JSON every call so the agent terminates fast."""
    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None, **kw):
        return '{"thought":"done","action":"FINISH","action_input":{},"confidence":0.9}'


@pytest.fixture
def toolbox():
    tb = create_toolbox(tools=[FinishTool()])
    yield tb
    tb.shutdown()


@pytest.fixture
def advisor_manager():
    m = AdvisorManager()
    m.register(NoOpAdvisor(max_calls_per_session=3))
    yield m
    m.shutdown()


@pytest.fixture
def agent(toolbox, advisor_manager):
    return KeryxAgent(
        executor_model=FixedOutputModel(),
        advisor_manager=advisor_manager,
        toolbox=toolbox,
        max_steps=5,
        confidence_threshold=0.6,
    )


# ------------------------------------------------------------------
# BudgetController
# ------------------------------------------------------------------

def test_budget_controller_can_proceed():
    model = MyLocalModel()
    bc = BudgetController(max_cost_usd=10.0, max_calls=100)
    assert bc.can_proceed(model) is True


def test_budget_controller_exhausted_calls():
    model = MyLocalModel()
    bc = BudgetController(max_cost_usd=10.0, max_calls=0)
    assert bc.can_proceed(model) is False


def test_budget_controller_record_call():
    model = MyLocalModel()
    bc = BudgetController(max_cost_usd=10.0, max_calls=10)
    bc.record_call(model, tokens_in=1000, tokens_out=500)
    assert bc.calls_made == 1
    # local model has zero cost
    assert bc.current_cost == pytest.approx(0.0)


def test_budget_remaining():
    bc = BudgetController(max_cost_usd=5.0, max_calls=10)
    assert bc.remaining_budget == pytest.approx(5.0)


def test_budget_to_dict():
    bc = BudgetController(max_cost_usd=5.0, max_calls=10)
    d = bc.to_dict()
    assert d["max_cost_usd"] == 5.0
    assert "remaining_budget" in d


# ------------------------------------------------------------------
# AgentStep
# ------------------------------------------------------------------

def test_agent_step_defaults():
    step = AgentStep(thought="t", action="read_file", action_input={})
    assert step.confidence == 0.0
    assert step.observation == ""


# ------------------------------------------------------------------
# KeryxAgent
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_run_completes(agent):
    result = await agent.run(target_path="/tmp/fake_target", resume=False)
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_run_returns_target(agent):
    result = await agent.run(target_path="/tmp/test_target", resume=False)
    assert result["target"] == "/tmp/test_target"


@pytest.mark.asyncio
async def test_agent_steps_taken(agent):
    result = await agent.run(target_path="/tmp/t", resume=False)
    assert result["steps_taken"] >= 1


@pytest.mark.asyncio
async def test_agent_report_has_keys(agent):
    result = await agent.run(target_path="/tmp/t", resume=False)
    for key in ("status", "target", "steps_taken", "confirmed_vulns", "parse_errors"):
        assert key in result


@pytest.mark.asyncio
async def test_agent_no_budget(toolbox, advisor_manager):
    agent = KeryxAgent(
        executor_model=FixedOutputModel(),
        advisor_manager=advisor_manager,
        toolbox=toolbox,
        max_steps=3,
        budget_usd=None,
    )
    result = await agent.run("/tmp/t", resume=False)
    assert result["budget"] is None


@pytest.mark.asyncio
async def test_agent_with_budget(toolbox, advisor_manager):
    agent = KeryxAgent(
        executor_model=FixedOutputModel(),
        advisor_manager=advisor_manager,
        toolbox=toolbox,
        max_steps=3,
        budget_usd=1.0,
    )
    result = await agent.run("/tmp/t", resume=False)
    assert result["budget"] is not None
    assert result["budget"]["max_cost_usd"] == 1.0


# ------------------------------------------------------------------
# Parse fallback path
# ------------------------------------------------------------------

class GarbageModel(MyLocalModel):
    """Always returns garbage so JSON parse always fails."""
    def generate(self, prompt, config=None, *, grammar=None, max_tokens=None, **kw):
        return "not json at all @@@@"


@pytest.mark.asyncio
async def test_agent_handles_parse_failures(toolbox, advisor_manager):
    agent = KeryxAgent(
        executor_model=GarbageModel(),
        advisor_manager=advisor_manager,
        toolbox=toolbox,
        max_steps=4,
    )
    result = await agent.run("/tmp/t", resume=False)
    assert result["status"] == "completed"
    assert result["parse_errors"] > 0
