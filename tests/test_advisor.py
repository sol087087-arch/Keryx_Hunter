"""AdvisorManager + BaseAdvisor: registration, dispatch, circuit breaker, cascade."""
import pytest
from keryx.advisors.base import (
    BaseAdvisor,
    AdvisorResponse,
    CascadeAdvisor,
    RuleBasedAdvisor,
)
from keryx.advisors.manager import AdvisorManager, create_advisor_manager
from keryx.core.shared_context import SharedContext


# ------------------------------------------------------------------
# Minimal concrete advisors
# ------------------------------------------------------------------

class AlwaysTriggerAdvisor(BaseAdvisor):
    name = "always"

    def should_trigger(self, context):
        return True

    async def advise(self, context):
        return AdvisorResponse(
            strategic_direction="go deeper",
            adjust_confidence_threshold=0.55,
        )


class NeverTriggerAdvisor(BaseAdvisor):
    name = "never"

    def should_trigger(self, context):
        return False

    async def advise(self, context):
        return AdvisorResponse()


class ErrorAdvisor(BaseAdvisor):
    name = "error_advisor"

    def should_trigger(self, context):
        return True

    async def advise(self, context):
        raise RuntimeError("advisor exploded")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def ctx():
    return SharedContext(target_path="/tmp/test", capability="deep_reasoning")


@pytest.fixture
def manager():
    m = create_advisor_manager()
    yield m
    m.shutdown()


# ------------------------------------------------------------------
# AdvisorResponse
# ------------------------------------------------------------------

def test_empty_response():
    r = AdvisorResponse()
    assert r.is_empty() is True


def test_non_empty_response():
    r = AdvisorResponse(strategic_direction="focus on IPC")
    assert r.is_empty() is False


def test_response_with_hypotheses():
    r = AdvisorResponse(suggested_hypotheses=["uaf in parse()"])
    assert r.is_empty() is False


# ------------------------------------------------------------------
# BaseAdvisor
# ------------------------------------------------------------------

def test_can_advise_initially():
    a = AlwaysTriggerAdvisor(max_calls_per_session=3)
    assert a.can_advise() is True


def test_can_advise_exhausted():
    a = AlwaysTriggerAdvisor(max_calls_per_session=1)
    a.calls_made = 1
    assert a.can_advise() is False


def test_reset():
    a = AlwaysTriggerAdvisor(max_calls_per_session=2)
    a.calls_made = 2
    a.reset()
    assert a.can_advise() is True


@pytest.mark.asyncio
async def test_advise_with_tracking_increments_count(ctx):
    a = AlwaysTriggerAdvisor(max_calls_per_session=5)
    response = await a.advise_with_tracking(ctx)
    assert a.calls_made == 1
    assert response.strategic_direction == "go deeper"


@pytest.mark.asyncio
async def test_advise_with_tracking_captures_exception(ctx):
    a = ErrorAdvisor(max_calls_per_session=5)
    response = await a.advise_with_tracking(ctx)
    assert "error" in response.metadata


@pytest.mark.asyncio
async def test_should_trigger_async(ctx):
    a = AlwaysTriggerAdvisor()
    result = await a.should_trigger_async(ctx)
    assert result is True


# ------------------------------------------------------------------
# CascadeAdvisor
# ------------------------------------------------------------------

def test_cascade_requires_at_least_one():
    with pytest.raises(ValueError):
        CascadeAdvisor(advisors=[])


@pytest.mark.asyncio
async def test_cascade_returns_first_non_empty(ctx):
    cascade = CascadeAdvisor(
        advisors=[NeverTriggerAdvisor(), AlwaysTriggerAdvisor()],
        max_calls_per_session=5,
    )
    response = await cascade.advise(ctx)
    assert response.strategic_direction == "go deeper"


@pytest.mark.asyncio
async def test_cascade_all_empty_returns_empty_metadata(ctx):
    cascade = CascadeAdvisor(
        advisors=[NeverTriggerAdvisor(), NeverTriggerAdvisor()],
        max_calls_per_session=5,
    )
    response = await cascade.advise(ctx)
    assert response.is_empty() is True
    assert "cascade" in response.metadata


# ------------------------------------------------------------------
# RuleBasedAdvisor
# ------------------------------------------------------------------

def test_rule_based_triggers_on_low_confidence(ctx):
    advisor = RuleBasedAdvisor(confidence_threshold=0.6, max_calls_per_session=5)
    # Inject a step with low confidence
    class _Step:
        action = "read_file"
        action_input = {}
        thought = ""
        confidence = 0.3
    ctx.add_step(_Step(), "content")
    assert advisor.should_trigger(ctx) is True


def test_rule_based_triggers_on_parse_errors(ctx):
    advisor = RuleBasedAdvisor(max_parse_errors=2, max_calls_per_session=5)
    ctx.parse_errors = 3
    assert advisor.should_trigger(ctx) is True


@pytest.mark.asyncio
async def test_rule_based_advice_content(ctx):
    ctx.parse_errors = 5
    advisor = RuleBasedAdvisor(max_parse_errors=3, max_calls_per_session=5)
    response = await advisor.advise(ctx)
    assert isinstance(response.strategic_direction, str)


# ------------------------------------------------------------------
# AdvisorManager
# ------------------------------------------------------------------

def test_register(manager):
    a = AlwaysTriggerAdvisor()
    manager.register(a, priority=80)
    assert manager.has_advisor("always")


def test_list_advisors(manager):
    manager.register(AlwaysTriggerAdvisor(), priority=50)
    manager.register(NeverTriggerAdvisor(), priority=90)
    names = manager.list_advisors()
    assert "always" in names
    assert "never" in names


def test_unregister(manager):
    manager.register(AlwaysTriggerAdvisor())
    manager.unregister("always")
    assert not manager.has_advisor("always")


def test_can_advise_no_advisors(manager):
    assert manager.can_advise() is False


def test_can_advise_with_advisor(manager):
    manager.register(AlwaysTriggerAdvisor(max_calls_per_session=3))
    assert manager.can_advise() is True


def test_calls_made_property(manager):
    assert manager.calls_made == 0


@pytest.mark.asyncio
async def test_get_advice_async_triggers(manager, ctx):
    manager.register(AlwaysTriggerAdvisor(max_calls_per_session=5))
    response = await manager.get_advice_async(ctx)
    assert manager.calls_made == 1
    assert response.strategic_direction == "go deeper"


@pytest.mark.asyncio
async def test_get_advice_async_not_found(manager, ctx):
    response = await manager.get_advice_async(ctx, advisor_name="ghost")
    assert "error" in response.metadata


@pytest.mark.asyncio
async def test_get_advice_async_no_trigger(manager, ctx):
    manager.register(NeverTriggerAdvisor(max_calls_per_session=5))
    response = await manager.get_advice_async(ctx)
    # triggered=False → empty response with metadata
    assert response.metadata.get("triggered") is False or response.is_empty()


def test_get_advice_sync(manager, ctx):
    manager.register(AlwaysTriggerAdvisor(max_calls_per_session=5))
    response = manager.get_advice(ctx)
    assert isinstance(response, AdvisorResponse)


def test_get_metrics(manager):
    m = manager.get_metrics()
    assert "total_calls" in m
    assert "advisor_health" in m
