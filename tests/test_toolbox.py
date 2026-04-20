"""ToolBox: registration, sync/async dispatch, error handling, metrics."""
import asyncio
import pytest
from keryx.tools.Toolbox import ToolBox, BaseTool, ToolResult, create_toolbox


# ------------------------------------------------------------------
# Minimal concrete tool
# ------------------------------------------------------------------

class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the input as output"

    async def execute(self, action_input, context=None):
        text = action_input.get("text", "")
        return ToolResult(success=True, output=f"ECHO:{text}")


class FailTool(BaseTool):
    name = "fail_tool"
    description = "Always fails"

    async def execute(self, action_input, context=None):
        return ToolResult(success=False, output="always fails", error="deliberate_failure")


class SlowTool(BaseTool):
    name = "slow_tool"
    description = "Sleeps longer than its timeout"
    default_timeout = 0.05  # 50 ms

    async def execute(self, action_input, context=None):
        await asyncio.sleep(10.0)
        return ToolResult(success=True, output="never reached")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def box():
    tb = create_toolbox(tools=[EchoTool(), FailTool(), SlowTool()])
    yield tb
    tb.shutdown()


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def test_register_and_list(box):
    tools = box.list_tools()
    assert "echo" in tools
    assert "fail_tool" in tools


def test_get_tool(box):
    t = box.get_tool("echo")
    assert t is not None
    assert t.name == "echo"


def test_get_unknown_tool(box):
    assert box.get_tool("nonexistent") is None


# ------------------------------------------------------------------
# Async execution
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_async_success(box):
    result = await box.execute_async("echo", {"text": "hello"})
    assert result.success is True
    assert "ECHO:hello" in result.output


@pytest.mark.asyncio
async def test_execute_async_unknown_tool(box):
    result = await box.execute_async("does_not_exist", {})
    assert result.success is False
    assert result.error == "tool_not_found"


@pytest.mark.asyncio
async def test_execute_async_fail_tool(box):
    result = await box.execute_async("fail_tool", {})
    assert result.success is False


@pytest.mark.asyncio
async def test_execute_async_timeout(box):
    result = await box.execute_async("slow_tool", {}, timeout=0.05)
    assert result.success is False
    assert result.error == "timeout"


@pytest.mark.asyncio
async def test_execute_async_invalid_input(box):
    result = await box.execute_async("echo", "not_a_dict")
    assert result.success is False
    assert result.error == "invalid_input_type"


# ------------------------------------------------------------------
# Sync execution
# ------------------------------------------------------------------

def test_execute_sync_success(box):
    result = box.execute("echo", {"text": "sync"})
    assert result.success is True
    assert "ECHO:sync" in result.output


def test_execute_sync_unknown_tool(box):
    result = box.execute("ghost", {})
    assert result.success is False


# ------------------------------------------------------------------
# ToolResult helpers
# ------------------------------------------------------------------

def test_tool_result_str(box):
    r = ToolResult(success=True, output="VULN_CONFIRMED heap-uaf")
    s = str(r)
    assert isinstance(s, str)


def test_tool_result_format_for_llm():
    r = ToolResult(success=False, output="something bad", error="oops")
    text = r.format_for_llm()
    assert "oops" in text or "ERROR" in text


# ------------------------------------------------------------------
# Batch execution
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_batch(box):
    actions = [
        ("echo", {"text": "a"}),
        ("echo", {"text": "b"}),
        ("fail_tool", {}),
    ]
    results = await box.execute_batch(actions)
    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is True
    assert results[2].success is False


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def test_metrics_after_calls(box):
    box.execute("echo", {"text": "x"})
    box.execute("echo", {"text": "y"})
    m = box.get_metrics()
    assert m["total_calls"] >= 2


def test_reset_metrics(box):
    box.execute("echo", {"text": "x"})
    box.reset_metrics()
    m = box.get_metrics()
    assert m["total_calls"] == 0
