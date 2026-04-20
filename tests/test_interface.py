"""ModelInterface contract tests using MyLocalModel (no llama_cpp required)."""
import pytest
from keryx.models.local_mock import MyLocalModel
from keryx.models.interface import (
    GenerationConfig,
    GenerationResult,
    CostEstimate,
    ModelCapabilities,
    ModelInterface,
)


@pytest.fixture
def model():
    return MyLocalModel()


def test_isinstance(model):
    assert isinstance(model, ModelInterface)


def test_is_local(model):
    assert model.is_local is True


def test_requires_network(model):
    assert model.requires_network is False


def test_is_healthy(model):
    assert model.is_healthy() is True


def test_generate_returns_str(model):
    result = model.generate("hello")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_config(model):
    cfg = GenerationConfig(max_tokens=50, temperature=0.5)
    result = model.generate("hello", config=cfg)
    assert isinstance(result, str)


def test_generate_result(model):
    r = model.generate_result("test prompt")
    assert isinstance(r, GenerationResult)
    assert isinstance(r.text, str)
    assert r.tokens_input > 0
    assert r.tokens_output > 0
    assert r.duration_ms >= 0.0


def test_generate_stream(model):
    chunks = list(model.generate_stream("test"))
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_generate_async(model):
    import asyncio
    result = asyncio.run(model.generate_async("async test"))
    assert isinstance(result, str)


def test_tokenize(model):
    tokens = model.tokenize("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_count_tokens(model):
    n = model.count_tokens("hello world")
    assert isinstance(n, int)
    assert n > 0


def test_get_context_length(model):
    length = model.get_context_length()
    assert isinstance(length, int)
    assert length > 0


def test_estimate_cost(model):
    cost = model.estimate_cost(100, 50)
    assert cost["input_cost_usd"] == 0.0
    assert cost["output_cost_usd"] == 0.0
    assert cost["total_cost_usd"] == 0.0


def test_get_usage_cost(model):
    cost = model.get_usage_cost()
    assert isinstance(cost["total_cost_usd"], float)


def test_get_capabilities(model):
    caps = model.get_capabilities()
    assert caps["is_local"] is True
    assert caps["supports_grammar"] is True
    assert isinstance(caps["max_context_length"], int)


def test_capabilities_property(model):
    caps = model.capabilities
    assert caps["is_local"] is True


def test_generate_with_tools(model):
    from keryx.models.interface import ToolDefinition
    tools = [ToolDefinition(name="noop", description="does nothing", parameters={})]
    result = model.generate_with_tools("test", tools)
    assert isinstance(result, str)


def test_unload_does_not_raise(model):
    model.unload()


def test_cost_per_token_is_zero(model):
    assert model.cost_per_1k_input_tokens == 0.0
    assert model.cost_per_1k_output_tokens == 0.0
