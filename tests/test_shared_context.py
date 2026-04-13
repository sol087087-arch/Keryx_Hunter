"""SharedContext: step tracking, hypothesis management, advisor state, serialization."""
import pytest
from keryx.core.shared_context import SharedContext


@pytest.fixture
def ctx():
    return SharedContext(target_path="/tmp/test_target", capability="deep_reasoning")


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_init(ctx):
    assert ctx.target_path == "/tmp/test_target"
    assert ctx.steps_taken == 0
    assert ctx.parse_errors == 0


# ------------------------------------------------------------------
# Step management
# ------------------------------------------------------------------

class _Step:
    def __init__(self, action, action_input=None, thought="", confidence=0.5):
        self.action = action
        self.action_input = action_input or {}
        self.thought = thought
        self.confidence = confidence


def test_add_step_increments_counter(ctx):
    ctx.add_step(_Step("codeql_query"), "some output")
    assert ctx.steps_taken == 1
    ctx.add_step(_Step("gdb_analyze"), "crash")
    assert ctx.steps_taken == 2


def test_get_last_confidence(ctx):
    assert ctx.get_last_confidence() is None
    ctx.add_step(_Step("codeql_query", confidence=0.75), "ok")
    conf = ctx.get_last_confidence()
    assert conf == pytest.approx(0.75)


def test_get_recent_history_empty(ctx):
    h = ctx.get_recent_history(5)
    assert isinstance(h, str)
    assert len(h) > 0


def test_get_recent_history_with_steps(ctx):
    ctx.add_step(_Step("read_file", {"file_path": "/foo.c"}, thought="look at foo"), "content")
    ctx.add_step(_Step("codeql_query", {}, thought="run query"), "result")
    h = ctx.get_recent_history(5)
    assert "read_file" in h or "codeql_query" in h


def test_trim_history(ctx):
    for i in range(20):
        ctx.add_step(_Step(f"action_{i}"), f"obs_{i}")
    ctx.trim_history(keep=6)
    # steps_taken must be preserved
    assert ctx.steps_taken == 20


# ------------------------------------------------------------------
# Hypothesis management
# ------------------------------------------------------------------

def test_add_hypothesis(ctx):
    ctx.add_hypothesis("buffer overflow in foo()")
    assert "buffer overflow in foo()" in ctx.hypotheses


def test_hypothesis_dedup(ctx):
    ctx.add_hypothesis("same")
    ctx.add_hypothesis("same")
    assert len([h for h in ctx.hypotheses if h == "same"]) == 1


def test_blacklist_hypothesis(ctx):
    ctx.add_hypothesis("false positive")
    ctx.blacklist_hypothesis("false positive")
    assert "false positive" not in ctx.hypotheses


def test_get_deduplicated_hypotheses_empty(ctx):
    result = ctx.get_deduplicated_hypotheses()
    assert isinstance(result, str)


def test_get_deduplicated_hypotheses_content(ctx):
    ctx.add_hypothesis("heap uaf in parse()")
    result = ctx.get_deduplicated_hypotheses()
    assert "heap uaf" in result


# ------------------------------------------------------------------
# Parse error tracking
# ------------------------------------------------------------------

def test_increment_parse_errors(ctx):
    ctx.increment_parse_errors()
    ctx.increment_parse_errors()
    assert ctx.parse_errors == 2


# ------------------------------------------------------------------
# Advisor state
# ------------------------------------------------------------------

def test_no_pending_advice_initially(ctx):
    assert ctx.has_pending_advisor_advice() is False


def test_add_and_consume_advice(ctx):
    ctx.add_advisor_advice({"strategy": "look deeper", "strategic_direction": "focus on IPC"})
    assert ctx.has_pending_advisor_advice() is True
    advice = ctx.get_pending_advisor_advice()
    assert advice is not None
    ctx.clear_pending_advisor_advice()
    assert ctx.has_pending_advisor_advice() is False


def test_set_get_advisor_guidance(ctx):
    ctx.set_advisor_guidance("focus on heap allocators")
    assert ctx.get_advisor_guidance() == "focus on heap allocators"


def test_clear_pending_advice(ctx):
    ctx.add_advisor_advice({"strategy": "test"})
    ctx.clear_pending_advisor_advice()
    assert ctx.has_pending_advisor_advice() is False


# ------------------------------------------------------------------
# Evidence / critique
# ------------------------------------------------------------------

def test_add_critique(ctx):
    ctx.add_critique("looks like a false positive")
    # no error = pass


def test_request_additional_evidence(ctx):
    ctx.add_step(_Step("read_file"), "content")
    ctx.request_additional_evidence("read_file")


# ------------------------------------------------------------------
# Serialization round-trip
# ------------------------------------------------------------------

def test_to_dict_from_dict_roundtrip(ctx):
    ctx.add_hypothesis("uaf in handle_msg()")
    ctx.add_step(_Step("codeql_query", confidence=0.6), "found issue")
    ctx.increment_parse_errors()

    d = ctx.to_dict()
    ctx2 = SharedContext.from_dict(d)

    assert ctx2.target_path == ctx.target_path
    assert ctx2.steps_taken == ctx.steps_taken
    assert ctx2.parse_errors == ctx.parse_errors
    assert "uaf in handle_msg()" in ctx2.hypotheses


# ------------------------------------------------------------------
# Metrics / summary
# ------------------------------------------------------------------

def test_get_metrics(ctx):
    m = ctx.get_metrics()
    assert "steps_count" in m
    assert "confirmed_vulns" in m


def test_build_summary(ctx):
    s = ctx.build_summary()
    assert isinstance(s, str)
    assert "/tmp/test_target" in s
