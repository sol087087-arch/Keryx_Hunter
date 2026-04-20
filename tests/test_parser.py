"""Unit tests for keryx/core/_parser.py.

Pure functions — no mocks, no agent, no context required.
"""
import pytest
from keryx.core._parser import extract_json, fallback_parse, parse_response


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:

    def test_plain_json_object(self) -> None:
        raw = '{"action": "read_file", "thought": "check it", "action_input": {}, "confidence": 0.7}'
        assert extract_json(raw) == raw

    def test_strips_markdown_fence(self) -> None:
        raw = '```json\n{"action": "FINISH", "thought": "done", "action_input": {}, "confidence": 0.9}\n```'
        result = extract_json(raw)
        assert result is not None
        assert result.startswith("{")

    def test_strips_plain_fence_without_json_label(self) -> None:
        raw = '```\n{"action": "NO_ACTION", "thought": "", "action_input": {}, "confidence": 0.1}\n```'
        result = extract_json(raw)
        assert result is not None
        assert result.startswith("{")

    def test_extracts_object_from_surrounding_text(self) -> None:
        raw = 'Here is my response: {"action": "codeql_query", "thought": "scan", "action_input": {}, "confidence": 0.8} done.'
        result = extract_json(raw)
        assert result is not None
        assert '"action"' in result

    def test_returns_none_on_empty_string(self) -> None:
        assert extract_json("") is None

    def test_returns_none_when_no_object(self) -> None:
        assert extract_json("no json here at all") is None

    def test_returns_none_on_unclosed_brace(self) -> None:
        assert extract_json('{"action": "read_file"') is None

    def test_strips_html_injection(self) -> None:
        raw = '<script>alert(1)</script>{"action": "FINISH", "thought": "t", "action_input": {}, "confidence": 0.9}'
        result = extract_json(raw)
        assert result is not None
        assert "<script>" not in result

    def test_strips_tool_call_tags(self) -> None:
        raw = '</tool_call>{"action": "NO_ACTION", "thought": "", "action_input": {}, "confidence": 0.1}'
        result = extract_json(raw)
        assert result is not None
        assert "</tool_call>" not in result

    def test_strips_c0_control_characters(self) -> None:
        # NUL byte and BEL before the JSON
        raw = '\x00\x07{"action": "FINISH", "thought": "t", "action_input": {}, "confidence": 0.9}'
        result = extract_json(raw)
        assert result is not None
        assert result.startswith("{")

    def test_uses_outermost_braces(self) -> None:
        # rfind("}") should pick the last closing brace
        raw = '{"action": "read_file", "action_input": {"path": "/tmp/f"}, "thought": "t", "confidence": 0.5}'
        result = extract_json(raw)
        assert result is not None
        assert result.endswith("}")
        assert '"path"' in result


# ---------------------------------------------------------------------------
# fallback_parse
# ---------------------------------------------------------------------------

class TestFallbackParse:

    @pytest.mark.parametrize("action_name", [
        "codeql_query", "gdb_analyze", "fuzzer_run", "git_blame",
        "injection_verifier", "read_file", "rag_search", "FINISH",
    ])
    def test_detects_known_action_in_free_text(self, action_name: str) -> None:
        raw = f"I think we should do {action_name} on the target file."
        step = fallback_parse(raw)
        assert step.action == action_name
        assert step.confidence == pytest.approx(0.3)
        assert step.thought == "fallback_parse_inferred"

    def test_returns_no_action_when_nothing_found(self) -> None:
        step = fallback_parse("gibberish @@@@ !!!")
        assert step.action == "NO_ACTION"
        assert step.confidence == pytest.approx(0.1)
        assert step.thought == "fallback_parse_failed"

    def test_no_action_keyword_does_not_self_match(self) -> None:
        # "NO_ACTION" in text should NOT match (it's excluded by design)
        step = fallback_parse("the output was NO_ACTION for sure")
        assert step.action == "NO_ACTION"
        assert step.thought == "fallback_parse_failed"

    def test_case_insensitive_match(self) -> None:
        step = fallback_parse("please run CODEQL_QUERY now")
        assert step.action == "codeql_query"

    def test_action_input_is_empty_dict(self) -> None:
        step = fallback_parse("run read_file please")
        assert step.action_input == {}

    def test_observation_is_empty(self) -> None:
        step = fallback_parse("run read_file please")
        assert step.observation == ""


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:

    def test_valid_json_returns_step_no_error(self) -> None:
        raw = '{"thought": "check", "action": "read_file", "action_input": {"path": "/f"}, "confidence": 0.75}'
        step, had_error = parse_response(raw)
        assert step.action == "read_file"
        assert step.confidence == pytest.approx(0.75)
        assert step.thought == "check"
        assert had_error is False

    def test_valid_json_action_input_preserved(self) -> None:
        raw = '{"thought": "t", "action": "codeql_query", "action_input": {"path": "/x.py"}, "confidence": 0.8}'
        step, _ = parse_response(raw)
        assert step.action_input == {"path": "/x.py"}

    def test_missing_action_key_returns_error(self) -> None:
        raw = '{"thought": "t", "action_input": {}, "confidence": 0.5}'
        step, had_error = parse_response(raw)
        assert had_error is True
        assert step.action == "NO_ACTION"

    def test_invalid_json_returns_error(self) -> None:
        raw = "not json at all @@@@"
        step, had_error = parse_response(raw)
        assert had_error is True

    def test_empty_string_returns_error(self) -> None:
        _, had_error = parse_response("")
        assert had_error is True

    def test_json_with_defaults(self) -> None:
        # Minimal valid JSON — missing optional fields get defaults
        raw = '{"action": "FINISH", "action_input": {}}'
        step, had_error = parse_response(raw)
        assert step.action == "FINISH"
        assert step.confidence == pytest.approx(0.5)   # default
        assert step.thought == ""                       # default
        assert had_error is False

    def test_confidence_coerced_to_float(self) -> None:
        raw = '{"thought": "t", "action": "read_file", "action_input": {}, "confidence": "0.6"}'
        step, had_error = parse_response(raw)
        assert isinstance(step.confidence, float)
        assert step.confidence == pytest.approx(0.6)
        assert had_error is False

    def test_markdown_fence_is_handled(self) -> None:
        raw = '```json\n{"thought": "t", "action": "FINISH", "action_input": {}, "confidence": 0.9}\n```'
        step, had_error = parse_response(raw)
        assert step.action == "FINISH"
        assert had_error is False

    def test_html_injection_stripped_before_parse(self) -> None:
        raw = '<script>evil()</script>{"thought":"t","action":"FINISH","action_input":{},"confidence":0.9}'
        step, had_error = parse_response(raw)
        assert step.action == "FINISH"
        assert had_error is False

    def test_fallback_fires_on_truncated_json(self) -> None:
        raw = '{"thought": "t", "action": "read_file", "action_input": {'
        step, had_error = parse_response(raw)
        assert had_error is True

    def test_no_error_flag_does_not_mean_no_action(self) -> None:
        # A successful parse that happens to produce NO_ACTION is still had_error=False
        raw = '{"thought": "wait", "action": "NO_ACTION", "action_input": {}, "confidence": 0.2}'
        step, had_error = parse_response(raw)
        assert step.action == "NO_ACTION"
        assert had_error is False
