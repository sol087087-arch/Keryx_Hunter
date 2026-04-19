"""Unit tests for keryx/core/verification_pipeline.py."""
from __future__ import annotations

import asyncio
import pytest

from keryx.core.verification_pipeline import VerificationPipeline
from keryx.core.shared_context import SharedContext
from keryx.tools.Toolbox import BaseTool, ToolBox, ToolResult, create_toolbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(target: str = "/tmp/git_blame.py") -> SharedContext:
    return SharedContext(target_path=target)


class _NamedTool(BaseTool):
    """Minimal tool registered under a given name."""
    def __init__(self, tool_name: str) -> None:
        self.name = tool_name

    async def execute(self, action_input, context=None):
        return ToolResult(success=True, output="ok")


def _tb(*tool_names: str) -> ToolBox:
    tools = [_NamedTool(n) for n in tool_names]
    return create_toolbox(tools=tools)


# ---------------------------------------------------------------------------
# resolve_target_tool
# ---------------------------------------------------------------------------

class TestResolveTargetTool:

    def test_exact_stem_match(self) -> None:
        tb = _tb("git_blame")
        vp = VerificationPipeline(tb, _ctx("/tmp/git_blame.py"))
        assert vp.resolve_target_tool() == "git_blame"
        tb.shutdown()

    def test_normalised_hyphen_to_underscore(self) -> None:
        tb = _tb("git_blame")
        vp = VerificationPipeline(tb, _ctx("/tmp/git-blame.py"))
        assert vp.resolve_target_tool() == "git_blame"
        tb.shutdown()

    def test_partial_match_stem_in_tool(self) -> None:
        tb = _tb("git_blame_extended")
        vp = VerificationPipeline(tb, _ctx("/tmp/git_blame.py"))
        assert vp.resolve_target_tool() == "git_blame_extended"
        tb.shutdown()

    def test_partial_match_tool_in_stem(self) -> None:
        tb = _tb("git_blame")
        vp = VerificationPipeline(tb, _ctx("/tmp/git_blame_wrapper.py"))
        assert vp.resolve_target_tool() == "git_blame"
        tb.shutdown()

    def test_returns_none_when_no_match(self) -> None:
        tb = _tb("codeql_query", "fuzzer_run")
        vp = VerificationPipeline(tb, _ctx("/tmp/unknown_tool.py"))
        assert vp.resolve_target_tool() is None
        tb.shutdown()

    def test_returns_none_on_empty_toolbox(self) -> None:
        tb = create_toolbox(tools=[])
        vp = VerificationPipeline(tb, _ctx("/tmp/git_blame.py"))
        assert vp.resolve_target_tool() is None
        tb.shutdown()


# ---------------------------------------------------------------------------
# extract_findings  (pure static method)
# ---------------------------------------------------------------------------

_GIT_OPTION_OBS = (
    "[HIGH] GIT_OPTION_INJECTION @ line 55: "
    'cmd.extend(["--upload-pack", author])\n'
)
_SHELL_TRUE_OBS = "[HIGH] SUBPROCESS_SHELL_TRUE @ line 12: subprocess.run(cmd, shell=True)\n"
_HARDCODED_OBS  = "[HIGH] HARDCODED_SECRET @ line 8: password = 'hunter2'\n"
_CLEAN_OBS      = "[AST] No findings in target.py (200 lines analyzed).\n"


class TestExtractFindings:

    def test_git_option_injection(self) -> None:
        findings = VerificationPipeline.extract_findings(_GIT_OPTION_OBS)
        assert len(findings) == 1
        rule, field, payload = findings[0]
        assert rule == "GIT_OPTION_INJECTION"
        assert field == "author"
        assert payload == "--upload-pack=test"

    def test_subprocess_shell_true(self) -> None:
        findings = VerificationPipeline.extract_findings(_SHELL_TRUE_OBS)
        assert len(findings) == 1
        rule, field, payload = findings[0]
        assert rule == "SUBPROCESS_SHELL_TRUE"
        assert field == "file"
        assert payload == "$(id)"

    def test_hardcoded_secret_skipped(self) -> None:
        # HARDCODED_SECRET has no payload → filtered out
        findings = VerificationPipeline.extract_findings(_HARDCODED_OBS)
        assert findings == []

    def test_clean_observation_returns_empty(self) -> None:
        assert VerificationPipeline.extract_findings(_CLEAN_OBS) == []

    def test_empty_string_returns_empty(self) -> None:
        assert VerificationPipeline.extract_findings("") == []

    def test_duplicate_rule_deduplicated(self) -> None:
        obs = _GIT_OPTION_OBS + _GIT_OPTION_OBS   # same rule twice
        findings = VerificationPipeline.extract_findings(obs)
        assert len(findings) == 1

    def test_multiple_rules_in_one_observation(self) -> None:
        obs = _GIT_OPTION_OBS + _SHELL_TRUE_OBS
        findings = VerificationPipeline.extract_findings(obs)
        rules = {f[0] for f in findings}
        assert "GIT_OPTION_INJECTION" in rules
        assert "SUBPROCESS_SHELL_TRUE" in rules

    def test_unsanitized_subprocess_arg(self) -> None:
        obs = "[HIGH] UNSANITIZED_SUBPROCESS_ARG @ line 20: cmd.append(file_path)\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert len(findings) == 1
        rule, field, payload = findings[0]
        assert rule == "UNSANITIZED_SUBPROCESS_ARG"
        assert field == "file"
        assert "passwd" in payload

    def test_open_user_path(self) -> None:
        obs = "[HIGH] OPEN_USER_PATH @ line 33: open(path)\n"
        findings = VerificationPipeline.extract_findings(obs)
        assert len(findings) == 1
        assert findings[0][1] == "path"


# ---------------------------------------------------------------------------
# auto_verify  (async, uses toolbox)
# ---------------------------------------------------------------------------

class _InjectionVerifier(BaseTool):
    """Stub injection_verifier: returns VULN_CONFIRMED when payload matches."""
    name = "injection_verifier"

    def __init__(self, confirm: bool = True) -> None:
        super().__init__()
        self._confirm = confirm

    async def execute(self, action_input, context=None):
        out = "VULN_CONFIRMED: payload accepted" if self._confirm else "REJECTED: sanitized"
        return ToolResult(success=True, output=out)


class _RaisingVerifier(BaseTool):
    """Stub injection_verifier that always raises."""
    name = "injection_verifier"

    def __init__(self) -> None:
        super().__init__()

    async def execute(self, action_input, context=None):
        raise RuntimeError("verifier exploded")


class TestAutoVerify:

    def _pipeline(self, tool: BaseTool, target: str = "/tmp/git_blame.py") -> tuple[VerificationPipeline, ToolBox]:
        tb = create_toolbox(tools=[_NamedTool("git_blame"), tool])
        ctx = _ctx(target)
        return VerificationPipeline(tb, ctx), tb

    @pytest.mark.asyncio
    async def test_returns_true_on_confirmed_finding(self) -> None:
        vp, tb = self._pipeline(_InjectionVerifier(confirm=True))
        confirmed, method = await vp.auto_verify(_GIT_OPTION_OBS)
        assert confirmed is True
        assert method == "injection_verifier"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_returns_false_when_rejected(self) -> None:
        vp, tb = self._pipeline(_InjectionVerifier(confirm=False))
        confirmed, method = await vp.auto_verify(_GIT_OPTION_OBS)
        assert confirmed is False
        assert method == "injection_verifier"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_returns_false_on_clean_observation(self) -> None:
        vp, tb = self._pipeline(_InjectionVerifier(confirm=True))
        confirmed, method = await vp.auto_verify(_CLEAN_OBS)
        assert confirmed is False
        assert method == "injection_verifier"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_returns_false_when_tool_not_resolved(self) -> None:
        # When target_tool is None, auto_verify falls back to fuzzer PoC.
        # The method tag must be "fuzz_poc"; confirmed depends on fuzzer outcome.
        from unittest.mock import patch
        from keryx.fuzzing.poc import PoCResult
        tb = create_toolbox(tools=[_InjectionVerifier(confirm=True)])
        vp = VerificationPipeline(tb, _ctx("/tmp/git_blame.py"))
        not_reproduced = PoCResult(rule="GIT_OPTION_INJECTION", reproduced=False,
                                   payload="", marker="M")
        with patch("keryx.core.verification_pipeline.VerificationPipeline._fuzz_verify",
                   return_value=False):
            confirmed, method = await vp.auto_verify(_GIT_OPTION_OBS)
        assert confirmed is False
        assert method == "fuzz_poc"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_tool_not_resolved_fuzz_confirmed(self) -> None:
        # When fuzzer PoC succeeds for an external target, method is "fuzz_poc".
        from unittest.mock import patch
        tb = create_toolbox(tools=[_InjectionVerifier(confirm=True)])
        vp = VerificationPipeline(tb, _ctx("/tmp/git_blame.py"))
        with patch("keryx.core.verification_pipeline.VerificationPipeline._fuzz_verify",
                   return_value=True):
            confirmed, method = await vp.auto_verify(_GIT_OPTION_OBS)
        assert confirmed is True
        assert method == "fuzz_poc"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_verifier_exception_is_caught(self) -> None:
        vp, tb = self._pipeline(_RaisingVerifier())
        confirmed, method = await vp.auto_verify(_GIT_OPTION_OBS)
        assert confirmed is False   # exception caught, continues → False
        assert method == "injection_verifier"
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_confirmed_step_written_to_context(self) -> None:
        vp, tb = self._pipeline(_InjectionVerifier(confirm=True))
        await vp.auto_verify(_GIT_OPTION_OBS)
        actions = [s["action"]["name"] for s in vp.context.steps]
        assert "injection_verifier" in actions
        tb.shutdown()

    @pytest.mark.asyncio
    async def test_confirmed_evidence_written_to_context(self) -> None:
        vp, tb = self._pipeline(_InjectionVerifier(confirm=True))
        await vp.auto_verify(_GIT_OPTION_OBS)
        locs = [e.location for e in vp.context.evidence]
        assert any("injection_verifier" in loc for loc in locs)
        tb.shutdown()
