"""Unit tests for keryx/core/_loop_guards.py.

Pure functions — no mocks, no asyncio, no agent needed.
"""
import pytest
from keryx.core._loop_guards import check_exit_clean, observation_confirms_vuln
from keryx.core.agent import AgentStep


# ---------------------------------------------------------------------------
# observation_confirms_vuln
# ---------------------------------------------------------------------------

class TestObservationConfirmsVuln:

    @pytest.mark.parametrize("signal", [
        "VULN_CONFIRMED: injection accepted",
        "AddressSanitizer: heap-use-after-free at 0x...",
        "heap-use-after-free in target function",
        "stack-buffer-overflow detected",
        "SEGFAULT at address 0x0",
        "CRASH in fuzz target",
        "UAF in object finalizer",
        "[HIGH] GIT_OPTION_INJECTION @ line 42",
        "GIT_OPTION_INJECTION found in git_blame.py",
        "SUBPROCESS_SHELL_TRUE at line 10",
        "HARDCODED_SECRET: password in config",
    ])
    def test_returns_true_for_known_signals(self, signal: str) -> None:
        assert observation_confirms_vuln(signal) is True

    @pytest.mark.parametrize("clean", [
        "No findings.",
        "[AST] No HIGH results in file (150 lines analyzed).",
        "everything is fine",
        "",
        "LOW: unused import",
        "MEDIUM: missing type hint",
    ])
    def test_returns_false_for_clean_output(self, clean: str) -> None:
        assert observation_confirms_vuln(clean) is False

    def test_case_sensitive(self) -> None:
        # Signals are uppercase — lowercase should NOT match
        assert observation_confirms_vuln("vuln_confirmed") is False
        assert observation_confirms_vuln("crash") is False

    def test_partial_match_in_longer_string(self) -> None:
        assert observation_confirms_vuln("line 1: nothing\nline 2: CRASH\nline 3: done") is True


# ---------------------------------------------------------------------------
# check_exit_clean
# ---------------------------------------------------------------------------

def _codeql(observation: str) -> AgentStep:
    return AgentStep(
        thought="scan",
        action="codeql_query",
        action_input={},
        observation=observation,
    )


def _other_action() -> AgentStep:
    return AgentStep(
        thought="read",
        action="read_file",
        action_input={},
        observation="file contents",
    )


_CLEAN_OBS = "[AST] No findings in target.py (200 lines analyzed)."
_HIGH_OBS  = "[HIGH] GIT_OPTION_INJECTION @ line 55: cmd.extend(['--upload-pack', author])"


class TestCheckExitClean:

    # ── non-codeql actions are always ignored ────────────────────────────

    def test_non_codeql_action_never_triggers(self) -> None:
        should_exit, new_count = check_exit_clean(_other_action(), 0, 1)
        assert should_exit is False
        assert new_count == 0   # counter untouched

    def test_non_codeql_does_not_change_existing_counter(self) -> None:
        should_exit, new_count = check_exit_clean(_other_action(), 3, 5)
        assert should_exit is False
        assert new_count == 3

    # ── HIGH finding resets counter ──────────────────────────────────────

    def test_high_finding_resets_counter_to_zero(self) -> None:
        should_exit, new_count = check_exit_clean(_codeql(_HIGH_OBS), 4, 5)
        assert should_exit is False
        assert new_count == 0

    def test_high_finding_never_triggers_exit(self) -> None:
        # Even if counter was at max_clean - 1, a HIGH finding should NOT exit
        should_exit, _ = check_exit_clean(_codeql(_HIGH_OBS), 4, 4)
        assert should_exit is False

    # ── clean scan increments counter ────────────────────────────────────

    def test_first_clean_scan_below_threshold(self) -> None:
        should_exit, new_count = check_exit_clean(_codeql(_CLEAN_OBS), 0, 2)
        assert should_exit is False
        assert new_count == 1

    def test_clean_scan_increments_counter(self) -> None:
        _, count_after_1 = check_exit_clean(_codeql(_CLEAN_OBS), 0, 3)
        _, count_after_2 = check_exit_clean(_codeql(_CLEAN_OBS), count_after_1, 3)
        assert count_after_1 == 1
        assert count_after_2 == 2

    # ── threshold logic ──────────────────────────────────────────────────

    def test_exits_when_counter_reaches_threshold(self) -> None:
        # max_clean=1: first clean scan should trigger exit
        should_exit, new_count = check_exit_clean(_codeql(_CLEAN_OBS), 0, 1)
        assert should_exit is True
        assert new_count == 1

    def test_exits_on_second_scan_when_threshold_is_2(self) -> None:
        _, count = check_exit_clean(_codeql(_CLEAN_OBS), 0, 2)   # first scan
        should_exit, final_count = check_exit_clean(_codeql(_CLEAN_OBS), count, 2)
        assert should_exit is True
        assert final_count == 2

    def test_does_not_exit_before_threshold(self) -> None:
        should_exit, _ = check_exit_clean(_codeql(_CLEAN_OBS), 0, 5)
        assert should_exit is False

    # ── pure function contract ────────────────────────────────────────────

    def test_does_not_mutate_action(self) -> None:
        action = _codeql(_CLEAN_OBS)
        original_obs = action.observation
        check_exit_clean(action, 0, 1)
        assert action.observation == original_obs

    def test_always_returns_two_tuple(self) -> None:
        result = check_exit_clean(_codeql(_CLEAN_OBS), 0, 1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)

    # ── high-then-clean scenario ──────────────────────────────────────────

    def test_high_then_clean_resets_and_counts_from_zero(self) -> None:
        # Build up count to 2, then see a HIGH (resets), then one clean
        _, count = check_exit_clean(_codeql(_CLEAN_OBS), 0, 5)   # count=1
        _, count = check_exit_clean(_codeql(_CLEAN_OBS), count, 5)  # count=2
        _, count = check_exit_clean(_codeql(_HIGH_OBS),  count, 5)  # HIGH → count=0
        should_exit, count = check_exit_clean(_codeql(_CLEAN_OBS), count, 5)  # count=1
        assert should_exit is False
        assert count == 1
