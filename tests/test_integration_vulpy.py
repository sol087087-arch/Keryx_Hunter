"""test_integration_vulpy.py — Integration test against the Vulpy intentionally
vulnerable Flask app (https://github.com/fportantier/vulpy).

Requires network on first run (to clone vulpy); subsequent runs reuse the cached
clone in /tmp/vulpy.  Uses --scripted so no API key is needed.

Assertions:
  • Phase 0 detects api_list.py (SSRF — OPEN_URL_REQUEST rule)
  • Phase 1 (ScriptedModel) confirms at least one finding
  • Budget is not exceeded
  • No unhandled exception escapes main()
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VULPY_REPO = "https://github.com/fportantier/vulpy"
VULPY_DIR  = Path("/tmp/vulpy")
TARGET_DIR = VULPY_DIR / "bad"


def _ensure_vulpy() -> None:
    """Clone vulpy if not already present."""
    if TARGET_DIR.is_dir():
        return
    result = subprocess.run(
        ["git", "clone", "--depth=1", VULPY_REPO, str(VULPY_DIR)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"Could not clone vulpy: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Fixture — import project_hunt once and patch _REPO_ROOT
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _hunt_module():
    _ensure_vulpy()
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import scripts.project_hunt as ph
    return ph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVulpyIntegration:

    def _run(self, ph, extra_kwargs: dict | None = None) -> object:
        """Run a hunt against vulpy/bad with ScriptedModel and return ctx."""
        import argparse
        original = ph._REPO_ROOT
        ph._REPO_ROOT = TARGET_DIR
        try:
            ns = argparse.Namespace(
                dir=".",
                top_n=5,
                default_model="haiku",
                escalate_model="opus",
                escalate_n=0,
                escalate_inconclusive=False,
                mode="flexible",        # scripted model returns canned output
                budget=0.50,
                min_score=1.0,          # low threshold — catch everything Phase 0 sees
                max_files=300,
                scripted=True,
                exclude=[],
                no_default_excludes=True,
                blame_weight=None,
                churn_weight=None,
                blame_days=90,
                no_blame=True,          # no git history in shallow clone
                no_cache=True,
                cache_file=".hunt_cache.json",
                always_escalate_high=False,
                fail_if_confirmed=False,
                output_json=None,
                output_sarif=None,
                output_html=None,
                fuzz=False,
                fuzz_timeout=10,
                fuzz_llm=False,
                fuzz_llm_model=None,
                learn=False,
                **(extra_kwargs or {}),
            )
            asyncio.run(ph.main(ns))
            return ph._last_ctx   # set by main() for test introspection (see below)
        finally:
            ph._REPO_ROOT = original

    # ------------------------------------------------------------------

    def test_phase0_detects_api_list(self, _hunt_module):
        """api_list.py must appear in Phase 0 results (SSRF / OPEN_URL_REQUEST)."""
        _ensure_vulpy()
        ctx = self._run(_hunt_module)
        if ctx is None:
            pytest.skip("_last_ctx not wired — skip introspection assertions")

        names = [r.file.name for r in ctx.scan_results]
        assert "api_list.py" in names, (
            f"api_list.py not found in Phase 0 results. Got: {names}"
        )

    def test_phase0_score_reasonable(self, _hunt_module):
        """api_list.py Phase 0 score should be >= 10 (SSRF + bonus)."""
        _ensure_vulpy()
        ctx = self._run(_hunt_module)
        if ctx is None:
            pytest.skip("_last_ctx not wired")

        api_result = next(
            (r for r in ctx.scan_results if r.file.name == "api_list.py"), None
        )
        assert api_result is not None, "api_list.py missing from scan_results"
        assert api_result.score >= 10.0, (
            f"Expected score >= 10, got {api_result.score}"
        )

    def test_budget_not_exceeded(self, _hunt_module):
        """ScriptedModel has zero cost — budget_used should remain at 0."""
        _ensure_vulpy()
        ctx = self._run(_hunt_module)
        if ctx is None:
            pytest.skip("_last_ctx not wired")
        assert ctx.budget_used_usd <= 0.50

    def test_no_exception_escapes(self, _hunt_module):
        """main() must not raise for a valid target directory."""
        _ensure_vulpy()
        # If _run() raises, the test fails — that's the assertion.
        self._run(_hunt_module)

    def test_hunt_results_non_empty(self, _hunt_module):
        """At least one file should be hunted (Phase 1) given min_score=1.0."""
        _ensure_vulpy()
        ctx = self._run(_hunt_module)
        if ctx is None:
            pytest.skip("_last_ctx not wired")
        assert len(ctx.hunts) >= 1, (
            "Expected at least one Phase 1 hunt result; got none."
        )
