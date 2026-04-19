#!/usr/bin/env python3
"""project_hunt.py — multi-file self-hunt across the KeryxHunter codebase.

Phase 0a (no LLM): ASTAnalyzerTool scans every .py file → raw ScanResult list
Phase 0b (git):    enrich_with_blame() adds recency + churn scores → re-rank
Phase 1  (LLM):    Two-tier hunt:
                     ★ top --escalate-n files  → --escalate-model (deep, e.g. opus)
                     → remaining top-N files   → --default-model  (fast, e.g. haiku)
Phase 2:           Aggregated report + per-model cost breakdown

Usage:
    python scripts/project_hunt.py [options]

    --dir            keryx/        directory to scan            (default: keryx/)
    --top-n          5             total files to hunt          (default: 5)
    --default-model  haiku         model for most files         (default: haiku)
    --escalate-model opus          model for top suspects       (default: opus)
    --escalate-n     0             top-scored files → escalate  (default: 0)
    --no-escalate-inconclusive     disable Phase 1c re-escalation (default: on)
    --mode           strict        verification_mode            (default: strict)
    --budget         0.50          shared USD cap per run       (default: 0.50)
    --scripted                     use ScriptedModel regardless of API key
    --min-score      1.0           skip files below this score  (default: 1.0)
    --max-files      300           Phase 0 file limit           (default: 300)

    Scoring weights (Phase 0):
    --blame-weight   4.0           weight for git recency score (default: 4.0)
    --churn-weight   2.5           weight for commit freq score (default: 2.5)
    --blame-days     90            git history window in days   (default: 90)
    --no-blame                     skip git enrichment (pure AST, faster)

    CI / output:
    --always-escalate-high         always route HIGH files to escalate_model
    --fail-if-confirmed            exit code 1 when any vuln confirmed
    --output-json PATH             write structured JSON report to PATH
    --output-sarif PATH            write SARIF 2.1.0 report to PATH (GitHub Security tab)
    --output-html PATH             write self-contained HTML report to PATH

    Incremental mode (Phase 0 cache):
    --no-cache                     ignore existing cache, force full rescan
    --cache-file PATH              cache location (default: .hunt_cache.json)

    Second run on unchanged codebase: 0 files re-scanned (100% cache hit).
    Any file touched since last run (git diff + mtime) is re-scanned automatically.

Examples:
    # All files through Haiku, git enrichment on
    python scripts/project_hunt.py --dir keryx --top-n 10

    # Top-3 through Opus, rest through Haiku, no git enrichment
    python scripts/project_hunt.py --dir keryx --top-n 10 --escalate-n 3 --no-blame

    # Custom weights
    python scripts/project_hunt.py --blame-weight 8.0 --churn-weight 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from keryx.advisors.base import RuleBasedAdvisor
from keryx.advisors.manager import create_advisor_manager
from keryx.core.agent import KeryxAgent
from keryx.models.interface import (
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelInterface,
    ToolCall,
    ToolDefinition,
)
from keryx.tools.Toolbox import ToolResult, create_default_toolbox
from keryx.tools.ast_analyzer import ASTAnalyzerTool

# Short-name → canonical API model ID
MODEL_IDS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# max_steps budget per tier
_STEPS_DEFAULT  = 10
_STEPS_ESCALATE = 15

# Always-excluded — never useful to scan
_EXCLUDE_ALWAYS = ("__pycache__", ".git")
# Default soft-excludes — overridable via --exclude / --no-default-excludes
_EXCLUDE_DEFAULT = ("tests/", "sandbox/", "scripts/")

# Incremental cache
_CACHE_FILE    = ".hunt_cache.json"
_CACHE_VERSION = 1

# Rule → recommended fix (shown in Phase 2 breakdown)
_RULE_FIX: dict[str, str] = {
    "SUBPROCESS_SHELL_TRUE":      "Pass a list instead of a string; remove shell=True",
    "GIT_OPTION_INJECTION":       "Use --flag=value form; add ['--', path] before path args",
    "GIT_FLAG_NAME_INJECTION":    "Never build flag names from user input; use an allow-list",
    "OPEN_USER_PATH":             "Resolve & validate path against an allowed root before open()",
    "HARDCODED_SECRET":           "Move secret to env var or secrets manager; rotate immediately",
    "SUBPROCESS_EXEC_STARRED":    "Expand *args before the call; validate each element",
    "UNSANITIZED_SUBPROCESS_ARG": "Validate/escape arg or use --flag=value; never pass raw user input",
    "LLM_OUTPUT_SINK":            "Parse LLM output with strict schema (JSON schema / Pydantic); never eval",
    "UNSAFE_EVAL_EXEC":           "Replace eval/exec with ast.literal_eval or a safe parser",
    "UNSAFE_PICKLE":              "Replace pickle with json/msgpack; if pickle required, sign payloads with hmac",
    "UNSAFE_YAML_LOAD":           "Use yaml.safe_load() or pass Loader=yaml.SafeLoader explicitly",
    "OS_SHELL_INJECTION":         "Use subprocess.run(list) instead of os.system/popen with a string",
    "SSRF":                       "Validate URL against an allowlist; reject private/internal addresses",
    "TEMPLATE_INJECTION":         "Never pass user input as a template string; use it only as render() context",
    "REGEX_DOS":                  "Compile patterns from constants only; reject untrusted regex input",
}

args_global: argparse.Namespace  # set in main() before any printing


# ---------------------------------------------------------------------------
# Phase 0 — data model
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    file:         Path
    high_count:   int        = 0
    medium_count: int        = 0
    rules:        list[str]  = field(default_factory=list)
    score:        float      = 0.0
    # Git enrichment signals — populated by enrich_with_blame()
    extra:        dict[str, float] = field(default_factory=dict)


class ScanScorer:
    """Extensible scorer. Weights are configurable; extra dict for future signals."""

    BASE_WEIGHTS: dict[str, float] = {"HIGH": 10.0, "MEDIUM": 3.0}

    # Default weights — overridable via CLI
    DEFAULT_EXTRA_WEIGHTS: dict[str, float] = {
        "blame": 4.0,   # git recency: how recently was the file touched?
        "churn": 2.5,   # git frequency: how often is it committed to?
    }

    def __init__(
        self,
        blame_weight: float | None = None,
        churn_weight: float | None = None,
    ) -> None:
        self.extra_weights = dict(self.DEFAULT_EXTRA_WEIGHTS)
        if blame_weight is not None:
            self.extra_weights["blame"] = blame_weight
        if churn_weight is not None:
            self.extra_weights["churn"] = churn_weight

    def compute_score(self, result: ScanResult) -> float:
        score = (
            result.high_count   * self.BASE_WEIGHTS["HIGH"]
            + result.medium_count * self.BASE_WEIGHTS["MEDIUM"]
        )
        for key, weight in self.extra_weights.items():
            if key in result.extra:
                score += result.extra[key] * weight
        result.score = round(score, 2)
        return result.score


# ---------------------------------------------------------------------------
# Incremental cache — Phase 0a persistence
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> tuple[dict[str, dict], str | None]:
    """Load Phase 0a cache.

    Returns (entries, last_commit).  entries is a dict keyed by repo-relative
    path; last_commit is the HEAD hash recorded when cache was written (or None).
    Returns ({}, None) on any miss/error/version mismatch.
    """
    if not cache_path.exists():
        return {}, None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != _CACHE_VERSION:
            return {}, None
        return data.get("entries", {}), data.get("commit")
    except Exception:
        return {}, None


def save_cache(
    cache_path: Path,
    entries:    dict[str, dict],
    commit:     str | None,
) -> None:
    """Persist Phase 0a scan results.  Silently ignores write errors."""
    try:
        cache_path.write_text(
            json.dumps(
                {"version": _CACHE_VERSION, "timestamp": time.time(),
                 "commit": commit, "entries": entries},
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[Cache] WARN: could not write {cache_path}: {exc}")


import subprocess as _sp   # noqa: E402  (used only in cache helpers)


def _git_head() -> str | None:
    """Return current HEAD commit hash, or None if not in a git repo."""
    try:
        r = _sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _git_changed_since(commit: str) -> set[str]:
    """Repo-relative paths changed since *commit* (committed + uncommitted).

    Combines:
      git diff --name-only <commit>   → files changed in commits after *commit*
      git status --porcelain          → staged / unstaged / untracked changes
    """
    changed: set[str] = set()
    try:
        # Committed changes since cached commit
        r = _sp.run(
            ["git", "diff", "--name-only", commit],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10,
        )
        changed.update(line for line in r.stdout.splitlines() if line)

        # Staged / unstaged / untracked changes not yet committed
        r = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if len(line) > 3:
                # "XY path" or "XY orig -> new"
                fname = line[3:].split(" -> ")[-1].strip().strip('"')
                changed.add(fname)
    except Exception:
        pass
    return changed


# ---------------------------------------------------------------------------
# Phase 0a — AST scanning
# ---------------------------------------------------------------------------

_ast_tool = ASTAnalyzerTool()
_scan_sem = asyncio.Semaphore(8)


def parse_ast_findings(result: ToolResult) -> tuple[int, int, list[str]]:
    """Extract (high_count, medium_count, rules) from ASTAnalyzerTool output."""
    if not result.success:
        return 0, 0, []

    findings = result.data.get("findings", [])
    if findings:
        high  = sum(1 for f in findings if f.get("severity") == "HIGH")
        med   = sum(1 for f in findings if f.get("severity") == "MEDIUM")
        rules = list({f["rule"] for f in findings})
        return high, med, rules

    # Text fallback (e.g. future external CodeQL integration)
    high  = len(re.findall(r'\[HIGH\]',   result.output))
    med   = len(re.findall(r'\[MEDIUM\]', result.output))
    rules = list(set(re.findall(r'\[(?:HIGH|MEDIUM)\]\s+(\w+)', result.output)))
    return high, med, rules


async def _ast_analyze(file: Path) -> ToolResult:
    async with _scan_sem:
        return await _ast_tool.execute({"file_path": str(file)})


async def scan_directory(
    dir_path:   Path,
    scorer:     ScanScorer,
    exclude:    tuple[str, ...],
    max_files:  int              = 300,
    min_score:  float            = 1.0,
    cache:      dict[str, dict] | None = None,
    changed:    set[str]        | None = None,
) -> tuple[list[ScanResult], dict[str, dict], int]:
    """Phase 0a: parallel AST scan → (scored ScanResult list, new cache entries, n_cache_hits).

    With *cache*:
      - Files absent from *changed* (or mtime-unchanged when *changed* is None)
        are loaded from cache without re-scanning.
      - Fresh / modified files are scanned and added to the returned entries.

    Returns results based on AST only (git enrichment applied later in Phase 0b).
    min_score pre-filters to avoid running git blame on obviously clean files.
    """
    py_files = [
        f for f in dir_path.rglob("*.py")
        if not any(ex in str(f) for ex in exclude)
    ][:max_files]

    if not py_files:
        print("[Phase0a] No Python files found.")
        return [], {}, 0

    # ── Split: cache-hit vs needs-scan ──────────────────────────────────────
    to_scan:     list[Path]      = []
    pre_results: list[ScanResult] = []   # loaded from cache
    new_entries: dict[str, dict]  = {}   # will be written back to cache

    if cache is not None:
        for f in py_files:
            rel   = _rel(f)
            entry = cache.get(rel)
            if entry is not None:
                # Use git-changed set when available; fall back to mtime comparison.
                if changed is not None:
                    hit = rel not in changed
                else:
                    hit = entry.get("mtime") == f.stat().st_mtime
                if hit:
                    sr = ScanResult(
                        file=f,
                        high_count=entry["high_count"],
                        medium_count=entry["medium_count"],
                        rules=entry["rules"],
                    )
                    scorer.compute_score(sr)
                    new_entries[rel] = entry          # carry forward unchanged
                    if sr.score >= min_score:
                        pre_results.append(sr)
                    continue
            to_scan.append(f)
    else:
        to_scan = list(py_files)

    n_hits   = len(py_files) - len(to_scan)
    soft     = [e for e in exclude if e not in _EXCLUDE_ALWAYS]
    excl_note = f"  excluding: {soft}" if soft else ""
    cache_note = (
        f"  ({n_hits} cached, {len(to_scan)} to scan)" if cache is not None else ""
    )
    print(f"[Phase0a] AST scan: {len(py_files)} file(s) in {dir_path}"
          f"{excl_note}{cache_note} …")

    # ── Scan only changed / uncached files ──────────────────────────────────
    t0 = time.perf_counter()
    if to_scan:
        raw = await asyncio.gather(*[_ast_analyze(f) for f in to_scan],
                                   return_exceptions=True)
    else:
        raw = []
    elapsed = time.perf_counter() - t0

    fresh_results: list[ScanResult] = []
    for file, r in zip(to_scan, raw):
        if isinstance(r, Exception):
            print(f"[Phase0a] WARN {file.name}: {r}")
            continue
        high, med, rules = parse_ast_findings(r)
        sr = ScanResult(file=file, high_count=high, medium_count=med, rules=rules)
        scorer.compute_score(sr)
        rel = _rel(file)
        new_entries[rel] = {
            "high_count":   high,
            "medium_count": med,
            "rules":        rules,
            "mtime":        file.stat().st_mtime,
        }
        if sr.score >= min_score:
            fresh_results.append(sr)

    elapsed_note = (
        f"  scanned {len(to_scan)}, cached {n_hits}" if cache is not None else ""
    )
    print(f"[Phase0a] Done in {elapsed:.2f}s{elapsed_note}")

    results = sorted(pre_results + fresh_results, key=lambda x: x.score, reverse=True)
    return results, new_entries, n_hits


# ---------------------------------------------------------------------------
# Phase 0b — git enrichment
# ---------------------------------------------------------------------------

_git_sem = asyncio.Semaphore(16)


async def _git_timestamps(
    file:      Path,
    repo_root: Path,
    since_days: int,
) -> list[int]:
    """Return list of commit Unix timestamps for file in the last N days."""
    try:
        async with _git_sem:
            proc = await asyncio.create_subprocess_exec(
                "git", "log", "--follow", "--format=%at",
                f"--since={since_days}.days",
                "--", str(file.relative_to(repo_root)),
                cwd=str(repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
        lines = stdout.decode().strip().splitlines()
        return [int(x) for x in lines if x.strip()]
    except Exception:
        return []


async def enrich_with_blame(
    results:    list[ScanResult],
    repo_root:  Path,
    scorer:     ScanScorer,
    since_days: int = 90,
) -> None:
    """Phase 0b: populate result.extra['blame'] and result.extra['churn'].

    blame  — recency score (0–10): 10 = committed today, 0 = >= since_days ago
    churn  — frequency score (0–10): normalized commit count in the window

    Mutates results in-place, then re-computes scores and re-sorts.
    """
    if not results:
        return

    t0 = time.perf_counter()
    print(f"[Phase0b] git enrichment: {len(results)} file(s) (window={since_days}d) …")

    ts_list = await asyncio.gather(
        *[_git_timestamps(sr.file, repo_root, since_days) for sr in results]
    )

    now = time.time()
    raw_churn = [len(ts) for ts in ts_list]
    max_churn = max(raw_churn, default=1) or 1

    for sr, ts, churn in zip(results, ts_list, raw_churn):
        sr.extra["churn"] = round(churn / max_churn * 10.0, 2)
        if ts:
            age_days = (now - max(ts)) / 86400
            sr.extra["blame"] = round(max(0.0, 1.0 - age_days / since_days) * 10.0, 2)
        else:
            sr.extra["blame"] = 0.0

    # Re-score with git signals and re-sort
    for sr in results:
        scorer.compute_score(sr)
    results.sort(key=lambda x: x.score, reverse=True)

    elapsed = time.perf_counter() - t0
    print(f"[Phase0b] Done in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    env_file = _REPO_ROOT / "keryx" / ".env"
    if env_file.exists():
        key = env_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def build_model(model_name: str, budget: float, scripted: bool = False) -> ModelInterface:
    """Resolve short name or raw model ID → ModelInterface."""
    if scripted:
        m = ScriptedModel()
        print(f"[Model] ScriptedModel (deterministic, requested={model_name!r})")
        return m
    api_key = _load_api_key()
    if api_key:
        from keryx.models.remote import create_anthropic_model
        resolved = MODEL_IDS.get(model_name, model_name)
        m = create_anthropic_model(api_key=api_key, model=resolved,
                                   max_budget_usd=budget)
        print(f"[Model] {resolved}  (budget cap ${budget})")
        return m
    m = ScriptedModel()
    print(f"[Model] ScriptedModel (no API key — fallback for {model_name!r})")
    return m


# ---------------------------------------------------------------------------
# ScriptedModel — deterministic fallback
# ---------------------------------------------------------------------------

class ScriptedModel(ModelInterface):
    """Scripted fallback: read_file → codeql_query → FINISH, per-prompt target."""

    model_name = "scripted-fallback"

    def __init__(self) -> None:
        self.main_calls = 0

    def generate(self, prompt: str, config: GenerationConfig | None = None,
                 *, grammar: str | None = None, max_tokens: int | None = None) -> str:
        if prompt.strip() == "ping":
            return "pong"
        if "vulnerability hypothesis" in prompt or "Does the evidence" in prompt:
            return "GENUINE CONCERN — AST finding appears valid."
        self.main_calls += 1
        target = self._target_from_prompt(prompt)
        step   = self._next_step(prompt, target)
        return json.dumps(step)

    @staticmethod
    def _target_from_prompt(prompt: str) -> str:
        m = re.search(r'Target file\s*:\s*(\S+)', prompt)
        return m.group(1) if m else "/tmp/unknown.py"

    @staticmethod
    def _next_step(prompt: str, target: str) -> dict:
        if ("] codeql_query" in prompt and "obs:" in prompt) or "[AST]" in prompt:
            return {"thought": "Analysis complete.",
                    "action": "FINISH", "action_input": {}, "confidence": 0.88}
        if "] read_file" in prompt and "obs:" in prompt:
            return {"thought": "Read target, running AST analysis.",
                    "action": "codeql_query",
                    "action_input": {"file_path": target}, "confidence": 0.82}
        return {"thought": "Starting: read target file.",
                "action": "read_file",
                "action_input": {"path": target}, "confidence": 0.75}

    async def generate_async(self, prompt, config=None,
                             *, grammar=None, max_tokens=None) -> str:
        return self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens)

    def generate_result(self, prompt, config=None) -> GenerationResult:
        return GenerationResult(text=self.generate(prompt, config),
                                tokens_input=10, tokens_output=10,
                                duration_ms=1.0, finish_reason="stop")

    def generate_stream(self, prompt, config=None) -> Iterator[str]:
        yield self.generate(prompt, config)

    def generate_with_tools(self, prompt, tools, config=None) -> str | ToolCall:
        return self.generate(prompt, config)

    def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def get_context_length(self) -> int:
        return 32_768

    def is_healthy(self) -> bool:
        return True

    def estimate_cost(self, i, o) -> CostEstimate:
        return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

    def get_usage_cost(self) -> CostEstimate:
        return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_context_length=32_768, supports_tool_calling=False,
            supports_grammar=True, supports_batching=False, supports_streaming=False,
            supports_speculative=False, supports_min_p=False, requires_gpu=False,
            is_local=True, is_quantized=False,
        )

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Phase 1 — per-file hunt
# ---------------------------------------------------------------------------

@dataclass
class HuntResult:
    file:        Path
    scan:        ScanResult
    steps:       int
    confirmed:   list[dict]
    elapsed_s:   float
    status:      str
    model_label: str
    escalated:   bool = False   # True → came from Phase 1c auto-escalation


async def hunt_file(
    scan:         ScanResult,
    model:        ModelInterface,
    mode:         str,
    max_steps:    int,
    model_label:  str,
    escalated:    bool       = False,
    context_hint: str | None = None,
) -> HuntResult:
    # Use the target's root: repo root when target is inside the repo,
    # otherwise the target file's parent directory (external hunt).
    try:
        scan.file.relative_to(_REPO_ROOT)
        _allowed = str(_REPO_ROOT)
    except ValueError:
        _allowed = str(scan.file.parent)
    toolbox = create_default_toolbox(allowed_root=_allowed)
    advisor = RuleBasedAdvisor(confidence_threshold=0.5)
    manager = create_advisor_manager(advisors=[(advisor, 10)])

    agent = KeryxAgent(
        executor_model=model,
        advisor_manager=manager,
        toolbox=toolbox,
        max_steps=max_steps,
        verification_mode=mode,
        max_clean_scans_before_exit=1,
    )

    t0     = time.perf_counter()
    result = await agent.run(target_path=str(scan.file), resume=False,
                             context_hint=context_hint)
    return HuntResult(
        file        = scan.file,
        scan        = scan,
        steps       = result.get("steps_taken", 0),
        confirmed   = result.get("confirmed_vulns", []),
        elapsed_s   = time.perf_counter() - t0,
        status      = result.get("status", "?"),
        model_label = model_label,
        escalated   = escalated,
    )


# ---------------------------------------------------------------------------
# Phase 2 — reporting
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _model_cost(model: ModelInterface) -> float:
    return model.get_usage_cost().get("total_cost_usd", 0.0)


def _model_calls(model: ModelInterface) -> int:
    return getattr(model, "_call_count", None) or getattr(model, "main_calls", 0)


def print_phase0(
    results:    list[ScanResult],
    top_n:      int,
    escalate_n: int,
    esc_label:  str,
    def_label:  str,
    git_enriched: bool,
    cache_hits:   int = 0,
    total_files:  int = 0,
) -> None:
    print("\n" + "=" * 70)
    print(f"PHASE 0 — AST scan results  (top {top_n} will be hunted)")
    if escalate_n:
        print(f"          ★ top {escalate_n} → {esc_label}   · rest → {def_label}")
    if git_enriched:
        print(f"          git-enriched scoring active  (blame + churn)")
    if cache_hits:
        pct = int(cache_hits / total_files * 100) if total_files else 0
        print(f"          cache: {cache_hits}/{total_files} files loaded from cache  ({pct}% hit rate)")
    print("=" * 70)

    if git_enriched:
        print(f"  {'score':>6}  {'H':>3}  {'M':>3}  {'blame':>5}  {'churn':>5}  file")
        print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*38}")
    else:
        print(f"  {'score':>6}  {'H':>3}  {'M':>3}  file")
        print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*40}")

    for i, r in enumerate(results):
        marker = "★" if i < escalate_n else ("→" if i < top_n else " ")
        if git_enriched:
            blame = r.extra.get("blame", 0.0)
            churn = r.extra.get("churn", 0.0)
            print(f"{marker} {r.score:6.1f}  {r.high_count:3d}  {r.medium_count:3d}"
                  f"  {blame:5.1f}  {churn:5.1f}  {_rel(r.file)}")
        else:
            print(f"{marker} {r.score:6.1f}  {r.high_count:3d}  {r.medium_count:3d}"
                  f"  {_rel(r.file)}")

    total_h = sum(r.high_count for r in results)
    total_m = sum(r.medium_count for r in results)
    print(f"\n  Total: {len(results)} file(s)  |  {total_h} HIGH  |  {total_m} MEDIUM")


def _print_hunt_rows(hunts: list[HuntResult]) -> None:
    for h in hunts:
        icon  = "✓" if h.confirmed else "·"
        label = f"{h.model_label}←{args_global.default_model}" if h.escalated else h.model_label
        print(f"\n  {icon}  {_rel(h.file)}  [{label}]")
        print(f"     steps={h.steps}  status={h.status}  elapsed={h.elapsed_s:.1f}s"
              f"  confirmed={len(h.confirmed)}")
        for v in h.confirmed:
            tag = v.get("confidence_tag", "?")
            obs = v.get("observation", "")[:100].replace("\n", " ")
            print(f"       [{tag}] {obs}")


def print_phase1(
    heavy_hunts:   list[HuntResult],
    default_hunts: list[HuntResult],
    esc_hunts:     list[HuntResult],
) -> None:
    all_hunts = heavy_hunts + default_hunts + esc_hunts
    all_vulns = [v for h in all_hunts for v in h.confirmed]
    print("\n" + "=" * 70)
    print(f"PHASE 1 — Hunt results  ({len(all_vulns)} confirmed across {len(all_hunts)} file(s))")
    print("=" * 70)

    if heavy_hunts:
        print(f"\n  [1a — score-based {args_global.escalate_model}]")
        _print_hunt_rows(heavy_hunts)

    if default_hunts:
        parallel_note = " (parallel)" if len(default_hunts) > 1 else ""
        print(f"\n  [1b — {args_global.default_model}{parallel_note}]")
        _print_hunt_rows(default_hunts)

    if esc_hunts:
        print(f"\n  [1c — auto-escalation: {args_global.default_model}→{args_global.escalate_model}]")
        _print_hunt_rows(esc_hunts)


def print_rule_breakdown(all_results: list[ScanResult]) -> None:
    """Aggregate per-rule file counts across all scanned results."""
    from collections import Counter
    rule_file_count: Counter[str] = Counter()
    for r in all_results:
        for rule in r.rules:
            rule_file_count[rule] += 1

    if not rule_file_count:
        return

    print("\n  RULE BREAKDOWN (AST findings across all scanned files):")
    print(f"  {'RULE':<35}  {'files':>5}  {'RECOMMENDED FIX'}")
    print(f"  {'-'*35}  {'-----':>5}  {'-'*50}")
    for rule, count in sorted(rule_file_count.items(), key=lambda x: -x[1]):
        fix = _RULE_FIX.get(rule, "—")
        print(f"  {rule:<35}  {count:5d}  {fix}")


def _run_fuzz_phase(
    all_hunts:       list[HuntResult],
    timeout:         int  = 10,
    llm_generate_fn       = None,
) -> None:
    """Phase 1.5 — attempt to reproduce each confirmed finding via PoC harness.

    When *llm_generate_fn* is provided (via ``--fuzz-llm``), it is passed to
    ``reproduce()`` as a fallback for rules that have no static template.
    """
    from keryx.fuzzing.poc import reproduce

    all_confirmed = [(h, v) for h in all_hunts for v in h.confirmed]
    if not all_confirmed:
        print("\n[--fuzz] No confirmed findings to reproduce.")
        return

    llm_note = "  (LLM fallback enabled)" if llm_generate_fn is not None else ""
    print(f"\n{'='*70}")
    print(f"PHASE 1.5 — PoC reproduction  ({len(all_confirmed)} confirmed finding(s)){llm_note}")
    print(f"{'='*70}")

    reproduced_total = 0
    for h, vuln in all_confirmed:
        rule    = vuln.get("rule", "UNKNOWN")
        finding = {"rule": rule, "observation": vuln.get("observation", "")}
        poc     = reproduce(finding, timeout=timeout, llm_generate_fn=llm_generate_fn)

        icon = "✓" if poc.reproduced else "·"
        print(f"\n  {icon}  rule={poc.rule}  file={_rel(h.file)}")
        if poc.error:
            print(f"     skip: {poc.error}")
        elif poc.reproduced:
            print(f"     REPRODUCED in {poc.elapsed_s:.2f}s  (confidence_boost=+{poc.confidence_boost:.2f})")
            print(f"     stdout: {poc.stdout[:120]!r}")
            reproduced_total += 1
            # Attach fuzz_proof to the vuln dict in-place so downstream report can use it
            vuln["fuzz_proof"] = {
                "reproduced":       poc.reproduced,
                "elapsed_s":        poc.elapsed_s,
                "confidence_boost": poc.confidence_boost,
                "stdout":           poc.stdout[:500],
            }
        else:
            print(f"     not reproduced in {poc.elapsed_s:.2f}s")
            if poc.stderr:
                print(f"     stderr: {poc.stderr[:80]!r}")

    print(f"\n  Fuzz summary: {reproduced_total}/{len(all_confirmed)} finding(s) reproduced")


def write_output_sarif(
    path:      str,
    all_hunts: list[HuntResult],
) -> None:
    """Serialize confirmed findings to a SARIF 2.1.0 file for GitHub Security tab."""
    from keryx.core.sarif import to_sarif

    hunt_dicts = [
        {
            "file":      _rel(h.file),
            "confirmed": h.confirmed,
        }
        for h in all_hunts
    ]
    sarif = to_sarif(hunt_dicts, repo_root=str(_REPO_ROOT))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sarif, indent=2, default=str), encoding="utf-8")
    n = sum(len(h.confirmed) for h in all_hunts)
    print(f"\n[SARIF] {n} result(s) written to {path}")


def write_output_json(
    path:        str,
    all_results: list[ScanResult],
    all_hunts:   list[HuntResult],
    models:      dict[str, ModelInterface],
    budget:      float,
    total_s:     float,
) -> None:
    """Serialize full run data to a JSON file for downstream tooling."""
    all_vulns = [v for h in all_hunts for v in h.confirmed]
    data = {
        "summary": {
            "files_scanned":       len(all_results),
            "files_hunted":        len(all_hunts),
            "confirmed_vulns":     len(all_vulns),
            "dynamically_verified": sum(1 for v in all_vulns if v.get("verified")),
            "total_cost_usd":      round(sum(_model_cost(m) for m in models.values()), 6),
            "total_elapsed_s":     round(total_s, 2),
        },
        "models": {
            label: {"cost_usd": round(_model_cost(m), 6), "calls": _model_calls(m)}
            for label, m in models.items()
        },
        "scan_results": [
            {
                "file":         _rel(r.file),
                "score":        r.score,
                "high_count":   r.high_count,
                "medium_count": r.medium_count,
                "rules":        r.rules,
                "blame":        r.extra.get("blame", 0.0),
                "churn":        r.extra.get("churn", 0.0),
            }
            for r in all_results
        ],
        "hunt_results": [
            {
                "file":            _rel(h.file),
                "model":           h.model_label,
                "escalated":       h.escalated,
                "steps":           h.steps,
                "status":          h.status,
                "elapsed_s":       round(h.elapsed_s, 2),
                "confirmed_count": len(h.confirmed),
                "confirmed":       h.confirmed,
            }
            for h in all_hunts
        ],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\n[JSON] Report written to {path}")


def write_output_html(
    path:        str,
    all_results: list[ScanResult],
    all_hunts:   list[HuntResult],
    models:      dict[str, ModelInterface],
    budget:      float,
    total_s:     float,
) -> None:
    """Write a self-contained HTML hunt report to *path*."""
    import html as _html
    from datetime import datetime, timezone

    all_vulns = [v for h in all_hunts for v in h.confirmed]
    total_cost = sum(_model_cost(m) for m in models.values())
    confirmed_count = len(all_vulns)
    verified_count  = sum(1 for v in all_vulns if v.get("verified"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── severity badge helper ────────────────────────────────────────────────
    _SEV_COLOURS = {"HIGH": "#c0392b", "MEDIUM": "#d68910", "LOW": "#1a8a1a"}

    def sev_badge(sev: str) -> str:
        colour = _SEV_COLOURS.get(str(sev).upper(), "#555")
        label  = _html.escape(str(sev).upper())
        return (
            f'<span style="background:{colour};color:#fff;padding:2px 7px;'
            f'border-radius:4px;font-size:0.78em;font-weight:bold;">'
            f'{label}</span>'
        )

    def esc(v: object) -> str:
        return _html.escape(str(v)) if v is not None else ""

    # ── confirmed findings rows ──────────────────────────────────────────────
    vuln_rows: list[str] = []
    for h in all_hunts:
        for v in h.confirmed:
            rule     = esc(v.get("rule", "?"))
            sev      = str(v.get("severity", "MEDIUM")).upper()
            file_str = esc(_rel(h.file))
            line     = esc(v.get("line", ""))
            msg      = esc(v.get("message") or v.get("observation", ""))[:200]
            src_type = esc(v.get("source_type", ""))
            verified = "Yes" if v.get("verified") else "No"
            badge    = sev_badge(sev)
            vuln_rows.append(
                f"<tr>"
                f"<td>{badge}</td>"
                f"<td><code>{rule}</code></td>"
                f"<td><code>{file_str}:{line}</code></td>"
                f"<td>{msg}</td>"
                f"<td>{src_type}</td>"
                f"<td>{verified}</td>"
                f"</tr>"
            )

    # ── scanned files rows ───────────────────────────────────────────────────
    file_rows: list[str] = []
    for r in sorted(all_results, key=lambda x: x.score, reverse=True)[:50]:
        file_rows.append(
            f"<tr>"
            f"<td><code>{esc(_rel(r.file))}</code></td>"
            f"<td>{r.score:.1f}</td>"
            f"<td>{r.high_count}</td>"
            f"<td>{r.medium_count}</td>"
            f"<td>{esc(', '.join(r.rules) if r.rules else '—')}</td>"
            f"</tr>"
        )

    # ── model cost table rows ────────────────────────────────────────────────
    model_rows: list[str] = []
    for label, m in models.items():
        model_rows.append(
            f"<tr>"
            f"<td>{esc(label)}</td>"
            f"<td>${_model_cost(m):.6f}</td>"
            f"<td>{_model_calls(m)}</td>"
            f"</tr>"
        )

    # ── assemble page ────────────────────────────────────────────────────────
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keryx Hunt Report</title>
<style>
  body{{font-family:system-ui,sans-serif;margin:0;background:#f5f6fa;color:#222}}
  header{{background:#1a1a2e;color:#fff;padding:1.2rem 2rem}}
  header h1{{margin:0;font-size:1.6rem;letter-spacing:.5px}}
  header p{{margin:.3rem 0 0;opacity:.7;font-size:.9rem}}
  .summary{{display:flex;gap:1.2rem;flex-wrap:wrap;padding:1.2rem 2rem}}
  .card{{background:#fff;border-radius:8px;padding:1rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,.08);min-width:130px}}
  .card .val{{font-size:2rem;font-weight:700;margin:0}}
  .card .lbl{{font-size:.78rem;color:#666;margin-top:.2rem}}
  section{{padding:.8rem 2rem 1.4rem}}
  h2{{font-size:1.15rem;border-bottom:2px solid #e0e0e0;padding-bottom:.4rem;margin-bottom:.8rem}}
  table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;
         overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.07);font-size:.88rem}}
  th{{background:#eef;text-align:left;padding:.55rem .8rem;white-space:nowrap}}
  td{{padding:.5rem .8rem;border-top:1px solid #eee;vertical-align:top;word-break:break-word}}
  tr:hover td{{background:#fafafa}}
  code{{font-size:.82em;background:#f0f0f0;padding:1px 4px;border-radius:3px}}
  .empty{{color:#888;font-style:italic;padding:.8rem 0}}
  footer{{padding:1rem 2rem;font-size:.78rem;color:#999;border-top:1px solid #e0e0e0}}
</style>
</head>
<body>
<header>
  <h1>Keryx Security Hunt Report</h1>
  <p>Generated {ts} &nbsp;|&nbsp; Budget ${budget:.2f} &nbsp;|&nbsp;
     Elapsed {total_s:.1f}s &nbsp;|&nbsp; Cost ${total_cost:.4f}</p>
</header>

<div class="summary">
  <div class="card"><p class="val">{len(all_results)}</p><p class="lbl">Files Scanned</p></div>
  <div class="card"><p class="val">{len(all_hunts)}</p><p class="lbl">Files Hunted</p></div>
  <div class="card"><p class="val">{confirmed_count}</p><p class="lbl">Confirmed Vulns</p></div>
  <div class="card"><p class="val">{verified_count}</p><p class="lbl">Dynamically Verified</p></div>
</div>

<section>
  <h2>Confirmed Vulnerabilities</h2>
  {'<p class="empty">No confirmed vulnerabilities.</p>' if not vuln_rows else f"""
  <table>
    <thead><tr>
      <th>Severity</th><th>Rule</th><th>Location</th>
      <th>Message</th><th>Source type</th><th>Verified</th>
    </tr></thead>
    <tbody>{''.join(vuln_rows)}</tbody>
  </table>"""}
</section>

<section>
  <h2>Top Scored Files (Phase 0)</h2>
  <table>
    <thead><tr>
      <th>File</th><th>Score</th><th>HIGH</th><th>MED</th><th>Rules</th>
    </tr></thead>
    <tbody>{''.join(file_rows) if file_rows else '<tr><td colspan="5" class="empty">No files.</td></tr>'}</tbody>
  </table>
</section>

<section>
  <h2>Model Cost Breakdown</h2>
  <table>
    <thead><tr><th>Model</th><th>Cost</th><th>Calls</th></tr></thead>
    <tbody>{''.join(model_rows) if model_rows else '<tr><td colspan="3" class="empty">No models.</td></tr>'}</tbody>
  </table>
</section>

<footer>Keryx Hunter &nbsp;|&nbsp; Rule coverage: R1-R13 &nbsp;|&nbsp;
  <a href="https://github.com/anthropics/claude-code">claude-code</a></footer>
</body>
</html>
"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"\n[HTML] Report written to {path}")


def print_summary(
    hunts:   list[HuntResult],
    models:  dict[str, ModelInterface],
    budget:  float,
    total_s: float,
) -> None:
    all_vulns  = [v for h in hunts for v in h.confirmed]
    iv_count   = sum(1 for v in all_vulns
                     if v.get("verified") and "injection_verifier" in v.get("confidence_tag", ""))
    fuzz_count = sum(1 for v in all_vulns
                     if v.get("verified") and "fuzz_poc" in v.get("confidence_tag", ""))
    verified   = iv_count + fuzz_count

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  files_hunted         : {len(hunts)}")
    print(f"  confirmed_vulns      : {len(all_vulns)}")
    dv_parts = []
    if iv_count:   dv_parts.append(f"{iv_count} via injection_verifier")
    if fuzz_count: dv_parts.append(f"{fuzz_count} via fuzz_poc")
    dv_label = "  (" + ", ".join(dv_parts) + ")" if dv_parts else "  (AST+injection_verifier or AST+fuzz_poc)"
    print(f"  dynamically_verified : {verified}{dv_label}")
    escalated_count = sum(1 for h in hunts if h.escalated)
    if escalated_count:
        print(f"  auto_escalated       : {escalated_count}  (haiku→{args_global.escalate_model})")
    print()

    total_cost = 0.0
    total_calls = 0
    for label, model in models.items():
        # Phase 1a + 1c both use escalate_model — separate the counts
        phase1a = [h for h in hunts if h.model_label == label and not h.escalated]
        phase1c = [h for h in hunts if h.model_label == label and h.escalated]
        all_m   = phase1a + phase1c
        confirmed   = sum(len(h.confirmed) for h in all_m)
        cost        = _model_cost(model)
        calls       = _model_calls(model)
        total_cost  += cost
        total_calls += calls
        esc_note = f"  (incl. {len(phase1c)} escalated)" if phase1c else ""
        print(f"  [{label:>8}]  files={len(all_m):2d}  confirmed={confirmed}"
              f"  calls={calls:3d}  cost=${cost:.5f}{esc_note}")

    budget_pct = (total_cost / budget * 100) if budget else 0.0
    print()
    print(f"  total_calls    : {total_calls}")
    print(f"  total_cost_usd : ${total_cost:.5f}  ({budget_pct:.1f}% of ${budget:.2f} budget)")
    print(f"  total_elapsed_s: {total_s:.1f}")

    # Manual review recommendations: HIGH findings, no confirmation after all passes
    manual = [
        h for h in hunts
        if h.scan.high_count > 0 and len(h.confirmed) == 0
    ]
    if manual:
        print(f"\n  MANUAL REVIEW RECOMMENDED ({len(manual)} file(s)):")
        for h in manual:
            tried = h.model_label
            if h.escalated:
                tried = f"{args_global.default_model}→{h.model_label}"
            print(f"    · {_rel(h.file)}  "
                  f"(AST: HIGH={h.scan.high_count}, tried: {tried})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    global args_global
    args_global = args

    scan_dir   = _REPO_ROOT / args.dir
    escalate_n = min(args.escalate_n, args.top_n)

    scorer = ScanScorer(
        blame_weight=args.blame_weight,
        churn_weight=args.churn_weight,
    )

    # ── Phase 0a: AST scan (with incremental cache) ──────────────────────
    soft_excludes  = () if args.no_default_excludes else _EXCLUDE_DEFAULT
    extra_excludes = tuple(args.exclude) if args.exclude else ()
    exclude = _EXCLUDE_ALWAYS + soft_excludes + extra_excludes

    cache_path = _REPO_ROOT / args.cache_file
    cache:   dict[str, dict] | None = None
    changed: set[str]        | None = None
    last_commit: str | None  = None

    if not args.no_cache:
        entries, last_commit = load_cache(cache_path)
        if entries:
            cache = entries
            if last_commit:
                changed = _git_changed_since(last_commit)
                n_changed = len([c for c in changed if c.endswith(".py")])
                print(f"[Cache] Loaded {len(entries)} entries  "
                      f"(commit {last_commit[:8]}, {n_changed} .py file(s) changed)")
            else:
                print(f"[Cache] Loaded {len(entries)} entries  (mtime-based invalidation)")

    all_results, new_entries, n_cache_hits = await scan_directory(
        scan_dir, scorer, exclude,
        max_files=args.max_files,
        min_score=args.min_score,
        cache=cache,
        changed=changed,
    )

    # Persist updated cache (always, so next run is warm)
    if not args.no_cache:
        save_cache(cache_path, new_entries, _git_head())

    if not all_results:
        print("Phase 0 found nothing to hunt.")
        return

    # ── Phase 0b: git enrichment (optional) ──────────────────────────────
    git_enriched = False
    if not args.no_blame:
        await enrich_with_blame(all_results, _REPO_ROOT, scorer,
                                since_days=args.blame_days)
        git_enriched = True

    targets = all_results[:args.top_n]
    print_phase0(
        all_results,
        top_n=args.top_n,
        escalate_n=escalate_n,
        esc_label=args.escalate_model,
        def_label=args.default_model,
        git_enriched=git_enriched,
        cache_hits=n_cache_hits,
        total_files=len(new_entries),
    )

    # ── Phase 1 — setup ───────────────────────────────────────────────────
    heavy_targets   = targets[:escalate_n]
    default_targets = targets[escalate_n:]
    can_escalate    = args.default_model != args.escalate_model

    # --always-escalate-high: promote every HIGH file to the escalate tier,
    # even if it sits below --top-n in the ranking.
    if args.always_escalate_high:
        heavy_paths = {r.file for r in heavy_targets}
        extra_high  = [r for r in all_results if r.high_count > 0 and r.file not in heavy_paths]
        if extra_high:
            print(f"\n[--always-escalate-high] Promoting {len(extra_high)} HIGH file(s) → "
                  f"{args.escalate_model}")
            for r in extra_high:
                print(f"  ↑ {_rel(r.file)}  score={r.score}  HIGH={r.high_count}")
            heavy_targets = heavy_targets + extra_high
            heavy_paths_updated = {r.file for r in heavy_targets}
            default_targets = [r for r in default_targets
                               if r.file not in heavy_paths_updated]

    print(f"\n{'='*70}")
    print(f"PHASE 1 — Hunting {len(targets)} file(s)  (mode={args.mode})")
    if heavy_targets:
        print(f"          ★ {len(heavy_targets)} file(s) → {args.escalate_model} ({_STEPS_ESCALATE} steps)")
    if default_targets:
        parallel_note = " parallel" if len(default_targets) > 1 else ""
        print(f"          → {len(default_targets)} file(s) → {args.default_model}"
              f" ({_STEPS_DEFAULT} steps,{parallel_note})")
    if can_escalate:
        print(f"          ↑ inconclusive HIGH → auto-escalate to {args.escalate_model}")
    print("=" * 70)

    models: dict[str, ModelInterface] = {}
    if default_targets:
        models[args.default_model] = build_model(args.default_model, args.budget,
                                                  scripted=args.scripted)
    if heavy_targets:
        models[args.escalate_model] = build_model(args.escalate_model, args.budget,
                                                   scripted=args.scripted)

    t0 = time.perf_counter()

    # ── Phase 1a: sequential Opus for score-based heavy targets ──────────
    heavy_hunts: list[HuntResult] = []
    for i, scan in enumerate(heavy_targets, 1):
        print(f"\n[1a {i}/{len(heavy_targets)}] ★ {_rel(scan.file)}"
              f"  score={scan.score}  HIGH={scan.high_count}  [{args.escalate_model}]")
        h = await hunt_file(scan, models[args.escalate_model],
                            mode=args.mode, max_steps=_STEPS_ESCALATE,
                            model_label=args.escalate_model)
        heavy_hunts.append(h)

    # ── Phase 1b: parallel Haiku for remaining targets ────────────────────
    default_hunts: list[HuntResult] = []
    if default_targets:
        print(f"\n[Phase 1b] Launching {len(default_targets)} {args.default_model} task(s) in parallel …")
        for i, scan in enumerate(default_targets, 1):
            print(f"  → [{i}] {_rel(scan.file)}  score={scan.score}  HIGH={scan.high_count}")
        default_hunts = list(await asyncio.gather(*[
            hunt_file(scan, models[args.default_model],
                      mode=args.mode, max_steps=_STEPS_DEFAULT,
                      model_label=args.default_model)
            for scan in default_targets
        ]))

    # ── Phase 1c: auto-escalate inconclusive results ─────────────────────
    esc_hunts: list[HuntResult] = []
    if can_escalate and args.escalate_inconclusive:
        inconclusive = [
            h for h in default_hunts
            if h.scan.high_count > 0 and len(h.confirmed) == 0
        ]
        if inconclusive:
            # ── Budget guard: estimate cost of escalation ─────────────────
            spent     = sum(_model_cost(m) for m in models.values())
            remaining = args.budget - spent
            # Rough per-file estimate for a deep Opus pass (~$0.04 conservative)
            _OPUS_COST_PER_FILE = 0.04
            affordable = max(0, int(remaining / _OPUS_COST_PER_FILE))
            skipped    = inconclusive[affordable:]
            inconclusive = inconclusive[:affordable]

            if skipped:
                print(f"\n[Phase 1c] Budget guard: ${remaining:.3f} remaining — "
                      f"can afford {affordable} escalation(s), "
                      f"{len(skipped)} file(s) moved to MANUAL REVIEW")
                for h in skipped:
                    print(f"  ⊘ {_rel(h.file)}  (budget insufficient)")

            if inconclusive:
                # Lazy-build escalate_model if Phase 1a didn't need it
                if args.escalate_model not in models:
                    models[args.escalate_model] = build_model(
                        args.escalate_model, args.budget, scripted=args.scripted
                    )
                print(f"\n[Phase 1c] Auto-escalating {len(inconclusive)} file(s) to"
                      f" {args.escalate_model}  (HIGH unconfirmed by {args.default_model})")
                for i, h in enumerate(inconclusive, 1):
                    print(f"  ↑ [{i}] {_rel(h.file)}  HIGH={h.scan.high_count}"
                          f"  {args.default_model}_steps={h.steps}"
                          f"  {args.default_model}_status={h.status}")
                for h in inconclusive:
                    # Build a one-line hint so Opus knows why it was escalated
                    rules_str = ", ".join(h.scan.rules) if h.scan.rules else "unknown"
                    hint = (
                        f"{args.default_model} found {h.scan.high_count} HIGH AST finding(s) "
                        f"({rules_str}) but could not confirm after {h.steps} step(s). "
                        f"Re-examine with deeper reasoning — focus on the HIGH rules listed."
                    )
                    esc = await hunt_file(
                        h.scan, models[args.escalate_model],
                        mode=args.mode, max_steps=_STEPS_ESCALATE,
                        model_label=args.escalate_model,
                        escalated=True,
                        context_hint=hint,
                    )
                    esc_hunts.append(esc)

    total_s = time.perf_counter() - t0

    # ── Phase 1.5 (optional): fuzz confirmed findings ─────────────────────
    all_hunts = heavy_hunts + default_hunts + esc_hunts
    if getattr(args, "fuzz", False):
        llm_fn = None
        if getattr(args, "fuzz_llm", False):
            from keryx.fuzzing.llm_harness import make_llm_generate_fn
            fuzz_model_name = getattr(args, "fuzz_llm_model", args.default_model)
            fuzz_model = build_model(fuzz_model_name, args.budget, scripted=args.scripted)
            llm_fn = make_llm_generate_fn(fuzz_model)
            print(f"\n[--fuzz-llm] LLM harness fallback enabled  (model={fuzz_model_name})")
        _run_fuzz_phase(all_hunts, timeout=getattr(args, "fuzz_timeout", 10),
                        llm_generate_fn=llm_fn)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print_phase1(heavy_hunts, default_hunts, esc_hunts)
    print_summary(all_hunts, models=models, budget=args.budget, total_s=total_s)
    print_rule_breakdown(all_results)

    if args.output_json:
        write_output_json(args.output_json, all_results, all_hunts, models,
                          args.budget, total_s)

    if args.output_sarif:
        write_output_sarif(args.output_sarif, all_hunts)

    if getattr(args, "output_html", None):
        write_output_html(args.output_html, all_results, all_hunts, models,
                          args.budget, total_s)

    if getattr(args, "learn", False):
        from keryx.core.rule_learner import RuleLearner, print_suggestions
        learner = RuleLearner([r.file for r in all_results])
        print_suggestions(learner.analyse())

    if args.fail_if_confirmed:
        all_confirmed = [v for h in all_hunts for v in h.confirmed]
        if all_confirmed:
            print(f"\n[--fail-if-confirmed] {len(all_confirmed)} confirmed vulnerability(ies)"
                  f" — exit 1")
            sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-file self-hunt")
    p.add_argument("--dir",            default="keryx",   help="Directory to scan")
    p.add_argument("--top-n",          type=int,  default=5,    dest="top_n")
    p.add_argument("--default-model",  default="haiku",   dest="default_model")
    p.add_argument("--escalate-model", default="opus",    dest="escalate_model")
    p.add_argument("--escalate-n",     type=int,  default=0,    dest="escalate_n")
    p.add_argument("--no-escalate-inconclusive", action="store_false",
                   dest="escalate_inconclusive",
                   help="Disable Phase 1c: do not re-escalate inconclusive HIGH files")
    p.set_defaults(escalate_inconclusive=True)
    p.add_argument("--mode",           default="strict",
                   choices=["strict", "flexible", "none"])
    p.add_argument("--budget",         type=float, default=0.50)
    p.add_argument("--min-score",      type=float, default=1.0,  dest="min_score")
    p.add_argument("--max-files",      type=int,   default=300,  dest="max_files")
    p.add_argument("--scripted",       action="store_true")
    # Exclusions (always-excluded: __pycache__, .git)
    p.add_argument("--exclude",        action="append", default=[], metavar="PATTERN",
                   help="Additional path patterns to exclude (repeatable)")
    p.add_argument("--no-default-excludes", action="store_true", dest="no_default_excludes",
                   help=f"Drop default soft-excludes {_EXCLUDE_DEFAULT} — scan everything")
    # Phase 0b — git enrichment
    p.add_argument("--blame-weight",   type=float, default=None, dest="blame_weight",
                   help="Weight for git recency score (default: 4.0)")
    p.add_argument("--churn-weight",   type=float, default=None, dest="churn_weight",
                   help="Weight for commit frequency score (default: 2.5)")
    p.add_argument("--blame-days",     type=int,   default=90,   dest="blame_days",
                   help="Git history window in days (default: 90)")
    p.add_argument("--no-blame",       action="store_true",      dest="no_blame",
                   help="Skip git enrichment (pure AST scoring)")
    # Incremental cache
    p.add_argument("--no-cache",      action="store_true",        dest="no_cache",
                   help="Ignore existing cache; force full Phase 0a rescan (still writes cache)")
    p.add_argument("--cache-file",    default=_CACHE_FILE,        dest="cache_file", metavar="PATH",
                   help=f"Cache file path (default: {_CACHE_FILE})")
    # CI / output
    p.add_argument("--always-escalate-high", action="store_true", dest="always_escalate_high",
                   help="Always route files with HIGH findings to escalate_model (ignores top-n cap)")
    p.add_argument("--fail-if-confirmed", action="store_true", dest="fail_if_confirmed",
                   help="Exit with code 1 if any vulnerability is confirmed (CI mode)")
    p.add_argument("--output-json",    default=None,             dest="output_json", metavar="PATH",
                   help="Write structured JSON report to PATH")
    p.add_argument("--output-sarif",   default=None,             dest="output_sarif", metavar="PATH",
                   help="Write SARIF 2.1.0 report to PATH (for GitHub Security tab)")
    p.add_argument("--output-html",    default=None,             dest="output_html",  metavar="PATH",
                   help="Write self-contained HTML report to PATH")
    # Phase 1.5 — PoC fuzzing
    p.add_argument("--fuzz",           action="store_true",      dest="fuzz",
                   help="After hunt, attempt to reproduce confirmed findings via sandboxed PoC harness")
    p.add_argument("--fuzz-timeout",   type=int, default=10,     dest="fuzz_timeout", metavar="SEC",
                   help="Per-harness sandbox timeout in seconds (default: 10)")
    p.add_argument("--fuzz-llm",       action="store_true",      dest="fuzz_llm",
                   help="Enable LLM-generated harness fallback for rules without a static template")
    p.add_argument("--fuzz-llm-model", default=None,             dest="fuzz_llm_model", metavar="MODEL",
                   help="Model for LLM harness generation (default: --default-model)")
    p.add_argument("--learn",          action="store_true",      dest="learn",
                   help="After the hunt, scan files for uncovered sink patterns (rule gap analysis)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
