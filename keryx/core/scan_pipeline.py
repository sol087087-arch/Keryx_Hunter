"""scan_pipeline.py — Phase 0: AST scanning, caching, and git enrichment.

Phase 0a: parallel AST scan with incremental cache.
Phase 0b: git enrichment (blame recency + churn frequency).

All functions receive repo_root explicitly — no global state.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess as _sp
import time
from pathlib import Path

from keryx.core.hunt_models import ScanResult, ScanScorer
from keryx.tools.Toolbox import ToolResult
from keryx.tools.ast_analyzer import ASTAnalyzerTool

# ---------------------------------------------------------------------------
# Cache constants
# ---------------------------------------------------------------------------

CACHE_VERSION = 1

# Always-excluded directories — never useful to scan
EXCLUDE_ALWAYS = ("__pycache__", ".git")

# ---------------------------------------------------------------------------
# Incremental cache — Phase 0a persistence
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> tuple[dict[str, dict], str | None]:
    """Load Phase 0a cache.

    Returns (entries, last_commit). entries is keyed by repo-relative path;
    last_commit is the HEAD hash recorded when cache was last written.
    Returns ({}, None) on any miss/error/version mismatch.
    """
    if not cache_path.exists():
        return {}, None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            print(f"[Cache] Version mismatch — discarding cache (expected {CACHE_VERSION})")
            return {}, None
        return data.get("entries", {}), data.get("commit")
    except Exception as exc:
        print(f"[Cache] WARN: could not load {cache_path}: {exc}")
        return {}, None


def save_cache(
    cache_path: Path,
    entries:    dict[str, dict],
    commit:     str | None,
) -> None:
    """Persist Phase 0a scan results. Silently ignores write errors."""
    try:
        cache_path.write_text(
            json.dumps(
                {"version": CACHE_VERSION, "timestamp": time.time(),
                 "commit": commit, "entries": entries},
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[Cache] WARN: could not write {cache_path}: {exc}")


# ---------------------------------------------------------------------------
# Git helpers — cache invalidation
# ---------------------------------------------------------------------------

def git_head(repo_root: Path) -> str | None:
    """Return current HEAD commit hash, or None if not in a git repo."""
    try:
        r = _sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def git_changed_since(commit: str, repo_root: Path) -> set[str]:
    """Repo-relative paths changed since *commit* (committed + uncommitted).

    Combines:
      git diff --name-only <commit>   → files changed in commits after *commit*
      git status --porcelain          → staged / unstaged / untracked changes
    """
    changed: set[str] = set()
    try:
        r = _sp.run(
            ["git", "diff", "--name-only", commit],
            cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
        changed.update(line for line in r.stdout.splitlines() if line)

        r = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if len(line) > 3:
                fname = line[3:].split(" -> ")[-1].strip().strip('"')
                changed.add(fname)
    except Exception as exc:
        print(f"[Cache] WARN: git change detection failed ({exc}) — treating all files as changed")
    return changed


# ---------------------------------------------------------------------------
# Phase 0a — AST scanning
# ---------------------------------------------------------------------------

_ast_tool = ASTAnalyzerTool()
_scan_sem  = asyncio.Semaphore(8)


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


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
    repo_root:  Path,
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

    to_scan:     list[Path]       = []
    pre_results: list[ScanResult] = []
    new_entries: dict[str, dict]  = {}

    if cache is not None:
        for f in py_files:
            rel   = _rel(f, repo_root)
            entry = cache.get(rel)
            if entry is not None:
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
                        line_count=entry.get("line_count", 0),
                    )
                    scorer.compute_score(sr)
                    new_entries[rel] = entry
                    if sr.score >= min_score:
                        pre_results.append(sr)
                    continue
            to_scan.append(f)
    else:
        to_scan = list(py_files)

    n_hits    = len(py_files) - len(to_scan)
    soft      = [e for e in exclude if e not in EXCLUDE_ALWAYS]
    excl_note = f"  excluding: {soft}" if soft else ""
    cache_note = (
        f"  ({n_hits} cached, {len(to_scan)} to scan)" if cache is not None else ""
    )
    print(f"[Phase0a] AST scan: {len(py_files)} file(s) in {dir_path}"
          f"{excl_note}{cache_note} …")

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
        try:
            line_count = file.read_text(encoding="utf-8", errors="ignore").count("\n")
        except OSError:
            line_count = 0
        sr  = ScanResult(file=file, high_count=high, medium_count=med,
                         rules=rules, line_count=line_count)
        scorer.compute_score(sr)
        rel = _rel(file, repo_root)
        new_entries[rel] = {
            "high_count":   high,
            "medium_count": med,
            "rules":        rules,
            "line_count":   line_count,
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
    file:       Path,
    repo_root:  Path,
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

    for sr in results:
        scorer.compute_score(sr)
    results.sort(key=lambda x: x.score, reverse=True)

    elapsed = time.perf_counter() - t0
    print(f"[Phase0b] Done in {elapsed:.2f}s")
