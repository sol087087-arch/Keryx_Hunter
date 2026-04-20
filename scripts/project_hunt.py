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
import sys
import time

from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from keryx.core.hunt_models import (
    HuntResult, ReportContext, ScanScorer, routing_score,
)
from keryx.core.reporters import (
    print_phase0,
    print_phase1,
    print_rule_breakdown,
    print_summary,
    run_fuzz_phase,
    write_output_html,
    write_output_json,
    write_output_sarif,
)
from keryx.core.hunt_pipeline import (
    STEPS_DEFAULT, STEPS_ESCALATE,
    build_model,
    run_escalate_tier, run_default_tier, run_escalation_phase,
)
from keryx.core.scan_pipeline import (
    EXCLUDE_ALWAYS,
    load_cache, save_cache,
    git_head, git_changed_since,
    scan_directory, enrich_with_blame,
)

# Default soft-excludes — overridable via --exclude / --no-default-excludes
_EXCLUDE_DEFAULT = ("tests/", "sandbox/", "scripts/")

# Incremental cache
_CACHE_FILE = ".hunt_cache.json"

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    scan_dir   = _REPO_ROOT / args.dir
    escalate_n = min(args.escalate_n, args.top_n)

    scorer = ScanScorer(
        blame_weight=args.blame_weight,
        churn_weight=args.churn_weight,
    )

    # ── Phase 0a: AST scan (with incremental cache) ──────────────────────
    soft_excludes  = () if args.no_default_excludes else _EXCLUDE_DEFAULT
    extra_excludes = tuple(args.exclude) if args.exclude else ()
    exclude = EXCLUDE_ALWAYS + soft_excludes + extra_excludes

    cache_path = _REPO_ROOT / args.cache_file
    cache:   dict[str, dict] | None = None
    changed: set[str]        | None = None
    last_commit: str | None  = None

    if not args.no_cache:
        entries, last_commit = load_cache(cache_path)
        if entries:
            cache = entries
            if last_commit:
                changed = git_changed_since(last_commit, _REPO_ROOT)
                n_changed = len([c for c in changed if c.endswith(".py")])
                print(f"[Cache] Loaded {len(entries)} entries  "
                      f"(commit {last_commit[:8]}, {n_changed} .py file(s) changed)")
            else:
                print(f"[Cache] Loaded {len(entries)} entries  (mtime-based invalidation)")

    all_results, new_entries, n_cache_hits = await scan_directory(
        scan_dir, scorer, exclude,
        repo_root=_REPO_ROOT,
        max_files=args.max_files,
        min_score=args.min_score,
        cache=cache,
        changed=changed,
    )

    # Persist updated cache (always, so next run is warm)
    if not args.no_cache:
        save_cache(cache_path, new_entries, git_head(_REPO_ROOT))

    if not all_results:
        print("Phase 0 found nothing to hunt.")
        return

    # ── Phase 0b: git enrichment (optional) ──────────────────────────────
    git_enriched = False
    if not args.no_blame:
        await enrich_with_blame(all_results, _REPO_ROOT, scorer,
                                since_days=args.blame_days)
        git_enriched = True

    targets = sorted(all_results, key=routing_score, reverse=True)[:args.top_n]

    ctx = ReportContext(
        repo_root=_REPO_ROOT,
        escalate_model=args.escalate_model,
        default_model=args.default_model,
        budget_limit_usd=args.budget,
        top_n=args.top_n,
        escalate_n=escalate_n,
        scan_results=all_results,
        git_enriched=git_enriched,
        cache_hits=n_cache_hits,
        total_files=len(new_entries),
    )
    print_phase0(ctx)

    # ── Phase 1 — setup ───────────────────────────────────────────────────
    heavy_targets   = targets[:escalate_n]
    default_targets = targets[escalate_n:]
    can_escalate    = args.default_model != args.escalate_model

    # --always-escalate-high: promote every HIGH file to the escalate tier
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
        print(f"          ★ {len(heavy_targets)} file(s) → {args.escalate_model} ({STEPS_ESCALATE} steps)")
    if default_targets:
        parallel_note = " parallel" if len(default_targets) > 1 else ""
        print(f"          → {len(default_targets)} file(s) → {args.default_model}"
              f" ({STEPS_DEFAULT} steps,{parallel_note})")
    if can_escalate:
        print(f"          ↑ inconclusive HIGH → auto-escalate to {args.escalate_model}")
    print("=" * 70)

    if default_targets:
        ctx.models[args.default_model] = build_model(args.default_model, args.budget,
                                                      _REPO_ROOT, scripted=args.scripted)
    if heavy_targets:
        ctx.models[args.escalate_model] = build_model(args.escalate_model, args.budget,
                                                       _REPO_ROOT, scripted=args.scripted)

    t0 = time.perf_counter()

    # ── Phase 1a ──────────────────────────────────────────────────────────
    heavy_hunts = await run_escalate_tier(
        heavy_targets,
        ctx.models.get(args.escalate_model) or
        build_model(args.escalate_model, args.budget, _REPO_ROOT, scripted=args.scripted),
        args.escalate_model, args.mode, _REPO_ROOT,
    )

    # ── Phase 1b ──────────────────────────────────────────────────────────
    default_hunts = await run_default_tier(
        default_targets,
        ctx.models.get(args.default_model) or
        build_model(args.default_model, args.budget, _REPO_ROOT, scripted=args.scripted),
        args.default_model, args.mode, _REPO_ROOT,
    )

    # ── Phase 1c ──────────────────────────────────────────────────────────
    esc_hunts: list[HuntResult] = []
    if can_escalate and args.escalate_inconclusive:
        esc_hunts = await run_escalation_phase(
            default_hunts, ctx.models, args.budget, args.scripted,
            args.mode, args.escalate_model, args.default_model, _REPO_ROOT,
        )

    ctx.total_elapsed_s = time.perf_counter() - t0
    ctx.hunts = heavy_hunts + default_hunts + esc_hunts
    ctx.budget_used_usd = sum(
        m.get_usage_cost().get("total_cost_usd", 0.0)
        for m in ctx.models.values()
    )

    # ── Phase 1.5 (optional): fuzz confirmed findings ─────────────────────
    if getattr(args, "fuzz", False):
        llm_fn = None
        if getattr(args, "fuzz_llm", False):
            from keryx.fuzzing.llm_harness import make_llm_generate_fn
            fuzz_model_name = getattr(args, "fuzz_llm_model", args.default_model)
            fuzz_model = build_model(fuzz_model_name, args.budget, _REPO_ROOT, scripted=args.scripted)
            llm_fn = make_llm_generate_fn(fuzz_model)
            print(f"\n[--fuzz-llm] LLM harness fallback enabled  (model={fuzz_model_name})")
        run_fuzz_phase(ctx,
                       timeout=getattr(args, "fuzz_timeout", 10),
                       llm_generate_fn=llm_fn)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print_phase1(ctx)
    print_summary(ctx)
    print_rule_breakdown(ctx)

    if args.output_json:
        write_output_json(args.output_json, ctx)

    if args.output_sarif:
        write_output_sarif(args.output_sarif, ctx)

    if getattr(args, "output_html", None):
        write_output_html(args.output_html, ctx)

    if getattr(args, "learn", False):
        from keryx.core.rule_learner import RuleLearner, print_suggestions
        learner = RuleLearner([r.file for r in all_results])
        print_suggestions(learner.analyse())

    if args.fail_if_confirmed:
        all_confirmed = [v for h in ctx.hunts for v in h.confirmed]
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
    p.add_argument("--min-score",      type=float, default=12.0, dest="min_score")
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
