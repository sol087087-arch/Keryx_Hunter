"""keryx — unified CLI entry point.

Usage:
    keryx hunt --target ./my-project
    keryx hunt --target ./my-project --model haiku --budget 0.30
    keryx hunt --target ./my-project --escalate-model opus --escalate-n 3 --top-n 10
    keryx hunt --target ./my-project --output-html report.html --output-json report.json
    keryx hunt --target ./my-project --scripted          # offline / CI mode
    keryx --version
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-command: hunt
# ---------------------------------------------------------------------------

def _add_hunt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--target", "-t", required=True, metavar="PATH",
        help="Directory to scan for vulnerabilities",
    )
    p.add_argument(
        "--model", default="haiku", dest="model",
        metavar="MODEL",
        help="Default model for most files (default: haiku)",
    )
    p.add_argument(
        "--escalate-model", default="opus", dest="escalate_model",
        metavar="MODEL",
        help="Model for top-priority files (default: opus)",
    )
    p.add_argument(
        "--escalate-n", type=int, default=0, dest="escalate_n",
        metavar="N",
        help="Top-N files to route to --escalate-model (default: 0)",
    )
    p.add_argument(
        "--top-n", type=int, default=5, dest="top_n",
        help="Total number of files to hunt (default: 5)",
    )
    p.add_argument(
        "--budget", type=float, default=0.50,
        help="Maximum spend in USD (default: 0.50)",
    )
    p.add_argument(
        "--mode", default="strict", choices=["strict", "flexible", "none"],
        help="Verification mode (default: strict)",
    )
    p.add_argument(
        "--min-score", type=float, default=12.0, dest="min_score",
        help="Minimum Phase 0 score to include a file (default: 12.0)",
    )
    p.add_argument(
        "--no-blame", action="store_true", dest="no_blame",
        help="Skip git enrichment (faster, pure AST scoring)",
    )
    p.add_argument(
        "--no-cache", action="store_true", dest="no_cache",
        help="Force full rescan, ignoring existing Phase 0 cache",
    )
    p.add_argument(
        "--output-json", default=None, dest="output_json", metavar="PATH",
        help="Write structured JSON report to PATH",
    )
    p.add_argument(
        "--output-html", default=None, dest="output_html", metavar="PATH",
        help="Write self-contained HTML report to PATH",
    )
    p.add_argument(
        "--output-sarif", default=None, dest="output_sarif", metavar="PATH",
        help="Write SARIF 2.1.0 report to PATH (GitHub Security tab)",
    )
    p.add_argument(
        "--scripted", action="store_true",
        help="Use ScriptedModel — for offline / CI testing without an API key",
    )
    p.add_argument(
        "--fail-if-confirmed", action="store_true", dest="fail_if_confirmed",
        help="Exit with code 1 if any vulnerability is confirmed (CI mode)",
    )


def _run_hunt(cli_args: argparse.Namespace) -> None:
    target = Path(cli_args.target).resolve()
    if not target.is_dir():
        print(f"[ERROR] --target {target} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Import project_hunt and override its _REPO_ROOT so paths resolve correctly
    # for external projects (not just the KeryxHunter repo itself).
    _repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_repo_root))
    import scripts.project_hunt as ph  # noqa: PLC0415

    original_repo_root = ph._REPO_ROOT
    ph._REPO_ROOT = target

    hunt_ns = argparse.Namespace(
        dir=".",                              # scan the target root
        top_n=cli_args.top_n,
        default_model=cli_args.model,
        escalate_model=cli_args.escalate_model,
        escalate_n=min(cli_args.escalate_n, cli_args.top_n),
        escalate_inconclusive=True,
        mode=cli_args.mode,
        budget=cli_args.budget,
        min_score=cli_args.min_score,
        max_files=300,
        scripted=cli_args.scripted,
        exclude=[],
        no_default_excludes=True,             # user targets their own project root
        blame_weight=None,
        churn_weight=None,
        blame_days=90,
        no_blame=cli_args.no_blame,
        no_cache=cli_args.no_cache,
        cache_file=".hunt_cache.json",
        always_escalate_high=False,
        fail_if_confirmed=cli_args.fail_if_confirmed,
        output_json=cli_args.output_json,
        output_sarif=cli_args.output_sarif,
        output_html=cli_args.output_html,
        fuzz=False,
        fuzz_timeout=10,
        fuzz_llm=False,
        fuzz_llm_model=None,
        learn=False,
    )

    try:
        asyncio.run(ph.main(hunt_ns))
    finally:
        ph._REPO_ROOT = original_repo_root


# ---------------------------------------------------------------------------
# Root parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="keryx",
        description="KeryxHunter — sovereign LLM-agent vulnerability hunter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  keryx hunt --target ./my-project
  keryx hunt --target ./my-project --model haiku --top-n 10 --budget 0.40
  keryx hunt --target ./my-project --escalate-model opus --escalate-n 3
  keryx hunt --target ./my-project --output-html report.html --no-blame
  keryx hunt --target ./my-project --scripted        # offline / CI
        """,
    )
    parser.add_argument(
        "--version", action="version", version="keryxhunter 0.9.0",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    hunt_p = sub.add_parser("hunt", help="Scan a directory for vulnerabilities")
    _add_hunt_args(hunt_p)

    args = parser.parse_args()

    if args.command == "hunt":
        _run_hunt(args)


if __name__ == "__main__":
    main()
