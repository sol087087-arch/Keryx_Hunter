#!/usr/bin/env python3
"""
run_hunt.py — Launch a real KeryxHunter vulnerability hunt.

Reads ANTHROPIC_API_KEY from keryx/.env (one line, raw key).
Targets keryx/tools/git_blame.py by default; override via CLI arg.

Usage:
    python run_hunt.py
    python run_hunt.py keryx/core/agent.py
    python run_hunt.py keryx/advisors/manager.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate project root and load API key
# ---------------------------------------------------------------------------
ROOT   = Path(__file__).parent.resolve()
DOTENV = ROOT / "keryx" / ".env"

if not DOTENV.exists():
    sys.exit(f"[ERROR] Key file not found: {DOTENV}")

API_KEY = DOTENV.read_text().strip()
if not API_KEY:
    sys.exit("[ERROR] keryx/.env is empty")


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
TARGET_REL = sys.argv[1] if len(sys.argv) > 1 else "keryx/tools/git_blame.py"
TARGET     = str(ROOT / TARGET_REL)

if not Path(TARGET).exists():
    sys.exit(f"[ERROR] Target not found: {TARGET}")


# ---------------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------------
async def main() -> None:
    from keryx.models.remote       import create_anthropic_model
    from keryx.tools.Toolbox       import create_default_toolbox
    from keryx.advisors.manager    import AdvisorManager
    from keryx.core.agent          import KeryxAgent

    print("=" * 70)
    print(f"  KeryxHunter — real hunt")
    print(f"  Target  : {TARGET_REL}")
    print(f"  Executor: claude-opus-4-6")
    print(f"  Critique: claude-haiku-4-5-20251001")
    print(f"  Budget  : $2.00  |  Max steps: 25")
    print("=" * 70)
    print()

    # Main reasoning model — Opus 4.6
    model = create_anthropic_model(
        api_key=API_KEY,
        model="claude-opus-4-6",
    )
    # _PROVIDER_MODELS stores $/MTok; cost_per_1k_* fields need $/1k.
    # Opus 4.6: $15.00/MTok in, $75.00/MTok out → divide by 1000.
    model.cost_per_1k_input_tokens  = 15.00 / 1000   # = $0.015 / 1k tokens
    model.cost_per_1k_output_tokens = 75.00 / 1000   # = $0.075 / 1k tokens

    # Lightweight critique / health-check model — Haiku 4.5
    # Used for self-critique, model health-pings, and low-stakes calls.
    # Haiku 4.5: $0.80/MTok in, $4.00/MTok out
    critique_model = create_anthropic_model(
        api_key=API_KEY,
        model="claude-haiku-4-5-20251001",
    )
    critique_model.cost_per_1k_input_tokens  = 0.80 / 1000
    critique_model.cost_per_1k_output_tokens = 4.00 / 1000

    # ---------------------------------------------------------------------------
    # Pre-flight: verify the API key and model name are valid before handing
    # control to KeryxAgent (which silently swallows health-check errors).
    # ---------------------------------------------------------------------------
    print("[Pre-flight] Testing Anthropic API...")
    try:
        ping = await model.generate_async("Say OK", max_tokens=5)
        print(f"[Pre-flight] OK — model responded: {ping!r}")
    except Exception as exc:
        print(f"[Pre-flight] FAILED — {type(exc).__name__}: {exc}")
        sys.exit(1)
    print()

    toolbox = create_default_toolbox(allowed_root=str(ROOT))

    # Advisor cascade: RuleBasedAdvisor (free) → CloudAdvisor/Haiku (smart)
    # RuleBasedAdvisor fires on stagnation/low-conf and gives deterministic advice.
    # CloudAdvisor fires on deeper stagnation and reasons about the actual context.
    from keryx.advisors.base import RuleBasedAdvisor, CascadeAdvisor
    from keryx.advisors.cloud_advisor import create_cloud_advisor

    rule_advisor = RuleBasedAdvisor(
        confidence_threshold=0.4,
        max_parse_errors=3,
        no_progress_after_steps=6,   # fires sooner than default
    )
    cloud_advisor = create_cloud_advisor(
        model=critique_model,          # Haiku — cheap, fast
        max_calls_per_session=3,
        confidence_threshold=0.35,
        no_progress_after_steps=5,
        budget_limit_usd=0.30,        # hard cap on advisor spend
        enable_scrubbing=False,        # local run, no privacy concern
        enable_digest=False,           # no local digest model available
    )

    cascade = CascadeAdvisor(
        advisors=[rule_advisor, cloud_advisor],
        name="keryx-cascade",
        max_calls_per_session=5,
    )

    advisor = AdvisorManager()
    advisor.register(cascade, priority=50)

    agent = KeryxAgent(
        executor_model=model,
        critique_model=critique_model,
        advisor_manager=advisor,
        toolbox=toolbox,
        max_steps=25,
        confidence_threshold=0.60,
        budget_usd=2.00,
        generate_timeout=60.0,
    )

    result = await agent.run(TARGET, capability="deep_reasoning", resume=False)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  HUNT COMPLETE")
    print("=" * 70)
    print(f"  Status       : {result.get('status')}")
    print(f"  Steps taken  : {result.get('steps_taken')}")
    print(f"  Parse errors : {result.get('parse_errors', 0)}")

    budget = result.get("budget")
    if budget:
        print(f"  Cost         : ${budget.get('current_cost', 0):.4f} / ${budget.get('max_cost_usd', 0):.2f}")

    hyps = result.get("hypotheses", [])
    clusters = result.get("hypothesis_clusters", [])
    print(f"\n  Hypotheses   : {len(hyps)} ({len(clusters)} distinct cluster(s))")
    for i, cluster in enumerate(clusters, 1):
        # Print the longest (most descriptive) hypothesis as the cluster representative
        rep = max(cluster, key=len)
        print(f"    [{i}] {rep}")
        if len(cluster) > 1:
            print(f"         + {len(cluster)-1} similar variant(s)")

    vulns = result.get("confirmed_vulns", [])
    print(f"\n  Confirmed    : {len(vulns)}")
    for v in vulns:
        print(f"    → {v}")

    blacklisted = result.get("blacklisted", [])
    if blacklisted:
        print(f"\n  Blacklisted  : {len(blacklisted)} (false positives)")
        for b in blacklisted[:3]:
            print(f"    ✗ {b[:100]}")

    evidence = result.get("evidence", [])
    if evidence:
        print(f"\n  Evidence     : {len(evidence)} items")
        for e in evidence[:5]:
            print(f"    • {e.get('location', '?')} — {e.get('description', '')[:80]}")

    critiques = result.get("critiques", [])
    if critiques:
        print(f"\n  Critiques    : {len(critiques)}")
        for c in critiques[:3]:
            print(f"    » {c[:120]}")

    print()
    summary = result.get("summary", "")
    if summary:
        print("  Summary:")
        for line in summary.splitlines():
            print(f"    {line}")

    print()
    print(f"  Full result JSON → hunt_result.json")
    (ROOT / "hunt_result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )


if __name__ == "__main__":
    asyncio.run(main())
