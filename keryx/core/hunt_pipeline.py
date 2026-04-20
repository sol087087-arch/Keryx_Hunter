"""hunt_pipeline.py — Phase 1: per-file LLM hunting and tier orchestration.

Phase 1a: sequential escalate-model for high-priority targets.
Phase 1b: parallel default-model for remaining targets.
Phase 1c: auto-escalation of inconclusive HIGH results.

All functions receive repo_root explicitly — no global state.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from keryx.advisors.base import RuleBasedAdvisor
from keryx.advisors.manager import create_advisor_manager
from keryx.core.agent import KeryxAgent
from keryx.core.hunt_models import HuntResult, ScanResult
from keryx.models.interface import ModelInterface
from keryx.models.scripted import ScriptedModel
from keryx.tools.Toolbox import create_default_toolbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_IDS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

STEPS_DEFAULT  = 10
STEPS_ESCALATE = 15

# Conservative per-file cost estimate for a deep escalate-model pass
_OPUS_COST_PER_FILE = 0.04


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _load_api_key(repo_root: Path) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    env_file = repo_root / "keryx" / ".env"
    if env_file.exists():
        key = env_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def build_model(
    model_name: str,
    budget:     float,
    repo_root:  Path,
    scripted:   bool = False,
) -> ModelInterface:
    """Resolve short name or raw model ID → ModelInterface."""
    if scripted:
        m = ScriptedModel()
        print(f"[Model] ScriptedModel (deterministic, requested={model_name!r})")
        return m
    api_key = _load_api_key(repo_root)
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
# Per-file hunt
# ---------------------------------------------------------------------------

async def hunt_file(
    scan:         ScanResult,
    model:        ModelInterface,
    mode:         str,
    max_steps:    int,
    model_label:  str,
    repo_root:    Path,
    escalated:    bool       = False,
    context_hint: str | None = None,
) -> HuntResult:
    try:
        scan.file.relative_to(repo_root)
        allowed = str(repo_root)
    except ValueError:
        allowed = str(scan.file.parent)
    toolbox = create_default_toolbox(allowed_root=allowed)
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
# Tier orchestration
# ---------------------------------------------------------------------------

def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


async def run_escalate_tier(
    heavy_targets:      list[ScanResult],
    model:              ModelInterface,
    escalate_model_name: str,
    mode:               str,
    repo_root:          Path,
) -> list[HuntResult]:
    """Phase 1a: sequential deep hunt for top-priority targets."""
    hunts: list[HuntResult] = []
    for i, scan in enumerate(heavy_targets, 1):
        print(f"\n[1a {i}/{len(heavy_targets)}] ★ {_rel(scan.file, repo_root)}"
              f"  score={scan.score}  HIGH={scan.high_count}  [{escalate_model_name}]")
        h = await hunt_file(scan, model, mode=mode, max_steps=STEPS_ESCALATE,
                            model_label=escalate_model_name, repo_root=repo_root)
        hunts.append(h)
    return hunts


async def run_default_tier(
    default_targets:    list[ScanResult],
    model:              ModelInterface,
    default_model_name: str,
    mode:               str,
    repo_root:          Path,
) -> list[HuntResult]:
    """Phase 1b: parallel hunt for remaining targets."""
    if not default_targets:
        return []
    print(f"\n[Phase 1b] Launching {len(default_targets)} {default_model_name} task(s) in parallel …")
    for i, scan in enumerate(default_targets, 1):
        print(f"  → [{i}] {_rel(scan.file, repo_root)}  score={scan.score}  HIGH={scan.high_count}")
    return list(await asyncio.gather(*[
        hunt_file(scan, model, mode=mode, max_steps=STEPS_DEFAULT,
                  model_label=default_model_name, repo_root=repo_root)
        for scan in default_targets
    ]))


async def run_escalation_phase(
    default_hunts:       list[HuntResult],
    models:              dict[str, ModelInterface],
    budget:              float,
    scripted:            bool,
    mode:                str,
    escalate_model_name: str,
    default_model_name:  str,
    repo_root:           Path,
) -> list[HuntResult]:
    """Phase 1c: auto-escalate inconclusive HIGH results to escalate_model."""
    inconclusive = [
        h for h in default_hunts
        if h.scan.high_count > 0 and len(h.confirmed) == 0
    ]
    if not inconclusive:
        return []

    spent     = sum(m.get_usage_cost().get("total_cost_usd", 0.0) for m in models.values())
    remaining = budget - spent
    affordable = max(0, int(remaining / _OPUS_COST_PER_FILE))
    skipped    = inconclusive[affordable:]
    inconclusive = inconclusive[:affordable]

    if skipped:
        print(f"\n[Phase 1c] Budget guard: ${remaining:.3f} remaining — "
              f"can afford {affordable} escalation(s), "
              f"{len(skipped)} file(s) moved to MANUAL REVIEW")
        for h in skipped:
            print(f"  ⊘ {_rel(h.file, repo_root)}  (budget insufficient)")

    if not inconclusive:
        return []

    if escalate_model_name not in models:
        models[escalate_model_name] = build_model(
            escalate_model_name, budget, repo_root, scripted=scripted
        )

    print(f"\n[Phase 1c] Auto-escalating {len(inconclusive)} file(s) to"
          f" {escalate_model_name}  (HIGH unconfirmed by {default_model_name})")
    for i, h in enumerate(inconclusive, 1):
        print(f"  ↑ [{i}] {_rel(h.file, repo_root)}  HIGH={h.scan.high_count}"
              f"  {default_model_name}_steps={h.steps}"
              f"  {default_model_name}_status={h.status}")

    esc_hunts: list[HuntResult] = []
    for h in inconclusive:
        rules_str = ", ".join(h.scan.rules) if h.scan.rules else "unknown"
        hint = (
            f"{default_model_name} found {h.scan.high_count} HIGH AST finding(s) "
            f"({rules_str}) but could not confirm after {h.steps} step(s). "
            f"Re-examine with deeper reasoning — focus on the HIGH rules listed."
        )
        esc = await hunt_file(
            h.scan, models[escalate_model_name],
            mode=mode, max_steps=STEPS_ESCALATE,
            model_label=escalate_model_name, repo_root=repo_root,
            escalated=True, context_hint=hint,
        )
        esc_hunts.append(esc)

    return esc_hunts
