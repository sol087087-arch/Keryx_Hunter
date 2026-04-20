"""reporters.py — all human-readable and machine-readable output for a hunt run.

Every public function receives its data as explicit parameters.
No global state is read — callers pass repo_root, model labels, etc.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from keryx.core.hunt_models import HuntResult, ReportContext, ScanResult

if TYPE_CHECKING:
    from keryx.models.interface import ModelInterface


# ---------------------------------------------------------------------------
# Rule → recommended fix (shown in Phase 2 breakdown)
# ---------------------------------------------------------------------------

RULE_FIX: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _model_cost(model: "ModelInterface") -> float:
    return model.get_usage_cost().get("total_cost_usd", 0.0)


def _model_calls(model: "ModelInterface") -> int:
    return getattr(model, "_call_count", None) or getattr(model, "main_calls", 0)


def _fuzz_level(v: dict) -> str:
    tag = v.get("confidence_tag", "")
    if "fuzz_poc:exploited" in tag: return "exploited"
    if "fuzz_poc:triggered" in tag: return "triggered"
    if "fuzz_poc:reachable" in tag: return "reachable"
    if "fuzz_poc"           in tag: return "exploited"  # legacy tag
    return ""


def _print_hunt_rows(
    hunts: list[HuntResult],
    ctx:   ReportContext,
) -> None:
    for h in hunts:
        icon  = "✓" if h.confirmed else "·"
        label = f"{h.model_label}←{ctx.default_model}" if h.escalated else h.model_label
        print(f"\n  {icon}  {_rel(h.file, ctx.repo_root)}  [{label}]")
        print(f"     steps={h.steps}  status={h.status}  elapsed={h.elapsed_s:.1f}s"
              f"  confirmed={len(h.confirmed)}")
        for v in h.confirmed:
            tag   = v.get("confidence_tag", "?")
            obs   = v.get("observation", "")[:100].replace("\n", " ")
            level = _fuzz_level(v)
            level_str = f"  verification={level}" if level else ""
            print(f"       [{tag}]{level_str}")
            print(f"       {obs}")


# ---------------------------------------------------------------------------
# Phase 0 report
# ---------------------------------------------------------------------------

def print_phase0(ctx: ReportContext) -> None:
    results = ctx.scan_results
    print("\n" + "=" * 70)
    print(f"PHASE 0 — AST scan results  (top {ctx.top_n} will be hunted)")
    if ctx.escalate_n and ctx.escalate_model:
        print(f"          ★ top {ctx.escalate_n} → {ctx.escalate_model}"
              f"   · rest → {ctx.default_model}")
    if ctx.git_enriched:
        print(f"          git-enriched scoring active  (blame + churn)")
    if ctx.cache_hits:
        pct = int(ctx.cache_hits / ctx.total_files * 100) if ctx.total_files else 0
        print(f"          cache: {ctx.cache_hits}/{ctx.total_files} files loaded from cache"
              f"  ({pct}% hit rate)")
    print("=" * 70)

    if ctx.git_enriched:
        print(f"  {'score':>6}  {'H':>3}  {'M':>3}  {'blame':>5}  {'churn':>5}  file")
        print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*38}")
    else:
        print(f"  {'score':>6}  {'H':>3}  {'M':>3}  file")
        print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*40}")

    for i, r in enumerate(results):
        marker = "★" if i < ctx.escalate_n else ("→" if i < ctx.top_n else " ")
        if ctx.git_enriched:
            blame = r.extra.get("blame", 0.0)
            churn = r.extra.get("churn", 0.0)
            print(f"{marker} {r.score:6.1f}  {r.high_count:3d}  {r.medium_count:3d}"
                  f"  {blame:5.1f}  {churn:5.1f}  {_rel(r.file, ctx.repo_root)}")
        else:
            print(f"{marker} {r.score:6.1f}  {r.high_count:3d}  {r.medium_count:3d}"
                  f"  {_rel(r.file, ctx.repo_root)}")

    total_h = sum(r.high_count for r in results)
    total_m = sum(r.medium_count for r in results)
    print(f"\n  Total: {len(results)} file(s)  |  {total_h} HIGH  |  {total_m} MEDIUM")


# ---------------------------------------------------------------------------
# Phase 1 report
# ---------------------------------------------------------------------------

def print_phase1(ctx: ReportContext) -> None:
    esc_label = ctx.escalate_model
    heavy   = [h for h in ctx.hunts
               if esc_label and h.model_label == esc_label and not h.escalated]
    default = [h for h in ctx.hunts
               if h.model_label == ctx.default_model and not h.escalated]
    esc     = [h for h in ctx.hunts if h.escalated]

    all_vulns = [v for h in ctx.hunts for v in h.confirmed]
    print("\n" + "=" * 70)
    print(f"PHASE 1 — Hunt results  ({len(all_vulns)} confirmed across {len(ctx.hunts)} file(s))")
    print("=" * 70)

    if heavy:
        print(f"\n  [1a — score-based {esc_label}]")
        _print_hunt_rows(heavy, ctx)

    if default:
        parallel_note = " (parallel)" if len(default) > 1 else ""
        print(f"\n  [1b — {ctx.default_model}{parallel_note}]")
        _print_hunt_rows(default, ctx)

    if esc:
        print(f"\n  [1c — auto-escalation: {ctx.default_model}→{esc_label}]")
        _print_hunt_rows(esc, ctx)


# ---------------------------------------------------------------------------
# Rule breakdown
# ---------------------------------------------------------------------------

def print_rule_breakdown(ctx: ReportContext) -> None:
    rule_file_count: Counter[str] = Counter()
    for r in ctx.scan_results:
        for rule in r.rules:
            rule_file_count[rule] += 1

    if not rule_file_count:
        return

    print("\n  RULE BREAKDOWN (AST findings across all scanned files):")
    print(f"  {'RULE':<35}  {'files':>5}  {'RECOMMENDED FIX'}")
    print(f"  {'-'*35}  {'-----':>5}  {'-'*50}")
    for rule, count in sorted(rule_file_count.items(), key=lambda x: -x[1]):
        fix = RULE_FIX.get(rule, "—")
        print(f"  {rule:<35}  {count:5d}  {fix}")


# ---------------------------------------------------------------------------
# Fuzz phase (Phase 1.5)
# ---------------------------------------------------------------------------

def run_fuzz_phase(
    ctx:            ReportContext,
    timeout:        int = 10,
    llm_generate_fn      = None,
) -> None:
    from keryx.fuzzing.poc import reproduce

    all_confirmed = [(h, v) for h in ctx.hunts for v in h.confirmed]
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
        print(f"\n  {icon}  rule={poc.rule}  file={_rel(h.file, ctx.repo_root)}")
        if poc.error:
            print(f"     skip: {poc.error}")
        elif poc.reproduced:
            print(f"     REPRODUCED in {poc.elapsed_s:.2f}s  (confidence_boost=+{poc.confidence_boost:.2f})")
            print(f"     stdout: {poc.stdout[:120]!r}")
            reproduced_total += 1
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(ctx: ReportContext) -> None:
    hunts      = ctx.hunts
    all_vulns  = [v for h in hunts for v in h.confirmed]
    iv_count   = sum(1 for v in all_vulns
                     if v.get("verified") and "injection_verifier" in v.get("confidence_tag", ""))
    fuzz_count = sum(1 for v in all_vulns
                     if v.get("verified") and "fuzz_poc" in v.get("confidence_tag", ""))
    verified   = iv_count + fuzz_count

    exploited_count = sum(1 for v in all_vulns if _fuzz_level(v) == "exploited")
    triggered_count = sum(1 for v in all_vulns if _fuzz_level(v) == "triggered")
    reachable_count = sum(1 for v in all_vulns if _fuzz_level(v) == "reachable")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  files_hunted         : {len(hunts)}")
    print(f"  confirmed_vulns      : {len(all_vulns)}")
    dv_parts = []
    if iv_count:        dv_parts.append(f"{iv_count} via injection_verifier")
    if exploited_count: dv_parts.append(f"{exploited_count} exploited (marker in output)")
    if triggered_count: dv_parts.append(f"{triggered_count} triggered (timeout/crash)")
    if reachable_count: dv_parts.append(f"{reachable_count} reachable (code path only)")
    dv_label = "  (" + ", ".join(dv_parts) + ")" if dv_parts else ""
    print(f"  dynamically_verified : {verified}{dv_label}")
    escalated_count = sum(1 for h in hunts if h.escalated)
    if escalated_count:
        print(f"  auto_escalated       : {escalated_count}"
              f"  ({ctx.default_model}→{ctx.escalate_model})")
    print()

    total_cost = 0.0
    total_calls = 0
    for label, model in ctx.models.items():
        tier_hunts  = [h for h in hunts if h.model_label == label]
        esc_hunts   = [h for h in tier_hunts if h.escalated]
        confirmed   = sum(len(h.confirmed) for h in tier_hunts)
        cost        = _model_cost(model)
        calls       = _model_calls(model)
        total_cost  += cost
        total_calls += calls
        esc_note = f"  (incl. {len(esc_hunts)} escalated)" if esc_hunts else ""
        print(f"  [{label:>8}]  files={len(tier_hunts):2d}  confirmed={confirmed}"
              f"  calls={calls:3d}  cost=${cost:.5f}{esc_note}")

    budget_pct = (total_cost / ctx.budget_limit_usd * 100) if ctx.budget_limit_usd else 0.0
    print()
    print(f"  total_calls    : {total_calls}")
    print(f"  total_cost_usd : ${total_cost:.5f}"
          f"  ({budget_pct:.1f}% of ${ctx.budget_limit_usd:.2f} budget)")
    print(f"  total_elapsed_s: {ctx.total_elapsed_s:.1f}")

    manual = [h for h in hunts if h.scan.high_count > 0 and len(h.confirmed) == 0]
    if manual:
        print(f"\n  MANUAL REVIEW RECOMMENDED ({len(manual)} file(s)):")
        for h in manual:
            tried = h.model_label
            if h.escalated:
                tried = f"{ctx.default_model}→{h.model_label}"
            print(f"    · {_rel(h.file, ctx.repo_root)}  "
                  f"(AST: HIGH={h.scan.high_count}, tried: {tried})")


# ---------------------------------------------------------------------------
# Machine-readable output
# ---------------------------------------------------------------------------

def write_output_sarif(path: str, ctx: ReportContext) -> None:
    from keryx.core.sarif import to_sarif

    hunt_dicts = [
        {"file": _rel(h.file, ctx.repo_root), "confirmed": h.confirmed}
        for h in ctx.hunts
    ]
    sarif = to_sarif(hunt_dicts, repo_root=str(ctx.repo_root))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sarif, indent=2, default=str), encoding="utf-8")
    n = sum(len(h.confirmed) for h in ctx.hunts)
    print(f"\n[SARIF] {n} result(s) written to {path}")


def write_output_json(path: str, ctx: ReportContext) -> None:
    all_vulns = [v for h in ctx.hunts for v in h.confirmed]
    data = {
        "summary": {
            "files_scanned":        len(ctx.scan_results),
            "files_hunted":         len(ctx.hunts),
            "confirmed_vulns":      len(all_vulns),
            "dynamically_verified": sum(1 for v in all_vulns if v.get("verified")),
            "total_cost_usd":       round(ctx.budget_used_usd, 6),
            "total_elapsed_s":      round(ctx.total_elapsed_s, 2),
        },
        "models": {
            label: {"cost_usd": round(_model_cost(m), 6), "calls": _model_calls(m)}
            for label, m in ctx.models.items()
        },
        "scan_results": [
            {
                "file":         _rel(r.file, ctx.repo_root),
                "score":        r.score,
                "high_count":   r.high_count,
                "medium_count": r.medium_count,
                "rules":        r.rules,
                "blame":        r.extra.get("blame", 0.0),
                "churn":        r.extra.get("churn", 0.0),
            }
            for r in ctx.scan_results
        ],
        "hunt_results": [
            {
                "file":            _rel(h.file, ctx.repo_root),
                "model":           h.model_label,
                "escalated":       h.escalated,
                "steps":           h.steps,
                "status":          h.status,
                "elapsed_s":       round(h.elapsed_s, 2),
                "confirmed_count": len(h.confirmed),
                "confirmed":       h.confirmed,
            }
            for h in ctx.hunts
        ],
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\n[JSON] Report written to {path}")


def write_output_html(path: str, ctx: ReportContext) -> None:
    import html as _html
    from datetime import datetime, timezone

    all_vulns = [v for h in ctx.hunts for v in h.confirmed]
    total_cost      = sum(_model_cost(m) for m in ctx.models.values())
    confirmed_count = len(all_vulns)
    verified_count  = sum(1 for v in all_vulns if v.get("verified"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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

    vuln_rows: list[str] = []
    for h in ctx.hunts:
        for v in h.confirmed:
            rule     = esc(v.get("rule", "?"))
            sev      = str(v.get("severity", "MEDIUM")).upper()
            file_str = esc(_rel(h.file, ctx.repo_root))
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

    file_rows: list[str] = []
    for r in sorted(ctx.scan_results, key=lambda x: x.score, reverse=True)[:50]:
        file_rows.append(
            f"<tr>"
            f"<td><code>{esc(_rel(r.file, ctx.repo_root))}</code></td>"
            f"<td>{r.score:.1f}</td>"
            f"<td>{r.high_count}</td>"
            f"<td>{r.medium_count}</td>"
            f"<td>{esc(', '.join(r.rules) if r.rules else '—')}</td>"
            f"</tr>"
        )

    model_rows: list[str] = []
    for label, m in ctx.models.items():
        model_rows.append(
            f"<tr>"
            f"<td>{esc(label)}</td>"
            f"<td>${_model_cost(m):.6f}</td>"
            f"<td>{_model_calls(m)}</td>"
            f"</tr>"
        )

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
  <p>Generated {ts} &nbsp;|&nbsp; Budget ${ctx.budget_limit_usd:.2f} &nbsp;|&nbsp;
     Elapsed {ctx.total_elapsed_s:.1f}s &nbsp;|&nbsp; Cost ${total_cost:.4f}</p>
</header>

<div class="summary">
  <div class="card"><p class="val">{len(ctx.scan_results)}</p><p class="lbl">Files Scanned</p></div>
  <div class="card"><p class="val">{len(ctx.hunts)}</p><p class="lbl">Files Hunted</p></div>
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
