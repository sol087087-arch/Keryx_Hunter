from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScanResult:
    file:         Path
    high_count:   int        = 0
    medium_count: int        = 0
    rules:        list[str]  = field(default_factory=list)
    score:        float      = 0.0
    line_count:   int        = 0
    # Git enrichment signals — populated by enrich_with_blame()
    extra:        dict[str, float] = field(default_factory=dict)


RULE_SCORE_BONUS: dict[str, float] = {
    # RCE / high-risk — maximum priority boost
    "SUBPROCESS_SHELL_TRUE":      28.0,
    "GIT_OPTION_INJECTION":       24.0,
    "GIT_FLAG_NAME_INJECTION":    22.0,
    "OS_SHELL_INJECTION":         21.0,
    "UNSAFE_EVAL_EXEC":           19.0,
    "UNSAFE_PICKLE":              17.0,
    "UNSANITIZED_SUBPROCESS_ARG": 16.0,
    "SUBPROCESS_EXEC_STARRED":    14.0,
    # Medium risk
    "SSRF":                       13.0,
    "TEMPLATE_INJECTION":         11.0,
    "UNSAFE_YAML_LOAD":           10.0,
    "HARDCODED_SECRET":            9.0,
    "OPEN_USER_PATH":              8.0,
    # Low priority / noise
    "REGEX_DOS":                   2.5,
    "UNSAFE_DESERIALIZATION":             1.5,
}


class ScanScorer:
    """Extensible scorer. Weights are configurable; extra dict for future signals."""

    BASE_WEIGHTS: dict[str, float] = {"HIGH": 10.0, "MEDIUM": 3.0}

    DEFAULT_EXTRA_WEIGHTS: dict[str, float] = {
        "blame": 4.0,   # git recency: how recently was the file touched?
        "churn": 2.5,   # git frequency: how often is it committed to?
    }

    _COMPLEXITY_THRESHOLD = 250  # lines above which complexity bonus kicks in
    _COMPLEXITY_RATE      = 1.5  # bonus points per 100 lines above threshold

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

    def _complexity_bonus(self, result: ScanResult) -> float:
        if result.line_count <= 300:
            return 0.0
        excess = result.line_count - 300
        return round(min(excess / 120 * 1.2, 5.0), 2)

    def compute_score(self, result: ScanResult) -> float:
        base       = (result.high_count   * self.BASE_WEIGHTS["HIGH"]
                      + result.medium_count * self.BASE_WEIGHTS["MEDIUM"])
        rule_bonus = sum(RULE_SCORE_BONUS.get(r, 0.0) for r in set(result.rules))
        complexity = self._complexity_bonus(result)
        for key, weight in self.extra_weights.items():
            if key in result.extra:
                base += result.extra[key] * weight
        result.score = round(base + rule_bonus + complexity, 2)
        return result.score


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


# ---------------------------------------------------------------------------
# Phase 1 routing weights — adjust Opus/Haiku tier assignment by rule class.
# RCE-class rules get a boost; DoS-only or low-signal rules are discounted.
# Does NOT affect Phase 0 display score — routing only.
# ---------------------------------------------------------------------------

RULE_ROUTING_WEIGHT: dict[str, float] = {
    "SUBPROCESS_SHELL_TRUE":      1.9,
    "GIT_OPTION_INJECTION":       1.8,
    "GIT_FLAG_NAME_INJECTION":    1.75,
    "OS_SHELL_INJECTION":         1.7,
    "UNSAFE_EVAL_EXEC":           1.6,
    "UNSAFE_PICKLE":              1.4,
    "UNSAFE_YAML_LOAD":           1.3,
    "TEMPLATE_INJECTION":         1.2,
    "HARDCODED_SECRET":           1.0,
    "SUBPROCESS_EXEC_STARRED":    1.1,
    "UNSANITIZED_SUBPROCESS_ARG": 1.0,
    "OPEN_USER_PATH":             0.9,
    "UNSAFE_DESERIALIZATION":            0.20,
    "REGEX_DOS":                  0.25,
}


def routing_score(sr: ScanResult) -> float:
    """Return sr.score adjusted by the highest-priority rule in the file."""
    if not sr.rules:
        return sr.score
    multiplier = max(
        (RULE_ROUTING_WEIGHT.get(r, 1.0) for r in sr.rules),
        default=1.0,
    )
    return sr.score * multiplier


# ---------------------------------------------------------------------------
# Reporting data structures
# ---------------------------------------------------------------------------

@dataclass
class ReportContext:
    """Single context object for the full reporting pipeline.

    Config fields (set once in main, not mutated afterward):
        repo_root, default_model, escalate_model, budget_limit_usd, top_n, escalate_n

    Runtime fields (populated as phases complete):
        scan_results, hunts, models, total_elapsed_s, budget_used_usd,
        git_enriched, cache_hits, total_files
    """

    # ── Config ─────────────────────────────────────────────────────────────
    repo_root:        Path
    default_model:    str
    budget_limit_usd: float
    escalate_model:   str | None = None
    top_n:            int        = 5
    escalate_n:       int        = 0

    # ── Runtime state ───────────────────────────────────────────────────────
    scan_results:    list[ScanResult]  = field(default_factory=list)
    hunts:           list[HuntResult]  = field(default_factory=list)
    models:          dict[str, Any]    = field(default_factory=dict)
    total_elapsed_s: float = 0.0
    budget_used_usd: float = 0.0
    git_enriched:    bool  = False
    cache_hits:      int   = 0
    total_files:     int   = 0
