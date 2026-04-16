# keryx/core/hunt_config.py
# HuntConfig — typed configuration bundle for KeryxAgent.
#
# Replaces scattered constructor kwargs with a named, validated dataclass.
# Three built-in presets cover the most common use cases; customise freely.
#
# Usage:
#   from keryx.core.hunt_config import HuntConfig
#
#   # Named preset
#   cfg = HuntConfig.fast()
#
#   # Custom
#   cfg = HuntConfig(max_steps=40, confidence_threshold=0.70, budget_usd=3.00)
#
#   # Create agent
#   agent = KeryxAgent(executor_model=model, advisor_manager=am,
#                      toolbox=tb, **cfg.to_agent_kwargs())

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class HuntConfig:
    """
    Typed configuration bundle for KeryxAgent.

    Attributes:
        max_steps:                   Hard step ceiling before the hunt aborts.
        confidence_threshold:        Minimum confidence to accept a vuln as real.
        budget_usd:                  Maximum API spend; None = unlimited.
        verification_mode:           How codeql HIGH findings are verified.
            "none"     — AST-only, immediate confirm, fastest.
            "strict"   — auto injection_verifier, no extra LLM tokens, CI-safe.
            "flexible" — agent-driven 2-step window then static fallback.
        max_clean_scans_before_exit: Exit after this many clean codeql scans.
                                     1 = exit on first clean scan (saves budget).
        generate_timeout:            Seconds before a single LLM call is aborted.
        max_prompt_chars:            Trim history when prompt exceeds this length.
        enforce_airgapped:           Abort if the executor requires network access.
    """
    max_steps:                   int                             = 25
    confidence_threshold:        float                           = 0.60
    budget_usd:                  float | None                    = 2.00
    verification_mode:           Literal["none", "strict", "flexible"] = "flexible"
    max_clean_scans_before_exit: int                             = 1
    generate_timeout:            float                           = 60.0
    max_prompt_chars:            int                             = 32_000
    enforce_airgapped:           bool                            = False

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def fast(cls) -> "HuntConfig":
        """
        CI / quick-triage preset.
        Strict verification, exits on first clean scan, tight budget.
        Optimised for: pull-request gates, batch scanning many small files.
        """
        return cls(
            max_steps=10,
            confidence_threshold=0.55,
            budget_usd=0.50,
            verification_mode="strict",
            max_clean_scans_before_exit=1,
            generate_timeout=30.0,
        )

    @classmethod
    def deep(cls) -> "HuntConfig":
        """
        Thorough investigation preset.
        Higher confidence bar, more steps, tolerates 2 clean scans before exit.
        Optimised for: manual security reviews, complex multi-file targets.
        """
        return cls(
            max_steps=50,
            confidence_threshold=0.75,
            budget_usd=5.00,
            verification_mode="flexible",
            max_clean_scans_before_exit=2,
            generate_timeout=90.0,
        )

    @classmethod
    def local(cls) -> "HuntConfig":
        """
        Air-gapped / local-model preset.
        No budget limit, airgap enforced, more steps to compensate for smaller models.
        Optimised for: offline environments, privacy-sensitive codebases.
        """
        return cls(
            max_steps=30,
            confidence_threshold=0.60,
            budget_usd=None,
            verification_mode="flexible",
            max_clean_scans_before_exit=1,
            generate_timeout=120.0,
            enforce_airgapped=True,
        )

    # ------------------------------------------------------------------
    # KeryxAgent integration
    # ------------------------------------------------------------------

    def to_agent_kwargs(self) -> dict[str, Any]:
        """
        Expand config into keyword arguments accepted by KeryxAgent.__init__.

        Example::
            agent = KeryxAgent(executor_model=model, advisor_manager=am,
                               toolbox=tb, **cfg.to_agent_kwargs())
        """
        return {
            "max_steps":                   self.max_steps,
            "confidence_threshold":        self.confidence_threshold,
            "budget_usd":                  self.budget_usd,
            "verification_mode":           self.verification_mode,
            "max_clean_scans_before_exit": self.max_clean_scans_before_exit,
            "generate_timeout":            self.generate_timeout,
            "max_prompt_chars":            self.max_prompt_chars,
            "enforce_airgapped":           self.enforce_airgapped,
        }

    def __repr__(self) -> str:
        return (
            f"HuntConfig(steps={self.max_steps}, mode={self.verification_mode!r}, "
            f"budget=${self.budget_usd}, clean_exit={self.max_clean_scans_before_exit})"
        )
