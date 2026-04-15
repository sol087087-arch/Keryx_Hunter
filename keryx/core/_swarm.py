# keryx/core/_swarm.py
# OrchestratorSwarmAdapter — thin integration layer between KeryxOrchestrator
# and the full SwarmDebate engine in keryx.models.swarm.
#
# Responsibility: translate the orchestrator's hunt-result dict into the
# engine's (hypotheses, context) interface, run the debate, and merge the
# typed SwarmResult back into the dict.  All debate logic lives in
# keryx/models/swarm.py.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..models.swarm import SwarmDebate

if TYPE_CHECKING:
    from ..models.interface import ModelInterface

logger = logging.getLogger("keryx.orchestrator.swarm")


class OrchestratorSwarmAdapter:
    """
    Adapts KeryxOrchestrator's hunt-result dict to SwarmDebate.debate() and
    back.  One instance is created per swarm run; stateless between hunts.

    Args:
        available_models: All models the orchestrator knows about.
        swarm_votes:      Shared mutable dict from OrchestratorMetrics —
                          written back after each debate so the orchestrator
                          can track per-hypothesis vote counts.
    """

    def __init__(
        self,
        available_models: dict[str, ModelInterface],
        swarm_votes: dict[str, int],
    ) -> None:
        self._models = available_models
        self._votes = swarm_votes

    async def run(
        self,
        base_result: dict[str, Any],
        debate_models: list[str],
    ) -> dict[str, Any]:
        """
        Run a swarm debate and merge results back into *base_result*.

        Returns the same dict (mutated), so the orchestrator can do:
            result = await swarm.run(result, plan.debate_models)
        """
        # Resolve model name strings → ModelInterface instances.
        models = [
            self._models[name]
            for name in debate_models
            if name in self._models
        ]
        if not models:
            base_result.update({
                "swarm_mode": True,
                "swarm_error": "no debate models available",
            })
            return base_result

        # Hypotheses are always list[str] from the agent — cast defensively.
        raw = base_result.get("hypotheses", [])
        hypotheses: list[str] = [
            h if isinstance(h, str) else str(h) for h in raw
        ]

        if not hypotheses:
            base_result.update({"swarm_mode": True, "consensus_reached": False})
            return base_result

        engine = SwarmDebate(models)
        sr = await engine.debate(hypotheses)

        # Write per-hypothesis vote counts back to orchestrator metrics.
        for hyp in sr.confirmed:
            k = hyp[:60]
            self._votes[k] = self._votes.get(k, 0) + 1

        # Merge SwarmResult into the result dict.
        base_result.update({
            "swarm_mode":      True,
            "swarm_confirmed": sr.confirmed,
            "consensus_ratio": sr.weighted_consensus_ratio,
            "swarm_voters":    sr.swarm_voters,
            "swarm_timeout":   sr.timeout_occurred,
            "debate_rounds":   sr.debate_rounds,
        })
        if sr.confirmed:
            base_result.setdefault("confirmed_vulns", []).extend(
                {"hypothesis": h, "swarm_verified": True}
                for h in sr.confirmed
            )

        logger.info(
            "[SwarmAdapter] Done | confirmed=%d/%d | voters=%d | ratio=%.2f",
            len(sr.confirmed),
            len(hypotheses),
            sr.swarm_voters,
            sr.weighted_consensus_ratio,
        )
        return base_result
