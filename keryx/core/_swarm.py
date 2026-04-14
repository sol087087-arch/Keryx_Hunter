# keryx/core/_swarm.py
# SwarmDebate — multi-model hypothesis voting for KeryxOrchestrator.

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.interface import ModelInterface

logger = logging.getLogger("keryx.orchestrator.swarm")

_SWARM_TIMEOUT = 60.0


def _hyp_key(hyp: str) -> str:
    """
    Short, readable key for vote tracking and log lines.
    Truncates to 60 chars — readable in debugger, unlike MD5 hashes.
    """
    return hyp[:60].replace("\n", " ")


class SwarmDebate:
    """
    Runs a multi-model debate over a set of vulnerability hypotheses.
    Stateless between hunts — receives available_models and a mutable
    swarm_votes dict from the orchestrator at construction time.
    """

    def __init__(
        self,
        available_models: dict[str, ModelInterface],
        swarm_votes: dict[str, int],
    ) -> None:
        self._models = available_models
        self._votes = swarm_votes  # shared ref — orchestrator metrics dict

    async def run(
        self, base_result: dict[str, Any], debate_models: list[str]
    ) -> dict[str, Any]:
        logger.info("Swarm debate | models=%s", debate_models)
        raw_hyps = base_result.get("hypotheses", [])
        hypotheses: list[str] = [
            h if isinstance(h, str) else json.dumps(h, ensure_ascii=False)
            for h in raw_hyps
        ]

        if not hypotheses:
            base_result.update({"swarm_mode": True, "consensus_reached": False})
            return base_result

        available_swarm = [m for m in debate_models if m in self._models]
        if not available_swarm:
            base_result.update({"swarm_mode": True, "swarm_error": "No debate models available"})
            return base_result

        tasks = [
            self._analyze_with_model(self._models[m], hypotheses)
            for m in available_swarm
        ]
        raw_results: list[Any] = []
        try:
            async with asyncio.timeout(_SWARM_TIMEOUT):
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        except TimeoutError:
            logger.warning("Swarm debate timed out after %.1fs", _SWARM_TIMEOUT)

        votes: dict[str, int] = {_hyp_key(h): 0 for h in hypotheses}
        valid_voters = 0
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("Swarm model '%s' failed: %s", available_swarm[i], r)
                continue
            valid_voters += 1
            for hyp, confirmed in r.items():
                if confirmed:
                    k = _hyp_key(hyp)
                    votes[k] = votes.get(k, 0) + 1
                    self._votes[k] = self._votes.get(k, 0) + 1

        swarm_confirmed: list[str] = []
        if valid_voters > 0:
            swarm_confirmed = [
                h for h in hypotheses
                if votes.get(_hyp_key(h), 0) / valid_voters >= 0.67
            ]

        base_result.update({
            "swarm_mode": True,
            "swarm_confirmed": swarm_confirmed,
            "consensus_ratio": len(swarm_confirmed) / len(hypotheses) if hypotheses else 0,
            "swarm_voters": valid_voters,
        })
        if swarm_confirmed:
            base_result.setdefault("confirmed_vulns", []).extend(
                [{"hypothesis": h, "swarm_verified": True} for h in swarm_confirmed]
            )
        return base_result

    async def _analyze_with_model(
        self, model: ModelInterface, hypotheses: list[str]
    ) -> dict[str, bool]:
        loop = asyncio.get_running_loop()
        results: dict[str, bool] = {}
        for hyp in hypotheses:
            try:
                prompt = (
                    f"Analyze this potential vulnerability:\n{hyp}\n\n"
                    "Is this real and exploitable? Answer: true or false only."
                )
                raw = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda p=prompt: model.generate(p, max_tokens=10, grammar=None),  # type: ignore[misc]
                    ),
                    timeout=30.0,
                )
                results[hyp] = "true" in raw.strip().lower()
            except Exception as exc:
                logger.debug("Swarm model failed on hypothesis: %s", exc)
                results[hyp] = False
        return results
