# keryx/core/orchestrator.py
# KeryxOrchestrator — progressive escalation, advisor strategy, swarm mode.
# Sovereign, air-gapped, production ready. Requires Python 3.11+.
#
# Heavy lifting is delegated to focused sub-modules:
#   _airgap.py      — network reachability probe
#   _checkpoint.py  — atomic JSON checkpoint save/load
#   _resources.py   — system resource checks
#   _swarm.py       — multi-model debate engine
#   _escalation.py  — stateless escalation helpers (pure functions)

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._airgap import detect_airgap
from ._checkpoint import CheckpointManager
from ._escalation import get_max_escalation, get_threshold, next_state, should_escalate
from ._resources import check_resources
from ._swarm import OrchestratorSwarmAdapter
from .agent import KeryxAgent
from .router import RoutingPlan, create_router
from .shared_context import SharedContext

try:
    from ..advisors.manager import AdvisorManager
    from ..models.interface import ModelInterface
    from ..tools.Toolbox import ToolBox
except ImportError:  # pragma: no cover
    if TYPE_CHECKING:  # pragma: no cover
        from ..advisors.manager import AdvisorManager
        from ..models.interface import ModelInterface
        from ..tools.Toolbox import ToolBox
    else:  # pragma: no cover
        ModelInterface = object  # type: ignore[assignment,misc]
        AdvisorManager = object  # type: ignore[assignment,misc]
        ToolBox = object  # type: ignore[assignment,misc]

logger = logging.getLogger("keryx.orchestrator")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorMetrics:
    routing_decisions: int = 0
    escalation_levels_used: int = 1
    advisor_calls: int = 0
    tools_executed: int = 0
    swarm_votes: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class KeryxOrchestrator:
    """
    Central brain of KeryxHunter.

    Implements the full Advisor Strategy:
    - Executor (small/fast) runs the main ReAct loop via KeryxAgent
    - Advisor (large/on-demand) triggers on low confidence or stalled progress
    - SharedContext is the single source of truth across escalation levels
    - Supports airgapped / hybrid / swarm modes
    - Progressive escalation (Level 1–4)
    """

    def __init__(
        self,
        available_models: dict[str, ModelInterface],
        advisor_manager: AdvisorManager,
        toolbox: ToolBox,
        config_path: Path | None = None,
        has_gpu: bool = False,
    ) -> None:
        self.available_models = available_models
        self.advisor_manager = advisor_manager
        self.toolbox = toolbox
        self.has_gpu = has_gpu
        self.config_path = config_path
        self.router = create_router(config_path, has_gpu=has_gpu)
        self._shared_context: SharedContext | None = None
        self._shutdown_event = asyncio.Event()
        self.metrics = OrchestratorMetrics()
        self._checkpoints = CheckpointManager()

        logger.info(
            "[Orchestrator] Initialized | models=%d | GPU=%s",
            len(available_models), has_gpu,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def hunt(
        self,
        target_path: str,
        capability: str = "deep_reasoning",
        mode: str = "hybrid",
        resume: bool = True,
        user_budget_usd: float | None = None,
    ) -> dict[str, Any]:
        logger.info(
            "[Orchestrator] Hunt started | mode=%s | capability=%s | target=%s",
            mode, capability, target_path,
        )
        self._setup_signal_handlers()
        try:
            return await self._run_hunt(target_path, capability, mode, resume, user_budget_usd)
        except asyncio.CancelledError:
            logger.warning("[Orchestrator] Hunt cancelled")
            return self._generate_cancelled_report(target_path, mode)
        finally:
            self._cleanup_signal_handlers()

    # ------------------------------------------------------------------
    # Internal hunt loop
    # ------------------------------------------------------------------
    async def _run_hunt(
        self,
        target_path: str,
        capability: str,
        mode: str,
        resume: bool,
        user_budget_usd: float | None,
    ) -> dict[str, Any]:
        is_airgapped = await detect_airgap(mode)
        plan: RoutingPlan = self.router.route(
            capability=capability,
            available_models=self.available_models,
            advisor_manager=self.advisor_manager,
            is_airgapped=is_airgapped,
            user_budget_usd=user_budget_usd,
        )
        self.metrics.routing_decisions += 1

        self._shared_context = self._checkpoints.load(target_path) if resume else None
        if self._shared_context is None:
            self._shared_context = SharedContext(
                target_path=target_path,
                capability=capability,
                mode=mode,
                routing_plan=plan,
            )

        escalation_level = self._shared_context.escalation_level
        max_escalation = get_max_escalation(is_airgapped)
        attempts_at_level = 0
        max_attempts = 3
        result: dict[str, Any] | None = None

        while escalation_level <= max_escalation and not self._shutdown_event.is_set():
            logger.info(
                "[Orchestrator] Escalation level %d/%d", escalation_level, max_escalation
            )
            logger.debug(
                "STATE level=%d | attempts=%d | hypotheses=%d | errors=%d | conf=%.2f",
                escalation_level,
                attempts_at_level,
                len(self._shared_context.hypotheses),
                self._shared_context.parse_errors,
                result.get("average_confidence", -1.0) if result else -1.0,
            )

            if not await check_resources():
                break

            agent = KeryxAgent(
                executor_model=plan.executor_model,
                advisor_manager=self.advisor_manager,
                toolbox=self.toolbox,
                max_steps=100,
                confidence_threshold=get_threshold(escalation_level),
                budget_usd=plan.budget_usd,
                enforce_airgapped=plan.enforce_no_network,
            )

            try:
                result = await agent.run(
                    target_path=target_path,
                    capability=capability,
                    resume=False,
                    context=self._shared_context,
                )
                self._shared_context = agent.context
            except Exception as exc:
                logger.error(
                    "[Orchestrator] Agent failed at level %d: %s", escalation_level, exc
                )
                self._shared_context = agent.context
                escalation_level, attempts_at_level = next_state(
                    escalation_level, attempts_at_level, max_attempts
                )
                if attempts_at_level == 0:
                    self._shared_context.set_escalation_level(escalation_level)
                await self._checkpoints.save(
                    target_path, self._shared_context, self._metrics_snapshot()
                )
                continue

            await self._checkpoints.save(
                target_path, self._shared_context, self._metrics_snapshot()
            )

            if should_escalate(result, self._shared_context, escalation_level, max_escalation):
                escalation_level, attempts_at_level = next_state(
                    escalation_level, attempts_at_level, max_attempts
                )
                if attempts_at_level == 0:
                    logger.info(
                        "[Orchestrator] Max attempts — escalating to level %d", escalation_level
                    )
                    self._shared_context.set_escalation_level(escalation_level)
                else:
                    logger.info(
                        "[Orchestrator] Retrying (%d/%d)", attempts_at_level, max_attempts
                    )
                continue
            break

        if mode == "swarm" and plan.debate_models and result and not self._shutdown_event.is_set():
            swarm = OrchestratorSwarmAdapter(self.available_models, self.metrics.swarm_votes)
            result = await swarm.run(result, plan.debate_models)

        final = self._build_final_result(result, mode, is_airgapped, escalation_level)
        logger.info(
            "[Orchestrator] Done | confirmed=%d", len(final.get("confirmed_vulns", []))
        )
        return final

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _setup_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _handler() -> None:
            logger.warning("[Orchestrator] Shutdown signal received")
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, OSError):
                loop.add_signal_handler(sig, _handler)

    def _cleanup_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError, OSError):
                loop.remove_signal_handler(sig)

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------
    def _build_final_result(
        self,
        result: dict[str, Any] | None,
        mode: str,
        is_airgapped: bool,
        escalation_level: int,
    ) -> dict[str, Any]:
        if result is None:
            return {
                "status": "cancelled",
                "mode": mode,
                "airgapped": is_airgapped,
                "escalation_level": escalation_level,
                "confirmed_vulns": [],
            }
        result.update({
            "mode": mode,
            "airgapped": is_airgapped,
            "escalation_level": escalation_level,
            "metrics": self.get_metrics(),
        })
        return result

    def _generate_cancelled_report(self, target_path: str, mode: str) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "target": target_path,
            "mode": mode,
            "confirmed_vulns": [],
            "metrics": self.get_metrics(),
        }

    def _metrics_snapshot(self) -> dict[str, Any]:
        return {
            "routing_decisions": self.metrics.routing_decisions,
            "escalation_levels_used": self.metrics.escalation_levels_used,
        }

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_shared_context(self) -> SharedContext | None:
        return self._shared_context

    def get_metrics(self) -> dict[str, Any]:
        return {
            "routing_decisions": self.metrics.routing_decisions,
            "escalation_levels": self.metrics.escalation_levels_used,
            "advisor_calls": self.metrics.advisor_calls,
            "tools_executed": self.metrics.tools_executed,
            "swarm_votes": self.metrics.swarm_votes,
            "shared_context": (
                self._shared_context.get_metrics() if self._shared_context else {}
            ),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_orchestrator(
    models: dict[str, ModelInterface],
    advisors: AdvisorManager,
    tools: ToolBox,
    config_path: Path | None = None,
    has_gpu: bool = False,
) -> KeryxOrchestrator:
    return KeryxOrchestrator(
        available_models=models,
        advisor_manager=advisors,
        toolbox=tools,
        config_path=config_path,
        has_gpu=has_gpu,
    )
