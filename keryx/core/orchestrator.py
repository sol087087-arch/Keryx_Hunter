# keryx/core/orchestrator.py
# Full Orchestrator for KeryxHunter - Advisor Strategy, Shared Context,
# progressive escalation, swarm/debate mode, true air-gapped detection.
# Sovereign, air-gapped, production ready.
# Requires Python 3.11+ (asyncio.timeout)

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import signal
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from .agent import KeryxAgent
from .router import RoutingPlan, create_router
from .shared_context import SharedContext

try:
    from ..advisors.manager import AdvisorManager
    from ..models.interface import ModelInterface
    from ..tools.Toolbox import ToolBox
except ImportError:
    if TYPE_CHECKING:
        from ..advisors.manager import AdvisorManager
        from ..models.interface import ModelInterface
        from ..tools.Toolbox import ToolBox
    else:
        ModelInterface = object  # type: ignore[assignment,misc]
        AdvisorManager = object  # type: ignore[assignment,misc]
        ToolBox = object  # type: ignore[assignment,misc]

logger = logging.getLogger("keryx.orchestrator")

# Global swarm debate timeout — one hung model won't kill the whole session
_SWARM_TIMEOUT = 60.0

# Checkpoint every N escalation loop iterations
_CHECKPOINT_EVERY_N = 1


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


def _hyp_key(hyp: str) -> str:
    """
    Short readable key for swarm vote tracking.
    DISPUTE RESOLVED: MD5 8-char hex keys are unreadable during debugging.
    We truncate the hypothesis text to 60 chars instead — saves memory,
    keeps logs meaningful.
    """
    return hyp[:60].replace("\n", " ")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class KeryxOrchestrator:
    """
    Central brain of KeryxHunter.
    Implements the full Advisor Strategy:
    - Executor (small/fast) runs the main ReAct loop
    - Advisor (large/on-demand) triggers on low confidence, no progress, patterns
    - SharedContext is the single source of truth across escalation levels
    - Supports airgapped / hybrid / swarm modes
    - Progressive escalation (Level 1–4)
    """

    def __init__(
        self,
        available_models: dict[str, "ModelInterface"],
        advisor_manager: "AdvisorManager",
        toolbox: "ToolBox",
        config_path: Path | None = None,
        has_gpu: bool = False,
    ):
        self.available_models = available_models
        self.advisor_manager = advisor_manager
        self.toolbox = toolbox
        self.has_gpu = has_gpu
        self.config_path = config_path
        self.router = create_router(config_path, has_gpu=has_gpu)
        self._shared_context: SharedContext | None = None
        self._shutdown_event = asyncio.Event()
        self.metrics = OrchestratorMetrics()

        logger.info(
            f"[Orchestrator] Initialized | models={len(available_models)} | GPU={has_gpu}"
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
            f"[Orchestrator] Hunt started | mode={mode} | "
            f"capability={capability} | target={target_path}"
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
        is_airgapped = await self._is_airgapped(mode)
        plan: RoutingPlan = self.router.route(
            capability=capability,
            available_models=self.available_models,
            advisor_manager=self.advisor_manager,
            is_airgapped=is_airgapped,
            user_budget_usd=user_budget_usd,
        )
        self.metrics.routing_decisions += 1

        self._shared_context = self._load_checkpoint(target_path) if resume else None
        if self._shared_context is None:
            self._shared_context = SharedContext(
                target_path=target_path,
                capability=capability,
                mode=mode,
                routing_plan=plan,
            )

        escalation_level = self._shared_context.escalation_level
        max_escalation = 4 if not is_airgapped else 2
        attempts_at_level = 0
        max_attempts = 3
        result: dict[str, Any] | None = None
        iteration = 0

        while escalation_level <= max_escalation and not self._shutdown_event.is_set():
            iteration += 1
            logger.info(f"[Orchestrator] Escalation level {escalation_level}/{max_escalation}")
            logger.debug(
                "STATE level=%d | attempts=%d | hypotheses=%d | errors=%d | conf=%.2f",
                escalation_level,
                attempts_at_level,
                len(self._shared_context.hypotheses),
                self._shared_context.parse_errors,
                result.get("average_confidence", -1.0) if result else -1.0,
            )

            if not await self._check_resources():
                logger.warning("[Orchestrator] Insufficient resources — stopping escalation")
                break

            agent = KeryxAgent(
                executor_model=plan.executor_model,
                advisor_manager=self.advisor_manager,
                toolbox=self.toolbox,
                max_steps=100,
                confidence_threshold=self._get_dynamic_threshold(escalation_level),
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
                logger.error(f"[Orchestrator] Agent failed at level {escalation_level}: {exc}")
                self._shared_context = agent.context  # save partial progress
                attempts_at_level += 1
                if attempts_at_level >= max_attempts:
                    escalation_level += 1
                    attempts_at_level = 0
                    self._shared_context.set_escalation_level(escalation_level)
                await self._save_checkpoint(target_path)
                continue

            # Checkpoint after every iteration
            await self._save_checkpoint(target_path)

            if self._should_escalate_further(result, escalation_level, max_escalation):
                attempts_at_level += 1
                if attempts_at_level >= max_attempts:
                    logger.info(f"[Orchestrator] Max attempts at level {escalation_level} — escalating")
                    escalation_level += 1
                    attempts_at_level = 0
                    self._shared_context.set_escalation_level(escalation_level)
                else:
                    logger.info(f"[Orchestrator] Retrying ({attempts_at_level}/{max_attempts})")
                continue
            break

        # Swarm debate
        if mode == "swarm" and plan.debate_models and result and not self._shutdown_event.is_set():
            result = await self._run_swarm_debate(result, plan.debate_models)

        final = self._build_final_result(result, mode, is_airgapped, escalation_level)
        logger.info(f"[Orchestrator] Done | confirmed={len(final.get('confirmed_vulns', []))}")
        return final

    # ------------------------------------------------------------------
    # Air-gapped detection
    # ------------------------------------------------------------------
    async def _is_airgapped(self, explicit_mode: str) -> bool:
        if explicit_mode == "airgapped":
            return True

        # Heuristic: proxy vars suggest network access
        if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
            logger.debug("[Orchestrator] Proxy env var detected — assuming network available")
            return False

        # Parallel TCP probe
        probes = [
            ("1.1.1.1", 53),   # Cloudflare DNS
            ("8.8.8.8", 53),   # Google DNS
            ("208.67.222.222", 53),  # OpenDNS
        ]

        async def _probe(host: str, port: int) -> bool:
            try:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: socket.create_connection((host, port), timeout=1.5),
                    ),
                    timeout=2.0,
                )
                return True
            except Exception:
                return False

        try:
            async with asyncio.timeout(3.0):
                results = await asyncio.gather(*[_probe(h, p) for h, p in probes])
                if any(results):
                    return False
        except TimeoutError:
            pass

        logger.info("[Orchestrator] Network unreachable — air-gapped mode active")
        return True

    # ------------------------------------------------------------------
    # Checkpoint — async, non-blocking write
    # ------------------------------------------------------------------
    def _checkpoint_path(self, target_path: str) -> Path:
        key = hashlib.sha256(target_path.encode()).hexdigest()[:12]
        return Path(".keryx_checkpoints") / f"orchestrator_{key}.json"

    async def _save_checkpoint(self, target_path: str) -> None:
        if not self._shared_context:
            return
        Path(".keryx_checkpoints").mkdir(exist_ok=True)
        path = self._checkpoint_path(target_path)

        state = {
            "context": self._shared_context.to_dict(),
            "metrics": {
                "routing_decisions": self.metrics.routing_decisions,
                "escalation_levels_used": self.metrics.escalation_levels_used,
            },
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_atomic_json, path, state)

    def _write_atomic_json(self, path: Path, data: dict[str, Any]) -> None:
        """Atomic write: .tmp → rename."""
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.rename(path)
            logger.debug(f"Checkpoint saved: {path.name}")
        except Exception as exc:
            logger.error(f"Atomic checkpoint write failed: {exc}")
            tmp.unlink(missing_ok=True)

    def _load_checkpoint(self, target_path: str) -> SharedContext | None:
        path = self._checkpoint_path(target_path)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ctx = SharedContext.from_dict(data["context"])
            logger.info(f"[Orchestrator] Resumed from checkpoint ({path.name})")
            return ctx
        except Exception as exc:
            logger.warning(f"Checkpoint load failed ({exc}) — starting fresh")
            return None

    # ------------------------------------------------------------------
    # Swarm debate
    # ------------------------------------------------------------------
    async def _run_swarm_debate(self, base_result: dict[str, Any], debate_models: list[str]) -> dict[str, Any]:
        logger.info(f"[Orchestrator] Swarm debate | models={debate_models}")
        raw_hyps = base_result.get("hypotheses", [])
        hypotheses: list[str] = [
            h if isinstance(h, str) else json.dumps(h, ensure_ascii=False)
            for h in raw_hyps
        ]
        if not hypotheses:
            base_result.update({"swarm_mode": True, "consensus_reached": False})
            return base_result

        available_swarm = [m for m in debate_models if m in self.available_models]
        if not available_swarm:
            base_result.update({"swarm_mode": True, "swarm_error": "No debate models available"})
            return base_result

        tasks = [
            self._analyze_with_model(self.available_models[m], hypotheses)
            for m in available_swarm
        ]
        results: list[Any] = []
        try:
            async with asyncio.timeout(_SWARM_TIMEOUT):
                results = await asyncio.gather(*tasks, return_exceptions=True)
        except TimeoutError:
            logger.warning(f"[Orchestrator] Swarm debate timed out after {_SWARM_TIMEOUT}s")

        votes: dict[str, int] = {_hyp_key(h): 0 for h in hypotheses}
        valid_voters = 0
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Swarm model '{available_swarm[i]}' failed: {r}")
                continue
            valid_voters += 1
            for hyp, confirmed in r.items():
                if confirmed:
                    k = _hyp_key(hyp)
                    votes[k] = votes.get(k, 0) + 1
                    self.metrics.swarm_votes[k] = self.metrics.swarm_votes.get(k, 0) + 1

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
        self, model: "ModelInterface", hypotheses: list[str]
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
                logger.debug(f"Swarm model failed on hypothesis: {exc}")
                results[hyp] = False
        return results

    # ------------------------------------------------------------------
    # Resource check / escalation helpers
    # ------------------------------------------------------------------
    async def _check_resources(self) -> bool:
        try:
            if psutil.virtual_memory().available < 512 * 1024 * 1024:
                logger.warning("Low memory — stopping escalation")
                return False
        except Exception:
            pass
        return True

    def _should_escalate_further(
        self, result: dict[str, Any], current_level: int, max_escalation: int
    ) -> bool:
        if not self._shared_context:
            return False
        if len(result.get("confirmed_vulns", [])) >= 1:
            return False
        low_confidence = result.get("average_confidence", 0.0) < 0.5
        no_progress = len(self._shared_context.hypotheses) == 0
        too_many_errors = self._shared_context.parse_errors > 5
        return (low_confidence or no_progress or too_many_errors) and current_level < max_escalation

    def _get_dynamic_threshold(self, escalation_level: int) -> float:
        return {1: 0.65, 2: 0.60, 3: 0.55, 4: 0.50}.get(escalation_level, 0.60)

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
# Convenience factory — sync, no async needed
# ---------------------------------------------------------------------------
def create_orchestrator(
    models: dict[str, "ModelInterface"],
    advisors: "AdvisorManager",
    tools: "ToolBox",
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
