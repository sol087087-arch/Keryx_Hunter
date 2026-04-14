# keryx/core/router.py
# Capability Router for KeryxHunter
# Decides executor / advisor / budget per task.
# Sovereign, air-gapped aware, cascade-advisor aware.

import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml  # type: ignore[import-untyped]

from ..advisors.manager import AdvisorManager
from ..models.interface import ModelInterface

logger = logging.getLogger("keryx.router")


# ---------------------------------------------------------------------------
# Model capability tiers
# ---------------------------------------------------------------------------
class ModelCapability(TypedDict):
    tier: int
    is_local: bool
    context_length: int
    requires_gpu: bool


_CAPABILITY_TIERS: dict[str, ModelCapability] = {
    "llama-4-70b": {"tier": 100, "is_local": True, "context_length": 32768, "requires_gpu": True},
    "qwen3.5-coder-32b": {"tier": 95, "is_local": True, "context_length": 131072, "requires_gpu": True},
    "deepseek-r1": {"tier": 90, "is_local": False, "context_length": 16384, "requires_gpu": False},
    "mistral-large": {"tier": 88, "is_local": False, "context_length": 32768, "requires_gpu": False},
    "codellama-34b": {"tier": 85, "is_local": True, "context_length": 16384, "requires_gpu": True},
    "llama-4-13b": {"tier": 50, "is_local": True, "context_length": 8192, "requires_gpu": False},
    "qwen3-coder-8b": {"tier": 40, "is_local": True, "context_length": 8192, "requires_gpu": False},
}


# Pre-computed fallback chains (module level)
def _build_fallback_chain(model_name: str) -> list[str]:
    """
    Return fallback candidates ordered by capability tier (descending),
    excluding the requested model itself.
    """
    _fallback: ModelCapability = cast(ModelCapability, {"tier": 0, "is_local": True, "context_length": 0, "requires_gpu": False})
    requested_tier = _CAPABILITY_TIERS.get(model_name, _fallback)["tier"]
    return [
        name
        for name, specs in sorted(
            _CAPABILITY_TIERS.items(), key=lambda x: x[1]["tier"], reverse=True
        )
        if name != model_name and specs["tier"] <= requested_tier
    ]


_FALLBACK_CHAINS: dict[str, list[str]] = {
    name: _build_fallback_chain(name) for name in _CAPABILITY_TIERS
}


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------
class RoutingPlan:
    """
    Concrete routing decision for a given capability + environment.
    Passed to KeryxAgent and AdvisorManager.
    """

    def __init__(
        self,
        executor_model: ModelInterface,
        advisor_name: str,
        advisor_chain: list[str],
        budget_usd: float | None,
        enforce_no_network: bool,
        require_consensus: bool,
        consensus_threshold: float,
        require_fuzzing_confirm: bool,
        debate_models: list[str],
        capability: str,
    ):
        self.executor_model = executor_model
        self.advisor_name = advisor_name
        self.advisor_chain = advisor_chain
        self.budget_usd = budget_usd
        self.enforce_no_network = enforce_no_network
        self.require_consensus = require_consensus
        self.consensus_threshold = consensus_threshold
        self.require_fuzzing_confirm = require_fuzzing_confirm
        self.debate_models = debate_models
        self.capability = capability
        self._validate()

    def _validate(self) -> None:
        if self.enforce_no_network and self.budget_usd is not None:
            raise ValueError(
                "Air-gapped mode cannot have a budget — local compute has no API cost."
            )
        if self.require_consensus and len(self.debate_models) < 2:
            raise ValueError(
                f"Consensus requires at least 2 debate models, got {self.debate_models!r}."
            )
        if self.advisor_chain is not None and len(self.advisor_chain) == 1:
            raise ValueError(
                f"Advisor chain must have ≥2 entries or be empty, got {self.advisor_chain!r}."
            )

    def __repr__(self) -> str:
        return (
            f"RoutingPlan(capability={self.capability!r}, "
            f"executor={self.executor_model.model_name!r}, "
            f"advisor={self.advisor_name!r}, "
            f"budget={self.budget_usd}, "
            f"airgapped={self.enforce_no_network})"
        )


# ---------------------------------------------------------------------------
# Default capability profiles
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, dict[str, Any]] = {
    "fast_pattern_matching": {
        "executor": "qwen3-coder-8b",
        "advisor": "none",
        "advisor_chain": [],
        "timeout_seconds": 5,
        "enforce_no_network": True,
        "budget_usd": None,
    },
    "deep_reasoning": {
        "executor": "qwen3-coder-8b",
        "advisor": "local-advisor",
        "advisor_chain": [],
        "fallback_advisor": "cloud-advisor",
        "timeout_seconds": 120,
        "self_critique_rounds": 2,
        "enforce_no_network": False,
        "budget_usd": 5.0,
    },
    "airgapped": {
        "executor": "qwen3-coder-8b",
        "advisor": "firefox-specialist",
        "advisor_chain": [],
        "enforce_no_network": True,
        "debate_models": ["llama-4-70b", "qwen3.5-coder-32b", "qwen3-coder-8b"],
        "budget_usd": None,
    },
    "verified_only": {
        "executor": "llama-4-70b",
        "advisor": "cascade-advisor",
        "advisor_chain": ["local-advisor", "cloud-advisor"],
        "require_consensus": True,
        "consensus_threshold": 0.67,
        "require_fuzzing_confirm": True,
        "enforce_no_network": False,
        "budget_usd": 10.0,
    },
}

_METRIC_KEYS = ("routing_decisions", "fallbacks_used", "health_check_failures", "gpu_skips")


class CapabilityRouter:
    """
    Routes hunt requests to the right executor + advisor combination
    based on capability profile, system state, and available models.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        has_gpu: bool = False,
    ):
        self.config_path = config_path or Path.home() / ".keryx" / "capabilities.yaml"
        self.capabilities = self._load_capabilities()
        self._has_gpu = has_gpu
        self.metrics: dict[str, int] = dict.fromkeys(_METRIC_KEYS, 0)

        logger.info(
            f"[Router] Loaded {len(self.capabilities)} capability profiles | gpu={has_gpu}"
        )

    def get_metrics(self) -> dict[str, int]:
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        self.metrics = dict.fromkeys(_METRIC_KEYS, 0)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------
    def route(
        self,
        capability: str,
        available_models: dict[str, ModelInterface],
        advisor_manager: AdvisorManager,
        is_airgapped: bool = False,
        user_budget_usd: float | None = None,
    ) -> RoutingPlan:
        self.metrics["routing_decisions"] += 1

        if capability not in self.capabilities:
            logger.warning(
                f"Unknown capability '{capability}' — falling back to 'deep_reasoning'"
            )
            capability = "deep_reasoning"

        cfg = self.capabilities[capability].copy()

        executor = self._resolve_executor(
            requested=cfg["executor"],
            available=available_models,
            is_airgapped=is_airgapped,
        )

        # Health check
        if not executor.is_healthy():
            self.metrics["health_check_failures"] += 1
            raise RuntimeError(
                f"Executor '{executor.model_name}' failed health check — "
                f"check model path and GPU availability."
            )

        advisor_name, advisor_chain = self._resolve_advisor(
            cfg=cfg,
            advisor_manager=advisor_manager,
            is_airgapped=is_airgapped,
        )

        budget = self._resolve_budget(
            profile_budget=cfg.get("budget_usd"),
            user_budget=user_budget_usd,
            enforce_no_network=cfg.get("enforce_no_network", False),
        )

        plan = RoutingPlan(
            executor_model=executor,
            advisor_name=advisor_name,
            advisor_chain=advisor_chain,
            budget_usd=budget,
            enforce_no_network=cfg.get("enforce_no_network", False),
            require_consensus=cfg.get("require_consensus", False),
            consensus_threshold=cfg.get("consensus_threshold", 0.67),
            require_fuzzing_confirm=cfg.get("require_fuzzing_confirm", False),
            debate_models=cfg.get("debate_models", []),
            capability=capability,
        )

        logger.debug(f"Resolved: {plan}")
        return plan

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def _resolve_executor(
        self,
        requested: str,
        available: dict[str, ModelInterface],
        is_airgapped: bool,
    ) -> ModelInterface:
        candidates = [requested] + _FALLBACK_CHAINS.get(requested, [])

        for name in candidates:
            if name not in available:
                continue

            model = available[name]
            _fallback_cap: ModelCapability = cast(ModelCapability, {"tier": 0, "is_local": False, "context_length": 0, "requires_gpu": False})
            caps: ModelCapability = _CAPABILITY_TIERS.get(name, _fallback_cap)

            if caps.get("requires_gpu", False) and not self._has_gpu:
                logger.warning(f"Skipping '{name}' — requires GPU")
                self.metrics["gpu_skips"] += 1
                continue

            if is_airgapped and not caps.get("is_local", False):
                logger.debug(f"Skipping '{name}' — not local (air-gapped mode)")
                continue

            if is_airgapped and not getattr(model, "is_local", False):
                logger.debug(f"Skipping '{name}' — model instance reports non-local")
                continue

            if name != requested:
                self.metrics["fallbacks_used"] += 1
                logger.info(f"Executor fallback: '{requested}' → '{name}'")

            return model

        raise RuntimeError(
            f"No suitable executor found for '{requested}' "
            f"(air-gapped={is_airgapped}, has_gpu={self._has_gpu})"
        )

    def _resolve_advisor(
        self,
        cfg: dict[str, Any],
        advisor_manager: AdvisorManager,
        is_airgapped: bool,
    ) -> tuple[str, list[str]]:
        requested = cfg.get("advisor", "none")
        fallback = cfg.get("fallback_advisor")
        chain = list(cfg.get("advisor_chain", []))

        if requested == "none":
            return "none", []

        def _try(name: str | None) -> str | None:
            if not name:
                return None
            advisor = advisor_manager.get_advisor(name)
            if advisor is None:
                logger.debug(f"Advisor '{name}' not registered")
                return None
            if is_airgapped and getattr(advisor, "requires_network", False):
                logger.info(f"Advisor '{name}' requires network — skipping (air-gapped)")
                return None
            return name

        # Primary
        resolved = _try(requested)
        if resolved:
            return resolved, chain

        # Fallback
        if fallback:
            resolved = _try(fallback)
            if resolved:
                logger.info(f"Advisor fallback: '{requested}' → '{fallback}'")
                return resolved, []

        # Cascade chain
        if chain:
            if is_airgapped:
                chain = [
                    a for a in chain
                    if not getattr(advisor_manager.get_advisor(a), "requires_network", True)
                ]
            if chain:
                return chain[0], chain

        logger.warning(f"No suitable advisor for '{requested}' — running without advisor")
        return "none", []

    @staticmethod
    def _resolve_budget(
        profile_budget: float | None,
        user_budget: float | None,
        enforce_no_network: bool,
    ) -> float | None:
        if enforce_no_network:
            return None
        if user_budget is not None and profile_budget is not None:
            return min(user_budget, profile_budget)
        return user_budget if user_budget is not None else profile_budget

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    def _load_capabilities(self) -> dict[str, dict[str, Any]]:
        merged = {k: v.copy() for k, v in _DEFAULTS.items()}

        if not self.config_path.exists():
            return merged

        try:
            with open(self.config_path) as f:
                user_cfg = yaml.safe_load(f) or {}
            user_caps = user_cfg.get("capabilities", {})

            for name, profile in user_caps.items():
                if name in merged:
                    merged[name].update(profile)
                else:
                    merged[name] = profile

            logger.info(f"Merged {len(user_caps)} profiles from {self.config_path}")
        except Exception as exc:
            logger.warning(f"Failed to load capabilities.yaml: {exc} — using defaults")

        return merged

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------
    def list_capabilities(self) -> list[str]:
        return sorted(self.capabilities.keys())

    def describe(self, capability: str) -> str:
        if capability not in self.capabilities:
            return f"Unknown capability: {capability!r}"
        lines = [f"Capability: {capability}"]
        for k, v in self.capabilities[capability].items():
            lines.append(f" {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_router(
    config_path: Path | None = None,
    has_gpu: bool = False,
) -> CapabilityRouter:
    return CapabilityRouter(config_path, has_gpu)
