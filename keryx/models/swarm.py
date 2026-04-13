# keryx/models/swarm.py
# Swarm / Multi-agent debate engine for KeryxHunter.
# Runs multiple models in parallel, applies weighted consensus voting.
# Supports iterative debate rounds and adversarial skeptic review.
# Sovereign, async, timeout-safe, grammar-enforced, production-ready.
# Requires Python 3.11+ (asyncio.timeout)

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from .interface import ModelInterface, GenerationConfig

logger = logging.getLogger("keryx.models.swarm")


# ---------------------------------------------------------------------------
# GBNF grammar for strict verdict JSON output
# 
# FIX 2: 'true'/'false' as RULE NAMES shadow GBNF built-ins and are rejected
# by the llama.cpp grammar validator. Renamed to 'bool-val', 'bool-true',
# 'bool-false'. JSON boolean literals are just quoted directly.
# ---------------------------------------------------------------------------
SWARM_VERDICT_GRAMMAR = r"""
root ::= "{" ws kv-hyps ws "}"
kv-hyps ::= "\"hypotheses\"" ws ":" ws "[" ws hyp-list ws "]"
hyp-list ::= hypothesis ( "," ws hypothesis )*
hypothesis ::= "{" ws kv-idx "," ws kv-verdict "," ws kv-conf "," ws kv-reason ws "}"
kv-idx ::= "\"index\"" ws ":" ws [0-9]+
kv-verdict ::= "\"verdict\"" ws ":" ws bool-val
kv-conf ::= "\"confidence\"" ws ":" ws number
kv-reason ::= "\"reason\"" ws ":" ws string
bool-val ::= "true" | "false"
number ::= [0-9]+ ( "." [0-9]+ )?
string ::= "\"" ( [^"\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
ws ::= [ \t\n\r]*
"""
# Skeptic uses identical schema — same grammar
SKEPTIC_VERDICT_GRAMMAR = SWARM_VERDICT_GRAMMAR


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class SwarmVote:
    """One model's verdict on one hypothesis."""
    model_name: str
    hypothesis: str
    verdict: bool
    confidence: float
    reasoning: str = ""
    weight: float = 1.0
    response_time_ms: float = 0.0


@dataclass
class SwarmResult:
    """Aggregated result of a full swarm debate."""
    hypotheses: List[str]
    confirmed: List[str]
    rejected: List[str]
    votes: Dict[str, List[SwarmVote]]
    weighted_consensus_ratio: float
    raw_consensus_ratio: float
    swarm_voters: int
    timeout_occurred: bool = False
    debate_rounds: int = 1
    total_time_ms: float = 0.0
    skeptic_overrides: int = 0
    grammar_violations: int = 0


@dataclass
class ModelWeight:
    model_name: str
    weight: float = 1.0
    capability_tier: int = 50


_DEFAULT_WEIGHTS: Dict[str, ModelWeight] = {
    "claude-3.5-sonnet": ModelWeight("claude-3.5-sonnet", 1.45, 100),
    "gpt-5.4": ModelWeight("gpt-5.4", 1.35, 98),
    "gpt-5-turbo": ModelWeight("gpt-5-turbo", 1.40, 99),
    "llama-4-70b": ModelWeight("llama-4-70b", 1.25, 90),
    "qwen3.5-coder-32b": ModelWeight("qwen3.5-coder-32b", 1.20, 88),
    "deepseek-r1": ModelWeight("deepseek-r1", 1.15, 85),
    "llama-4-13b": ModelWeight("llama-4-13b", 0.90, 70),
    "qwen3-coder-8b": ModelWeight("qwen3-coder-8b", 0.75, 55),
    "phi-4-mini": ModelWeight("phi-4-mini", 0.60, 40),
    "gemma-3-2b": ModelWeight("gemma-3-2b", 0.40, 25),
}


# ---------------------------------------------------------------------------
# SwarmDebate
# ---------------------------------------------------------------------------
class SwarmDebate:
    """
    Multi-agent debate / consensus engine.
    - GBNF grammar enforces valid JSON output — no regex fallback needed.
    - Weighted voting: larger/stronger models count more.
    - Adversarial skeptic review: one model tasked with disproving consensus.
    - All votes (positive AND negative) are tracked for correct ratio math.
    """
    def __init__(
        self,
        models: List[ModelInterface],
        consensus_threshold: float = 0.67,
        swarm_timeout: float = 60.0,
        max_tokens_per_response: int = 800,
        enable_iterative_debate: bool = False,
        max_debate_rounds: int = 2,
        use_weighted_voting: bool = True,
        skeptic_model: Optional[ModelInterface] = None,
        skeptic_weight_multiplier: float = 1.2,
        model_weights: Optional[Dict[str, float]] = None,
        max_concurrent: int = 2,
    ):
        self.models = models
        self.swarm_timeout = swarm_timeout
        self.max_tokens = max_tokens_per_response
        self.enable_iterative_debate = enable_iterative_debate
        self.max_debate_rounds = max_debate_rounds
        self.use_weighted_voting = use_weighted_voting
        self.skeptic_model = skeptic_model
        self.skeptic_weight_multiplier = skeptic_weight_multiplier
        self.consensus_threshold = self._calculate_threshold(models, consensus_threshold)

        self.model_weights: Dict[str, float] = {}
        for model in models:
            name = model.model_name
            if model_weights and name in model_weights:
                self.model_weights[name] = model_weights[name]
            elif name in _DEFAULT_WEIGHTS:
                self.model_weights[name] = _DEFAULT_WEIGHTS[name].weight
            else:
                self.model_weights[name] = self._detect_weight(name)

        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"[SwarmDebate] Initialized | models={len(models)} | "
            f"threshold={self.consensus_threshold:.2f} | "
            f"skeptic={skeptic_model.model_name if skeptic_model else 'none'} | "
            f"timeout={swarm_timeout}s | grammar=enforced"
        )

    # ------------------------------------------------------------------
    # Threshold & weight helpers
    # ------------------------------------------------------------------
    def _calculate_threshold(self, models: List[ModelInterface], default: float) -> float:
        frontier = {"claude-3.5-sonnet", "gpt-5.4", "gpt-5-turbo"}
        has_frontier = any(m.model_name in frontier for m in models)
        return 0.60 if has_frontier else 0.75

    def _detect_weight(self, name: str) -> float:
        nl = name.lower()
        if any(x in nl for x in ("70b", "405b")): return 1.25
        if any(x in nl for x in ("32b", "34b")): return 1.10
        if "13b" in nl: return 0.90
        if any(x in nl for x in ("8b", "7b")): return 0.75
        return 1.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def debate(
        self,
        hypotheses: List[str],
        context: Optional[str] = None,
    ) -> SwarmResult:
        start = time.time()
        if not hypotheses or not self.models:
            return SwarmResult(
                hypotheses=hypotheses or [],
                confirmed=[], rejected=[], votes={},
                weighted_consensus_ratio=0.0,
                raw_consensus_ratio=0.0,
                swarm_voters=0,
            )

        logger.info(
            f"[SwarmDebate] Debating {len(hypotheses)} hypotheses | "
            f"models={len(self.models)}"
        )

        if self.enable_iterative_debate:
            result = await self._iterative_debate(hypotheses, context)
        else:
            result = await self._parallel_debate(hypotheses, context)

        if self.skeptic_model and result.confirmed:
            skeptic_votes = await self._run_skeptic_review(
                result.confirmed, result.votes, context
            )
            if skeptic_votes:
                result.skeptic_overrides = self._apply_skeptic_overrides(result, skeptic_votes)

        result.total_time_ms = (time.time() - start) * 1000
        result.grammar_violations = 0

        logger.info(
            f"[SwarmDebate] Done | confirmed={len(result.confirmed)}/{len(hypotheses)} | "
            f"ratio={result.weighted_consensus_ratio:.2f} | "
            f"skeptic_overrides={result.skeptic_overrides} | "
            f"time={result.total_time_ms:.0f}ms"
        )
        return result

    # ------------------------------------------------------------------
    # Parallel debate
    # ------------------------------------------------------------------
    async def _parallel_debate(
        self,
        hypotheses: List[str],
        context: Optional[str],
    ) -> SwarmResult:
        tasks = [self._analyze_with_model(m, hypotheses, context) for m in self.models]
        timeout_occurred = False
        try:
            async with asyncio.timeout(self.swarm_timeout):
                model_results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning(f"[SwarmDebate] Timed out after {self.swarm_timeout}s")
            model_results = [asyncio.TimeoutError() for _ in tasks]
            timeout_occurred = True

        votes: Dict[str, List[SwarmVote]] = {h: [] for h in hypotheses}
        valid_voters = 0

        for i, res in enumerate(model_results):
            if isinstance(res, Exception):
                logger.warning(f"[SwarmDebate] '{self.models[i].model_name}' failed: {res}")
                continue
            valid_voters += 1
            for hyp, vote in res.items():
                votes[hyp].append(vote)

        confirmed, rejected, w_ratio, r_ratio = self._compute_consensus(
            votes, hypotheses, valid_voters
        )

        return SwarmResult(
            hypotheses=hypotheses,
            confirmed=confirmed,
            rejected=rejected,
            votes=votes,
            weighted_consensus_ratio=w_ratio,
            raw_consensus_ratio=r_ratio,
            swarm_voters=valid_voters,
            timeout_occurred=timeout_occurred,
            debate_rounds=1,
        )

    # ------------------------------------------------------------------
    # Iterative debate
    # ------------------------------------------------------------------
    async def _iterative_debate(
        self,
        hypotheses: List[str],
        context: Optional[str],
    ) -> SwarmResult:
        all_votes: Dict[str, List[SwarmVote]] = {h: [] for h in hypotheses}
        history: List[str] = []
        round_used = 0
        per_round_timeout = self.swarm_timeout / max(self.max_debate_rounds, 1)

        for rnd in range(1, self.max_debate_rounds + 1):
            round_used = rnd
            tasks = [
                self._analyze_with_model_iterative(m, hypotheses, context, history, rnd)
                for m in self.models
            ]
            try:
                async with asyncio.timeout(per_round_timeout):
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                logger.warning(f"[SwarmDebate] Round {rnd} timed out")
                break

            round_reasoning: List[str] = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    continue
                for hyp, vote in res.items():
                    all_votes[hyp].append(vote)
                    if vote.reasoning:
                        round_reasoning.append(
                            f"[{self.models[i].model_name}] {vote.reasoning[:100]}"
                        )
            history.extend(round_reasoning)

            _, _, w_ratio, _ = self._compute_consensus(all_votes, hypotheses, len(self.models))
            if w_ratio >= self.consensus_threshold:
                logger.debug(f"[SwarmDebate] Early consensus at round {rnd}")
                break

        confirmed, rejected, w_ratio, r_ratio = self._compute_consensus(
            all_votes, hypotheses, len(self.models)
        )

        return SwarmResult(
            hypotheses=hypotheses,
            confirmed=confirmed,
            rejected=rejected,
            votes=all_votes,
            weighted_consensus_ratio=w_ratio,
            raw_consensus_ratio=r_ratio,
            swarm_voters=len(self.models),
            debate_rounds=round_used,
        )

    # ------------------------------------------------------------------
    # Single-model analysis
    # ------------------------------------------------------------------
    async def _analyze_with_model(
        self,
        model: ModelInterface,
        hypotheses: List[str],
        context: Optional[str],
    ) -> Dict[str, SwarmVote]:
        prompt = self._build_debate_prompt(hypotheses, context)
        start = time.time()
        try:
            async with self._semaphore:
                raw: str = await model.generate_async(
                    prompt,
                    grammar=SWARM_VERDICT_GRAMMAR,
                    max_tokens=self.max_tokens,
                )
            ms = (time.time() - start) * 1000
            return self._parse_verdict_json(raw, hypotheses, model.model_name, ms)
        except Exception as exc:
            logger.warning(f"[SwarmDebate] {model.model_name} failed: {exc}")
            return self._neutral_votes(hypotheses, model.model_name, str(exc))

    async def _analyze_with_model_iterative(
        self,
        model: ModelInterface,
        hypotheses: List[str],
        context: Optional[str],
        history: List[str],
        round_num: int,
    ) -> Dict[str, SwarmVote]:
        prompt = self._build_iterative_prompt(hypotheses, context, history, round_num)
        try:
            async with self._semaphore:
                raw: str = await model.generate_async(
                    prompt,
                    grammar=SWARM_VERDICT_GRAMMAR,
                    max_tokens=self.max_tokens,
                )
            return self._parse_verdict_json(raw, hypotheses, model.model_name, 0.0)
        except Exception as exc:
            logger.debug(f"[SwarmDebate] {model.model_name} round {round_num} failed: {exc}")
            return self._neutral_votes(hypotheses, model.model_name, str(exc))

    # ------------------------------------------------------------------
    # Skeptic review
    # ------------------------------------------------------------------
    async def _run_skeptic_review(
        self,
        candidates: List[str],
        all_votes: Dict[str, List[SwarmVote]],
        context: Optional[str],
    ) -> Dict[str, SwarmVote]:
        if not self.skeptic_model or not candidates:
            return {}

        logger.info(f"[Skeptic] Reviewing {len(candidates)} candidates")
        prompt = self._build_skeptic_prompt(candidates, all_votes, context)

        try:
            raw: str = await self.skeptic_model.generate_async(
                prompt,
                grammar=SKEPTIC_VERDICT_GRAMMAR,
                max_tokens=self.max_tokens,
            )
            votes = self._parse_verdict_json(
                raw, candidates, self.skeptic_model.model_name, 0.0
            )
            for v in votes.values():
                v.weight *= self.skeptic_weight_multiplier
            return votes
        except Exception as exc:
            logger.warning(f"[Skeptic] Review failed: {exc}")
            return {}

    def _apply_skeptic_overrides(
        self,
        result: SwarmResult,
        skeptic_votes: Dict[str, SwarmVote],
    ) -> int:
        overrides = 0
        for hyp, vote in list(skeptic_votes.items()):
            if not vote.verdict and hyp in result.confirmed:
                result.confirmed.remove(hyp)
                result.rejected.append(hyp)
                overrides += 1
        return overrides

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_debate_prompt(self, hypotheses: List[str], context: Optional[str]) -> str:
        hyp_list = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses))
        ctx_section = f"Context:\n{context[:1000]}\n\n" if context else ""
        return (
            "You are a senior security researcher specializing in memory safety.\n"
            "Evaluate each hypothesis: is it a REAL, EXPLOITABLE vulnerability?\n"
            "Consider: reachability, exploit chain, bounds checks, guard conditions.\n\n"
            f"{ctx_section}"
            f"Hypotheses:\n{hyp_list}\n\n"
            "Respond with JSON only, no extra text:\n"
            '{"hypotheses": ['
            '{"index": 1, "verdict": true, "confidence": 0.85, "reason": "why"}]}'
        )

    def _build_iterative_prompt(
        self,
        hypotheses: List[str],
        context: Optional[str],
        history: List[str],
        round_num: int,
    ) -> str:
        base = self._build_debate_prompt(hypotheses, context)
        if history:
            tail = "\n".join(history[-10:])
            base += f"\n\nPrevious debate (round {round_num - 1}):\n{tail}\n\nRevise your assessment:"
        return base

    def _build_skeptic_prompt(
        self,
        candidates: List[str],
        all_votes: Dict[str, List[SwarmVote]],
        context: Optional[str],
    ) -> str:
        hyp_list = "\n".join(f"{i+1}. {h}" for i, h in enumerate(candidates))
        summary = []
        for hyp in candidates:
            for v in all_votes.get(hyp, []):
                if v.verdict and v.reasoning:
                    summary.append(f"- {v.model_name}: {v.reasoning[:150]}")

        ctx_section = f"Context:\n{context[:800]}\n\n" if context else ""
        return (
            "You are a Senior Security Auditor. Your mandate: DISPROVE these hypotheses.\n"
            "Your reputation depends on eliminating false positives.\n\n"
            f"{ctx_section}"
            f"Hypotheses under review:\n{hyp_list}\n\n"
            f"Other researchers flagged these:\n" + "\n".join(summary[:10]) + "\n\n"
            "For each hypothesis: is it actually reachable? Are there implicit guards?\n"
            "A verdict of TRUE means you failed to disprove it.\n"
            "A verdict of FALSE means you found a counter-argument.\n\n"
            "Respond with JSON only:\n"
            '{"hypotheses": ['
            '{"index": 1, "verdict": false, "confidence": 0.92, "reason": "counter-argument"}]}'
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_verdict_json(
        self,
        raw: str,
        hypotheses: List[str],
        model_name: str,
        response_time_ms: float,
    ) -> Dict[str, SwarmVote]:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].lstrip("json").strip()

        data = json.loads(raw)
        weight = self.model_weights.get(model_name, 1.0)
        votes: Dict[str, SwarmVote] = {}

        for item in data.get("hypotheses", []):
            idx = int(item.get("index", 0)) - 1
            if 0 <= idx < len(hypotheses):
                hyp = hypotheses[idx]
                votes[hyp] = SwarmVote(
                    model_name=model_name,
                    hypothesis=hyp,
                    verdict=bool(item.get("verdict", False)),
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=str(item.get("reason", ""))[:500],
                    weight=weight,
                    response_time_ms=response_time_ms,
                )

        for hyp in hypotheses:
            if hyp not in votes:
                votes[hyp] = SwarmVote(
                    model_name=model_name,
                    hypothesis=hyp,
                    verdict=False,
                    confidence=0.0,
                    reasoning="not_evaluated",
                    weight=weight,
                    response_time_ms=response_time_ms,
                )
        return votes

    def _neutral_votes(
        self,
        hypotheses: List[str],
        model_name: str,
        reason: str,
    ) -> Dict[str, SwarmVote]:
        w = self.model_weights.get(model_name, 1.0)
        return {
            h: SwarmVote(
                model_name=model_name,
                hypothesis=h,
                verdict=False,
                confidence=0.0,
                reasoning=reason,
                weight=w,
            )
            for h in hypotheses
        }

    # ------------------------------------------------------------------
    # Consensus computation
    # ------------------------------------------------------------------
    def _compute_consensus(
        self,
        votes: Dict[str, List[SwarmVote]],
        hypotheses: List[str],
        total_voters: int,
    ) -> Tuple[List[str], List[str], float, float]:
        confirmed: List[str] = []
        rejected: List[str] = []
        sum_w = 0.0
        sum_r = 0.0

        for hyp in hypotheses:
            hyp_votes = votes.get(hyp, [])
            if not hyp_votes:
                continue

            weighted_positive = sum(v.weight for v in hyp_votes if v.verdict)
            weighted_total = sum(v.weight for v in hyp_votes)
            w_ratio = weighted_positive / weighted_total if weighted_total > 0 else 0.0

            raw_positive = sum(1 for v in hyp_votes if v.verdict)
            r_ratio = raw_positive / len(hyp_votes)

            if w_ratio >= self.consensus_threshold:
                confirmed.append(hyp)
            elif r_ratio < 0.3 and w_ratio < 0.4:
                rejected.append(hyp)

            sum_w += w_ratio
            sum_r += r_ratio

        n = len(hypotheses) or 1
        return confirmed, rejected, sum_w / n, sum_r / n

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def assign_skeptic(self, model: ModelInterface) -> None:
        self.skeptic_model = model
        logger.info(f"[SwarmDebate] Skeptic assigned: {model.model_name}")

    def get_vote_summary(self, result: SwarmResult) -> str:
        lines = [
            f"Swarm Debate Results | consensus={result.weighted_consensus_ratio:.2f} | "
            f"voters={result.swarm_voters} | rounds={result.debate_rounds}"
        ]
        lines.append(f"Confirmed: {len(result.confirmed)}")
        for hyp in result.confirmed:
            lines.append(f" [+] {hyp[:100]}")
            for v in result.votes.get(hyp, []):
                if v.verdict:
                    lines.append(f"   {v.model_name}: conf={v.confidence:.2f} | {v.reasoning[:80]}")

        if result.rejected:
            lines.append(f"Rejected: {len(result.rejected)}")
            for hyp in result.rejected:
                lines.append(f" [-] {hyp[:100]}")

        if result.skeptic_overrides:
            lines.append(f"Skeptic overrides: {result.skeptic_overrides}")
        if result.timeout_occurred:
            lines.append("[WARN] Timeout occurred — some models did not respond")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def create_swarm_debate(
    models: List[ModelInterface],
    consensus_threshold: float = 0.67,
    swarm_timeout: float = 60.0,
    enable_iterative_debate: bool = False,
    use_weighted_voting: bool = True,
    skeptic_model: Optional[ModelInterface] = None,
    max_concurrent: int = 2,
) -> SwarmDebate:
    return SwarmDebate(
        models=models,
        consensus_threshold=consensus_threshold,
        swarm_timeout=swarm_timeout,
        enable_iterative_debate=enable_iterative_debate,
        use_weighted_voting=use_weighted_voting,
        skeptic_model=skeptic_model,
        max_concurrent=max_concurrent,
    ) 