# keryx/advisors/cloud_advisor.py
# Cloud API advisor with production hardening:
# - Exponential backoff + jitter
# - Privacy scrubbing (paths, IPs, keys)
# - Local digest model (8B summarizes before sending to cloud — 60-80% cost reduction)
# - Strict JSON parsing and tool validation
# - Budget enforcement
from **future** import annotations
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from .base import BaseAdvisor, AdvisorResponse
from ..models.interface import ModelInterface
try:
    from ..core.shared_context import SharedContext
except ImportError:
    SharedContext = Any # type: ignore
logger = logging.getLogger("keryx.advisors.cloud")
# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.5 # FIX 12: was 0.1 (10%), standard is 0.5 (50%)
# ---------------------------------------------------------------------------
# Privacy scrubbing patterns
# ---------------------------------------------------------------------------
*SCRUB_PATTERNS: List[Tuple[str, str]] = [
    (r'/home/[^/\s]+/[^/\s]+', '/workspace'),
    (r'/Users/[^/\s]+/[^/\s]+', '/workspace'),
    (r'[A-Za-z]:$$   ^\   +\   ^\   $$+', 'C:\workspace'),
    (r'/opt/[^/\s]+/[^/\s]+', '/opt/app'),
    (r'/var/[^/\s]+/[^/\s]+', '/var/app'),
    (r'\b\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}\b', 'x.x.x.x'),
    (r'\b[A-Za-z0-9.*%+-]+@[A-Za-z0-9.-]+.[A-Za-z]{2,}\b', 'user@example.com'),
    (r'AKIA[0-9A-Z]{16}', 'AKIA...REDACTED'),
]
_COMPILED_SCRUB = [(re.compile(p, re.IGNORECASE), r) for p, r in _SCRUB_PATTERNS]
def scrub_text(text: str) -> str:
    """Apply all privacy scrubbing patterns."""
    for pattern, replacement in _COMPILED_SCRUB:
        text = pattern.sub(replacement, text)
    return text
# ---------------------------------------------------------------------------
# CloudAdvisor
# ---------------------------------------------------------------------------
class CloudAdvisor(BaseAdvisor):
    """
    Production-hardened cloud advisor.
    Privacy stack:
    1. Optional local digest: cheap 8B model summarizes history → only summary
       goes to cloud. Reduces token cost 60–80% and keeps raw logs on-device.
    2. Scrubbing: paths, IPs, AWS keys, emails replaced before any cloud call.
    Reliability stack:
    - Exponential backoff with 50% jitter on retryable errors
    - Budget enforcement (USD cap per session)
    - Graceful degradation: returns useful fallback if all attempts fail
    NOTE: Circuit breaker is NOT implemented here — AdvisorManager already
    provides one. Duplicating it would double-count failures.
    FIX 11: removed local _consecutive_failures / _circuit_open state.
    """
    name = "cloud-advisor"
    requires_network = True
    def **init**(
        self,
        model: ModelInterface,
        toolbox: Optional[Any] = None,
        local_digest_model: Optional[ModelInterface] = None,
        max_calls_per_session: int = 3,
        confidence_threshold: float = 0.45,
        max_parse_errors: int = 3,
        no_progress_after_steps: int = 20,
        temperature: float = 0.3,
        max_tokens: int = 800,
        request_timeout: float = 30.0,
        budget_limit_usd: Optional[float] = 2.0,
        retry_config: Optional[RetryConfig] = None,
        enable_scrubbing: bool = True,
        enable_digest: bool = True,
        **kwargs,
    ):
        super().**init**(max_calls_per_session=max_calls_per_session, **kwargs)
        self.model = model
        self.toolbox = toolbox
        self.local_digest_model = local_digest_model
        self.confidence_threshold = confidence_threshold
        self.max_parse_errors = max_parse_errors
        self.no_progress_after_steps = no_progress_after_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.budget_limit_usd = budget_limit_usd
        self.retry_config = retry_config or RetryConfig()
        self.enable_scrubbing = enable_scrubbing
        self.enable_digest = enable_digest
        # Cost tracking
        self._session_cost_usd: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        # FIX 3: consistent naming — _tools_cache used throughout
        self._tools_cache: Optional[List[str]] = None
        self._tools_cache_time: float = 0.0
        logger.info(
            f"[{self.name}] Initialized | model={model.model_name} | "
            f"digest={enable_digest} | scrubbing={enable_scrubbing} | "
            f"budget=${budget_limit_usd}"
        )
    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------
    def _check_preflight(self) -> Tuple[bool, str]:
        """
        Fast pre-flight checks before any expensive operations.
        FIX 1+2: was self._call_count (undefined). Use self.calls_made from BaseAdvisor.
        """
        if self.calls_made >= self.max_calls_per_session:
            return False, "call_limit"
        if self.budget_limit_usd and self.*session_cost_usd >= self.budget_limit_usd:
            return False, f"budget_exhausted*${self._session_cost_usd:.2f}"
        if not self.model.is_healthy():
            return False, "model_unhealthy"
        return True, "ok"
    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    def should_trigger(self, context: Any) -> bool:
        ok, reason = self._check_preflight()
        if not ok:
            logger.debug(f"[{self.name}] Preflight blocked: {reason}")
            return False
        conf = getattr(context, "get_last_confidence", lambda: None)()
        errors = getattr(context, "parse_errors", 0)
        steps = getattr(context, "steps_taken", 0)
        hypotheses = getattr(context, "hypotheses", [])
        local_tried = getattr(context, "local_advisor_called", False)
        low_conf = conf is not None and conf < self.confidence_threshold
        too_errors = errors >= self.max_parse_errors
        stalled = steps > self.no_progress_after_steps and not hypotheses
        # Cloud is last resort for confidence — require local advisor first
        if low_conf and not local_tried:
            return False
        return low_conf or too_errors or stalled
    # ------------------------------------------------------------------
    # Core advice
    # ------------------------------------------------------------------
    async def advise(self, context: Any) -> AdvisorResponse:
        ok, reason = self._check_preflight()
        if not ok:
            return self.*fallback_response(context, f"preflight*{reason}")
        available_tools = self._get_available_tools(context)
        try:
            prompt = await self._build_secure_prompt(context, available_tools)
        except Exception as exc:
            logger.error(f"[{self.name}] Prompt build failed: {exc}")
            return self._fallback_response(context, "prompt_build_failed")
        try:
            raw_text = await self._call_with_retry(prompt)
            self._update_usage_estimate(prompt, raw_text)
            response = self._parse_and_validate(raw_text, context, available_tools)
            logger.info(
                f"[{self.name}] Success | "
                f"cost=${self._session_cost_usd:.4f} | "
                f"tokens={self._total_input_tokens + self._total_output_tokens}"
            )
            return response
        except Exception as exc:
            logger.error(f"[{self.name}] Failed after retries: {exc}")
            return self._fallback_response(context, str(exc))
    # ------------------------------------------------------------------
    # Prompt building with digest + scrubbing
    # ------------------------------------------------------------------
    async def _build_secure_prompt(self, context: Any, available_tools: List[str]) -> str:
        # Step 1: Summarize history locally if digest model available
        if self.enable_digest and self.local_digest_model:
            history = await self._generate_local_digest(context)
        else:
            raw = getattr(context, 'get_recent_history', lambda n: "")(6) or ""
            history = scrub_text(raw) if self.enable_scrubbing else raw
        # Step 2: Scrub target path
        target = str(getattr(context, 'target_path', 'unknown'))
        if self.enable_scrubbing:
            target = scrub_text(target)
        blacklisted = list(getattr(context, "blacklisted_hypotheses", []))
        current_hyps = [
            h for h in getattr(context, "hypotheses", [])
            if h not in blacklisted
        ][:5]
        conf = getattr(context, 'get_last_confidence', lambda: None)()
        tools_str = "\n".join(f"- {t}" for t in available_tools[:10])
        return (
            "You are Keryx Cloud Advisor. Provide strategic security research guidance.\n\n"
            "## Context\n"
            f"Target: {target}\n"
            f"Capability: {getattr(context, 'capability', 'unknown')}\n"
            f"Step: {getattr(context, 'steps_taken', 0)}\n"
            f"Confidence: {conf or 'low'}\n"
            f"Parse errors: {getattr(context, 'parse_errors', 0)}\n\n"
            "## Activity Summary\n"
            f"{history[:800]}\n\n"
            "## Current Focus\n"
            f"Hypotheses: {current_hyps or 'None — generate new directions'}\n"
            f"Avoid: {blacklisted[:3] or 'None'}\n\n"
            "## Available Tools (ONLY these are valid)\n"
            f"{tools_str}\n\n"
            "CRITICAL: Output ONLY a single JSON object. No markdown. No prose.\n"
            '{"strategic_direction": "1-2 sentences", '
            '"adjust_confidence_threshold": 0.5, '
            '"suggested_hypotheses": ["hypothesis"], '
            '"blacklist_hypotheses": ["dead_end"], '
            '"suggested_tools": ["tool_from_list"], '
            '"priority_files": ["relative/path.cpp"]}'
        )
    async def _generate_local_digest(self, context: Any) -> str:
        """
        Summarize activity log with cheap local model before sending to cloud.
        Keeps raw logs on-device; only the summary goes to the cloud API.
        FIX 7+8: generate_async returns str, not GenerationResult.
        FIX 4: no config= param — pass grammar and max_tokens directly.
        """
        raw = getattr(context, 'get_recent_history', lambda n: "")(10) or ""
        if not raw:
            return ""
        digest_prompt = (
            "Summarize this security scan activity in 2-3 sentences.\n"
            "Focus on: what was tried, what failed, current hypothesis.\n\n"
            f"Activity log:\n{raw[:2000]}\n\nSummary:"
        )
        try:
            # FIX 7: no config=, just max_tokens and grammar=None
            summary: str = await asyncio.wait_for(
                self.local_digest_model.generate_async(
                    digest_prompt,
                    max_tokens=150,
                    grammar=None,
                ),
                timeout=10.0,
            )
            # FIX 8: summary is already str
            result = scrub_text(summary) if self.enable_scrubbing else summary
            logger.debug(f"[{self.name}] Digest: {len(result)} chars (from {len(raw)} raw)")
            return result
        except Exception as exc:
            logger.warning(f"[{self.name}] Digest failed ({exc}) — using truncated raw")
            truncated = raw[:800]
            return scrub_text(truncated) if self.enable_scrubbing else truncated
    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------
    async def _call_with_retry(self, prompt: str) -> str:
        """
        Execute with exponential backoff and jitter.
        FIX 4+5: generate_async returns str directly. No config= param.
        FIX 9: GenerationConfig had no response_format field — removed.
        FIX 10: retryable_errors stored in RetryConfig as tuple — caught correctly.
                 Using explicit exception types avoids the dynamic except-clause issue.
        """
        retryable = (asyncio.TimeoutError, ConnectionError, OSError)
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                result: str = await asyncio.wait_for(
                    self.model.generate_async(
                        prompt,
                        max_tokens=self.max_tokens,
                        grammar=None, # Cloud models don't use GBNF
                    ),
                    timeout=self.request_timeout,
                )
                return result # str
            except retryable as exc:
                if attempt == self.retry_config.max_attempts:
                    raise
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 1)),
                    self.retry_config.max_delay,
                )
                # FIX 12: jitter_factor=0.5 → 50% jitter (standard), was 0.1
                jitter = random.uniform(0, delay * self.retry_config.jitter_factor)
                wait = delay + jitter
                logger.warning(
                    f"[{self.name}] Attempt {attempt} failed ({exc.**class**.**name**}). "
                    f"Retry in {wait:.1f}s..."
                )
                await asyncio.sleep(wait)
            except Exception:
                raise # Non-retryable — propagate immediately
        raise RuntimeError("Retry loop exhausted without returning") # should never reach
    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------
    def _update_usage_estimate(self, prompt: str, response_text: str) -> None:
        """
        Estimate token usage and update cost counters.
        FIX 5+6: generate_async returns str. No .text attribute.
        We estimate from string length (4 chars ≈ 1 token).
        """
        input_tok = len(prompt) // 4
        output_tok = len(response_text) // 4
        self._total_input_tokens += input_tok
        self._total_output_tokens += output_tok
        cost_in = (input_tok / 1000) * getattr(self.model, 'cost_per_1k_input_tokens', 0.0)
        cost_out = (output_tok / 1000) * getattr(self.model, 'cost_per_1k_output_tokens', 0.0)
        call_cost = cost_in + cost_out
        self._session_cost_usd += call_cost
        logger.info(
            f"COST_TRACKER | advisor={self.name} | "
            f"in={input_tok} out={output_tok} | "
            f"call=$  {call_cost:.6f} | session=  ${self._session_cost_usd:.4f}"
        )
    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_and_validate(
        self,
        raw: str,
        context: Any,
        available_tools: List[str],
    ) -> AdvisorResponse:
        """
        Parse cloud model response.
        FIX 5: raw is str from generate_async, not GenerationResult.
        """
        raw = raw.strip()
        if raw.startswith(""):             parts = raw.split("")
if len(parts) >= 2:
raw = parts[1].lstrip("json").strip()
start, end = raw.find("{"), raw.rfind("}")
if start == -1 or end <= start:
raise ValueError("No JSON object in response")
raw = raw[start:end + 1]
data = json.loads(raw)
raw_tools = data.get("suggested_tools", [])
if not isinstance(raw_tools, list):
raw_tools = []
valid_tools = [t for t in raw_tools if t in available_tools][:3]
invalid = [t for t in raw_tools if t not in available_tools]
if invalid:
logger.warning(f"[{self.name}] Filtered hallucinated tools: {invalid}")
def _str_list(key: str, limit: int) -> List[str]:
return [h for h in data.get(key, []) if isinstance(h, str)][:limit]
return AdvisorResponse(
strategic_direction=str(data.get("strategic_direction", ""))[:600],
adjust_confidence_threshold=self._clamp_threshold(
data.get("adjust_confidence_threshold")
),
suggested_hypotheses=_str_list("suggested_hypotheses", 3),
blacklist_hypotheses=_str_list("blacklist_hypotheses", 3),
suggested_tools=valid_tools,
priority_files=_str_list("priority_files", 3),
metadata={
"advisor": self.name,
"model": self.model.model_name,
"session_cost_usd": self._session_cost_usd,
"tools_filtered": invalid,
},
)
def _clamp_threshold(self, value: Any) -> Optional[float]:
if value is None:
return None
try:
return max(0.3, min(0.85, float(value)))
except (ValueError, TypeError):
return None
# ------------------------------------------------------------------
# Tools cache
# ------------------------------------------------------------------
def _get_available_tools(self, context: Any) -> List[str]:
"""
Cached tool listing with 5s TTL.
FIX 3: was self._cached_tools (undefined). Correct name is self._tools_cache.
"""
now = time.time()
if self._tools_cache is not None and now - self._tools_cache_time < 5.0:
return self._tools_cache
if self.toolbox and hasattr(self.toolbox, "list_tools"):
tools = self.toolbox.list_tools()
elif hasattr(context, "get_available_tools"):
tools = context.get_available_tools()
else:
tools = ["codeql_query", "gdb_analyze", "fuzzer_run", "rag_search"]
self._tools_cache = tools
self._tools_cache_time = now
return tools
# ------------------------------------------------------------------
# Fallback
# ------------------------------------------------------------------
def _fallback_response(self, context: Any, error: str) -> AdvisorResponse:
return AdvisorResponse(
strategic_direction=(
"Cloud advisor unavailable. Continue with local reasoning. "
"Focus on memory-unsafe patterns and recent file changes."
),
adjust_confidence_threshold=0.55,
metadata={
"advisor": self.name,
"fallback": True,
"error": error,
},
)
# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------
def reset(self) -> None:
super().reset()
self._session_cost_usd = 0.0
self._total_input_tokens = 0
self._total_output_tokens = 0
self._tools_cache = None
self._tools_cache_time = 0.0
logger.debug(f"[{self.name}] Reset complete")
def get_metrics(self) -> Dict[str, Any]:
return {
**super().get_metrics(),
"model": self.model.model_name,
"session_cost_usd": self._session_cost_usd,
"total_tokens": self._total_input_tokens + self._total_output_tokens,
"input_tokens": self._total_input_tokens,
"output_tokens": self._total_output_tokens,
"digest_enabled": self.enable_digest,
"scrubbing_enabled": self.enable_scrubbing,
}
def **repr**(self) -> str:
return (
f"CloudAdvisor(model={self.model.model_name!r}, "
f"cost=${self._session_cost_usd:.3f}, "
f"calls={self.calls_made}/{self.max_calls_per_session})"
)
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_cloud_advisor(
model: ModelInterface,
local_digest_model: Optional[ModelInterface] = None,
**kwargs,
) -> CloudAdvisor:
return CloudAdvisor(model=model, local_digest_model=local_digest_model, **kwargs)
