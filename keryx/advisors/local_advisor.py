# keryx/advisors/local_advisor.py
# Local large-model advisor: dynamic grammar, tool validation, blacklist tracking.
# Sovereign, air-gapped, zero network.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from ..models.interface import ModelInterface
from .base import AdvisorResponse, BaseAdvisor

try:
    from ..core.shared_context import SharedContext
except ImportError:
    SharedContext = Any  # type: ignore

logger = logging.getLogger("keryx.advisors.local")


# ---------------------------------------------------------------------------
# GBNF grammar builder
# ---------------------------------------------------------------------------
_GRAMMAR_BASE = r"""
root ::= "{" ws kv-strat "," ws kv-adj "," ws kv-sug "," ws kv-black "," ws kv-tools "," ws kv-prio ws "}"
kv-strat ::= "\"strategic_direction\"" ws ":" ws string
kv-adj ::= "\"adjust_confidence_threshold\"" ws ":" ws (number | "null")
kv-sug ::= "\"suggested_hypotheses\"" ws ":" ws string-array
kv-black ::= "\"blacklist_hypotheses\"" ws ":" ws string-array
kv-tools ::= "\"suggested_tools\"" ws ":" ws TOOL_RULE
kv-prio ::= "\"priority_files\"" ws ":" ws string-array
string-array ::= "[" ws ( string ( "," ws string )* )? ws "]"
string ::= "\"" ( [^"\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
number ::= "-"? [0-9]+ ( "." [0-9]+ )?
ws ::= [ \t\n\r]*
"""

_FALLBACK_TOOL_RULE = 'tool-array ::= "[" ws "]"\n'


def build_dynamic_grammar(available_tools: list[str]) -> str:
    """
    Build GBNF with strict tool enumeration.
    Prevents model from hallucinating tool names that don't exist.
    """
    if not available_tools:
        tool_rule = 'tool-array ::= "[" ws "]"'
    else:
        alts = " | ".join(f'"{t}"' for t in available_tools)
        tool_rule = f'tool-array ::= "[" ws ( ( {alts} ) ( "," ws ( {alts} ) )* )? ws "]"'

    return _GRAMMAR_BASE.replace("TOOL_RULE", "tool-array") + tool_rule + "\n"


# ---------------------------------------------------------------------------
# Context cache key
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContextCacheKey:
    """Immutable key for KV cache deduplication between advisor calls."""
    target_path: str
    capability: str
    history_hash: str
    tool_outputs_hash: str

    @classmethod
    def from_context(cls, context: Any, history_depth: int = 5) -> ContextCacheKey:
        steps = list(getattr(context, 'steps', []))[-history_depth:]
        step_parts = []
        for s in steps:
            if hasattr(s, 'action'):
                step_parts.append(f"{getattr(s, 'action', '')}:{getattr(s, 'observation', '')[:80]}")
            elif isinstance(s, dict):
                step_parts.append(json.dumps(s, sort_keys=True, default=str)[:100])
            else:
                step_parts.append(str(s)[:80])

        history_hash = hashlib.sha256("\n".join(step_parts).encode()).hexdigest()[:16]
        tools_used = list(getattr(context, 'tools_used', []))
        tools_hash = hashlib.sha256(json.dumps(sorted(tools_used)).encode()).hexdigest()[:16]

        return cls(
            target_path=str(getattr(context, 'target_path', '')),
            capability=str(getattr(context, 'capability', '')),
            history_hash=history_hash,
            tool_outputs_hash=tools_hash,
        )


# ---------------------------------------------------------------------------
# LocalAdvisor
# ---------------------------------------------------------------------------
class LocalAdvisor(BaseAdvisor):
    """
    Advisor backed by a local large model.
    Features:
    - Dynamic GBNF grammar: only registered tools are grammatically valid
    - Blacklist tracking: already-tried hypotheses and dead tools are excluded
    - Context-aware fallback
    - Deduplication of repeated advice
    """
    name = "local-advisor"
    requires_network = False

    def __init__(
        self,
        model: ModelInterface,
        toolbox: Any | None = None,
        max_calls_per_session: int = 4,
        confidence_threshold: float = 0.55,
        max_parse_errors: int = 4,
        no_progress_after_steps: int = 18,
        max_context_tokens: int = 6000,
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(max_calls_per_session=max_calls_per_session, **kwargs)

        if not getattr(model, "is_local", False):
            logger.warning(f"[{self.name}] Model '{model.model_name}' is not local")

        self.model = model
        self.toolbox = toolbox
        self.confidence_threshold = confidence_threshold
        self.max_parse_errors = max_parse_errors
        self.no_progress_after_steps = no_progress_after_steps
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature

        # State
        self._last_advice_hash: int | None = None
        self._suggested_dead_ends: set[str] = set()
        self._last_cache_key: ContextCacheKey | None = None

        logger.info(
            f"[{self.name}] Initialized | model={model.model_name} | "
            f"temp={temperature} | max_calls={max_calls_per_session}"
        )

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    def should_trigger(self, context: Any) -> bool:
        conf = getattr(context, 'get_last_confidence', lambda: None)()
        errors = getattr(context, 'parse_errors', 0)
        steps = getattr(context, 'steps_taken', 0)
        hypotheses = list(getattr(context, 'hypotheses', []))
        blacklisted = set(getattr(context, 'blacklisted_hypotheses', []))

        novel_hyps = [h for h in hypotheses if h not in blacklisted]

        low_conf = conf is not None and conf < self.confidence_threshold
        too_errors = errors >= self.max_parse_errors
        stalled = steps > self.no_progress_after_steps and not novel_hyps

        triggered = low_conf or too_errors or stalled

        if triggered:
            logger.debug(
                f"[{self.name}] Trigger | conf={conf} low={low_conf} "
                f"errors={too_errors} stalled={stalled}"
            )
        return triggered

    # ------------------------------------------------------------------
    # Advice
    # ------------------------------------------------------------------
    async def advise(self, context: Any) -> AdvisorResponse:
        if not self.model.is_healthy():
            return self._error_response("Model unhealthy")

        available_tools = self._get_available_tools(context)
        grammar = build_dynamic_grammar(available_tools)
        prompt = self._build_prompt(context, available_tools)
        prompt = self._truncate_to_tokens(prompt, self.max_context_tokens)

        cache_key = ContextCacheKey.from_context(context)
        if cache_key == self._last_cache_key:
            logger.debug(f"[{self.name}] Context unchanged since last call")
        self._last_cache_key = cache_key

        try:
            raw: str = await asyncio.wait_for(
                self.model.generate_async(
                    prompt,
                    grammar=grammar,
                    max_tokens=1000,
                ),
                timeout=45.0,
            )

            response = self._parse_and_validate(raw, context, available_tools)
            self._persist_blacklists(context, response)

            # Deduplication warning
            advice_hash = hash(response.strategic_direction[:100])
            if advice_hash == self._last_advice_hash and not response.is_empty():
                response.strategic_direction += " [same advice repeated — try a different approach]"

            self._last_advice_hash = advice_hash

            logger.info(
                f"[{self.name}] Advice generated | "
                f"direction={len(response.strategic_direction)}c | "
                f"tools={response.suggested_tools}"
            )
            return response

        except TimeoutError:
            logger.error(f"[{self.name}] Generation timeout")
            return self._fallback_response(context, "timeout")
        except Exception as exc:
            logger.error(f"[{self.name}] Failed: {exc}", exc_info=True)
            return self._fallback_response(context, str(exc))

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------
    def _get_available_tools(self, context: Any) -> list[str]:
        if self.toolbox is not None and hasattr(self.toolbox, 'list_tools'):
            return self.toolbox.list_tools()
        if hasattr(context, 'get_available_tools'):
            return context.get_available_tools()
        return ["codeql_query", "gdb_analyze", "fuzzer_run", "rag_search"]

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(self, context: Any, available_tools: list[str]) -> str:
        blacklisted_hyps = list(getattr(context, 'blacklisted_hypotheses', []))
        failed_tools = list(getattr(context, 'failed_tools', []))
        all_blacklisted = set(blacklisted_hyps) | self._suggested_dead_ends

        recent = ""
        if hasattr(context, 'get_recent_history'):
            recent = context.get_recent_history(6) or ""

        tool_outputs = self._extract_tool_outputs(context, max_chars=800)

        current_hyps = [
            h for h in getattr(context, 'hypotheses', [])
            if h not in all_blacklisted
        ][:5]

        tools_str = "\n".join(f"- {t}" for t in available_tools)
        conf = getattr(context, 'get_last_confidence', lambda: None)()

        return (
            "You are Keryx Local Advisor — senior security researcher.\n\n"
            "## Current State\n"
            f"Target: {getattr(context, 'target_path', 'unknown')}\n"
            f"Capability: {getattr(context, 'capability', 'unknown')}\n"
            f"Step: {getattr(context, 'steps_taken', 0)}\n"
            f"Confidence: {conf or 'N/A'}\n"
            f"Parse errors: {getattr(context, 'parse_errors', 0)}\n\n"
            "## FORBIDDEN (do not suggest)\n"
            f"Hypotheses: {list(all_blacklisted)[:5] or 'none'}\n"
            f"Failed tools: {failed_tools[:3] or 'none'}\n\n"
            "## Recent Activity\n"
            f"{recent[:1200]}\n\n"
            "## Tool Outputs\n"
            f"{tool_outputs}\n\n"
            "## Active Hypotheses\n"
            f"{current_hyps or 'None — generate new directions'}\n\n"
            "## Available Tools (ONLY these are valid)\n"
            f"{tools_str}\n\n"
            "Output ONLY valid JSON with these fields:\n"
            '{"strategic_direction": "1-2 sentences", '
            '"adjust_confidence_threshold": 0.5, '
            '"suggested_hypotheses": ["..."], '
            '"blacklist_hypotheses": ["..."], '
            '"suggested_tools": ["codeql_query"], '
            '"priority_files": ["path/to/file.cpp"]}'
        )

    def _extract_tool_outputs(self, context: Any, max_chars: int) -> str:
        steps = list(getattr(context, 'steps', []))[-5:]
        outputs = []
        budget = max_chars
        for step in reversed(steps):
            obs = ""
            if hasattr(step, 'observation'):
                obs = str(step.observation)
            elif isinstance(step, dict):
                obs = str(step.get('observation', ''))
            if not obs:
                continue
            is_failure = any(kw in obs.lower() for kw in ('error', 'fail', 'timeout', 'crash'))
            prefix = "[FAIL] " if is_failure else "[OK] "
            snippet = (prefix + obs)[: budget // 2]
            outputs.append(snippet)
            budget -= len(snippet)
            if budget <= 0:
                break
        return "\n".join(reversed(outputs)) or "No tool outputs yet."

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        half = max_chars // 2 - 50
        return text[:half] + "\n...[context truncated]...\n" + text[-half:]

    # ------------------------------------------------------------------
    # Parsing + validation
    # ------------------------------------------------------------------
    def _parse_and_validate(
        self,
        raw: str,
        context: Any,
        available_tools: list[str],
    ) -> AdvisorResponse:
        raw = raw.strip()

        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].lstrip("json").strip()

        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in response")

        raw = raw[start:end + 1]
        data = json.loads(raw)

        raw_tools = data.get("suggested_tools", [])
        valid_tools = [t for t in raw_tools if t in available_tools]
        invalid_tools = [t for t in raw_tools if t not in available_tools]

        if invalid_tools:
            logger.warning(f"[{self.name}] Filtered hallucinated tools: {invalid_tools}")

        blacklisted = set(getattr(context, 'blacklisted_hypotheses', []))
        suggestions = [
            h for h in data.get("suggested_hypotheses", [])
            if h not in blacklisted
        ][:3]

        return AdvisorResponse(
            strategic_direction=str(data.get("strategic_direction", ""))[:500],
            adjust_confidence_threshold=self._clamp_threshold(
                data.get("adjust_confidence_threshold")
            ),
            suggested_hypotheses=suggestions,
            blacklist_hypotheses=data.get("blacklist_hypotheses", [])[:3],
            suggested_tools=valid_tools[:2],
            priority_files=data.get("priority_files", [])[:3],
            metadata={
                "advisor": self.name,
                "model": self.model.model_name,
                "invalid_tools_filtered": invalid_tools,
            },
        )

    def _clamp_threshold(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return max(0.3, min(0.8, float(value)))
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Blacklist persistence
    # ------------------------------------------------------------------
    def _persist_blacklists(self, context: Any, response: AdvisorResponse) -> None:
        for hyp in response.blacklist_hypotheses:
            self._suggested_dead_ends.add(hyp)
            if hasattr(context, 'blacklist_hypothesis'):
                context.blacklist_hypothesis(hyp)
            elif hasattr(context, 'add_blacklisted_hypothesis'):
                context.add_blacklisted_hypothesis(hyp)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def _fallback_response(self, context: Any, error: str) -> AdvisorResponse:
        errors = getattr(context, 'parse_errors', 0)
        hyps = getattr(context, 'hypotheses', [])

        if errors > 3:
            direction = "Simplify actions. Use explicit parameters. Avoid nested tool calls."
        elif not hyps:
            direction = "Switch to rag_search with CVE patterns. Try different file directories."
        else:
            direction = "Continue current strategy with reduced confidence threshold."

        return AdvisorResponse(
            strategic_direction=direction,
            metadata={"advisor": self.name, "fallback": True, "error": error},
        )

    def _error_response(self, error: str) -> AdvisorResponse:
        return AdvisorResponse(
            strategic_direction=f"Advisor unavailable: {error}. Continue with executor defaults.",
            metadata={"advisor": self.name, "error": error},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._last_advice_hash = None
        self._suggested_dead_ends = set()
        self._last_cache_key = None

    def get_metrics(self) -> dict[str, Any]:
        return {
            **super().get_metrics(),
            "model": self.model.model_name,
            "temperature": self.temperature,
            "dead_ends_tracked": len(self._suggested_dead_ends),
        }

    def __repr__(self) -> str:
        return f"LocalAdvisor(model={self.model.model_name!r}, calls={self.calls_made})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_local_advisor(
    model: ModelInterface,
    toolbox: Any | None = None,
    **kwargs,
) -> LocalAdvisor:
    return LocalAdvisor(model=model, toolbox=toolbox, **kwargs)
