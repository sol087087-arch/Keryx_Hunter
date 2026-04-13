from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterator, List, Optional, Union

from keryx.models.interface import (
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelInterface,
    ToolCall,
    ToolDefinition,
)


class MyLocalModel(ModelInterface):
    """Minimal mock implementation for tests and development."""

    def __init__(self) -> None:
        self.model_name = "my-local-llama-3"
        self.cost_per_1k_input_tokens = 0.0
        self.cost_per_1k_output_tokens = 0.0

    # ── Primary sync path ────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return "Analysis: Potential buffer overflow in handle_packet..."

    def generate_result(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        text = self.generate(prompt, config)
        return GenerationResult(
            text=text,
            tokens_input=120,
            tokens_output=35,
            duration_ms=245.0,
            finish_reason="stop",
            metadata={"source": "local_mock"},
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        yield self.generate(prompt, config)

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, ToolCall]:
        # Для мока пока возвращаем обычный текст
        return self.generate(prompt, config)

    # ── Async path ───────────────────────────────────────────────────────
    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Default implementation via base class is fine, but we override for clarity."""
        return self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens)

    # ── Context ──────────────────────────────────────────────────────────
    def tokenize(self, text: str) -> List[int]:
        # Rough approximation: ~4 chars per token
        return [0] * max(1, len(text) // 4)

    def get_context_length(self) -> int:
        return 32768

    # ── Health ───────────────────────────────────────────────────────────
    def is_healthy(self) -> bool:
        return True

    # ── Cost ─────────────────────────────────────────────────────────────
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        return CostEstimate(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
        )

    def get_usage_cost(self) -> CostEstimate:
        return CostEstimate(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
        )

    # ── Capabilities ─────────────────────────────────────────────────────
    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_context_length=32768,
            supports_tool_calling=False,
            supports_grammar=True,
            supports_batching=False,
            supports_streaming=False,
            supports_speculative=False,
            supports_min_p=False,
            requires_gpu=False,
            is_local=True,
            is_quantized=False,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────
    def unload(self) -> None:
        pass