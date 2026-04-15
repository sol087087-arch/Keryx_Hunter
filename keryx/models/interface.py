# keryx/models/interface.py
# Abstract base interface for ALL model backends in KeryxHunter.
# Local (llama.cpp, MLX, etc.) and remote (Anthropic, OpenAI, Groq, etc.)

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    TypedDict,
)

logger = logging.getLogger("keryx.models")


@dataclass
class GenerationConfig:
    """Structured generation parameters."""
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.05
    top_k: int = 0
    repeat_penalty: float = 1.1
    stop: list[str] | None = None
    grammar: str | None = None
    seed: int | None = None
    logits_post_processor: Callable | None = field(default=None, repr=False)


@dataclass
class GenerationResult:
    """Result of a model generation call."""
    text: str
    tokens_input: int
    tokens_output: int
    duration_ms: float
    time_to_first_token_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]


class ToolDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class CostEstimate(TypedDict):
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class ModelCapabilities(TypedDict):
    max_context_length: int
    supports_tool_calling: bool
    supports_grammar: bool
    supports_batching: bool
    supports_streaming: bool
    supports_speculative: bool
    supports_min_p: bool
    requires_gpu: bool
    is_local: bool
    is_quantized: bool


class ModelInterface(ABC):
    """Core contract that all models (local and remote) must satisfy."""

    model_name: str
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        pass  # pragma: no cover

    @abstractmethod
    def generate_result(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        pass  # pragma: no cover

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        pass  # pragma: no cover

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: GenerationConfig | None = None,
    ) -> str | ToolCall:
        pass  # pragma: no cover

    async def generate_async(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Default async implementation via thread pool."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens),
        )

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        pass  # pragma: no cover

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    @abstractmethod
    def get_context_length(self) -> int:
        pass  # pragma: no cover

    def get_context_used(self) -> int:
        return 0

    def clear_context(self, keep_tokens: int = 0) -> None:  # noqa: B027
        """No-op default — subclasses override if context windowing is supported."""

    @abstractmethod
    def is_healthy(self) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        pass  # pragma: no cover

    @abstractmethod
    def get_usage_cost(self) -> CostEstimate:
        pass  # pragma: no cover

    def reset_cost_tracking(self) -> None:  # noqa: B027
        """No-op default — subclasses override if cost accumulation is tracked."""

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        pass  # pragma: no cover

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.get_capabilities()

    @property
    def is_local(self) -> bool:
        override = self.__dict__.get("_is_local")
        if override is not None:
            return override
        return self.capabilities.get("is_local", True)

    @is_local.setter
    def is_local(self, value: bool) -> None:
        self.__dict__["_is_local"] = value

    @property
    def requires_network(self) -> bool:
        return not self.is_local

    async def load(self) -> None:  # noqa: B027
        """No-op default — subclasses override for async model loading."""

    @abstractmethod
    def unload(self) -> None:
        pass  # pragma: no cover

    def get_metrics(self) -> dict[str, Any]:
        return {}


# Exceptions
class ModelError(Exception):
    pass


class ModelLoadError(ModelError):
    pass


class GenerationError(ModelError):
    pass


class BudgetExceededError(ModelError):
    pass


class ContextOverflowError(ModelError):
    pass


class ToolCallError(ModelError):
    pass
