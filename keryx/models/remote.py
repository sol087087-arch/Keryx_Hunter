# keryx/models/interface.py
# Abstract base interface for ALL model backends in KeryxHunter.
# Local (llama.cpp, MLX, TensorRT) and remote (Anthropic, OpenAI, Groq) share this contract.
# Sovereign, air-gapped, async-first.
from **future** import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, AsyncIterator, Callable, Dict, List, Optional, Union
)
# TypedDict is stdlib since 3.8
from typing import TypedDict
logger = logging.getLogger("keryx.models")
# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """
    Structured generation parameters — works for both local and cloud models.
    Fields not supported by a backend are silently ignored.
    """
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.05 # effective for local models; reduces repetition
    top_k: int = 0
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    grammar: Optional[str] = None # GBNF string for llama.cpp
    seed: Optional[int] = None
    # Callable excluded from serialization and comparison
    logits_post_processor: Optional[Callable] = field(default=None, repr=False, compare=False)
    def to_dict(self) -> Dict[str, Any]:
        """Safe serialization — excludes non-serializable Callable."""
        return {
            k: v for k, v in self.**dict**.items()
            if k != "logits_post_processor"
        }
    @classmethod
    def default(cls) -> "GenerationConfig":
        return cls()
    @classmethod
    def for_critique(cls) -> "GenerationConfig":
        """Short, focused output for self-critique steps."""
        return cls(max_tokens=350, temperature=0.3, grammar=None)
    @classmethod
    def for_structured(cls, grammar: str) -> "GenerationConfig":
        """Grammar-constrained JSON output."""
        return cls(max_tokens=1000, temperature=0.1, grammar=grammar)
# ---------------------------------------------------------------------------
# Result types — ONE definition, used everywhere
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """
    Full generation result with metrics.
    time_to_first_token_ms enables streaming UI progress bars.
    FIX: previous local.py used GenerationResult with tokens_generated.
    Canonical field names are tokens_input / tokens_output here.
    local.py updated to match.
    """
    text: str
    tokens_input: int
    tokens_output: int
    duration_ms: float
    time_to_first_token_ms: float = 0.0
    finish_reason: str = "stop" # stop | length | tool_call
    metadata: Dict[str, Any] = field(default_factory=dict)
    @property
    def tokens_per_second(self) -> float:
        return self.tokens_output / (self.duration_ms / 1000) if self.duration_ms > 0 else 0.0
# ---------------------------------------------------------------------------
# Tool types — ONE definition
# ---------------------------------------------------------------------------
class ToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]
class ToolDefinition(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any] # JSON Schema object
# ---------------------------------------------------------------------------
# Cost types
# ---------------------------------------------------------------------------
class CostEstimate(TypedDict):
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
# ---------------------------------------------------------------------------
# Capability descriptor — single source of truth for router
# ---------------------------------------------------------------------------
class ModelCapabilities(TypedDict):
    max_context_length: int
    supports_tool_calling: bool
    supports_grammar: bool # GBNF / constrained decoding
    supports_batching: bool
    supports_streaming: bool
    supports_speculative: bool # speculative decoding with draft model
    supports_min_p: bool
    requires_gpu: bool
    is_local: bool # False = requires network (cloud API)
    is_quantized: bool
# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------
class ModelInterface(ABC):
    """
    The single contract for every model backend in KeryxHunter.
    Design decisions:
    - generate() is SYNC. Callers that need async wrap it in run_in_executor.
      Reason: llama-cpp-python is a C extension — wrapping with async def
      create_task() does nothing; the GIL blocks anyway. Be honest about this.
    - generate_stream() is a SYNC Iterator. Async streaming is handled by
      generate_stream_async() which uses Queue + call_soon_threadsafe.
    - Tools are passed AT CALL TIME, not registered on the model.
      Models are stateless; tool state lives in the agent/toolbox.
    - tokenize() is mandatory — agents need exact counts before sending.
    - get_capabilities() is the single source of truth for the router.
    """
    # ------------------------------------------------------------------
    # Identity — as a plain attribute, not abstract property.
    # Abstract property conflicts with setting self.model_name in **init**.
    # Subclasses set: self.model_name = "llama-70b"
    # ------------------------------------------------------------------
    model_name: str # must be set in **init** of every subclass
    # ------------------------------------------------------------------
    # Sync generation (primary path)
    # ------------------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None, # convenience override
        max_tokens: Optional[int] = None, # convenience override
    ) -> str:
        """
        Synchronous generation. Returns plain text.
        Convenience kwargs (grammar, max_tokens) override config fields
        so callers don't need to build a full GenerationConfig for simple calls.
        This matches how agent.py and orchestrator.py call the model.
        """
    @abstractmethod
    def generate_result(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Synchronous generation with full metrics.
        Use when you need token counts for budget tracking.
        """
    # ------------------------------------------------------------------
    # Streaming (sync iterator — see local.py generate_stream_async for async)
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> "Iterator[str]":
        """
        Sync streaming iterator.
        FIX: 'async def ... -> AsyncIterator' is ambiguous and wrong.
        A sync Iterator is the correct return type for a C-backed model.
        For async streaming, use generate_stream_async() in the concrete class.
        """
    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, ToolCall]:
        """
        Generate with tool calling. Model decides: text or tool invocation.
        Tools are passed here, not stored on the model.
        Returns str for text response, ToolCall for tool invocation.
        """
    # ------------------------------------------------------------------
    # Async wrappers (default impl — concrete classes may override)
    # ------------------------------------------------------------------
    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        grammar: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Async wrapper around sync generate(). Override for true async backends."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens),
        )
    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Native tokenization — mandatory.
        Agents call this BEFORE sending to ensure prompt fits in context window.
        """
    def count_tokens(self, text: str) -> int:
        """Convenience: just the count."""
        return len(self.tokenize(text))
    @abstractmethod
    def get_context_length(self) -> int:
        """Maximum context window in tokens."""
    def get_context_used(self) -> int:
        """Current tokens in KV cache. 0 if not tracked."""
        return 0
    def get_context_remaining(self) -> int:
        return self.get_context_length() - self.get_context_used()
    def clear_context(self, keep_tokens: int = 0) -> None:
        """
        Clear KV cache.
        keep_tokens: preserve this many tokens at start (system prompt prefix).
        """
    # ------------------------------------------------------------------
    # Health — required by router.py
    # ------------------------------------------------------------------
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Lightweight check that the model is loaded and responsive.
        Called by CapabilityRouter before committing to a routing plan.
        Should be cheap: tokenize a short string, not a full generation.
        """
    # ------------------------------------------------------------------
    # Cost — required by BudgetController in agent.py
    # ------------------------------------------------------------------
    # These two attributes must be set in **init** of every subclass.
    # BudgetController accesses them via getattr(model, "cost_per_1k_input_tokens", 0.0).
    cost_per_1k_input_tokens: float # 0.0 for local models
    cost_per_1k_output_tokens: float # 0.0 for local models
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        """Estimate cost for a call before making it."""
    @abstractmethod
    def get_usage_cost(self) -> CostEstimate:
        """Total accumulated cost since last reset_cost_tracking()."""
    def reset_cost_tracking(self) -> None:
        """Reset accumulated cost counters."""
    # ------------------------------------------------------------------
    # Capabilities — single source of truth for router
    # ------------------------------------------------------------------
    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """
        Everything the router needs to make routing decisions.
        Called once at routing time and cached by the router.
        """
    @property
    def capabilities(self) -> ModelCapabilities:
        return self.get_capabilities()
    # Convenience shorthands derived from capabilities
    @property
    def is_local(self) -> bool:
        return self.capabilities.get("is_local", True)
    @property
    def requires_network(self) -> bool:
        return not self.is_local
    @property
    def requires_gpu(self) -> bool:
        return self.capabilities.get("requires_gpu", False)
    @property
    def supports_grammar(self) -> bool:
        return self.capabilities.get("supports_grammar", False)
    @property
    def supports_speculative(self) -> bool:
        return self.capabilities.get("supports_speculative", False)
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def load(self) -> None:
        """
        Async model loading — for backends where loading is I/O bound
        (downloading weights, waiting for GPU memory allocation).
        Default: no-op (most local models load in **init**).
        """
    @abstractmethod
    def unload(self) -> None:
        """Release all resources: GPU memory, file handles, etc."""
    async def warmup(self, prompt: str = "ping") -> None:
        """
        Async warmup — run a trivial generation to prime KV cache and GPU.
        Default impl calls generate_async with minimal config.
        """
        try:
            await self.generate_async(prompt, max_tokens=1)
        except Exception as exc:
            logger.debug(f"[{self.**class**.**name**}] Warmup skipped: {exc}")
    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Generation metrics: total calls, tokens, latency, errors."""
        return {}
    def get_memory_usage(self) -> Dict[str, float]:
        """RAM and GPU memory in GB."""
        return {}
    def get_version_info(self) -> Dict[str, str]:
        return {
            "backend": self.**class**.**name**,
            "model_name": getattr(self, "model_name", "unknown"),
        }
    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def **repr**(self) -> str:
        caps = self.get_capabilities()
        return (
            f"{self.**class**.**name**}("
            f"name={getattr(self, 'model_name', '?')!r}, "
            f"local={caps.get('is_local')}, "
            f"ctx={caps.get('max_context_length')})"
        )
# ---------------------------------------------------------------------------
# Exceptions — full hierarchy
# ---------------------------------------------------------------------------
class ModelError(Exception):
    """Base for all model errors."""
class ModelLoadError(ModelError):
    """Failed to load model weights or initialize backend."""
class GenerationError(ModelError):
    """Generation failed (OOM, timeout, grammar violation, etc.)."""
class BudgetExceededError(ModelError):
    """Cloud API budget limit reached."""
class ContextOverflowError(ModelError):
    """Prompt exceeds model context window."""
class ToolCallError(ModelError):
    """Model returned malformed tool call."""
# ---------------------------------------------------------------------------
# Type alias for convenience imports
# ---------------------------------------------------------------------------
from typing import Iterator # noqa: E402 — needed for generate_stream signature
