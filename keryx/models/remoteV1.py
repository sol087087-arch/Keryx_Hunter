# keryx/models/remote.py
# RemoteModel — cloud LLM backend for KeryxHunter.
# Implements ModelInterface for Anthropic, OpenAI, Groq, DeepSeek.
# Production-hardened: tiktoken token counting, privacy scrubbing,
# async context manager, exponential backoff + Retry-After,
# precise budget checks BEFORE the call.

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

from .interface import (
    BudgetExceededError,
    ContextOverflowError,
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelError,
    ModelInterface,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger("keryx.models.remote")


# ---------------------------------------------------------------------------
# Per-provider context lengths and costs (per 1k tokens)
# ---------------------------------------------------------------------------

_PROVIDER_MODELS: dict[str, dict[str, Any]] = {
    # Anthropic
    "claude-opus-4.6":           {"ctx": 200_000, "in": 15.00,  "out": 75.00},
    "claude-sonnet-4.6":         {"ctx": 200_000, "in":  3.00,  "out": 15.00},
    "claude-haiku-4.6":          {"ctx": 200_000, "in":  0.80,  "out":  4.00},
    # OpenAI
    "gpt-5.4":                    {"ctx": 128_000, "in":  2.50,  "out": 10.00},
    "gpt-5.4-mini":               {"ctx": 128_000, "in":  0.15,  "out":  0.60},
    "gpt-5-turbo":               {"ctx": 128_000, "in": 10.00,  "out": 30.00},
    "o3-mini":                   {"ctx": 200_000, "in":  1.10,  "out":  4.40},
    # Groq (free tier, approximate)
    "llama-4-70b-versatile":   {"ctx":  32_768, "in":  0.59,  "out":  0.79},
    "llama-3.1-8b-instant":      {"ctx": 131_072, "in":  0.05,  "out":  0.08},
    "mixtral-8x7b-32768":        {"ctx":  32_768, "in":  0.24,  "out":  0.24},
    # DeepSeek
    "deepseek-v4":             {"ctx":  64_000, "in":  0.14,  "out":  0.28},
    "deepseek-r1":            {"ctx":  16_000, "in":  0.14,  "out":  0.28},
}

_DEFAULT_CTX = 32_768


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RemoteModelConfig:
    provider:    str              # anthropic | openai | groq | deepseek
    api_key:     str
    model_name:  str
    base_url:    str | None   = None
    max_retries: int             = 3
    timeout:     float           = 60.0
    max_budget_usd: float | None = None


# ---------------------------------------------------------------------------
# Privacy scrubbing (self-contained)
# ---------------------------------------------------------------------------

_SCRUB_PATS = [
    (re.compile(r'/home/[^/\s]+',    re.I), '/workspace'),
    (re.compile(r'/Users/[^/\s]+',   re.I), '/workspace'),
    (re.compile(r'C:\\Users\\[^\\\s]+', re.I), 'C:\\workspace'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'sk-[A-Za-z0-9]{40,}'), '[API_KEY]'),
    (re.compile(r'AKIA[0-9A-Z]{16}'), '[AWS_KEY]'),
]


def _scrub(text: str) -> str:
    for pat, repl in _SCRUB_PATS:
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# RemoteModel
# ---------------------------------------------------------------------------

class RemoteModel(ModelInterface):
    """
    Remote cloud model backend (Anthropic, OpenAI-compatible, Groq, DeepSeek).

    Async-primary: use generate_async() directly from async code.
    Sync generate() is available for agent.py compatibility but runs
    the coroutine on a dedicated background thread loop.

    generate_async() accepts (prompt, grammar=None, max_tokens=None) kwargs
    so agent.py can call it the same way as LocalModel.
    Grammar is silently ignored — cloud models don't support GBNF.
    """

    def __init__(self, config: RemoteModelConfig) -> None:
        self.config     = config
        self.model_name = config.model_name
        self.is_local   = False

        spec = _PROVIDER_MODELS.get(config.model_name, {})
        self.cost_per_1k_input_tokens  = spec.get("in",  0.0)
        self.cost_per_1k_output_tokens = spec.get("out", 0.0)
        self._ctx_length               = spec.get("ctx", _DEFAULT_CTX)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        self._session_cost_usd    = 0.0
        self._total_input_tokens  = 0
        self._total_output_tokens = 0
        self._call_count          = 0
        self._healthy             = True

        self._tokenizer = self._get_tokenizer(config.model_name)

        # Provider routing
        if config.provider == "anthropic":
            self._base_url = config.base_url or "https://api.anthropic.com/v1"
            self._headers  = {
                "x-api-key":         config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            }
        else:
            self._base_url = config.base_url or "https://api.openai.com/v1"
            self._headers  = {
                "Authorization": f"Bearer {config.api_key}",
                "content-type":  "application/json",
            }

        # Background sync loop (FIX 2)
        self._sync_loop:   asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None           = None
        self._loop_ready   = threading.Event()

        logger.info(
            f"[RemoteModel] {config.provider}/{config.model_name} | "
            f"ctx={self._ctx_length:,} | budget=${config.max_budget_usd}"
        )

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------

    def _get_tokenizer(self, model_name: str) -> Any:
        if not _TIKTOKEN_AVAILABLE:
            return None
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def tokenize(self, text: str) -> list[int]:
        """FIX 6: mandatory abstract method."""
        if self._tokenizer:
            return self._tokenizer.encode(text)
        return list(range(len(text) // 3))   # rough fallback

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _check_budget(self, estimated_cost: float = 0.0) -> None:
        if self.config.max_budget_usd is not None and self._session_cost_usd + estimated_cost > self.config.max_budget_usd:
            raise BudgetExceededError(
                f"Budget ${self.config.max_budget_usd:.2f} would be exceeded "
                f"(used: ${self._session_cost_usd:.4f}, est: ${estimated_cost:.4f})"
            )

    def _update_cost(self, input_t: int, output_t: int) -> None:
        cost = (
            input_t  * self.cost_per_1k_input_tokens  / 1000 +
            output_t * self.cost_per_1k_output_tokens / 1000
        )
        self._session_cost_usd    += cost
        self._total_input_tokens  += input_t
        self._total_output_tokens += output_t

    # ------------------------------------------------------------------
    # Core async generation
    # FIX 4: accept grammar= and max_tokens= kwargs (ignored/overridden)
    # ------------------------------------------------------------------

    async def generate_async(
        self,
        prompt:     str,
        config:     GenerationConfig | None = None,
        *,
        grammar:    str | None = None,    # GBNF — silently ignored for cloud
        max_tokens: int | None = None,
    ) -> str:
        """
        FIX 4+5: returns str (not GenerationResult) to match agent.py expectations.
        grammar is accepted but ignored — cloud APIs don't support GBNF.
        """
        cfg = config or GenerationConfig()
        if max_tokens is not None:
            cfg = GenerationConfig(
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                stop=cfg.stop,
            )

        scrubbed = _scrub(prompt)
        input_tokens = self._count_tokens(scrubbed)

        if input_tokens > self._ctx_length:
            raise ContextOverflowError(
                f"Prompt {input_tokens} tokens exceeds context {self._ctx_length}"
            )

        estimated = (
            input_tokens * self.cost_per_1k_input_tokens / 1000 +
            cfg.max_tokens * self.cost_per_1k_output_tokens / 1000
        )
        self._check_budget(estimated)
        self._call_count += 1

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if self.config.provider == "anthropic":
                    payload = self._build_anthropic_payload(scrubbed, cfg)
                    url     = f"{self._base_url}/messages"
                else:
                    payload = self._build_openai_payload(scrubbed, cfg)
                    url     = f"{self._base_url}/chat/completions"

                response = await self._client.post(url, json=payload, headers=self._headers)
                response.raise_for_status()
                data     = response.json()
                text, in_t, out_t = self._parse_response(data)
                self._update_cost(in_t, out_t)
                self._healthy = True
                return text

            except httpx.HTTPStatusError as exc:
                last_error = exc
                status     = exc.response.status_code
                if status == 429:
                    wait = float(exc.response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"[RemoteModel] Rate limited — retry in {wait}s")
                    await asyncio.sleep(wait)
                elif status in (500, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ModelError(f"HTTP {status}: {exc.response.text[:200]}") from exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        self._healthy = False
        raise ModelError(f"Failed after {self.config.max_retries + 1} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Sync generate — FIX 2: dedicated background loop (no asyncio.run)
    # FIX 5: returns str
    # ------------------------------------------------------------------

    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        if self._sync_loop and not self._sync_loop.is_closed():
            return self._sync_loop
        self._loop_ready.clear()
        loop = asyncio.new_event_loop()
        self._sync_loop = loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            self._loop_ready.set()
            loop.run_forever()

        self._sync_thread = threading.Thread(target=_run, daemon=True, name="remote-loop")
        self._sync_thread.start()
        self._loop_ready.wait(timeout=3.0)
        return loop

    def generate(
        self,
        prompt:     str,
        config:     GenerationConfig | None = None,
        *,
        grammar:    str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Sync wrapper — safe from any thread context."""
        loop   = self._ensure_sync_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.generate_async(prompt, config, grammar=grammar, max_tokens=max_tokens),
            loop,
        )
        return future.result(timeout=self.config.timeout + 10.0)

    def generate_result(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Returns full GenerationResult with token counts for budget tracking."""
        start = time.time()
        text  = self.generate(prompt, config)
        return GenerationResult(
            text=text,
            tokens_input=self._count_tokens(prompt),
            tokens_output=self._count_tokens(text),
            duration_ms=(time.time() - start) * 1000,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Sync streaming — yields full response as single chunk (no true streaming for remote)."""
        yield self.generate(prompt, config)

    # ------------------------------------------------------------------
    # Tool calling — FIX 3: Anthropic branch added
    # ------------------------------------------------------------------

    async def generate_with_tools(
        self,
        prompt: str,
        tools:  list[ToolDefinition],
        config: GenerationConfig | None = None,
    ) -> str | ToolCall:
        cfg      = config or GenerationConfig()
        scrubbed = _scrub(prompt)
        self._check_budget()

        if self.config.provider == "anthropic":
            return await self._generate_with_tools_anthropic(scrubbed, tools, cfg)
        return await self._generate_with_tools_openai(scrubbed, tools, cfg)

    async def _generate_with_tools_anthropic(
        self,
        prompt: str,
        tools:  list[ToolDefinition],
        cfg:    GenerationConfig,
    ) -> str | ToolCall:
        anth_tools = [
            {
                "name":        t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        payload = {
            "model":      self.model_name,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "tools":      anth_tools,
            "messages":   [{"role": "user", "content": prompt}],
        }
        response = await self._client.post(
            f"{self._base_url}/messages", json=payload, headers=self._headers
        )
        response.raise_for_status()
        data = response.json()
        self._update_cost(
            data["usage"]["input_tokens"],
            data["usage"]["output_tokens"],
        )
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                return ToolCall(name=block["name"], arguments=block.get("input", {}))
        text = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"), ""
        )
        return text

    async def _generate_with_tools_openai(
        self,
        prompt: str,
        tools:  list[ToolDefinition],
        cfg:    GenerationConfig,
    ) -> str | ToolCall:
        payload = {
            "model":       self.model_name,
            "max_tokens":  cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages":    [{"role": "user", "content": prompt}],
            "tools":       tools,
            "tool_choice": "auto",
        }
        response = await self._client.post(
            f"{self._base_url}/chat/completions", json=payload, headers=self._headers
        )
        response.raise_for_status()
        data    = response.json()
        usage   = data.get("usage", {})
        self._update_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        message = data["choices"][0]["message"]
        if message.get("tool_calls"):
            tc = message["tool_calls"][0]
            return ToolCall(
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"]),
            )
        return message.get("content") or ""

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _build_anthropic_payload(self, prompt: str, cfg: GenerationConfig) -> dict:
        return {
            "model":       self.model_name,
            "max_tokens":  cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages":    [{"role": "user", "content": prompt}],
        }

    def _build_openai_payload(self, prompt: str, cfg: GenerationConfig) -> dict:
        return {
            "model":       self.model_name,
            "max_tokens":  cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages":    [{"role": "user", "content": prompt}],
        }

    def _parse_response(self, data: dict) -> tuple[str, int, int]:
        if self.config.provider == "anthropic":
            text    = data["content"][0]["text"]
            input_t = data["usage"]["input_tokens"]
            output_t = data["usage"]["output_tokens"]
        else:
            text    = data["choices"][0]["message"].get("content") or ""
            input_t = data["usage"]["prompt_tokens"]
            output_t = data["usage"]["completion_tokens"]
        return text, input_t, output_t

    # ------------------------------------------------------------------
    # ModelInterface abstract methods — FIX 6,7,8,9
    # ------------------------------------------------------------------

    def get_context_length(self) -> int:
        return self._ctx_length

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_context_length=self._ctx_length,
            supports_tool_calling=True,
            supports_grammar=False,       # cloud APIs don't support GBNF
            supports_batching=False,
            supports_streaming=False,     # not implemented
            supports_speculative=False,
            supports_min_p=False,
            requires_gpu=False,
            is_local=False,
            is_quantized=False,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        in_cost  = input_tokens  * self.cost_per_1k_input_tokens  / 1000
        out_cost = output_tokens * self.cost_per_1k_output_tokens / 1000
        return CostEstimate(
            input_cost_usd=in_cost,
            output_cost_usd=out_cost,
            total_cost_usd=in_cost + out_cost,
        )

    def get_usage_cost(self) -> CostEstimate:
        return CostEstimate(
            input_cost_usd=self._total_input_tokens  * self.cost_per_1k_input_tokens  / 1000,
            output_cost_usd=self._total_output_tokens * self.cost_per_1k_output_tokens / 1000,
            total_cost_usd=self._session_cost_usd,
        )

    def reset_cost_tracking(self) -> None:
        self._session_cost_usd    = 0.0
        self._total_input_tokens  = 0
        self._total_output_tokens = 0

    def is_healthy(self) -> bool:
        return self._healthy

    # ------------------------------------------------------------------
    # Metrics and lifecycle
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        return {
            "provider":         self.config.provider,
            "model":            self.model_name,
            "calls":            self._call_count,
            "input_tokens":     self._total_input_tokens,
            "output_tokens":    self._total_output_tokens,
            "session_cost_usd": round(self._session_cost_usd, 6),
            "budget_limit_usd": self.config.max_budget_usd,
            "healthy":          self._healthy,
        }

    async def load(self) -> None:
        pass

    def unload(self) -> None:
        """Close async HTTP client via sync wrapper."""
        if self._sync_loop and not self._sync_loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._client.aclose(), self._sync_loop
            )
            with contextlib.suppress(Exception):
                future.result(timeout=5.0)
        if self._sync_loop and not self._sync_loop.is_closed():
            self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)

    @asynccontextmanager
    async def session(self):
        try:
            yield self
        finally:
            await self._client.aclose()

    def __repr__(self) -> str:
        return (
            f"RemoteModel(provider={self.config.provider!r}, "
            f"model={self.model_name!r}, "
            f"cost=${self._session_cost_usd:.4f})"
        )


# ---------------------------------------------------------------------------
# Factories — FIX 10: cost table uses direct per-1k values, no double division
# FIX 11: updated model names
# ---------------------------------------------------------------------------

def create_anthropic_model(
    api_key:        str,
    model:          str            = "claude-sonnet-4.6",
    max_budget_usd: float | None = None,
) -> RemoteModel:
    spec = _PROVIDER_MODELS.get(model, {"ctx": 200_000, "in": 3.0, "out": 15.0})
    cfg  = RemoteModelConfig(
        provider="anthropic",
        api_key=api_key,
        model_name=model,
        max_budget_usd=max_budget_usd,
    )
    m = RemoteModel(cfg)
    m.cost_per_1k_input_tokens  = spec["in"]
    m.cost_per_1k_output_tokens = spec["out"]
    return m


def create_openai_model(
    api_key:        str,
    model:          str            = "gpt-5.4",
    max_budget_usd: float | None = None,
) -> RemoteModel:
    spec = _PROVIDER_MODELS.get(model, {"ctx": 128_000, "in": 2.5, "out": 10.0})
    cfg  = RemoteModelConfig(
        provider="openai",
        api_key=api_key,
        model_name=model,
        max_budget_usd=max_budget_usd,
    )
    m = RemoteModel(cfg)
    m.cost_per_1k_input_tokens  = spec["in"]
    m.cost_per_1k_output_tokens = spec["out"]
    return m


def create_groq_model(
    api_key:        str,
    model:          str            = "llama-4-70b-versatile",
    max_budget_usd: float | None = None,
) -> RemoteModel:
    spec = _PROVIDER_MODELS.get(model, {"ctx": 32_768, "in": 0.59, "out": 0.79})
    cfg  = RemoteModelConfig(
        provider="groq",
        api_key=api_key,
        model_name=model,
        base_url="https://api.groq.com/openai/v1",
        max_budget_usd=max_budget_usd,
    )
    m = RemoteModel(cfg)
    m.cost_per_1k_input_tokens  = spec["in"]
    m.cost_per_1k_output_tokens = spec["out"]
    return m


def create_deepseek_model(
    api_key:        str,
    model:          str            = "deepseek-v4",
    max_budget_usd: float | None = None,
) -> RemoteModel:
    spec = _PROVIDER_MODELS.get(model, {"ctx": 64_000, "in": 0.14, "out": 0.28})
    cfg  = RemoteModelConfig(
        provider="deepseek",
        api_key=api_key,
        model_name=model,
        base_url="https://api.deepseek.com/v1",
        max_budget_usd=max_budget_usd,
    )
    m = RemoteModel(cfg)
    m.cost_per_1k_input_tokens  = spec["in"]
    m.cost_per_1k_output_tokens = spec["out"]
    return m
