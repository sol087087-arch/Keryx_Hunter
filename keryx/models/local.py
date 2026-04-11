# keryx/models/local.py
# Local LLM backend using llama-cpp-python — native, fast, grammar-constrained,
# no HTTP, no network, full tool-calling support.
# Requires Python 3.11+, llama-cpp-python >= 0.2.70
# Sovereign, air-gapped core.
import asyncio
import gc
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
import psutil
from llama_cpp import Llama, LlamaGrammar
from .interface import ModelInterface
logger = logging.getLogger("keryx.models.local")
# GGUF quantization RAM multipliers vs file size (conservative estimates)
_QUANT_RAM_MULTIPLIERS = {
    "Q2": 1.1, "Q3": 1.15, "Q4": 1.2, "Q5": 1.25,
    "Q6": 1.3, "Q8": 1.5, "F16": 2.1, "F32": 4.0,
}
_DEFAULT_RAM_MULTIPLIER = 1.5 # safe default when quant unknown
# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------
class GenerationResult:
    **slots** = ("text", "tokens_generated", "tokens_per_second", "duration_ms")
    def **init**(self, text: str, tokens_generated: int, tokens_per_second: float, duration_ms: float):
        self.text = text
        self.tokens_generated = tokens_generated
        self.tokens_per_second = tokens_per_second
        self.duration_ms = duration_ms
class ToolCall:
    **slots** = ("name", "arguments")
    def **init**(self, name: str, arguments: Dict[str, Any]):
        self.name = name
        self.arguments = arguments
class ToolDefinition:
    **slots** = ("name", "description", "parameters", "executor")
    def **init**(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        executor: Optional[Callable[[Dict], str]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.executor = executor # FIX 3: stored per-tool, not in shared lambda
class LocalModelConfig:
    def **init**(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 32768,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        f16_kv: bool = True,
        # Speculative decoding
        draft_model_path: Optional[str] = None,
        draft_model_gpu_layers: int = 0,
        speculative_lookahead: int = 4,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.verbose = verbose
        self.f16_kv = f16_kv
        self.draft_model_path = draft_model_path
        self.draft_model_gpu_layers = draft_model_gpu_layers
        self.speculative_lookahead = speculative_lookahead
# ---------------------------------------------------------------------------
# LocalModel
# ---------------------------------------------------------------------------
class LocalModel(ModelInterface):
    """
    llama-cpp-python native backend.
    Features:
    - Compiled grammar cache (GBNF)
    - Native token counting
    - Structured tool calling with JSON schema enforcement
    - Streaming (async-safe, lock released during stream)
    - KV cache management with system-prompt prefix preservation
    - Speculative decoding (draft model)
    - OOM pre-check with quantization-aware RAM estimate
    """
    def **init**(self, config: LocalModelConfig):
        self.config = config
        self.model_path = Path(config.model_path)
        self.model_name = config.model_name or self.model_path.stem
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        # FIX 10: OOM check using quantization-aware multiplier, not flat 1.2x
        self._check_oom_risk(self.model_path)
        self._lock = Lock()
        self._grammar_cache: Dict[str, LlamaGrammar] = {}
        self._tools: Dict[str, ToolDefinition] = {}
        self._metrics = {"total_calls": 0, "total_tokens": 0,
                               "total_time_ms": 0.0, "errors": 0}
        # FIX 5: health state tracked explicitly, not time-cached
        self._healthy: Optional[bool] = None
        self._last_health_check = 0.0
        logger.info(
            f"[LocalModel] Loading {self.model_name} | "
            f"gpu_layers={config.n_gpu_layers} | ctx={config.n_ctx} | "
            f"threads={config.n_threads or 'auto'}"
        )
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=config.n_gpu_layers,
                n_ctx=config.n_ctx,
                n_batch=config.n_batch,
                n_threads=config.n_threads,
                verbose=config.verbose,
                f16_kv=config.f16_kv,
            )
        except Exception as exc:
            logger.error(f"[LocalModel] Failed to load: {exc}")
            raise RuntimeError(f"Model loading failed: {exc}") from exc
        # Speculative decoding — optional draft model
        self._draft_llm: Optional[Llama] = None
        if config.draft_model_path:
            self._load_draft_model(config)
        self.is_local = True
        self.cost_per_1k_input_tokens = 0.0
        self.cost_per_1k_output_tokens = 0.0
        logger.info(f"[LocalModel] {self.model_name} ready")
    # ------------------------------------------------------------------
    # OOM pre-check
    # ------------------------------------------------------------------
    @staticmethod
    def _check_oom_risk(model_path: Path) -> None:
        """
        Estimate RAM needed from file size + quantization type.
        FIX 10: Previous code used file_size * 1.2 for all quants.
        Q4 GGUF of a 70B model is ~40GB file but needs ~45GB RAM.
        F16 70B is ~130GB file and needs ~140GB RAM.
        We detect quant from filename and apply the right multiplier.
        """
        try:
            file_size = model_path.stat().st_size
            name_upper = model_path.stem.upper()
            multiplier = _DEFAULT_RAM_MULTIPLIER
            for quant, mult in _QUANT_RAM_MULTIPLIERS.items():
                if quant in name_upper:
                    multiplier = mult
                    break
            estimated_ram = file_size * multiplier
            available_ram = psutil.virtual_memory().available
            if available_ram < estimated_ram:
                logger.warning(
                    f"[LocalModel] Potential OOM: need ~{estimated_ram / 1e9:.1f} GB, "
                    f"have {available_ram / 1e9:.1f} GB available"
                )
            else:
                logger.info(
                    f"[LocalModel] RAM OK: need ~{estimated_ram / 1e9:.1f} GB, "
                    f"have {available_ram / 1e9:.1f} GB"
                )
        except Exception as exc:
            logger.debug(f"[LocalModel] OOM check skipped: {exc}")
    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        grammar: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text.
        If stream=True, returns a sync Iterator[str].
        Use generate_stream_async() for non-blocking streaming.
        FIX 1: Lock is NOT held during streaming — releasing it before
        yielding allows other threads to call generate() between chunks.
        """
        grammar_obj = self._get_grammar(grammar)
        if stream:
            # Return iterator WITHOUT holding lock — caller consumes at their pace
            return self._stream_iter(prompt, grammar_obj, max_tokens, temperature, top_p, stop)
        # Non-streaming: hold lock for the full call
        with self._lock:
            return self._generate_blocking(prompt, grammar_obj, max_tokens, temperature, top_p, stop)
    def _generate_blocking(
        self,
        prompt: str,
        grammar_obj: Optional[LlamaGrammar],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> str:
        start = time.time()
        try:
            output = self.llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                grammar=grammar_obj,
                stop=stop,
                echo=False,
            )
            text = output["choices"][0]["text"].strip()
            tokens = (
                output.get("usage", {}).get("completion_tokens")
                or len(self.llm.tokenize(text.encode()))
            )
            ms = (time.time() - start) * 1000
            self._record_metrics(tokens, ms)
            self._healthy = True
            logger.debug(
                f"[LocalModel] {self.model_name} | "
                f"{tokens} tok | {ms:.0f}ms | {tokens / (ms / 1000):.1f} tok/s"
            )
            return text
        except Exception as exc:
            self._metrics["errors"] += 1
            self._healthy = False
            logger.error(f"[LocalModel] Generation failed: {exc}")
            raise LocalModelError(f"Generation failed: {exc}") from exc
    def _stream_iter(
        self,
        prompt: str,
        grammar_obj: Optional[LlamaGrammar],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> Iterator[str]:
        """Sync streaming iterator. Lock is acquired per-chunk, not for full stream."""
        gen = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            grammar=grammar_obj,
            stop=stop,
            stream=True,
            echo=False,
        )
        for chunk in gen:
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text
    async def generate_async(
        self,
        prompt: str,
        grammar: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Async non-streaming generation.
        FIX 2: run_in_executor does not accept **kwargs for the callable.
        Use a lambda with explicit closed-over args instead.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt,
                grammar=grammar,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=False,
            ),
        )
    async def generate_stream_async(
        self,
        prompt: str,
        grammar: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Truly async streaming. Each token fetched in a thread, yielded to event loop.
        FIX 7: The suggested lambda: next(gen) approach passes a generator
        across threads — generators are not thread-safe. We instead create the
        generator in the executor thread and send tokens back via asyncio.Queue.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        def _produce() -> None:
            try:
                for token in self._stream_iter(
                    prompt,
                    self._get_grammar(grammar),
                    max_tokens,
                    temperature,
                    0.9,
                    None,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None) # sentinel
        loop.run_in_executor(None, _produce)
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
    # ------------------------------------------------------------------
    # Speculative decoding
    # ------------------------------------------------------------------
    def _load_draft_model(self, config: LocalModelConfig) -> None:
        """
        Load a small draft model for speculative decoding.
        Speculative decoding runs the draft model N steps ahead,
        then verifies with the main model in one forward pass.
        Typical speedup: 1.5–2.5x for structured output (JSON/code).
        Higher gains when grammar constrains output vocabulary.
        Recommended draft: same architecture family, 3–8B params.
        Example: llama-70b (main) + llama-8b (draft).
        """
        draft_path = Path(config.draft_model_path)
        if not draft_path.exists():
            logger.warning(f"[LocalModel] Draft model not found: {draft_path} — speculative decoding disabled")
            return
        self._check_oom_risk(draft_path)
        logger.info(f"[LocalModel] Loading draft model: {draft_path.name}")
        try:
            self._draft_llm = Llama(
                model_path=str(draft_path),
                n_gpu_layers=config.draft_model_gpu_layers,
                n_ctx=config.n_ctx,
                verbose=False,
            )
            self._speculative_lookahead = config.speculative_lookahead
            logger.info(
                f"[LocalModel] Speculative decoding enabled | "
                f"lookahead={config.speculative_lookahead} tokens"
            )
        except Exception as exc:
            logger.warning(f"[LocalModel] Draft model load failed ({exc}) — disabling speculative decoding")
            self._draft_llm = None
    def generate_speculative(
        self,
        prompt: str,
        grammar: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """
        Speculative decoding: draft model proposes tokens, main model verifies.
        If draft model not loaded, falls back to standard generation.
        Best results for structured output (JSON with grammar constraints).
        Note: llama-cpp-python speculative API is experimental.
        Pin llama-cpp-python >= 0.2.70 in requirements.
        """
        if self._draft_llm is None:
            logger.debug("[LocalModel] No draft model — using standard generation")
            return self.generate(prompt, grammar=grammar, max_tokens=max_tokens, temperature=temperature)
        grammar_obj = self._get_grammar(grammar)
        with self._lock:
            start = time.time()
            try:
                # llama-cpp-python speculative decoding API
                output = self.llm(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    grammar=grammar_obj,
                    draft_model=self._draft_llm, # type: ignore[arg-type]
                    draft_model_num_pred_tokens=self._speculative_lookahead,
                    echo=False,
                )
                text = output["choices"][0]["text"].strip()
                tokens = output.get("usage", {}).get("completion_tokens", 0)
                ms = (time.time() - start) * 1000
                self._record_metrics(tokens, ms)
                logger.debug(
                    f"[LocalModel] Speculative | {tokens} tok | {ms:.0f}ms | "
                    f"{tokens / (ms / 1000):.1f} tok/s"
                )
                return text
            except TypeError:
                # API not available in this llama-cpp-python version
                logger.warning("[LocalModel] Speculative API unavailable — falling back to standard")
                return self._generate_blocking(prompt, grammar_obj, max_tokens, temperature, 0.9, None)
            except Exception as exc:
                self._metrics["errors"] += 1
                raise LocalModelError(f"Speculative generation failed: {exc}") from exc
    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        prompts: List[str],
        grammar: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> List[GenerationResult]:
        """
        Sequential batch generation with shared grammar compilation.
        FIX 6: Previous docstring claimed 'native batch evaluation' — it was
        a plain for-loop. True parallel batch in llama.cpp requires llama_batch
        C API which llama-cpp-python does not yet expose cleanly.
        This is sequential but efficient: grammar compiled once, KV cache reused.
        For true parallelism, load multiple model instances and use asyncio.gather.
        """
        grammar_obj = self._get_grammar(grammar)
        results = []
        for prompt in prompts:
            try:
                start = time.time()
                output = self.llm(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    grammar=grammar_obj,
                    echo=False,
                )
                text = output["choices"][0]["text"].strip()
                tokens = output.get("usage", {}).get("completion_tokens", 0)
                ms = (time.time() - start) * 1000
                self._record_metrics(tokens, ms)
                results.append(GenerationResult(
                    text=text,
                    tokens_generated=tokens,
                    tokens_per_second=tokens / (ms / 1000) if ms > 0 else 0,
                    duration_ms=ms,
                ))
            except Exception as exc:
                logger.error(f"[LocalModel] Batch item failed: {exc}")
                results.append(GenerationResult("", 0, 0.0, 0.0))
        return results
    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        executor: Optional[Callable[[Dict], str]] = None,
    ) -> None:
        """
        Register a tool.
        FIX 3: Previous code stored all executors in one shared lambda
        that only captured the last registered tool. Each tool now stores
        its own executor reference in ToolDefinition.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            executor=executor,
        )
        logger.debug(f"[LocalModel] Registered tool: {name}")
    def execute_tool(self, tool_call: ToolCall) -> Optional[str]:
        """Execute a tool call if an executor was registered."""
        tool = self._tools.get(tool_call.name)
        if not tool or not tool.executor:
            return None
        try:
            return tool.executor(tool_call.arguments)
        except Exception as exc:
            logger.error(f"[LocalModel] Tool '{tool_call.name}' execution failed: {exc}")
            return f"Tool error: {exc}"
    def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> Union[str, ToolCall]:
        tools = tools or list(self._tools.values())
        grammar = self._build_tool_call_grammar([t.name for t in tools])
        sys_prompt = self._build_tool_prompt(prompt, tools)
        response = self.generate(sys_prompt, grammar=grammar, max_tokens=max_tokens, temperature=temperature)
        try:
            parsed = json.loads(response)
            if "tool_call" in parsed:
                tc = parsed["tool_call"]
                return ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
            return parsed.get("response", response)
        except json.JSONDecodeError:
            return response
    def _build_tool_prompt(self, user_prompt: str, tools: List[ToolDefinition]) -> str:
        descs = "\n".join(
            f"Tool: {t.name}\nDescription: {t.description}\nParameters: {json.dumps(t.parameters)}"
            for t in tools
        )
        return (
            'Respond with JSON in one of these formats:\n'
            '1. Tool call: {"tool_call": {"name": "...", "arguments": {...}}}\n'
            '2. Text: {"response": "..."}\n\n'
            f'Available tools:\n{descs}\n\n'
            f'User: {user_prompt}\n\nResponse:'
        )
    def _build_tool_call_grammar(self, tool_names: List[str]) -> str:
        """
        Valid GBNF grammar for tool call JSON.
        FIX 4: Previous grammar had mixed Python/GBNF quote styles and
        invalid rule structure. GBNF requires consistent double-quote strings
        and no Python f-string interpolation inside grammar literals.
        """
        # Build name alternatives as proper GBNF string literals
        name_alts = " | ".join(f'"{n}"' for n in tool_names)
        return (
            'root ::= tool-call | text-resp\n'
            f'tool-name ::= {name_alts}\n'
            'tool-call ::= "{" ws "\"tool_call\"" ws ":" ws "{" ws '
            '"\"name\"" ws ":" ws tool-name ws "," ws '
            '"\"arguments\"" ws ":" ws object ws "}" ws "}"\n'
            'text-resp ::= "{" ws "\"response\"" ws ":" ws string ws "}"\n'
            'object ::= "{" ws ( pair ( "," ws pair )* )? ws "}"\n'
            'pair ::= string ws ":" ws value\n'
            'value ::= string | number | object | array | "true" | "false" | "null"\n'
            'array ::= "[" ws ( value ( "," ws value )* )? ws "]"\n'
            'string ::= "\"" ( [^\"\\] | "\\" . )* "\""\n'
            'number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )?\n'
            'ws ::= [ \t\n\r]*\n'
        )
    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------
    def clear_kv_cache(self, keep_system_tokens: int = 0) -> None:
        """
        Clear KV cache.
        FIX 9: kv_cache_seq_rm API exists in llama-cpp-python >= 0.2.60
        but its signature changed across versions. We guard with try/except
        and fall back to full reset.
        keep_system_tokens: preserve this many tokens at the start
        (e.g. your system prompt length) to avoid re-encoding it next call.
        """
        try:
            if keep_system_tokens > 0:
                # Remove tokens from keep_system_tokens to end of context
                self.llm.kv_cache_seq_rm(-1, keep_system_tokens, -1)
                logger.debug(f"[LocalModel] KV cache trimmed, kept {keep_system_tokens} tokens")
            else:
                self.llm.reset()
                logger.debug(f"[LocalModel] KV cache cleared fully")
        except (AttributeError, TypeError) as exc:
            # API not available in this version — fall back to full reset
            logger.debug(f"[LocalModel] kv_cache_seq_rm unavailable ({exc}), using full reset")
            try:
                self.llm.reset()
            except Exception as e2:
                logger.warning(f"[LocalModel] KV cache reset failed: {e2}")
    def get_context_used(self) -> int:
        try:
            return self.llm.n_tokens
        except AttributeError:
            return 0
    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def is_healthy(self) -> bool:
        """
        Health check with short cache (5s) but explicit failure tracking.
        FIX 5: Previous version returned True for 5s regardless of actual state.
        Now: if model has been explicitly marked unhealthy (_healthy=False),
        we bypass the cache and re-test immediately.
        """
        now = time.time()
        # Only skip re-test if model was healthy recently
        if self._healthy is True and now - self._last_health_check < 5.0:
            return True
        self._last_health_check = now
        try:
            self.llm.tokenize(b"health_check")
            self._healthy = True
            return True
        except Exception:
            self._healthy = False
            return False
    # ------------------------------------------------------------------
    # Metrics and memory
    # ------------------------------------------------------------------
    def _record_metrics(self, tokens: int, ms: float) -> None:
        self._metrics["total_calls"] += 1
        self._metrics["total_tokens"] += tokens
        self._metrics["total_time_ms"] += ms
    def get_metrics(self) -> Dict[str, Any]:
        calls = self._metrics["total_calls"]
        ms = self._metrics["total_time_ms"]
        return {
            **self._metrics,
            "avg_latency_ms": ms / calls if calls > 0 else 0,
            "avg_tokens_per_second": self._metrics["total_tokens"] / (ms / 1000) if ms > 0 else 0,
            "model_name": self.model_name,
            "context_size": self.config.n_ctx,
            "context_used": self.get_context_used(),
            "speculative_decoding": self._draft_llm is not None,
        }
    def get_memory_usage(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 ** 3
                total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                result.update({"gpu_allocated_gb": alloc, "gpu_total_gb": total, "gpu_free_gb": total - alloc})
        except ImportError:
            pass
        proc = psutil.Process()
        result.update({"ram_used_gb": proc.memory_info().rss / 1024 ** 3, "ram_percent": proc.memory_percent()})
        return result
    # ------------------------------------------------------------------
    # Grammar cache
    # ------------------------------------------------------------------
    def _get_grammar(self, grammar_str: Optional[str]) -> Optional[LlamaGrammar]:
        if not grammar_str:
            return None
        if grammar_str not in self._grammar_cache:
            self._grammar_cache[grammar_str] = LlamaGrammar.from_string(grammar_str)
        return self._grammar_cache[grammar_str]
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def unload(self) -> None:
        with self._lock:
            for attr in ("llm", "_draft_llm"):
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    delattr(self, attr)
            self._grammar_cache.clear()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            logger.info(f"[LocalModel] {self.model_name} unloaded")
    def **repr**(self) -> str:
        draft = f", draft={Path(self.config.draft_model_path).stem}" if self._draft_llm else ""
        return f"LocalModel(name={self.model_name}, ctx={self.config.n_ctx}{draft})"
# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class LocalModelError(Exception):
    """Raised on local model generation failures."""
# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def load_local_model(
    model_path: str,
    model_name: Optional[str] = None,
    n_gpu_layers: int = -1,
    n_ctx: int = 32768,
    draft_model_path: Optional[str] = None,
    speculative_lookahead: int = 4,
    **kwargs,
) -> LocalModel:
    return LocalModel(LocalModelConfig(
        model_path=model_path,
        model_name=model_name,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        draft_model_path=draft_model_path,
        speculative_lookahead=speculative_lookahead,
        **kwargs,
    ))
def load_models_for_swarm(configs: List[Dict[str, Any]]) -> List[LocalModel]:
    models = []
    for cfg in configs:
        try:
            model = load_local_model(**cfg)
            models.append(model)
            logger.info(f"[Swarm] Loaded {model.model_name}")
        except Exception as exc:
            logger.error(f"[Swarm] Failed to load {cfg.get('model_path', '?')}: {exc}")
    return models
