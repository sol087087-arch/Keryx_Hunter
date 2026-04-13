# keryx/tools/toolbox.py
# ToolBox — hardened registry with structured returns, isolation, and intelligent formatting.

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("keryx.tools")


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Standardized result from any tool execution.

    agent.py reads:
        str(result)  — calls __str__ which returns format_for_llm output.
        The agent checks 'VULN_CONFIRMED' in str(result).

    FIX 8: __str__ uses ASCII markers, not emoji (Windows UTF-8 codepage issue).
    FIX 9: __str__ must return str so agent.py 'VULN_CONFIRMED' in observation works.
    """
    success:  bool
    output:   str
    error:    Optional[str]  = None
    data:     Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_for_llm(self, max_chars: int = 1000) -> str:
        """Format for agent prompt — structured summary with finding counts."""
        if not self.success:
            # FIX 8: ASCII instead of emoji
            return f"[ERROR:{self.error}]: {self.output[:max_chars]}"

        parts = []
        if "findings" in self.data:
            parts.append(f"[OK] Found {len(self.data['findings'])} potential issues")
        elif "count" in self.data:
            parts.append(f"[OK] Found {self.data['count']} items")
        else:
            parts.append("[OK] Success")

        preview = self.output[:max_chars]
        if len(self.output) > max_chars:
            preview += f"... [{len(self.output)} chars total]"
        parts.append(preview)
        return "\n".join(parts)

    def to_prompt_fragment(self) -> str:
        return self.format_for_llm(max_chars=800)

    def __str__(self) -> str:
        """
        FIX 9: agent.py does 'VULN_CONFIRMED' in observation where observation = str(result).
        Must return str that preserves the raw output text so signal strings survive.
        """
        return self.format_for_llm(max_chars=500)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":  self.success,
            "output":   self.output[:2000],
            "error":    self.error,
            "data":     self.data,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class BaseTool:
    """
    Abstract base for all Keryx tools.

    FIX 1+2: _lock, _call_count etc were class-level variables — shared across
    ALL instances of the same subclass. Moved to __init__ as instance variables.

    FIX 3: execute() signature does NOT include timeout — ToolBox handles
    timeout via asyncio.wait_for externally. Tools don't need to know about it.
    """

    name:            str   = "base_tool"
    description:     str   = "Base tool"
    default_timeout: float = 60.0

    def __init__(self) -> None:
        # FIX 1+2: instance-level, not class-level
        self._call_count:    int   = 0
        self._success_count: int   = 0
        self._total_time_ms: float = 0.0
        self._lock = threading.Lock()   # per-instance lock

    async def execute(
        self,
        action_input: Dict[str, Any],
        context: Any = None,
        # FIX 3: no timeout param — ToolBox applies wait_for externally
    ) -> ToolResult:
        raise NotImplementedError

    def get_command(self, action_input: Dict[str, Any]) -> Optional[List[str]]:
        """Return subprocess command, or None for Python-native tools."""
        return None

    def is_available(self) -> bool:
        return True

    def _record_call(self, success: bool, duration_ms: float) -> None:
        with self._lock:
            self._call_count    += 1
            self._total_time_ms += duration_ms
            if success:
                self._success_count += 1

    def reset_metrics(self) -> None:
        with self._lock:
            self._call_count    = 0
            self._success_count = 0
            self._total_time_ms = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name":         self.name,
                "calls":        self._call_count,
                "successes":    self._success_count,
                "success_rate": self._success_count / max(self._call_count, 1),
                "avg_time_ms":  self._total_time_ms / max(self._call_count, 1),
            }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# ToolBox
# ---------------------------------------------------------------------------

class ToolBox:
    """
    Hardened tool registry and dispatcher.

    agent.py:           execute(action, action_input) → str via __str__
    orchestrator.py:    execute_batch(actions) → List[ToolResult]
    advisors:           list_tools() → List[str]
    """

    def __init__(self, default_timeout: float = 60.0) -> None:
        self._tools:           Dict[str, BaseTool] = {}
        self.default_timeout   = default_timeout

        # FIX 7: CPU*4 causes OOM for CPU-bound tools. Cap at reasonable default.
        cpu_count  = os.cpu_count() or 1
        max_workers = min(8, cpu_count * 2)   # I/O bound: 2x; CPU bound: keep low
        if max_workers > 6:
            logger.warning(
                f"[ToolBox] {max_workers} workers — if tools are CPU-bound "
                "(local LLM), consider passing max_workers=2"
            )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="keryx_worker",
        )
        logger.info(f"[ToolBox] Initialized | workers={max_workers} (CPU={cpu_count})")

        # Persistent background loop for sync bridge
        self._sync_loop:   Optional[asyncio.AbstractEventLoop] = None
        self._sync_thread: Optional[threading.Thread]          = None
        self._loop_lock    = threading.Lock()
        self._loop_ready   = threading.Event()

        # Global metrics
        self._total_calls:   int = 0
        self._timeout_count: int = 0
        self._error_count:   int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"[ToolBox] Registered: {tool.name!r}")

    def register_many(self, tools: List[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def list_tools(self) -> List[str]:
        return [n for n, t in self._tools.items() if t.is_available()]

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_command(self, action: str, action_input: Dict[str, Any]) -> Optional[List[str]]:
        tool = self._tools.get(action)
        return tool.get_command(action_input) if tool and tool.is_available() else None

    # ------------------------------------------------------------------
    # Async execution
    # ------------------------------------------------------------------

    async def execute_async(
        self,
        action:       str,
        action_input: Dict[str, Any],
        context:      Any            = None,
        timeout:      Optional[float] = None,
    ) -> ToolResult:
        start = time.time()

        # Validate input
        if not isinstance(action_input, dict):
            return ToolResult(
                success=False,
                output=f"Invalid action_input type: {type(action_input).__name__}",
                error="invalid_input_type",
            )

        # Sanitize output limits
        sanitized = dict(action_input)
        try:
            sanitized["max_output_len"] = min(int(sanitized.get("max_output_len", 2000)), 10000)
        except (ValueError, TypeError):
            sanitized["max_output_len"] = 2000

        # Resolve tool
        tool = self._tools.get(action)
        if tool is None:
            self._total_calls += 1
            return ToolResult(
                success=False,
                output=f"Unknown tool: {action!r}",
                error="tool_not_found",
                metadata={"available": self.list_tools()},
            )
        if not tool.is_available():
            self._total_calls += 1
            return ToolResult(success=False, output=f"Tool {action!r} unavailable", error="tool_unavailable")

        # FIX 5: timeout managed by wait_for only — NOT passed into tool.execute()
        effective_timeout = timeout or getattr(tool, 'default_timeout', self.default_timeout)

        try:
            result = await asyncio.wait_for(
                tool.execute(sanitized, context),   # FIX 3+5: no timeout kwarg
                timeout=effective_timeout,
            )

            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, output=str(result))

            duration_ms = (time.time() - start) * 1000
            tool._record_call(result.success, duration_ms)
            self._total_calls += 1

            result.metadata.update({
                "tool":          action,
                "duration_ms":   duration_ms,
                "timeout_limit": effective_timeout,
            })
            return result

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start) * 1000
            tool._record_call(False, duration_ms)
            self._total_calls  += 1
            self._timeout_count += 1
            logger.warning(f"[ToolBox] Timeout: {action!r} after {effective_timeout}s")
            return ToolResult(
                success=False,
                output=f"Tool {action!r} timed out after {effective_timeout}s",
                error="timeout",
                metadata={"tool": action, "timeout": effective_timeout},
            )

        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            tool._record_call(False, duration_ms)
            self._total_calls  += 1
            self._error_count  += 1
            logger.error(f"[ToolBox] Unhandled in {action!r}: {exc}", exc_info=True)
            return ToolResult(
                success=False,
                output=f"Unhandled error in {action}: {exc}",
                error="unhandled_exception",
                metadata={"tool": action, "exception_type": type(exc).__name__},
            )

    # ------------------------------------------------------------------
    # Sync bridge
    # FIX 4: RuntimeError from get_running_loop() = NOT in async = CORRECT path
    # ------------------------------------------------------------------

    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._sync_loop is not None and not self._sync_loop.is_closed():
                return self._sync_loop

            self._loop_ready.clear()
            loop = asyncio.new_event_loop()
            self._sync_loop = loop

            def _run() -> None:
                asyncio.set_event_loop(loop)
                self._loop_ready.set()
                try:
                    loop.run_forever()
                except Exception as exc:
                    logger.error(f"[ToolBox] Sync loop crashed: {exc}")

            self._sync_thread = threading.Thread(target=_run, daemon=True, name="toolbox-loop")
            self._sync_thread.start()
            if not self._loop_ready.wait(timeout=3.0):
                raise RuntimeError("[ToolBox] Sync event loop failed to start")
            return loop

    def execute(
        self,
        action:       str,
        action_input: Dict[str, Any],
        context:      Any            = None,
        timeout:      Optional[float] = None,
    ) -> ToolResult:
        """
        Synchronous dispatch — returns ToolResult.

        agent.py uses str(result) to get the text observation.
        FIX 4: get_running_loop() raises RuntimeError when NOT in async context.
        That's the EXPECTED case here. We warn only if somehow called from async.
        """
        try:
            asyncio.get_running_loop()
            # We're inside an async context — this is a misuse
            logger.warning(
                f"[ToolBox] execute() called from async context for {action!r}. "
                "Use execute_async() instead."
            )
        except RuntimeError:
            pass   # FIX 4: RuntimeError = not in async = correct, proceed normally

        try:
            loop   = self._ensure_sync_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.execute_async(action, action_input, context, timeout),
                loop,
            )
            wall_timeout = (timeout or self.default_timeout) + 5.0
            return future.result(timeout=wall_timeout)

        except Exception as exc:
            logger.error(f"[ToolBox] Sync bridge failure for {action!r}: {exc}")
            return ToolResult(
                success=False,
                output=f"Sync bridge error: {exc}",
                error="sync_bridge_error",
            )

    # ------------------------------------------------------------------
    # Batch execution
    # FIX 6: return_exceptions=True is fine — execute_with_limit can still
    # raise if asyncio itself has a bug. Keep it, but document clearly.
    # ------------------------------------------------------------------

    async def execute_batch(
        self,
        actions:        List[tuple],
        context:        Any          = None,
        max_concurrent: int          = 4,
    ) -> List[ToolResult]:
        """
        Execute multiple tools with semaphore-controlled concurrency.
        Each tool's execute() already catches all errors and returns ToolResult,
        so exceptions here only arise from asyncio internals — still safe to catch.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run(action: str, action_input: Dict) -> ToolResult:
            async with semaphore:
                return await self.execute_async(action, action_input, context)

        results = await asyncio.gather(
            *[run(a, i) for a, i in actions],
            return_exceptions=True,
        )

        out: List[ToolResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                out.append(ToolResult(
                    success=False,
                    output=f"Batch error: {r}",
                    error="batch_exception",
                    metadata={"action": actions[i][0], "exception": type(r).__name__},
                ))
            else:
                out.append(r)
        return out

    # ------------------------------------------------------------------
    # Metrics and lifecycle
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_calls":   self._total_calls,
            "timeouts":      self._timeout_count,
            "errors":        self._error_count,
            "tools":         {n: t.get_metrics() for n, t in self._tools.items()},
        }

    def reset_metrics(self) -> None:
        self._total_calls   = 0
        self._timeout_count = 0
        self._error_count   = 0
        for tool in self._tools.values():
            tool.reset_metrics()

    def shutdown(self) -> None:
        logger.info("[ToolBox] Shutting down...")
        self._executor.shutdown(wait=False)
        with self._loop_lock:
            if self._sync_loop and not self._sync_loop.is_closed():
                self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)
                if self._sync_thread and self._sync_thread.is_alive():
                    self._sync_thread.join(timeout=3.0)
                try:
                    self._sync_loop.close()
                except Exception:
                    pass
        logger.info("[ToolBox] Done")

    def __repr__(self) -> str:
        return f"ToolBox(tools={list(self._tools.keys())})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_toolbox(
    tools:           Optional[List[BaseTool]] = None,
    default_timeout: float                    = 60.0,
) -> ToolBox:
    tb = ToolBox(default_timeout=default_timeout)
    if tools:
        tb.register_many(tools)
    return tb
# ---------------------------------------------------------------------------
# Default toolbox factory — именно её ждёт агент
# ---------------------------------------------------------------------------

def create_default_toolbox(
    allowed_root: Optional[str] = None,
    default_timeout: float = 60.0,
) -> ToolBox:
    """
    Создаёт Toolbox со всеми инструментами, которые должны быть доступны агенту.
    Это точка сборки, которую использует orchestrator и agent.
    """
    from .read_file import create_read_file_tool
    from .git_blame import create_git_blame_tool
    # from .codeql import create_codeql_tool      # раскомментировать позже
    # from .gdb import create_gdb_tool
    # from .fuzzer import create_fuzzer_tool

    toolbox = ToolBox(default_timeout=default_timeout)

    # Критический минимум (без него агент падает на первом шаге)
    toolbox.register(create_read_file_tool(allowed_root=allowed_root))
    toolbox.register(create_git_blame_tool())

    # Остальные инструменты — добавляем по мере готовности
    # toolbox.register(create_codeql_tool())
    # toolbox.register(create_gdb_tool())
    # toolbox.register(create_fuzzer_tool())

    logger.info(f"[ToolBox] Default toolbox created with {len(toolbox.list_tools())} tools")
    return toolbox