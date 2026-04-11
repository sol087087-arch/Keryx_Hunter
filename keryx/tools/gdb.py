# keryx/tools/gdb.py
# GDB tool wrapper for KeryxHunter — production-hardened.
# Features: process group kill, universal parsing x86/x64/ARM, exploit pattern detection,
# ASLR entropy analysis, privacy scrubbing.
# Sovereign, async-safe, timeout-guaranteed.

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import shutil
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# FIX 3: correct relative import path
try:
    from ..core.shared_context import SharedContext
except ImportError:
    SharedContext = Any  # type: ignore

from .toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.gdb")

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Privacy scrubbing — self-contained, no runtime imports from other modules
# FIX 8: was importing from cloud_advisor at call time — hidden coupling
# ---------------------------------------------------------------------------

_SCRUB_PATTERNS = [
    (re.compile(r'/home/[^/\s]+',  re.IGNORECASE), '/home/user'),
    (re.compile(r'/Users/[^/\s]+', re.IGNORECASE), '/Users/user'),
    (re.compile(r'C:\\Users\\[^\\\s]+', re.IGNORECASE), 'C:\\Users\\user'),
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), 'x.x.x.x'),
    # FIX: don't scrub memory addresses — they're the payload of GDB output
]

def _scrub(text: str) -> str:
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# GDBResult — lean dataclass, only fields actually populated
# FIX 9: removed redundant fields that were never set in _run_gdb
# ---------------------------------------------------------------------------

@dataclass
class GDBResult:
    command:          str
    stdout:           str
    stderr:           str
    exit_code:        int
    execution_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# GDBTool
# ---------------------------------------------------------------------------

class GDBTool(BaseTool):   # FIX 1: inherit BaseTool

    name        = "gdb_analyze"
    description = (
        "Run GDB commands for crash analysis, stack traces, registers, "
        "memory inspection, and ASLR evaluation."
    )
    default_timeout = 45.0

    PRESET_COMMANDS: Dict[str, str] = {
        "backtrace":      "bt full",
        "registers":      "info registers",
        "all_registers":  "info all-registers",
        "stack":          "info stack",
        "frame":          "info frame",
        "locals":         "info locals",
        "args":           "info args",
        "threads":        "info threads",
        "disas":          "disas",
        "full_analysis":  "bt full; info registers; info frame; info locals; info proc mappings",
        "memory_maps":    "info proc mappings",
    }

    def __init__(
        self,
        gdb_path:         Optional[str] = None,
        timeout_seconds:  float         = 45.0,
        max_output_chars: int           = 50_000,
        enable_scrubbing: bool          = True,
    ):
        super().__init__()   # initialises _lock, _call_count etc.
        self.gdb_path         = gdb_path or shutil.which("gdb") or "gdb"
        self.timeout          = timeout_seconds
        self.max_output_chars = max_output_chars
        self.enable_scrubbing = enable_scrubbing
        self._available       = bool(shutil.which(self.gdb_path))

        if not self._available:
            logger.warning(f"[GDBTool] Binary not found: {self.gdb_path!r}")
        logger.info(
            f"[GDBTool] Initialized | available={self._available} | "
            f"binary={self.gdb_path} | timeout={timeout_seconds}s"
        )

    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_input: Dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        if not self._available:
            return ToolResult(
                success=False,
                output="GDB not available. Install gdb and add it to PATH.",
                error="gdb_not_found",
            )

        # Resolve command
        preset  = action_input.get("preset")
        command = action_input.get("command")
        if preset and preset in self.PRESET_COMMANDS:
            command = self.PRESET_COMMANDS[preset]

        if not command:
            return ToolResult(
                success=False,
                output="Missing 'command' or valid 'preset' in action_input",
                error="command_missing",
            )

        core_path      = action_input.get("core")
        pid            = action_input.get("pid")
        executable     = action_input.get("executable")
        extra_commands = action_input.get("extra_commands", [])

        start = time.time()
        try:
            gdb_result = await self._run_gdb(
                command, core_path, pid, executable, extra_commands
            )
            duration_ms = (time.time() - start) * 1000

            # FIX 2: call self._scrub_output(), not undefined scrub_gdb_output()
            output = gdb_result.stdout
            if self.enable_scrubbing:
                output = self._scrub_output(output)

            if len(output) > self.max_output_chars:
                output = output[:self.max_output_chars] + "\n...[truncated]..."

            parsed = self._parse_gdb_output(output)

            if gdb_result.exit_code == 0:
                logger.info(
                    f"[GDBTool] Success | {duration_ms:.0f}ms | "
                    f"frames={len(parsed.get('stack_trace', []))}"
                )
                return ToolResult(
                    success=True,
                    output=output,
                    data={
                        "findings":        parsed.get("findings", [])[:100],
                        "stack_trace":     parsed.get("stack_trace", [])[:50],
                        "registers":       parsed.get("registers", {}),
                        "signal_info":     parsed.get("signal_info"),
                        "fault_address":   parsed.get("fault_address"),
                        "aslr_entropy":    parsed.get("aslr_entropy"),
                        "execution_time_ms": duration_ms,
                    },
                    metadata={
                        "tool":      self.name,
                        "command":   command[:100],
                        "exit_code": gdb_result.exit_code,
                    },
                )
            else:
                logger.warning(f"[GDBTool] Failed | exit_code={gdb_result.exit_code}")
                return ToolResult(
                    success=False,
                    output=gdb_result.stderr or output,
                    error=f"exit_code_{gdb_result.exit_code}",
                )

        except asyncio.TimeoutError:
            logger.error(f"[GDBTool] Timeout after {self.timeout}s")
            return ToolResult(
                success=False,
                output=f"GDB timed out after {self.timeout}s",
                error="timeout",
            )
        except Exception as exc:
            logger.error(f"[GDBTool] Unexpected: {exc}", exc_info=True)
            return ToolResult(success=False, output=str(exc), error="unexpected_error")

    # ------------------------------------------------------------------
    # GDB subprocess with correct command ordering
    # FIX 7: load file/core FIRST, then run analysis commands
    # FIX 4: platform guard for os.setsid (Unix only)
    # ------------------------------------------------------------------

    async def _run_gdb(
        self,
        command:        str,
        core_path:      Optional[str]       = None,
        pid:            Optional[int]        = None,
        executable:     Optional[str]        = None,
        extra_commands: Optional[List[str]]  = None,
    ) -> GDBResult:
        cmd = [self.gdb_path, "--batch", "-n", "--quiet", "--nx"]

        # FIX 7: load target FIRST so subsequent commands have context
        if executable:
            cmd += ["-ex", f"file {executable}"]
        if core_path:
            cmd += ["-ex", f"core-file {core_path}"]
        elif pid is not None:
            cmd += ["-ex", f"attach {pid}"]

        # THEN add analysis commands
        for c in (c.strip() for c in command.split(";") if c.strip()):
            cmd += ["-ex", c]

        if extra_commands:
            for ec in (e.strip() for e in extra_commands if e.strip()):
                cmd += ["-ex", ec]

        logger.debug(f"[GDBTool] Running: {' '.join(cmd[:8])}...")

        # FIX 4: preexec_fn=os.setsid is Unix-only
        preexec = os.setsid if not _IS_WINDOWS else None

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
            return GDBResult(
                command=command,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )

        except asyncio.TimeoutError:
            # Kill entire process group (Unix) or just the process (Windows)
            if not _IS_WINDOWS:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            else:
                proc.kill()
            raise

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_gdb_output(self, output: str) -> Dict[str, Any]:
        """Universal parser for x86/x64/ARM output."""
        findings:     List[Dict]       = []
        stack_trace:  List[str]        = []
        registers:    Dict[str, str]   = {}
        signal_info:  Optional[str]    = None
        fault_address: Optional[str]   = None

        # FIX 5: compile signal name set as lowercase for case-insensitive matching
        _SIGNAL_NAMES = frozenset(
            s.lower() for s in ("sigsegv", "sigabrt", "sigill", "sigfpe", "sigbus")
        )

        for line in output.splitlines():
            ls = line.strip()

            # Stack frames (#0, #1, ...)
            if ls.startswith("#") and ls[1:2].isdigit():
                stack_trace.append(ls)
                findings.append({"type": "stack_frame", "raw": ls[:200]})
                continue

            # FIX 5: lowercase comparison for signal names
            ls_lower = ls.lower()
            if "signal" in ls_lower and any(s in ls_lower for s in _SIGNAL_NAMES):
                signal_info = ls
                findings.append({"type": "signal", "raw": ls})
                m = re.search(r"at (0x[0-9a-fA-F]+)", ls)
                if m:
                    fault_address = m.group(1)

            # Registers (rax/eax/r0 format)
            reg_m = re.match(r"^([a-z][a-z0-9]{1,4})\s+(0x[0-9a-fA-F]+)", ls, re.IGNORECASE)
            if reg_m:
                registers[reg_m.group(1)] = reg_m.group(2)

            # Memory map lines
            if re.match(r"0x[0-9a-fA-F]+-0x[0-9a-fA-F]+", ls):
                findings.append({"type": "memory_map", "raw": ls[:150]})

        # Exploit pattern detection
        if re.search(r"0x41414141|0x61616161|0x42424242", output, re.IGNORECASE):
            findings.append({
                "type":   "exploit_pattern",
                "detail": "Repeating pattern detected (AAAA/aaaa/BBBB) — possible buffer overflow control",
            })

        # UAF / heap corruption signals
        if re.search(r"heap.use.after.free|double.free|heap.buffer.overflow", output, re.IGNORECASE):
            findings.append({
                "type":   "asan_signal",
                "detail": "AddressSanitizer: memory corruption detected",
            })

        return {
            "findings":     findings[:100],
            "stack_trace":  stack_trace,
            "registers":    registers,
            "signal_info":  signal_info,
            "fault_address": fault_address,
            "aslr_entropy": self._analyze_aslr_entropy(output),
        }

    @staticmethod
    def _analyze_aslr_entropy(maps_output: str) -> Optional[float]:
        """
        Rough ASLR entropy from memory maps.

        FIX 6: clarified comment — addr[:8] = highest 8 hex chars (top 32 bits
        of a 64-bit address). Low entropy here = weak ASLR.
        """
        addresses = re.findall(r"0x([0-9a-fA-F]{8,16})", maps_output)
        if len(addresses) < 5:
            return None
        # Count unique top-32-bit prefixes
        unique_high = len(set(addr[:8] for addr in addresses))
        entropy = (unique_high / len(addresses)) * 32.0
        return round(entropy, 2)

    def _scrub_output(self, text: str) -> str:
        """FIX 2: was calling undefined scrub_gdb_output(). Now calls _scrub()."""
        return _scrub(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_preset_commands(self) -> Dict[str, str]:
        return self.PRESET_COMMANDS.copy()

    def __repr__(self) -> str:
        return (
            f"GDBTool(available={self._available}, "
            f"binary={self.gdb_path!r}, timeout={self.timeout}s)"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_gdb_tool(
    gdb_path:        Optional[str] = None,
    timeout_seconds: float         = 45.0,
    **kwargs,
) -> GDBTool:
    return GDBTool(gdb_path=gdb_path, timeout_seconds=timeout_seconds, **kwargs)
