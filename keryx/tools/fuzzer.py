# keryx/tools/fuzzer.py
# Fuzzer tool wrapper for KeryxHunter — hardened edition.
# Supports AFL++ and libFuzzer with real process group kill on timeout,
# crash triage, ASan integration, privacy scrubbing, and environment fixes.
# Sovereign, async-safe, production-ready.

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import re
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# FIX 1: correct import path
try:
    from ..core.shared_context import SharedContext
except ImportError:
    SharedContext = Any  # type: ignore

from .Toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.fuzzer")

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class FuzzerResult:
    fuzzer_type:       str
    crashes_found:     int
    unique_crashes:    int
    execution_time_ms: float
    findings:          list[dict[str, Any]] = field(default_factory=list)
    crash_files:       list[str]            = field(default_factory=list)
    stdout:            str                  = ""
    stderr:            str                  = ""
    exit_code:         int                  = 0


# ---------------------------------------------------------------------------
# ASan output parser
# ---------------------------------------------------------------------------

_ASAN_PATTERNS: dict[str, str] = {
    "heap-use-after-free":      r"==\d+==ERROR: AddressSanitizer: heap-use-after-free",
    "heap-buffer-overflow":     r"==\d+==ERROR: AddressSanitizer: heap-buffer-overflow",
    "stack-buffer-overflow":    r"==\d+==ERROR: AddressSanitizer: stack-buffer-overflow",
    "global-buffer-overflow":   r"==\d+==ERROR: AddressSanitizer: global-buffer-overflow",
    "use-after-poison":         r"==\d+==ERROR: AddressSanitizer: use-after-poison",
    "container-overflow":       r"==\d+==ERROR: AddressSanitizer: container-overflow",
    "double-free":              r"==\d+==ERROR: AddressSanitizer: double-free",
    "alloc-dealloc-mismatch":   r"==\d+==ERROR: AddressSanitizer: alloc-dealloc-mismatch",
}


def parse_asan_output(output: str) -> list[dict[str, Any]]:
    detected = next(
        (ct for ct, pat in _ASAN_PATTERNS.items() if re.search(pat, output)),
        None,
    )
    if not detected:
        return []

    stack: list[str] = []
    in_stack = False
    for line in output.splitlines():
        if re.match(r"\s+#\d+ ", line):
            in_stack = True
            stack.append(line.strip())
        elif in_stack and not line.strip():
            break

    addr = None
    m = re.search(r"0x[0-9a-fA-F]{8,16}", output)
    if m:
        addr = m.group(0)

    return [{
        "type":        "crash",
        "crash_type":  detected,
        "stack_trace": stack[:20],
        "fault_address": addr,
    }]


# ---------------------------------------------------------------------------
# Privacy scrubbing
# ---------------------------------------------------------------------------

_FUZZER_SCRUB = [
    (re.compile(r'/home/[^/\s]+',    re.IGNORECASE), '/workspace'),
    (re.compile(r'/Users/[^/\s]+',   re.IGNORECASE), '/workspace'),
    (re.compile(r'C:\\Users\\[^\\]+', re.IGNORECASE), 'C:\\workspace'),
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), 'x.x.x.x'),
    # Note: intentionally NOT scrubbing memory addresses — they are the payload
]


def scrub_fuzzer_output(text: str) -> str:
    for pat, repl in _FUZZER_SCRUB:
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# FuzzerTool
# ---------------------------------------------------------------------------

class FuzzerTool(BaseTool):
    """
    Fuzzer integration.

    Supports:
    - AFL++ (for native binaries with corpus)
    - libFuzzer (ASan-instrumented binary IS the fuzzer — no separate binary)

    FIX 3: libFuzzer is not a standalone binary. A libFuzzer-instrumented
    target IS its own fuzzer. We just run the target with fuzzing flags.
    The 'libfuzzer_path' concept is removed — caller passes the instrumented
    binary in action_input["binary"] and sets fuzzer_type="libfuzzer".
    """

    name            = "fuzzer_run"
    description     = "Run AFL++ or libFuzzer on a target binary with seeds and timeout."
    default_timeout = 120.0

    def __init__(
        self,
        afl_path:         str | None = None,
        timeout_seconds:  float         = 120.0,
        max_output_chars: int           = 50_000,
        enable_scrubbing: bool          = True,
        default_seed_dir: str | None = None,
    ):
        super().__init__()   # FIX 8: super sets _call_count etc — don't repeat
        self.afl_path         = afl_path or shutil.which("afl-fuzz") or "afl-fuzz"
        self.timeout          = timeout_seconds
        self.max_output_chars = max_output_chars
        self.enable_scrubbing = enable_scrubbing
        self.default_seed_dir = default_seed_dir or str(
            Path(__file__).parent / "seeds"
        )
        self._afl_available = bool(shutil.which(self.afl_path))
        logger.info(
            f"[FuzzerTool] Initialized | afl={self._afl_available} | "
            f"timeout={timeout_seconds}s"
        )

    def is_available(self) -> bool:
        # libFuzzer availability depends on the binary — always considered available
        return self._afl_available or True

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_input: dict[str, Any],
        context:      Any = None,
    ) -> ToolResult:
        fuzzer_type = action_input.get("fuzzer_type", "afl")
        binary      = action_input.get("binary")

        if not binary:
            return ToolResult(
                success=False,
                output="Missing 'binary' in action_input",
                error="binary_missing",
            )
        if not Path(binary).exists():
            return ToolResult(
                success=False,
                output=f"Binary not found: {binary}",
                error="binary_not_found",
            )

        if fuzzer_type == "afl" and not self._afl_available:
            return ToolResult(
                success=False,
                output="AFL++ not installed. Install afl-fuzz.",
                error="afl_not_found",
            )

        start = time.time()
        output_dir: str | None = None

        try:
            if fuzzer_type == "afl":
                result, output_dir = await self._run_afl(action_input, binary)
            elif fuzzer_type == "libfuzzer":
                result, output_dir = await self._run_libfuzzer(action_input, binary)
            else:
                return ToolResult(
                    success=False,
                    output=f"Unknown fuzzer_type: {fuzzer_type!r}",
                    error="unknown_fuzzer",
                )

            duration_ms            = (time.time() - start) * 1000
            result.execution_time_ms = duration_ms
            self._record_call(success=result.crashes_found >= 0, duration_ms=duration_ms)

            # FIX 6: _process_crash_files defined below
            findings = self._process_crash_files(result.crash_files)
            result.findings = findings

            output = result.stdout
            if self.enable_scrubbing:
                output = scrub_fuzzer_output(output)
            if len(output) > self.max_output_chars:
                output = output[:self.max_output_chars] + "\n...[truncated]..."

            if result.crashes_found > 0:
                logger.info(
                    f"[FuzzerTool] Crashes: {result.crashes_found} unique={result.unique_crashes} | "
                    f"{duration_ms:.0f}ms"
                )

            return ToolResult(
                success=True,
                output=output,
                data={
                    "crashes_found":    result.crashes_found,
                    "unique_crashes":   result.unique_crashes,
                    "findings":         findings,
                    "crash_files":      result.crash_files[:20],
                    "execution_time_ms": duration_ms,
                },
                metadata={
                    "tool":         self.name,
                    "binary":       binary,
                    "fuzzer_type":  fuzzer_type,
                },
            )

        except TimeoutError:
            self._record_call(success=False, duration_ms=(time.time() - start) * 1000)
            logger.error(f"[FuzzerTool] Timed out after {self.timeout}s")
            return ToolResult(
                success=False,
                output=f"Fuzzer timed out after {self.timeout}s",
                error="timeout",
            )
        except Exception as exc:
            self._record_call(success=False, duration_ms=(time.time() - start) * 1000)
            logger.error(f"[FuzzerTool] Unexpected: {exc}", exc_info=True)
            return ToolResult(success=False, output=str(exc), error="unexpected_error")

        finally:
            # FIX 5: clean up temp output dirs on success too
            if output_dir and output_dir.startswith(tempfile.gettempdir()):
                import shutil as _sh
                with contextlib.suppress(Exception):
                    _sh.rmtree(output_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # AFL++
    # ------------------------------------------------------------------

    async def _run_afl(
        self,
        action_input: dict[str, Any],
        binary:       str,
    ) -> tuple[FuzzerResult, str]:
        """Returns (FuzzerResult, output_dir) so caller can clean up."""
        seed_dir   = action_input.get("seed_dir", self.default_seed_dir)
        output_dir = action_input.get("output_dir") or tempfile.mkdtemp(prefix="keryx_afl_")
        use_stdin  = action_input.get("use_stdin", False)
        extra_args = action_input.get("extra_args", [])

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        seed_path = Path(seed_dir)
        if not seed_path.exists():
            seed_path.mkdir(parents=True, exist_ok=True)
            # FIX 9: write bytes, not text — AFL reads binary seeds
            (seed_path / "seed1").write_bytes(b"AAAA")

        env = os.environ.copy()
        env.update({
            "AFL_SKIP_CPU_CHECK": "1",
            "AFL_I_HERE_BY_RECALL_THAT_ASAN_AND_MSAN_ARE_NOT_COMPATIBLE": "1",
            "AFL_NO_UI": "1",
        })

        cmd = [
            self.afl_path,
            "-i", str(seed_dir),
            "-o", str(output_dir),
            "-t", "5000",
            "-m", "none",
        ]
        if action_input.get("dict"):
            cmd += ["-x", action_input["dict"]]
        cmd += ["--", binary]
        if not use_stdin:
            cmd.append("@@")
        cmd.extend(extra_args)

        logger.debug(f"[FuzzerTool] AFL++: {' '.join(cmd[:8])}...")
        preexec = os.setsid if not _IS_WINDOWS else None

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            preexec_fn=preexec,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except TimeoutError:
            _kill_pg(proc)
            raise

        # FIX 7: only files in crashes/ subdir, not hangs/ or queue/
        crash_dir   = Path(output_dir) / "crashes"
        crash_files = []
        if crash_dir.exists():
            crash_files = [
                str(p) for p in crash_dir.iterdir()
                if p.is_file() and p.name.startswith("id:")
                and "README" not in p.name
            ]

        return FuzzerResult(
            fuzzer_type="afl++",
            crashes_found=len(crash_files),
            unique_crashes=len(crash_files),
            execution_time_ms=0.0,
            crash_files=crash_files,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        ), output_dir

    # ------------------------------------------------------------------
    # libFuzzer
    # FIX 3: libFuzzer binary = the target itself, compiled with -fsanitize=fuzzer
    # FIX 4: fixed stdout/stderr capture after timeout
    # ------------------------------------------------------------------

    async def _run_libfuzzer(
        self,
        action_input: dict[str, Any],
        binary:       str,
    ) -> tuple[FuzzerResult, str]:
        """
        Run a libFuzzer-instrumented binary.

        The binary compiled with -fsanitize=fuzzer,address IS the fuzzer.
        We pass flags to control corpus, timeout, and artifact output.
        """
        seed_dir   = action_input.get("seed_dir", self.default_seed_dir)
        output_dir = action_input.get("output_dir") or tempfile.mkdtemp(prefix="keryx_libf_")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            binary,
            "-runs=0",
            "-timeout=10",
            f"-max_total_time={int(self.timeout)}",
            f"-artifact_prefix={output_dir}/",
        ]
        if Path(seed_dir).exists():
            cmd.append(str(seed_dir))

        logger.debug(f"[FuzzerTool] libFuzzer: {' '.join(cmd[:6])}...")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout + 5.0  # slight grace over -max_total_time
            )
        except TimeoutError:
            try:
                proc.terminate()
                # FIX 4: read buffered output BEFORE wait, not after
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            except TimeoutError:
                proc.kill()
                stdout_b, stderr_b = await proc.communicate()

        crash_files = [
            str(p) for p in Path(output_dir).iterdir()
            if p.is_file() and p.name not in ("README.txt",) and not p.name.endswith(".log")
        ]

        return FuzzerResult(
            fuzzer_type="libfuzzer",
            crashes_found=len(crash_files),
            unique_crashes=len(crash_files),
            execution_time_ms=0.0,
            crash_files=crash_files,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        ), output_dir

    # ------------------------------------------------------------------
    # FIX 6: _process_crash_files — was called but never defined
    # ------------------------------------------------------------------

    def _process_crash_files(self, crash_files: list[str]) -> list[dict[str, Any]]:
        """
        Read crash artifacts and parse ASan output from each.
        Returns aggregated findings list.
        """
        all_findings: list[dict[str, Any]] = []
        for path_str in crash_files[:20]:   # cap to avoid excessive I/O
            try:
                content = Path(path_str).read_text(encoding="utf-8", errors="replace")
                findings = parse_asan_output(content)
                for f in findings:
                    f["crash_file"] = path_str
                all_findings.extend(findings)
            except Exception as exc:
                logger.debug(f"[FuzzerTool] Could not read crash file {path_str}: {exc}")
        return all_findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_preset_commands(self) -> dict[str, str]:
        return {
            "quick_afl":      "fuzzer_type=afl, timeout=60s",
            "thorough_afl":   "fuzzer_type=afl, timeout=300s, extra_args=[-D]",
            "libfuzzer_asan": "fuzzer_type=libfuzzer (binary must be -fsanitize=fuzzer,address)",
        }

    def __repr__(self) -> str:
        return (
            f"FuzzerTool(afl={self._afl_available}, "
            f"timeout={self.timeout}s)"
        )


# ---------------------------------------------------------------------------
# Process group kill helper (shared with GDB pattern)
# ---------------------------------------------------------------------------

def _kill_pg(proc: asyncio.subprocess.Process) -> None:
    """Kill process group on Unix, process on Windows."""
    if _IS_WINDOWS:
        proc.kill()
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_fuzzer_tool(
    afl_path:        str | None = None,
    timeout_seconds: float         = 120.0,
    **kwargs,
) -> FuzzerTool:
    return FuzzerTool(afl_path=afl_path, timeout_seconds=timeout_seconds, **kwargs)
