"""Sandboxed subprocess execution for PoC harnesses.

Isolation strategy (MVP):
  - subprocess.run() with wall-clock timeout (no hang risk)
  - preexec_fn applies RLIMIT_CPU and RLIMIT_FSIZE via the ``resource`` module
    before the child process exec()s — effective on Linux and macOS.
  - No seccomp/namespaces yet; firecracker/gVisor is the next tier.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    elapsed_s: float
    timed_out: bool = False


def _make_preexec(cpu_seconds: int):
    """Return a preexec_fn that applies resource limits in the child process."""
    def _preexec():
        try:
            import resource  # noqa: PLC0415
            # Hard CPU-time cap — process gets SIGKILL if it exceeds it
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            # Cap individual file writes at 4 MB — prevents runaway output files
            resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
        except Exception:  # noqa: BLE001
            pass  # best-effort; unsupported limits are silently ignored
    return _preexec


def run(script: str, *, timeout: int = 10) -> SandboxResult:
    """Execute *script* as a Python one-liner in an isolated subprocess.

    The child is killed after *timeout* wall-clock seconds regardless of
    CPU usage (double-bounded by RLIMIT_CPU + subprocess timeout).
    """
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_make_preexec(timeout),
        )
        return SandboxResult(
            stdout    = proc.stdout,
            stderr    = proc.stderr,
            exit_code = proc.returncode,
            elapsed_s = time.monotonic() - t0,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            stdout    = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr    = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            exit_code = None,
            elapsed_s = time.monotonic() - t0,
            timed_out = True,
        )
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(
            stdout    = "",
            stderr    = str(exc),
            exit_code = -1,
            elapsed_s = time.monotonic() - t0,
        )
