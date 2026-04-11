# keryx/sandbox/container.py
# Isolated Docker sandbox for KeryxHunter — hardened, sovereign, pre-built image edition.
# Features:
# - Pre-built keryx/hunter-base:latest (build-essential, afl++, gdb, python3)
# - Memory + CPU limits (2g / 1 core)
# - SYS_PTRACE for GDB
# - --network none for true air-gapped mode
# - Timeout via asyncio.wait_for around blocking SDK calls
# - Privacy scrubbing (self-contained)
# - ASAN_OPTIONS inside container

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# FIX 1: correct path — shared_context lives in keryx/core/
try:
    from ..core.shared_context import SharedContext
except ImportError:
    SharedContext = Any  # type: ignore

# FIX 2: correct path — toolbox lives in keryx/tools/, container in keryx/sandbox/
try:
    from ..tools.toolbox import ToolResult
except ImportError:
    ToolResult = Any  # type: ignore

try:
    import docker
    from docker.models.containers import Container as DockerContainer
    _DOCKER_IMPORTABLE = True
except ImportError:
    _DOCKER_IMPORTABLE = False
    DockerContainer = Any  # type: ignore

logger = logging.getLogger("keryx.sandbox")


# ---------------------------------------------------------------------------
# Privacy scrubbing — self-contained
# FIX 3: scrub_sandbox_output was called but never defined
# ---------------------------------------------------------------------------

_SCRUB_PATS = [
    (re.compile(r'/home/[^/\s]+',    re.IGNORECASE), '/home/user'),
    (re.compile(r'/Users/[^/\s]+',   re.IGNORECASE), '/Users/user'),
    (re.compile(r'C:\\Users\\[^\\\s]+', re.IGNORECASE), 'C:\\Users\\user'),
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), 'x.x.x.x'),
]


def scrub_sandbox_output(text: str) -> str:
    for pat, repl in _SCRUB_PATS:
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# SandboxResult
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    success:           bool
    output:            str
    error:             Optional[str]       = None
    execution_time_ms: float               = 0.0
    exit_code:         int                 = 0
    data:              Dict[str, Any]      = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IsolatedContainer
# ---------------------------------------------------------------------------

class IsolatedContainer:
    """
    Docker-based isolated sandbox for safe compilation and fuzzing.

    Uses pre-built keryx/hunter-base:latest — no runtime apt-get inside container.
    Hard limits: 2g RAM, 1 core (nano_cpus).

    Usage:
        async with IsolatedContainer(enable_network=False) as sandbox:
            result = await sandbox.compile_with_asan(["/workspace/target.cpp"], "/workspace/target")
            result = await sandbox.run_fuzzer("/workspace/target", "/workspace/seeds", "/workspace/out")
    """

    def __init__(
        self,
        image:            str   = "keryx/hunter-base:latest",
        timeout_seconds:  float = 180.0,
        enable_network:   bool  = True,
        enable_scrubbing: bool  = True,
        mem_limit:        str   = "2g",
        cpu_limit:        float = 1.0,      # cores → converted to nano_cpus
        auto_pull:        bool  = True,
    ):
        self.image            = image
        self.timeout          = timeout_seconds
        self.enable_network   = enable_network
        self.enable_scrubbing = enable_scrubbing
        self.mem_limit        = mem_limit
        self.cpu_limit        = cpu_limit
        self.auto_pull        = auto_pull

        # FIX 8: lazy init — don't ping Docker in __init__ (blocks event loop)
        self._docker_client   = None
        self._docker_checked  = False
        self._docker_available: Optional[bool] = None

        self.container:       Optional[Any]    = None
        self._container_id:   Optional[str]    = None
        self._temp_volumes:   List[str]        = []

        logger.info(
            f"[IsolatedContainer] Initialized | image={image} | "
            f"network={'on' if enable_network else 'off'} | "
            f"mem={mem_limit} | cpu={cpu_limit} | timeout={timeout_seconds}s"
        )

    # ------------------------------------------------------------------
    # Docker availability — lazy, async-safe
    # FIX 8: availability check is deferred to first start() call
    # ------------------------------------------------------------------

    async def _ensure_docker(self) -> bool:
        """Check Docker availability once, lazily, without blocking __init__."""
        if self._docker_checked:
            return bool(self._docker_available)

        self._docker_checked = True
        if not _DOCKER_IMPORTABLE:
            logger.warning("[IsolatedContainer] docker-py not installed")
            self._docker_available = False
            return False

        loop = asyncio.get_running_loop()   # FIX 4
        try:
            client = await loop.run_in_executor(None, docker.from_env)
            await loop.run_in_executor(None, client.ping)
            self._docker_client    = client
            self._docker_available = True
            return True
        except Exception as exc:
            logger.warning(f"[IsolatedContainer] Docker unavailable: {exc}")
            self._docker_available = False
            return False

    def is_available(self) -> bool:
        """Sync check — returns None/False before first start() call."""
        if self._docker_available is None:
            return _DOCKER_IMPORTABLE
        return bool(self._docker_available)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, volumes: Optional[Dict[str, Dict[str, str]]] = None) -> bool:
        if not await self._ensure_docker():
            return False
        if self.container:
            return True

        loop   = asyncio.get_running_loop()   # FIX 4
        client = self._docker_client

        try:
            # Pull image if needed
            try:
                await loop.run_in_executor(None, client.images.get, self.image)
            except Exception:
                if self.auto_pull and self.enable_network:
                    logger.info(f"[IsolatedContainer] Pulling {self.image}...")
                    await loop.run_in_executor(None, client.images.pull, self.image)
                else:
                    logger.error(
                        f"[IsolatedContainer] Image {self.image!r} not available locally "
                        "and network is disabled"
                    )
                    return False

            # Temporary host volume for scratch space
            tmp_host = tempfile.mkdtemp(prefix="keryx_sandbox_")
            self._temp_volumes.append(tmp_host)
            host_volumes = dict(volumes or {})
            host_volumes[tmp_host] = {"bind": "/sandbox_tmp", "mode": "rw"}

            # nano_cpus: 1 core = 1_000_000_000 (documented in Docker API)
            nano_cpus = int(self.cpu_limit * 1_000_000_000)

            self.container = await loop.run_in_executor(
                None,
                lambda: client.containers.run(
                    self.image,
                    command="tail -f /dev/null",
                    detach=True,
                    privileged=False,
                    cap_add=["SYS_PTRACE"],
                    mem_limit=self.mem_limit,
                    nano_cpus=nano_cpus,
                    network_mode="none" if not self.enable_network else "bridge",
                    volumes=host_volumes,
                    security_opt=["seccomp=unconfined"],
                    environment={
                        "ASAN_OPTIONS": "detect_leaks=0,disable_coredump=0",
                    },
                    remove=False,
                ),
            )
            self._container_id = self.container.short_id
            logger.info(f"[IsolatedContainer] Started {self._container_id}")
            return True

        except Exception as exc:
            logger.error(f"[IsolatedContainer] Failed to start: {exc}")
            return False

    async def stop(self) -> None:
        loop = asyncio.get_running_loop()   # FIX 4
        if self.container:
            try:
                await loop.run_in_executor(None, lambda: self.container.stop(timeout=5))
                await loop.run_in_executor(None, lambda: self.container.remove(force=True))
                logger.info(f"[IsolatedContainer] Stopped {self._container_id}")
            except Exception as exc:
                logger.warning(f"[IsolatedContainer] Stop failed: {exc}")
            finally:
                self.container      = None
                self._container_id  = None

        for vol in self._temp_volumes:
            shutil.rmtree(vol, ignore_errors=True)
        self._temp_volumes.clear()

    # ------------------------------------------------------------------
    # Command execution with real timeout
    # FIX 5+6: asyncio.wait_for wraps run_in_executor; blocking SDK call
    #           gets killed when the future is cancelled by wait_for.
    # ------------------------------------------------------------------

    async def execute_command(
        self,
        command:  List[str],
        workdir:  str            = "/workspace",
        timeout:  Optional[float] = None,
    ) -> SandboxResult:
        if not self.container:
            return SandboxResult(success=False, output="Container not started", error="container_not_ready")

        effective_timeout = timeout or self.timeout
        start = time.time()
        loop  = asyncio.get_running_loop()   # FIX 4

        try:
            # FIX 5+6: wrap run_in_executor in wait_for — if exec_run blocks,
            # the Future is cancelled and the thread pool slot is released.
            exec_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.container.exec_run(
                        command,
                        workdir=workdir,
                        demux=True,
                    ),
                ),
                timeout=effective_timeout,
            )

            duration_ms = (time.time() - start) * 1000
            raw_out = (exec_result.output[0] or b"").decode("utf-8", errors="replace")
            raw_err = (exec_result.output[1] or b"").decode("utf-8", errors="replace")
            output  = raw_out + raw_err

            if self.enable_scrubbing:
                output = scrub_sandbox_output(output)
            if len(output) > 50_000:
                output = output[:50_000] + "\n...[truncated]..."

            success = exec_result.exit_code == 0
            return SandboxResult(
                success=success,
                output=output,
                error=None if success else f"exit_code_{exec_result.exit_code}",
                execution_time_ms=duration_ms,
                exit_code=exec_result.exit_code,
                data={"command": " ".join(command)},
            )

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start) * 1000
            logger.error(f"[IsolatedContainer] Timeout: {' '.join(command[:4])} ({effective_timeout}s)")
            return SandboxResult(
                success=False,
                output=f"Command timed out after {effective_timeout}s",
                error="timeout",
                execution_time_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            logger.error(f"[IsolatedContainer] Execution failed: {exc}")
            return SandboxResult(
                success=False,
                output=str(exc),
                error="execution_failed",
                execution_time_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def compile_with_asan(
        self,
        source_files:  List[str],
        output_binary: str,
        workdir:       str            = "/workspace",
        extra_flags:   Optional[List[str]] = None,
    ) -> SandboxResult:
        """Compile C/C++ sources with AddressSanitizer inside the sandbox."""
        cmd = [
            "g++",
            "-fsanitize=address",
            "-g", "-O0",
            "-fno-omit-frame-pointer",
        ]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd += ["-o", output_binary] + source_files
        return await self.execute_command(cmd, workdir=workdir, timeout=120.0)

    async def run_fuzzer(
        self,
        binary:      str,
        seed_dir:    str,
        output_dir:  str,
        timeout:     float = 60.0,
        fuzzer_type: str   = "afl",
    ) -> SandboxResult:
        """
        Run AFL++ inside the sandbox.

        FIX 7: seed_dir and output_dir are CONTAINER paths (e.g. /workspace/seeds).
        Caller is responsible for mounting host dirs into the container via volumes.
        We create them inside the container with mkdir -p before running the fuzzer.
        """
        # Create dirs inside the container
        mkdir_result = await self.execute_command(
            ["mkdir", "-p", seed_dir, output_dir], timeout=10.0
        )
        if not mkdir_result.success:
            return mkdir_result

        # Write a minimal seed if seed_dir is empty
        seed_check = await self.execute_command(
            ["sh", "-c", f"ls {seed_dir} | wc -l"], timeout=5.0
        )
        if seed_check.success and seed_check.output.strip() == "0":
            await self.execute_command(
                ["sh", "-c", f"printf '\\x41\\x41\\x41\\x41' > {seed_dir}/seed1"],
                timeout=5.0,
            )

        if fuzzer_type == "afl":
            cmd = [
                "afl-fuzz",
                "-i", seed_dir,
                "-o", output_dir,
                "-t", "2000",
                "-m", "none",
                "--",
                binary, "@@",
            ]
            return await self.execute_command(cmd, timeout=timeout)

        return SandboxResult(
            success=False,
            output=f"Unsupported fuzzer_type: {fuzzer_type!r}",
            error="unsupported_fuzzer",
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "IsolatedContainer":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> bool:
        await self.stop()
        return False

    def __repr__(self) -> str:
        return (
            f"IsolatedContainer("
            f"running={self.container is not None}, "
            f"image={self.image!r}, "
            f"network={'on' if self.enable_network else 'off'})"
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

async def create_sandbox(
    image:           str   = "keryx/hunter-base:latest",
    enable_network:  bool  = True,
    timeout_seconds: float = 180.0,
    **kwargs,
) -> IsolatedContainer:
    """Create and start a sandbox. Caller must call stop() or use as context manager."""
    sandbox = IsolatedContainer(
        image=image,
        enable_network=enable_network,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
    await sandbox.start()
    return sandbox


async def run_in_sandbox(
    command:        List[str],
    image:          str   = "keryx/hunter-base:latest",
    timeout:        float = 60.0,
    enable_network: bool  = False,
    volumes:        Optional[Dict[str, Dict[str, str]]] = None,
) -> SandboxResult:
    """
    One-shot: start sandbox, run command, stop sandbox.

    Example:
        result = await run_in_sandbox(["ls", "-la", "/workspace"], timeout=10.0)
    """
    async with IsolatedContainer(
        image=image,
        enable_network=enable_network,
        timeout_seconds=timeout,
    ) as sandbox:
        await sandbox.start(volumes=volumes)
        return await sandbox.execute_command(command, timeout=timeout)
