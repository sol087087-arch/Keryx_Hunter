# keryx/tools/git_blame.py
# Git Blame / Log / Hotspots Tool for KeryxHunter.
# Sovereign, async-safe, timeout-guaranteed, privacy scrubbing.
# Supports: blame, log, hotspots (churn analysis).

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..core.schemas import GitBlameInput, HotspotEntry, MetricsSnapshot, RiskLevel
from .Toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.git_blame")

# Rich is optional — used for human-readable operator output when available.
try:
    from rich.console import Console
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


@dataclass
class _GitResult:
    """Internal result from a raw git subprocess call."""
    command:   str
    output:    str
    stderr:    str
    exit_code: int
    authors:   list[str] = field(default_factory=list)


class GitBlameTool(BaseTool):
    """
    Git history tool.

    Supports:
    - git blame <file>[:line]
    - git log with optional --grep / --author / --since
    - hotspots: churn analysis for Regression-Driven Vulnerability Research

    Input is validated via Pydantic (GitBlameInput schema).
    Output uses toolbox.ToolResult so ToolBox dispatch pipeline stays intact.
    Rich table is printed to stderr when available and stderr is a TTY.
    """

    name        = "git_blame"
    description = (
        "Run git blame, git log, or churn analysis on a repository. "
        "Helps the agent understand code history, authorship, "
        "and identify high-risk hotspots (frequently changed files)."
    )

    _DEFAULT_EXTENSIONS: frozenset[str] = frozenset([
        ".c", ".cpp", ".cc", ".cxx",
        ".h", ".hpp",
        ".rs", ".go", ".java",
    ])

    def __init__(
        self,
        git_path:         str | None = None,
        timeout_seconds:  float         = 30.0,
        max_output_lines: int           = 200,
        enable_scrubbing: bool          = True,
        rich_output:      bool          = True,
    ) -> None:
        # FIX: BaseTool.__init__ owns _call_count, _success_count, _lock.
        # Never redeclare them here.
        super().__init__()

        # FIX: resolve once; _available = binary found, no double which() call.
        resolved         = git_path or shutil.which("git")
        self.git_path    = resolved or "git"
        self._available  = resolved is not None

        self.timeout          = timeout_seconds
        self.max_output_lines = max_output_lines
        self.enable_scrubbing = enable_scrubbing

        # Rich console on stderr, only when stderr is a TTY and rich_output=True.
        self._console: Any | None = None
        if rich_output and _RICH_AVAILABLE and sys.stderr.isatty():
            self._console = Console(stderr=True, highlight=False, soft_wrap=True)

        if not self._available:
            logger.warning("[GitBlameTool] git binary not found — tool disabled.")
        else:
            logger.info(
                "[GitBlameTool] Initialized | git=%s | timeout=%.0fs | "
                "max_lines=%d | rich=%s",
                self.git_path, timeout_seconds, max_output_lines,
                self._console is not None,
            )

    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_input: dict[str, Any],
        context:      Any = None,
    ) -> ToolResult:
        """
        Entry point. Validates input via Pydantic before dispatching.

        Expected action_input keys:
            command: "blame" | "log" | "hotspots"  (default: "blame")

            blame:
                file (str, required), line (int, optional)

            log:
                n (int), path (str), author (str), grep (str), since (str)

            hotspots:
                path (str), n (int), since (str),
                extensions (list[str]), min_changes (int)
        """
        if not self._available:
            return ToolResult(
                success=False,
                output="git not available. Install git and ensure it is in PATH.",
                error="git_not_found",
            )

        # Pydantic validation — catches missing fields, wrong types, wrong values.
        try:
            params = GitBlameInput.model_validate(action_input)
        except ValidationError as exc:
            logger.error("[GitBlameTool] Input validation failed: %s", exc)
            return ToolResult(
                success=False,
                output="Invalid action_input — see error for details.",
                error="validation_error",
                data={"validation_errors": exc.errors()},
            )

        start = time.time()

        try:
            if params.command == "hotspots":
                return await self._run_hotspots(params)

            if params.command == "blame":
                result = await self._git_blame(params)
            else:
                result = await self._git_log(params)

        except TimeoutError:
            duration_ms = (time.time() - start) * 1000
            self._record_call(success=False, duration_ms=duration_ms)
            logger.error("[GitBlameTool] %s timed out after %.0fs", params.command, self.timeout)
            return ToolResult(
                success=False,
                output=f"git {params.command} timed out after {self.timeout:.0f}s",
                error="timeout",
            )
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            self._record_call(success=False, duration_ms=duration_ms)
            logger.error("[GitBlameTool] Unexpected error: %s", exc, exc_info=True)
            return ToolResult(
                success=False,
                output=str(exc),
                error="unexpected_error",
            )

        duration_ms = (time.time() - start) * 1000
        success     = result.exit_code == 0
        self._record_call(success=success, duration_ms=duration_ms)

        if not success:
            return ToolResult(
                success=False,
                output=self._maybe_scrub(result.stderr) or self._maybe_scrub(result.output),
                error=f"git_{params.command}_failed",
                data={"exit_code": result.exit_code},
            )

        output = self._maybe_scrub(result.output)
        output = self._truncate(output)

        target = params.file or params.path
        return ToolResult(
            success=True,
            output=output,
            data={
                "command":           params.command,
                "target":            target,
                "lines":             len(output.splitlines()),
                "authors":           result.authors,
                "execution_time_ms": round(duration_ms, 1),
                "exit_code":         result.exit_code,
            },
            metadata={"tool": self.name, "command": params.command, "target": target},
        )

    # ------------------------------------------------------------------
    # blame
    # ------------------------------------------------------------------

    async def _git_blame(self, params: GitBlameInput) -> _GitResult:
        if not params.file:
            raise ValueError("'file' is required for blame")
        if not Path(params.file).exists():
            raise FileNotFoundError(f"File not found: {params.file}")

        cmd = [self.git_path, "blame", "--line-porcelain"]
        if params.line is not None:
            cmd.extend(["-L", f"{params.line},{params.line}"])
        cmd.append(params.file)

        return await self._run_cmd(cmd, extract_authors="blame")

    # ------------------------------------------------------------------
    # log
    # ------------------------------------------------------------------

    async def _git_log(self, params: GitBlameInput) -> _GitResult:
        cmd = [
            self.git_path, "log",
            f"-{params.n}",
            "--no-color",
            "--format=%h|%an|%ae|%s",
        ]
        if params.author:
            cmd.extend(["--author", params.author])
        if params.grep:
            cmd.extend(["--grep", params.grep])
        if params.since:
            cmd.extend(["--since", params.since])
        if params.path:
            cmd.append(params.path)

        return await self._run_cmd(cmd, extract_authors="log")

    # ------------------------------------------------------------------
    # hotspots
    # ------------------------------------------------------------------

    async def _run_hotspots(self, params: GitBlameInput) -> ToolResult:
        """
        Churn analysis: files with high change frequency and many authors.
        Complexity = changes x (1 + sqrt(authors))
        """
        target     = params.path or params.file or "."
        extensions = frozenset(params.extensions or self._DEFAULT_EXTENSIONS)
        min_churn  = params.min_changes

        start = time.time()

        try:
            # Step 1: collect changed file names
            cmd = [
                self.git_path, "log",
                f"-n{params.n}",
                "--pretty=format:",
                "--name-only",
            ]
            if params.since:
                cmd.extend(["--since", params.since])
            cmd.append(target)

            raw = await self._run_cmd(cmd)
            if raw.exit_code != 0:
                self._record_call(success=False, duration_ms=(time.time() - start) * 1000)
                return ToolResult(
                    success=False,
                    output=self._maybe_scrub(raw.stderr),
                    error="git_log_failed",
                    data={"exit_code": raw.exit_code},
                )

            # Step 2: count churn per file
            churn: dict[str, int] = {}
            for line in raw.output.splitlines():
                line = line.strip()
                if not line or not any(line.endswith(ext) for ext in extensions):
                    continue
                norm = os.path.normpath(line)
                churn[norm] = churn.get(norm, 0) + 1

            # Step 3: fetch authors for top-15 by churn
            candidates = sorted(churn.items(), key=lambda kv: kv[1], reverse=True)[:15]
            authors_by_file: dict[str, set[str]] = {}

            for filepath, changes in candidates:
                if changes < min_churn:
                    continue
                # FIX: build command declaratively — no fragile list.insert()
                author_cmd = [self.git_path, "log", f"-n{params.n}"]
                if params.since:
                    author_cmd.extend(["--since", params.since])
                author_cmd.extend(["--format=%an", "--follow", "--", filepath])

                ar = await self._run_cmd(author_cmd)
                if ar.exit_code == 0:
                    authors_by_file[filepath] = {
                        ln.strip() for ln in ar.output.splitlines() if ln.strip()
                    }

            # Step 4: score and classify
            hotspots: list[HotspotEntry] = []
            for filepath, changes in churn.items():
                if changes < min_churn:
                    continue
                authors   = authors_by_file.get(filepath, set())
                n_authors = len(authors)
                complexity = changes * (1.0 + math.sqrt(n_authors))

                if changes > 10 and n_authors > 3:
                    risk = RiskLevel.CRITICAL
                elif changes > 5 and n_authors > 2:
                    risk = RiskLevel.HIGH
                elif changes > 3:
                    risk = RiskLevel.MEDIUM
                else:
                    risk = RiskLevel.LOW

                hotspots.append(HotspotEntry(
                    file=filepath,
                    changes=changes,
                    authors=n_authors,
                    top_authors=sorted(authors)[:3],
                    complexity_score=round(complexity, 1),
                    risk_level=risk,
                ))

            hotspots.sort(key=lambda h: h.complexity_score, reverse=True)
            top = hotspots[:10]

            # Step 5: Rich table for human operator (stderr, TTY-only)
            self._print_rich_hotspots(top, target, params)

            # Step 6: plain-text output for LLM (no ANSI codes)
            llm_output = self._format_hotspots_text(top)
            llm_output = self._truncate(llm_output)

            duration_ms = (time.time() - start) * 1000
            self._record_call(success=True, duration_ms=duration_ms)

            return ToolResult(
                success=True,
                output=llm_output,
                data={
                    "hotspots":            [h.model_dump() for h in top],
                    "total_files_scanned": len(churn),
                    "analysis_period":     params.since or f"last {params.n} commits",
                    "execution_time_ms":   round(duration_ms, 1),
                },
                metadata={
                    "tool":             self.name,
                    "command":          "hotspots",
                    "target":           target,
                    "records_returned": len(top),
                },
            )

        except TimeoutError:
            duration_ms = (time.time() - start) * 1000
            self._record_call(success=False, duration_ms=duration_ms)
            logger.error("[GitBlameTool] Hotspots timed out after %.0fs", self.timeout)
            return ToolResult(
                success=False,
                output=f"Hotspots analysis timed out after {self.timeout:.0f}s",
                error="timeout",
            )
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            self._record_call(success=False, duration_ms=duration_ms)
            logger.error("[GitBlameTool] Hotspots failed: %s", exc, exc_info=True)
            return ToolResult(
                success=False,
                output=str(exc),
                error="analysis_failed",
            )

    # ------------------------------------------------------------------
    # Low-level subprocess
    # ------------------------------------------------------------------

    async def _run_cmd(
        self,
        cmd:             list[str],
        extract_authors: str | None = None,
    ) -> _GitResult:
        """
        Run a git command with timeout. Kill on timeout.

        FIX: SIGKILL (not SIGTERM) — git won't respond to SIGTERM when hung.
        FIX: returncode checked properly — None means process didn't finish.
        """
        proc = await _create_proc(cmd)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except TimeoutError:
            _kill_proc(proc)
            raise

        # FIX: proc.returncode can be None if event loop is in a bad state
        exit_code = proc.returncode if proc.returncode is not None else -1
        output    = stdout.decode("utf-8", errors="replace")
        err_out   = stderr.decode("utf-8", errors="replace")

        authors: list[str] = []
        if extract_authors == "blame":
            authors = _extract_blame_authors(output)
        elif extract_authors == "log":
            authors = _extract_log_authors(output)

        return _GitResult(
            command=   " ".join(cmd),
            output=    output,
            stderr=    err_out,
            exit_code= exit_code,
            authors=   authors,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print_rich_hotspots(
        self,
        top:    list[HotspotEntry],
        target: str,
        params: GitBlameInput,
    ) -> None:
        """Print Rich table to stderr. No-op if Rich unavailable or not a TTY."""
        if self._console is None or not _RICH_AVAILABLE:
            return

        # Local import to satisfy strict type checkers when rich is available.
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        _RISK_STYLE = {
            RiskLevel.CRITICAL: "bold red",
            RiskLevel.HIGH:     "orange3",
            RiskLevel.MEDIUM:   "yellow",
            RiskLevel.LOW:      "green",
        }

        has_critical  = any(h.risk_level == RiskLevel.CRITICAL for h in top)
        title_style   = "bold red" if has_critical else "bold yellow"

        table = Table(
            title=f"Code Hotspots: {target}",
            title_style=title_style,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("File",  style="cyan", width=45, no_wrap=True)
        table.add_column("Churn", justify="right", style="green")
        table.add_column("Auth",  justify="right", style="yellow")
        table.add_column("Risk",  justify="center")
        table.add_column("Score", justify="right", style="blue")

        for h in top:
            display = h.file if len(h.file) <= 44 else "..." + h.file[-41:]
            table.add_row(
                display,
                str(h.changes),
                str(h.authors),
                Text(h.risk_level.value, style=_RISK_STYLE[h.risk_level]),
                f"{h.complexity_score:.1f}",
            )

        self._console.print(table)
        period = params.since or f"last {params.n} commits"
        self._console.print(f"[dim]Period: {period}[/dim]")
        self._console.print("[dim]CRITICAL: >10 changes & >3 authors[/dim]")

    @staticmethod
    def _format_hotspots_text(top: list[HotspotEntry]) -> str:
        """Plain-text table for LLM context. No ANSI codes."""
        header = f"{'File':<45} {'Churn':>6} {'Auth':>5} {'Risk':>8} {'Score':>6}"
        sep    = "-" * 72
        rows: list[str] = []
        for h in top:
            name = h.file if len(h.file) <= 44 else "..." + h.file[-41:]
            rows.append(
                f"{name:<45} {h.changes:>6} {h.authors:>5} "
                f"{h.risk_level.value:>8} {h.complexity_score:>6.1f}"
            )
        footer = [
            "",
            "Churn = commits touching the file in the analysis window.",
            "Score = churn x (1 + sqrt(authors))  (coordination complexity).",
            "CRITICAL (>10 churn, >3 authors) -> audit before fuzzing.",
        ]
        return "\n".join([header, sep, *rows, *footer])

    def _maybe_scrub(self, text: str) -> str:
        return _scrub(text) if self.enable_scrubbing else text

    def _truncate(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= self.max_output_lines:
            return text
        return "\n".join(lines[:self.max_output_lines]) + (
            f"\n...[{len(lines) - self.max_output_lines} lines truncated]"
        )

    def get_typed_metrics(self) -> MetricsSnapshot:
        """Typed metrics (extends BaseTool.get_metrics())."""
        raw = self.get_metrics()
        return MetricsSnapshot(**raw)

    def __repr__(self) -> str:
        return f"GitBlameTool(available={self._available}, timeout={self.timeout}s)"


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------

def _extract_blame_authors(output: str) -> list[str]:
    seen: set[str] = set()
    for line in output.splitlines():
        if line.startswith("author "):
            name = line[7:].strip()
            if name:
                seen.add(name)
    return sorted(seen)[:10]


def _extract_log_authors(output: str) -> list[str]:
    seen: set[str] = set()
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            name = parts[1].strip()
            if name:
                seen.add(name)
    return sorted(seen)[:10]


def _scrub(text: str) -> str:
    text = re.sub(r"/home/[^/\s]+",     "/workspace", text)
    text = re.sub(r"/Users/[^/\s]+",    "/workspace", text)
    text = re.sub(r"C:\\Users\\[^\\]+", r"C:\\workspace", text)
    text = re.sub(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
    )
    # Shorten full 40-char SHAs; keep 7 chars for cross-reference
    text = re.sub(r"\b([0-9a-f]{7})[0-9a-f]{33}\b", r"\1...", text)
    return text


async def _create_proc(cmd: list[str]) -> asyncio.subprocess.Process:
    kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    if sys.platform != "win32":
        kwargs["preexec_fn"] = os.setsid
    return await asyncio.create_subprocess_exec(*cmd, **kwargs)


def _kill_proc(proc: asyncio.subprocess.Process) -> None:
    """Kill subprocess immediately (SIGKILL, not SIGTERM)."""
    if sys.platform == "win32":
        proc.kill()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        with contextlib.suppress(Exception):
            proc.kill()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_git_blame_tool(
    git_path:        str | None = None,
    timeout_seconds: float         = 30.0,
    **kwargs: Any,
) -> GitBlameTool:
    return GitBlameTool(git_path=git_path, timeout_seconds=timeout_seconds, **kwargs)


# ---------------------------------------------------------------------------
# One-shot helpers (scripts / tests)
# ---------------------------------------------------------------------------

async def get_file_blame(
    file_path: str,
    line:      int | None = None,
) -> dict[str, Any]:
    tool   = create_git_blame_tool()
    result = await tool.execute({"command": "blame", "file": file_path, "line": line})
    return {
        "success": result.success,
        "output":  result.output,
        "authors": result.data.get("authors", []) if result.success else [],
        "error":   result.error,
    }


async def get_commit_log(
    n:      int           = 10,
    path:   str | None = None,
    author: str | None = None,
    grep:   str | None = None,
    since:  str | None = None,
) -> dict[str, Any]:
    tool   = create_git_blame_tool()
    result = await tool.execute({
        "command": "log",
        "n": n, "path": path,
        "author": author, "grep": grep, "since": since,
    })
    return {
        "success": result.success,
        "output":  result.output,
        "authors": result.data.get("authors", []) if result.success else [],
        "error":   result.error,
    }


async def get_hotspots(
    path:       str           = ".",
    n_commits:  int           = 100,
    since:      str | None = None,
    extensions: list[str] | None = None,
) -> dict[str, Any]:
    tool   = create_git_blame_tool()
    result = await tool.execute({
        "command": "hotspots",
        "path": path, "n": n_commits,
        "since": since, "extensions": extensions,
    })
    return {
        "success":  result.success,
        "output":   result.output,
        "hotspots": result.data.get("hotspots", []) if result.success else [],
        "error":    result.error,
    }
