# keryx/tools/git_blame.py
# Git Blame / Log / Hotspots Tool for KeryxHunter.
# Sovereign, async-safe, timeout-guaranteed, privacy scrubbing.
# Supports: blame, log, hotspots (churn analysis).

from __future__ import annotations

import asyncio
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
from typing import Any, Dict, List, Optional

from .toolbox import ToolResult, BaseTool

logger = logging.getLogger("keryx.tools.git_blame")


@dataclass
class _GitResult:
    """Internal result from a raw git subprocess call."""
    command:    str
    output:     str
    stderr:     str
    exit_code:  int
    authors:    List[str] = field(default_factory=list)


class GitBlameTool(BaseTool):
    """
    Git history tool.

    Supports:
    - git blame <file>[:line]
    - git log --oneline -n <count> <path>
    - git log --grep / --author / --since
    - hotspots: churn analysis for Regression-Driven Vulnerability Research

    Real process kill on timeout + privacy scrubbing.
    """

    name        = "git_blame"
    description = (
        "Run git blame, git log, or churn analysis on a repository. "
        "Helps the agent understand code history, authorship, "
        "and identify high-risk hotspots (frequently changed files)."
    )

    # Security-critical file extensions for hotspot analysis
    _DEFAULT_EXTENSIONS = frozenset([
        ".c", ".cpp", ".cc", ".cxx",
        ".h", ".hpp",
        ".rs", ".go", ".java",
    ])

    def __init__(
        self,
        git_path:         Optional[str] = None,
        timeout_seconds:  float         = 30.0,
        max_output_lines: int           = 200,
        enable_scrubbing: bool          = True,
    ) -> None:
        super().__init__()
        # BUG-FIX 1: Do NOT redeclare _call_count / _success_count — BaseTool owns them.
        # All counting goes through self._record_call(success, duration_ms).

        # BUG-FIX 2: resolve git binary once; availability = binary exists on disk.
        resolved = git_path or shutil.which("git")
        self.git_path    = resolved or "git"
        self._available  = resolved is not None

        self.timeout          = timeout_seconds
        self.max_output_lines = max_output_lines
        self.enable_scrubbing = enable_scrubbing

        if not self._available:
            logger.warning(
                "[GitBlameTool] git binary not found — tool will be disabled."
            )
        else:
            logger.info(
                "[GitBlameTool] Initialized | git=%s | timeout=%.0fs | max_lines=%d",
                self.git_path, timeout_seconds, max_output_lines,
            )

    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_input: Dict[str, Any],
        context:      Any = None,
    ) -> ToolResult:
        """
        Expected action_input keys:

        command: "blame" | "log" | "hotspots"  (default: "blame")

        blame:
            file (str, required)   — path to file
            line (int, optional)   — single line to inspect

        log:
            n      (int)           — number of commits (default 10)
            path   (str)           — path filter
            author (str)           — author filter
            grep   (str)           — message grep pattern
            since  (str)           — date filter, e.g. "1 week ago"

        hotspots:
            path        (str)      — root to analyse (default ".")
            n           (int)      — commits to scan (default 100)
            since       (str)      — date filter
            extensions  (list)     — file extensions (default: C/C++/Rust/Go/Java)
            min_changes (int)      — minimum churn to include (default 2)
        """
        if not self._available:
            return ToolResult(
                success=False,
                output="git not available. Install git and ensure it is in PATH.",
                error="git_not_found",
            )

        cmd_type = action_input.get("command", "blame").lower()

        if cmd_type not in ("blame", "log", "hotspots"):
            return ToolResult(
                success=False,
                output=f"Unknown command: {cmd_type!r}. Use 'blame', 'log', or 'hotspots'.",
                error="unknown_command",
            )

        start = time.time()

        try:
            if cmd_type == "hotspots":
                # hotspots has its own timing / _record_call inside
                return await self._run_hotspots(action_input)

            if cmd_type == "blame":
                result = await self._git_blame(action_input)
            else:
                result = await self._git_log(action_input)

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start) * 1000
            self._record_call(success=False, duration_ms=duration_ms)
            logger.error("[GitBlameTool] %s timed out after %.0fs", cmd_type, self.timeout)
            return ToolResult(
                success=False,
                output=f"git {cmd_type} timed out after {self.timeout:.0f}s",
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
                error=f"git_{cmd_type}_failed",
                data={"exit_code": result.exit_code},
            )

        output = self._maybe_scrub(result.output)
        output = self._truncate(output)

        file_path = action_input.get("file") or action_input.get("path")
        return ToolResult(
            success=True,
            output=output,
            data={
                "command":          cmd_type,
                "target":           file_path,
                "lines":            len(output.splitlines()),
                "authors":          result.authors,
                "execution_time_ms": round(duration_ms, 1),
                "exit_code":        result.exit_code,
            },
            metadata={
                "tool":    self.name,
                "command": cmd_type,
                "target":  file_path,
            },
        )

    # ------------------------------------------------------------------
    # blame
    # ------------------------------------------------------------------

    async def _git_blame(self, action_input: Dict[str, Any]) -> _GitResult:
        file_path = action_input.get("file", "")
        line_num  = action_input.get("line")

        if not file_path:
            raise ValueError("Missing 'file' for git blame")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        cmd = [self.git_path, "blame", "--line-porcelain"]
        if line_num is not None:
            cmd.extend(["-L", f"{line_num},{line_num}"])
        cmd.append(file_path)

        return await self._run_cmd(cmd, extract_authors="blame")

    # ------------------------------------------------------------------
    # log
    # ------------------------------------------------------------------

    async def _git_log(self, action_input: Dict[str, Any]) -> _GitResult:
        n_commits = int(action_input.get("n", 10))
        log_path  = action_input.get("path")
        author    = action_input.get("author")
        grep      = action_input.get("grep")
        since     = action_input.get("since")

        cmd = [
            self.git_path, "log",
            f"-{n_commits}",
            "--no-color",
            "--format=%h|%an|%ae|%s",
        ]
        if author:
            cmd.extend(["--author", author])
        if grep:
            cmd.extend(["--grep", grep])
        if since:
            cmd.extend(["--since", since])
        if log_path:
            cmd.append(log_path)

        return await self._run_cmd(cmd, extract_authors="log")

    # ------------------------------------------------------------------
    # hotspots
    # ------------------------------------------------------------------

    async def _run_hotspots(self, action_input: Dict[str, Any]) -> ToolResult:
        """
        Churn analysis: files with high change frequency and many authors.

        Complexity = changes × (1 + √authors)
        High changes + many authors = coordination complexity = regression risk.
        """
        path       = action_input.get("path", ".")
        n_commits  = int(action_input.get("n", 100))
        since      = action_input.get("since")
        extensions = frozenset(action_input.get("extensions") or self._DEFAULT_EXTENSIONS)
        min_churn  = int(action_input.get("min_changes", 2))

        start = time.time()

        try:
            # ── Step 1: collect changed file names ─────────────────────────
            cmd = [
                self.git_path, "log",
                f"-n{n_commits}",
                "--pretty=format:",   # suppress commit lines
                "--name-only",
            ]
            if since:
                cmd.extend(["--since", since])
            cmd.append(path)

            raw = await self._run_cmd(cmd)
            if raw.exit_code != 0:
                self._record_call(success=False, duration_ms=(time.time() - start) * 1000)
                return ToolResult(
                    success=False,
                    output=self._maybe_scrub(raw.stderr),
                    error="git_log_failed",
                    data={"exit_code": raw.exit_code},
                )

            # ── Step 2: count churn per file ───────────────────────────────
            churn: Dict[str, int] = {}
            for line in raw.output.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not any(line.endswith(ext) for ext in extensions):
                    continue
                norm = os.path.normpath(line)
                churn[norm] = churn.get(norm, 0) + 1

            # ── Step 3: fetch authors for top-15 candidates ────────────────
            candidates = sorted(churn.items(), key=lambda kv: kv[1], reverse=True)[:15]
            authors_by_file: Dict[str, set] = {}

            for filepath, changes in candidates:
                if changes < min_churn:
                    continue
                # BUG-FIX 5: build command cleanly, no fragile list.insert()
                author_cmd = [self.git_path, "log", f"-n{n_commits}"]
                if since:
                    author_cmd.extend(["--since", since])
                author_cmd.extend(["--format=%an", "--follow", "--", filepath])

                ar = await self._run_cmd(author_cmd)
                if ar.exit_code == 0:
                    authors_by_file[filepath] = {
                        ln.strip() for ln in ar.output.splitlines() if ln.strip()
                    }

            # ── Step 4: score and classify ─────────────────────────────────
            hotspots = []
            for filepath, changes in churn.items():
                if changes < min_churn:
                    continue
                authors     = authors_by_file.get(filepath, set())
                n_authors   = len(authors)
                complexity  = changes * (1.0 + math.sqrt(n_authors))

                if changes > 10 and n_authors > 3:
                    risk = "CRITICAL"
                elif changes > 5 and n_authors > 2:
                    risk = "HIGH"
                elif changes > 3:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

                hotspots.append({
                    "file":             filepath,
                    "changes":          changes,
                    "authors":          n_authors,
                    "top_authors":      sorted(authors)[:3],
                    "complexity_score": round(complexity, 1),
                    "risk_level":       risk,
                })

            hotspots.sort(key=lambda h: h["complexity_score"], reverse=True)
            top = hotspots[:10]

            # ── Step 5: format for LLM ─────────────────────────────────────
            header = (
                f"{'File':<45} {'Churn':>6} {'Auth':>5} {'Risk':>8} {'Score':>6}"
            )
            separator = "-" * 72
            rows = []
            for h in top:
                display = self._maybe_scrub(h["file"])
                if len(display) > 44:
                    display = "..." + display[-41:]
                rows.append(
                    f"{display:<45} {h['changes']:>6} {h['authors']:>5} "
                    f"{h['risk_level']:>8} {h['complexity_score']:>6.1f}"
                )

            footer = [
                "",
                "Churn = commits touching the file in the analysis window.",
                "Score = churn × (1 + √authors)  (coordination complexity).",
                "CRITICAL (>10 churn, >3 authors) → audit before fuzzing.",
            ]

            output = "\n".join([header, separator, *rows, *footer])
            output = self._truncate(output)

            duration_ms = (time.time() - start) * 1000
            self._record_call(success=True, duration_ms=duration_ms)

            return ToolResult(
                success=True,
                output=output,
                data={
                    "hotspots":            top,
                    "total_files_scanned": len(churn),
                    "analysis_period":     since or f"last {n_commits} commits",
                    "execution_time_ms":   round(duration_ms, 1),
                },
                metadata={
                    "tool":             self.name,
                    "command":          "hotspots",
                    "target":           path,
                    "records_returned": len(top),
                },
            )

        except asyncio.TimeoutError:
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
    # Low-level subprocess helpers
    # ------------------------------------------------------------------

    async def _run_cmd(
        self,
        cmd:             List[str],
        extract_authors: Optional[str] = None,
    ) -> _GitResult:
        """
        Run a git command, wait up to self.timeout, kill on timeout.

        BUG-FIX 4: SIGKILL on timeout (not SIGTERM) — git is unlikely to respond
        to SIGTERM when hung on a remote or a large pack-file operation.
        BUG-FIX 3: returncode is checked properly — None means process didn't finish.
        """
        proc = await self._create_proc(cmd)
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            self._kill_proc(proc)
            raise

        # BUG-FIX 3: proc.returncode can be None if communicate() contract is broken
        exit_code = proc.returncode if proc.returncode is not None else -1
        output    = stdout.decode("utf-8", errors="replace")
        err_out   = stderr.decode("utf-8", errors="replace")

        authors: List[str] = []
        if extract_authors == "blame":
            authors = _extract_blame_authors(output)
        elif extract_authors == "log":
            authors = _extract_log_authors(output)

        return _GitResult(
            command=  " ".join(cmd),
            output=   output,
            stderr=   err_out,
            exit_code=exit_code,
            authors=  authors,
        )

    @staticmethod
    async def _create_proc(cmd: List[str]) -> asyncio.subprocess.Process:
        kwargs: Dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if sys.platform != "win32":
            kwargs["preexec_fn"] = os.setsid   # new process group for clean kill
        return await asyncio.create_subprocess_exec(*cmd, **kwargs)

    @staticmethod
    def _kill_proc(proc: asyncio.subprocess.Process) -> None:
        """Kill the subprocess (and its group on Unix) immediately."""
        # BUG-FIX 4: SIGKILL not SIGTERM; no magic numbers
        if sys.platform == "win32":
            proc.kill()
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _maybe_scrub(self, text: str) -> str:
        return _scrub(text) if self.enable_scrubbing else text

    def _truncate(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= self.max_output_lines:
            return text
        kept = lines[:self.max_output_lines]
        kept.append(f"...[{len(lines) - self.max_output_lines} lines truncated]")
        return "\n".join(kept)

    def __repr__(self) -> str:
        return f"GitBlameTool(available={self._available}, timeout={self.timeout}s)"


# ---------------------------------------------------------------------------
# Pure-function helpers (no self dependency — easier to test)
# ---------------------------------------------------------------------------

def _extract_blame_authors(output: str) -> List[str]:
    seen: set = set()
    for line in output.splitlines():
        if line.startswith("author "):
            name = line[7:].strip()
            if name:
                seen.add(name)
    return sorted(seen)[:10]


def _extract_log_authors(output: str) -> List[str]:
    seen: set = set()
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            name = parts[1].strip()
            if name:
                seen.add(name)
    return sorted(seen)[:10]


def _scrub(text: str) -> str:
    """Privacy scrubbing: paths, emails, full commit hashes."""
    # Absolute home paths
    text = re.sub(r"/home/[^/\s]+",        "/workspace", text)
    text = re.sub(r"/Users/[^/\s]+",       "/workspace", text)
    text = re.sub(r"C:\\Users\\[^\\]+",    r"C:\\workspace", text)
    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
    )
    # Full 40-char commit SHAs → keep first 7 for reference
    text = re.sub(r"\b([0-9a-f]{7})[0-9a-f]{33}\b", r"\1…", text)
    return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_git_blame_tool(
    git_path:        Optional[str] = None,
    timeout_seconds: float         = 30.0,
    **kwargs: Any,
) -> GitBlameTool:
    return GitBlameTool(git_path=git_path, timeout_seconds=timeout_seconds, **kwargs)


# ---------------------------------------------------------------------------
# Standalone one-shot helpers (scripts / tests)
# ---------------------------------------------------------------------------

async def get_file_blame(
    file_path: str,
    line:      Optional[int] = None,
) -> Dict[str, Any]:
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
    path:   Optional[str] = None,
    author: Optional[str] = None,
    grep:   Optional[str] = None,
    since:  Optional[str] = None,
) -> Dict[str, Any]:
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
    since:      Optional[str] = None,
    extensions: Optional[List[str]] = None,
) -> Dict[str, Any]:
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
