# keryx/tools/codeql.py
# CodeQL tool with deep flow analysis, build command support, and resource isolation.
# Sovereign, async-safe, memory-protected.

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from .Toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.codeql")


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------

def get_safe_ram_mb() -> int:
    try:
        return int(psutil.virtual_memory().available / (1024 * 1024) * 0.7)
    except Exception:
        return 4096


def kill_process_tree(pid: int, timeout: float = 5.0) -> None:
    """Kill process and all children to prevent zombies."""
    try:
        parent   = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.terminate()
        gone, alive = psutil.wait_procs(children, timeout=timeout / 2)
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        try:
            parent.terminate()
            parent.wait(timeout=timeout / 2)
        except psutil.TimeoutExpired:
            parent.kill()
            parent.wait()
    except psutil.NoSuchProcess:
        pass
    except Exception as exc:
        logger.warning(f"[CodeQLTool] kill_process_tree failed: {exc}")


# ---------------------------------------------------------------------------
# Privacy scrubbing
# FIX 9: removed r'[a-f0-9]{32,}' — matches hex in code/addresses (false positives)
# ---------------------------------------------------------------------------

_SCRUB_PATTERNS = [
    (re.compile(r'/home/[^/\s]+', re.IGNORECASE),       '/home/user'),
    (re.compile(r'/Users/[^/\s]+', re.IGNORECASE),      '/Users/user'),
    (re.compile(r'C:\\Users\\[^\\\s]+', re.IGNORECASE),  'C:\\Users\\user'),
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), 'x.x.x.x'),
    (re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'),          'user@example.com'),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'),                 'sk-***'),
]

def scrub_codeql_output(text: str) -> str:
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# CodeQLFinding
# ---------------------------------------------------------------------------

@dataclass
class CodeQLFinding:
    rule_id:    str
    message:    str
    file:       str
    line:       int
    severity:   str                  = "warning"
    flow_steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id":    self.rule_id,
            "message":    self.message[:300],
            "file":       self.file,
            "line":       self.line,
            "severity":   self.severity,
            "flow_steps": self.flow_steps[:10],
        }

    def summary(self) -> str:
        lines = [f"{self.rule_id}: {self.message[:200]}",
                 f"  at {self.file}:{self.line}"]
        if self.flow_steps:
            lines.append("  Data flow:")
            for i, s in enumerate(self.flow_steps[:5], 1):
                lines.append(f"    {i}. {s.get('file','?')}:{s.get('line','?')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CodeQLTool
# ---------------------------------------------------------------------------

class CodeQLTool(BaseTool):
    """
    Hardened CodeQL integration:
    - SARIF output for rich flow analysis
    - Explicit build command (no autobuild guessing)
    - RAM limits via --ram flag
    - Process tree cleanup (no zombies)
    - Context manager for guaranteed cleanup
    """

    name        = "codeql_query"
    description = "Run CodeQL security queries with data flow analysis"

    QUERY_BUNDLES = {
        "security-extended":      "codeql/cpp-queries:Security/CWE",
        "security-and-quality":   "codeql/cpp-queries:Security and Quality",
        "buffer-overflow":        "codeql/cpp-queries:Security/CWE/CWE-119",
        "use-after-free":         "codeql/cpp-queries:Security/CWE/CWE-416",
        "sql-injection":          "codeql/cpp-queries:Security/CWE/CWE-089",
        "command-injection":      "codeql/cpp-queries:Security/CWE/CWE-078",
    }

    def __init__(
        self,
        codeql_path:      str | None   = None,
        default_db_path:  str | None   = None,
        timeout_seconds:  float           = 180.0,
        max_results:      int             = 50,
        enable_scrubbing: bool            = True,
        auto_create_db:   bool            = True,
        persistent_db:    bool            = False,
        ram_limit_mb:     int | None   = None,
    ):
        self.codeql_path     = codeql_path or shutil.which("codeql") or "codeql"
        self.default_db_path = Path(default_db_path or ".keryx_codeql_db")
        self.timeout         = timeout_seconds
        self.max_results     = max_results
        self.enable_scrubbing = enable_scrubbing
        self.auto_create_db  = auto_create_db
        self.persistent_db   = persistent_db
        self.ram_limit_mb    = ram_limit_mb or get_safe_ram_mb()

        self._available = bool(shutil.which(self.codeql_path))

        # FIX 7: track (db_path, language, build_command_hash) to detect stale DBs
        self._db_session_keys: set[str]  = set()
        self._temp_dbs:         set[str] = set()   # for cleanup

        if not self._available:
            logger.warning(f"[CodeQLTool] Binary not found: {self.codeql_path}")
        logger.info(
            f"[CodeQLTool] Initialized | available={self._available} | "
            f"ram={self.ram_limit_mb}MB | timeout={timeout_seconds}s"
        )

    def is_available(self) -> bool:
        return self._available

    def list_available_queries(self) -> dict[str, str]:
        return self.QUERY_BUNDLES.copy()

    # ------------------------------------------------------------------
    # FIX 8: get_command() for native subprocess execution by agent.py
    # ------------------------------------------------------------------

    def get_command(self, action_input: dict[str, Any]) -> list[str] | None:
        """
        Return None — CodeQL runs as Python-managed subprocess internally.
        agent.py falls back to run_in_executor → toolbox.execute().
        """
        return None

    # ------------------------------------------------------------------
    # Context manager for guaranteed cleanup
    # FIX 6: replaces unreliable __del__
    # ------------------------------------------------------------------

    def __enter__(self) -> CodeQLTool:
        return self

    def __exit__(self, *_: Any) -> bool:
        self.cleanup()
        return False

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_input: dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        if not self._available:
            return ToolResult(success=False, output="CodeQL not installed", error="codeql_not_found")

        query = action_input.get("query")
        if not query:
            return ToolResult(success=False, output="Missing 'query'", error="query_missing")

        resolved = self.QUERY_BUNDLES.get(query, query)
        language      = action_input.get("language", "cpp")
        build_command = action_input.get("build_command")
        db_path       = str(action_input.get("database") or self.default_db_path)
        include_flow  = action_input.get("include_flow", True)
        max_results   = action_input.get("max_results", self.max_results)

        if language in ("cpp", "c", "c++") and not build_command and self.auto_create_db:
            logger.warning("[CodeQLTool] C/C++ without build_command — likely to fail")

        start = time.time()

        try:
            # FIX 7: DB session key includes language + build_command hash
            db_key = f"{db_path}:{language}:{hash(build_command or '')}"
            if self.auto_create_db and db_key not in self._db_session_keys:
                ok = await self._create_database(db_path, language, build_command, context)
                if not ok:
                    return ToolResult(
                        success=False,
                        output="Database creation failed. Check build logs.",
                        error="db_creation_failed",
                    )
                self._db_session_keys.add(db_key)
                if not self.persistent_db:
                    self._temp_dbs.add(db_path)

            exit_code, stdout, stderr = await self._run_query(resolved, db_path)
            duration_ms = (time.time() - start) * 1000

            if self.enable_scrubbing:
                stdout = scrub_codeql_output(stdout)
                stderr = scrub_codeql_output(stderr)

            if exit_code != 0:
                return ToolResult(
                    success=False,
                    output=stderr[:500] or stdout[:500],
                    error=f"exit_code_{exit_code}",
                )

            findings = self._parse_sarif_findings(stdout, max_results, include_flow)
            summary  = self._build_summary(findings)

            logger.info(
                f"[CodeQLTool] findings={len(findings)} | "
                f"flow_steps={sum(len(f.flow_steps) for f in findings)} | "
                f"{duration_ms:.0f}ms"
            )
            return ToolResult(
                success=True,
                output=summary,
                data={
                    "findings":    [f.to_dict() for f in findings],
                    "count":       len(findings),
                    "duration_ms": duration_ms,
                    "query":       query,
                },
                metadata={
                    "tool":             self.name,
                    "findings_count":   len(findings),
                    "has_flow_analysis": include_flow,
                },
            )

        except TimeoutError:
            return ToolResult(success=False, output=f"Timeout after {self.timeout}s", error="timeout")
        except Exception as exc:
            logger.error(f"[CodeQLTool] Unexpected: {exc}", exc_info=True)
            return ToolResult(success=False, output=str(exc), error="unexpected")

    # ------------------------------------------------------------------
    # Database creation
    # ------------------------------------------------------------------

    async def _create_database(
        self,
        db_path:       str,
        language:      str,
        build_command: str | None,
        context:       Any,
    ) -> bool:
        source_root = str(getattr(context, 'target_path', '.')) if context else '.'
        db_dir = Path(db_path)

        if db_dir.exists() and not self.persistent_db:
            shutil.rmtree(db_dir, ignore_errors=True)

        cmd = [
            self.codeql_path, "database", "create", db_path,
            "--language", language,
            "--source-root", source_root,
            "--ram", str(self.ram_limit_mb),
        ]
        if build_command:
            cmd += ["--command", build_command]
        else:
            cmd.append("--no-run-unnecessary-builds")

        logger.info(f"[CodeQLTool] Creating DB | lang={language}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            if proc.returncode == 0:
                return True
            logger.error(f"[CodeQLTool] DB creation failed: {stderr.decode()[:400]}")
            return False
        except TimeoutError:
            kill_process_tree(proc.pid)
            return False

    # ------------------------------------------------------------------
    # Query execution
    # FIX 3: 'database analyze' doesn't support --output - (stdout).
    #         Write to temp file, then read.
    # FIX 4: BQRS → SARIF requires 'bqrs interpret', not 'bqrs decode'.
    # ------------------------------------------------------------------

    async def _run_query(
        self,
        query:   str,
        db_path: str,
    ) -> tuple[int, str, str]:
        """Route to suite-analysis or single-query path."""
        is_suite = query.startswith("codeql/") or ":" in query
        if is_suite:
            return await self._run_suite(query, db_path)
        return await self._run_single_query(query, db_path)

    async def _run_suite(self, query: str, db_path: str) -> tuple[int, str, str]:
        """
        Run a query suite via 'database analyze'.
        FIX 3: analyze writes to a file, not stdout. Use temp file.
        """
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as tmp:
            sarif_path = tmp.name

        cmd = [
            self.codeql_path, "database", "analyze",
            db_path, query,
            "--format", "sarif-latest",
            "--output", sarif_path,       # FIX 3: file, not "-"
            "--ram", str(self.ram_limit_mb),
            "--threads", str(min(4, os.cpu_count() or 2)),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            if proc.returncode != 0:
                return proc.returncode, "", stderr.decode('utf-8', errors='replace')
            try:
                content = Path(sarif_path).read_text(encoding='utf-8')
            except Exception:
                content = ""
            return 0, content, ""
        except TimeoutError:
            kill_process_tree(proc.pid)
            raise
        finally:
            Path(sarif_path).unlink(missing_ok=True)

    async def _run_single_query(self, query: str, db_path: str) -> tuple[int, str, str]:
        """
        Run single .ql file: query run → BQRS → bqrs interpret → SARIF.
        FIX 4: 'bqrs decode' doesn't produce SARIF — use 'bqrs interpret'.
        """
        with tempfile.NamedTemporaryFile(suffix=".bqrs", delete=False) as tmp:
            bqrs_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as tmp:
            sarif_path = tmp.name

        try:
            # Step 1: query run → BQRS
            run_cmd = [
                self.codeql_path, "query", "run",
                query,
                "--database", db_path,
                "--output", bqrs_path,
                "--ram", str(self.ram_limit_mb),
            ]
            proc = await asyncio.create_subprocess_exec(
                *run_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                if proc.returncode != 0:
                    return proc.returncode, "", stderr.decode()
            except TimeoutError:
                kill_process_tree(proc.pid)
                raise

            # Step 2: bqrs interpret → SARIF  (FIX 4: was 'bqrs decode')
            interp_cmd = [
                self.codeql_path, "bqrs", "interpret",
                bqrs_path,
                "--format", "sarif-latest",
                "--output", sarif_path,
                "--query-metadata-path", query,  # required for interpret
                "--sarif-add-query-help",
            ]
            proc2 = await asyncio.create_subprocess_exec(
                *interp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr2 = await asyncio.wait_for(proc2.communicate(), timeout=30.0)
                if proc2.returncode != 0:
                    return proc2.returncode, "", stderr2.decode()
                content = Path(sarif_path).read_text(encoding='utf-8')
                return 0, content, ""
            except TimeoutError:
                kill_process_tree(proc2.pid)
                raise

        finally:
            Path(bqrs_path).unlink(missing_ok=True)
            Path(sarif_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # SARIF parsing
    # FIX 5: build rule_id → severity dict once, not O(R*N) per result
    # ------------------------------------------------------------------

    def _parse_sarif_findings(
        self,
        sarif_json:   str,
        limit:        int,
        include_flow: bool,
    ) -> list[CodeQLFinding]:
        if not sarif_json.strip():
            return []
        try:
            data     = json.loads(sarif_json)
            findings = []
            for run in data.get("runs", []):
                # FIX 5: build severity lookup once per run
                rule_severity: dict[str, str] = {}
                for rule in run.get("tool", {}).get("driver", {}).get("rules", []):
                    rid = rule.get("id", "")
                    rule_severity[rid] = (
                        rule.get("defaultConfiguration", {}).get("level", "warning")
                    )

                for result in run.get("results", [])[:limit]:
                    rule_id = result.get("ruleId", "unknown")
                    message = result.get("message", {}).get("text", "")
                    locs    = result.get("locations", [])
                    if not locs:
                        continue
                    loc  = locs[0].get("physicalLocation", {})
                    file = loc.get("artifactLocation", {}).get("uri", "unknown")
                    line = loc.get("region", {}).get("startLine", 0)

                    finding = CodeQLFinding(
                        rule_id=rule_id,
                        message=message,
                        file=file,
                        line=line,
                        severity=rule_severity.get(rule_id, "warning"),
                    )
                    if include_flow:
                        finding.flow_steps = self._extract_flow_steps(result)
                    findings.append(finding)
            return findings
        except json.JSONDecodeError as exc:
            logger.warning(f"[CodeQLTool] SARIF parse error: {exc}")
            return []

    def _extract_flow_steps(self, result: dict) -> list[dict[str, Any]]:
        steps: list[dict] = []
        for flow in result.get("codeFlows", []):
            for thread in flow.get("threadFlows", []):
                for loc_entry in thread.get("locations", []):
                    ploc = loc_entry.get("location", {}).get("physicalLocation", {})
                    step = {
                        "file":   ploc.get("artifactLocation", {}).get("uri", "?"),
                        "line":   ploc.get("region", {}).get("startLine", 0),
                        "column": ploc.get("region", {}).get("startColumn", 0),
                    }
                    if not steps or steps[-1] != step:
                        steps.append(step)
        return steps

    def _build_summary(self, findings: list[CodeQLFinding]) -> str:
        if not findings:
            return "No vulnerabilities found by CodeQL."
        lines = [f"Found {len(findings)} potential issue(s):\n"]
        for i, f in enumerate(findings[:10], 1):
            lines.append(f"{i}. {f.summary()}\n")
        if len(findings) > 10:
            lines.append(f"... and {len(findings) - 10} more")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup — via context manager, not __del__
    # FIX 6: __del__ removed, use 'with CodeQLTool(...) as tool:'
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove temporary databases. Call explicitly or use context manager."""
        if self.persistent_db:
            return
        for db_path in list(self._temp_dbs):
            try:
                shutil.rmtree(db_path, ignore_errors=True)
                logger.debug(f"[CodeQLTool] Cleaned up: {db_path}")
            except Exception as exc:
                logger.warning(f"[CodeQLTool] Cleanup failed for {db_path}: {exc}")
        self._temp_dbs.clear()

    def __repr__(self) -> str:
        return (
            f"CodeQLTool(available={self._available}, "
            f"ram={self.ram_limit_mb}MB, persistent={self.persistent_db})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_codeql_tool(
    codeql_path:     str | None = None,
    timeout_seconds: float         = 180.0,
    persistent_db:   bool          = False,
    **kwargs,
) -> CodeQLTool:
    return CodeQLTool(
        codeql_path=codeql_path,
        timeout_seconds=timeout_seconds,
        persistent_db=persistent_db,
        **kwargs,
    )
