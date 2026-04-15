# keryx/tools/read_file.py
# Simple file reading tool — critical for agent's early steps.
# Supports path traversal protection, encoding fallback, timeout, and line limits.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .Toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.read_file")


class ReadFileTool(BaseTool):
    """
    Simple file reading tool.
    Used by agent in early steps to explore the codebase.
    """

    name = "read_file"
    description = (
        "Read the content of a file from the target codebase. "
        "Returns the file content as text."
    )

    def __init__(
        self,
        max_chars: int = 100_000,
        max_lines: int | None = None,
        encoding: str = "utf-8",
        allowed_root: str | None = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__()
        self.max_chars = max_chars
        self.max_lines = max_lines
        self.encoding = encoding
        self.allowed_root = Path(allowed_root).resolve() if allowed_root else None
        self.timeout = timeout_seconds

        logger.info(
            f"[ReadFileTool] Initialized | max_chars={max_chars} | "
            f"max_lines={max_lines} | encoding={encoding} | timeout={timeout_seconds}s"
        )

    async def execute(
        self,
        action_input: dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        file_path = action_input.get("file_path") or action_input.get("path")
        encoding = action_input.get("encoding", self.encoding)
        max_chars = action_input.get("max_chars", self.max_chars)
        max_lines = action_input.get("max_lines", self.max_lines)
        start_line = action_input.get("start_line")   # 1-based, inclusive
        end_line   = action_input.get("end_line")     # 1-based, inclusive

        if not file_path:
            return ToolResult(
                success=False,
                output="Missing 'file_path' in action_input",
                error="file_path_missing",
            )

        try:
            path = Path(file_path).resolve()

            # Security: restrict path traversal
            if self.allowed_root:
                try:
                    path.relative_to(self.allowed_root)
                except ValueError:
                    logger.warning(f"Path traversal attempt blocked: {file_path}")
                    return ToolResult(
                        success=False,
                        output="Access denied: path outside allowed root",
                        error="path_traversal_blocked",
                    )

            if not path.exists():
                return ToolResult(
                    success=False,
                    output=f"File not found: {file_path}",
                    error="file_not_found",
                )
            if not path.is_file():
                return ToolResult(
                    success=False,
                    output=f"Path is not a file: {file_path}",
                    error="not_a_file",
                )

            # Get file stats once
            stat = path.stat()

            # Read file with timeout + memory-efficient read
            loop = asyncio.get_running_loop()
            try:
                content = await asyncio.wait_for(
                    loop.run_in_executor(None, self._read_file, path, encoding, max_chars),
                    timeout=self.timeout,
                )
            except TimeoutError:
                logger.error(f"ReadFileTool timeout on {file_path} after {self.timeout}s")
                return ToolResult(
                    success=False,
                    output=f"File read timed out after {self.timeout}s",
                    error="timeout",
                )

            # Total line count (before any slicing)
            all_lines = content.splitlines(keepends=True)
            total_lines = len(all_lines)
            truncated = False

            # start_line / end_line slicing (1-based)
            if start_line is not None or end_line is not None:
                sl = max(1, int(start_line or 1)) - 1          # to 0-based
                el = min(total_lines, int(end_line or total_lines))
                all_lines = all_lines[sl:el]
                content = "".join(all_lines)
                if sl > 0 or el < total_lines:
                    content = (
                        f"[Lines {sl+1}–{el} of {total_lines}]\n" + content
                    )

            # Apply line limit
            line_count = len(all_lines)
            if max_lines is not None and line_count > max_lines:
                content = "".join(all_lines[:max_lines])
                content += f"\n... [truncated after {max_lines} lines, total {total_lines} lines]"
                truncated = True

            # Final character limit
            if len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]"
                truncated = True

            return ToolResult(
                success=True,
                output=content,
                data={
                    "file_path":    str(path),
                    "size_bytes":   stat.st_size,
                    "chars_read":   len(content),
                    "lines_read":   line_count,
                    "total_lines":  total_lines,
                    "truncated":    truncated,
                    "encoding_used": encoding,
                },
                metadata={
                    "tool": self.name,
                    "file": str(path),
                    "size": stat.st_size,
                },
            )

        except UnicodeDecodeError as e:
            logger.error(f"ReadFileTool decode error on {file_path}: {e}")
            return ToolResult(
                success=False,
                output=f"File encoding error: {e}. Try specifying different 'encoding'.",
                error="decode_error",
            )
        except Exception as exc:
            logger.error(f"ReadFileTool failed on {file_path}: {exc}")
            return ToolResult(
                success=False,
                output=f"Failed to read file: {exc}",
                error="read_error",
            )

    def _read_file(self, path: Path, encoding: str, max_chars: int) -> str:
        """
        Memory-efficient file read.
        Reads only the required amount instead of loading the entire file.
        """
        read_limit = max_chars + 2000  # небольшой запас для корректного line truncation

        with open(path, encoding=encoding, errors="replace") as f:
            return f.read(read_limit)

    def get_command(self, action_input: dict[str, Any]) -> list[str] | None:
        """No subprocess command — pure Python tool."""
        return None

    def __repr__(self) -> str:
        return (
            f"ReadFileTool(max_chars={self.max_chars}, max_lines={self.max_lines}, "
            f"encoding={self.encoding}, allowed_root={self.allowed_root})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_read_file_tool(
    max_chars: int = 100_000,
    max_lines: int | None = None,
    encoding: str = "utf-8",
    allowed_root: str | None = None,
    timeout_seconds: float = 10.0,
) -> ReadFileTool:
    """Factory function for ReadFileTool."""
    return ReadFileTool(
        max_chars=max_chars,
        max_lines=max_lines,
        encoding=encoding,
        allowed_root=allowed_root,
        timeout_seconds=timeout_seconds,
    )
