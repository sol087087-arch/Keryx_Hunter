#!/usr/bin/env python3
"""vuln_cli.py — intentionally vulnerable CLI utility.

Fetches git-log entries filtered by author name and/or date range.

VULNERABILITY: GIT_OPTION_INJECTION
    The `author` and `since` arguments are appended to the git command list
    without a "--" end-of-options separator.  An attacker who controls either
    value can inject arbitrary git options.

    Example exploit:
        python vuln_cli.py --author "--upload-pack=touch /tmp/pwned"
        # git interprets the value as a flag rather than a name string.

    Fix: insert "--" before user-supplied positional values, or validate that
    the string does not start with "-".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def fetch_commits(
    repo_path: str,
    author: str | None = None,
    since: str | None = None,
    max_count: int = 20,
) -> str:
    """Return recent commits, optionally filtered by author and date.

    Security issue (R2): author and since are passed directly as git option
    arguments with no "--" separator and no leading-dash validation.
    """
    cmd = ["git", "log", "--oneline", f"--max-count={max_count}"]

    if author:
        cmd.extend(["--author", author])   # [HIGH] GIT_OPTION_INJECTION

    if since:
        cmd.extend(["--since", since])     # [HIGH] GIT_OPTION_INJECTION

    cmd.append(repo_path)                  # [MEDIUM] UNSANITIZED_SUBPROCESS_ARG

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch git commits by author")
    parser.add_argument("repo",        nargs="?", default=".", help="Path to git repo")
    parser.add_argument("--author",    default=None, help="Filter by author name")
    parser.add_argument("--since",     default=None, help="Show commits after this date")
    parser.add_argument("--max-count", type=int,    default=20)
    args = parser.parse_args()

    print(fetch_commits(args.repo, args.author, args.since, args.max_count))


if __name__ == "__main__":
    main()
