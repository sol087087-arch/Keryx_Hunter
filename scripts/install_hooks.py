#!/usr/bin/env python3
"""Install the Keryx pre-commit hook into .git/hooks/.

Usage:
    python scripts/install_hooks.py [--force]

The pre-commit hook:
  1. Collects staged .py files.
  2. Runs the AST scanner on them (Phase 0a only — fast, no LLM).
  3. Blocks the commit if any HIGH finding is detected.
  4. Prints a short report so the developer knows what to fix.

Exit codes from the hook script itself:
  0  — no HIGH findings (commit proceeds)
  1  — HIGH findings found (commit blocked)

To bypass in an emergency:
    git commit --no-verify
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

_HOOK_BODY = r"""#!/usr/bin/env python3
# keryx pre-commit hook — installed by scripts/install_hooks.py
# Blocks commits that introduce HIGH-severity AST findings.
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from keryx.tools.ast_analyzer import ASTAnalyzerTool

# Collect staged Python files
result = subprocess.run(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
    capture_output=True, text=True, cwd=str(_REPO_ROOT),
)
staged = [
    _REPO_ROOT / p for p in result.stdout.splitlines()
    if p.endswith(".py") and (_REPO_ROOT / p).exists()
]

if not staged:
    sys.exit(0)

tool = ASTAnalyzerTool()
high_findings: list[dict] = []

for path in staged:
    try:
        raw = tool.execute({"file_path": str(path)})
        findings = raw.get("findings", []) if isinstance(raw, dict) else []
        for f in findings:
            if isinstance(f, dict) and f.get("severity") == "HIGH":
                f["file"] = str(path.relative_to(_REPO_ROOT))
                high_findings.append(f)
            elif hasattr(f, "severity") and f.severity == "HIGH":
                high_findings.append({
                    "file":  str(path.relative_to(_REPO_ROOT)),
                    "rule":  f.rule,
                    "line":  f.line,
                    "message": f.message,
                })
    except Exception as exc:
        print(f"[keryx] Warning: could not scan {path.name}: {exc}", file=sys.stderr)

if not high_findings:
    print(f"[keryx] pre-commit: {len(staged)} file(s) scanned — no HIGH findings.")
    sys.exit(0)

print(f"\n[keryx] pre-commit BLOCKED — {len(high_findings)} HIGH finding(s):\n",
      file=sys.stderr)
for f in high_findings:
    rule = f.get("rule", "?")
    line = f.get("line", "?")
    msg  = f.get("message", "")
    file = f.get("file", "?")
    print(f"  [{rule}] {file}:{line}  {msg}", file=sys.stderr)

print(
    "\nFix the issues above, or use  git commit --no-verify  to bypass.\n",
    file=sys.stderr,
)
sys.exit(1)
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Install Keryx pre-commit hook")
    ap.add_argument("--force", action="store_true", help="Overwrite existing hook")
    ap.add_argument("--repo-dir", default=None, metavar="PATH",
                    help="Repository root (default: parent of this script's parent)")
    args = ap.parse_args()

    repo_root = Path(args.repo_dir).resolve() if args.repo_dir else Path(__file__).resolve().parent.parent
    hooks_dir = repo_root / ".git" / "hooks"

    if not hooks_dir.exists():
        print(f"Error: {hooks_dir} not found — are you inside a git repository?",
              file=sys.stderr)
        return 1

    hook_path = hooks_dir / "pre-commit"

    if hook_path.exists() and not args.force:
        print(f"Hook already exists at {hook_path}.")
        print("Run with --force to overwrite.")
        return 0

    hook_path.write_text(_HOOK_BODY, encoding="utf-8")
    # Make the hook executable
    current = hook_path.stat().st_mode
    hook_path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Installed Keryx pre-commit hook → {hook_path}")
    print("The hook will block commits with HIGH AST findings.")
    print("To bypass in an emergency: git commit --no-verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
