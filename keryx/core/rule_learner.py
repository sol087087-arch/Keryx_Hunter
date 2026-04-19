# keryx/core/rule_learner.py
"""Post-hunt rule gap analysis.

``RuleLearner`` scans the Python source files that were already analysed and
looks for sink patterns that are *structurally similar* to the current rule set
but not yet covered.  It produces a ranked list of ``RuleSuggestion`` objects
that can be printed after Phase 2 (``--learn`` flag) or consumed
programmatically.

Design principles
-----------------
* No re-running AST analysis — works purely from already-read source text via
  lightweight regex passes, so it adds < 0.1 s to a full hunt.
* Every suggestion includes example file paths so a developer can quickly
  verify whether it is a genuine gap or a false positive.
* Suggestions are ranked by occurrence count (most common pattern first).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RuleSuggestion:
    suggestion_id: str      # e.g. "SSRF_HTTPX"
    rule_family:   str      # nearest existing rule, e.g. "SSRF"
    occurrences:   int      # how many files contain the pattern
    files:         list[str] = field(default_factory=list)   # up to 5 examples
    example:       str = ""  # one concrete code snippet
    description:   str = ""
    recommended:   str = ""  # proposed fix / rule addition

    def __str__(self) -> str:
        files_str = ", ".join(self.files[:3])
        if len(self.files) > 3:
            files_str += f" (+{len(self.files) - 3} more)"
        return (
            f"[{self.suggestion_id}]  family={self.rule_family}  "
            f"occurrences={self.occurrences}\n"
            f"  desc     : {self.description}\n"
            f"  fix/rule : {self.recommended}\n"
            f"  files    : {files_str}\n"
            f"  example  : {self.example}"
        )


# ---------------------------------------------------------------------------
# Gap-pattern registry
#
# Each entry: (suggestion_id, rule_family, regex_pattern, description, recommended)
# The regex is searched across source file text (re.MULTILINE).
# Groups are used to extract an example snippet (group 0 = full match if no groups).
# ---------------------------------------------------------------------------

_GAP_PATTERNS: list[tuple[str, str, str, str, str]] = [
    # --- SSRF family ---------------------------------------------------------
    (
        "SSRF_HTTPX",
        "SSRF",
        r"httpx\.(?:get|post|put|delete|patch|request)\([^\"'][^)]*\)",
        "httpx HTTP methods called with a non-constant URL argument.",
        "Extend the SSRF rule (R11) to cover httpx.* in addition to requests.*",
    ),
    (
        "SSRF_URLLIB",
        "SSRF",
        r"urllib\.request\.urlopen\([^\"'][^)]*\)",
        "urllib.request.urlopen called with a non-constant URL argument.",
        "Add an SSRF sub-rule for urllib.request.urlopen(url_var).",
    ),
    (
        "SSRF_AIOHTTP",
        "SSRF",
        r"(?:session|client)\.(?:get|post|put|delete|patch)\([^\"'][^)]*\)",
        "aiohttp / httpx async session method called with a non-constant URL.",
        "Add an async SSRF rule covering ClientSession.get(url_var) patterns.",
    ),
    # --- Command injection / subprocess -------------------------------------
    (
        "SHELL_FORMAT_STRING",
        "SUBPROCESS_SHELL_TRUE",
        r'(?:os\.system|subprocess\.run)\(\s*f["\']',
        "os.system or subprocess.run called with an f-string command.",
        "Flag f-string arguments to os.system / subprocess.run as HIGH "
        "(the entire shell command is user-controlled via the interpolated variables).",
    ),
    # --- Deserialization ----------------------------------------------------
    (
        "UNSAFE_MARSHAL",
        "UNSAFE_PICKLE",
        r"marshal\.loads\([^\"'][^)]*\)",
        "marshal.loads called with a non-constant argument.",
        "Add UNSAFE_MARSHAL rule: marshal can execute arbitrary bytecode on load.",
    ),
    (
        "UNSAFE_SHELVE",
        "UNSAFE_PICKLE",
        r"shelve\.open\([^\"'][^)]*\)",
        "shelve.open called with a non-constant path — shelve uses pickle internally.",
        "Add UNSAFE_SHELVE rule or extend UNSAFE_PICKLE to cover shelve.open.",
    ),
    # --- XML / XXE ----------------------------------------------------------
    (
        "XXE_ELEMENTTREE",
        "LLM_OUTPUT_SINK",   # closest structural family (untrusted deserialization)
        r"(?:ET|ElementTree|etree)\.(?:parse|fromstring|XML)\([^\"'][^)]*\)",
        "xml.etree.ElementTree.parse/fromstring with a non-constant argument — "
        "possible XML External Entity (XXE) injection.",
        "Add an XXE rule: parse XML only from trusted sources; use defusedxml.",
    ),
    # --- Dynamic import -----------------------------------------------------
    (
        "UNSAFE_IMPORT",
        "UNSAFE_EVAL_EXEC",
        r"importlib\.import_module\([^\"'][^)]*\)",
        "importlib.import_module called with a non-constant module name.",
        "Add UNSAFE_IMPORT rule: dynamic module loading with user input enables "
        "arbitrary code execution via crafted module names.",
    ),
    # --- Template engines ---------------------------------------------------
    (
        "TEMPLATE_MAKO",
        "TEMPLATE_INJECTION",
        r"mako\.template\.Template\([^\"'][^)]*\)",
        "Mako Template instantiated with a non-constant string (SSTI).",
        "Extend the TEMPLATE_INJECTION rule (R12) to explicitly match mako.template.Template.",
    ),
    (
        "TEMPLATE_CHAMELEON",
        "TEMPLATE_INJECTION",
        r"chameleon\.PageTemplate\([^\"'][^)]*\)",
        "Chameleon PageTemplate instantiated with a non-constant string (SSTI).",
        "Add a TEMPLATE_INJECTION sub-rule for chameleon.PageTemplate.",
    ),
    # --- Regex --------------------------------------------------------------
    (
        "REGEX_DOS_COMPILED",
        "REGEX_DOS",
        r"re\.compile\([^r\"\'][^)]*\)\.(?:match|search|fullmatch)",
        "Compiled regex pattern from a non-constant source used immediately — "
        "chained call obscures the ReDoS risk.",
        "Extend REGEX_DOS (R13) to detect chained re.compile(...).match(...) patterns.",
    ),
]

# Pre-compiled for speed
_COMPILED: list[tuple[str, str, re.Pattern[str], str, str]] = [
    (sid, fam, re.compile(pat, re.MULTILINE), desc, rec)
    for sid, fam, pat, desc, rec in _GAP_PATTERNS
]


# ---------------------------------------------------------------------------
# RuleLearner
# ---------------------------------------------------------------------------

class RuleLearner:
    """Scan source files for uncovered sink patterns and rank the gaps."""

    def __init__(self, files: list[Path]) -> None:
        """
        Parameters
        ----------
        files:
            Python source files to scan (already opened/parsed in Phase 0).
            The learner reads their text content via Path.read_text().
        """
        self._files = files

    def analyse(self) -> list[RuleSuggestion]:
        """Return suggestions sorted by occurrence count (highest first)."""
        hits: dict[str, dict] = {}   # suggestion_id → aggregated data

        for path in self._files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for sid, fam, pat, desc, rec in _COMPILED:
                match = pat.search(text)
                if match is None:
                    continue
                if sid not in hits:
                    hits[sid] = {
                        "family":  fam,
                        "desc":    desc,
                        "rec":     rec,
                        "files":   [],
                        "example": match.group(0)[:120].strip(),
                    }
                hits[sid]["files"].append(str(path))

        suggestions = [
            RuleSuggestion(
                suggestion_id = sid,
                rule_family   = data["family"],
                occurrences   = len(data["files"]),
                files         = data["files"][:5],
                example       = data["example"],
                description   = data["desc"],
                recommended   = data["rec"],
            )
            for sid, data in hits.items()
        ]
        suggestions.sort(key=lambda s: s.occurrences, reverse=True)
        return suggestions


def print_suggestions(suggestions: list[RuleSuggestion]) -> None:
    """Pretty-print a list of RuleSuggestion objects."""
    if not suggestions:
        print("\n[Learn] No rule gaps detected in scanned files.")
        return

    print(f"\n{'='*70}")
    print(f"RULE GAP ANALYSIS — {len(suggestions)} potential gap(s) detected")
    print("=" * 70)
    for s in suggestions:
        print(f"\n{s}")
    print(
        "\nThese are suggestions for new or extended AST rules.\n"
        "Verify each against the flagged files before adding a rule."
    )
