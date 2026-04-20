# keryx/core/verification_pipeline.py
"""Strict-mode auto-verification pipeline.

Resolves the target tool from the toolbox, extracts HIGH findings from codeql
output, and probes each one via injection_verifier — zero extra LLM tokens.

VerificationPipeline is a collaboration object: it accepts a toolbox and a
context as constructor arguments and writes results back to the context
(add_step, add_evidence).  extract_findings() is pure and static.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ._constants import PAYLOAD_MAP, VARNAME_TO_FIELD
from .shared_context import SharedContext

if TYPE_CHECKING:
    from ..tools.Toolbox import ToolBox


class VerificationPipeline:

    def __init__(self, toolbox: "ToolBox", context: SharedContext) -> None:
        self.toolbox = toolbox
        self.context = context

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------

    def resolve_target_tool(self) -> str | None:
        """Infer which registered tool corresponds to the file being analyzed.

        Strategy (A then B):
        1. Exact stem match: git_blame.py → "git_blame"
        2. Normalised stem (hyphens/spaces → underscores)
        3. Partial match: any tool name contained in the stem, or vice versa
        4. Return None and warn if nothing found.
        """
        available = set(self.toolbox.list_tools())
        stem = Path(self.context.target_path).stem

        if stem in available:
            return stem
        normalized = stem.replace("-", "_").replace(" ", "_")
        if normalized in available:
            return normalized
        for tool in available:
            if tool in stem or stem in tool:
                return tool

        print(
            f"[INFO] {stem!r} has no registered tool — "
            "falling back to fuzzer PoC verification"
        )
        return None

    # ------------------------------------------------------------------
    # Finding extraction (pure)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_findings(
        codeql_observation: str,
    ) -> list[tuple[str, str, str]]:
        """Parse HIGH findings from codeql output.

        Returns a list of (rule, inject_field, payload) tuples ready for
        injection_verifier.  Skips rules with no payload (e.g. HARDCODED_SECRET).
        One attempt per rule — duplicates are ignored.
        """
        high_lines = re.findall(
            r'\[HIGH\] (\w+) @.*?:\s*(.+?)(?:\n|$)', codeql_observation
        )

        results: list[tuple[str, str, str]] = []
        seen_rules: set[str] = set()

        for rule, context_text in high_lines:
            if rule in seen_rules:
                continue
            payload = PAYLOAD_MAP.get(rule)
            if payload is None:
                continue

            inject_field: str | None = None

            if rule == "GIT_OPTION_INJECTION":
                m = re.search(r'cmd\.extend\(\["--[\w-]+",\s*(\w+)\]', context_text)
                if m:
                    inject_field = VARNAME_TO_FIELD.get(m.group(1), m.group(1))

            elif rule in ("UNSANITIZED_SUBPROCESS_ARG", "SUBPROCESS_EXEC_STARRED"):
                m = re.search(r'cmd\.(?:append|exec)\((\w+)\)', context_text)
                if m:
                    inject_field = VARNAME_TO_FIELD.get(m.group(1), m.group(1))

            elif rule == "OPEN_USER_PATH":
                m = re.search(r'open\((\w+)\)', context_text)
                if m:
                    inject_field = VARNAME_TO_FIELD.get(m.group(1), m.group(1))

            elif rule == "SUBPROCESS_SHELL_TRUE":
                inject_field = "file"

            elif rule == "TEMPLATE_INJECTION":
                # Extract the variable name passed to Template(...) or from_string(...)
                m = re.search(r'(?:Template|from_string)\((\w+)\)', context_text)
                if m:
                    inject_field = VARNAME_TO_FIELD.get(m.group(1), "template")
                else:
                    inject_field = "template"

            elif rule == "REGEX_DOS":
                # Extract the variable name passed as the pattern to re.<func>(...)
                m = re.search(r're\.\w+\((\w+)', context_text)
                if m:
                    inject_field = VARNAME_TO_FIELD.get(m.group(1), "pattern")
                else:
                    inject_field = "pattern"

            if inject_field:
                results.append((rule, inject_field, payload))
                seen_rules.add(rule)

        return results

    # ------------------------------------------------------------------
    # Async verification
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fuzzer fallback (external targets)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_high_rules(codeql_observation: str) -> list[str]:
        """Return deduplicated HIGH rule names from a codeql observation."""
        seen: dict[str, None] = {}
        for m in re.finditer(r"\[HIGH\]\s+(\w+)", codeql_observation):
            seen.setdefault(m.group(1), None)
        return list(seen)

    async def _fuzz_verify(self, codeql_observation: str) -> tuple[bool, str]:
        """Fallback for external targets: run fuzzer PoC for each HIGH rule.

        Returns (confirmed, verification_level) where level is one of:
          "exploited" — exploit marker found in output
          "triggered" — anomalous behaviour (timeout / crash) without marker
          "reachable" — code path reached, no anomaly
        """
        from ..fuzzing.poc import reproduce   # local import — avoids circular dep

        rules = self._extract_high_rules(codeql_observation)
        if not rules:
            print("[FuzzVerify] No HIGH rules to probe — AST-only.")
            return False, "reachable"

        best_level = "reachable"
        for rule in rules:
            poc = reproduce({"rule": rule}, timeout=10)
            if poc.reproduced:
                print(
                    f"[FuzzVerify] CONFIRMED — rule={rule}  level={poc.verification_level}"
                    f"  elapsed={poc.elapsed_s:.2f}s  marker={poc.marker}"
                )
                return True, poc.verification_level
            if poc.error:
                print(f"[FuzzVerify] skip {rule}: {poc.error}")
            else:
                print(f"[FuzzVerify] not reproduced — rule={rule}  level={poc.verification_level}"
                      f"  ({poc.elapsed_s:.2f}s)")
                if poc.verification_level == "triggered" and best_level == "reachable":
                    best_level = "triggered"

        return False, best_level

    # ------------------------------------------------------------------
    # Async verification
    # ------------------------------------------------------------------

    async def auto_verify(self, codeql_observation: str) -> tuple[bool, str]:
        """Run dynamic verification on all HIGH findings.

        For internal targets (file stem matches a registered tool) uses the
        injection_verifier.  For external targets falls back to fuzzer PoC.

        Returns ``(confirmed, method_tag)`` where method_tag is one of:
          - ``"injection_verifier"``  — tool-level injection test confirmed
          - ``"fuzz_poc"``            — sandboxed PoC harness confirmed
        """
        from .agent import AgentStep   # local import avoids circular dependency

        target_tool = self.resolve_target_tool()
        if target_tool is None:
            confirmed, level = await self._fuzz_verify(codeql_observation)
            return confirmed, f"fuzz_poc:{level}"

        findings = self.extract_findings(codeql_observation)
        if not findings:
            print("[AutoVerify] No verifiable HIGH findings — static-only confirmation.")
            return False, "injection_verifier"

        for rule, inject_field, payload in findings:
            print(
                f"[AutoVerify] {target_tool!r} | rule={rule} | "
                f"field={inject_field!r} | payload={payload!r}"
            )
            try:
                result = await asyncio.wait_for(
                    self.toolbox.execute_async(
                        "injection_verifier",
                        {
                            "target_tool":       target_tool,
                            "inject_field":      inject_field,
                            "payload":           payload,
                            "expected_behavior": "reject",
                            "base_overrides":    {"file": self.context.target_path},
                        },
                    ),
                    timeout=20.0,
                )
            except Exception as exc:
                print(f"[AutoVerify] Verifier call failed ({rule}): {exc}")
                continue

            observation = result.output if hasattr(result, "output") else str(result)
            self.context.add_step(
                AgentStep(
                    thought=f"auto_verify_{rule}",
                    action="injection_verifier",
                    action_input={
                        "target_tool": target_tool,
                        "inject_field": inject_field,
                        "payload": payload,
                    },
                    confidence=0.9 if "VULN_CONFIRMED" in observation else 0.35,
                ),
                observation,
            )
            self.context.add_evidence(
                location=f"injection_verifier:{target_tool}.{inject_field}",
                description=f"strict-mode auto-verification of {rule}",
            )

            if "VULN_CONFIRMED" in observation:
                print(f"[AutoVerify] CONFIRMED — {target_tool}.{inject_field}")
                return True, "injection_verifier"
            print(f"[AutoVerify] REJECTED — {target_tool}.{inject_field}")

        return False, "injection_verifier"
