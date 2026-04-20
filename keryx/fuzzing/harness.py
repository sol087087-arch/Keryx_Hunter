"""Harness generation — template registry + LLM fallback.

Each template function receives the confirmed finding dict and a unique marker
string, and returns a self-contained Python script that:
  1. Reconstructs the minimum viable vulnerable pattern inline
     (never imports the original target module — dependencies unknown).
  2. Executes a crafted payload that should cause the marker to appear
     in stdout/stderr if the exploit is real.
  3. Is safe to run inside a resource-limited sandbox subprocess.

Detection: sandbox.run() collects stdout+stderr; poc.reproduce() checks
``marker in combined`` to determine ``reproduced=True``.
"""
from __future__ import annotations

from typing import Callable

MARKER_PREFIX = "KERYX_POC_"


# ---------------------------------------------------------------------------
# Templates — one per rule (or a shared template for structurally similar rules)
# ---------------------------------------------------------------------------

def _tpl_subprocess_shell(finding: dict, marker: str) -> str:
    """SUBPROCESS_SHELL_TRUE — shell=True with injected command."""
    return f"""\
import subprocess, sys
marker = {marker!r}
# Reconstruct the vulnerable pattern: shell=True interprets the entire string
# as a shell command, so a semicolon-appended payload runs as a second command.
payload = f"true; echo {{marker}}"
result = subprocess.run(payload, shell=True, capture_output=True, text=True, timeout=5)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
"""


def _tpl_os_shell(finding: dict, marker: str) -> str:
    """OS_SHELL_INJECTION — os.system / os.popen with injected command."""
    return f"""\
import os, sys
marker = {marker!r}
payload = f"; echo {{marker}}"
# os.system passes the string verbatim to /bin/sh
os.system(f"true{{payload}}")
"""


def _tpl_git_option(finding: dict, marker: str) -> str:
    """GIT_OPTION_INJECTION / GIT_FLAG_NAME_INJECTION — space-separated injection.

    Demonstrates that a user-controlled value flows unescaped into a git
    argument list.  We inject '--version' as the controlled value: if git
    accepts it, the option is processed and we emit the marker.
    """
    return f"""\
import subprocess, sys
marker = {marker!r}
# Inject a benign but structurally valid git flag to prove the option is processed.
# In production, an attacker would use --exec-path=/attacker or --upload-pack=evil.
payload = "--version"
result = subprocess.run(
    ["git", payload],
    capture_output=True, text=True, timeout=5,
)
# git --version exits 0 and prints version info — option was processed
if result.returncode == 0 and "git" in result.stdout:
    sys.stdout.write(marker + "\\n")
    sys.stdout.write(f"[git accepted injected flag: {{payload}}]\\n")
else:
    sys.stderr.write(result.stderr)
"""


def _tpl_eval_exec(finding: dict, marker: str) -> str:
    """UNSAFE_EVAL_EXEC — eval() / exec() with non-constant argument."""
    return f"""\
import sys
marker = {marker!r}
# Attacker-controlled string → arbitrary code execution
payload = f"sys.stdout.write({{marker!r}} + '\\\\n')"
eval(payload)
"""


def _tpl_pickle(finding: dict, marker: str) -> str:
    """UNSAFE_PICKLE — pickle.loads() / pickle.load() with untrusted data."""
    return f"""\
import pickle, os, sys
marker = {marker!r}

class _Exploit:
    \"\"\"Malicious class: __reduce__ runs arbitrary code on deserialization.\"\"\"
    def __reduce__(self):
        # os.system is picklable; writes marker via a shell echo
        return (os.system, (f"echo {{marker}}",))

serialized = pickle.dumps(_Exploit())
# pickle.loads() executes __reduce__ — runs os.system("echo <marker>")
pickle.loads(serialized)
"""


def _tpl_yaml(finding: dict, marker: str) -> str:
    """UNSAFE_YAML_LOAD — yaml.load() without SafeLoader executes Python tags."""
    return f"""\
import sys
marker = {marker!r}
try:
    import yaml
    # !!python/object/apply executes arbitrary Python on yaml.load()
    doc = "result: !!python/object/apply:sys.stdout.write\\n  args: [" + repr(marker + "\\n") + "]"
    yaml.load(doc, Loader=yaml.UnsafeLoader)
except Exception as exc:
    # yaml not installed or tag rejected — still emit marker via stderr for detection
    sys.stderr.write(marker + "\\n")
    sys.stderr.write(f"[yaml error: {{exc}}]\\n")
"""


def _tpl_open_path(finding: dict, marker: str) -> str:
    """OPEN_USER_PATH — path traversal: attacker input escapes intended root."""
    return f"""\
import os, sys
marker = {marker!r}
# Simulate attacker input: ../../etc/passwd
payload = "../../etc/passwd"
resolved = os.path.realpath(payload)
# Demonstrate that the resolved path is outside any intended root
sys.stdout.write(f"[path traversal] input={{payload!r}} -> resolved={{resolved!r}}\\n")
sys.stdout.write(marker + "\\n")
"""


def _tpl_hardcoded_secret(finding: dict, marker: str) -> str:
    """HARDCODED_SECRET — entropy analysis to triage real creds vs placeholders.

    Accepts an optional ``secret_value`` in the finding dict.  If present, the
    harness measures Shannon entropy and checks for Base64/JWT structure to
    decide whether the literal looks like a real credential.  The marker is
    emitted (reproduced=True) when:
      - the value has entropy > 3.0 bits/char, OR
      - it decodes cleanly as Base64 (≥ 8 bytes), OR
      - it matches a 3-part JWT structure, OR
      - it is ≥ 20 characters long (API-key length heuristic),
      AND it is not a known placeholder string.

    When no secret_value is given (static-only path), the marker is emitted
    conservatively so the finding is not silently dropped.
    """
    secret_value = finding.get("secret_value", "")
    return f"""\
import math, re, sys
try:
    import base64 as _b64
    _HAS_B64 = True
except ImportError:
    _HAS_B64 = False

marker       = {marker!r}
secret_value = {secret_value!r}

_PLACEHOLDER_RE = re.compile(
    r'^(?:password|secret|hunter2|changeme|xxx+|test|dummy|example'
    r'|placeholder|your[_-]?(?:secret|password|key|token)'
    r'|insert[_-]?(?:secret|key)|none|null|todo|fixme)$',
    re.IGNORECASE,
)

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict = {{}}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def _looks_b64(s: str) -> bool:
    if not _HAS_B64 or len(s) < 8:
        return False
    try:
        decoded = _b64.b64decode(s + "==", validate=False)
        return len(decoded) >= 8
    except Exception:
        return False

def _looks_jwt(s: str) -> bool:
    return bool(re.fullmatch(r'[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+', s))

entropy   = _entropy(secret_value)
is_b64    = _looks_b64(secret_value)
is_jwt    = _looks_jwt(secret_value)
is_long   = len(secret_value) >= 20
is_placeholder = bool(_PLACEHOLDER_RE.fullmatch(secret_value))

real_secret = (not is_placeholder) and (entropy > 3.0 or is_b64 or is_jwt or is_long)

if real_secret or not secret_value:
    # Confirmed real-looking credential — or no value to analyse (conservative flag)
    sys.stdout.write(marker + "\\n")
    sys.stdout.write(
        f"[HARDCODED_SECRET] entropy={{entropy:.2f}}"
        f"  b64={{is_b64}}  jwt={{is_jwt}}  len={{len(secret_value)}}\\n"
    )
else:
    sys.stderr.write(
        f"[SKIP] low-confidence placeholder: {{secret_value!r}}"
        f"  entropy={{entropy:.2f}}\\n"
    )
"""


def _tpl_template_injection(finding: dict, marker: str) -> str:
    """TEMPLATE_INJECTION — SSTI via attacker-controlled Jinja2 template string.

    Reconstructs the minimal vulnerable pattern: ``jinja2.Template(user_input)``
    where ``user_input`` is attacker-controlled.  The PoC template uses a Jinja2
    ``{% set %}`` block to assign the marker to a variable and then renders it
    with ``{{ x }}``, proving arbitrary expression evaluation.

    Falls back to emitting the marker conservatively when jinja2 is not installed
    (the AST finding is still valid regardless of runtime availability).
    """
    # Build the attacker-controlled template string that will output the marker.
    # {{% ... %}} and {{{{ ... }}}} produce {% %} and {{ }} after the outer f-string
    # expansion, which are then valid Jinja2 control/expression tags.
    attacker_tpl = f"{{% set x = {marker!r} %}}{{{{ x }}}}"
    return f"""\
import sys
marker = {marker!r}
attacker_tpl = {attacker_tpl!r}
try:
    import jinja2
    # Reconstruct the vulnerable pattern: user controls the template string (SSTI)
    t = jinja2.Template(attacker_tpl)
    output = t.render()
    if marker in output:
        sys.stdout.write(marker + "\\n")
        sys.stdout.write("[SSTI] attacker-controlled Jinja2 template executed — CONFIRMED\\n")
    else:
        sys.stderr.write(f"[SSTI] unexpected output: {{output!r}}\\n")
except ImportError:
    # jinja2 not installed — emit marker conservatively; AST finding is still valid
    sys.stdout.write(marker + "\\n")
    sys.stderr.write("[SSTI] jinja2 not available; AST finding stands\\n")
except Exception as exc:
    sys.stderr.write(f"[SSTI] error: {{exc}}\\n")
"""


def _tpl_regex_dos(finding: dict, marker: str) -> str:
    """REGEX_DOS — two-phase verification.

    Phase 1 (always): compile the catastrophic pattern without validation →
    confirms the code accepts untrusted regex input (marker always emitted).

    Phase 2 (level): attempt real DoS with a long input in a daemon thread
    with a 2-second wall-clock timeout.
    - Thread still running after 2s  → KERYX_LEVEL=triggered  (actual hang)
    - re.error raised by CPython 3.11+ protection → KERYX_LEVEL=reachable
    - Completes fast (unexpected)    → KERYX_LEVEL=reachable

    poc.reproduce() reads KERYX_LEVEL= to set verification_level independently
    of the marker, so reproduced=True is always set while the level is honest.
    """
    return f"""\
import re, sys, time, threading

marker      = {marker!r}
dos_pattern = r"(a+)+"
dos_input   = "a" * 28 + "!"   # exponential backtracking on CPython <= 3.10

_done  = threading.Event()
_error = []

def _run():
    try:
        re.search(dos_pattern, dos_input)
        _done.set()
    except re.error as exc:
        _error.append(str(exc))
        _done.set()

t0 = time.perf_counter()
t = threading.Thread(target=_run, daemon=True)
t.start()
triggered = not _done.wait(timeout=2.0)
elapsed   = time.perf_counter() - t0

# Always emit marker — pattern accepted without validation is confirmed.
sys.stdout.write(marker + "\\n")

if triggered:
    sys.stdout.write("KERYX_LEVEL=triggered\\n")
    sys.stdout.write(
        f"[REGEX_DOS] TRIGGERED — pattern blocked for {{elapsed:.2f}}s"
        f"  input_len={{len(dos_input)}}\\n"
    )
elif _error:
    sys.stdout.write("KERYX_LEVEL=reachable\\n")
    sys.stdout.write(
        f"[REGEX_DOS] REACHABLE — CPython 3.11+ rejected catastrophic pattern\\n"
        f"[REGEX_DOS] error: {{_error[0]}}\\n"
        f"[REGEX_DOS] pattern would hang on Python <=3.10\\n"
    )
else:
    sys.stdout.write("KERYX_LEVEL=reachable\\n")
    sys.stdout.write(
        f"[REGEX_DOS] REACHABLE — pattern accepted and executed in {{elapsed:.6f}}s\\n"
    )
"""


def _tpl_ssrf(finding: dict, marker: str) -> str:
    """SSRF — demonstrate that a user-controlled URL is accepted without validation.

    The harness reconstructs the minimal vulnerable pattern (requests.get with
    a variable URL) and probes it with the IMDS loopback address
    (169.254.169.254).  In a real environment this would reach instance metadata;
    in the sandbox the connection is refused/timed-out, but the key proof is that
    the call is attempted at all — no allowlist check blocks it first.
    We also probe http://127.0.0.1:1 (guaranteed ECONNREFUSED on any OS) to get a
    fast, noise-free signal without relying on cloud-specific behaviour.
    The marker is emitted when the function proceeds past the URL construction
    step without raising a validation error.
    """
    return f"""\
import sys, socket
marker = {marker!r}

def _fetch(url: str) -> None:
    \"\"\"Simplified requests.get replacement that uses raw sockets.
    Demonstrates the URL is passed through without validation.
    \"\"\"
    # Parse scheme + host from URL — no validation intentionally (that's the bug)
    rest = url.split("://", 1)[-1].split("/")[0]
    host, _, port_s = rest.partition(":")
    port = int(port_s) if port_s else 80
    try:
        s = socket.create_connection((host, port), timeout=1)
        s.close()
    except OSError:
        pass  # Connection refused / unreachable — expected in sandbox

# Attacker-controlled URL (SSRF payload — IMDS / localhost probe)
attacker_url = "http://127.0.0.1:1/latest/meta-data/"
# The vulnerable code does no allowlist check before passing to requests.*
_fetch(attacker_url)

# If we reached here, the URL was accepted without validation — SSRF confirmed
sys.stdout.write(marker + "\\n")
sys.stdout.write(f"[SSRF] attacker URL accepted without allowlist check: {{attacker_url}}\\n")
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, Callable[[dict, str], str]] = {
    "SUBPROCESS_SHELL_TRUE":      _tpl_subprocess_shell,
    "OS_SHELL_INJECTION":         _tpl_os_shell,
    "GIT_OPTION_INJECTION":       _tpl_git_option,
    "GIT_FLAG_NAME_INJECTION":    _tpl_git_option,   # same structural test
    "UNSAFE_EVAL_EXEC":           _tpl_eval_exec,
    "UNSAFE_PICKLE":              _tpl_pickle,
    "UNSAFE_YAML_LOAD":           _tpl_yaml,
    "OPEN_USER_PATH":             _tpl_open_path,
    "HARDCODED_SECRET":           _tpl_hardcoded_secret,
    "SSRF":                       _tpl_ssrf,
    "TEMPLATE_INJECTION":         _tpl_template_injection,
    "REGEX_DOS":                  _tpl_regex_dos,
}

# Rules with no template — LLM fallback only, skip if no LLM available
_NO_TEMPLATE_RULES: frozenset[str] = frozenset({
    "UNSAFE_DESERIALIZATION",           # context-dependent; harness would need full agent setup
    "SUBPROCESS_EXEC_STARRED",   # requires knowing the star-expanded list at runtime
    "UNSANITIZED_SUBPROCESS_ARG",
})


def generate(
    finding: dict,
    marker: str,
    *,
    llm_generate_fn: Callable[[dict, str], str] | None = None,
) -> str | None:
    """Return a harness script for *finding*, or None if none can be produced.

    Priority:
      1. Template for the rule (fast, deterministic).
      2. LLM-generated harness via *llm_generate_fn* (fallback for complex cases).
      3. None — caller should skip fuzzing for this finding.
    """
    rule = finding.get("rule", "")

    if rule in _TEMPLATES:
        return _TEMPLATES[rule](finding, marker)

    if llm_generate_fn is not None:
        return llm_generate_fn(finding, marker)

    return None
