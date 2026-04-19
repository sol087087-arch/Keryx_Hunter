"""Deliberately vulnerable demo application — hunt target for --fuzz smoke test.

Each function contains a real, exploitable vulnerability that the fuzzer
should be able to reproduce without LLM calls (mode=none, scripted).
"""
import subprocess
import os
import pickle
import yaml
import re
import requests
from jinja2 import Template


def run_report(report_name: str) -> str:
    """R1 — SUBPROCESS_SHELL_TRUE: user-controlled report_name flows into shell."""
    result = subprocess.run(
        f"cat reports/{report_name}.txt",
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def export_data(output_path: str) -> None:
    """R3 — OPEN_USER_PATH: attacker controls output_path."""
    with open(output_path, "w") as f:
        f.write("export data\n")


def run_cleanup(script_arg: str) -> None:
    """R10 — OS_SHELL_INJECTION: os.system with user input."""
    os.system(f"rm -f /tmp/{script_arg}")


def load_session(raw_bytes: bytes) -> object:
    """R8 — UNSAFE_PICKLE: deserializes attacker-controlled bytes."""
    return pickle.loads(raw_bytes)


def load_config(stream) -> dict:
    """R9 — UNSAFE_YAML_LOAD: yaml.load without SafeLoader."""
    return yaml.load(stream, Loader=yaml.FullLoader)


def eval_formula(formula: str) -> float:
    """R7 — UNSAFE_EVAL_EXEC: eval with user-controlled formula."""
    return eval(formula)


def fetch_resource(url: str) -> str:
    """R11 — SSRF: user-controlled URL passed directly to requests.get."""
    response = requests.get(url, timeout=5)
    return response.text


def render_page(user_template: str, **ctx) -> str:
    """R12 — TEMPLATE_INJECTION: user controls the Jinja2 template string (SSTI)."""
    t = Template(user_template)
    return t.render(**ctx)


def search_logs(user_pattern: str, log_line: str) -> bool:
    """R13 — REGEX_DOS: user-controlled pattern passed directly to re.search."""
    return bool(re.search(user_pattern, log_line))
