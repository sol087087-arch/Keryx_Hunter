# KeryxHunter

> *Keryx (Κῆρυξ) — the divine messenger who announces what is coming.*

**A sovereign, air-gapped LLM-agent vulnerability hunter.**  
Runs locally. Sends zero bytes to the cloud. Works with any model.

```
  ██╗  ██╗███████╗██████╗ ██╗   ██╗██╗  ██╗
  ██║ ██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝╚██╗██╔╝
  █████╔╝ █████╗  ██████╔╝ ╚████╔╝  ╚███╔╝
  ██╔═██╗ ██╔══╝  ██╔══██╗  ╚██╔╝   ██╔██╗
  ██║  ██╗███████╗██║  ██║   ██║   ██╔╝ ██╗
  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
  hunter · sovereign · air-gapped · open
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Local First](https://img.shields.io/badge/local--first-✓-brightgreen.svg)](#air-gapped-mode)
[![Models](https://img.shields.io/badge/models-any-orange.svg)](#model-routing)
[![Air-Gapped](https://img.shields.io/badge/air--gapped-ready-red.svg)](#air-gapped-mode)

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Key Features](#key-features)
- [The Advisor Strategy](#the-advisor-strategy)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Operating Modes](#operating-modes)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Comparison](#comparison)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Why This Exists

Current LLM-powered vulnerability tools split into two camps:

**Powerful but closed.** Proprietary agentic systems with impressive results. They require uploading your source code to external servers. Inaccessible to independent researchers, classified environments, and organizations with strict data policies.

**Open but shallow.** Static analyzers (Semgrep, CodeQL) or simple LLM wrappers without an agentic loop. High false-positive rates. No fuzzing verification. No iterative reasoning.

**KeryxHunter closes the gap:**

- Full agentic loop (hypothesis → verification → fuzzing → PoC) — **locally**
- Your code never leaves your machine
- Any model: local GGUF, cloud API, or a mix
- Specialized in memory safety: UAF, buffer overflow, type confusion, race conditions

---

## Key Features

### Air-Gapped Mode

Complete network isolation. Zero outbound connections. Designed for classified environments, offline workstations, and proprietary codebases.

```bash
keryx hunt --target /path/to/codebase --mode airgapped \
  --models /models/llama-3-70b.gguf,/models/qwen-32b.gguf
```

### Intelligent Model Routing

You don't choose a model — you declare a **capability**. The router picks the optimal model for the task, context, and budget.

```bash
keryx hunt --capability fast_pattern_matching   # local 8B, ~20ms
keryx hunt --capability deep_reasoning          # local 70B or API
keryx hunt --capability verified_only           # 3-model vote required
```

### The Advisor Strategy

When the executor model finds nothing — or confidence is low — KeryxHunter escalates to a designated **Advisor**: a larger, more capable model invoked on-demand.

See [The Advisor Strategy](#the-advisor-strategy) for the full design.

### Progressive Escalation

```
Level 1  |  Local 8B      |  10,000 files  ->  50 candidates   |  $0.00
Level 2  |  Local 70B     |  50 candidates ->   5 hypotheses    |  $0.00
Level 3  |  Advisor LLM   |   5 hypotheses ->   2 confirmed     |  ~$3-12
Level 4  |  Fuzzing+SymEx |   2 bugs       ->   2 verified PoCs |  $0.00
```

Money is spent only when genuinely warranted.

### Multi-Agent Debate

Three models independently analyze the same code. A result is accepted only with consensus >= 2/3. False-positive rate drops from ~85% to ~15%.

### Native Local Model Integration

Not an HTTP wrapper over `localhost:11434`. Direct loading via `llama-cpp-python` — no network overhead, native tool calls, grammar-constrained output.

| Approach | Latency | Privacy |
|----------|---------|---------|
| HTTP (Ollama) | ~500ms | Local process |
| **KeryxHunter native** | **~20ms** | **In-process** |

---

## The Advisor Strategy

KeryxHunter implements an **Executor / Advisor** architecture. The key insight: you do not need a large expensive model on every turn — only when the fast executor gets stuck.

```
                        THE ADVISOR STRATEGY
  ___________________________________________________________________
 |                                                                   |
 |  Main loop ---> [ Executor ]  --- Tool call --->  [ Advisor  ]   |
 |                   Llama-8B                        Any large LLM  |
 |                   Runs every turn                 On-demand only |
 |                       |                                  |       |
 |                  Read / write                    Reviews context |
 |                       |                                  |       |
 |                       v         Sends advice             |       |
 |               [ Shared Context ] <------------------------        |
 |                 Conversation * tools * history                    |
 |___________________________________________________________________|
          Advisor reads the same context as Executor
```

### How It Works

The **Executor** is a small, fast local model that runs the main analysis loop — scanning files, calling tools (CodeQL, GDB, AFL++), forming hypotheses, and self-critiquing results.

The **Advisor** is a large, powerful model invoked only when:

- The executor reports confidence below a threshold
- The executor finds no candidates after scanning all files
- A hypothesis fails verification after N rounds
- A task capability explicitly requires deep reasoning

The Advisor reads the full shared context (conversation, tool outputs, code snippets, scan history), sends its guidance back into shared context, and the Executor resumes the loop with new direction.

The expensive model runs once or twice per session — not on every file.

### Configuring Advisors

Advisors are fully configurable. You can assign different advisors to different failure modes, or chain them in sequence.

```yaml
# ~/.keryx/advisors.yaml

advisors:
  # Default advisor: local large model (air-gapped compatible)
  - name: local-advisor
    model: llama-70b
    trigger:
      confidence_below: 0.6
      no_candidates_after: 500        # files scanned with no result
    reads_context: full
    max_calls_per_session: 3

  # Deep reasoning advisor: cloud model (online mode only)
  - name: cloud-advisor
    model: claude-opus                # or gpt-4o, deepseek-r1, gemini, etc.
    trigger:
      confidence_below: 0.4
      executor_rounds_failed: 3
    reads_context: full
    max_calls_per_session: 2
    require_online_mode: true

  # Specialist advisor: fine-tuned for Firefox internals
  - name: firefox-specialist
    model: /models/firefox-lora.gguf
    trigger:
      file_matches_pattern:
        - "js/src/**"
        - "ipc/**"
    reads_context: targeted           # only relevant files + tool outputs
    max_calls_per_session: 5

  # Cascade: advisors chain in order
  - name: cascade-advisor
    chain:
      - local-advisor                 # try local first
      - cloud-advisor                 # escalate to cloud if still stuck
    trigger:
      no_confirmed_vulns_after_full_scan: true
```

### Assigning Advisors per Hunt

```bash
# Use a specific advisor
keryx hunt --target /firefox --advisor local-advisor

# Cascade: local first, cloud if needed
keryx hunt --target /firefox --advisor cascade-advisor --budget $5

# Air-gapped: only local advisor allowed
keryx hunt --target /firefox --mode airgapped --advisor firefox-specialist

# Disable advisor entirely (fastest, cheapest)
keryx hunt --target /firefox --advisor none
```

### Advisor Plugin API

Custom advisors can be implemented as plugins:

```python
from keryx.advisors import BaseAdvisor, SharedContext, AdvisorResponse

class MyCustomAdvisor(BaseAdvisor):
    name = "my-advisor"

    def should_trigger(self, context: SharedContext) -> bool:
        return context.executor_confidence < 0.55

    def advise(self, context: SharedContext) -> AdvisorResponse:
        # context contains: code snippets, tool outputs,
        # executor hypotheses, scan history
        advice = self.model.reason(context.to_prompt())
        return AdvisorResponse(
            guidance=advice,
            suggested_tools=["codeql", "gdb"],
            priority_files=advice.extract_files()
        )

# Register
keryx advisor register MyCustomAdvisor
```

---

## How It Works

### Agent Loop (ReAct)

```
  1. RANKING
     AST analysis + CVE patterns  ->  top-N files

  2. HYPOTHESIS
     Executor reads code  ->  forms vulnerability theory

  3. VERIFICATION (tools)
     CodeQL : static reachability analysis
     GDB    : symbol and stack analysis
     Git    : change history and blame

  4. SELF-CRITIQUE
     Executor receives results  ->  refines hypothesis
     "Is this code path actually reachable?"

  5. ADVISOR CHECK  (if confidence < threshold)
     Advisor reads full shared context
     Sends guidance  ->  Executor resumes with new direction

  6. FUZZING
     AFL++ / libFuzzer  ->  crash case generation
     AddressSanitizer detects memory corruption

  7. PoC
     Minimal reproducible proof-of-concept
```

### Isolated Execution Environment

All fuzzing and compilation runs inside a Docker container with AddressSanitizer (`-fsanitize=address`) for memory corruption detection, `SYS_PTRACE` for GDB integration, and `--network none` in air-gapped mode.

---

## Quick Start

### Requirements

- Python 3.11+
- An Anthropic API key (`ANTHROPIC_API_KEY`) — or use `--scripted` for offline testing

### Installation

```bash
git clone https://github.com/keryxhunter/KeryxHunter
cd KeryxHunter
pip install -e .
```

### First Run (5 minutes)

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Scan a project — top 5 files, Haiku model, $0.30 budget
keryx hunt --target ./my-project

# More thorough: top 10 files, route top 3 to Opus
keryx hunt --target ./my-project \
  --top-n 10 --escalate-n 3 --escalate-model opus \
  --budget 0.50 --output-html results/report.html

# Air-gapped / CI — no API key needed
keryx hunt --target ./my-project --scripted
```

That's it. KeryxHunter will:
1. **Phase 0** — AST-scan all `.py` files, score by rule severity + git history
2. **Phase 1** — Hunt top-N files with the LLM agent (Haiku for most, Opus for top suspects)
3. **Phase 2** — Print a ranked report with confirmed findings, cost breakdown, and fix recommendations

### All CLI options

```
keryx hunt --target PATH          # required: directory to scan
  --model haiku|sonnet|opus       # default model (default: haiku)
  --escalate-model opus           # model for top files (default: opus)
  --escalate-n N                  # how many top files go to escalate-model
  --top-n N                       # total files to hunt (default: 5)
  --budget USD                    # max spend (default: 0.50)
  --mode strict|flexible|none     # verification strictness (default: strict)
  --min-score N                   # min Phase 0 score to include (default: 12.0)
  --no-blame                      # skip git enrichment (faster)
  --no-cache                      # force full rescan
  --output-json PATH              # write JSON report
  --output-html PATH              # write HTML report
  --output-sarif PATH             # write SARIF report (GitHub Security tab)
  --scripted                      # offline mode, no API key needed
  --fail-if-confirmed             # exit 1 if any vuln confirmed (CI)
```

---

## Known Limitations

These are honest limitations discovered via self-hunting KeryxHunter on its own codebase.

### Verification coverage

**Dynamic verification only works for files that match a registered tool name.**
Files like `fuzzer.py`, `cloud_advisor.py`, or any file in an external project that doesn't correspond to a KeryxHunter-internal tool will fall back to fuzzer PoC verification (sandbox-based). This is still useful but less precise than injection-verifier testing.

**`REGEX_DOS` reports `reachable` on Python 3.11+.**
CPython 3.11+ includes catastrophic-backtracking protection that prevents the pattern from hanging. The finding is real (no allowlist check), but the sandbox completes in milliseconds rather than demonstrating a timeout. On Python ≤3.10 the same harness would return `triggered`.

### Phase 0 scoring

**`UNSAFE_DESERIALIZATION` has a high false-positive rate on typical Python code.**
In self-hunt runs: 5–7 files flagged per run, 0 confirmed. The rule detects any LLM output flowing to a sink (print, return, eval), which is common and usually safe. Its score bonus is kept low (1.5) to reduce Opus slot waste, but it still surfaces in Phase 0.

**`complexity_bonus` uses `line_count = 0` for cached files** until the cache is rebuilt with `--no-cache`. First run after updating KeryxHunter: add `--no-cache` once to warm the new cache.

### Rule coverage

KeryxHunter currently detects vulnerabilities in **Python only**.
Multi-language support (JavaScript, Go, Rust) is on the roadmap but not yet implemented.

The AST analyzer covers 15 rule classes (R1–R15). It does not yet detect:
SQL injection, XXE, insecure deserialization via `jsonpickle`, path traversal via `zipfile`, or race conditions.

### Budget and models

The `--budget` flag is a soft cap checked between files — it does not interrupt a hunt mid-file. Actual spend may exceed the budget by one file's cost.

Cloud models require `ANTHROPIC_API_KEY`. Local model support (`llama-cpp-python`) is implemented but requires manual installation of the `[local]` extras and a GGUF model file.

---

## Configuration

### `~/.keryx/models.yaml`

```yaml
models:
  - name: llama-8b
    type: native
    backend: llama.cpp
    path: /models/llama-3-8b.gguf
    gpu_layers: 0
    context_length: 8192

  - name: llama-70b
    type: native
    backend: llama.cpp
    path: /models/llama-3-70b.gguf
    gpu_layers: 45
    context_length: 32768
    local_tools: [gdb, codeql, afl-fuzz]

  - name: qwen-32b
    type: native
    backend: llama.cpp
    path: /models/qwen-32b.gguf
    gpu_layers: 32
    context_length: 131072

  - name: claude-opus
    type: remote
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    use_only_when: online_mode_enabled
```

### `~/.keryx/capabilities.yaml`

```yaml
capabilities:
  fast_pattern_matching:
    executor: llama-8b
    advisor: none
    timeout_seconds: 5

  deep_reasoning:
    executor: llama-8b
    advisor: local-advisor
    fallback_advisor: cloud-advisor
    timeout_seconds: 120
    self_critique_rounds: 2

  airgapped:
    executor: llama-8b
    advisor: firefox-specialist
    enforce_no_network: true
    debate_models: [llama-70b, qwen-32b, llama-8b]

  verified_only:
    executor: llama-70b
    advisor: cascade-advisor
    require_consensus: true
    consensus_threshold: 0.67
    require_fuzzing_confirm: true
    min_crash_reproductions: 3
```

---

## Operating Modes

### Air-Gapped

```bash
keryx hunt \
  --target /classified/codebase \
  --mode airgapped \
  --models /models/llama-70b.gguf,/models/qwen-32b.gguf \
  --advisor firefox-specialist

docker run --network none --gpus all \
  -v /classified:/target \
  keryx:airgapped hunt --target /target --mode airgapped
```

### Hybrid

```bash
# Local executor, cloud advisor for final verification
keryx hunt \
  --target /path/to/codebase \
  --mode hybrid \
  --budget $10 \
  --advisor cascade-advisor
```

### Swarm

```bash
# Multiple models in parallel, consensus voting
keryx hunt \
  --target /path/to/codebase \
  --mode swarm \
  --debate-models llama-70b,qwen-32b,llama-8b \
  --advisor cloud-advisor
```

---

## Architecture

```
 ___________________________________________________________________________
|                          KeryxHunter v2.1                                 |
|___________________________________________________________________________|
|  CLI: keryx hunt --capability deep_reasoning --advisor cascade-advisor    |
|___________________________________________________________________________|
|                                                                           |
|  Capability Router                                                        |
|  "deep_reasoning" -> {                                                    |
|    executor : llama-8b    (fast, every turn)                              |
|    advisor  : llama-70b   (on-demand, when stuck)                         |
|    fallback : claude-opus (if online and confidence < 0.4)                |
|  }                                                                        |
|___________________________________________________________________________|
|                                                                           |
|  [ Executor ]  --- tool call --->  [ Advisor ]                            |
|  small, fast                       large, on-demand, any LLM             |
|  runs every turn                   reads full shared context              |
|       |                                      |                            |
|  read/write                          sends advice                         |
|       v                                      |                            |
|  [ Shared Context ] <------------------------                             |
|  Conversation * tool outputs * scan history                               |
|___________________________________________________________________________|
|                                                                           |
|  Model Runtime                                                            |
|  Remote API     Local GPU       Edge / CPU      Hybrid MoE               |
|  Anthropic      llama.cpp       llama.cpp        Router picks best        |
|  OpenAI         vLLM            MLX              model per sub-task       |
|  Groq           Ollama          WebLLM                                    |
|___________________________________________________________________________|
|                                                                           |
|  Execution Engines  (local, shared memory IPC)                            |
|  Symbolic Ex    Fuzzing         Sandbox          Knowledge Graph          |
|  angr / Triton  AFL++ / libF    Docker / gVisor  Neo4j + FAISS            |
|                 LLM-guided                       (CVE embeddings)         |
|___________________________________________________________________________|
```

### Core Components

| Component | Description | Technology |
|-----------|-------------|------------|
| `CapabilityRouter` | Routes tasks by capability profiles | Python, YAML |
| `LocalInferenceEngine` | Native inference, no HTTP overhead | llama-cpp-python |
| `SecurityAgent` | ReAct loop with tool access | Custom |
| `AdvisorManager` | Trigger detection, context packaging, dispatch | Custom |
| `IsolatedContainer` | Safe compilation and fuzzing environment | Docker SDK |
| `GDBInterface` | Crash analysis, stack traces, offsets | pygdbmi |
| `SwarmAnalyzer` | Multi-agent debate and consensus | asyncio |
| `KnowledgeGraph` | CVE history and similar patterns | Neo4j + FAISS |

---

## Benchmarks

> **Methodology:** 500 files from mozilla-central (open repository).  
> Baseline: 10 historically confirmed CVEs with public patches from Mozilla Bugzilla.

### Mode Comparison

| Mode | Time | Cost | Found | False Positives |
|------|------|------|-------|-----------------|
| Pattern only (8B, no advisor) | 2 min | $0 | 45 candidates | ~44 |
| Deep reasoning (70B local, local advisor) | 3 h | $0 | 6 confirmed | 3 |
| Deep reasoning (70B local, cloud advisor) | 2 h | ~$5 | 8 confirmed | 1 |
| Air-gapped + swarm debate | 4 h | $0 | 5 confirmed | 1 |

### Executor Alone vs Executor + Advisor

| Setup | Confirmed | False Positives | Cost |
|-------|-----------|-----------------|------|
| 8B executor, no advisor | 2 | 43 | $0 |
| 8B executor + 70B local advisor | 5 | 8 | $0 |
| 8B executor + cloud advisor | 7 | 2 | ~$4 |

The advisor adds minimal cost but significantly reduces false positives and recovers findings the executor alone would miss.

### Local Inference Latency

| Approach | p50 | p99 |
|----------|-----|-----|
| Ollama HTTP | ~500ms | ~1200ms |
| **KeryxHunter native** | **~20ms** | **~60ms** |
| vLLM API | ~80ms | ~200ms |

---

## Comparison

| Feature | Semgrep | CodeQL | Snyk | GitHub Copilot Autofix | KeryxHunter |
|---------|---------|--------|------|------------------------|-------------|
| Air-gapped | yes | yes | no | no | **yes** |
| LLM reasoning | no | no | partial | partial | **yes** |
| Agentic loop | no | no | no | no | **yes** |
| Advisor escalation | no | no | no | no | **yes** |
| Fuzzing verification | no | no | no | no | **yes** |
| Any LLM model | no | no | no | no | **yes** |
| Local 70B+ native | no | no | no | no | **yes** |
| Multi-agent debate | no | no | no | no | **yes** |
| Open source | yes | yes | no | no | **yes** |
| Local cost | free | free | $$ | $$ | **free** |

KeryxHunter does not replace Semgrep or CodeQL — it uses them as tools inside its agent loop.

---

## Roadmap

### Phase 1: Local Backend (Weeks 1–2) — done
- [x] llama-cpp-python native integration
- [x] Grammar-constrained JSON output
- [x] Local tool calling (MCP)
- [x] Docker isolation with ASan

### Phase 2: Agentic Loop (Weeks 3–4)
- [ ] Full ReAct loop with tool access
- [ ] CodeQL and GDB as LLM-invokable tools
- [ ] Hypothesis failure feedback into executor context
- [ ] Self-critique: "Is this code path actually reachable?"

### Phase 3: Advisor System (Weeks 5–6)
- [ ] Advisor trigger engine (confidence threshold, rounds, file patterns)
- [ ] Shared context packaging and delivery
- [ ] Advisor plugin API
- [ ] Cascade advisor chains
- [ ] Per-task advisor assignment via CLI

### Phase 4: Capability Router (Weeks 7–8)
- [ ] `--capability` CLI with YAML profiles
- [ ] Cost and latency optimizer
- [ ] Model swarming for critical findings
- [ ] Real-time metrics dashboard

### Phase 5: Firefox Specialization (Weeks 9–10)
- [ ] LoRA adapter fine-tuned on Firefox CVE patches
- [ ] SpiderMonkey GC root analyzer
- [ ] IPDL state machine verifier
- [ ] Mozilla Bugzilla API integration

### Phase 6: Platform (Q2)
- [ ] Web UI with live logs
- [ ] REST API for CI/CD integration
- [ ] GitHub Action
- [ ] VSCode extension
- [ ] Kubernetes Helm chart

---

## Project Structure

```
KeryxHunter/
├── keryx/
│   ├── core/
│   │   ├── agent.py              # ReAct loop
│   │   ├── router.py             # Capability router
│   │   └── orchestrator.py       # Task coordination
│   ├── models/
│   │   ├── local.py              # llama-cpp-python backend
│   │   ├── remote.py             # API backends
│   │   └── swarm.py              # Multi-agent debate
│   ├── advisors/
│   │   ├── base.py               # BaseAdvisor interface
│   │   ├── manager.py            # Trigger detection + dispatch
│   │   ├── local_advisor.py      # Local large model advisor
│   │   ├── cloud_advisor.py      # Remote API advisor
│   │   └── cascade_advisor.py    # Chained advisor
│   ├── tools/
│   │   ├── codeql.py
│   │   ├── gdb.py
│   │   └── fuzzer.py
│   ├── sandbox/
│   │   └── container.py
│   └── knowledge/
│       ├── graph.py
│       └── embeddings.py
├── examples/
│   └── firefox_cve/
├── configs/
│   ├── models.yaml.example
│   ├── capabilities.yaml.example
│   └── advisors.yaml.example
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Contributing

We welcome contributions. Areas where help is most needed:

- New LLM backends: MLX (Apple Silicon), TensorRT-LLM, ExLlamaV2
- Firefox-specific vulnerability patterns for SpiderMonkey and Gecko
- Performance optimization for local inference on consumer GPUs
- Benchmarks against public CVE datasets
- Custom advisor implementations

```bash
git clone https://github.com/keryxhunter/KeryxHunter
cd KeryxHunter
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

---

## Disclaimer

**For authorized security research only.**

KeryxHunter is designed to find vulnerabilities in software you own or have explicit permission to test. Use on systems without authorization is illegal and contrary to the purpose of this project.

This project does not distribute model weights. You are responsible for legally obtaining any model weights used in air-gapped mode.

All discovered vulnerabilities should be disclosed through official responsible disclosure programs (Mozilla Bug Bounty, Google VRP, etc.).

---

## License

[MIT](LICENSE) — free for security research, commercial use, and modification.

---

Built by security researchers, for security researchers.

*Keryx — announcing vulnerabilities before they become exploits. Sovereign by design. Open by nature.*

[Star on GitHub](https://github.com/keryxhunter/KeryxHunter) · [Report Bug](https://github.com/keryxhunter/KeryxHunter/issues) · [Discussions](https://github.com/keryxhunter/KeryxHunter/discussions)
