# Keryx Hunter — Session Change Report

---

## Сессия 2026-04-18 (рефакторинг монолита + фиксы)

**Ветка:** `v2`  
**Коммит на момент начала:** `c29db43` — "chore: prepare for project_hunt refactoring"

### Функциональные изменения

#### R12 TEMPLATE_INJECTION — точный детект алиасов

**Файл:** `keryx/tools/ast_analyzer.py`

Исправлена ложная пропасть: `from jinja2 import Template as T; T(x)` не детектировалось. `import jinja2 as j; j.Template(x)` давало MEDIUM вместо HIGH.

- Добавлены `_ssti_aliases: set[str]` и `_module_aliases: dict[str, str]` в `_VulnVisitor.__init__`
- `visit_ImportFrom` заполняет `_ssti_aliases` при импорте `Template` из опасного модуля под любым псевдонимом
- `visit_Import` отслеживает `import jinja2 as j` → `_module_aliases["j"] = "jinja2"`
- Паттерн A (голый вызов `Template`): HIGH если имя в `_ssti_aliases`, MEDIUM если модуль неизвестен
- Паттерн A (qualified `mod.Template`): разрешает алиасы модулей, HIGH/MEDIUM по каноническому имени
- Паттерн B (`from_string`): HIGH только для известных шаблонных движков — убраны ложные срабатывания на `LlamaGrammar.from_string` и подобные

#### Трёхуровневая верификация PoC

**Файл:** `keryx/fuzzing/poc.py` — добавлено поле `verification_level: Literal["reachable", "triggered", "exploited"]` в `PoCResult`.

- `exploited` — маркер найден в stdout (доказанное выполнение)
- `triggered` — ненулевой exit code или timeout (потенциальное срабатывание, покрывает REGEX_DOS)
- `reachable` — процесс запустился, маркер не найден

**Файл:** `keryx/core/verification_pipeline.py` — `_fuzz_verify` теперь возвращает `tuple[bool, str]`. `auto_verify` кодирует уровень в `method_tag`: `"fuzz_poc:exploited"` / `"fuzz_poc:triggered"` / `"fuzz_poc:reachable"`.

#### Rule Routing Weights

**Файл:** `keryx/core/hunt_models.py`

- Добавлен `RULE_ROUTING_WEIGHT: dict[str, float]` — множители приоритета для отдельных правил
- Добавлена функция `routing_score(sr: ScanResult) -> float` — применяет максимальный вес к итоговому скору при сортировке перед Phase 1 (не влияет на отображаемый AST-скор)

---

### Рефакторинг: разбиение монолита `project_hunt.py`

До рефакторинга в `scripts/project_hunt.py` было ~1427 строк — весь код в одном месте. После — 350 строк (CLI + `main()` + координация).

#### Новый файл: `keryx/core/scan_pipeline.py`

Перенесено из `project_hunt.py`:
- `load_cache`, `save_cache` — инкрементальный кэш Phase 0a
- `git_head(repo_root)`, `git_changed_since(commit, repo_root)` — инвалидация кэша по git
- `parse_ast_findings`, `scan_directory(..., repo_root, ...)` — параллельное AST-сканирование
- `enrich_with_blame(results, repo_root, scorer, since_days)` — git-обогащение Phase 0b
- Константы `CACHE_VERSION`, `EXCLUDE_ALWAYS`

Ключевое: все функции принимают `repo_root: Path` явно — глобальная переменная `_REPO_ROOT` не используется внутри модуля.

#### Новый файл: `keryx/core/hunt_pipeline.py`

Перенесено из `project_hunt.py`:
- `MODEL_IDS`, `STEPS_DEFAULT = 10`, `STEPS_ESCALATE = 15`
- `_load_api_key(repo_root)`, `build_model(model_name, budget, repo_root, scripted)`
- `hunt_file(scan, model, mode, max_steps, model_label, repo_root, ...)`

Новые публичные функции:
- `run_escalate_tier(...)` — Phase 1a: последовательный hunt для топ-файлов
- `run_default_tier(...)` — Phase 1b: параллельный hunt для остальных
- `run_escalation_phase(...)` — Phase 1c: авто-эскалация inconclusive HIGH + бюджет-гард

#### Итоговое состояние модулей

| Файл | Ответственность |
|---|---|
| `keryx/core/hunt_models.py` | Датаклассы, скоринг, routing weights |
| `keryx/models/scripted.py` | Детерминированная fallback-модель |
| `keryx/core/reporters.py` | Весь вывод: консоль, JSON, SARIF, HTML |
| `keryx/core/scan_pipeline.py` | Phase 0a/0b: AST-скан + git-обогащение |
| `keryx/core/hunt_pipeline.py` | Phase 1a/1b/1c: построение моделей + тиры |
| `scripts/project_hunt.py` | CLI, `main()`, координация (350 строк) |

---

### Тесты

Исправлены после изменения `_fuzz_verify` (теперь возвращает `tuple`):
- `test_auto_verify_no_target_tool`
- `test_returns_false_when_tool_not_resolved`
- `test_tool_not_resolved_fuzz_confirmed`

Исправлен `TestHtmlReport.test_vuln_row_rendered` и `test_xss_escaped_in_message` — импорт `ScanResult`, `HuntResult` перенесён с `scripts.project_hunt` на `keryx.core.hunt_models`.

**Итог: 1112 тестов, все проходят, coverage 88.5%.**

---
---

## Предыдущая сессия — добавление capability-слоёв

**Branch:** `v2`  
**Baseline commit:** `dde94da` — "fix: update import test cascade_advisor → cascade (post-reset cleanup)"  
**Report date:** 2026-04-18

---

## Overview

This session added three major capability pillars to Keryx Hunter: a complete fuzzing subsystem, advanced static analysis rules, three self-improvement mechanisms, and a CI/CD integration layer. The work extended the project from a 6-rule AST scanner with no dynamic verification into a 13-rule scanner with sandboxed proof-of-concept generation, SARIF reporting, and a full GitHub Actions pipeline.

---

## What Changed

### 1. New AST Security Rules (R11–R13)

The AST analyzer previously covered 6 vulnerability classes (R1–R6 plus R7–R10 added in prior sessions). This session added three new detection rules:

- **R11 — SSRF (Server-Side Request Forgery):** Detects calls to `requests.get/post/put/delete/patch` and similar HTTP methods where the URL argument comes from a variable or parameter rather than a constant string. The risk is that user-controlled URLs can force the server to make internal network requests.

- **R12 — TEMPLATE_INJECTION (Server-Side Template Injection):** Detects Jinja2 and similar template engines being instantiated with a non-constant string as the template source. When user input becomes the template itself (rather than just a render variable), it enables arbitrary code execution.

- **R13 — REGEX_DOS (Regular Expression Denial of Service):** Detects calls to `re.compile`, `re.match`, `re.search`, and similar functions where the regex pattern itself comes from user input. Attacker-crafted patterns with catastrophic backtracking can freeze a server for seconds or minutes.

The AST analyzer grew from **364 lines to 810 lines** to accommodate these rules plus the source_type tracking system described below.

---

### 2. Source-Type Annotation on Findings

Every finding produced by the AST analyzer now carries a `source_type` field that classifies where the dangerous argument came from:

- **`param`** — the argument is a function parameter or directly derived from one. Highest-risk: the taint flows directly from a caller.
- **`local`** — the argument is a local variable computed inside the function. Medium-risk: may be sanitized somewhere in the body.
- **`expression`** — the argument is a compound expression (f-string, concatenation, subscript, etc.) that contains at least one parameter name. High-risk.
- **`constant`** — the argument is a string or numeric literal. Not a finding in practice; used for safe-path verification.
- **`unknown`** — the call is at module level, outside any function scope.

This classification is achieved by maintaining a parameter-name stack that is pushed/popped as the visitor enters and exits function definitions, including nested and async functions.

---

### 3. Fuzzing Subsystem (new `keryx/fuzzing/` package)

A complete sandboxed proof-of-concept generation system was created from scratch. It consists of five modules:

- **`harness.py`** — The template engine. For each rule, it knows how to construct a minimal Python script that attempts to exercise the vulnerability. Templates exist for HARDCODED_SECRET (entropy analysis), SSRF (raw socket probe), TEMPLATE_INJECTION (Jinja2 SSTI payload), REGEX_DOS (catastrophic backtracking pattern), UNSAFE_PICKLE, UNSAFE_EVAL_EXEC, UNSAFE_YAML_LOAD, and several subprocess/command injection rules. Rules without a static template fall back to LLM generation.

- **`sandbox.py`** — Executes generated scripts in an isolated subprocess with a configurable CPU/time limit. Captures stdout and checks for the presence of a unique marker string to confirm exploitation.

- **`poc.py`** — Orchestrates the harness + sandbox loop: given a finding dict, it generates a script, runs it, and returns a structured result indicating whether the vulnerability was dynamically confirmed.

- **`tool.py`** — Wraps the PoC system as a `ToolResult`-compatible interface so the main agent can call it like any other tool.

- **`llm_harness.py`** — A factory that produces an LLM-powered harness generator. When called, it sends a structured prompt to any `ModelInterface` asking it to write a self-contained Python exploit script for a given finding. The returned script is validated for syntactic correctness and presence of the exploitation marker before being handed to the sandbox.

---

### 4. SARIF 2.1.0 Output

A new module `keryx/core/sarif.py` converts hunt results into the SARIF 2.1.0 format used by the GitHub Security tab. Key behaviors:

- Every confirmed finding becomes a SARIF `result` with physical location (file + line), severity level, rule metadata, and a human-readable message.
- All 13 rules have entries in the rule metadata table with camelCase IDs, short descriptions, and remediation guidance.
- File URIs are handled in three cases: already-relative paths, absolute paths within the repository (made relative), and absolute paths outside the repository (kept as-is with a `%SRCROOT%` prefix).
- When a finding was dynamically verified by the fuzzer, the fuzz proof is attached to the result's `properties` object.

The `--output-sarif PATH` flag in `project_hunt.py` writes this file.

---

### 5. Verification Pipeline

A new `keryx/core/verification_pipeline.py` module bridges the LLM hunt output and the fuzzer. It:

- Parses the raw text output from the LLM agent to extract structured finding dictionaries.
- Maps variable names seen in the code context to the correct fuzzing injection field using a `VARNAME_TO_FIELD` lookup table.
- Handles TEMPLATE_INJECTION and REGEX_DOS as special cases, since the injection point (template string vs. regex pattern) needs to be identified from the code snippet.
- Enriches findings with the `payload` from a central `PAYLOAD_MAP` (one canonical test payload per rule).

---

### 6. Self-Improvement: Rule Gap Detector

A new module `keryx/core/rule_learner.py` implements a lightweight post-hunt analysis pass. After the main hunt completes, it scans the same source files using a set of hand-written regex patterns to look for sink patterns not currently covered by the AST rule set. It produces a ranked list of suggestions:

- **10 gap patterns** covering: httpx HTTP methods, urllib.request.urlopen, aiohttp async sessions, f-string arguments to os.system, marshal.loads, shelve.open, xml.etree.ElementTree parsing, importlib.import_module with dynamic names, Mako templates, Chameleon templates, and chained re.compile calls.
- Each suggestion includes: a suggestion ID, the closest existing rule family it belongs to, how many files contain the pattern, up to five example file paths, a concrete code snippet, a description of the risk, and a recommended rule addition.
- Suggestions are ranked by occurrence count so the most widespread gaps appear first.

The `--learn` flag in `project_hunt.py` enables this pass.

---

### 7. Self-Improvement: LLM Harness Generator

`keryx/fuzzing/llm_harness.py` provides the `make_llm_generate_fn(model)` factory. It wraps any `ModelInterface` and returns a callable that the fuzzing harness can use as a fallback when no static template exists for a rule. The prompt instructs the LLM to write a self-contained, stdlib-only Python script that reconstructs the vulnerable pattern and writes a unique marker to stdout if exploitation succeeds. Two layers of validation are applied before the script reaches the sandbox: it must contain the marker string and must parse as valid Python.

The `--fuzz-llm` and `--fuzz-llm-model` flags in `project_hunt.py` enable this.

---

### 8. Self-Improvement: Source-Type Annotation

Described above under section 2. This is the "symbolic hints" component — it gives the LLM agent a pre-computed signal about whether a dangerous argument is directly parameter-tainted, helping it prioritize which findings are worth deeper investigation.

---

### 9. GitHub Actions Workflow (`.github/workflows/keryx-hunt.yml`)

A new dedicated security scan workflow separate from the existing CI workflow. Key design decisions:

- Triggers on push to `main`/`v2`, on every pull request targeting `main`, on a nightly cron schedule (02:17 UTC), and on manual dispatch with configurable parameters.
- Automatically detects whether an API key is available; falls back to `--scripted` mode for fork pull requests where secrets are not exposed.
- Uploads the SARIF report to the GitHub Security tab using `github/codeql-action/upload-sarif`.
- Uploads JSON, SARIF, and HTML reports as build artifacts with a 30-day retention period.
- Requires `security-events: write` permission for the SARIF upload.
- Fails the build if any confirmed vulnerability is found (`--fail-if-confirmed`).

---

### 10. Pre-Commit Hook Installer (`scripts/install_hooks.py`)

A new installer script that writes a git pre-commit hook into `.git/hooks/`. The installed hook:

- Collects all staged Python files from `git diff --cached`.
- Runs the AST scanner on each one.
- Blocks the commit with exit code 1 if any HIGH-severity finding is detected, printing a report of which rule fired on which file and line.
- Does nothing and exits 0 if all staged files are clean.

The installer accepts `--force` to overwrite an existing hook and `--repo-dir` to point at a custom repository root (used in tests). Bypass remains possible via `git commit --no-verify`.

---

### 11. HTML Report (`write_output_html` + `--output-html`)

A self-contained single-file HTML report generator added to `project_hunt.py`. It produces a page with:

- Summary cards: total files scanned, files hunted by the LLM, confirmed vulnerabilities, dynamically verified count.
- A confirmed vulnerabilities table with colored severity badges, rule name, file location, message, source type, and whether the fuzzer confirmed it.
- A top-50 scored files table showing AST score, HIGH/MEDIUM counts, and triggered rule names.
- A model cost breakdown table.

All user-supplied data (finding messages, file paths, rule names) is passed through `html.escape()` to prevent XSS in the report itself.

---

## Files Created This Session

| File | Lines | Purpose |
|---|---|---|
| `keryx/core/_constants.py` | 77 | Shared payload map and variable-name-to-field lookup table |
| `keryx/core/sarif.py` | 272 | SARIF 2.1.0 serialization |
| `keryx/core/verification_pipeline.py` | 250 | LLM output parsing and finding enrichment |
| `keryx/core/rule_learner.py` | 229 | Post-hunt rule gap detector |
| `keryx/fuzzing/__init__.py` | — | Package marker |
| `keryx/fuzzing/harness.py` | 385 | Static PoC template engine |
| `keryx/fuzzing/llm_harness.py` | 116 | LLM-powered harness factory |
| `keryx/fuzzing/poc.py` | 68 | PoC orchestration (harness + sandbox) |
| `keryx/fuzzing/sandbox.py` | 75 | Isolated subprocess executor |
| `keryx/fuzzing/tool.py` | 71 | Tool-compatible wrapper |
| `scripts/project_hunt.py` | 1,427 | Main hunt orchestrator (heavily extended) |
| `scripts/install_hooks.py` | 129 | Pre-commit hook installer |
| `.github/workflows/keryx-hunt.yml` | 94 | GitHub Actions security scan workflow |

---

## Files Modified This Session

| File | Change summary |
|---|---|
| `keryx/tools/ast_analyzer.py` | Added R11–R13 detection methods, source_type annotation, param-stack tracking. Grew from 364 → 810 lines. |
| `tests/test_keryx_suite.py` | Added 221 new test methods across 17 new test classes. Grew from 693 → 914 test functions. |

---

## Test Coverage

| Metric | Before session | After session |
|---|---|---|
| Total test functions | 693 | 914 |
| Tests passing | 693 | 1,112 (includes tests from other test files) |
| Line coverage | not measured here | **98.22%** |
| Coverage threshold | 60% | 60% (well exceeded) |
| Uncovered lines | — | 63 out of 3,544 |

---

## New Test Classes and What They Verify

| Test class | # tests | What it verifies |
|---|---|---|
| `TestHardcodedSecretHarness` | 11 | Entropy-based secret detection: JWT, Base64, high-entropy strings flagged; known placeholders and empty values treated conservatively |
| `TestSarifModule` | 21 | SARIF structure validity, rule metadata for all 13 rules, artifact URI handling for relative/absolute/external paths, fuzz proof attachment |
| `TestSSRFRule` | 18 | AST detection of SSRF patterns with requests library; safe constant URLs not flagged; source_type correctly identified |
| `TestSSRFFuzzerHarness` | 3 | SSRF harness generates a script; sandbox runs it; marker detection logic |
| `TestTemplateInjectionRule` | 16 | Jinja2 Template(var) and env.from_string(var) flagged; Template.render(name=var) not flagged; both sync and async contexts |
| `TestTemplateInjectionHarness` | 5 | Template injection PoC script generation and execution |
| `TestTemplateInjectionExtractFindings` | 5 | Verification pipeline correctly extracts injection field from template code context |
| `TestTemplateInjectionSarifMeta` | 2 | SARIF rule table includes TEMPLATE_INJECTION with correct severity and description |
| `TestRegexDosRule` | 20 | REGEX_DOS fires on user-controlled pattern argument; safe constant patterns not flagged; chained re.compile not a false negative |
| `TestRegexDosHarness` | 6 | ReDoS PoC script constructs catastrophic backtracking pattern; harness emits marker when pattern accepted without validation |
| `TestRegexDosExtractFindings` | 4 | Verification pipeline maps regex variable names to the pattern injection field |
| `TestRegexDosSarifMeta` | 2 | SARIF rule table includes REGEX_DOS |
| `TestSourceTypeAnnotation` | 15 | source_type is "param" for function parameters, "local" for computed locals, "param" for expressions containing params, "unknown" at module level, correct behavior with nested functions |
| `TestLLMHarnessFactory` | 8 | make_llm_generate_fn returns callable; validates marker presence; validates Python syntax; returns None on generation failure or garbage output |
| `TestRuleLearner` | 12 | Gap patterns fire on matching source text; multiple patterns detected across files; ranking by occurrence count; empty file list; unreadable files skipped gracefully |
| `TestGitHubActionsWorkflow` | 12 | YAML is valid; push/PR/schedule triggers present; SARIF upload step present; artifact upload present; security-events permission declared; --fail-if-confirmed and --output-html flags present |
| `TestPreCommitHookInstaller` | 7 | Hook created and made executable; --force overwrites; without --force existing hook is preserved; missing .git/ directory returns error code 1; hook body is syntactically valid Python |
| `TestHtmlReport` | 8 | File is created; DOCTYPE present; summary cards rendered; vulnerability row appears with correct rule and severity; empty-finding message shown; XSS characters in messages are HTML-escaped; parent directories created; --output-html accepted by argparse |

---

## Key Design Decisions

1. **No re-running AST analysis in the rule learner** — it operates purely by regex over already-read source text, adding less than 0.1 seconds to a full hunt.

2. **Three-case SARIF URI handling** — `Path.as_uri()` raises `ValueError` on relative paths, so the serializer explicitly handles already-relative paths, absolute in-repo paths (made relative), and absolute external paths.

3. **Stack-based param tracking** — the AST visitor maintains a stack of parameter name sets when entering/leaving function scopes, so nested functions and async functions are handled correctly without false "param" classifications leaking across scope boundaries.

4. **Double validation for LLM harnesses** — generated scripts must both contain the unique marker string and parse as valid Python via `ast.parse()` before reaching the sandbox. This prevents nonsensical or incomplete output from being executed.

5. **HTML XSS safety** — all dynamic content in the HTML report passes through `html.escape()`. The report is self-contained with no external dependencies (no CDN, no JavaScript frameworks).

6. **Scripted fallback in CI** — the GitHub Actions workflow detects the absence of `ANTHROPIC_API_KEY` and automatically adds `--scripted`, so fork pull requests (which cannot access secrets) still get a meaningful AST-only scan result rather than failing with an authentication error.
