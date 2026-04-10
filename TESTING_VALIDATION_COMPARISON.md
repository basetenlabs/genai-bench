# Testing & Validation Comparison: Origin vs Upstream

**Date:** 2026-04-08
**Fork:** `basetenlabs/genai-bench` (origin)
**Upstream:** `sgl-project/genai-bench` (upstream)

---

## CI Pipeline

| | Origin (Baseten) | Upstream |
|---|---|---|
| **Python matrix** | `3.11`, `3.12` | `3.10`, `3.11`, `3.12` |
| **Python requirement** | `>=3.11,<3.14` | `>=3.10,<3.13` |
| **Coverage threshold** | `89%` | `93%` |
| **Install style** | `python -m uv pip install --system` | `uv sync --extra dev --extra multi-cloud` |
| **Test runner** | `python -m pytest` | `uv run pytest` |
| **GHA action versions** | `checkout@v4`, `setup-python@v4`, `setup-uv@v3` | `checkout@v6`, `setup-python@v6`, `setup-uv@v7` |

**Workflow files:** Both sides have the same set — `ci.yml`, `release.yml`, `docs.yml`, `labeler.yml`.

---

## Linting / Formatting

| | Origin | Upstream |
|---|---|---|
| **ruff target-version** | `py313` | `py310` |
| **black target-version** | `py313` | `py310` |
| **Ignores E501** (line length) | Yes | No |

Origin ignores `E501` (line-too-long) in ruff, meaning some origin code has lines longer than 88 chars that upstream's ruff config would flag.

---

## Coverage Configuration

- **`.coveragerc`:** Origin adds one extra omit entry: `*/_remote_module_non_scriptable` (a Baseten-internal module path). Otherwise identical.
- **`pyproject.toml`:** Neither side has a `[tool.pytest.ini_options]` section — pytest config is driven solely by `.coveragerc` and CLI flags in CI.

---

## Test File Inventory

### Origin-only (17 files)

| Directory | Files | Tests for |
|-----------|-------|-----------|
| `tests/async_runner/` | 13 files (`test_arrival_metrics`, `test_arrival_pacing`, `test_base_async_runner`, `test_closed_loop_runner`, `test_factory`, `test_midstream_error`, `test_network_timing`, `test_open_loop_runner`, `test_qps`, `test_session_lifecycle`, `test_streaming`, `test_timeout_semantics`, `__init__.py`) | Async execution engine |
| `tests/auth/` | `test_baseten_auth.py` | Baseten auth provider |
| `tests/cli/` | `test_cli_async.py` | Async CLI options |
| `tests/metrics/` | `test_metrics_math_safety.py` | Division-by-zero, negative TTFT, edge cases |
| `tests/user/` | `test_baseten_user.py` | Baseten user/API client |

### Upstream-only (3 files)

| File | Tests for |
|------|-----------|
| `tests/metrics/test_metrics_interval.py` | `metrics_refresh_interval` on `AggregatedMetricsCollector` |
| `tests/user/test_oci_openai_user.py` | `OCIOpenAIUser` |
| `tests/user/test_together_user.py` | `TogetherUser` (uses `genai_logging`) |

### Shared conftest difference

Upstream adds an `autouse` fixture `reset_warning_once_cache` in `tests/conftest.py` that clears `genai_logging._warning_once_keys` between tests. Origin does not have this fixture.

---

## Merge Implications

### 1. Coverage threshold conflict

Origin lowered the threshold to 89%; upstream requires 93%. Origin-only code (async runner, baseten user) may not meet the 93% bar. Decision needed: keep 89% or invest in coverage to hit 93%.

### 2. Origin tests depend on origin source modules

The 17 origin-only test files import origin-only modules:
- `tests/async_runner/*` imports `genai_bench.async_runner`
- `tests/auth/test_baseten_auth.py` imports `genai_bench.auth.baseten`
- `tests/user/test_baseten_user.py` imports `genai_bench.user.baseten_user`

These will pass as long as origin's source modules are present in the merged codebase.

### 3. Upstream tests depend on upstream source and fixtures

`test_together_user.py` imports `genai_bench.user.together_user` and references `genai_logging._warning_once_keys`. The `reset_warning_once_cache` conftest fixture is also upstream-only. These must land together in the merge or tests will fail at import time.

### 4. Python 3.10 compatibility

Origin code targets `>=3.11` and uses `py313` as the ruff target. Upstream's CI matrix includes Python 3.10. After merge, any origin code using 3.11-only syntax (e.g., `tomllib` stdlib, `match` statements without backport, `ExceptionGroup`) will fail upstream's 3.10 CI job.

**Decision needed:** Drop 3.10 support (origin's stance) or ensure all code is 3.10-compatible.

### 5. Ruff formatting divergence

Upstream ran a codebase-wide ruff reformat (`01939ac`) targeting `py310`. Origin targets `py313` and ignores `E501`. After merge, running `ruff check` or `ruff format` will produce different results depending on which config wins. Aligning on a single ruff config is needed to avoid CI lint failures.

### 6. Stub test file

Origin's `tests/async_runner/test_midstream_error.py` is a pass-only stub that adds no coverage. If the coverage threshold is raised back toward 93%, this dead weight will need to be either implemented or removed.
