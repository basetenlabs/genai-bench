# Upstream Sync Analysis

**Date:** 2026-04-08
**Fork:** `basetenlabs/genai-bench` (origin)
**Upstream:** `sgl-project/genai-bench` (upstream)
**Common ancestor:** `dacc5be91a89144308914bcb17184086364e97bb`
**Divergence:** 75 commits on origin, 49 on upstream

---

## Baseten-Specific Changes (75 commits)

### 1. Baseten Provider Integration (10 commits)

New auth provider, `baseten_user.py`, `--disable-streaming` flag, API key filtering from metadata. Full test suite.

| Commit | Description |
|--------|-------------|
| `e5d4992` | Added Baseten auth classes, `--disable-streaming` CLI flag, routing in `unified_factory.py` |
| `6d04d88` | Expanded `baseten_user.py` with non-OpenAI endpoint support; full test suite (725 lines) |
| `946a257` | Added `docs/user-guide/baseten-support.md` |
| `d927fe2` | Fixed image URL construction in `baseten_user.py` |
| `cdee631` | Fixed `BASETEN_API_KEY` env var support in `cli.py` and `validation.py` |
| `372bfb2` | Validation also checks `MODEL_API_KEY` env var |
| `eaeee9f` | Filters sensitive info (API keys) from run metadata |
| `4d7bc05` | Demoted request preview to debug logging |
| `e89b3d1` | Added `custom_message` override, enhanced `data/config.py` |
| `c256424` | Formatting cleanup |

**Key files:** `auth/baseten/`, `user/baseten_user.py`, `cli.py`, `validation.py`

### 2. Async Runner (3 commits)

Complete alternative execution engine (`--execution-engine async`) with open-loop and closed-loop modes.

| Commit | Description |
|--------|-------------|
| `059245a` | Full `async_runner/` package: `base.py`, `closed_loop.py`, `open_loop.py`, `factory.py`, `runner.py`; CLI integration; 10+ test files |
| `ae1d8a4` | Removed per-request timeout from closed-loop runner |
| `b8a0204` | Fixed metrics collection for incomplete/cancelled async tasks |

**Key files:** `async_runner/` (entire new package), `cli.py`, `tests/async_runner/`

### 3. Benchmarking Features (13 commits)

| Commit | Description |
|--------|-------------|
| `f2b7bec` | Multi-model benchmarking: `--api-model-name` accepts multiple values, `--tokenizer-from-model` flag |
| `5eeed23` | Fixed tokenizer not updating between models |
| `0941d33` | `prefix-repetition` traffic scenario for KV cache benchmarking |
| `7a8cbf1` | `TextSampler` prefix-repetition support |
| `38e79fc` | Opt-in network latency metrics (`--track-network-timing`) |
| `c13e019` | Sets `min_tokens` and `max_tokens` in requests |
| `93a1e2f` | `--no-min-tokens` flag for backends that don't support it |
| `cd4085c` / `3b30426` / `99631d3` | Reasoning tag support for GPT-OSS models |
| `7ee2b24` | Ensured `time_at_first_token` is never `None` |
| `92c2ebe` | TRT-LLM engine support; defaulted task/backend |
| `94ab808` | H100 MIG to allowed GPU list |

**Key files:** `cli.py`, `option_groups.py`, `sampling/text.py`, `sampling/base.py`, `scenarios/text.py`, `async_runner/base.py`, `metrics/metrics.py`, `protocol.py`, `user/openai_user.py`

### 4. Prompt Instruction Tuning (7 commits)

Iterative refinement of system prompt in `sampling/text.py` to maximize output token generation.

| Commit | Description |
|--------|-------------|
| `2c58b44`, `a1c3914`, `3f2b478`, `5c8e160`, `34ff11d`, `aece312`, `956201c` | Series of prompt placement, wording, and logging changes |

**Key files:** `sampling/text.py`

### 5. Metrics & Bug Fixes (8 commits)

| Commit | Description |
|--------|-------------|
| `faad8b6` | Division-by-zero guards, negative TTFT clamping, zero `run_duration` guard; `test_metrics_math_safety.py` (409 lines); added `CLAUDE.md` |
| `8f37ff3` | Removed filtering of incomplete requests from metrics/analysis |
| `46ce804` | Formatting pass |
| `fcbaa64` / `ebb6905` / `7a4b445` / `51d036f` / `e81fdae` | Token count adjustment logic and separator handling fixes |

**Key files:** `aggregated_metrics_collector.py`, `request_metrics_collector.py`, `sampling/text.py`, `excel_report.py`, `test_metrics_math_safety.py`

### 6. CI, Docs, Housekeeping (12 commits)

| Commit | Description |
|--------|-------------|
| `9896b7d` / `06605a6` | Coverage threshold bump, `.coveragerc` exclusion |
| `0b05e47` | Python version fix in CI |
| `c731ad1` / `e033671` / `944533c` | Fixed broken tests |
| `c737c99` | Added `torch` dependency |
| `8dedfc0` | Public repo warning in `CLAUDE.md` |
| `bf06f74` | mkdocs config for basetenlabs fork |
| `6e2f268` | CPU-only PyTorch install docs |
| `10147f7` | Performance troubleshooting guide |
| `00ef727` | Test cleanup |

**Key files:** `.github/workflows/ci.yml`, `.coveragerc`, `pyproject.toml`, `CLAUDE.md`, `docs/`

---

## Upstream Changes (49 commits)

### 1. New Features

| Commit | Description |
|--------|-------------|
| `5893a41` | **Text-to-image** task support (DALL-E style, new protocol classes, OCI OpenAI user, image prompts dataset) |
| `eb7317f` / `5dabdd8` | **Text-to-rerank** task for OpenAI-compatible backends |
| `ed72aac` | **Prefix caching** via `--prefix-len` CLI option |
| `7ce286d` | **HuggingFace local dataset loading** |
| `e26e59a` | **Min refresh interval** for metrics throttling |

### 2. Reasoning Token Reporting (9 commits)

Cross-cutting: adds `num_reasoning_tokens` to protocol/metrics across all user backends.

| Commit | Backend |
|--------|---------|
| `1cc360d` | Core API (`protocol.py`, `metrics.py`) |
| `fd4ec84` | OpenAI |
| `e0f637a` | Azure OpenAI |
| `aa8e3d8` | Cohere |
| `0cd8b46` | Together |
| `bcceb31` | OCI (genai + cohere) |
| `81ccb82` | AWS Bedrock |
| `7af3973` | GCP Vertex |
| `ae91f9c` / `ef148f2` | OCI GenAI + vLLM Harmony bugfixes |

### 3. Bug Fixes (9 commits)

| Commit | Description |
|--------|-------------|
| `39abdcb` | Filter TPOT/inference speed by output latency |
| `9991107` | Fix metrics filtering for non-streaming tasks |
| `41c4e9f` | OCI 429 error: metrics were being dropped |
| `3e86928` | SSE parser whitespace handling |
| `7eff699` | Better "Invalid Response" error messages |
| `5d32068` | Together auth Bearer token fix |
| `8e3d11a` | Together URL trailing slash handling |
| `ee65f3f` | Chat template for prefill token counting |
| `dd994aa` | Better token counting via OCI GenAI usage data |

### 4. Breaking Changes

| Commit | Description |
|--------|-------------|
| `e6ad58a` | **Removes `mean_total_chars_per_hour`** and `character_token_ratio` from metrics/protocol/CLI; replaced with `tokens/min` |
| `ee65f3f` | **Moves prefill token counting** from user `on_start` to `sampling/text.py`; breaks forks overriding `on_start` |
| `c872e2f` | **Gevent monkey-patching** at import time in `__init__.py`; async UI update changes |

### 5. CI/Infra

| Commit | Description |
|--------|-------------|
| `2a6c3c9` | Switch CI to `uv sync`, pin ruff |
| `01939ac` | Ruff upgrade + codebase reformat |
| `0a6b1f0` | Dependabot config |
| 8 commits | Dependabot GHA version bumps |
| `b434b38` | Ruff lint fix |
| `6d40268` | Deduplicate log warnings |
| `d0780ee` / `4c595eb` / `ed72999` | Fix `release.yml` errors |

### 6. Releases & Docs

| Commit | Description |
|--------|-------------|
| `e9a639f` | Version bump to 0.0.3 |
| `e11e3fd` | Version bump to 0.0.4 |
| `2365cee` | PR template update |
| `60f55df` | Bump minimum OCI SDK version |

---

## Conflict Zones

### HIGH Severity (same functions/lines modified by both sides)

| File | Origin Changes | Upstream Changes |
|------|---------------|-----------------|
| `genai_bench/cli/cli.py` | Execution engine options, `--no-min-tokens`, `disable_streaming`, param filtering | Gevent import, `metrics_options`, `prefix_len`/`prefix_ratio`, `oci-openai` backend |
| `genai_bench/metrics/aggregated_metrics_collector.py` | Disabled metrics filtering, added `OPTIONAL_METRICS_FIELDS` skip for network timing | Time-based update throttling, renamed `_should_filter_metrics`, nullifies bad values |
| `genai_bench/sampling/text.py` | `PrefixRepetitionScenario`, `_shared_prefix_cache`, conditional `ignore_eos` | Text-to-image, `MultiModality`, `prefix_len`/`prefix_ratio`, `warning_once` |
| `genai_bench/user/openai_user.py` | `disable_streaming` flag, conditional stream/stream_options | Text-to-rerank/image, reasoning content parsing, `warning_once` |
| `pyproject.toml` | Python `>=3.11,<3.14`, added `aiohttp`, `orjson`, `torch`, `pytest-asyncio` | Version 0.0.4, relaxed dep bounds, bumped `oci`, added `openai`, `httpx` |
| `tests/sampling/test_text.py` | `PrefixRepetitionScenario` tests, tokenizer mock resets | `ImageModality`, `DeterministicDistribution`, `warning_once` tests |

### MEDIUM Severity (nearby sections)

| File | Notes |
|------|-------|
| `genai_bench/analysis/flexible_plot_report.py` | Near-identical formatting changes, one line-split difference |
| `tests/metrics/test_metrics.py` | Origin inverted filtering assertion vs upstream adding reasoning tests |
| `.github/workflows/ci.yml` | Origin dropped Python 3.10 from matrix; upstream updated action versions |
| `.github/workflows/release.yml` | Coverage threshold touched by both; action version updates differ |

### Auto-Merged Cleanly (13 files)

`README.md`, `docs/user-guide/run-benchmark.md`, `docs/user-guide/scenario-definition.md`, `genai_bench/analysis/excel_report.py`, `genai_bench/cli/option_groups.py`, `genai_bench/cli/validation.py`, `genai_bench/metrics/metrics.py`, `genai_bench/metrics/request_metrics_collector.py`, `genai_bench/protocol.py`, `genai_bench/scenarios/base.py`, `genai_bench/user/aws_bedrock_user.py`, `tests/user/test_oci_genai_user.py`, `tests/user/test_openai_user.py`

### No Conflict Risk

- **98 files** only modified by origin
- **41 files** only modified by upstream

---

## Sync Strategy Considerations

1. **10 files** require manual conflict resolution
2. The hardest conflicts are in `cli.py`, `sampling/text.py`, `openai_user.py`, and `aggregated_metrics_collector.py` — both sides added substantial features to the same functions
3. Upstream breaking changes to watch for:
   - `mean_total_chars_per_hour` removal (check if Baseten code references it)
   - Prefill token counting move from `on_start` to `sampling/text.py`
   - Gevent monkey-patching at import time in `__init__.py`
4. Baseten's `prefix-repetition` scenario vs upstream's `--prefix-len` are different approaches to prefix caching benchmarks — decide whether to keep both or converge
