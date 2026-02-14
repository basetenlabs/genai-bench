# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **PUBLIC REPOSITORY**: This is an open-source project. Do NOT commit:
> - API keys, secrets, or credentials
> - Internal Baseten URLs, model IDs, or infrastructure details
> - Proprietary business logic or customer data
> - Internal documentation or runbooks
>
> When in doubt, ask before committing.

## Build & Test Commands

```bash
# Install dependencies (uses uv + hatchling)
uv pip install -e ".[dev,multi-cloud]"

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/metrics/test_metrics.py -v

# Run with coverage report (CI threshold: 89%, badge says 93%)
uv run pytest tests/ --cov --cov-report=term-missing

# Lint and format
uv run ruff check genai_bench tests
uv run ruff format genai_bench tests

# Type checking
uv run mypy genai_bench --config-file=mypy.ini
```

## Architecture Overview

genai-bench is an LLM benchmarking CLI tool (`genai-bench benchmark|excel|plot`) built on Click. It measures token-level performance (TTFT, TPOT, E2E latency, throughput) across multiple cloud providers.

### Core Flow

The CLI entry point (`genai_bench/cli/cli.py`) orchestrates: auth setup → tokenizer loading → data loading → sampler creation → runner execution → metrics aggregation → report generation.

### Two Execution Engines

- **Locust engine** (default): Uses `DistributedRunner` (`genai_bench/distributed/runner.py`) with gevent-based concurrency. Each backend has a `BaseUser` subclass (Locust `HttpUser`) in `genai_bench/user/` that handles request formatting and response parsing.
- **Async engine** (`--execution-engine=async`): Uses `aiohttp` with either `OpenLoopRunner` (QPS-based) or `ClosedLoopRunner` (concurrency-based) from `genai_bench/async_runner/`. Factory in `async_runner/factory.py` selects based on `--qps-level` vs `--num-concurrency`.

### Key Abstractions (all use subclass registries)

- **Scenario** (`scenarios/base.py`): Defines traffic patterns via string DSL. `Scenario.from_string()` parses e.g. `N(480,240)/(300,150)`, `D(100,100)`, `E(64)`, `I(512,512)`, `dataset`. Subclasses auto-register via `__init_subclass__`.
- **Sampler** (`sampling/base.py`): Generates `UserRequest` objects from scenarios. Registry keyed by `input_modality`. `TextSampler` and `ImageSampler` are the main implementations.
- **Auth** (`auth/`): Two abstract hierarchies — `ModelAuthProvider` (API endpoint auth) and `StorageAuthProvider` (cloud storage auth). `UnifiedAuthFactory` creates providers for: openai, oci, aws-bedrock, azure-openai, gcp-vertex, baseten, together.
- **Storage** (`storage/`): `BaseStorage` ABC with implementations for OCI, AWS S3, Azure Blob, GCP Cloud Storage, GitHub. Created via `StorageFactory`.

### Request/Response Protocol

`genai_bench/protocol.py` defines Pydantic models: `UserRequest` → `UserChatRequest`, `UserImageChatRequest`, `UserEmbeddingRequest`, `UserReRankRequest`. Responses: `UserResponse` → `UserChatResponse`.

### Metrics Pipeline

`RequestMetricsCollector` computes per-request metrics → `AggregatedMetricsCollector` aggregates across a run → results saved as JSON. `MetricStats` provides percentile statistics (p25–p99). Analysis module generates Excel reports and plots.

### Backend User Classes

Each API backend has a user class in `genai_bench/user/` (e.g., `OpenAIUser`, `BasetenUser`, `AWSBedrockUser`). Mapped in `cli/validation.py:API_BACKEND_USER_MAP`. vLLM and SGLang reuse `OpenAIUser`.

## Test Conventions

- Tests mirror source structure under `tests/`
- `tests/conftest.py` provides `mock_tokenizer` (local bert-base-uncased from `tests/fixtures/`) and auto-resets `OpenAIUser` class attributes between tests
- Uses `pytest-asyncio` for async tests
- gevent monkey-patching is present for Locust compatibility
- Coverage config in `.coveragerc` omits UI, logging, analysis/plot modules

## Style

- Python 3.11+ (CI tests 3.11, 3.12)
- Ruff for linting and formatting (line length 88)
- isort with "black" profile and custom `LOCUST` section for locust imports
- Pydantic v2 for all data models
