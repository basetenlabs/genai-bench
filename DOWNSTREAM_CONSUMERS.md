# Downstream Consumers of genai-bench

**Date:** 2026-04-10

genai-bench has two known downstream consumers. Neither is an upstream dependency — genai-bench has zero references to either project. Both install genai-bench from `origin/main` HEAD at runtime with no version pin.

---

## genai-bench-ui

**Repo:** `genai-bench-ui`
**Direction:** genai-bench-ui depends on genai-bench (one-way).

### How it uses genai-bench

Installs genai-bench from GitHub at runtime inside a Baseten training job (`truss train`), then invokes `genai-bench benchmark` as a CLI subprocess. No Python imports cross the boundary.

**Installation (in `jobs/training_runner.py`):**
```
uv run -p 3.13 --with tiktoken \
  --with 'git+https://github.com/basetenlabs/genai-bench.git' \
  genai-bench benchmark <args>
```

**CLI command construction:** `jobs/views.py:_build_genai_bench_cmd`

### Data contract

**Output files consumed:**
- JSON result files matching `*_(concurrency|batch_size)_*_time_*s.json`
- `experiment_metadata.json`
- Read from the job's checkpoint directory (`$BT_CHECKPOINT_DIR`)

**Fields read (in `results/utils.py`):**
- `aggregated_metrics.stats.ttft.{mean,p50,p90,p99}`
- `aggregated_metrics.stats.e2e_latency.{mean,p50,p90,p99}`
- `aggregated_metrics.mean_output_throughput_tokens_per_s`
- `experiment_metadata.json` for run-level config (`api-base`, `model`, `traffic_scenario`, `num_concurrency`, etc.)

Parsed results are uploaded to S3 under `genai-bench-ui/{workspace_id}/{run_id}/`.

---

## model-registry

**Repo:** `model-registry`
**Direction:** model-registry depends on genai-bench (one-way).

### How it uses genai-bench

A GitHub Actions workflow (`llm_benchmark.yml`, manually dispatched) installs genai-bench from GitHub at runtime, then invokes the CLI. No pinned version — always pulls latest `main`.

**Installation (in `.github/actions/run-llm-bench/main.py:build_command()`):**
```
uv run --with git+https://github.com/basetenlabs/genai-bench.git \
  genai-bench benchmark <args>
```

**CLI flags passed:**
- `--api-backend baseten`
- `--execution-engine async`
- `--task text-to-text`
- `--api-base <endpoint_url>` (format: `https://model-{model_id}.api.baseten.co/deployment/{deployment_id}/predict`)
- `--api-model-name <hf_model_id>` (parsed from `config.yaml` weights source)
- `--max-time-per-run`, `--max-requests-per-run`, `--experiment-base-dir`, `--tokenizer-from-model`
- `--traffic-scenario` (e.g. `D(100,100)`, `D(1024,1024)`)
- `--num-concurrency` (e.g. 1, 4, 8)
- `--server-gpu-type`, `--server-gpu-count`, `--server-engine`

### Data contract

**Output files consumed:** Same JSON result file pattern as genai-bench-ui.

**Fields read:**
- `aggregated_metrics.stats.ttft.{mean,p50,p90,p99}` (seconds, converted to ms)
- `aggregated_metrics.stats.tpot.{mean,p50,p90,p99}`
- `aggregated_metrics.stats.e2e_latency.{mean,p50,p90,p99}`
- `aggregated_metrics.mean_output_throughput_tokens_per_s`
- `aggregated_metrics.mean_total_tokens_throughput_tokens_per_s`
- `aggregated_metrics.requests_per_second`
- `aggregated_metrics.error_rate`
- `aggregated_metrics.num_completed_requests`
- `aggregated_metrics.scenario`
- `aggregated_metrics.num_concurrency`

**Results stored:** Committed to `benchmarking/<model-directory>/results.csv` and uploaded to S3.

**Trigger:** `workflow_dispatch` with inputs `model-directory` (e.g. `llm/qwen3.5-4b/latency`) and `profile` (`smoke` or `full`).

---

## Sync Risk Assessment

Both consumers install from `origin/main` HEAD with no version pin. Any breaking change to genai-bench's CLI flags or JSON output schema will immediately affect them.

### Upstream changes that could break consumers

1. **`mean_total_chars_per_hour` removal** (upstream commit `e6ad58a`) — if either consumer reads this field, it will be missing after merge. Needs verification.
2. **New metric fields** (e.g. `num_reasoning_tokens`) — additive, shouldn't break, but parsers with strict schemas might reject unknown fields.
3. **CLI flag renames or removals** — both consumers construct CLI commands programmatically; any flag change breaks invocation.
4. **JSON output structure changes** — any restructuring of `aggregated_metrics.stats.*` would break both consumers' parsing logic.
