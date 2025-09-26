#!/bin/bash
# GenAI Bench - Benchmark Runner Script

set -e

echo "🎯 Starting GenAI Bench benchmark..."

uv run --with git+https://github.com/basetenlabs/genai-bench.git genai-bench benchmark \
    --api-backend baseten \
    --api-base "https://model-yqvy8neq.api.baseten.co/environments/production/predict" \
    --server-gpu-type "H100" \
    --server-gpu-count 1 \
    --task "text-to-text" \
    --api-model-name "Qwen3-30B-A3B-Instruct-2507-FP8" \
    --model-tokenizer "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --max-requests-per-run 64 \
    --max-time-per-run 30 \
    --num-concurrency 2 \
    --num-concurrency 4 \
    --num-concurrency 8 \
    --num-concurrency 16 \
    --num-concurrency 32 \
    --num-concurrency 64 \
    --traffic-scenario "N(2000,200)/(200,20)" \
    --experiment-folder-name "baseten_training_benchmark" \
    --experiment-base-dir "$BT_CHECKPOINT_DIR"

echo "✅ Benchmark completed!"
