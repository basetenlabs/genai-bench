#!/bin/bash
# Baseten Benchmark Test Script
# Endpoint: https://model-e3mgvgo3.api.baseten.co/deployment/qeev0pq/predict

echo "🚀 GenAI Bench - Baseten Endpoint Test"
echo "======================================"
echo ""

# Configuration
export BASETEN_API_KEY="<baseten_api_key>"
API_BASE="https://model-e3mgvgo3.api.baseten.co/deployment/qeev0pq/predict"
STREAMING_PORT=8080


echo "📡 Endpoint: $API_BASE"
echo "🔌 Streaming Port: $STREAMING_PORT"
echo ""

# Check if API key is provided
if [ -z "$BASETEN_API_KEY" ]; then
    echo "⚠️  WARNING: BASETEN_API_KEY environment variable not set"
    echo "   Please set your Baseten API key:"
    echo "   export BASETEN_API_KEY='your-api-key-here'"
    echo ""
    read -p "Enter your Baseten API key: " BASETEN_API_KEY
    export BASETEN_API_KEY
fi

echo "🔧 Running benchmark with streaming..."
echo ""

# Run the benchmark
uv run genai-bench benchmark \
    --api-backend baseten \
    --api-base "$API_BASE" \
    --api-key "$BASETEN_API_KEY" \
    --task "text-to-text" \
    --api-model-name "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --model-tokenizer "Qwen/Qwen3-Next-80B-A3B-Instruct" \
    --max-requests-per-run 64 \
    --num-concurrency 2 \
    --num-concurrency 4 \
    --num-concurrency 8 \
    --num-concurrency 16 \
    --num-concurrency 24 \
    --num-concurrency 32 \
    --max-time-per-run 600 \
    --traffic-scenario "D(2000,500)" \
    --enable-streaming \
    --streaming-port $STREAMING_PORT

echo ""
echo "✅ Benchmark completed!"
echo "🌐 Dashboard was available at: http://localhost:$STREAMING_PORT"
echo "🔌 WebSocket URL was: ws://localhost:$STREAMING_PORT/ws"
