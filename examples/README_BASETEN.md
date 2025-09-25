# Baseten Benchmark Examples

This directory contains examples for running benchmarks against Baseten model endpoints with real-time streaming capabilities.

## Quick Start

### 1. Test the Endpoint

First, test that your Baseten endpoint is working:

```bash
python test_baseten_endpoint.py --api-key "your-api-key" --model "your-model-name"
```

### 2. Run a Benchmark

Run a full benchmark with streaming:

```bash
python run_baseten_benchmark.py --api-key "your-api-key" --model "your-model-name"
```

### 3. View Results

Open your browser and navigate to `http://localhost:8080` to see the real-time dashboard.

## Files

### `test_baseten_endpoint.py`
Simple test script to verify the Baseten endpoint is working before running benchmarks.

**Features:**
- Tests basic connectivity
- Verifies response format
- Tests concurrent requests
- Measures basic latency

**Usage:**
```bash
python test_baseten_endpoint.py --api-key "pt_abc123" --model "my-model"
```

### `run_baseten_benchmark.py`
Full benchmark runner for Baseten models with streaming.

**Features:**
- Runs comprehensive benchmarks
- Real-time streaming dashboard
- Configurable parameters
- Results storage

**Usage:**
```bash
python run_baseten_benchmark.py --api-key "pt_abc123" --model "my-model"
```

### `streaming_example.py`
Comprehensive example showing all streaming capabilities.

**Features:**
- CLI usage examples
- Programmatic usage
- Baseten-specific examples
- Custom frontend examples

**Usage:**
```bash
python streaming_example.py
```

## Configuration

### Required Parameters

- `--api-key`: Your Baseten API key (starts with `pt_`)
- `--model`: The name of your model on Baseten

### Optional Parameters

- `--streaming-port`: Port for the streaming dashboard (default: 8080)
- `--max-time`: Maximum time per run in seconds (default: 60)
- `--max-requests`: Maximum requests per run (default: 100)
- `--concurrency`: Concurrency levels to test (default: 1,2,4)
- `--scenario`: Traffic scenario (default: D(100,100))

## Baseten-Specific Configuration

### API Endpoint
The examples use the Baseten endpoint:
```
https://model-yqvy8neq.api.baseten.co/environments/production/predict
```

### Authentication
Baseten uses API key authentication with the header:
```
Authorization: Bearer pt_your_api_key_here
```

### Model Format
Baseten models are referenced by their model name as configured in your Baseten workspace.

## Example Workflow

### 1. Setup
```bash
# Install dependencies
pip install genai-bench aiohttp

# Set your API key (optional)
export BASETEN_API_KEY="pt_your_api_key_here"
```

### 2. Test Connection
```bash
python test_baseten_endpoint.py --api-key "pt_your_api_key" --model "your-model"
```

### 3. Run Benchmark
```bash
python run_baseten_benchmark.py --api-key "pt_your_api_key" --model "your-model"
```

### 4. Monitor Results
- Open `http://localhost:8080` in your browser
- Watch real-time metrics and progress
- View completed benchmark results

## CLI Usage

You can also use the genai-bench CLI directly:

```bash
genai-bench benchmark \
    --api-backend baseten \
    --api-base "https://model-yqvy8neq.api.baseten.co/environments/production/predict" \
    --api-key "pt_your_api_key" \
    --api-model-name "your-model" \
    --model-tokenizer "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --model "your-model" \
    --task text-to-text \
    --max-time-per-run 60 \
    --max-requests-per-run 100 \
    --num-concurrency 1,2,4 \
    --traffic-scenario "D(100,100)" \
    --enable-streaming \
    --streaming-port 8080
```

## Troubleshooting

### Common Issues

**"Endpoint test failed"**
- Check your API key is correct
- Verify the model name exists in your Baseten workspace
- Ensure the endpoint URL is correct

**"Benchmark failed"**
- Check network connectivity
- Verify API rate limits
- Ensure model is deployed and running

**"Dashboard not loading"**
- Check if port 8080 is available
- Verify firewall settings
- Try a different port with `--streaming-port`

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python run_baseten_benchmark.py --api-key "pt_your_api_key" --model "your-model"
```

## Results

### Output Files
Benchmarks create results in the `experiments/` directory:
- JSON files with detailed metrics
- Excel reports with aggregated data
- Plots and visualizations

### Dashboard Features
- Real-time latency and throughput metrics
- Interactive histograms and scatter plots
- Progress tracking
- Live log streaming
- Historical data access

## Advanced Usage

### Custom Parameters
```bash
python run_baseten_benchmark.py \
    --api-key "pt_your_api_key" \
    --model "your-model" \
    --max-time 120 \
    --max-requests 500 \
    --concurrency "1,2,4,8" \
    --scenario "D(100,100),D(200,200)"
```

### Programmatic Usage
```python
from genai_bench.streaming import StreamingDashboard
import asyncio

async def run_custom_benchmark():
    dashboard = StreamingDashboard(port=8080)
    await dashboard.start()
    
    # Your custom benchmark logic here
    # The dashboard will automatically receive metrics
    
    await dashboard.stop()

asyncio.run(run_custom_benchmark())
```

## Support

For issues with:
- **Baseten API**: Check the [Baseten documentation](https://docs.baseten.co/)
- **GenAI Bench**: Check the [GenAI Bench documentation](https://github.com/sgl-project/genai-bench)
- **Streaming**: Check the streaming guide in `docs/streaming-guide.md`
