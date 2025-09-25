# GenAI Bench Streaming Guide

This guide explains how to use GenAI Bench's real-time streaming capabilities to monitor benchmarks through a web dashboard.

## Overview

GenAI Bench now supports real-time streaming of benchmark data to web clients, allowing you to:

- **Monitor benchmarks in real-time** through a web browser
- **View live metrics** including latency, throughput, and progress
- **See interactive visualizations** with charts and histograms
- **Access historical data** from completed benchmarks
- **Create custom frontends** that connect to the streaming API

## Quick Start

### 1. Run a Benchmark with Streaming

Enable streaming by adding the `--enable-streaming` flag to your benchmark command:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8082" \
    --api-key "your-api-key" \
    --model "gpt-3.5-turbo" \
    --task text-to-text \
    --max-time-per-run 60 \
    --max-requests-per-run 200 \
    --enable-streaming \
    --streaming-port 8080
```

### 2. Open the Dashboard

Once the benchmark starts, open your web browser and navigate to:

```
http://localhost:8080
```

You'll see a real-time dashboard showing:
- **Benchmark Status**: Current scenario, progress, and overall status
- **Live Metrics**: Real-time latency and throughput data
- **Visualizations**: Histograms and scatter plots
- **Logs**: Live log messages from the benchmark

## Features

### Real-Time Data Streaming

The streaming dashboard provides real-time updates for:

- **Metrics Panels**: Input/output latency and throughput
- **Histograms**: Distribution of latency values
- **Scatter Plots**: Throughput vs latency correlations
- **Progress Bars**: Overall and per-run progress
- **Log Messages**: Real-time log streaming

### WebSocket API

The streaming server exposes a WebSocket API at `/ws` for custom integrations:

```javascript
// Connect to the streaming API
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.event_type) {
        case 'metrics':
            // Handle live metrics data
            updateMetrics(data.data.live_metrics);
            break;
        case 'status':
            // Handle status updates
            updateStatus(data.data);
            break;
        case 'log':
            // Handle log messages
            addLog(data.data.message);
            break;
    }
};
```

### REST API Endpoints

The streaming server also provides REST endpoints:

- `GET /api/status` - Current benchmark status
- `GET /api/historical-data` - Historical benchmark data
- `GET /api/metrics` - Current metrics snapshot

## Advanced Usage

### Custom Frontend Integration

You can create custom frontends that connect to the streaming API:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Custom GenAI Bench Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div id="metrics"></div>
    <canvas id="chart"></canvas>
    
    <script>
        const ws = new WebSocket('ws://localhost:8080/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.event_type === 'metrics') {
                updateCustomDashboard(data.data.live_metrics);
            }
        };
        
        function updateCustomDashboard(metrics) {
            // Your custom visualization logic here
            document.getElementById('metrics').innerHTML = 
                `TTFT: ${metrics.stats.ttft.mean.toFixed(3)}s`;
        }
    </script>
</body>
</html>
```

### Programmatic Usage

You can also use the streaming dashboard programmatically:

```python
from genai_bench.streaming import StreamingDashboard
import asyncio

async def run_benchmark_with_streaming():
    # Create streaming dashboard
    dashboard = StreamingDashboard(port=8080)
    
    # Start the streaming server
    server_task = asyncio.create_task(dashboard.start())
    
    # Update benchmark status
    dashboard.update_benchmark_status(
        status="running",
        current_scenario="D(100,100)",
        progress_percentage=25.0
    )
    
    # Update metrics
    live_metrics = {
        "ttft": [0.1, 0.15, 0.12],
        "stats": {
            "ttft": {"mean": 0.12, "min": 0.1, "max": 0.15}
        }
    }
    dashboard.update_metrics_panels(live_metrics)
    
    # Keep running
    await asyncio.sleep(60)
    
    # Stop the server
    server_task.cancel()
    await dashboard.stop()

# Run the example
asyncio.run(run_benchmark_with_streaming())
```

### Standalone Streaming Server

You can start a standalone streaming server for viewing historical data:

```bash
genai-bench start-streaming-server --port 8080
```

This starts only the web server without running a benchmark, useful for viewing results from previously completed benchmarks.

## Configuration Options

### Streaming Options

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-streaming` | `False` | Enable real-time streaming |
| `--streaming-port` | `8080` | Port for the streaming server |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_STREAMING` | `false` | Enable streaming (alternative to CLI flag) |
| `STREAMING_PORT` | `8080` | Streaming server port |

## Architecture

### Components

1. **StreamingDashboard**: Main dashboard class that handles data streaming
2. **StreamingServer**: WebSocket and HTTP server for client connections
3. **Event System**: Asynchronous event queue for real-time updates
4. **Web Frontend**: Built-in HTML dashboard with Chart.js visualizations

### Data Flow

```
Benchmark Execution → Metrics Collection → StreamingDashboard → WebSocket → Frontend
```

1. **Benchmark runs** and collects metrics
2. **Metrics are aggregated** in real-time
3. **StreamingDashboard** queues events for streaming
4. **WebSocket server** broadcasts events to connected clients
5. **Frontend** receives and visualizes the data

### Event Types

The streaming system supports these event types:

- `metrics`: Live metrics data with statistics
- `histogram`: Histogram data for latency distributions
- `scatter`: Scatter plot data for throughput vs latency
- `status`: Benchmark status and progress
- `log`: Log messages
- `progress`: Progress updates
- `heartbeat`: Connection keep-alive

## Troubleshooting

### Common Issues

**Dashboard not loading:**
- Check if the streaming server is running on the correct port
- Ensure no firewall is blocking the connection
- Verify the benchmark is running with `--enable-streaming`

**No data appearing:**
- Check that the benchmark is actively running
- Verify WebSocket connection in browser developer tools
- Look for error messages in the benchmark logs

**Performance issues:**
- Reduce the number of connected clients
- Increase the event queue timeout
- Monitor server resource usage

### Debug Mode

Enable debug logging for the streaming components:

```bash
export LOG_LEVEL=DEBUG
genai-bench benchmark --enable-streaming ...
```

## Examples

See the `examples/streaming_example.py` file for complete examples of:

1. Running benchmarks with streaming via CLI
2. Using the streaming dashboard programmatically
3. Creating custom frontends
4. Integrating with existing applications

## Dependencies

The streaming functionality requires these optional dependencies:

```bash
pip install fastapi uvicorn websockets
```

If these are not installed, the streaming functionality will be disabled with a warning message.

## API Reference

### StreamingDashboard

```python
class StreamingDashboard:
    def __init__(self, port: int = 8080)
    async def start()
    async def stop()
    def update_metrics_panels(live_metrics: LiveMetricsData)
    def update_histogram_panel(live_metrics: LiveMetricsData)
    def update_benchmark_status(**kwargs)
    def add_log_message(message: str, level: str = "INFO")
```

### WebSocket API

```javascript
// Event structure
{
    "event_type": "metrics|status|log|histogram|scatter|progress|heartbeat",
    "timestamp": 1234567890.123,
    "data": { /* event-specific data */ }
}
```

### REST API

```bash
# Get current status
curl http://localhost:8080/api/status

# Get historical data
curl http://localhost:8080/api/historical-data

# Get current metrics
curl http://localhost:8080/api/metrics
```

## Contributing

To extend the streaming functionality:

1. **Add new event types** in `StreamingDashboard`
2. **Extend the WebSocket API** in `StreamingServer`
3. **Update the frontend** to handle new events
4. **Add tests** for new functionality

See the existing code in `genai_bench/streaming/` for examples of the current implementation.
