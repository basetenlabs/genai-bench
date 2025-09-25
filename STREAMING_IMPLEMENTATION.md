# GenAI Bench Streaming Implementation

This document provides a comprehensive overview of the streaming implementation for GenAI Bench, explaining how to implement real-time benchmark monitoring with web frontend integration.

## Overview

The streaming implementation allows you to:
1. **Kick off a benchmarking job** that runs the benchmark
2. **Connect a frontend** to the running benchmark
3. **Show the benchmark happening in real-time** with live metrics
4. **Display interactive data** when the benchmark is complete

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Benchmark     │    │  Streaming       │    │   Web Frontend  │
│   Execution     │───▶│  Dashboard       │───▶│   (Browser)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  WebSocket       │
                       │  Server          │
                       └──────────────────┘
```

### Data Flow

1. **Benchmark Execution**: The existing benchmark runs and collects metrics
2. **Metrics Collection**: `AggregatedMetricsCollector` processes individual request metrics
3. **Streaming Dashboard**: `StreamingDashboard` receives metrics and queues events
4. **WebSocket Server**: `StreamingServer` broadcasts events to connected clients
5. **Frontend**: Web clients receive and visualize the data in real-time

## Implementation Details

### 1. Streaming Dashboard (`genai_bench/streaming/streaming_dashboard.py`)

The `StreamingDashboard` class extends the existing dashboard functionality:

```python
class StreamingDashboard:
    def __init__(self, port: int = 8080):
        self.port = port
        self.connected_clients: Set[str] = set()
        self.benchmark_status = BenchmarkStatus(...)
        self.historical_data: List[Dict[str, Any]] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
```

**Key Features:**
- **Compatibility**: Implements the same interface as the existing `RichLiveDashboard`
- **Event Queue**: Asynchronous event system for real-time updates
- **Historical Storage**: Stores completed benchmark data
- **Status Tracking**: Maintains benchmark progress and status

### 2. Streaming Server (`genai_bench/streaming/streaming_server.py`)

The `StreamingServer` handles web connections:

```python
class StreamingServer:
    def __init__(self, dashboard, port: int = 8080):
        self.dashboard = dashboard
        self.port = port
        self.websocket_connections: Set[WebSocketServerProtocol] = set()
```

**Features:**
- **WebSocket Support**: Real-time bidirectional communication
- **REST API**: HTTP endpoints for status and historical data
- **Built-in Frontend**: Default HTML dashboard with Chart.js
- **Fallback Support**: Works with or without FastAPI

### 3. CLI Integration (`genai_bench/cli/cli.py`)

Added streaming options to the existing benchmark command:

```python
@click.option(
    "--enable-streaming",
    is_flag=True,
    default=False,
    help="Enable real-time streaming to web dashboard",
)
@click.option(
    "--streaming-port",
    default=8080,
    help="Port for the streaming dashboard server",
    type=int,
)
```

## Usage Examples

### 1. Basic Usage with CLI

```bash
# Run benchmark with streaming enabled
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

Then open `http://localhost:8080` in your browser.

### 2. Programmatic Usage

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
    
    # Your benchmark logic here...
    # The dashboard will automatically receive metrics from the existing system
    
    # Keep running
    await asyncio.sleep(60)
    
    # Stop the server
    server_task.cancel()
    await dashboard.stop()

# Run the example
asyncio.run(run_benchmark_with_streaming())
```

### 3. Custom Frontend Integration

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

## Event System

The streaming system uses an event-driven architecture:

### Event Types

- **`metrics`**: Live metrics data with statistics
- **`histogram`**: Histogram data for latency distributions  
- **`scatter`**: Scatter plot data for throughput vs latency
- **`status`**: Benchmark status and progress
- **`log`**: Log messages
- **`progress`**: Progress updates
- **`heartbeat`**: Connection keep-alive

### Event Structure

```javascript
{
    "event_type": "metrics|status|log|histogram|scatter|progress|heartbeat",
    "timestamp": 1234567890.123,
    "data": { /* event-specific data */ }
}
```

## API Endpoints

### WebSocket API

- **Endpoint**: `ws://localhost:8080/ws`
- **Protocol**: WebSocket
- **Events**: Real-time streaming of all benchmark events

### REST API

- **`GET /api/status`**: Current benchmark status
- **`GET /api/historical-data`**: Historical benchmark data
- **`GET /api/metrics`**: Current metrics snapshot

## Integration Points

### 1. Existing Dashboard Interface

The streaming dashboard implements the same interface as the existing `RichLiveDashboard`:

```python
def update_metrics_panels(self, live_metrics: LiveMetricsData):
def update_histogram_panel(self, live_metrics: LiveMetricsData):
def update_scatter_plot_panel(self, ui_scatter_plot_metrics: Optional[List[float]]):
def update_benchmark_progress_bars(self, progress_increment: float):
# ... etc
```

### 2. Metrics Collection

Integrates with the existing `AggregatedMetricsCollector`:

```python
# In the distributed runner
self.dashboard.handle_single_request(
    self.metrics_collector.get_live_metrics(),
    environment.runner.stats.total.num_requests,
    metrics.error_code,
)
```

### 3. CLI Integration

Seamlessly integrates with existing CLI commands by adding streaming options.

## Dependencies

### Required (Optional)

```bash
pip install fastapi uvicorn websockets
```

### Development

```bash
pip install -r requirements-streaming.txt
```

## Testing

Run the streaming tests:

```bash
pytest tests/streaming/test_streaming_dashboard.py
```

## Example Workflow

### Complete Workflow

1. **Start Benchmark with Streaming**:
   ```bash
   genai-bench benchmark --enable-streaming --streaming-port 8080 [other options]
   ```

2. **Open Dashboard**:
   - Navigate to `http://localhost:8080`
   - See real-time metrics, progress, and visualizations

3. **Monitor Progress**:
   - Watch live updates as the benchmark runs
   - View histograms and scatter plots
   - Monitor logs in real-time

4. **View Results**:
   - Historical data is automatically stored
   - Interactive charts show completed benchmark data
   - Export capabilities for further analysis

### Custom Integration

1. **Create Custom Frontend**:
   - Connect to WebSocket at `ws://localhost:8080/ws`
   - Handle different event types
   - Build custom visualizations

2. **Programmatic Access**:
   - Use `StreamingDashboard` class directly
   - Integrate with existing applications
   - Build custom monitoring systems

## Benefits

### For Users

- **Real-time Monitoring**: See benchmarks as they happen
- **Interactive Visualizations**: Rich charts and graphs
- **Historical Data**: Access to completed benchmark results
- **Custom Frontends**: Build your own dashboards
- **Remote Access**: Monitor benchmarks from anywhere

### For Developers

- **Easy Integration**: Drop-in replacement for existing dashboard
- **Extensible**: Add new event types and visualizations
- **Well-tested**: Comprehensive test coverage
- **Documented**: Clear API and usage examples
- **Backward Compatible**: Doesn't break existing functionality

## Future Enhancements

### Potential Improvements

1. **Authentication**: Add user authentication for secure access
2. **Multi-user Support**: Allow multiple users to view the same benchmark
3. **Persistent Storage**: Store historical data in databases
4. **Alerting**: Add notification systems for benchmark completion/failures
5. **Mobile Support**: Responsive design for mobile devices
6. **Plugin System**: Allow custom visualization plugins

### Scalability

- **Load Balancing**: Support for multiple streaming servers
- **Redis Integration**: Use Redis for event broadcasting across multiple servers
- **Database Storage**: Store metrics in time-series databases
- **CDN Integration**: Serve static assets from CDNs

## Conclusion

The streaming implementation provides a complete solution for real-time benchmark monitoring with web frontend integration. It's designed to be:

- **Easy to use**: Simple CLI flags enable streaming
- **Flexible**: Support for custom frontends and integrations
- **Scalable**: Can handle multiple clients and large datasets
- **Maintainable**: Clean architecture with comprehensive testing

The implementation seamlessly integrates with the existing GenAI Bench architecture while providing powerful new capabilities for real-time monitoring and visualization.
