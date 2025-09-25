"""Streaming server for real-time benchmark data."""

import asyncio
import json
import uuid
from typing import Dict, Set, Optional
from pathlib import Path

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    websockets = None
    WebSocketServerProtocol = None

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from genai_bench.logging import init_logger

logger = init_logger(__name__)


class StreamingServer:
    """
    WebSocket server for streaming real-time benchmark data to frontend clients.
    
    Supports both WebSocket connections for real-time data and HTTP endpoints
    for historical data and static file serving.
    """
    
    def __init__(self, dashboard, port: int = 8080):
        self.dashboard = dashboard
        self.port = port
        self.websocket_connections: Set[WebSocketServerProtocol] = set()
        self.app = None
        self.server = None
        
    async def start(self):
        """Start the streaming server."""
        if FASTAPI_AVAILABLE:
            await self._start_fastapi_server()
        elif websockets:
            await self._start_websockets_server()
        else:
            logger.warning("No web framework available. Install fastapi or websockets.")
            
    async def stop(self):
        """Stop the streaming server."""
        if self.server:
            try:
                if hasattr(self.server, 'close'):
                    self.server.close()
                if hasattr(self.server, 'wait_closed'):
                    await self.server.wait_closed()
                elif hasattr(self.server, 'shutdown'):
                    await self.server.shutdown()
            except Exception as e:
                # Ignore shutdown errors
                pass
            
    async def _start_fastapi_server(self):
        """Start FastAPI server with WebSocket support."""
        self.app = FastAPI(title="GenAI Bench Streaming Server")
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            client_id = str(uuid.uuid4())
            self.dashboard.connected_clients.add(client_id)
            
            try:
                # Send initial status
                await websocket.send_json({
                    "event_type": "status",
                    "data": self.dashboard.benchmark_status.__dict__
                })
                
                # Send historical data
                await websocket.send_json({
                    "event_type": "historical_data",
                    "data": {"historical_data": self.dashboard.get_historical_data()}
                })
                
                # Process events from dashboard
                while True:
                    try:
                        # Wait for events from dashboard
                        event = await asyncio.wait_for(
                            self.dashboard.event_queue.get(), 
                            timeout=1.0
                        )
                        
                        # Debug log
                        logger.debug(f"Processing event: {event.event_type}")
                        
                        # Send event to client
                        await websocket.send_json({
                            "event_type": event.event_type,
                            "timestamp": event.timestamp,
                            "data": event.data
                        })
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({
                            "event_type": "heartbeat",
                            "timestamp": asyncio.get_event_loop().time(),
                            "data": {}
                        })
                        
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.dashboard.connected_clients.discard(client_id)
                
        # REST API endpoints
        @self.app.get("/api/status")
        async def get_status():
            return self.dashboard.benchmark_status.__dict__
            
        @self.app.get("/api/historical-data")
        async def get_historical_data():
            return {"historical_data": self.dashboard.get_historical_data()}
            
        @self.app.get("/api/metrics")
        async def get_current_metrics():
            # Get current metrics from the metrics collector if available
            if hasattr(self.dashboard, 'metrics_collector'):
                return self.dashboard.metrics_collector.get_live_metrics()
            return {}
            
        # Serve static files (frontend)
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
        else:
            # Fallback HTML
            @self.app.get("/", response_class=HTMLResponse)
            async def get_frontend():
                return self._get_default_html()
                
        # Start server
        config = uvicorn.Config(
            self.app, 
            host="0.0.0.0", 
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()
        
    async def _start_websockets_server(self):
        """Start pure WebSocket server."""
        async def websocket_handler(websocket, path):
            client_id = str(uuid.uuid4())
            self.websocket_connections.add(websocket)
            
            try:
                # Send initial status
                await websocket.send(json.dumps({
                    "event_type": "status",
                    "data": self.dashboard.benchmark_status.__dict__
                }))
                
                # Process events
                while True:
                    try:
                        event = await asyncio.wait_for(
                            self.dashboard.event_queue.get(),
                            timeout=1.0
                        )
                        
                        await websocket.send(json.dumps({
                            "event_type": event.event_type,
                            "timestamp": event.timestamp,
                            "data": event.data
                        }))
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send(json.dumps({
                            "event_type": "heartbeat",
                            "timestamp": asyncio.get_event_loop().time(),
                            "data": {}
                        }))
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client {client_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.discard(websocket)
                
        self.server = await websockets.serve(
            websocket_handler, 
            "0.0.0.0", 
            self.port
        )
        await self.server.wait_closed()
        
    def _get_default_html(self) -> str:
        """Get default HTML frontend."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI Bench - Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .status {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .chart-container {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .logs {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background-color: #007bff;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GenAI Bench - Live Dashboard</h1>
            <p>Real-time benchmark monitoring</p>
        </div>
        
        <div class="status">
            <h3>Benchmark Status</h3>
            <div id="status-info">Connecting...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Input Latency (TTFT)</h4>
                <div id="ttft-metrics">No data</div>
            </div>
            <div class="metric-card">
                <h4>Output Latency</h4>
                <div id="output-latency-metrics">No data</div>
            </div>
            <div class="metric-card">
                <h4>Input Throughput</h4>
                <div id="input-throughput-metrics">No data</div>
            </div>
            <div class="metric-card">
                <h4>Output Throughput</h4>
                <div id="output-throughput-metrics">No data</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Latency Distribution</h3>
            <canvas id="latency-chart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Throughput vs Latency</h3>
            <canvas id="scatter-chart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>RPS vs E2E Latency</h3>
            <canvas id="rps-latency-chart" width="400" height="200"></canvas>
        </div>
        
        <div class="logs">
            <h3>Logs</h3>
            <div id="log-container"></div>
        </div>
    </div>

    <script>
        let ws;
        let latencyChart, scatterChart, rpsLatencyChart;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('Connected to GenAI Bench');
                document.getElementById('status-info').innerHTML = 'Connected';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };
            
            ws.onclose = function() {
                console.log('Disconnected from GenAI Bench');
                document.getElementById('status-info').innerHTML = 'Disconnected - Reconnecting...';
                setTimeout(connect, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleEvent(event) {
            switch(event.event_type) {
                case 'status':
                    updateStatus(event.data);
                    break;
                case 'metrics':
                    updateMetrics(event.data.live_metrics);
                    break;
                case 'histogram':
                    updateHistogram(event.data);
                    break;
                case 'scatter':
                    updateScatter(event.data);
                    break;
                case 'rps_vs_latency':
                    updateRpsVsLatency(event.data);
                    break;
                case 'log':
                    addLog(event.data.message, event.data.level);
                    break;
                case 'progress':
                    updateProgress(event.data.progress);
                    break;
            }
        }
        
        function updateStatus(status) {
            const statusHtml = `
                <strong>Status:</strong> ${status.status}<br>
                <strong>Scenario:</strong> ${status.current_scenario}<br>
                <strong>Progress:</strong> ${status.progress_percentage.toFixed(1)}%<br>
                <strong>Total Scenarios:</strong> ${status.total_scenarios}<br>
                <strong>Total Iterations:</strong> ${status.total_iterations}
            `;
            document.getElementById('status-info').innerHTML = statusHtml;
            updateProgress(status.progress_percentage);
        }
        
        function updateMetrics(liveMetrics) {
            if (liveMetrics.stats) {
                const stats = liveMetrics.stats;
                
                if (stats.ttft) {
                    document.getElementById('ttft-metrics').innerHTML = `
                        Avg: ${stats.ttft.mean?.toFixed(3)}s<br>
                        P50: ${stats.ttft.p50?.toFixed(3)}s<br>
                        P99: ${stats.ttft.p99?.toFixed(3)}s
                    `;
                }
                
                if (stats.output_latency) {
                    document.getElementById('output-latency-metrics').innerHTML = `
                        Avg: ${stats.output_latency.mean?.toFixed(3)}s<br>
                        P50: ${stats.output_latency.p50?.toFixed(3)}s<br>
                        P99: ${stats.output_latency.p99?.toFixed(3)}s
                    `;
                }
                
                if (stats.input_throughput) {
                    document.getElementById('input-throughput-metrics').innerHTML = `
                        Avg: ${stats.input_throughput.mean?.toFixed(1)} tokens/s<br>
                        Min: ${stats.input_throughput.min?.toFixed(1)} tokens/s<br>
                        Max: ${stats.input_throughput.max?.toFixed(1)} tokens/s
                    `;
                }
                
                if (stats.output_throughput) {
                    document.getElementById('output-throughput-metrics').innerHTML = `
                        Avg: ${stats.output_throughput.mean?.toFixed(1)} tokens/s<br>
                        Min: ${stats.output_throughput.min?.toFixed(1)} tokens/s<br>
                        Max: ${stats.output_throughput.max?.toFixed(1)} tokens/s
                    `;
                }
            }
        }
        
        function updateHistogram(data) {
            if (!latencyChart) {
                const ctx = document.getElementById('latency-chart').getContext('2d');
                latencyChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'TTFT',
                            data: [],
                            backgroundColor: 'rgba(54, 162, 235, 0.5)'
                        }, {
                            label: 'Output Latency',
                            data: [],
                            backgroundColor: 'rgba(255, 99, 132, 0.5)'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
            
            if (data.ttft_histogram && data.ttft_histogram.bins) {
                latencyChart.data.labels = data.ttft_histogram.bins.slice(0, -1).map(bin => bin.toFixed(3));
                latencyChart.data.datasets[0].data = data.ttft_histogram.counts;
            }
            
            if (data.output_latency_histogram && data.output_latency_histogram.bins) {
                latencyChart.data.datasets[1].data = data.output_latency_histogram.counts;
            }
            
            latencyChart.update();
        }
        
        function updateScatter(data) {
            if (!scatterChart) {
                const ctx = document.getElementById('scatter-chart').getContext('2d');
                scatterChart = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Throughput vs Latency',
                            data: [],
                            backgroundColor: 'rgba(75, 192, 192, 0.5)'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Throughput (tokens/s)'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Latency (s)'
                                }
                            }
                        }
                    }
                });
            }
            
            scatterChart.data.datasets[0].data.push({
                x: data.input_throughput,
                y: data.ttft
            });
            
            // Keep only last 100 points
            if (scatterChart.data.datasets[0].data.length > 100) {
                scatterChart.data.datasets[0].data.shift();
            }
            
            scatterChart.update();
        }
        
        function updateRpsVsLatency(data) {
            if (!rpsLatencyChart) {
                const ctx = document.getElementById('rps-latency-chart').getContext('2d');
                rpsLatencyChart = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'RPS vs E2E Latency',
                            data: [],
                            backgroundColor: 'rgba(255, 159, 64, 0.5)'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Requests per Second (RPS)'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'E2E Latency (s)'
                                }
                            }
                        }
                    }
                });
            }
            
            rpsLatencyChart.data.datasets[0].data.push({
                x: data.rps,
                y: data.e2e_latency
            });
            
            // Keep only last 100 points
            if (rpsLatencyChart.data.datasets[0].data.length > 100) {
                rpsLatencyChart.data.datasets[0].data.shift();
            }
            
            rpsLatencyChart.update();
        }
        
        function updateProgress(progress) {
            document.getElementById('progress-fill').style.width = progress + '%';
        }
        
        function addLog(message, level) {
            const logContainer = document.getElementById('log-container');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `<span style="color: #666;">[${timestamp}]</span> <span style="color: ${level === 'ERROR' ? 'red' : level === 'WARNING' ? 'orange' : 'black'};">${message}</span>`;
            logContainer.appendChild(logEntry);
            
            // Keep only last 50 log entries
            while (logContainer.children.length > 50) {
                logContainer.removeChild(logContainer.firstChild);
            }
            
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        // Connect on page load
        connect();
    </script>
</body>
</html>
        """
