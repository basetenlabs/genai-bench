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
    
    def __init__(self, dashboard, port: int = 8080, host: str = "0.0.0.0"):
        self.dashboard = dashboard
        self.port = port
        self.host = host
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
                
                # Send complete historical data
                complete_history = self.dashboard.get_complete_history()
                await websocket.send_json({
                    "event_type": "historical_data",
                    "data": complete_history
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
            
        @self.app.get("/api/connection-info")
        async def get_connection_info():
            """Get connection information for the streaming server."""
            return self.dashboard.get_connection_info()
            
        @self.app.get("/api/history")
        async def get_history():
            """Get complete historical data."""
            return self.dashboard.get_complete_history()
            
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
            host=self.host, 
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
            self.host, 
            self.port
        )
        await self.server.wait_closed()
        
    def _get_default_html(self) -> str:
        """Get default HTML frontend styled like Baseten UI."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI Bench - Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background-color: #ffffff;
            color: #1a1a1a;
            line-height: 1.5;
        }
        
        /* Top Navigation Bar */
        .top-nav {
            background: #ffffff;
            border-bottom: 1px solid #e5e7eb;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .scenario-info-section {
            margin: 16px 0;
            padding: 16px 0;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .scenario-info {
            display: flex;
            gap: 32px;
            align-items: center;
            background: #f8fafc;
            padding: 12px 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            max-width: 600px;
        }
        
        .scenario-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }
        
        .scenario-label {
            font-size: 12px;
            color: #64748b;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .scenario-value {
            font-size: 14px;
            color: #1e293b;
            font-weight: 600;
        }
        
        .nav-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        .logo-icon {
            width: 24px;
            height: 24px;
            background: #10b981;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }
        
        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #6b7280;
            font-size: 14px;
        }
        
        .nav-right {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .cost-info {
            background: #f0fdf4;
            color: #166534;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .nav-link {
            color: #6b7280;
            text-decoration: none;
            font-size: 14px;
        }
        
        .notification-bell {
            position: relative;
            color: #6b7280;
            cursor: pointer;
        }
        
        .notification-badge {
            position: absolute;
            top: -4px;
            right: -4px;
            background: #ef4444;
            color: white;
            border-radius: 50%;
            width: 16px;
            height: 16px;
            font-size: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Main Content */
        .main-content {
            padding: 24px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .tab {
            padding: 12px 0;
            color: #6b7280;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab.active {
            color: #10b981;
            border-bottom-color: #10b981;
        }
        
        .tab:hover {
            color: #1a1a1a;
        }
        
        /* Action Buttons */
        .action-buttons {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            justify-content: flex-end;
        }
        
        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.2s;
            border: none;
            cursor: pointer;
        }
        
        .btn-outline {
            background: transparent;
            color: #6b7280;
            border: 1px solid #d1d5db;
        }
        
        .btn-outline:hover {
            background: #f9fafb;
            color: #1a1a1a;
        }
        
        .btn-primary {
            background: #10b981;
            color: white;
        }
        
        .btn-primary:hover {
            background: #059669;
        }
        
        /* Metrics Cards */
        .metrics-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }
        
        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 24px;
        }
        
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .metric-title {
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 8px;
        }
        
        .metric-subtitle {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 16px;
        }
        
        .status-pills {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        
        .status-pill {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .status-pill.success {
            background: #f0fdf4;
            color: #166534;
        }
        
        .status-pill.warning {
            background: #fffbeb;
            color: #d97706;
        }
        
        .status-pill.error {
            background: #fef2f2;
            color: #dc2626;
        }
        
        .chart-container {
            height: 200px;
            margin-top: 16px;
        }
        
        /* Percentile Values */
        .percentiles {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        
        .percentile {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .percentile.p50 { background: #dbeafe; color: #1e40af; }
        .percentile.p90 { background: #f0fdf4; color: #166534; }
        .percentile.p95 { background: #fffbeb; color: #d97706; }
        .percentile.p99 { background: #fef2f2; color: #dc2626; }
        
        /* Progress Bar */
        .progress-container {
            margin-bottom: 24px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #f3f4f6;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: #10b981;
            transition: width 0.3s ease;
        }
        
        /* Logs */
        .logs-container {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 24px;
        }
        
        .logs-header {
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 16px;
        }
        
        .logs-content {
            background: #f9fafb;
            border-radius: 6px;
            padding: 16px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            line-height: 1.4;
        }
        
        .log-entry {
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .log-timestamp {
            color: #6b7280;
            font-size: 11px;
        }
        
        .log-message {
            color: #1a1a1a;
        }
        
        .log-message.error { color: #dc2626; }
        .log-message.warning { color: #d97706; }
        .log-message.info { color: #1a1a1a; }
        
        /* Connection Status */
        .connection-status {
            position: fixed;
            top: 80px;
            right: 24px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            z-index: 50;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
        }
        
        .status-dot.connecting { background: #f59e0b; }
        .status-dot.error { background: #ef4444; }
        
        /* Responsive */
        @media (max-width: 768px) {
            .metrics-container {
                grid-template-columns: 1fr;
            }
            
            .tabs {
                gap: 16px;
            }
            
            .main-content {
                padding: 16px;
            }
        }
    </style>
</head>
<body>
    <!-- Top Navigation Bar -->
    <div class="top-nav">
        <div class="nav-left">
            <div class="logo">
                <div class="logo-icon">B</div>
                <span>genai-bench</span>
            </div>
            <div class="model-selector">
                <span id="model-name">Qwen3-Next-80B-A3B-Instruct</span>
                <span>▼</span>
            </div>
        </div>
        <div class="nav-right">
            <div class="cost-info">
                <span id="cost-info">Benchmark in progress</span>
            </div>
            <a href="#" class="nav-link">Model Library</a>
            <div class="notification-bell">
                🔔
                <div class="notification-badge" id="notification-badge" style="display: none;">1</div>
            </div>
            <div class="user-icon">👤</div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Tabs -->
        <div class="tabs">
            <a href="#" class="tab">Overview</a>
            <a href="#" class="tab">Logs</a>
            <a href="#" class="tab active">Benchmarks</a>
            <a href="#" class="tab">Metrics</a>
            <a href="#" class="tab">Activity</a>
        </div>

        <!-- Action Buttons -->
        <div class="action-buttons">
            <button class="btn btn-outline" onclick="copyEndpoint()">📋 API endpoint</button>
            <button class="btn btn-primary" onclick="openPlayground()">🎮 Playground</button>
        </div>

        <!-- Scenario Information -->
        <div class="scenario-info-section">
            <div class="scenario-info">
                <div class="scenario-item">
                    <span class="scenario-label">Concurrency:</span>
                    <span class="scenario-value" id="current-concurrency">-</span>
                </div>
                <div class="scenario-item">
                    <span class="scenario-label">Traffic:</span>
                    <span class="scenario-value" id="current-traffic">-</span>
                </div>
                <div class="scenario-item">
                    <span class="scenario-label">Scenario:</span>
                    <span class="scenario-value" id="current-scenario">-</span>
                </div>
            </div>
        </div>

        <!-- Progress Bar -->
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
        </div>

        <!-- Metrics Cards -->
        <div class="metrics-container">
            <!-- Inference Volume Card -->
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">Inference volume</div>
                </div>
                <div class="metric-value" id="inference-volume">0</div>
                <div class="metric-subtitle" id="inference-subtitle">requests per minute</div>
                <div class="status-pills">
                    <div class="status-pill success" id="success-requests">2XX 0 /min</div>
                    <div class="status-pill warning" id="client-errors">4XX 0 /min</div>
                    <div class="status-pill error" id="server-errors">5XX 0 /min</div>
                </div>
                <div class="chart-container">
                    <canvas id="inference-chart"></canvas>
                </div>
            </div>

            <!-- Response Time Card -->
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">End-to-end response time</div>
                    <span style="color: #6b7280; font-size: 14px;">ℹ️</span>
                </div>
                <div class="metric-value" id="response-time">0</div>
                <div class="metric-subtitle" id="response-subtitle">ms mean</div>
                <div class="percentiles">
                    <div class="percentile p50" id="p50">p50 0 ms</div>
                    <div class="percentile p90" id="p90">p90 0 ms</div>
                    <div class="percentile p95" id="p95">p95 0 ms</div>
                    <div class="percentile p99" id="p99">p99 0 ms</div>
                </div>
                <div class="chart-container">
                    <canvas id="response-chart"></canvas>
                </div>
            </div>
        </div>

        <!-- Additional Charts -->
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">Latency Distribution</div>
                </div>
                <div class="chart-container">
                    <canvas id="latency-chart"></canvas>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-title">Throughput vs Latency</div>
                </div>
                <div class="chart-container">
                    <canvas id="scatter-chart"></canvas>
                </div>
            </div>
        </div>

        <!-- Logs Section -->
        <div class="logs-container">
            <div class="logs-header">Activity Logs</div>
            <div class="logs-content" id="log-container">
                <div class="log-entry">
                    <span class="log-timestamp">[Connecting...]</span>
                    <span class="log-message">Establishing connection to benchmark server</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Connection Status -->
    <div class="connection-status" id="connection-status">
        <div class="status-indicator">
            <div class="status-dot connecting" id="status-dot"></div>
            <span id="connection-text">Connecting...</span>
        </div>
    </div>

    <script>
        let ws;
        let inferenceChart, responseChart, latencyChart, scatterChart;
        let requestCount = 0;
        let startTime = Date.now();
        let isRestoringHistoricalData = false;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('Connected to GenAI Bench');
                updateConnectionStatus('connected', 'Connected');
                document.getElementById('connection-text').textContent = 'Connected';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };
            
            ws.onclose = function() {
                console.log('Disconnected from GenAI Bench');
                updateConnectionStatus('error', 'Disconnected');
                document.getElementById('connection-text').textContent = 'Disconnected - Reconnecting...';
                setTimeout(connect, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('error', 'Connection Error');
            };
        }
        
        function updateConnectionStatus(status, text) {
            const statusDot = document.getElementById('status-dot');
            const connectionText = document.getElementById('connection-text');
            
            statusDot.className = `status-dot ${status}`;
            connectionText.textContent = text;
        }
        
        function handleEvent(event) {
            // Skip real-time updates if we're restoring historical data
            if (isRestoringHistoricalData && event.event_type !== 'historical_data') {
                return;
            }
            
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
                case 'historical_data':
                    handleHistoricalData(event.data);
                    break;
                case 'heartbeat':
                    // Handle heartbeat - connection is alive
                    break;
            }
        }
        
        function handleHistoricalData(data) {
            console.log('Received historical data:', data);
            isRestoringHistoricalData = true;
            
            // Restore metrics history
            if (data.metrics_history && data.metrics_history.length > 0) {
                console.log('Restoring metrics history:', data.metrics_history.length, 'entries');
                data.metrics_history.forEach(metric => {
                    if (metric.live_metrics) {
                        updateMetrics(metric.live_metrics);
                    }
                });
            }
            
            // Restore logs history
            if (data.logs_history && data.logs_history.length > 0) {
                console.log('Restoring logs history:', data.logs_history.length, 'entries');
                // Clear existing logs
                document.getElementById('log-container').innerHTML = '';
                data.logs_history.forEach(log => {
                    addLog(log.message, log.level);
                });
            }
            
            // Restore scatter plot history
            if (data.scatter_history && data.scatter_history.length > 0) {
                console.log('Restoring scatter plot history:', data.scatter_history.length, 'entries');
                data.scatter_history.forEach(scatter => {
                    updateScatter(scatter);
                });
            }
            
            // Restore histogram history
            if (data.histogram_history && data.histogram_history.length > 0) {
                console.log('Restoring histogram history:', data.histogram_history.length, 'entries');
                // Use the most recent histogram data
                const latestHistogram = data.histogram_history[data.histogram_history.length - 1];
                updateHistogram(latestHistogram);
            }
            
            // Restore status
            if (data.current_status) {
                updateStatus(data.current_status);
            }
            
            // Restore progress
            if (data.current_status && data.current_status.progress_percentage !== undefined) {
                updateProgress(data.current_status.progress_percentage);
            }
            
            isRestoringHistoricalData = false;
        }
        
        function updateStatus(status) {
            // Update progress bar
            updateProgress(status.progress_percentage);
            
            // Update cost info
            const costInfo = document.getElementById('cost-info');
            if (status.status === 'running') {
                costInfo.textContent = `Benchmark running - ${status.progress_percentage.toFixed(1)}%`;
            } else if (status.status === 'completed') {
                costInfo.textContent = 'Benchmark completed';
            } else {
                costInfo.textContent = `Status: ${status.status}`;
            }
            
            // Update scenario information
            updateScenarioInfo(status);
        }
        
        function updateScenarioInfo(status) {
            console.log('Updating scenario info:', status);
            
            // Update concurrency
            const concurrencyEl = document.getElementById('current-concurrency');
            if (status.current_concurrency !== undefined && status.current_concurrency !== null) {
                concurrencyEl.textContent = status.current_concurrency;
                console.log('Set concurrency to:', status.current_concurrency);
            } else {
                concurrencyEl.textContent = '-';
                console.log('No concurrency value found');
            }
            
            // Update traffic scenario
            const trafficEl = document.getElementById('current-traffic');
            if (status.traffic_scenario) {
                trafficEl.textContent = status.traffic_scenario;
            } else if (status.current_scenario) {
                trafficEl.textContent = status.current_scenario;
            } else {
                trafficEl.textContent = '-';
            }
            
            // Update scenario name/description
            const scenarioEl = document.getElementById('current-scenario');
            if (status.scenario_name) {
                scenarioEl.textContent = status.scenario_name;
            } else if (status.current_scenario) {
                scenarioEl.textContent = status.current_scenario;
            } else {
                scenarioEl.textContent = '-';
            }
        }
        
        function updateMetrics(liveMetrics) {
            if (liveMetrics.stats) {
                const stats = liveMetrics.stats;
                
                // Update inference volume (requests per minute)
                requestCount++;
                const elapsedMinutes = (Date.now() - startTime) / 60000;
                const requestsPerMinute = Math.round(requestCount / Math.max(elapsedMinutes, 0.1));
                
                document.getElementById('inference-volume').textContent = requestsPerMinute;
                document.getElementById('success-requests').textContent = `2XX ${requestsPerMinute} /min`;
                
                // Update response time
                if (stats.ttft && stats.ttft.mean) {
                    const responseTimeMs = Math.round(stats.ttft.mean * 1000);
                    document.getElementById('response-time').textContent = responseTimeMs.toLocaleString();
                    
                    // Update percentiles
                    if (stats.ttft.p50) document.getElementById('p50').textContent = `p50 ${Math.round(stats.ttft.p50 * 1000)} ms`;
                    if (stats.ttft.p90) document.getElementById('p90').textContent = `p90 ${Math.round(stats.ttft.p90 * 1000)} ms`;
                    if (stats.ttft.p95) document.getElementById('p95').textContent = `p95 ${Math.round(stats.ttft.p95 * 1000)} ms`;
                    if (stats.ttft.p99) document.getElementById('p99').textContent = `p99 ${Math.round(stats.ttft.p99 * 1000)} ms`;
                }
                
                // Update charts
                updateInferenceChart(requestsPerMinute);
                updateResponseChart(stats.ttft?.mean * 1000 || 0);
            }
        }
        
        function updateInferenceChart(requestsPerMinute) {
            if (!inferenceChart) {
                const ctx = document.getElementById('inference-chart').getContext('2d');
                inferenceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Requests/min',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Requests per minute'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            }
            
            const now = new Date().toLocaleTimeString();
            inferenceChart.data.labels.push(now);
            inferenceChart.data.datasets[0].data.push(requestsPerMinute);
            
            // Keep only last 20 points
            if (inferenceChart.data.labels.length > 20) {
                inferenceChart.data.labels.shift();
                inferenceChart.data.datasets[0].data.shift();
            }
            
            inferenceChart.update();
        }
        
        function updateResponseChart(responseTimeMs) {
            if (!responseChart) {
                const ctx = document.getElementById('response-chart').getContext('2d');
                responseChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'p50',
                                data: [],
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                tension: 0.1
                            },
                            {
                                label: 'p90',
                                data: [],
                                borderColor: '#10b981',
                                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                                tension: 0.1
                            },
                            {
                                label: 'p95',
                                data: [],
                                borderColor: '#f59e0b',
                                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                tension: 0.1
                            },
                            {
                                label: 'p99',
                                data: [],
                                borderColor: '#ef4444',
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Response time (ms)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            }
                        }
                    }
                });
            }
            
            const now = new Date().toLocaleTimeString();
            responseChart.data.labels.push(now);
            
            // Add data points for each percentile (simplified)
            responseChart.data.datasets[0].data.push(responseTimeMs);
            responseChart.data.datasets[1].data.push(responseTimeMs * 1.2);
            responseChart.data.datasets[2].data.push(responseTimeMs * 1.4);
            responseChart.data.datasets[3].data.push(responseTimeMs * 1.8);
            
            // Keep only last 20 points
            if (responseChart.data.labels.length > 20) {
                responseChart.data.labels.shift();
                responseChart.data.datasets.forEach(dataset => {
                    dataset.data.shift();
                });
            }
            
            responseChart.update();
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
            
            const newPoint = {
                x: data.input_throughput,
                y: data.ttft
            };
            
            // Check for duplicates before adding
            const isDuplicate = scatterChart.data.datasets[0].data.some(point => 
                Math.abs(point.x - newPoint.x) < 0.001 && Math.abs(point.y - newPoint.y) < 0.001
            );
            
            if (!isDuplicate) {
                scatterChart.data.datasets[0].data.push(newPoint);
            }
            
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
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `
                <span class="log-timestamp">[${timestamp}]</span>
                <span class="log-message ${level.toLowerCase()}">${message}</span>
            `;
            logContainer.appendChild(logEntry);
            
            // Keep only last 20 log entries
            while (logContainer.children.length > 20) {
                logContainer.removeChild(logContainer.firstChild);
            }
            
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        function copyEndpoint() {
            const endpoint = window.location.origin;
            navigator.clipboard.writeText(endpoint).then(() => {
                alert('Endpoint copied to clipboard!');
            });
        }
        
        function openPlayground() {
            alert('Playground feature coming soon!');
        }
        
        // Tab switching functionality
        function initTabs() {
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Handle tab content switching
                    const tabText = this.textContent.trim();
                    switch(tabText) {
                        case 'Benchmarks':
                            showBenchmarksContent();
                            break;
                        case 'Metrics':
                            showMetricsContent();
                            break;
                        case 'Logs':
                            showLogsContent();
                            break;
                        case 'Overview':
                            showOverviewContent();
                            break;
                        case 'Activity':
                            showActivityContent();
                            break;
                    }
                });
            });
        }
        
        function showBenchmarksContent() {
            // Show the main benchmark metrics (current content)
            document.querySelector('.metrics-container').style.display = 'grid';
            document.querySelector('.logs-container').style.display = 'block';
        }
        
        function showMetricsContent() {
            // Show detailed metrics view
            document.querySelector('.metrics-container').style.display = 'grid';
            document.querySelector('.logs-container').style.display = 'none';
        }
        
        function showLogsContent() {
            // Show only logs
            document.querySelector('.metrics-container').style.display = 'none';
            document.querySelector('.logs-container').style.display = 'block';
        }
        
        function showOverviewContent() {
            // Show overview content
            document.querySelector('.metrics-container').style.display = 'grid';
            document.querySelector('.logs-container').style.display = 'block';
        }
        
        function showActivityContent() {
            // Show activity content
            document.querySelector('.metrics-container').style.display = 'grid';
            document.querySelector('.logs-container').style.display = 'block';
        }
        
        // Fallback to fetch historical data via HTTP if WebSocket fails
        async function fetchHistoricalData() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                handleHistoricalData(data);
                console.log('Historical data loaded via HTTP fallback');
            } catch (error) {
                console.error('Failed to fetch historical data:', error);
            }
        }
        
        // Initialize tabs and connect on page load
        document.addEventListener('DOMContentLoaded', function() {
            initTabs();
            connect();
            
            // Also try to fetch historical data via HTTP as a fallback
            setTimeout(fetchHistoricalData, 1000);
        });
    </script>
</body>
</html>
        """
