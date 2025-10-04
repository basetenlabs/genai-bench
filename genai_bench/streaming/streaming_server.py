"""Streaming server for real-time benchmark data."""

import asyncio
import json
import time
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
    
    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle messages from WebSocket clients."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "update_parameters":
                # Handle parameter updates
                parameters = data.get("parameters", {})
                await self._handle_parameter_update(websocket, parameters)
            elif message_type == "get_parameters":
                # Send current parameters
                await self._send_current_parameters(websocket)
            elif message_type == "start_benchmark":
                # Handle benchmark start request
                config = data.get("config", {})
                await self._handle_benchmark_start(websocket, config)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _handle_parameter_update(self, websocket: WebSocket, parameters: dict):
        """Handle parameter update requests."""
        try:
            # Update dashboard with new parameters
            if hasattr(self.dashboard, 'update_parameters'):
                self.dashboard.update_parameters(parameters)
            
            # Broadcast parameter update to all clients
            await self._broadcast_to_all_clients({
                "event_type": "parameters_updated",
                "data": parameters
            })
            
            # Send confirmation back to requesting client
            await websocket.send_json({
                "event_type": "parameter_update_confirmed",
                "data": {"status": "success", "parameters": parameters}
            })
            
            logger.info(f"Parameters updated: {parameters}")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            await websocket.send_json({
                "event_type": "parameter_update_error",
                "data": {"error": str(e)}
            })
    
    async def _send_current_parameters(self, websocket: WebSocket):
        """Send current parameters to client."""
        try:
            # Get current parameters from dashboard
            current_params = getattr(self.dashboard, 'current_parameters', {})
            
            await websocket.send_json({
                "event_type": "current_parameters",
                "data": current_params
            })
            
        except Exception as e:
            logger.error(f"Error sending current parameters: {e}")
    
    async def _handle_benchmark_start(self, websocket: WebSocket, config: dict):
        """Handle benchmark start requests."""
        try:
            # This would integrate with the actual benchmark system
            # For now, just acknowledge the request
            await websocket.send_json({
                "event_type": "benchmark_start_requested",
                "data": {"config": config, "status": "received"}
            })
            
            logger.info(f"Benchmark start requested with config: {config}")
            
        except Exception as e:
            logger.error(f"Error handling benchmark start: {e}")
    
    async def _broadcast_to_all_clients(self, message: dict):
        """Broadcast message to all connected clients."""
        # This would need to be implemented to track all WebSocket connections
        # For now, this is a placeholder
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
            logger.info(f"Client {client_id} connected. Total clients: {len(self.dashboard.connected_clients)}")
            
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
                
                # Process events from dashboard and handle client messages
                last_heartbeat = time.time()
                heartbeat_interval = 30  # Send heartbeat every 30 seconds instead of every second
                
                while True:
                    try:
                        # Wait for either dashboard events or client messages with longer timeout
                        dashboard_task = asyncio.create_task(
                            asyncio.wait_for(self.dashboard.event_queue.get(), timeout=5.0)
                        )
                        client_task = asyncio.create_task(
                            asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                        )
                        
                        done, pending = await asyncio.wait(
                            [dashboard_task, client_task],
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=5.0
                        )
                        
                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                        
                        # Handle dashboard events
                        if dashboard_task in done:
                            try:
                                event = dashboard_task.result()
                                logger.debug(f"Processing event: {event.event_type}")
                                
                                # Send event to client
                                await websocket.send_json({
                                    "event_type": event.event_type,
                                    "timestamp": event.timestamp,
                                    "data": event.data
                                })
                            except Exception as e:
                                logger.error(f"Error processing dashboard event: {e}")
                        
                        # Handle client messages
                        if client_task in done:
                            try:
                                message = client_task.result()
                                await self._handle_client_message(websocket, message)
                            except Exception as e:
                                logger.error(f"Error handling client message: {e}")
                        
                        # Send heartbeat only if no events for a while
                        current_time = time.time()
                        if current_time - last_heartbeat > heartbeat_interval:
                            try:
                                await websocket.send_json({
                                    "event_type": "heartbeat",
                                    "timestamp": current_time,
                                    "data": {}
                                })
                                last_heartbeat = current_time
                            except Exception as e:
                                logger.debug(f"Heartbeat failed: {e}")
                                break
                        
                    except asyncio.TimeoutError:
                        # Only send heartbeat if we haven't sent one recently
                        current_time = time.time()
                        if current_time - last_heartbeat > heartbeat_interval:
                            try:
                                await websocket.send_json({
                                    "event_type": "heartbeat",
                                    "timestamp": current_time,
                                    "data": {}
                                })
                                last_heartbeat = current_time
                            except Exception as e:
                                logger.debug(f"Heartbeat failed: {e}")
                                break
                        
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected cleanly")
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                self.dashboard.connected_clients.discard(client_id)
                logger.info(f"Client {client_id} removed. Total clients: {len(self.dashboard.connected_clients)}")
                
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
        
        /* Modal Styles */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: white;
            border-radius: 8px;
            width: 90%;
            max-width: 800px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 24px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .modal-header h2 {
            margin: 0;
            font-size: 20px;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #6b7280;
            padding: 0;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-close:hover {
            color: #1a1a1a;
        }
        
        .modal-body {
            padding: 24px;
        }
        
        .modal-footer {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            padding: 20px 24px;
            border-top: 1px solid #e5e7eb;
        }
        
        /* Parameter Form Styles */
        .parameter-section {
            margin-bottom: 32px;
        }
        
        .parameter-section h3 {
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .form-group {
            margin-bottom: 16px;
        }
        
        .form-group label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            color: #374151;
            margin-bottom: 6px;
        }
        
        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.2s;
        }
        
        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #10b981;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        }
        
        .form-group textarea {
            resize: vertical;
            min-height: 80px;
        }
        
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
            
            .modal-content {
                width: 95%;
                margin: 20px;
            }
            
            .modal-body {
                padding: 16px;
            }
            
            .modal-footer {
                padding: 16px;
                flex-direction: column;
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
            <button class="btn btn-outline" onclick="toggleParameterEditor()">⚙️ Edit Parameters</button>
            <button class="btn btn-primary" onclick="startBenchmark()">🚀 Start Benchmark</button>
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

        <!-- Parameter Editor Modal -->
        <div id="parameter-editor-modal" class="modal" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Edit Benchmark Parameters</h2>
                    <button class="modal-close" onclick="closeParameterEditor()">&times;</button>
                </div>
                <div class="modal-body">
                    <form id="parameter-form">
                        <!-- API Configuration -->
                        <div class="parameter-section">
                            <h3>API Configuration</h3>
                            <div class="form-group">
                                <label for="api_backend">API Backend:</label>
                                <select id="api_backend" name="api_backend">
                                    <option value="baseten">Baseten</option>
                                    <option value="openai">OpenAI</option>
                                    <option value="anthropic">Anthropic</option>
                                    <option value="azure">Azure</option>
                                    <option value="gcp">Google Cloud</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="api_base">API Base URL:</label>
                                <input type="text" id="api_base" name="api_base" placeholder="https://api.example.com/v1">
                            </div>
                            <div class="form-group">
                                <label for="api_key">API Key:</label>
                                <input type="password" id="api_key" name="api_key" placeholder="Your API key">
                            </div>
                            <div class="form-group">
                                <label for="api_model_name">Model Name:</label>
                                <input type="text" id="api_model_name" name="api_model_name" placeholder="gpt-4">
                            </div>
                            <div class="form-group">
                                <label for="task">Task Type:</label>
                                <select id="task" name="task">
                                    <option value="text-to-text">Text to Text</option>
                                    <option value="text-to-image">Text to Image</option>
                                    <option value="image-to-text">Image to Text</option>
                                    <option value="multimodal">Multimodal</option>
                                </select>
                            </div>
                        </div>

                        <!-- Load Testing Parameters -->
                        <div class="parameter-section">
                            <h3>Load Testing Parameters</h3>
                            <div class="form-group">
                                <label for="max_requests_per_run">Max Requests per Run:</label>
                                <input type="number" id="max_requests_per_run" name="max_requests_per_run" min="1" max="10000" value="64">
                            </div>
                            <div class="form-group">
                                <label for="max_time_per_run">Max Time per Run (seconds):</label>
                                <input type="number" id="max_time_per_run" name="max_time_per_run" min="1" max="3600" value="600">
                            </div>
                            <div class="form-group">
                                <label for="num_concurrency">Concurrency Levels (comma-separated):</label>
                                <input type="text" id="num_concurrency" name="num_concurrency" placeholder="2,4,8,16,24,32" value="2,4,8,16,24,32">
                            </div>
                            <div class="form-group">
                                <label for="traffic_scenario">Traffic Scenario:</label>
                                <input type="text" id="traffic_scenario" name="traffic_scenario" placeholder="D(2000,500)" value="D(2000,500)">
                            </div>
                        </div>

                        <!-- Advanced Parameters -->
                        <div class="parameter-section">
                            <h3>Advanced Parameters</h3>
                            <div class="form-group">
                                <label for="batch_size">Batch Size:</label>
                                <input type="text" id="batch_size" name="batch_size" placeholder="1,2,4,8" value="1">
                            </div>
                            <div class="form-group">
                                <label for="experiment_folder_name">Experiment Folder Name:</label>
                                <input type="text" id="experiment_folder_name" name="experiment_folder_name" placeholder="my-experiment">
                            </div>
                            <div class="form-group">
                                <label for="num_workers">Number of Workers:</label>
                                <input type="number" id="num_workers" name="num_workers" min="1" max="100" value="1">
                            </div>
                        </div>

                        <!-- Request Parameters -->
                        <div class="parameter-section">
                            <h3>Request Parameters</h3>
                            <div class="form-group">
                                <label for="additional_request_params">Additional Request Parameters (JSON):</label>
                                <textarea id="additional_request_params" name="additional_request_params" rows="3" placeholder='{"temperature": 0.7, "max_tokens": 1000}'></textarea>
                            </div>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" onclick="resetParameters()">Reset to Defaults</button>
                    <button class="btn btn-outline" onclick="closeParameterEditor()">Cancel</button>
                    <button class="btn btn-primary" onclick="saveParameters()">Save Parameters</button>
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
        let reconnectAttempts = 0;
        let maxReconnectAttempts = 10;
        let reconnectDelay = 5000; // Start with 5 seconds
        let isConnecting = false;
        
        function connect() {
            if (isConnecting) {
                console.log('Connection already in progress, skipping...');
                return;
            }
            
            isConnecting = true;
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            console.log(`Attempting to connect to ${wsUrl} (attempt ${reconnectAttempts + 1})`);
            updateConnectionStatus('connecting', 'Connecting...');
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('Connected to GenAI Bench');
                updateConnectionStatus('connected', 'Connected');
                document.getElementById('connection-text').textContent = 'Connected';
                reconnectAttempts = 0; // Reset on successful connection
                reconnectDelay = 5000; // Reset delay
                isConnecting = false;
            };
            
            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleEvent(data);
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
            
            ws.onclose = function(event) {
                console.log('Disconnected from GenAI Bench', event.code, event.reason);
                isConnecting = false;
                
                // Don't reconnect if it was a clean close or too many attempts
                if (event.code === 1000 || reconnectAttempts >= maxReconnectAttempts) {
                    updateConnectionStatus('error', 'Disconnected');
                    document.getElementById('connection-text').textContent = 'Disconnected';
                    return;
                }
                
                updateConnectionStatus('error', 'Disconnected');
                document.getElementById('connection-text').textContent = `Disconnected - Reconnecting in ${reconnectDelay/1000}s...`;
                
                reconnectAttempts++;
                console.log(`Reconnection attempt ${reconnectAttempts}/${maxReconnectAttempts}`);
                
                setTimeout(() => {
                    connect();
                }, reconnectDelay);
                
                // Exponential backoff with max delay of 30 seconds
                reconnectDelay = Math.min(reconnectDelay * 1.5, 30000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('error', 'Connection Error');
                isConnecting = false;
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
                case 'parameters_updated':
                    handleParametersUpdated(event.data);
                    break;
                case 'current_parameters':
                    handleCurrentParameters(event.data);
                    break;
                case 'parameter_update_confirmed':
                    handleParameterUpdateConfirmed(event.data);
                    break;
                case 'parameter_update_error':
                    handleParameterUpdateError(event.data);
                    break;
                case 'benchmark_start_requested':
                    handleBenchmarkStartRequested(event.data);
                    break;
                case 'heartbeat':
                    // Handle heartbeat - connection is alive, no need to log
                    console.debug('Received heartbeat');
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
        
        // Parameter Editor Functions
        function toggleParameterEditor() {
            const modal = document.getElementById('parameter-editor-modal');
            if (modal.style.display === 'none' || modal.style.display === '') {
                modal.style.display = 'flex';
                loadCurrentParameters();
            } else {
                modal.style.display = 'none';
            }
        }
        
        function closeParameterEditor() {
            document.getElementById('parameter-editor-modal').style.display = 'none';
        }
        
        function loadCurrentParameters() {
            // Request current parameters from server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'get_parameters'
                }));
            }
        }
        
        function saveParameters() {
            const form = document.getElementById('parameter-form');
            const formData = new FormData(form);
            const parameters = {};
            
            // Convert form data to object
            for (let [key, value] of formData.entries()) {
                if (key === 'num_concurrency' || key === 'batch_size') {
                    // Convert comma-separated string to array
                    parameters[key] = value.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
                } else if (key === 'additional_request_params') {
                    // Parse JSON
                    try {
                        parameters[key] = value ? JSON.parse(value) : {};
                    } catch (e) {
                        alert('Invalid JSON in Additional Request Parameters');
                        return;
                    }
                } else if (key === 'max_requests_per_run' || key === 'max_time_per_run' || key === 'num_workers') {
                    parameters[key] = parseInt(value);
                } else {
                    parameters[key] = value;
                }
            }
            
            // Send parameters to server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'update_parameters',
                    parameters: parameters
                }));
                closeParameterEditor();
            } else {
                alert('Not connected to server');
            }
        }
        
        function resetParameters() {
            if (confirm('Are you sure you want to reset all parameters to defaults?')) {
                // Reset form to default values
                document.getElementById('api_backend').value = 'baseten';
                document.getElementById('api_base').value = '';
                document.getElementById('api_key').value = '';
                document.getElementById('api_model_name').value = '';
                document.getElementById('task').value = 'text-to-text';
                document.getElementById('max_requests_per_run').value = '64';
                document.getElementById('max_time_per_run').value = '600';
                document.getElementById('num_concurrency').value = '2,4,8,16,24,32';
                document.getElementById('traffic_scenario').value = 'D(2000,500)';
                document.getElementById('batch_size').value = '1';
                document.getElementById('experiment_folder_name').value = '';
                document.getElementById('num_workers').value = '1';
                document.getElementById('additional_request_params').value = '';
            }
        }
        
        function startBenchmark() {
            // Get current parameters and start benchmark
            const form = document.getElementById('parameter-form');
            const formData = new FormData(form);
            const config = {};
            
            // Convert form data to config object
            for (let [key, value] of formData.entries()) {
                if (key === 'num_concurrency' || key === 'batch_size') {
                    config[key] = value.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
                } else if (key === 'additional_request_params') {
                    try {
                        config[key] = value ? JSON.parse(value) : {};
                    } catch (e) {
                        alert('Invalid JSON in Additional Request Parameters');
                        return;
                    }
                } else if (key === 'max_requests_per_run' || key === 'max_time_per_run' || key === 'num_workers') {
                    config[key] = parseInt(value);
                } else {
                    config[key] = value;
                }
            }
            
            // Send benchmark start request
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'start_benchmark',
                    config: config
                }));
                addLog('Benchmark start requested', 'info');
            } else {
                alert('Not connected to server');
            }
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
        
        // Parameter Editor Event Handlers
        function handleParametersUpdated(data) {
            console.log('Parameters updated:', data);
            addLog(`Parameters updated: ${Object.keys(data.updated_parameters).join(', ')}`, 'info');
        }
        
        function handleCurrentParameters(data) {
            console.log('Current parameters received:', data);
            // Populate form with current parameters
            populateParameterForm(data);
        }
        
        function handleParameterUpdateConfirmed(data) {
            console.log('Parameter update confirmed:', data);
            addLog('Parameters saved successfully', 'info');
        }
        
        function handleParameterUpdateError(data) {
            console.error('Parameter update error:', data);
            addLog(`Parameter update failed: ${data.error}`, 'error');
            alert(`Failed to update parameters: ${data.error}`);
        }
        
        function handleBenchmarkStartRequested(data) {
            console.log('Benchmark start requested:', data);
            addLog('Benchmark start request received by server', 'info');
        }
        
        function populateParameterForm(parameters) {
            // Populate form fields with current parameters
            Object.keys(parameters).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    if (key === 'num_concurrency' || key === 'batch_size') {
                        // Convert array to comma-separated string
                        element.value = Array.isArray(parameters[key]) ? parameters[key].join(',') : parameters[key];
                    } else if (key === 'additional_request_params') {
                        // Convert object to JSON string
                        element.value = JSON.stringify(parameters[key] || {}, null, 2);
                    } else {
                        element.value = parameters[key] || '';
                    }
                }
            });
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
