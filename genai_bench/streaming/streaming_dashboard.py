"""Streaming dashboard for real-time web frontend updates."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from genai_bench.protocol import LiveMetricsData
from genai_bench.logging import init_logger

logger = init_logger(__name__)


@dataclass
class BenchmarkStatus:
    """Status information for a running benchmark."""
    status: str  # "running", "completed", "failed"
    current_scenario: str
    current_iteration: int
    total_scenarios: int
    total_iterations: int
    progress_percentage: float
    start_time: float
    estimated_end_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class StreamEvent:
    """Event structure for streaming data."""
    event_type: str  # "metrics", "status", "log", "error"
    timestamp: float
    data: Dict[str, Any]


class StreamingDashboard:
    """
    A dashboard implementation that streams real-time benchmark data to web clients.
    
    This class extends the existing dashboard functionality to support:
    - WebSocket connections for real-time updates
    - Event-driven data streaming
    - Benchmark status tracking
    - Historical data storage for completed benchmarks
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.connected_clients: Set[str] = set()
        self.benchmark_status = BenchmarkStatus(
            status="idle",
            current_scenario="",
            current_iteration=0,
            total_scenarios=0,
            total_iterations=0,
            progress_percentage=0.0,
            start_time=time.time()
        )
        self.historical_data: List[Dict[str, Any]] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        # Initialize run tracking variables for RPS calculations
        self.start_time: Optional[float] = None
        self.run_time: Optional[int] = None
        self.max_requests_per_run: Optional[int] = None
        
        # Add layout attribute for compatibility with LoggingManager
        # Create a minimal layout structure to avoid None errors
        self.layout = type('MockLayout', (), {
            '__getitem__': lambda self, key: type('MockPanel', (), {
                'update': lambda self, content: None
            })(),
            '__setitem__': lambda self, key, value: None
        })()
        
    async def start(self):
        """Start the streaming server."""
        from .streaming_server import StreamingServer
        self.server = StreamingServer(self, self.port)
        self._running = True
        await self.server.start()
        
    async def stop(self):
        """Stop the streaming server."""
        self._running = False
        if hasattr(self, 'server'):
            await self.server.stop()
            
    def update_benchmark_status(self, **kwargs):
        """Update benchmark status and broadcast to clients."""
        for key, value in kwargs.items():
            if hasattr(self.benchmark_status, key):
                setattr(self.benchmark_status, key, value)
        
        # Broadcast status update
        self._queue_event("status", asdict(self.benchmark_status))
        
    def update_metrics_panels(self, live_metrics: LiveMetricsData):
        """Update metrics and stream to clients."""
        # Create metrics event
        metrics_event = {
            "live_metrics": live_metrics,
            "timestamp": time.time()
        }
        self._queue_event("metrics", metrics_event)
        
    def update_histogram_panel(self, live_metrics: LiveMetricsData):
        """Update histogram data and stream to clients."""
        # Extract histogram data from live metrics
        histogram_data = {
            "ttft_histogram": self._create_histogram_data(live_metrics.get("ttft", [])),
            "output_latency_histogram": self._create_histogram_data(live_metrics.get("output_latency", [])),
            "timestamp": time.time()
        }
        self._queue_event("histogram", histogram_data)
        
    def update_scatter_plot_panel(self, ui_scatter_plot_metrics: Optional[List[float]]):
        """Update scatter plot data and stream to clients."""
        if ui_scatter_plot_metrics:
            scatter_data = {
                "ttft": ui_scatter_plot_metrics[0],
                "output_latency": ui_scatter_plot_metrics[1],
                "input_throughput": ui_scatter_plot_metrics[2],
                "output_throughput": ui_scatter_plot_metrics[3],
                "timestamp": time.time()
            }
            self._queue_event("scatter", scatter_data)
            
    def update_rps_vs_latency_plot(self, rps: float, e2e_latency: float):
        """Update RPS vs E2E latency scatter plot data."""
        rps_latency_data = {
            "rps": rps,
            "e2e_latency": e2e_latency,
            "timestamp": time.time()
        }
        self._queue_event("rps_vs_latency", rps_latency_data)
        
        # Add debug log for RPS vs latency updates
        logger.debug(f"RPS vs Latency update: RPS={rps:.2f}, E2E_Latency={e2e_latency:.3f}s")
        
    def update_iteration_rps_vs_latency(self, concurrency: int, aggregated_metrics, run_time: float, total_requests: int):
        """Update RPS vs E2E latency for a completed iteration."""
        if run_time > 0 and total_requests > 0:
            # Calculate RPS for this iteration
            rps = total_requests / run_time
            
            # Extract latency from aggregated metrics
            e2e_latency = 0
            if aggregated_metrics and hasattr(aggregated_metrics, 'stats'):
                stats = aggregated_metrics.stats
                # Try to get TTFT mean as primary latency metric
                if hasattr(stats, 'ttft') and stats.ttft and stats.ttft.mean:
                    e2e_latency = stats.ttft.mean
                # Fallback to output_latency if TTFT not available
                elif hasattr(stats, 'output_latency') and stats.output_latency and stats.output_latency.mean:
                    e2e_latency = stats.output_latency.mean
                # Fallback to e2e_latency if available
                elif hasattr(stats, 'e2e_latency') and stats.e2e_latency and stats.e2e_latency.mean:
                    e2e_latency = stats.e2e_latency.mean
            
            # Only update if we have valid latency data
            if e2e_latency > 0:
                self.update_rps_vs_latency_plot(rps, e2e_latency)
                logger.info(f"Iteration RPS vs Latency: Concurrency={concurrency}, RPS={rps:.2f}, E2E_Latency={e2e_latency:.3f}s")
            else:
                logger.debug(f"No valid latency data found in aggregated metrics for concurrency {concurrency}")
                logger.debug(f"Aggregated metrics stats: {aggregated_metrics.stats if aggregated_metrics else 'None'}")
        else:
            logger.debug(f"Invalid run data for RPS calculation: run_time={run_time}, total_requests={total_requests}")
            
    def update_benchmark_progress_bars(self, progress_increment: float):
        """Update progress and stream to clients."""
        self.benchmark_status.progress_percentage = progress_increment
        self._queue_event("progress", {"progress": progress_increment})
        
    def create_benchmark_progress_task(self, run_name: str):
        """Create a new benchmark task and stream to clients."""
        self._queue_event("task_created", {"run_name": run_name})
        
    def update_total_progress_bars(self, total_runs: int):
        """Update total progress and stream to clients."""
        self._queue_event("total_progress", {"total_runs": total_runs})
        
    def start_run(self, run_time: int, start_time: float, max_requests_per_run: int):
        """Start a new run and stream to clients."""
        # Store run parameters for RPS calculations
        self.start_time = start_time
        self.run_time = run_time
        self.max_requests_per_run = max_requests_per_run
        
        run_data = {
            "run_time": run_time,
            "start_time": start_time,
            "max_requests_per_run": max_requests_per_run
        }
        self._queue_event("run_started", run_data)
        
    def handle_single_request(
        self,
        live_metrics: LiveMetricsData,
        total_requests: int,
        error_code: int | None,
    ):
        """Handle a single request and stream updates to clients."""
        request_data = {
            "total_requests": total_requests,
            "error_code": error_code,
            "timestamp": time.time()
        }
        
        if error_code is None:
            # Only stream metrics for successful requests
            self.update_metrics_panels(live_metrics)
            self.update_histogram_panel(live_metrics)
            
            # Note: RPS vs latency is now calculated at iteration level, not per-request
            # This method is kept for compatibility but RPS vs latency updates happen elsewhere
            
        self._queue_event("request_processed", request_data)
        
    def reset_plot_metrics(self):
        """Reset plot metrics and stream to clients."""
        self._queue_event("metrics_reset", {})
        
    def reset_run_tracking(self):
        """Reset run tracking variables for new runs."""
        self.start_time = None
        self.run_time = None
        self.max_requests_per_run = None
        
    def reset_panels(self):
        """Reset panels and stream to clients."""
        self._queue_event("panels_reset", {})
        
    def add_log_message(self, message: str, level: str = "INFO"):
        """Add a log message and stream to clients."""
        log_data = {
            "message": message,
            "level": level,
            "timestamp": time.time()
        }
        self._queue_event("log", log_data)
        
    def add_historical_data(self, data: Dict[str, Any]):
        """Add completed benchmark data to historical storage."""
        data["timestamp"] = time.time()
        self.historical_data.append(data)
        
    def get_historical_data(self) -> List[Dict[str, Any]]:
        """Get all historical benchmark data."""
        return self.historical_data
        
    def _queue_event(self, event_type: str, data: Dict[str, Any]):
        """Queue an event for streaming to clients."""
        if not self._running:
            logger.debug(f"Dashboard not running, skipping event: {event_type}")
            return
            
        event = StreamEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        
        # Add to queue for async processing
        try:
            # Use asyncio.create_task if event loop is running
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self.event_queue.put(event))
                logger.debug(f"Queued event {event_type} using create_task")
            else:
                # Fallback: put directly if no event loop
                self.event_queue.put_nowait(event)
                logger.debug(f"Queued event {event_type} using put_nowait")
        except (RuntimeError, asyncio.QueueFull) as e:
            # Event loop not running or queue full, skip
            logger.debug(f"Failed to queue event {event_type}: {e}")
            pass
            
    def _create_histogram_data(self, values: List[float]) -> Dict[str, Any]:
        """Create histogram data from raw values."""
        if not values:
            return {"bins": [], "counts": []}
            
        import numpy as np
        
        # Create histogram bins
        hist, bin_edges = np.histogram(values, bins=10)
        
        return {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist(),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values))
        }
        
    @property
    def live(self):
        """Compatibility with existing dashboard interface."""
        return self._create_context_manager()
        
    def _create_context_manager(self):
        """Create a context manager compatible with existing dashboard interface."""
        class StreamingContextManager:
            def __init__(self, dashboard):
                self.dashboard = dashboard
                self.is_started = True  # Add expected attribute
                
            def __enter__(self):
                return self.dashboard
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
        return StreamingContextManager(self)
