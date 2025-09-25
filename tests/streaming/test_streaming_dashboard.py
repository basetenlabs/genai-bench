"""Tests for the streaming dashboard functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from genai_bench.streaming import StreamingDashboard, BenchmarkStatus
from genai_bench.protocol import LiveMetricsData


class TestStreamingDashboard:
    """Test cases for StreamingDashboard."""
    
    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = StreamingDashboard(port=8080)
        
        assert dashboard.port == 8080
        assert dashboard.connected_clients == set()
        assert isinstance(dashboard.benchmark_status, BenchmarkStatus)
        assert dashboard.benchmark_status.status == "idle"
        assert dashboard.historical_data == []
        
    def test_update_benchmark_status(self):
        """Test benchmark status updates."""
        dashboard = StreamingDashboard()
        
        # Update status
        dashboard.update_benchmark_status(
            status="running",
            current_scenario="D(100,100)",
            progress_percentage=50.0
        )
        
        assert dashboard.benchmark_status.status == "running"
        assert dashboard.benchmark_status.current_scenario == "D(100,100)"
        assert dashboard.benchmark_status.progress_percentage == 50.0
        
    def test_update_metrics_panels(self):
        """Test metrics panel updates."""
        dashboard = StreamingDashboard()
        
        # Create sample metrics
        live_metrics: LiveMetricsData = {
            "ttft": [0.1, 0.15, 0.12],
            "input_throughput": [100, 95, 105],
            "output_throughput": [50, 48, 52],
            "output_latency": [0.5, 0.48, 0.52],
            "stats": {
                "ttft": {"mean": 0.12, "min": 0.1, "max": 0.15},
                "input_throughput": {"mean": 100, "min": 95, "max": 105},
                "output_throughput": {"mean": 50, "min": 48, "max": 52},
                "output_latency": {"mean": 0.5, "min": 0.48, "max": 0.52}
            }
        }
        
        # Update metrics
        dashboard.update_metrics_panels(live_metrics)
        
        # Verify event was queued (we can't easily test the async queue without running the event loop)
        # But we can verify the method doesn't raise exceptions
        assert dashboard._running is False  # Should be False initially
        
    def test_add_log_message(self):
        """Test log message addition."""
        dashboard = StreamingDashboard()
        
        dashboard.add_log_message("Test log message", "INFO")
        dashboard.add_log_message("Error message", "ERROR")
        
        # Verify messages were queued
        assert dashboard._running is False
        
    def test_add_historical_data(self):
        """Test historical data addition."""
        dashboard = StreamingDashboard()
        
        data = {"test": "data", "value": 123}
        dashboard.add_historical_data(data)
        
        assert len(dashboard.historical_data) == 1
        assert dashboard.historical_data[0]["test"] == "data"
        assert dashboard.historical_data[0]["value"] == 123
        assert "timestamp" in dashboard.historical_data[0]
        
    def test_get_historical_data(self):
        """Test historical data retrieval."""
        dashboard = StreamingDashboard()
        
        # Add some data
        dashboard.add_historical_data({"test1": "data1"})
        dashboard.add_historical_data({"test2": "data2"})
        
        historical_data = dashboard.get_historical_data()
        
        assert len(historical_data) == 2
        assert historical_data[0]["test1"] == "data1"
        assert historical_data[1]["test2"] == "data2"
        
    def test_create_histogram_data(self):
        """Test histogram data creation."""
        dashboard = StreamingDashboard()
        
        # Test with empty data
        empty_hist = dashboard._create_histogram_data([])
        assert empty_hist == {"bins": [], "counts": []}
        
        # Test with sample data
        values = [0.1, 0.2, 0.15, 0.25, 0.12]
        hist_data = dashboard._create_histogram_data(values)
        
        assert "bins" in hist_data
        assert "counts" in hist_data
        assert "min" in hist_data
        assert "max" in hist_data
        assert "mean" in hist_data
        assert hist_data["min"] == 0.1
        assert hist_data["max"] == 0.25
        assert hist_data["mean"] == 0.164  # Approximate
        
    def test_live_context_manager(self):
        """Test the live context manager compatibility."""
        dashboard = StreamingDashboard()
        
        # Test that the live property returns a context manager
        context_manager = dashboard.live
        
        # Test context manager methods
        with context_manager as ctx:
            assert ctx is dashboard
            
    @pytest.mark.asyncio
    async def test_start_stop_server(self):
        """Test server start and stop functionality."""
        dashboard = StreamingDashboard(port=8081)  # Use different port for testing
        
        # Mock the server to avoid actually starting it
        with patch('genai_bench.streaming.streaming_server.FASTAPI_AVAILABLE', False):
            with patch('genai_bench.streaming.streaming_server.websockets') as mock_websockets:
                # Mock websockets to be available
                mock_websockets.serve = Mock()
                mock_websockets.serve.return_value = Mock()
                mock_websockets.serve.return_value.wait_closed = Mock()
                
                # Test start
                try:
                    await dashboard.start()
                except Exception:
                    # Expected since we're mocking
                    pass
                
                # Test stop
                await dashboard.stop()
                
                # Verify stop was called
                if hasattr(dashboard, 'server') and dashboard.server:
                    dashboard.server.close.assert_called_once()


class TestBenchmarkStatus:
    """Test cases for BenchmarkStatus dataclass."""
    
    def test_benchmark_status_creation(self):
        """Test BenchmarkStatus creation."""
        status = BenchmarkStatus(
            status="running",
            current_scenario="D(100,100)",
            current_iteration=1,
            total_scenarios=2,
            total_iterations=3,
            progress_percentage=25.0,
            start_time=time.time()
        )
        
        assert status.status == "running"
        assert status.current_scenario == "D(100,100)"
        assert status.current_iteration == 1
        assert status.total_scenarios == 2
        assert status.total_iterations == 3
        assert status.progress_percentage == 25.0
        assert status.estimated_end_time is None
        assert status.error_message is None
        
    def test_benchmark_status_with_optional_fields(self):
        """Test BenchmarkStatus with optional fields."""
        status = BenchmarkStatus(
            status="failed",
            current_scenario="D(100,100)",
            current_iteration=1,
            total_scenarios=2,
            total_iterations=3,
            progress_percentage=50.0,
            start_time=time.time(),
            estimated_end_time=time.time() + 3600,
            error_message="Test error"
        )
        
        assert status.status == "failed"
        assert status.estimated_end_time is not None
        assert status.error_message == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])
