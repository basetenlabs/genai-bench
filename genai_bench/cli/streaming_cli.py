"""Streaming CLI for real-time benchmark execution."""

import asyncio
import threading
import time
from typing import Optional

import click

from genai_bench.cli.cli import benchmark as original_benchmark
from genai_bench.cli.option_groups import (
    api_options,
    distributed_locust_options,
    experiment_options,
    model_auth_options,
    object_storage_options,
    oci_auth_options,
    sampling_options,
    server_options,
    storage_auth_options,
)
from genai_bench.streaming import StreamingDashboard
from genai_bench.logging import init_logger

logger = init_logger(__name__)


@click.command(context_settings={"show_default": True})
@api_options
@model_auth_options
@oci_auth_options
@server_options
@experiment_options
@sampling_options
@distributed_locust_options
@object_storage_options
@storage_auth_options
@click.option(
    "--streaming-port",
    default=8080,
    help="Port for the streaming dashboard server",
    type=int,
)
@click.option(
    "--enable-streaming",
    is_flag=True,
    default=True,
    help="Enable real-time streaming to web dashboard",
)
@click.pass_context
def streaming_benchmark(
    ctx,
    streaming_port: int,
    enable_streaming: bool,
    **kwargs
):
    """
    Run a benchmark with real-time streaming to web dashboard.
    
    This command runs the same benchmark as the regular benchmark command
    but also starts a web server that streams real-time data to a frontend.
    
    The web dashboard will be available at http://localhost:{streaming_port}
    """
    
    if not enable_streaming:
        # Fall back to regular benchmark
        return original_benchmark.callback(ctx, **kwargs)
    
    # Create streaming dashboard
    dashboard = StreamingDashboard(port=streaming_port)
    
    # Start the streaming server in a separate thread
    def run_streaming_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(dashboard.start())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
    
    streaming_thread = threading.Thread(target=run_streaming_server, daemon=True)
    streaming_thread.start()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    logger.info(f"🚀 Streaming dashboard started at http://localhost:{streaming_port}")
    logger.info("📊 Open the URL in your browser to view real-time benchmark data")
    
    try:
        # Run the benchmark with the streaming dashboard
        # We need to modify the original benchmark function to use our dashboard
        run_benchmark_with_streaming(ctx, dashboard, **kwargs)
    except KeyboardInterrupt:
        logger.info("🛑 Benchmark interrupted by user")
    finally:
        # Stop the streaming server
        asyncio.run(dashboard.stop())
        logger.info("🔌 Streaming server stopped")


def run_benchmark_with_streaming(ctx, dashboard: StreamingDashboard, **kwargs):
    """
    Run benchmark with streaming dashboard integration.
    
    This function modifies the original benchmark execution to use the streaming dashboard
    instead of the regular Rich dashboard.
    """
    # Import the original benchmark function's logic
    from genai_bench.cli.cli import benchmark as original_benchmark_func
    
    # Store original dashboard creation
    original_create_dashboard = None
    try:
        from genai_bench.ui.dashboard import create_dashboard as original_create_dashboard_func
        original_create_dashboard = original_create_dashboard_func
    except ImportError:
        pass
    
    # Monkey patch the dashboard creation to use our streaming dashboard
    def create_streaming_dashboard():
        return dashboard
    
    # Replace the dashboard creation function
    if original_create_dashboard:
        import genai_bench.ui.dashboard
        genai_bench.ui.dashboard.create_dashboard = create_streaming_dashboard
    
    try:
        # Run the original benchmark function
        original_benchmark_func.callback(ctx, **kwargs)
    finally:
        # Restore original dashboard creation
        if original_create_dashboard:
            genai_bench.ui.dashboard.create_dashboard = original_create_dashboard


@click.command()
@click.option(
    "--port",
    default=8080,
    help="Port for the streaming dashboard server",
    type=int,
)
def start_streaming_server(port: int):
    """
    Start a standalone streaming server for viewing historical benchmark data.
    
    This command starts only the web server without running a benchmark.
    Useful for viewing results from previously completed benchmarks.
    """
    dashboard = StreamingDashboard(port=port)
    
    logger.info(f"🚀 Starting streaming server on port {port}")
    logger.info(f"📊 Dashboard will be available at http://localhost:{port}")
    
    try:
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    finally:
        asyncio.run(dashboard.stop())
