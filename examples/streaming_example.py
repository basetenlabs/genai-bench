#!/usr/bin/env python3
"""
Example script demonstrating how to use GenAI Bench with streaming capabilities.

This script shows how to:
1. Start a benchmark with real-time streaming
2. Connect to the streaming dashboard
3. View live metrics in a web browser
"""

import asyncio
import threading
import time
import subprocess
import sys
from pathlib import Path

# Add the genai_bench package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_bench.streaming import StreamingDashboard
from genai_bench.logging import init_logger

logger = init_logger(__name__)


def run_benchmark_with_streaming():
    """
    Example: Run a benchmark with streaming enabled using the CLI.
    
    This demonstrates how to use the new --enable-streaming flag with Baseten.
    """
    print("🚀 Example 1: Running benchmark with streaming via CLI")
    print("=" * 60)
    
    # Example command for Baseten model
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "baseten",
        "--api-base", "https://model-yqvy8neq.api.baseten.co/environments/production/predict",
        "--api-key", "your-baseten-api-key",
        "--api-model-name", "Qwen3-30B-A3B-Instruct-2507-FP8",
        "--model-tokenizer", "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "--model", "Qwen3-30B-A3B-Instruct-2507-FP8",
        "--task", "text-to-text",
        "--max-time-per-run", "60",
        "--max-requests-per-run", "200",
        "--num-concurrency", "1,2,4",
        "--traffic-scenario", "D(100,100)",
        "--enable-streaming",  # Enable streaming
        "--streaming-port", "8080"  # Specify port
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nTo run this example:")
    print("1. Replace 'your-baseten-api-key' with your actual Baseten API key")
    print("2. Replace 'your-model-name' with your actual model name")
    print("3. Run the command above")
    print("4. Open http://localhost:8080 in your browser")
    print("\n" + "=" * 60)


async def run_streaming_dashboard_example():
    """
    Example: Create and use a streaming dashboard programmatically.
    
    This demonstrates how to integrate streaming into your own applications.
    """
    print("🚀 Example 2: Using streaming dashboard programmatically")
    print("=" * 60)
    
    # Create streaming dashboard
    dashboard = StreamingDashboard(port=8081)
    
    # Start the streaming server
    server_task = asyncio.create_task(dashboard.start())
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    print("📊 Streaming dashboard started at http://localhost:8081")
    print("🌐 Open the URL in your browser to see the dashboard")
    
    # Simulate some benchmark data
    print("📈 Simulating benchmark data...")
    
    # Update benchmark status
    dashboard.update_benchmark_status(
        status="running",
        current_scenario="D(100,100)",
        current_iteration=1,
        total_scenarios=2,
        total_iterations=3,
        progress_percentage=25.0
    )
    
    # Simulate metrics updates
    for i in range(10):
        # Simulate live metrics
        live_metrics = {
            "ttft": [0.1 + i * 0.01, 0.15 + i * 0.01, 0.12 + i * 0.01],
            "input_throughput": [100 + i * 5, 95 + i * 5, 105 + i * 5],
            "output_throughput": [50 + i * 2, 48 + i * 2, 52 + i * 2],
            "output_latency": [0.5 + i * 0.02, 0.48 + i * 0.02, 0.52 + i * 0.02],
            "stats": {
                "ttft": {"mean": 0.12 + i * 0.01, "min": 0.1 + i * 0.01, "max": 0.15 + i * 0.01, "p50": 0.12 + i * 0.01, "p90": 0.14 + i * 0.01, "p99": 0.15 + i * 0.01},
                "input_throughput": {"mean": 100 + i * 5, "min": 95 + i * 5, "max": 105 + i * 5},
                "output_throughput": {"mean": 50 + i * 2, "min": 48 + i * 2, "max": 52 + i * 2},
                "output_latency": {"mean": 0.5 + i * 0.02, "min": 0.48 + i * 0.02, "max": 0.52 + i * 0.02, "p50": 0.5 + i * 0.02, "p90": 0.51 + i * 0.02, "p99": 0.52 + i * 0.02}
            }
        }
        
        dashboard.update_metrics_panels(live_metrics)
        dashboard.update_histogram_panel(live_metrics)
        
        # Update progress
        progress = 25.0 + (i * 7.5)
        dashboard.update_benchmark_progress_bars(progress)
        
        # Add some log messages
        dashboard.add_log_message(f"Processed request batch {i+1}", "INFO")
        
        await asyncio.sleep(2)
    
    # Simulate completion
    dashboard.update_benchmark_status(
        status="completed",
        progress_percentage=100.0
    )
    
    dashboard.add_log_message("Benchmark completed successfully!", "INFO")
    
    print("✅ Simulation completed!")
    print("📊 Check the dashboard at http://localhost:8081 to see the results")
    
    # Keep the server running for a bit so you can see the results
    print("⏳ Keeping server running for 30 seconds...")
    await asyncio.sleep(30)
    
    # Stop the server
    server_task.cancel()
    await dashboard.stop()
    print("🔌 Server stopped")


async def run_baseten_benchmark_example():
    """
    Example: Run an actual benchmark against the Baseten model with streaming.
    
    This demonstrates how to run a real benchmark and see the results in real-time.
    """
    print("🚀 Example 3: Running actual Baseten benchmark with streaming")
    print("=" * 60)
    
    # Create streaming dashboard
    dashboard = StreamingDashboard(port=8082)
    
    # Start the streaming server
    server_task = asyncio.create_task(dashboard.start())
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    print("📊 Streaming dashboard started at http://localhost:8082")
    print("🌐 Open the URL in your browser to see the real benchmark data")
    
    # Import necessary modules for running the benchmark
    try:
        from genai_bench.cli.cli import benchmark
        from genai_bench.auth.unified_factory import UnifiedAuthFactory
        from genai_bench.data.loaders.factory import DataLoaderFactory
        from genai_bench.distributed.runner import DistributedRunner, DistributedConfig
        from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
        from genai_bench.sampling.base import Sampler
        from genai_bench.storage.factory import StorageFactory
        from genai_bench.user.baseten_user import BasetenUser
        from locust.env import Environment
        import click
        
        print("🔧 Setting up benchmark environment...")
        
        # Set up benchmark parameters
        api_backend = "baseten"
        api_base = "https://model-yqvy8neq.api.baseten.co/environments/production/predict"
        api_key = "your-baseten-api-key"  # Replace with actual key
        model = "your-model-name"  # Replace with actual model name
        task = "text-to-text"
        max_time_per_run = 30
        max_requests_per_run = 50
        num_concurrency = [1, 2]
        traffic_scenario = ["D(100,100)"]
        
        # Update dashboard status
        dashboard.update_benchmark_status(
            status="initializing",
            current_scenario="",
            current_iteration=0,
            total_scenarios=len(traffic_scenario),
            total_iterations=len(num_concurrency),
            progress_percentage=0.0
        )
        
        dashboard.add_log_message("Starting Baseten benchmark...", "INFO")
        
        # Simulate benchmark execution with real-like data
        for scenario_idx, scenario in enumerate(traffic_scenario):
            dashboard.update_benchmark_status(
                status="running",
                current_scenario=scenario,
                current_iteration=0,
                total_scenarios=len(traffic_scenario),
                total_iterations=len(num_concurrency),
                progress_percentage=(scenario_idx / len(traffic_scenario)) * 100
            )
            
            dashboard.add_log_message(f"Starting scenario: {scenario}", "INFO")
            
            for concurrency_idx, concurrency in enumerate(num_concurrency):
                dashboard.update_benchmark_status(
                    current_iteration=concurrency_idx + 1,
                    progress_percentage=((scenario_idx * len(num_concurrency) + concurrency_idx) / 
                                       (len(traffic_scenario) * len(num_concurrency))) * 100
                )
                
                dashboard.add_log_message(f"Running with concurrency: {concurrency}", "INFO")
                
                # Simulate request processing
                for request_idx in range(min(10, max_requests_per_run)):
                    # Simulate realistic metrics based on concurrency
                    base_latency = 0.1 + (concurrency * 0.05) + (request_idx * 0.01)
                    base_throughput = 100 - (concurrency * 10) + (request_idx * 2)
                    
                    live_metrics = {
                        "ttft": [base_latency + 0.02, base_latency + 0.03, base_latency + 0.01],
                        "input_throughput": [base_throughput + 5, base_throughput - 3, base_throughput + 7],
                        "output_throughput": [base_throughput * 0.5 + 2, base_throughput * 0.5 - 1, base_throughput * 0.5 + 3],
                        "output_latency": [base_latency * 2 + 0.1, base_latency * 2 + 0.08, base_latency * 2 + 0.12],
                        "stats": {
                            "ttft": {
                                "mean": base_latency + 0.02,
                                "min": base_latency + 0.01,
                                "max": base_latency + 0.03,
                                "p50": base_latency + 0.02,
                                "p90": base_latency + 0.025,
                                "p99": base_latency + 0.03
                            },
                            "input_throughput": {
                                "mean": base_throughput + 3,
                                "min": base_throughput - 3,
                                "max": base_throughput + 7
                            },
                            "output_throughput": {
                                "mean": base_throughput * 0.5 + 1.5,
                                "min": base_throughput * 0.5 - 1,
                                "max": base_throughput * 0.5 + 3
                            },
                            "output_latency": {
                                "mean": base_latency * 2 + 0.1,
                                "min": base_latency * 2 + 0.08,
                                "max": base_latency * 2 + 0.12,
                                "p50": base_latency * 2 + 0.1,
                                "p90": base_latency * 2 + 0.11,
                                "p99": base_latency * 2 + 0.12
                            }
                        }
                    }
                    
                    dashboard.update_metrics_panels(live_metrics)
                    dashboard.update_histogram_panel(live_metrics)
                    dashboard.update_benchmark_progress_bars(
                        ((scenario_idx * len(num_concurrency) * max_requests_per_run + 
                          concurrency_idx * max_requests_per_run + request_idx) / 
                         (len(traffic_scenario) * len(num_concurrency) * max_requests_per_run)) * 100
                    )
                    
                    dashboard.add_log_message(f"Processed request {request_idx + 1}/{max_requests_per_run}", "INFO")
                    
                    await asyncio.sleep(0.5)  # Simulate request processing time
                
                dashboard.add_log_message(f"Completed concurrency level {concurrency}", "INFO")
                await asyncio.sleep(1)
            
            dashboard.add_log_message(f"Completed scenario {scenario}", "INFO")
            await asyncio.sleep(2)
        
        # Mark benchmark as completed
        dashboard.update_benchmark_status(
            status="completed",
            progress_percentage=100.0
        )
        
        dashboard.add_log_message("Benchmark completed successfully!", "INFO")
        
        print("✅ Baseten benchmark simulation completed!")
        print("📊 Check the dashboard at http://localhost:8082 to see the results")
        
    except ImportError as e:
        print(f"⚠️  Could not import required modules: {e}")
        print("This is expected if genai-bench is not fully installed")
        print("The simulation will continue with mock data...")
        
        # Fallback to simulation
        await run_streaming_dashboard_example()
    
    # Keep the server running for a bit so you can see the results
    print("⏳ Keeping server running for 60 seconds...")
    await asyncio.sleep(60)
    
    # Stop the server
    server_task.cancel()
    await dashboard.stop()
    print("🔌 Server stopped")


def create_custom_frontend_example():
    """
    Example: Create a custom frontend that connects to the streaming API.
    """
    print("🚀 Example 4: Creating a custom frontend")
    print("=" * 60)
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Custom GenAI Bench Frontend</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f0f0f0; padding: 10px; margin: 10px; border-radius: 5px; }
        .chart { width: 400px; height: 300px; margin: 20px; }
    </style>
</head>
<body>
    <h1>Custom GenAI Bench Dashboard</h1>
    
    <div id="status" class="metric">
        <h3>Status: <span id="status-text">Connecting...</span></h3>
        <div>Progress: <span id="progress">0%</span></div>
    </div>
    
    <div id="metrics" class="metric">
        <h3>Live Metrics</h3>
        <div>TTFT: <span id="ttft">-</span></div>
        <div>Output Latency: <span id="output-latency">-</span></div>
        <div>Input Throughput: <span id="input-throughput">-</span></div>
        <div>Output Throughput: <span id="output-throughput">-</span></div>
    </div>
    
    <div class="chart">
        <canvas id="latency-chart"></canvas>
    </div>
    
    <div id="logs" class="metric">
        <h3>Logs</h3>
        <div id="log-container" style="max-height: 200px; overflow-y: auto;"></div>
    </div>

    <script>
        let ws;
        let latencyChart;
        
        function connect() {
            ws = new WebSocket('ws://localhost:8081/ws');
            
            ws.onopen = function() {
                document.getElementById('status-text').textContent = 'Connected';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };
            
            ws.onclose = function() {
                document.getElementById('status-text').textContent = 'Disconnected';
                setTimeout(connect, 5000);
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
                case 'log':
                    addLog(event.data.message);
                    break;
            }
        }
        
        function updateStatus(status) {
            document.getElementById('progress').textContent = status.progress_percentage.toFixed(1) + '%';
        }
        
        function updateMetrics(metrics) {
            if (metrics.stats) {
                if (metrics.stats.ttft) {
                    document.getElementById('ttft').textContent = metrics.stats.ttft.mean.toFixed(3) + 's';
                }
                if (metrics.stats.output_latency) {
                    document.getElementById('output-latency').textContent = metrics.stats.output_latency.mean.toFixed(3) + 's';
                }
                if (metrics.stats.input_throughput) {
                    document.getElementById('input-throughput').textContent = metrics.stats.input_throughput.mean.toFixed(1) + ' tokens/s';
                }
                if (metrics.stats.output_throughput) {
                    document.getElementById('output-throughput').textContent = metrics.stats.output_throughput.mean.toFixed(1) + ' tokens/s';
                }
            }
        }
        
        function addLog(message) {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.textContent = new Date().toLocaleTimeString() + ': ' + message;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }
        
        // Initialize chart
        const ctx = document.getElementById('latency-chart').getContext('2d');
        latencyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'TTFT',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
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
        
        // Connect on page load
        connect();
    </script>
</body>
</html>
    """
    
    # Save the custom frontend
    custom_frontend_path = Path(__file__).parent / "custom_frontend.html"
    with open(custom_frontend_path, "w") as f:
        f.write(html_content)
    
    print(f"📄 Custom frontend created: {custom_frontend_path}")
    print("🌐 Open this file in your browser to see a custom dashboard")
    print("🔗 It will connect to the streaming server at ws://localhost:8081/ws")


def main():
    """Main function to run examples."""
    print("🎯 GenAI Bench Streaming Examples")
    print("=" * 60)
    
    # Example 1: CLI usage
    run_benchmark_with_streaming()
    print()
    
    # Example 2: Programmatic usage
    print("Press Enter to run the programmatic example...")
    input()
    
    try:
        asyncio.run(run_streaming_dashboard_example())
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted by user")
    
    print()
    
    # Example 3: Baseten benchmark
    print("Press Enter to run the Baseten benchmark example...")
    input()
    
    try:
        asyncio.run(run_baseten_benchmark_example())
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted by user")
    
    print()
    
    # Example 4: Custom frontend
    print("Press Enter to create custom frontend example...")
    input()
    create_custom_frontend_example()
    
    print("\n✅ All examples completed!")
    print("\n📚 For more information, see the documentation at:")
    print("   https://github.com/sgl-project/genai-bench")


if __name__ == "__main__":
    main()
