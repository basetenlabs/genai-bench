#!/usr/bin/env python3
"""
Script to run a benchmark against the Baseten model endpoint with streaming.

Usage:
    python run_baseten_benchmark.py --api-key YOUR_API_KEY --model YOUR_MODEL_NAME

Example:
    python run_baseten_benchmark.py --api-key "pt_abc123" --model "my-model"
"""

import asyncio
import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add the genai_bench package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_bench.logging import init_logger

logger = init_logger(__name__)


def run_baseten_benchmark(api_key: str, model_name: str, streaming_port: int = 8080, enable_streaming: bool = True):
    """
    Run a benchmark against the Baseten model endpoint.
    
    Args:
        api_key: Baseten API key
        model_name: Name of the model to benchmark
        streaming_port: Port for the streaming dashboard
    """
    
    # Set environment variables
    env = os.environ.copy()
    env["BASETEN_API_KEY"] = api_key
    
    # Construct the benchmark command
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "baseten",
        "--api-base", "https://model-yqvy8neq.api.baseten.co/environments/production/predict",
        "--api-key", api_key,
        "--api-model-name", model_name,
        "--model-tokenizer", "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "--model", model_name,
        "--task", "text-to-text",
        "--max-time-per-run", "60",
        "--max-requests-per-run", "10",
        "--num-concurrency", "1",
        "--num-concurrency", "2", 
        "--num-concurrency", "4",
        "--traffic-scenario", "D(100,100)",
    ]
    
    # Add streaming options if enabled
    if enable_streaming:
        cmd.extend([
            "--enable-streaming",
            "--streaming-port", str(streaming_port),
        ])
    
    # Add experiment folder name
    cmd.extend([
        "--experiment-folder-name", f"baseten_benchmark_{model_name}",
        # "--upload-results", "false"  # Set to true if you want to upload results
    ])
    
    print(f"🚀 Starting Baseten benchmark for model: {model_name}")
    print(f"📊 Streaming dashboard will be available at: http://localhost:{streaming_port}")
    print(f"🔑 Using API key: {api_key[:8]}...")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the benchmark
        result = subprocess.run(cmd, env=env, check=True)
        print("✅ Benchmark completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed with exit code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ genai-bench command not found. Please install genai-bench first.")
        print("   pip install genai-bench")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        return 1


def main():
    """Main function to parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run a benchmark against the Baseten model endpoint with streaming"
    )
    
    parser.add_argument(
        "--api-key",
        default=os.getenv("BASETEN_API_KEY"),
        help="Baseten API key (or set BASETEN_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the model to benchmark"
    )
    
    parser.add_argument(
        "--streaming-port",
        type=int,
        default=8080,
        help="Port for the streaming dashboard (default: 8080)"
    )
    
    parser.add_argument(
        "--max-time",
        type=int,
        default=60,
        help="Maximum time per run in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--max-requests",
        type=int,
        default=100,
        help="Maximum requests per run (default: 100)"
    )
    
    parser.add_argument(
        "--concurrency",
        default="1,2,4",
        help="Concurrency levels to test (default: 1,2,4)"
    )
    
    parser.add_argument(
        "--scenario",
        default="D(100,100)",
        help="Traffic scenario (default: D(100,100))"
    )
    
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        default=True,
        help="Enable real-time streaming to web dashboard (default: True)"
    )
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("❌ Error: API key is required!")
        print("   Either provide --api-key argument or set BASETEN_API_KEY environment variable")
        return 1
    
    print("🎯 Baseten Benchmark Runner")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"API Key: {args.api_key[:8]}...")
    print(f"Streaming Port: {args.streaming_port}")
    print(f"Max Time: {args.max_time}s")
    print(f"Max Requests: {args.max_requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Scenario: {args.scenario}")
    print("=" * 50)
    print()
    
    # Run the benchmark
    return run_baseten_benchmark(
        api_key=args.api_key,
        model_name=args.model,
        streaming_port=args.streaming_port,
        enable_streaming=args.enable_streaming
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
