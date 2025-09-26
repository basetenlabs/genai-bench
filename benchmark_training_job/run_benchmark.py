#!/usr/bin/env python3
"""
Simple benchmark runner for Baseten Training Job
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def main():
    """Run the benchmark shell script"""
    print("🎯 Running GenAI Bench...")
    
    # Run the shell script
    try:
        result = subprocess.run(["./run_benchmark.sh"], check=True)
        print("✅ Benchmark completed successfully!")
        
        # Save results to checkpoint directory
        checkpoint_dir = os.getenv("BT_CHECKPOINT_DIR", "/mnt/ckpts")
        print(f"📁 Checkpoint directory: {checkpoint_dir}")
        print(f"📁 Checkpoint directory exists: {os.path.exists(checkpoint_dir)}")
        
        if os.path.exists(checkpoint_dir):
            # Find benchmark results (typically in current directory or experiment folders)
            current_dir = Path(".")
            print(f"📁 Current directory contents: {list(current_dir.iterdir())}")
            
            experiment_dirs = list(current_dir.glob("*_benchmark_*"))
            print(f"📁 Found experiment directories: {experiment_dirs}")
            
            if experiment_dirs:
                # Copy the most recent experiment directory
                latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
                print(f"📁 Copying results from {latest_experiment} to {checkpoint_dir}")
                shutil.copytree(latest_experiment, Path(checkpoint_dir) / latest_experiment.name, dirs_exist_ok=True)
            else:
                # Copy all JSON files and other results
                print(f"📁 Copying all results to {checkpoint_dir}")
                for file_path in current_dir.glob("*.json"):
                    print(f"📁 Copying {file_path} to {checkpoint_dir}")
                    shutil.copy2(file_path, checkpoint_dir)
                for file_path in current_dir.glob("*.png"):
                    print(f"📁 Copying {file_path} to {checkpoint_dir}")
                    shutil.copy2(file_path, checkpoint_dir)
                for file_path in current_dir.glob("*.xlsx"):
                    print(f"📁 Copying {file_path} to {checkpoint_dir}")
                    shutil.copy2(file_path, checkpoint_dir)
        else:
            print(f"❌ Checkpoint directory {checkpoint_dir} does not exist!")
            # Try to create it
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"✅ Created checkpoint directory {checkpoint_dir}")
            except Exception as e:
                print(f"❌ Failed to create checkpoint directory: {e}")
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed with return code: {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
