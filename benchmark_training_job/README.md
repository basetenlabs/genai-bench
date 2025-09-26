# GenAI Bench - Baseten Training Job

Simple benchmark training job for running GenAI Bench on Baseten.

## 🚀 Quick Start

### 1. Deploy the Training Job

```bash
truss train push benchmark_training_config.py --tail
```

### 2. Download Results

#### Option A: Download via Truss CLI (basic files)
```bash
truss train download --job-id <JOB_ID> --target-directory ./benchmark_results
```

#### Option B: Download all checkpoint files (recommended)
After the training job completes, you'll get a JSON file with checkpoint URLs. Use our download script:

```bash
# Download all checkpoint files with proper directory structure
./download_all_checkpoints.sh genai-benchmark_<JOB_ID>_checkpoints.json

# Or use the Python script directly
python3 download_checkpoints.py genai-benchmark_<JOB_ID>_checkpoints.json
```

This will download all benchmark results including:
- JSON files for each concurrency level
- Performance plots (PNG files)
- Excel summary reports
- Experiment metadata

## 📋 What It Does

This training job runs the GenAI Bench benchmark with:

- **Model**: Qwen3-30B-A3B-Instruct-2507-FP8
- **Endpoint**: https://model-yqvy8neq.api.baseten.co/environments/production/predict
- **Concurrency Levels**: 2, 4, 8, 16, 32, 64
- **Traffic Scenario**: N(2000,200)/(200,20)

## 📁 Files

- `benchmark_training_config.py` - Baseten training job config
- `run_benchmark.sh` - The benchmark command as a shell script
- `run_benchmark.py` - Simple Python script that runs the shell script
- `download_checkpoints.py` - Python script to download checkpoint files
- `download_all_checkpoints.sh` - Shell script wrapper for downloading checkpoints
- `README.md` - This file

## 🔧 Customization

To change the benchmark parameters, edit `run_benchmark.sh` with your desired command.

## 📊 Results

After downloading, you'll get:
- JSON files for each concurrency level
- Performance plots
- Excel reports
