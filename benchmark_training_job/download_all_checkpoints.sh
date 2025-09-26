#!/bin/bash
# Download all checkpoint files from Baseten training job

set -e

# Check if JSON file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoints_json_file>"
    echo "Example: $0 genai-benchmark_nwxky73_checkpoints.json"
    exit 1
fi

JSON_FILE="$1"

# Check if file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "❌ File $JSON_FILE not found!"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not available!"
    exit 1
fi

# Run the download script
echo "🚀 Starting checkpoint download..."
python3 download_checkpoints.py "$JSON_FILE"

echo "✅ Download complete!"
