#!/usr/bin/env python3
"""
Script to download checkpoint files from Baseten training job results
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

def download_file(url, local_path):
    """Download a file from URL to local path"""
    try:
        print(f"📥 Downloading {url}")
        
        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download using urllib (built-in)
        urllib.request.urlretrieve(url, local_path)
        
        print(f"✅ Downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def main():
    """Main function to download checkpoint files"""
    if len(sys.argv) != 2:
        print("Usage: python download_checkpoints.py <checkpoints_json_file>")
        print("Example: python download_checkpoints.py genai-benchmark_nwxky73_checkpoints.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ File {json_file} not found!")
        sys.exit(1)
    
    # Load the checkpoint data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    job_id = data['job']['id']
    output_dir = Path(f"checkpoint_results_{job_id}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Downloading checkpoints to {output_dir}")
    
    # Download each checkpoint file
    success_count = 0
    total_count = len(data['checkpoint_artifacts'])
    
    for artifact in data['checkpoint_artifacts']:
        url = artifact['url']
        relative_path = artifact['relative_file_name']
        
        # Create local path
        local_path = output_dir / relative_path
        
        if download_file(url, local_path):
            success_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   Total files: {total_count}")
    print(f"   Successfully downloaded: {success_count}")
    print(f"   Failed: {total_count - success_count}")
    print(f"   Output directory: {output_dir.absolute()}")
    
    if success_count > 0:
        print(f"\n📋 Files downloaded:")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                print(f"   {file_path.relative_to(output_dir)}")

if __name__ == "__main__":
    main()
