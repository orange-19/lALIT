#!/usr/bin/env python3
"""
Hugging Face Model Download Script
Downloads the lalit-03/gemma3-1b-summarization-finetuned model
"""

import os
import sys
from huggingface_hub import HfApi, snapshot_download

def download_model(repo_id="lalit-03/gemma3-1b-summarization-finetuned", cache_dir="./my-model"):
    """
    Download model from Hugging Face Hub with error handling
    """
    try:
        print(f"Fetching model info for: {repo_id}")
        api = HfApi()
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        
        print("\nModel files:")
        total_size_mb = 0
        for f in info.siblings:
            size_mb = f.size / (1024**2) if f.size else 0
            total_size_mb += size_mb
            print(f"  {f.rfilename}: {size_mb:.1f} MB")
        
        print(f"\nTotal size: {total_size_mb:.1f} MB")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"\nDownloading to: {os.path.abspath(cache_dir)}")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        print("\n✅ Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        return False

def main():
    """
    Main function
    """
    success = download_model()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
