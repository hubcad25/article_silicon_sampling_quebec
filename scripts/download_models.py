#!/usr/bin/env python3
"""
Pre-download models from Hugging Face for offline use on compute nodes.
Run this on the login node (which has internet access).
"""

import argparse
import os
from huggingface_hub import snapshot_download

MODEL_MAP = {
    "0.5b": "unsloth/Qwen2.5-0.5B-bnb-4bit",
    "1b": "unsloth/Llama-3.2-1B-bnb-4bit",
    "8b": "unsloth/Llama-3.1-8B-bnb-4bit",
    "70b": "unsloth/Llama-3.1-70B-bnb-4bit",
}

def main():
    parser = argparse.ArgumentParser(description="Pre-download models for offline SFT")
    parser.add_argument("--size", type=str, choices=["0.5b", "1b", "8b", "70b", "all"], default="all")
    parser.add_argument("--token", type=str, help="HF Token (required for Llama models)", default=os.environ.get("HF_TOKEN"))
    
    args = parser.parse_args()
    
    sizes = ["0.5b", "1b", "8b", "70b"] if args.size == "all" else [args.size]
    
    for size in sizes:
        model_id = MODEL_MAP[size]
        print(f"Downloading {model_id} ...")
        snapshot_download(
            repo_id=model_id,
            token=args.token,
            # We skip heavy files not needed by Unsloth if any, 
            # but usually snapshot_download is safest.
        )
        print(f"Successfully cached {model_id}")

if __name__ == "__main__":
    main()
