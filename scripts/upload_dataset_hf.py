#!/usr/bin/env python3
"""
Upload the processed datasets (including the 2.2GB JSONL) to a private Hugging Face dataset repo.
This allows us to run stateless RunPod instances without needing a Network Volume.

Usage:
    python scripts/upload_dataset_hf.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Load token from .env (without printing it!)
load_dotenv()
token = os.getenv("HF_API_KEY")

if not token:
    logger.error("Missing HF_API_KEY in .env file.")
    sys.exit(1)

# Target repository details
HF_USER = "hubcad25"
REPO_NAME = "article_silicon_sampling_quebec_data"
REPO_ID = f"{HF_USER}/{REPO_NAME}"

# Files to upload
FILES_TO_UPLOAD = [
    "data/processed/finetune_train.jsonl",
    "data/processed/questions.parquet",
    "data/processed/respondents.parquet",
]

def main():
    api = HfApi(token=token)

    # 1. Ensure the repo exists (and is private)
    logger.info(f"Ensuring private repo '{REPO_ID}' exists...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=token)
        logger.info("Repo is ready.")
    except Exception as e:
        logger.error(f"Failed to create/access repo: {e}")
        sys.exit(1)

    # 2. Upload the files
    for file_path_str in FILES_TO_UPLOAD:
        file_path = Path(file_path_str)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}. Skipping.")
            continue

        size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Uploading {file_path.name} ({size_mb:.1f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {file_path.name}",
            )
            logger.info(f"✓ {file_path.name} uploaded successfully.")
        except Exception as e:
            logger.error(f"Failed to upload {file_path.name}: {e}")

    logger.info("All uploads complete.")
    logger.info(f"You can view your dataset at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
