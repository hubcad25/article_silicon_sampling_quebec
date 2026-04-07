#!/usr/bin/env python3
"""
Upload the processed datasets to a private Hugging Face dataset repo.

Usage:
    # Upload everything (shared files + all available finetune datasets)
    python scripts/upload_dataset_hf.py

    # Upload only condition 4A finetune dataset
    python scripts/upload_dataset_hf.py --condition 4a

    # Upload only condition 4B finetune dataset
    python scripts/upload_dataset_hf.py --condition 4b
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
token = os.getenv("HF_API_KEY")

if not token:
    logger.error("Missing HF_API_KEY in .env file.")
    sys.exit(1)

HF_USER = "hubcad25"
REPO_NAME = "article_silicon_sampling_quebec_data"
REPO_ID = f"{HF_USER}/{REPO_NAME}"

# Shared processed files always uploaded (unless --condition restricts scope)
SHARED_FILES = [
    "data/processed/questions.parquet",
    "data/processed/respondents.parquet",
]

# Finetune datasets keyed by condition
FINETUNE_FILES = {
    "4a": "data/processed/finetune_train.jsonl",
    "4b": "data/processed/finetune_train_4b.jsonl",
}

# Destination name in the HF repo for each finetune file
FINETUNE_REPO_NAMES = {
    "4a": "finetune_train.jsonl",
    "4b": "finetune_train_4b.jsonl",
}


def upload_file(api: HfApi, local_path: Path, repo_path: str) -> None:
    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(f"Uploading {local_path} -> {repo_path} ({size_mb:.1f} MB)...")
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload {repo_path}",
        )
        logger.info(f"✓ {repo_path} uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload {repo_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload processed datasets to HuggingFace")
    parser.add_argument(
        "--condition",
        choices=["4a", "4b"],
        default=None,
        help="Upload only the finetune dataset for a specific condition (default: all available)",
    )
    parser.add_argument(
        "--skip-shared",
        action="store_true",
        help="Skip uploading shared files (questions.parquet, respondents.parquet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi(token=token)

    logger.info(f"Ensuring private repo '{REPO_ID}' exists...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=token)
        logger.info("Repo is ready.")
    except Exception as e:
        logger.error(f"Failed to create/access repo: {e}")
        sys.exit(1)

    # Upload shared files unless restricted
    if not args.skip_shared:
        for file_path_str in SHARED_FILES:
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue
            upload_file(api, file_path, file_path.name)

    # Determine which finetune conditions to upload
    conditions = [args.condition] if args.condition else list(FINETUNE_FILES.keys())

    for condition in conditions:
        local_path = Path(FINETUNE_FILES[condition])
        repo_name = FINETUNE_REPO_NAMES[condition]
        if not local_path.exists():
            logger.warning(f"Finetune dataset for condition {condition} not found: {local_path}. Skipping.")
            continue
        upload_file(api, local_path, repo_name)

    logger.info("All uploads complete.")
    logger.info(f"Dataset repo: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
