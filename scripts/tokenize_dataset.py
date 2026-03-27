#!/usr/bin/env python3
"""
Pre-tokenize finetune_train.jsonl and push the tokenized dataset to HuggingFace.

Run this ONCE locally (CPU is fine — no GPU needed) before launching cloud training.
The tokenized dataset is pushed to HF and reused by finetune.py on any pod,
skipping the expensive tokenization step on the GPU machine.

Usage:
    source .venv/bin/activate
    pip install -e ".[finetune]"
    python scripts/tokenize_dataset.py

Output:
    Pushes tokenized dataset to: hubcad25/article_silicon_sampling_quebec_tokenized
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

HF_DATASET_ID_RAW = "hubcad25/article_silicon_sampling_quebec_data"
HF_DATASET_ID_TOKENIZED = "hubcad25/article_silicon_sampling_quebec_tokenized"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Same tokenizer as Llama-3 for local CPU run
DEFAULT_DATA = Path("data/processed/finetune_train.jsonl")
MAX_SEQ_LEN = 3072  # Generous ceiling — samples are ~2600 tokens
EVAL_SPLIT = 0.05
SEED = 42


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    hf_token = os.getenv("HF_API_KEY")
    if not hf_token:
        logger.error("Missing HF_API_KEY in .env")
        sys.exit(1)

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        logger.error('Run: pip install -e ".[finetune]"')
        sys.exit(1)

    # 1. Load raw dataset
    if DEFAULT_DATA.exists():
        logger.info("Loading from local file %s ...", DEFAULT_DATA)
        ds = load_dataset("json", data_files=str(DEFAULT_DATA), split="train")
    else:
        logger.info("Loading from HF repo %s ...", HF_DATASET_ID_RAW)
        ds = load_dataset(HF_DATASET_ID_RAW, data_files="finetune_train.jsonl", split="train")
    logger.info("Raw samples: %d", len(ds))

    # 2. Load tokenizer (TinyLlama = same vocab as Llama-3, runs on CPU)
    # On the GPU pod, the actual Llama-3.1-8B tokenizer will be used for inference —
    # but for pre-tokenization the vocab is identical so this is safe.
    logger.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Pre-format: input + output → text
    logger.info("Formatting samples (input + output -> text) ...")
    ds = ds.map(
        lambda x: {"text": x["input"] + x["output"]},
        remove_columns=["input", "output"],
        num_proc=4,
        desc="Formatting",
    )

    # 4. Tokenize
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    logger.info("Tokenizing %d samples (max_length=%d) ...", len(ds), MAX_SEQ_LEN)
    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    logger.info("Tokenization complete.")

    # 5. Shuffle + split
    ds = ds.shuffle(seed=SEED)
    split = ds.train_test_split(test_size=EVAL_SPLIT, seed=SEED)
    logger.info("Train: %d | Eval: %d", len(split["train"]), len(split["test"]))

    # 6. Push to HF
    logger.info("Pushing tokenized dataset to %s ...", HF_DATASET_ID_TOKENIZED)
    split.push_to_hub(
        HF_DATASET_ID_TOKENIZED,
        private=True,
        token=hf_token,
    )
    logger.info("Done. Dataset available at: https://huggingface.co/datasets/%s", HF_DATASET_ID_TOKENIZED)


if __name__ == "__main__":
    main()
