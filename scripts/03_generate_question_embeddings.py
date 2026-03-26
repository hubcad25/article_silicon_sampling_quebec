#!/usr/bin/env python3
"""Generate semantic embeddings for the CES 2021 questions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import polars as pl
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_REVISION = "936af83a2ecce5fe87a09109ff5cbcefe073173a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add sentence-transformers embeddings to the questions parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Source parquet with question metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Target parquet path that receives the embedding column",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name (HF alias or local path)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Hugging Face model revision (commit SHA)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size passed to the sentence-transformers encoder",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings if the column already exists",
    )
    return parser.parse_args()


def best_text(record: dict[str, Any]) -> str:
    for key in ("question", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fallback = record.get("variable_name") or ""
    return fallback.strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input parquet: {args.input}")

    logger.info("Loading question metadata...")
    df = pl.read_parquet(args.input)

    if "embedding" in df.columns and not args.force:
        raise SystemExit(
            "Embedding column already exists. Use --force to recompute."
        )

    logger.info("Preparing embedding texts...")
    records = df.to_dicts()
    texts = [best_text(record) for record in records]

    logger.info("Loading %s model (revision=%s)...", args.model, args.revision)
    model = SentenceTransformer(args.model, revision=args.revision)

    logger.info("Computing embeddings for %d questions...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    logger.info("Attaching embeddings to dataframe...")
    embedding_values = [emb.astype("float32").tolist() for emb in embeddings]
    if "embedding" in df.columns:
        df = df.drop("embedding")
    df = df.with_columns(pl.Series("embedding", embedding_values))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    logger.info("Saved enriched questions parquet to %s", args.output)


if __name__ == "__main__":
    main()
