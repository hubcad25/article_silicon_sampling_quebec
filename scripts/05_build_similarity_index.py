#!/usr/bin/env python3
"""Build a semantic similarity index for RAG retrieval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from article_silicon_sampling_quebec.rag_similarity import build_similarity_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-k semantic neighbors for each target question"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Questions parquet with an embedding column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/rag_similarity.csv"),
        help="Output path for top-k neighbors (.csv or .parquet)",
    )
    parser.add_argument(
        "--target-split",
        type=str,
        default="test",
        help="Question split used as retrieval targets",
    )
    parser.add_argument(
        "--candidate-split",
        type=str,
        default="train",
        help="Question split used as retrieval candidates",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar neighbors retained per target",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if not args.questions.exists():
        raise SystemExit(f"Missing questions parquet: {args.questions}")

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"Output already exists: {args.output}. Use --force to overwrite."
        )

    logger.info("Loading questions from %s", args.questions)
    questions = pl.read_parquet(args.questions)
    if "embedding" not in questions.columns:
        raise SystemExit(
            "Questions parquet is missing 'embedding'. "
            "Run scripts/03_generate_question_embeddings.py first."
        )

    logger.info(
        "Building similarity index (target=%s, candidates=%s, top_k=%d)",
        args.target_split,
        args.candidate_split,
        args.top_k,
    )
    index = build_similarity_index(
        questions,
        target_split=args.target_split,
        candidate_split=args.candidate_split,
        top_k=args.top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        index.write_csv(args.output)
    elif args.output.suffix.lower() == ".parquet":
        index.write_parquet(args.output)
    else:
        raise SystemExit("Unsupported output format. Use .csv or .parquet")
    logger.info("Saved %d rows to %s", index.height, args.output)


if __name__ == "__main__":
    main()
