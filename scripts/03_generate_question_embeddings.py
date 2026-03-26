#!/usr/bin/env python3
"""Generate semantic embeddings for the CES 2021 questions."""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import polars as pl
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_REVISION = "936af83a2ecce5fe87a09109ff5cbcefe073173a"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "au", "aux", "avec", "be", "by", "ce", "cet", "cette", "ces",
    "comme", "dans", "de", "des", "do", "does", "du", "elle", "en", "est", "et", "etre", "for", "from",
    "had", "has", "have", "il", "in", "is", "it", "je", "la", "le", "les", "leur", "leurs", "lui", "mais",
    "me", "mes", "moi", "mon", "ne", "ni", "nos", "notre", "nous", "of", "on", "or", "ou", "par", "pas",
    "pour", "qu", "que", "qui", "sa", "sans", "se", "ses", "son", "sont", "sur", "that", "the", "these",
    "this", "to", "un", "une", "vos", "votre", "vous", "what", "which", "who", "whom", "why", "when",
    "where", "with", "would", "you", "your",
}

DISPLAY_CHOICE_RE = re.compile(r"display\s+this\s+choice:?", flags=re.IGNORECASE)


def strip_diacritics(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    cleaned = DISPLAY_CHOICE_RE.sub(" ", value)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" :;-\n\t")


def extract_content_terms(text: str) -> list[str]:
    normalized = strip_diacritics(text.lower())
    tokens = re.split(r"[^a-z0-9]+", normalized)

    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def build_embedding_input(base_text: str) -> str:
    terms = extract_content_terms(base_text)[:24]
    if not terms:
        return base_text
    return f"{base_text}\n\nkey_terms: {' '.join(terms)}"


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
    question = clean_text(record.get("question"))
    label = clean_text(record.get("label"))

    base_text = ""
    if label and question:
        if question.lower() in label.lower():
            base_text = label
        elif label.lower() in question.lower():
            base_text = question
        else:
            base_text = f"{label}. {question}"
    elif label:
        base_text = label
    elif question:
        base_text = question
    else:
        fallback = record.get("variable_name") or ""
        base_text = str(fallback).strip()

    return build_embedding_input(base_text)


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
