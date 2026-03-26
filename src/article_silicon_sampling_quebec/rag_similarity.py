"""Utilities for semantic similarity search over CES questions."""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl


def _embedding_from_value(value: Any) -> list[float] | None:
    if value is None:
        return None

    if isinstance(value, list):
        return [float(x) for x in value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        return None

    return None


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must share the same dimension")

    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot = sum(lx * rx for lx, rx in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _rows_with_embeddings(df: pl.DataFrame) -> list[dict[str, Any]]:
    required = {"variable_name", "split", "embedding"}
    missing = required.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in questions table: {missing_list}")

    rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        embedding = _embedding_from_value(row.get("embedding"))
        if embedding:
            row_copy = dict(row)
            row_copy["embedding"] = embedding
            rows.append(row_copy)
    return rows


def build_similarity_index(
    questions: pl.DataFrame,
    *,
    target_split: str = "test",
    candidate_split: str = "train",
    top_k: int = 5,
    exclude_same_variable: bool = True,
) -> pl.DataFrame:
    """Build top-k nearest semantic neighbors per target question."""
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    rows = _rows_with_embeddings(questions)
    targets = [r for r in rows if r.get("split") == target_split]
    candidates = [r for r in rows if r.get("split") == candidate_split]

    if not targets:
        raise ValueError(f"No target rows found for split='{target_split}'")
    if not candidates:
        raise ValueError(f"No candidate rows found for split='{candidate_split}'")

    out_rows: list[dict[str, Any]] = []
    for target in targets:
        scored: list[tuple[str, float, str | None]] = []
        target_var = target["variable_name"]
        target_embedding = target["embedding"]

        for candidate in candidates:
            candidate_var = candidate["variable_name"]
            if exclude_same_variable and target_var == candidate_var:
                continue

            score = cosine_similarity(target_embedding, candidate["embedding"])
            scored.append(
                (
                    candidate_var,
                    score,
                    candidate.get("thematic_domain"),
                )
            )

        scored.sort(key=lambda item: item[1], reverse=True)
        for rank, (neighbor_var, score, neighbor_domain) in enumerate(
            scored[:top_k],
            start=1,
        ):
            out_rows.append(
                {
                    "target_variable_name": target_var,
                    "target_split": target_split,
                    "target_thematic_domain": target.get("thematic_domain"),
                    "neighbor_rank": rank,
                    "neighbor_variable_name": neighbor_var,
                    "neighbor_split": candidate_split,
                    "neighbor_thematic_domain": neighbor_domain,
                    "cosine_similarity": score,
                }
            )

    return pl.DataFrame(out_rows).sort(
        ["target_variable_name", "neighbor_rank"]
    )
