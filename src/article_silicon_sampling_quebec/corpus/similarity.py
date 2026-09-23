"""Item-to-item semantic similarity index — step 1.5 of phase 1.

Backs the "semantic distance" axis of the paper (§2.3 of docs/plan_article.md):
for any target item, how far is it from a given set of items (typically the
training corpus)?

Two artefacts on disk, both produced by `scripts/12_build_similarity_index.py`:

    data/item_embeddings.parquet   survey_id, variable, question_text, embedding
    data/item_similarity.parquet   one row per (item, neighbour) pair, k rows/item

This module holds the pure logic (no network, no Azure): the embed-text recipe,
cosine math, and the two loaders. Everything here is importable in tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Default artefact locations, relative to the repo root.
EMBEDDINGS_PATH = Path("data/item_embeddings.parquet")
SIMILARITY_PATH = Path("data/item_similarity.parquet")

# Neighbours kept per item in the materialised index.
DEFAULT_K = 50

ItemKey = tuple[str, str]  # (survey_id, variable)


# --------------------------------------------------------------------------
# Embed text
# --------------------------------------------------------------------------


def build_embed_text(
    question_text: str,
    display_label: str | None = None,
    option_labels: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
) -> str:
    """Text fed to the embedding model for one item.

    Mirrors `ingestion/build_docs.py::embed_text` in the search-engine repo, so
    our vectors live in the same space as the ones already serving production.
    Order matters only for readability; the model sees a single string.
    """
    parts: list[str] = [question_text or ""]
    if display_label:
        parts.append(display_label)
    for label in option_labels or ():
        if label:
            parts.append(label)
    if concepts:
        parts.append(", ".join(concepts))
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Cosine math (pure numpy, no I/O)
# --------------------------------------------------------------------------


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left at zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of `a` and every row of `b`."""
    return l2_normalize(a) @ l2_normalize(b).T


def top_k_neighbors(
    matrix: np.ndarray,
    k: int = DEFAULT_K,
    block: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours of every row, self excluded.

    Returns (indices, similarities), both (n, k), sorted by decreasing
    similarity. Computed in row blocks so the full n x n matrix is never
    materialised.
    """
    unit = l2_normalize(matrix)
    n = unit.shape[0]
    k = min(k, max(n - 1, 0))
    idx_out = np.empty((n, k), dtype=np.int32)
    sim_out = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = unit[start:stop] @ unit.T
        # Self-exclusion by position, not by value: two identical wordings can
        # legitimately sit at cosine 1.0 and must stay visible to each other.
        rows = np.arange(stop - start)
        sims[rows, np.arange(start, stop)] = -np.inf

        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        idx_out[start:stop] = np.take_along_axis(part, order, axis=1)
        sim_out[start:stop] = np.take_along_axis(part_sims, order, axis=1)

    return idx_out, sim_out


# --------------------------------------------------------------------------
# Embedding store
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemEmbeddings:
    """Items of the perimeter plus their unit-norm embedding matrix."""

    items: pl.DataFrame  # survey_id, variable, question_text
    matrix: np.ndarray  # (n, d), L2-normalised
    _index: dict[ItemKey, int]

    @property
    def keys(self) -> list[ItemKey]:
        return list(self._index)

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def position(self, key: ItemKey) -> int:
        try:
            return self._index[key]
        except KeyError:
            raise KeyError(f"unknown item: {key}") from None

    def vector(self, key: ItemKey) -> np.ndarray:
        return self.matrix[self.position(key)]

    def question_text(self, key: ItemKey) -> str:
        return self.items["question_text"][self.position(key)]

    def similarity_to_set(
        self,
        key: ItemKey,
        others: Iterable[ItemKey],
    ) -> np.ndarray:
        """Cosine similarity of one item to each item of `others`, in order.

        `key` is dropped from `others` if present: an item is never its own
        neighbour, including when measuring distance to a corpus it belongs to.
        """
        target = self.vector(key)
        positions = [self.position(o) for o in others if o != key]
        if not positions:
            return np.empty(0, dtype=np.float32)
        return (self.matrix[positions] @ target).astype(np.float32)

    def distance_to_set(
        self,
        key: ItemKey,
        others: Iterable[ItemKey],
    ) -> float:
        """Cosine distance (1 - max cosine) from one item to a set of items.

        This is the quantity plotted on the x-axis of the coverage-performance
        curve (§2.3): distance from a test item to its nearest neighbour in the
        training corpus. Returns 1.0 (maximally far) when the set is empty.
        """
        sims = self.similarity_to_set(key, others)
        if sims.size == 0:
            return 1.0
        return float(1.0 - sims.max())

    def nearest_in_set(
        self,
        key: ItemKey,
        others: Iterable[ItemKey],
    ) -> tuple[ItemKey | None, float]:
        """Closest member of `others` and its cosine similarity."""
        candidates = [o for o in others if o != key]
        sims = self.similarity_to_set(key, candidates)
        if sims.size == 0:
            return None, float("-inf")
        best = int(np.argmax(sims))
        return candidates[best], float(sims[best])


def build_embeddings(
    items: pl.DataFrame,
    vectors: np.ndarray,
) -> ItemEmbeddings:
    """Assemble an `ItemEmbeddings` from an item table and a raw matrix."""
    required = {"survey_id", "variable", "question_text"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items is missing columns: {sorted(missing)}")
    if items.height != np.asarray(vectors).shape[0]:
        raise ValueError("items and vectors have different lengths")

    keys = list(zip(items["survey_id"], items["variable"], strict=True))
    index = {key: i for i, key in enumerate(keys)}
    if len(index) != len(keys):
        raise ValueError("duplicate (survey_id, variable) in items")

    return ItemEmbeddings(
        items=items.select("survey_id", "variable", "question_text"),
        matrix=l2_normalize(vectors),
        _index=index,
    )


def load_embeddings(path: Path | str = EMBEDDINGS_PATH) -> ItemEmbeddings:
    """Load `item_embeddings.parquet`."""
    frame = pl.read_parquet(path)
    vectors = np.asarray(frame["embedding"].to_list(), dtype=np.float32)
    return build_embeddings(frame.drop("embedding"), vectors)


# --------------------------------------------------------------------------
# Similarity index
# --------------------------------------------------------------------------

SIMILARITY_COLUMNS = [
    "survey_id",
    "variable",
    "rank",
    "neighbor_survey_id",
    "neighbor_variable",
    "cosine",
    "same_survey",
]


def build_similarity_frame(
    items: pl.DataFrame,
    vectors: np.ndarray,
    k: int = DEFAULT_K,
) -> pl.DataFrame:
    """Materialise the k-nearest-neighbour table from items + vectors."""
    emb = build_embeddings(items, vectors)
    idx, sims = top_k_neighbors(emb.matrix, k=k)
    n, kk = idx.shape

    survey = np.asarray(items["survey_id"].to_list(), dtype=object)
    variable = np.asarray(items["variable"].to_list(), dtype=object)

    frame = pl.DataFrame(
        {
            "survey_id": np.repeat(survey, kk),
            "variable": np.repeat(variable, kk),
            "rank": np.tile(np.arange(1, kk + 1, dtype=np.int32), n),
            "neighbor_survey_id": survey[idx.ravel()],
            "neighbor_variable": variable[idx.ravel()],
            "cosine": sims.ravel().astype(np.float32),
        }
    )
    return frame.with_columns(
        (pl.col("survey_id") == pl.col("neighbor_survey_id")).alias("same_survey")
    ).select(SIMILARITY_COLUMNS)


@dataclass(frozen=True)
class SimilarityIndex:
    """Materialised k-NN index, queried by (survey_id, variable)."""

    frame: pl.DataFrame

    @property
    def k(self) -> int:
        return int(self.frame["rank"].max() or 0)

    def items(self) -> list[ItemKey]:
        unique = self.frame.select("survey_id", "variable").unique(maintain_order=True)
        return list(zip(unique["survey_id"], unique["variable"], strict=True))

    def neighbors(
        self,
        key: ItemKey,
        k: int | None = None,
        exclude_same_survey: bool = False,
    ) -> pl.DataFrame:
        """k nearest neighbours of one item, best first.

        `exclude_same_survey=True` drops neighbours coming from the same survey
        — the honest cross-survey view, since a same-survey neighbour is often
        the same battery reworded rather than an independent item.
        """
        survey_id, variable = key
        out = self.frame.filter(
            (pl.col("survey_id") == survey_id) & (pl.col("variable") == variable)
        )
        if exclude_same_survey:
            out = out.filter(~pl.col("same_survey"))
        out = out.sort("cosine", descending=True)
        if k is not None:
            out = out.head(k)
        return out

    def nearest(
        self,
        key: ItemKey,
        exclude_same_survey: bool = False,
    ) -> tuple[ItemKey, float] | None:
        """Single nearest neighbour, or None if the index holds none."""
        row = self.neighbors(key, k=1, exclude_same_survey=exclude_same_survey)
        if row.height == 0:
            return None
        return (
            (row["neighbor_survey_id"][0], row["neighbor_variable"][0]),
            float(row["cosine"][0]),
        )


def load_index(path: Path | str = SIMILARITY_PATH) -> SimilarityIndex:
    """Load `item_similarity.parquet`."""
    return SimilarityIndex(pl.read_parquet(path))
