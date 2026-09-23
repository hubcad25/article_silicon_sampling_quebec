"""Unit tests for the item-to-item similarity index. No network calls."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from article_silicon_sampling_quebec.corpus.similarity import (
    SimilarityIndex,
    build_embed_text,
    build_embeddings,
    build_similarity_frame,
    cosine_similarity,
    l2_normalize,
    top_k_neighbors,
)

RNG = np.random.default_rng(1234)


@pytest.fixture
def items() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "survey_id": ["s1", "s1", "s1", "s2", "s2", "s3"],
            "variable": ["a", "b", "c", "a", "d", "e"],
            "question_text": [f"q{i}" for i in range(6)],
        }
    )


@pytest.fixture
def vectors() -> np.ndarray:
    # Deliberately not unit-norm, to check normalisation happens downstream.
    return RNG.normal(size=(6, 16)).astype(np.float32) * 3.7


# --------------------------------------------------------------------------
# Embed text
# --------------------------------------------------------------------------


def test_embed_text_follows_production_recipe():
    text = build_embed_text("Aimez-vous X ?", "Appréciation de X", ["Oui", "Non"], ["x"])
    assert text == "Aimez-vous X ?\nAppréciation de X\nOui\nNon\nx"


def test_embed_text_tolerates_missing_parts():
    assert build_embed_text("Q", None, None, None) == "Q"
    assert build_embed_text("Q", None, [None, "Oui"], []) == "Q\nOui"


def test_embed_text_disambiguates_truncated_battery_stems():
    """The 80-char Stata cap makes sub-items identical without display_label."""
    stem = "Here are some things people can do to participate in politics. Please indicate h"
    a = build_embed_text(stem, "Participation: Signed a petition", ["Never"], None)
    b = build_embed_text(stem, "Participation: Donated money", ["Never"], None)
    assert a != b


# --------------------------------------------------------------------------
# Cosine math
# --------------------------------------------------------------------------


def test_l2_normalize_gives_unit_rows_and_tolerates_zeros():
    matrix = np.vstack([RNG.normal(size=(4, 8)), np.zeros((1, 8))])
    unit = l2_normalize(matrix)
    assert np.allclose(np.linalg.norm(unit[:4], axis=1), 1.0, atol=1e-6)
    assert np.all(unit[4] == 0.0)


def test_cosine_similarity_is_symmetric(vectors):
    sims = cosine_similarity(vectors, vectors)
    assert np.allclose(sims, sims.T, atol=1e-6)


def test_cosine_similarity_is_bounded(vectors):
    sims = cosine_similarity(vectors, vectors)
    assert sims.min() >= -1.0 - 1e-6
    assert sims.max() <= 1.0 + 1e-6


def test_cosine_similarity_self_is_one(vectors):
    sims = cosine_similarity(vectors, vectors)
    assert np.allclose(np.diag(sims), 1.0, atol=1e-6)


def test_cosine_similarity_is_scale_invariant(vectors):
    assert np.allclose(
        cosine_similarity(vectors, vectors),
        cosine_similarity(vectors * 100.0, vectors * 0.01),
        atol=1e-5,
    )


def test_top_k_excludes_self(vectors):
    idx, _ = top_k_neighbors(vectors, k=3)
    for i, row in enumerate(idx):
        assert i not in row


def test_top_k_is_sorted_and_bounded(vectors):
    idx, sims = top_k_neighbors(vectors, k=5)
    assert idx.shape == sims.shape == (6, 5)
    assert np.all(np.diff(sims, axis=1) <= 1e-6)
    assert sims.min() >= -1.0 - 1e-6 and sims.max() <= 1.0 + 1e-6


def test_top_k_matches_brute_force(vectors):
    idx, sims = top_k_neighbors(vectors, k=2, block=2)
    full = cosine_similarity(vectors, vectors)
    np.fill_diagonal(full, -np.inf)
    for i in range(vectors.shape[0]):
        expected = np.argsort(-full[i])[:2]
        assert list(idx[i]) == list(expected)
        assert np.allclose(sims[i], full[i][expected], atol=1e-6)


def test_top_k_caps_at_n_minus_one(vectors):
    idx, _ = top_k_neighbors(vectors, k=1000)
    assert idx.shape[1] == vectors.shape[0] - 1


def test_identical_vectors_stay_visible_to_each_other():
    """Self-exclusion is positional; true duplicates must remain findable."""
    matrix = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
    idx, sims = top_k_neighbors(matrix, k=2)
    assert np.allclose(sims, 1.0, atol=1e-6)
    for i, row in enumerate(idx):
        assert i not in row


# --------------------------------------------------------------------------
# ItemEmbeddings
# --------------------------------------------------------------------------


def test_build_embeddings_rejects_length_mismatch(items):
    with pytest.raises(ValueError):
        build_embeddings(items, RNG.normal(size=(5, 4)))


def test_build_embeddings_rejects_duplicate_keys():
    dup = pl.DataFrame(
        {"survey_id": ["s1", "s1"], "variable": ["a", "a"], "question_text": ["x", "y"]}
    )
    with pytest.raises(ValueError):
        build_embeddings(dup, RNG.normal(size=(2, 4)))


def test_same_variable_in_two_surveys_is_two_items(items, vectors):
    """('s1','a') and ('s2','a') must not collide — the key is the pair."""
    emb = build_embeddings(items, vectors)
    assert emb.position(("s1", "a")) != emb.position(("s2", "a"))


def test_unknown_key_raises(items, vectors):
    emb = build_embeddings(items, vectors)
    with pytest.raises(KeyError):
        emb.vector(("nope", "nope"))


def test_similarity_to_set_drops_the_item_itself(items, vectors):
    emb = build_embeddings(items, vectors)
    others = [("s1", "a"), ("s1", "b"), ("s2", "d")]
    sims = emb.similarity_to_set(("s1", "a"), others)
    assert sims.size == 2
    assert sims.max() <= 1.0 + 1e-6 and sims.min() >= -1.0 - 1e-6


def test_distance_to_set_matches_one_minus_max_cosine(items, vectors):
    emb = build_embeddings(items, vectors)
    others = [("s1", "b"), ("s1", "c"), ("s2", "d")]
    expected = 1.0 - emb.similarity_to_set(("s1", "a"), others).max()
    assert emb.distance_to_set(("s1", "a"), others) == pytest.approx(expected)


def test_distance_to_a_set_containing_only_itself_is_maximal(items, vectors):
    emb = build_embeddings(items, vectors)
    assert emb.distance_to_set(("s1", "a"), [("s1", "a")]) == 1.0
    assert emb.distance_to_set(("s1", "a"), []) == 1.0


def test_distance_to_set_is_bounded(items, vectors):
    emb = build_embeddings(items, vectors)
    others = emb.keys
    for key in emb.keys:
        assert 0.0 <= emb.distance_to_set(key, others) <= 2.0


def test_nearest_in_set_agrees_with_distance(items, vectors):
    emb = build_embeddings(items, vectors)
    others = [("s1", "b"), ("s2", "d"), ("s3", "e")]
    key, sim = emb.nearest_in_set(("s1", "a"), others)
    assert key in others
    assert 1.0 - sim == pytest.approx(emb.distance_to_set(("s1", "a"), others))


# --------------------------------------------------------------------------
# SimilarityIndex
# --------------------------------------------------------------------------


def test_similarity_frame_shape_and_columns(items, vectors):
    frame = build_similarity_frame(items, vectors, k=3)
    assert frame.columns == [
        "survey_id",
        "variable",
        "rank",
        "neighbor_survey_id",
        "neighbor_variable",
        "cosine",
        "same_survey",
    ]
    assert frame.height == items.height * 3
    assert frame["cosine"].max() <= 1.0 + 1e-6
    assert frame["cosine"].min() >= -1.0 - 1e-6
    assert sorted(frame["rank"].unique().to_list()) == [1, 2, 3]


def test_similarity_frame_never_lists_the_item_itself(items, vectors):
    frame = build_similarity_frame(items, vectors, k=5)
    self_rows = frame.filter(
        (pl.col("survey_id") == pl.col("neighbor_survey_id"))
        & (pl.col("variable") == pl.col("neighbor_variable"))
    )
    assert self_rows.height == 0


def test_same_survey_flag_is_correct(items, vectors):
    frame = build_similarity_frame(items, vectors, k=5)
    assert frame["same_survey"].to_list() == [
        a == b
        for a, b in zip(frame["survey_id"], frame["neighbor_survey_id"], strict=True)
    ]


def test_similarity_is_symmetric_at_full_k(items, vectors):
    """With k = n-1 the table holds every pair; cosine(a,b) must equal cosine(b,a)."""
    frame = build_similarity_frame(items, vectors, k=items.height - 1)
    flipped = frame.select(
        pl.col("neighbor_survey_id").alias("survey_id"),
        pl.col("neighbor_variable").alias("variable"),
        pl.col("survey_id").alias("neighbor_survey_id"),
        pl.col("variable").alias("neighbor_variable"),
        pl.col("cosine").alias("cosine_rev"),
    )
    joined = frame.join(
        flipped,
        on=["survey_id", "variable", "neighbor_survey_id", "neighbor_variable"],
        how="inner",
    )
    assert joined.height == frame.height
    assert np.allclose(joined["cosine"], joined["cosine_rev"], atol=1e-6)


def test_neighbors_are_sorted_best_first(items, vectors):
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=4))
    out = index.neighbors(("s1", "a"))
    assert out["cosine"].to_list() == sorted(out["cosine"].to_list(), reverse=True)


def test_neighbors_k_truncates(items, vectors):
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=5))
    assert index.neighbors(("s1", "a"), k=2).height == 2


def test_exclude_same_survey_drops_same_survey_neighbours(items, vectors):
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=5))
    out = index.neighbors(("s1", "a"), exclude_same_survey=True)
    assert out.height > 0
    assert not any(out["same_survey"].to_list())
    assert set(out["neighbor_survey_id"].to_list()) <= {"s2", "s3"}


def test_nearest_respects_same_survey_exclusion(items, vectors):
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=5))
    cross = index.nearest(("s1", "a"), exclude_same_survey=True)
    assert cross is not None
    assert cross[0][0] != "s1"


def test_nearest_returns_none_for_unknown_item(items, vectors):
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=3))
    assert index.nearest(("ghost", "x")) is None


def test_index_items_and_k(items, vectors):
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=3))
    assert index.k == 3
    assert len(index.items()) == items.height


def test_index_nearest_matches_embeddings_distance(items, vectors):
    """The materialised index and the on-the-fly path must agree."""
    emb = build_embeddings(items, vectors)
    index = SimilarityIndex(build_similarity_frame(items, vectors, k=items.height - 1))
    key = ("s1", "a")
    others = [k for k in emb.keys if k != key]
    _, sim = index.nearest(key)
    assert 1.0 - sim == pytest.approx(emb.distance_to_set(key, others), abs=1e-5)
