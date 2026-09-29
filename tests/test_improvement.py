from __future__ import annotations

import polars as pl
import pytest

from article_silicon_sampling_quebec.improvement import (
    average_arms,
    regularize_between_cells,
    summarize_methods,
)
from article_silicon_sampling_quebec.prompts import ItemSpec, Option


def distributions() -> pl.DataFrame:
    rows = []
    values = {
        "A": {(1, "x"): (0.8, 0.2), (1, "y"): (0.4, 0.6)},
        "A20": {(1, "x"): (0.6, 0.4), (1, "y"): (0.2, 0.8)},
    }
    for arm, pairs in values.items():
        for (item_idx, cell), shares in pairs.items():
            for code, share in zip(("1", "2"), shares, strict=True):
                rows.append({
                    "arm": arm, "item_idx": item_idx, "cell": cell,
                    "temperature": 1.0, "code": code, "share": share, "n": 10,
                })
    return pl.DataFrame(rows)


def test_average_arms_uses_equal_probability_weights():
    result = average_arms(distributions(), ("A", "A20"), "AVG")
    value = result.filter(
        (pl.col("cell") == "x") & (pl.col("code") == "1")
    )["share"][0]
    assert value == pytest.approx(0.7)


def test_regularization_retains_frozen_fraction_of_cell_deviation():
    result = regularize_between_cells(distributions(), "A", "A_REG", 0.25)
    # Item mean for code 1 is .6; x's deviation is +.2, of which .05 remains.
    value = result.filter(
        (pl.col("cell") == "x") & (pl.col("code") == "1")
    )["share"][0]
    assert value == pytest.approx(0.65)
    sums = result.group_by("item_idx", "cell").agg(pl.col("share").sum())
    assert sums["share"].to_list() == pytest.approx([1.0, 1.0])


def test_method_summary_is_question_weighted_and_paired():
    rows = []
    for arm, values in {"A": (0.2, 0.4), "X": (0.1, 0.2)}.items():
        for item_idx, value in enumerate(values):
            rows.append({
                "arm": arm, "item_idx": item_idx, "mean_tv": value,
                "mean_kl": value * 2, "n_cells": 3, "block": "b",
                "distance_bin": "d",
            })
    summary = summarize_methods(
        pl.DataFrame(rows), methods=("A", "X"), repetitions=100, seed=4
    )
    x = summary.filter(pl.col("arm") == "X").row(0, named=True)
    assert x["mean_tv"] == pytest.approx(0.15)
    assert x["difference_vs_a"] == pytest.approx(-0.15)


def test_canonical_code_recovers_unique_zero_padded_modality():
    item = ItemSpec(
        survey_id="survey", variable="q", text="Question",
        options=(Option("01", "First"), Option("02", "Second")), language="fr",
    )
    assert item.canonical_code("1") == "01"
    assert item.canonical_code("02") == "02"
