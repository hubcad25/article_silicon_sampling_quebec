import numpy as np
import polars as pl
import pytest

from article_silicon_sampling_quebec.metrics import (
    evaluate_distributions,
    evaluate_ses_subgroups,
    flattening_diagnostics,
    kl_divergence,
    total_variation,
)


def fake_analysis_tables():
    distribution_rows = [
        {"arm": "observed", "item_idx": 1, "cell": "c", "temperature": None,
         "code": "1", "share": 0.75, "n": 2},
        {"arm": "observed", "item_idx": 1, "cell": "c", "temperature": None,
         "code": "2", "share": 0.25, "n": 2},
    ]
    diagnostic_rows = []
    for arm, shares in (("A", (0.5, 0.5)), ("B", (0.75, 0.25)),
                        ("B0", (0.5, 0.5)), ("R", (0.25, 0.75)),
                        ("BS", (0.75, 0.25))):
        for code, share in zip(("1", "2"), shares, strict=True):
            distribution_rows.append({
                "arm": arm, "item_idx": 1, "cell": "c", "temperature": 0.7,
                "code": code, "share": share, "n": 4,
            })
        diagnostic_rows.append({
            "arm": arm, "item_idx": 1, "cell": "c", "temperature": 0.7,
            "invalid_rate": 0.0,
        })
    responses = pl.DataFrame({
        "item_idx": [1, 1], "cell": ["c", "c"], "respondent_id": ["a", "b"],
        "code": ["1", "2"], "weight": [3.0, 1.0],
    })
    return pl.DataFrame(distribution_rows), pl.DataFrame(diagnostic_rows), responses


def test_kl_smooths_only_the_model_counts():
    value = kl_divergence(np.array([1.0, 0.0]), np.array([0, 4]), alpha=0.5)
    assert value == pytest.approx(np.log(10))


def test_total_variation_has_expected_scale():
    assert total_variation(np.array([0.75, 0.25]), np.array([0.5, 0.5])) == 0.25


def test_evaluation_accepts_exact_statistical_probabilities():
    distributions, diagnostics, responses = fake_analysis_tables()
    statistical = pl.DataFrame([
        {"arm": "S", "item_idx": 1, "cell": "c", "temperature": 0.7,
         "code": "1", "share": 0.8, "n": None},
        {"arm": "S", "item_idx": 1, "cell": "c", "temperature": 0.7,
         "code": "2", "share": 0.2, "n": None},
    ]).with_columns(pl.col("n").cast(pl.Int64))
    distributions = pl.concat([distributions, statistical], how="vertical")
    diagnostics = pl.concat([diagnostics, pl.DataFrame([{
        "arm": "S", "item_idx": 1, "cell": "c", "temperature": 0.7,
        "invalid_rate": 0.0,
    }])], how="vertical")

    metrics, _, summaries, contrasts = evaluate_distributions(
        distributions, diagnostics, responses, repetitions=20, seed=4,
    )

    statistical_metric = metrics.filter(pl.col("arm") == "S").row(0, named=True)
    assert statistical_metric["model_n"] is None
    assert statistical_metric["tv"] == pytest.approx(0.05)
    expected_kl = 0.75 * np.log(0.75 / 0.8) + 0.25 * np.log(0.25 / 0.2)
    assert statistical_metric["kl"] == pytest.approx(expected_kl)
    assert summaries.filter(pl.col("arm") == "S").height == 1
    assert contrasts.filter(pl.col("contrast").str.ends_with(" - S")).height == 4


def test_evaluation_is_reproducible_and_pairs_all_requested_contrasts():
    tables = fake_analysis_tables()
    first = evaluate_distributions(*tables, repetitions=50, seed=7)
    second = evaluate_distributions(*tables, repetitions=50, seed=7)

    for left, right in zip(first, second, strict=True):
        assert left.equals(right)
    metrics, items, summaries, contrasts = first
    assert metrics.height == items.height == summaries.height == 5
    assert contrasts.height == 6
    assert metrics.filter(pl.col("arm") == "B")["kl"][0] < metrics.filter(
        pl.col("arm") == "A"
    )["kl"][0]
    assert contrasts.filter(pl.col("contrast") == "B - A")["mean_kl_difference"][0] < 0
    assert contrasts.filter(pl.col("contrast") == "B - A")["mean_tv_difference"][0] < 0


def test_flattening_keeps_all_invalid_groups_as_missing():
    distributions, _, _ = fake_analysis_tables()
    extra_cell = distributions.with_columns(
        pl.lit("d").alias("cell"),
        pl.when(pl.col("arm") == "observed").then(pl.lit(100)).otherwise(pl.col("n")).alias("n"),
    )
    distributions = pl.concat([distributions, extra_cell]).with_columns(
        pl.when(pl.col("arm") == "observed").then(pl.lit(100)).otherwise(pl.col("n")).alias("n"),
        pl.when(pl.col("arm") == "BS").then(pl.lit(None)).otherwise(pl.col("share")).alias("share"),
    )

    result = flattening_diagnostics(distributions)

    bs = result.filter(pl.col("arm") == "BS")
    assert bs["n_cells"].to_list() == [0, 0]
    assert bs["variance_ratio"].null_count() == 2


def test_summary_averages_cells_within_items_before_items():
    distribution_rows = []
    diagnostic_rows = []
    response_rows = []
    cells = {1: ["a"], 2: ["a", "b", "c"]}
    for item_idx, item_cells in cells.items():
        for cell in item_cells:
            for code, share in (("1", 1.0), ("2", 0.0)):
                distribution_rows.append({
                    "arm": "observed", "item_idx": item_idx, "cell": cell,
                    "temperature": None, "code": code, "share": share, "n": 1,
                })
            response_rows.append({
                "item_idx": item_idx, "cell": cell, "respondent_id": f"{item_idx}-{cell}",
                "code": "1", "weight": 1.0,
            })
            for arm in ("A", "B", "B0", "R", "BS"):
                model_shares = (1.0, 0.0) if item_idx == 1 else (0.0, 1.0)
                for code, share in zip(("1", "2"), model_shares, strict=True):
                    distribution_rows.append({
                        "arm": arm, "item_idx": item_idx, "cell": cell,
                        "temperature": 1.0, "code": code, "share": share, "n": 2,
                    })
                diagnostic_rows.append({
                    "arm": arm, "item_idx": item_idx, "cell": cell,
                    "temperature": 1.0, "invalid_rate": 0.0,
                })

    _, _, summaries, _ = evaluate_distributions(
        pl.DataFrame(distribution_rows), pl.DataFrame(diagnostic_rows),
        pl.DataFrame(response_rows), repetitions=10, seed=2,
    )

    summary = summaries.filter((pl.col("scope") == "all") & (pl.col("arm") == "A"))
    assert summary["mean_tv"][0] == 0.5


def test_ses_summary_preserves_item_weighting_and_available_contrasts():
    distributions, diagnostics, responses = fake_analysis_tables()
    metrics, _, _, _ = evaluate_distributions(
        distributions, diagnostics, responses, repetitions=10, seed=3,
    )
    attributes = pl.DataFrame({"item_idx": [1], "cell": ["c"], "gender": ["woman"]})

    summaries, contrasts = evaluate_ses_subgroups(
        metrics, attributes, dimensions=("gender",), temperature=0.7,
        repetitions=10, seed=3,
    )

    assert summaries.filter(pl.col("scope") == "all").height == 5
    assert contrasts.filter(pl.col("scope") == "all").height == 6
    assert set(summaries["level"]) == {"woman"}
