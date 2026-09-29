"""Derived methods and question-bootstrap summaries for the final 48-item test."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl


RAW_ARMS = ("A", "A20", "B020", "BR20")
METHODS = ("A", "A20", "B020", "BR20", "AVG3", "AVG4", "A_REG", "AVG4_REG")
METHOD_LABELS = {
    "A": "Entraîné 8k",
    "A20": "Entraîné 20k",
    "B020": "Indices retirés 20k",
    "BR20": "Avec répondants 20k",
    "AVG3": "Moyenne sans répondants",
    "AVG4": "Moyenne des quatre modèles",
    "A_REG": "Entraîné 8k régularisé",
    "AVG4_REG": "Moyenne régularisée",
}
KEYS = ("item_idx", "cell", "temperature", "code")


def average_arms(distributions: pl.DataFrame, arms: Sequence[str], name: str) -> pl.DataFrame:
    """Average arm probabilities option by option with equal arm weights."""
    selected = distributions.filter(pl.col("arm").is_in(arms)).sort(*KEYS, "arm")
    counts = selected.group_by(*KEYS, maintain_order=True).agg(
        pl.col("arm").n_unique().alias("n_arms")
    )
    if counts.filter(pl.col("n_arms") != len(arms)).height:
        raise ValueError(f"arms do not have identical option coverage: {arms}")
    return selected.group_by(*KEYS, maintain_order=True).agg(
        pl.col("share").mean().alias("share")
    ).with_columns(
        pl.lit(name).alias("arm"),
        pl.lit(None, dtype=pl.Int64).alias("n"),
    ).select("arm", *KEYS, "share", "n")


def regularize_between_cells(
    distributions: pl.DataFrame, arm: str, name: str, deviation_retained: float
) -> pl.DataFrame:
    """Retain a frozen fraction of each cell's deviation from its item mean."""
    if not 0 <= deviation_retained <= 1:
        raise ValueError("deviation_retained must be between zero and one")
    selected = distributions.filter(pl.col("arm") == arm).sort("item_idx", "code", "cell")
    if selected.is_empty():
        raise ValueError(f"missing arm {arm}")
    return selected.with_columns(
        pl.col("share").mean().over("item_idx", "code").alias("item_mean")
    ).with_columns(
        (
            pl.col("item_mean")
            + deviation_retained * (pl.col("share") - pl.col("item_mean"))
        ).alias("share"),
        pl.lit(name).alias("arm"),
        pl.lit(None, dtype=pl.Int64).alias("n"),
    ).drop("item_mean").select("arm", *KEYS, "share", "n")


def build_derived_methods(raw: pl.DataFrame) -> pl.DataFrame:
    """Build the four methods frozen before opening the final results."""
    if set(raw["arm"].unique()) != set(RAW_ARMS):
        raise ValueError(f"expected raw arms {RAW_ARMS}")
    avg3 = average_arms(raw, ("A", "A20", "B020"), "AVG3")
    avg4 = average_arms(raw, RAW_ARMS, "AVG4")
    a_reg = regularize_between_cells(raw, "A", "A_REG", deviation_retained=0.20)
    avg4_reg = regularize_between_cells(avg4, "AVG4", "AVG4_REG", deviation_retained=0.55)
    return pl.concat([avg3, avg4, a_reg, avg4_reg], how="vertical").sort(
        "arm", "item_idx", "cell", "code"
    )


def cell_metrics(
    observed: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    alpha: float = 0.5,
    derived_effective_n: int = 250,
) -> pl.DataFrame:
    """Compute TV and smoothed KL for every method × item × cell."""
    truth = observed.filter(pl.col("arm") == "observed").select(
        "item_idx", "cell", "code", pl.col("share").alias("observed_share"),
        pl.col("n").alias("observed_n"),
    )
    joined = predictions.join(
        truth, on=["item_idx", "cell", "code"], validate="m:1"
    ).sort("arm", "item_idx", "cell", "code")
    expected = predictions.height
    if joined.height != expected:
        raise ValueError("predictions and observed distributions have different option coverage")
    joined = joined.with_columns(
        pl.len().over("arm", "item_idx", "cell").alias("n_options"),
        pl.col("n").fill_null(derived_effective_n).cast(pl.Float64).alias("metric_n"),
    ).with_columns(
        (
            (pl.col("share") * pl.col("metric_n") + alpha)
            / (pl.col("metric_n") + alpha * pl.col("n_options"))
        ).alias("smoothed_share")
    )
    result = joined.group_by("arm", "item_idx", "cell", maintain_order=True).agg(
        (0.5 * (pl.col("share") - pl.col("observed_share")).abs().sum()).alias("tv"),
        pl.when(pl.col("observed_share") > 0)
        .then(pl.col("observed_share") * (pl.col("observed_share") / pl.col("smoothed_share")).log())
        .otherwise(0.0).sum().alias("kl"),
        pl.col("observed_n").first(),
    )
    has_null_metric = result.select(
        pl.col("tv").is_null().any() | pl.col("kl").is_null().any()
    ).item()
    if has_null_metric:
        raise ValueError("null metric produced")
    return result.sort("arm", "item_idx", "cell")


def item_metrics(cells: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Average cells within question and attach frozen question categories."""
    items = cells.sort("arm", "item_idx", "cell").group_by(
        "arm", "item_idx", maintain_order=True
    ).agg(
        pl.col("tv").mean().alias("mean_tv"),
        pl.col("kl").mean().alias("mean_kl"),
        pl.len().alias("n_cells"),
    )
    return items.join(
        metadata.select("item_idx", "block", "distance_bin"),
        on="item_idx", how="left", validate="m:1",
    ).sort("arm", "item_idx")


def summarize_methods(
    items: pl.DataFrame,
    *,
    baseline: str = "A",
    methods: Sequence[str] = METHODS,
    repetitions: int = 2_000,
    seed: int = 20_260_925,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Summarize methods and paired differences by bootstrapping questions."""
    wide_tv = items.select("item_idx", "arm", "mean_tv").pivot(
        on="arm", index="item_idx", values="mean_tv"
    ).sort("item_idx")
    wide_kl = items.select("item_idx", "arm", "mean_kl").pivot(
        on="arm", index="item_idx", values="mean_kl"
    ).sort("item_idx")
    missing = set(methods) - set(wide_tv.columns)
    if missing or baseline not in wide_tv.columns:
        raise ValueError(f"missing methods: {sorted(missing)}")
    if wide_tv.select(methods).null_count().row(0) != (0,) * len(methods):
        raise ValueError("methods do not cover identical questions")
    n_items = wide_tv.height
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_items, size=(repetitions, n_items))
    tail = (1 - confidence) / 2
    baseline_values = wide_tv[baseline].to_numpy()
    rows = []
    for method in methods:
        tv = wide_tv[method].to_numpy()
        kl = wide_kl[method].to_numpy()
        difference = tv - baseline_values
        tv_boot = tv[sampled].mean(axis=1)
        difference_boot = difference[sampled].mean(axis=1)
        rows.append({
            "arm": method,
            "label": METHOD_LABELS.get(method, method),
            "n_items": n_items,
            "n_cells": int(items.filter(pl.col("arm") == method)["n_cells"].sum()),
            "mean_tv": float(tv.mean()),
            "mean_tv_ci_low": float(np.quantile(tv_boot, tail)),
            "mean_tv_ci_high": float(np.quantile(tv_boot, 1 - tail)),
            "difference_vs_a": float(difference.mean()),
            "difference_ci_low": float(np.quantile(difference_boot, tail)),
            "difference_ci_high": float(np.quantile(difference_boot, 1 - tail)),
            "mean_kl": float(kl.mean()),
        })
    return pl.DataFrame(rows)


def category_breakdown(
    items: pl.DataFrame,
    selected: str,
    category: str,
    *,
    baseline: str = "A",
    labels: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Compare the globally selected method with A within frozen categories."""
    frame = items.filter(pl.col("arm").is_in([baseline, selected]))
    grouped = frame.group_by(category, "arm").agg(
        pl.col("mean_tv").mean().alias("mean_tv"),
        pl.len().alias("n_items"),
    ).pivot(on="arm", index=category, values=["mean_tv", "n_items"])
    result = grouped.select(
        pl.col(category),
        pl.col(f"n_items_{baseline}").alias("n_items"),
        pl.col(f"mean_tv_{baseline}").alias("baseline_tv"),
        pl.col(f"mean_tv_{selected}").alias("selected_tv"),
    ).with_columns(
        (pl.col("selected_tv") - pl.col("baseline_tv")).alias("difference")
    )
    if labels:
        result = result.with_columns(
            pl.col(category).replace_strict(labels, default=pl.col(category)).alias("label")
        )
    return result.sort(category)


__all__ = [
    "METHOD_LABELS", "METHODS", "RAW_ARMS", "average_arms", "build_derived_methods",
    "category_breakdown", "cell_metrics", "item_metrics", "regularize_between_cells",
    "summarize_methods",
]
