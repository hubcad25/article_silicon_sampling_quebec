"""Distributional metrics for the frozen five-arm inference pilot."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import polars as pl


DEFAULT_ALPHA = 0.5
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20_260_925
DEFAULT_CONFIDENCE = 0.95
PRIMARY_TEMPERATURE = 1.0
MIN_CELL_N = 30
FLATTENING_MIN_CELL_N = 100
CONTRASTS = (
    ("B", "A"),
    ("B", "B0"),
    ("BS", "A"),
    ("BS", "B0"),
    ("B", "BS"),
    ("A", "R"),
    ("A", "S"),
    ("B0", "S"),
    ("B", "S"),
    ("BS", "S"),
)


def total_variation(observed: np.ndarray, model: np.ndarray) -> float:
    """Return total variation distance between two discrete distributions."""
    if observed.shape != model.shape:
        raise ValueError("observed and model distributions must have the same shape")
    return float(0.5 * np.abs(observed - model).sum())


def kl_divergence(observed: np.ndarray, model_counts: np.ndarray,
                  *, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute KL(observed || model) after additive smoothing of model counts."""
    if alpha <= 0 or model_counts.sum() <= 0:
        raise ValueError("alpha and the model effective sample size must be positive")
    model = (model_counts + alpha) / (model_counts.sum() + alpha * len(model_counts))
    positive = observed > 0
    return float(np.sum(observed[positive] * np.log(observed[positive] / model[positive])))


def _model_counts(shares: Sequence[float | None], n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(len(shares), dtype=np.int64)
    values = np.asarray(shares, dtype=float)
    counts = np.rint(values * n).astype(np.int64)
    if counts.sum() != n or np.any(counts < 0):
        raise ValueError(f"model shares do not reconstruct {n} integer draws")
    return counts


def _observed_bootstrap(codes: np.ndarray, weights: np.ndarray, option_count: int,
                        repetitions: int, rng: np.random.Generator) -> np.ndarray:
    sampled = rng.integers(0, len(codes), size=(repetitions, len(codes)))
    sampled_codes = codes[sampled]
    sampled_weights = weights[sampled]
    masses = np.column_stack([
        np.where(sampled_codes == option, sampled_weights, 0.0).sum(axis=1)
        for option in range(option_count)
    ])
    totals = masses.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("an observed bootstrap replicate has zero weight")
    return masses / totals[:, None]


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


def evaluate_distributions(
    distributions: pl.DataFrame,
    diagnostics: pl.DataFrame,
    observed_responses: pl.DataFrame,
    *,
    repetitions: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
    min_cell_n: int = MIN_CELL_N,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return cell metrics, item means, arm summaries, and paired contrasts."""
    if repetitions < 1 or not 0 < confidence < 1 or min_cell_n < 1:
        raise ValueError("invalid bootstrap, confidence, or cell-size parameter")
    required_distributions = {"arm", "item_idx", "cell", "temperature", "code", "share", "n"}
    required_diagnostics = {"arm", "item_idx", "cell", "temperature", "invalid_rate"}
    required_responses = {"item_idx", "cell", "code", "weight"}
    for name, frame, required in (
        ("distributions", distributions, required_distributions),
        ("diagnostics", diagnostics, required_diagnostics),
        ("observed_responses", observed_responses, required_responses),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    distributions = distributions.with_columns(pl.col("code").cast(pl.Utf8))
    observed_responses = observed_responses.with_columns(pl.col("code").cast(pl.Utf8))
    observed = distributions.filter(pl.col("arm") == "observed")
    model = distributions.filter(pl.col("arm") != "observed")
    if observed.is_empty() or model.is_empty():
        raise ValueError("both observed and model distributions are required")

    rng = np.random.default_rng(seed)
    observed_by_pair: dict[tuple[int, str], tuple[list[str], np.ndarray, int]] = {}
    observed_boot: dict[tuple[int, str], np.ndarray] = {}
    for key, group in observed.group_by("item_idx", "cell", maintain_order=True):
        item_idx, cell = int(key[0]), str(key[1])
        group = group.sort("code")
        options = group["code"].to_list()
        response_group = observed_responses.filter(
            (pl.col("item_idx") == item_idx) & (pl.col("cell") == cell)
        )
        option_index = {code: index for index, code in enumerate(options)}
        try:
            codes = np.asarray([option_index[code] for code in response_group["code"]], dtype=int)
        except KeyError as exc:
            raise ValueError(f"observed response has an unknown option for item {item_idx}") from exc
        weights = response_group["weight"].to_numpy().astype(float)
        observed_n = int(group["n"][0])
        if len(codes) != observed_n:
            raise ValueError(f"observed response count mismatch for item {item_idx}, cell {cell}")
        observed_by_pair[(item_idx, cell)] = (options, group["share"].to_numpy(), observed_n)
        observed_boot[(item_idx, cell)] = _observed_bootstrap(
            codes, weights, len(options), repetitions, rng
        )

    invalid_rates = {
        (str(arm), int(item_idx), str(cell), float(temperature)): float(invalid_rate)
        for arm, item_idx, cell, temperature, invalid_rate in diagnostics.select(
            "arm", "item_idx", "cell", "temperature", "invalid_rate"
        ).iter_rows()
    }
    metric_rows: list[dict] = []
    bootstraps: dict[tuple[str, int, str, float, str], np.ndarray] = {}
    for key, group in model.group_by(
        "arm", "item_idx", "cell", "temperature", maintain_order=True
    ):
        arm, item_idx, cell, temperature = str(key[0]), int(key[1]), str(key[2]), float(key[3])
        pair_key = (item_idx, cell)
        if pair_key not in observed_by_pair:
            raise ValueError(f"missing observed distribution for item {item_idx}, cell {cell}")
        options, observed_shares, observed_n = observed_by_pair[pair_key]
        group = group.sort("code")
        if group["code"].to_list() != options:
            raise ValueError(f"option mismatch for arm {arm}, item {item_idx}, cell {cell}")
        if group["n"].n_unique() != 1:
            raise ValueError(f"model n varies across options for item {item_idx}, cell {cell}")
        raw_n_model = group["n"][0]
        n_model = None if raw_n_model is None else int(raw_n_model)
        diagnostic_key = (arm, item_idx, cell, temperature)
        if diagnostic_key not in invalid_rates:
            raise ValueError(f"missing diagnostics for {diagnostic_key}")
        row = {
            "arm": arm, "item_idx": item_idx, "cell": cell, "temperature": temperature,
            "tv": None, "tv_ci_low": None, "tv_ci_high": None,
            "kl": None, "kl_ci_low": None, "kl_ci_high": None,
            "observed_n": observed_n, "small_cell": observed_n < min_cell_n,
            "model_n": n_model, "invalid_rate": invalid_rates[diagnostic_key],
        }
        if n_model is None:
            model_shares = group["share"].to_numpy().astype(float)
            if not np.isfinite(model_shares).all() or np.any(model_shares <= 0):
                raise ValueError(f"deterministic probabilities are invalid for {diagnostic_key}")
            if not np.isclose(model_shares.sum(), 1.0, atol=1e-6):
                raise ValueError(f"deterministic probabilities do not sum to one for {diagnostic_key}")
            row["tv"] = total_variation(observed_shares, model_shares)
            positive = observed_shares > 0
            row["kl"] = float(np.sum(
                observed_shares[positive]
                * np.log(observed_shares[positive] / model_shares[positive])
            ))
            sampled_observed = observed_boot[pair_key]
            tv_boot = 0.5 * np.abs(sampled_observed - model_shares).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl_boot = np.where(
                    sampled_observed > 0,
                    sampled_observed * np.log(sampled_observed / model_shares),
                    0.0,
                ).sum(axis=1)
            row["tv_ci_low"], row["tv_ci_high"] = _interval(tv_boot, confidence)
            row["kl_ci_low"], row["kl_ci_high"] = _interval(kl_boot, confidence)
            bootstraps[(*diagnostic_key, "tv")] = tv_boot
            bootstraps[(*diagnostic_key, "kl")] = kl_boot
        elif n_model:
            counts = _model_counts(group["share"].to_list(), n_model)
            model_shares = counts / n_model
            row["tv"] = total_variation(observed_shares, model_shares)
            row["kl"] = kl_divergence(observed_shares, counts, alpha=alpha)
            sampled_counts = rng.multinomial(n_model, model_shares, size=repetitions)
            sampled_model = sampled_counts / n_model
            sampled_observed = observed_boot[pair_key]
            tv_boot = 0.5 * np.abs(sampled_observed - sampled_model).sum(axis=1)
            smoothed_model = (sampled_counts + alpha) / (n_model + alpha * len(options))
            with np.errstate(divide="ignore", invalid="ignore"):
                kl_boot = np.where(
                    sampled_observed > 0,
                    sampled_observed * np.log(sampled_observed / smoothed_model),
                    0.0,
                ).sum(axis=1)
            row["tv_ci_low"], row["tv_ci_high"] = _interval(tv_boot, confidence)
            row["kl_ci_low"], row["kl_ci_high"] = _interval(kl_boot, confidence)
            bootstraps[(*diagnostic_key, "tv")] = tv_boot
            bootstraps[(*diagnostic_key, "kl")] = kl_boot
        metric_rows.append(row)

    metrics = pl.DataFrame(metric_rows).sort("arm", "temperature", "item_idx", "cell")
    item_rows: list[dict] = []
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    item_ids = sorted(int(value) for value in observed["item_idx"].unique())
    temperatures = sorted(float(value) for value in model["temperature"].unique())

    for scope, scoped in (
        ("all", metrics),
        (f"n_ge_{min_cell_n}", metrics.filter(~pl.col("small_cell"))),
    ):
        for key, group in scoped.drop_nulls(["tv", "kl"]).group_by(
            "arm", "temperature", "item_idx", maintain_order=True
        ):
            arm, temperature, item_idx = str(key[0]), float(key[1]), int(key[2])
            item_rows.append({
                "scope": scope, "arm": arm, "temperature": temperature,
                "item_idx": item_idx, "n_cells": group.height,
                "mean_tv": float(group["tv"].mean()), "mean_kl": float(group["kl"].mean()),
                "mean_invalid_rate": float(group["invalid_rate"].mean()),
            })

        for arm in sorted(str(value) for value in model["arm"].unique()):
            for temperature in temperatures:
                group = scoped.filter(
                    (pl.col("arm") == arm) & (pl.col("temperature") == temperature)
                ).drop_nulls(["tv", "kl"])
                represented = sorted(int(value) for value in group["item_idx"].unique())
                if represented != item_ids:
                    continue
                item_points: dict[str, list[float]] = {"tv": [], "kl": []}
                item_boots: dict[str, list[np.ndarray]] = {"tv": [], "kl": []}
                for item_idx in item_ids:
                    cells = group.filter(pl.col("item_idx") == item_idx)
                    keys = [(arm, item_idx, str(cell), temperature) for cell in cells["cell"]]
                    for metric in ("tv", "kl"):
                        item_points[metric].append(float(cells[metric].mean()))
                        item_boots[metric].append(np.vstack([
                            bootstraps[(*cell_key, metric)] for cell_key in keys
                        ]).mean(axis=0))
                row = {
                    "scope": scope, "arm": arm, "temperature": temperature,
                    "n_items": len(item_ids), "n_cells": group.height,
                }
                for metric in ("tv", "kl"):
                    item_boot = np.vstack(item_boots[metric])
                    sampled_items = rng.integers(
                        0, len(item_ids), size=(repetitions, len(item_ids))
                    )
                    replicate = np.arange(repetitions)[:, None]
                    aggregate_boot = item_boot[sampled_items, replicate].mean(axis=1)
                    row[f"mean_{metric}"] = float(np.mean(item_points[metric]))
                    row[f"mean_{metric}_ci_low"], row[f"mean_{metric}_ci_high"] = _interval(
                        aggregate_boot, confidence
                    )
                summary_rows.append(row)

        for left, right in CONTRASTS:
            for temperature in temperatures:
                left_rows = scoped.filter(
                    (pl.col("arm") == left) & (pl.col("temperature") == temperature)
                ).drop_nulls(["tv", "kl"])
                right_rows = scoped.filter(
                    (pl.col("arm") == right) & (pl.col("temperature") == temperature)
                ).drop_nulls(["tv", "kl"])
                pair_keys = sorted(
                    set(zip(left_rows["item_idx"], left_rows["cell"], strict=True))
                    & set(zip(right_rows["item_idx"], right_rows["cell"], strict=True))
                )
                represented = sorted({int(item_idx) for item_idx, _ in pair_keys})
                if represented != item_ids:
                    continue
                point_by_metric: dict[str, list[float]] = {"tv": [], "kl": []}
                boot_by_metric: dict[str, list[np.ndarray]] = {"tv": [], "kl": []}
                for item_idx in item_ids:
                    cells = [str(cell) for candidate, cell in pair_keys if int(candidate) == item_idx]
                    for metric in ("tv", "kl"):
                        left_points = [float(left_rows.filter(
                            (pl.col("item_idx") == item_idx) & (pl.col("cell") == cell)
                        )[metric][0]) for cell in cells]
                        right_points = [float(right_rows.filter(
                            (pl.col("item_idx") == item_idx) & (pl.col("cell") == cell)
                        )[metric][0]) for cell in cells]
                        point_by_metric[metric].append(np.mean(left_points) - np.mean(right_points))
                        boot_by_metric[metric].append(np.vstack([
                            bootstraps[(left, item_idx, cell, temperature, metric)]
                            - bootstraps[(right, item_idx, cell, temperature, metric)]
                            for cell in cells
                        ]).mean(axis=0))
                sampled_items = rng.integers(
                    0, len(item_ids), size=(repetitions, len(item_ids))
                )
                row = {
                    "scope": scope, "contrast": f"{left} - {right}",
                    "temperature": temperature, "n_items": len(item_ids),
                    "n_cells": len(pair_keys),
                }
                replicate = np.arange(repetitions)[:, None]
                for metric in ("tv", "kl"):
                    item_boot = np.vstack(boot_by_metric[metric])
                    contrast_boot = item_boot[sampled_items, replicate].mean(axis=1)
                    row[f"mean_{metric}_difference"] = float(np.mean(point_by_metric[metric]))
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = _interval(
                        contrast_boot, confidence
                    )
                contrast_rows.append(row)

    return (
        metrics,
        pl.DataFrame(item_rows).sort("scope", "arm", "temperature", "item_idx"),
        pl.DataFrame(summary_rows).sort("scope", "temperature", "mean_tv"),
        pl.DataFrame(contrast_rows).sort("scope", "contrast", "temperature"),
    )


def flattening_diagnostics(
    distributions: pl.DataFrame,
    *,
    min_cell_n: int = FLATTENING_MIN_CELL_N,
) -> pl.DataFrame:
    """Compute model/observed between-cell variance ratios by item and option."""
    observed = distributions.filter(pl.col("arm") == "observed").select(
        "item_idx", "cell", "code", pl.col("share").alias("observed_share"),
        pl.col("n").alias("observed_n"),
    )
    joined = distributions.filter(pl.col("arm") != "observed").join(
        observed, on=["item_idx", "cell", "code"], how="inner", validate="m:1"
    ).filter(pl.col("observed_n") >= min_cell_n)
    rows = []
    for key, group in joined.group_by("arm", "temperature", "item_idx", "code"):
        arm, temperature, item_idx, code = str(key[0]), float(key[1]), int(key[2]), str(key[3])
        paired = group.drop_nulls(["observed_share", "share"])
        n_cells = paired.height
        observed_variance = (
            float(paired["observed_share"].var(ddof=0)) if n_cells >= 2 else math.nan
        )
        model_variance = float(paired["share"].var(ddof=0)) if n_cells >= 2 else math.nan
        ratio = (
            model_variance / observed_variance
            if math.isfinite(observed_variance) and observed_variance > 0
            else None
        )
        rows.append({
            "arm": arm, "temperature": temperature, "item_idx": item_idx, "code": code,
            "n_cells": n_cells, "observed_variance": observed_variance,
            "model_variance": model_variance, "variance_ratio": ratio,
        })
    return pl.DataFrame(rows).sort("arm", "temperature", "item_idx", "code")


def evaluate_ses_subgroups(
    cell_metrics: pl.DataFrame,
    cell_attributes: pl.DataFrame,
    *,
    dimensions: Sequence[str] = ("age", "gender", "education"),
    temperature: float = PRIMARY_TEMPERATURE,
    repetitions: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return exploratory SES summaries and item-bootstrap paired contrasts."""
    keys = ["item_idx", "cell"]
    if cell_attributes.select(keys).n_unique() != cell_attributes.height:
        raise ValueError("cell attributes must be unique by item_idx and cell")
    available_dimensions = [name for name in dimensions if name in cell_attributes.columns]
    frame = cell_metrics.filter(pl.col("temperature") == temperature).join(
        cell_attributes.select(*keys, *available_dimensions),
        on=keys,
        how="inner",
        validate="m:1",
    )
    rng = np.random.default_rng(seed)
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    for scope, scoped in (
        ("all", frame),
        (f"n_ge_{MIN_CELL_N}", frame.filter(~pl.col("small_cell"))),
    ):
        for dimension in available_dimensions:
            levels = sorted(str(value) for value in scoped[dimension].drop_nulls().unique())
            for level in levels:
                group = scoped.filter(pl.col(dimension) == level)
                item_means = group.group_by("arm", "item_idx").agg(
                    pl.col("tv").mean().alias("mean_tv"),
                    pl.col("kl").mean().alias("mean_kl"),
                    pl.len().alias("n_cells"),
                )
                for arm_group in item_means.partition_by("arm", maintain_order=True):
                    summary_rows.append({
                        "scope": scope,
                        "dimension": dimension,
                        "level": level,
                        "arm": str(arm_group["arm"][0]),
                        "temperature": temperature,
                        "n_items": arm_group.height,
                        "n_cells": int(arm_group["n_cells"].sum()),
                        "mean_tv": float(arm_group["mean_tv"].mean()),
                        "mean_kl": float(arm_group["mean_kl"].mean()),
                    })
                for left, right in CONTRASTS:
                    left_rows = group.filter(pl.col("arm") == left)
                    right_rows = group.filter(pl.col("arm") == right)
                    pair_keys = sorted(
                        set(zip(left_rows["item_idx"], left_rows["cell"], strict=True))
                        & set(zip(right_rows["item_idx"], right_rows["cell"], strict=True))
                    )
                    if not pair_keys:
                        continue
                    item_differences: dict[str, list[float]] = {"tv": [], "kl": []}
                    for item_idx in sorted({int(item) for item, _ in pair_keys}):
                        cells = [str(cell) for item, cell in pair_keys if int(item) == item_idx]
                        for metric in ("tv", "kl"):
                            differences = []
                            for cell in cells:
                                left_value = left_rows.filter(
                                    (pl.col("item_idx") == item_idx) & (pl.col("cell") == cell)
                                )[metric][0]
                                right_value = right_rows.filter(
                                    (pl.col("item_idx") == item_idx) & (pl.col("cell") == cell)
                                )[metric][0]
                                differences.append(float(left_value) - float(right_value))
                            item_differences[metric].append(float(np.mean(differences)))
                    n_items = len(item_differences["tv"])
                    sampled_items = rng.integers(0, n_items, size=(repetitions, n_items))
                    row = {
                        "scope": scope, "dimension": dimension, "level": level,
                        "contrast": f"{left} - {right}", "temperature": temperature,
                        "n_items": n_items, "n_cells": len(pair_keys),
                    }
                    for metric in ("tv", "kl"):
                        values = np.asarray(item_differences[metric])
                        bootstrap = values[sampled_items].mean(axis=1)
                        row[f"mean_{metric}_difference"] = float(values.mean())
                        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = _interval(
                            bootstrap, confidence
                        )
                    contrast_rows.append(row)
    return (
        pl.DataFrame(summary_rows).sort("scope", "dimension", "level", "mean_tv"),
        pl.DataFrame(contrast_rows).sort("scope", "dimension", "level", "contrast"),
    )


__all__ = [
    "CONTRASTS", "DEFAULT_ALPHA", "DEFAULT_BOOTSTRAPS", "DEFAULT_CONFIDENCE",
    "DEFAULT_SEED", "FLATTENING_MIN_CELL_N", "MIN_CELL_N", "PRIMARY_TEMPERATURE",
    "evaluate_distributions", "evaluate_ses_subgroups", "flattening_diagnostics",
    "kl_divergence", "total_variation",
]
