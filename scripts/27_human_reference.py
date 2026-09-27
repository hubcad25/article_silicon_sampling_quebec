"""Human-human reference: TV between the context half and the evaluation half, per item x cell.

Measures the disagreement expected from respondent sampling alone. It is not a method for a new
question (it uses answers to the target question); it sets the scale for the model arms.

Usage:
    .venv/bin/python scripts/27_human_reference.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.corpus import blob  # noqa: E402
from article_silicon_sampling_quebec.evaluation import build_observed_tables  # noqa: E402
from article_silicon_sampling_quebec.inference import (  # noqa: E402
    HELDOUT_HALVES_PATH,
    HELDOUT_PATH,
    build_item_cells,
)
from article_silicon_sampling_quebec.metrics import (  # noqa: E402
    DEFAULT_BOOTSTRAPS,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    MIN_CELL_N,
    PRIMARY_TEMPERATURE,
    total_variation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=REPO / "data" / "analysis")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    return parser.parse_args()


def half_distributions(pairs, heldout: pl.DataFrame, halves: pl.DataFrame, half: str,
                       surveys: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Weighted distribution per item x cell in one half; pairs with no valid answer are skipped."""
    members = heldout.join(
        halves.filter(pl.col("half") == half).select("__survey_id", "__respondent_id", "cell"),
        on=["__survey_id", "__respondent_id", "cell"], how="inner", validate="1:1",
    )
    frames = []
    for pair in pairs:
        try:
            observed, _ = build_observed_tables(
                [pair], members, validate_pair_n=False,
                survey_loader=lambda survey_id, columns: surveys[survey_id].select(columns),
            )
        except ValueError as error:
            if "mass is zero" not in str(error):
                raise
            continue
        frames.append(observed)
    return pl.concat(frames).select("item_idx", "cell", "code", "share", "n")


def item_bootstrap(values_by_item: dict[int, float], repetitions: int, seed: int,
                   confidence: float) -> tuple[float, float, float]:
    """Unweighted mean over items, with a percentile item bootstrap."""
    values = np.asarray(list(values_by_item.values()))
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    tail = (1 - confidence) / 2
    return float(values.mean()), float(np.quantile(boot, tail)), float(np.quantile(boot, 1 - tail))


def main() -> None:
    args = parse_args()
    pairs = build_item_cells()
    heldout = pl.read_parquet(HELDOUT_PATH).with_columns(pl.col("__respondent_id").cast(pl.Utf8))
    halves = pl.read_csv(HELDOUT_HALVES_PATH, schema_overrides={"__respondent_id": pl.Utf8})

    variables: dict[str, set[str]] = {}
    for pair in pairs:
        variables.setdefault(pair.survey_id, set()).add(pair.variable)
    surveys = {
        survey_id: blob.read_survey(survey_id, ["__respondent_id", *sorted(names)]).with_columns(
            pl.col("__respondent_id").cast(pl.Utf8)
        )
        for survey_id, names in variables.items()
    }

    context = half_distributions(pairs, heldout, halves, "context", surveys)
    evaluation = half_distributions(pairs, heldout, halves, "eval", surveys)
    joined = evaluation.join(
        context, on=["item_idx", "cell", "code"], how="left", suffix="_context",
    )
    rows = []
    for (item_idx, cell), group in joined.group_by("item_idx", "cell", maintain_order=True):
        group = group.sort("code")
        context_n = group["n_context"].drop_nulls()
        rows.append({
            "item_idx": int(item_idx), "cell": str(cell),
            "observed_n": int(group["n"][0]),
            "context_n": int(context_n[0]) if context_n.len() else 0,
            "tv": (total_variation(group["share"].to_numpy(),
                                   group["share_context"].to_numpy())
                   if context_n.len() else None),
        })
    cells = pl.DataFrame(rows).with_columns(
        (pl.col("observed_n") < MIN_CELL_N).alias("small_cell"),
    ).sort("item_idx", "cell")

    arms = pl.read_csv(args.analysis_root / "cell_metrics.csv").filter(
        pl.col("temperature") == PRIMARY_TEMPERATURE
    ).select("arm", "item_idx", "cell", "tv")

    summary_rows, contrast_rows = [], []
    for scope, scoped in (("all", cells), (f"n_ge_{MIN_CELL_N}", cells.filter(~pl.col("small_cell")))):
        human = scoped.drop_nulls("tv")
        by_item = {int(k[0]): float(g["tv"].mean()) for k, g in human.group_by("item_idx")}
        mean, low, high = item_bootstrap(by_item, args.bootstrap, args.seed, args.confidence)
        summary_rows.append({
            "scope": scope, "arm": "human", "n_items": len(by_item), "n_cells": human.height,
            "mean_tv": mean, "mean_tv_ci_low": low, "mean_tv_ci_high": high,
        })
        for arm in sorted(arms["arm"].unique().to_list()):
            paired = arms.filter(pl.col("arm") == arm).join(
                human.select("item_idx", "cell", pl.col("tv").alias("tv_human")),
                on=["item_idx", "cell"], how="inner", validate="1:1",
            ).drop_nulls("tv")
            by_item_arm = {int(k[0]): float(g["tv"].mean()) for k, g in paired.group_by("item_idx")}
            by_item_diff = {
                int(k[0]): float((g["tv"] - g["tv_human"]).mean())
                for k, g in paired.group_by("item_idx")
            }
            mean, low, high = item_bootstrap(by_item_arm, args.bootstrap, args.seed, args.confidence)
            summary_rows.append({
                "scope": scope, "arm": arm, "n_items": len(by_item_arm), "n_cells": paired.height,
                "mean_tv": mean, "mean_tv_ci_low": low, "mean_tv_ci_high": high,
            })
            mean, low, high = item_bootstrap(by_item_diff, args.bootstrap, args.seed, args.confidence)
            contrast_rows.append({
                "scope": scope, "contrast": f"{arm} - human", "n_items": len(by_item_diff),
                "n_cells": paired.height, "mean_tv_difference": mean,
                "tv_ci_low": low, "tv_ci_high": high,
            })

    outputs = {
        "human_reference_cells.csv": cells,
        "human_reference_summary.csv": pl.DataFrame(summary_rows),
        "human_reference_contrasts.csv": pl.DataFrame(contrast_rows),
    }
    for name, frame in outputs.items():
        frame.write_csv(args.analysis_root / name)
        print(args.analysis_root / name)
    print(pl.DataFrame(summary_rows))
    print(pl.DataFrame(contrast_rows))


if __name__ == "__main__":
    main()
