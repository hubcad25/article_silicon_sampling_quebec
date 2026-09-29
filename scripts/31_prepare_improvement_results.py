"""Produce the confirmatory results for the 48-question improvement note."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.evaluation import (  # noqa: E402
    build_model_tables,
    build_observed_tables,
)
from article_silicon_sampling_quebec.improvement import (  # noqa: E402
    METHODS,
    RAW_ARMS,
    build_derived_methods,
    category_breakdown,
    cell_metrics,
    item_metrics,
    summarize_methods,
)
from article_silicon_sampling_quebec.inference import (  # noqa: E402
    HELDOUT_HALVES_PATH,
    HELDOUT_PATH,
    build_item_cells,
)

INFERENCE = REPO / "data" / "analysis" / "inference" / "final48-250"
OUTPUT = REPO / "data" / "analysis" / "improvement_48"
ARM_PATHS = {
    "A": INFERENCE / "c0-8k" / "A.csv",
    "A20": INFERENCE / "c0-20k" / "A20.csv",
    "B020": INFERENCE / "c1-20k" / "B020.csv",
    "BR20": INFERENCE / "c1-20k" / "BR20.csv",
}
THEME_LABELS = {
    "partis_vote": "Partis et vote",
    "democratie_engagement": "Démocratie et engagement",
    "valeurs_sociales": "Valeurs sociales",
    "sante": "Santé",
    "etat_economie": "État et économie",
    "economie_percue": "Économie perçue",
    "identite_qc_federalisme": "Identité et fédéralisme",
}
DISTANCE_LABELS = {
    "isolated": "Isolée (< 0,70)",
    "far": "Loin (0,70–0,775)",
    "moderate": "Modérée (0,775–0,85)",
    "near": "Proche (0,85–0,95)",
    "quasi_duplicate": "Quasi-doublon (≥ 0,95)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_and_read_arms() -> dict[str, pl.DataFrame]:
    frames = {}
    for arm, csv_path in ARM_PATHS.items():
        manifest_path = csv_path.with_suffix(".manifest.json")
        progress_path = csv_path.with_suffix(".progress.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = manifest["expected"]
        settings = manifest["settings"]
        if not (
            expected == {
                "frozen_item_cell_pairs": 885,
                "questions": 48,
                "selected_item_cell_pairs": 885,
                "subset": "remaining",
                "tasks": 221_250,
            }
            and settings["arm"] == arm
            and settings["draws"] == 250
            and settings["subset"] == "remaining"
            and settings["temperatures"] == [1.0]
            and progress["state"] == "completed"
            and progress["done"] == progress["tasks"] == 221_250
        ):
            raise ValueError(f"{arm} does not satisfy the frozen final48-250 contract")
        frames[arm] = pl.read_csv(
            csv_path,
            schema_overrides={"arm": pl.Utf8, "cell": pl.Utf8, "matched_code": pl.Utf8},
            null_values={"matched_code": ""},
        )
    return frames


def observed_tables(pairs):
    heldout = pl.read_parquet(HELDOUT_PATH).with_columns(
        pl.col("__respondent_id").cast(pl.Utf8)
    )
    halves = pl.read_csv(
        HELDOUT_HALVES_PATH, schema_overrides={"__respondent_id": pl.Utf8}
    ).filter(pl.col("half") == "eval")
    eval_heldout = heldout.join(
        halves.select("__survey_id", "__respondent_id", "cell"),
        on=["__survey_id", "__respondent_id", "cell"], how="inner", validate="1:1",
    )
    if eval_heldout.height != halves.height:
        raise ValueError("heldout eval rows do not match the frozen split")
    return build_observed_tables(pairs, eval_heldout, validate_pair_n=False)


def contrast(items: pl.DataFrame, left: str, right: str, repetitions: int) -> dict:
    selected = items.filter(pl.col("arm").is_in([left, right])).select(
        "item_idx", "arm", "mean_tv", "mean_kl"
    ).pivot(on="arm", index="item_idx", values=["mean_tv", "mean_kl"]).sort("item_idx")
    tv = selected[f"mean_tv_{left}"].to_numpy() - selected[f"mean_tv_{right}"].to_numpy()
    kl = selected[f"mean_kl_{left}"].to_numpy() - selected[f"mean_kl_{right}"].to_numpy()
    rng = np.random.default_rng(20_260_925)
    sampled = rng.integers(0, len(tv), size=(repetitions, len(tv)))
    boot = tv[sampled].mean(axis=1)
    return {
        "contrast": f"{left} - {right}", "n_items": len(tv),
        "mean_tv_difference": float(tv.mean()),
        "tv_ci_low": float(np.quantile(boot, 0.025)),
        "tv_ci_high": float(np.quantile(boot, 0.975)),
        "mean_kl_difference": float(kl.mean()),
    }


def main() -> None:
    args = parse_args()
    pairs = build_item_cells(subset="remaining")
    observed, responses = observed_tables(pairs)
    raw, diagnostics = build_model_tables(
        validate_and_read_arms(), pairs, temperatures=(1.0,), draws_per_group=250,
        arms=RAW_ARMS,
    )
    derived = build_derived_methods(raw)
    predictions = pl.concat([raw, derived], how="vertical")
    cells = cell_metrics(observed, predictions)
    metadata = pl.read_csv(REPO / "data" / "analysis" / "test_blocks.csv").filter(
        ~pl.col("pilot")
    )
    items = item_metrics(cells, metadata)
    summary = summarize_methods(items, repetitions=args.bootstrap)
    selected = str(summary.sort("mean_tv")["arm"][0])
    themes = category_breakdown(items, selected, "block", labels=THEME_LABELS)
    proximity = category_breakdown(
        items, selected, "distance_bin", labels=DISTANCE_LABELS
    )
    contrast_pairs = [(method, "A") for method in METHODS if method != "A"] + [
        ("BR20", "B020"), ("BR20", "AVG4"), ("AVG4_REG", "AVG4")
    ]
    contrasts = pl.DataFrame([
        contrast(items, left, right, args.bootstrap) for left, right in contrast_pairs
    ])
    invalidity = diagnostics.group_by("arm").agg(
        pl.col("transport_n").sum(), pl.col("invalid_n").sum(),
    ).with_columns(
        (pl.col("invalid_n") / pl.col("transport_n")).alias("invalid_rate")
    ).sort("arm")

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "distributions.csv": pl.concat([observed, predictions], how="vertical"),
        "observed_responses.csv": responses,
        "distribution_diagnostics.csv": diagnostics,
        "cell_metrics.csv": cells,
        "item_metrics.csv": items,
        "method_summary.csv": summary,
        "method_contrasts.csv": contrasts,
        "theme_breakdown.csv": themes,
        "proximity_breakdown.csv": proximity,
        "invalidity.csv": invalidity,
    }
    for name, frame in outputs.items():
        frame.write_csv(args.output_root / name)
    source_paths = [
        *ARM_PATHS.values(),
        *(path.with_suffix(".manifest.json") for path in ARM_PATHS.values()),
        *(path.with_suffix(".progress.json") for path in ARM_PATHS.values()),
        HELDOUT_PATH, HELDOUT_HALVES_PATH,
        REPO / "data" / "analysis" / "test_blocks.csv",
        REPO / "docs" / "adr" / "0006-inference-48-questions-amelioration.md",
    ]
    manifest = {
        "selected_method": selected,
        "parameters": {
            "questions": 48, "item_cell_pairs": 885, "draws": 250,
            "temperature": 1.0, "bootstrap_repetitions": args.bootstrap,
            "bootstrap_seed": 20_260_925,
            "aggregation": "unweighted cells within question, then equal weight to questions",
            "derived_method_kl_effective_n": 250,
            "regularization": {"A_REG_deviation_retained": 0.20,
                               "AVG4_REG_deviation_retained": 0.55},
        },
        "sources": {str(path.relative_to(REPO)): sha256(path) for path in source_paths},
        "outputs": {name: sha256(args.output_root / name) for name in outputs},
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary)
    print(f"selected method: {selected}")
    print(f"outputs: {args.output_root}")


if __name__ == "__main__":
    main()
