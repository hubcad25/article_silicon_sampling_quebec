"""Compute the pre-specified five-arm pilot analysis and annex tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.metrics import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAPS,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    MIN_CELL_N,
    PRIMARY_TEMPERATURE,
    evaluate_distributions,
    evaluate_ses_subgroups,
    flattening_diagnostics,
)
from article_silicon_sampling_quebec.inference import build_item_cells  # noqa: E402


def verified_statistical_outputs(root: Path) -> list[Path] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    required = {"distributions.csv", "distribution_diagnostics.csv"}
    if not required <= set(outputs):
        raise ValueError("statistical benchmark manifest is incomplete")
    for name, expected_hash in outputs.items():
        path = root / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"statistical benchmark output does not match manifest: {path}")
    return [root / name for name in sorted(outputs)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=REPO / "data" / "analysis")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.input_root
    distributions = pl.read_csv(
        args.input_root / "distributions.csv",
        schema_overrides={"code": pl.Utf8, "temperature": pl.Float64},
    )
    diagnostics = pl.read_csv(args.input_root / "distribution_diagnostics.csv")
    statistical_root = args.input_root / "statistical_benchmark"
    statistical_outputs = verified_statistical_outputs(statistical_root)
    if statistical_outputs is not None:
        statistical_distributions = pl.read_csv(
            statistical_root / "distributions.csv",
            schema_overrides={"code": pl.Utf8, "temperature": pl.Float64, "n": pl.Int64},
        )
        statistical_diagnostics = pl.read_csv(
            statistical_root / "distribution_diagnostics.csv",
            schema_overrides={"transport_n": pl.Int64, "effective_n": pl.Int64},
        )
        distributions = pl.concat([distributions, statistical_distributions], how="vertical")
        diagnostics = pl.concat([diagnostics, statistical_diagnostics], how="vertical")
    responses = pl.read_csv(
        args.input_root / "observed_responses.csv", schema_overrides={"code": pl.Utf8}
    )
    metrics, items, summaries, contrasts = evaluate_distributions(
        distributions, diagnostics, responses, repetitions=args.bootstrap,
        seed=args.seed, alpha=args.alpha, confidence=args.confidence,
    )
    flattening = flattening_diagnostics(distributions)
    cell_attributes = []
    for pair in build_item_cells():
        row = {"item_idx": pair.item_idx, "cell": pair.cell}
        row.update(dict(zip(pair.dimensions, pair.cell.split("|"), strict=True)))
        cell_attributes.append(row)
    ses_summaries, ses_contrasts = evaluate_ses_subgroups(
        metrics, pl.DataFrame(cell_attributes), repetitions=args.bootstrap,
        seed=args.seed, confidence=args.confidence,
    )
    main_summary = summaries.filter(
        (pl.col("temperature") == PRIMARY_TEMPERATURE) & (pl.col("scope") == "all")
    )
    main_contrasts = contrasts.filter(
        (pl.col("temperature") == PRIMARY_TEMPERATURE) & (pl.col("scope") == "all")
    )
    sensitivity_contrasts = contrasts.filter(
        (pl.col("temperature") == PRIMARY_TEMPERATURE)
        & (pl.col("scope") == f"n_ge_{MIN_CELL_N}")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sensitivity_summary = summaries.filter(
        (pl.col("temperature") == PRIMARY_TEMPERATURE)
        & (pl.col("scope") == f"n_ge_{MIN_CELL_N}")
    )
    outputs = {
        "cell_metrics": (output_root / "cell_metrics.csv", metrics),
        "item_metrics": (output_root / "item_metrics.csv", items),
        "main_arm_summary": (output_root / "main_arm_summary.csv", main_summary),
        "main_arm_contrasts": (output_root / "main_arm_contrasts.csv", main_contrasts),
        "sensitivity_item_metrics_n_ge30": (
            output_root / "sensitivity_item_metrics_n_ge30.csv",
            items.filter(
                (pl.col("scope") == f"n_ge_{MIN_CELL_N}")
                & (pl.col("temperature") == PRIMARY_TEMPERATURE)
            ),
        ),
        "sensitivity_n_ge30": (output_root / "sensitivity_n_ge30.csv", sensitivity_summary),
        "sensitivity_contrasts_n_ge30": (
            output_root / "sensitivity_contrasts_n_ge30.csv", sensitivity_contrasts,
        ),
        "annex_temperature": (output_root / "annex_temperature.csv", summaries),
        "annex_temperature_contrasts": (
            output_root / "annex_temperature_contrasts.csv", contrasts,
        ),
        "annex_invalidity": (
            output_root / "annex_invalidity.csv",
            metrics.select(
                "arm", "item_idx", "cell", "temperature", "model_n", "invalid_rate"
            ),
        ),
        "annex_flattening": (output_root / "annex_flattening.csv", flattening),
        "exploratory_ses_summary": (
            output_root / "exploratory_ses_summary.csv", ses_summaries,
        ),
        "exploratory_ses_contrasts": (
            output_root / "exploratory_ses_contrasts.csv", ses_contrasts,
        ),
        "analysis_parameters": (
            output_root / "analysis_parameters.csv",
            pl.DataFrame([{
                "provisional": False,
                "truth": "heldout half=eval only",
                "primary_metric": "total variation",
                "robustness_metric": "KL(observed || model)",
                "primary_temperature": PRIMARY_TEMPERATURE,
                "small_cell_rule": (
                    "retain in paired main analysis; flag observed_n<30; "
                    "report sensitivity observed_n>=30"
                ),
                "aggregation": "unweighted cells within item, then equal weight to 12 items",
                "kl_direction": "observed || model",
                "model_count_pseudocount": args.alpha,
                "observed_smoothing": False,
                "bootstrap_repetitions": args.bootstrap,
                "bootstrap_seed": args.seed,
                "confidence": args.confidence,
                "interval": "percentile",
                "observed_resampling": "respondents uniformly with replacement; retain weights",
                "model_resampling": "valid draws with replacement",
                "contrast_resampling": "paired cells; items with replacement; same items in both arms",
                "statistical_benchmark_s": (
                    "conditional logit with frozen text-embedding-3-large alternatives"
                    if "S" in set(distributions["arm"])
                    else "not available; not analysed"
                ),
            }]),
        ),
    }
    for name, (path, frame) in outputs.items():
        frame.write_csv(path)
        print(f"{name}: {path}")
    source_files = [
        args.input_root / "distributions.csv",
        args.input_root / "distribution_diagnostics.csv",
        args.input_root / "observed_responses.csv",
        REPO / "data" / "split" / "heldout_halves.csv",
        REPO / "docs" / "adr" / "0003-analyse-comparative-pilote.md",
    ]
    if statistical_outputs is not None:
        source_files.extend([*statistical_outputs, statistical_root / "manifest.json"])
    manifest = {
        "parameters": outputs["analysis_parameters"][1].row(0, named=True),
        "sources": {
            str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in source_files
        },
    }
    manifest_path = output_root / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(f"analysis_manifest: {manifest_path}")
    print(
        f"analysis parameters: alpha={args.alpha}, bootstrap={args.bootstrap}, "
        f"seed={args.seed}, confidence={args.confidence}"
    )


if __name__ == "__main__":
    main()
