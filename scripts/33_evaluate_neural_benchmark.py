"""Evaluate exploratory neural benchmark N against S and trained 8k arm A."""

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
    PRIMARY_TEMPERATURE,
    evaluate_distributions,
)

ANALYSIS = REPO / "data" / "analysis"
OUTPUT = ANALYSIS / "neural_benchmark_evaluation"
CONTRASTS = (("N", "S"), ("N", "A"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_tables(root: Path, expected_arm: str) -> tuple[pl.DataFrame, pl.DataFrame, list[Path]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("arm") != expected_arm:
        raise ValueError(f"expected arm {expected_arm} in {manifest_path}")
    required = {"distributions.csv", "distribution_diagnostics.csv"}
    outputs = manifest.get("outputs", {})
    if not required <= set(outputs):
        raise ValueError(f"incomplete benchmark manifest: {manifest_path}")
    paths = []
    for name, expected_hash in outputs.items():
        path = root / name
        if not path.exists() or _sha256(path) != expected_hash:
            raise ValueError(f"benchmark output does not match manifest: {path}")
        paths.append(path)
    distributions = pl.read_csv(
        root / "distributions.csv",
        schema_overrides={"code": pl.Utf8, "temperature": pl.Float64, "n": pl.Int64},
    )
    diagnostics = pl.read_csv(
        root / "distribution_diagnostics.csv",
        schema_overrides={"transport_n": pl.Int64, "effective_n": pl.Int64},
    )
    return distributions, diagnostics, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ANALYSIS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_distributions_path = args.input_root / "distributions.csv"
    base_diagnostics_path = args.input_root / "distribution_diagnostics.csv"
    responses_path = args.input_root / "observed_responses.csv"
    distributions = pl.read_csv(
        base_distributions_path,
        schema_overrides={"code": pl.Utf8, "temperature": pl.Float64},
    ).filter(
        (pl.col("arm") == "observed")
        | ((pl.col("arm") == "A") & (pl.col("temperature") == PRIMARY_TEMPERATURE))
    )
    diagnostics = pl.read_csv(base_diagnostics_path).filter(
        (pl.col("arm") == "A") & (pl.col("temperature") == PRIMARY_TEMPERATURE)
    )
    source_paths = [base_distributions_path, base_diagnostics_path, responses_path]
    for folder, arm in (("statistical_benchmark", "S"), ("neural_benchmark", "N")):
        benchmark_root = args.input_root / folder
        benchmark_distributions, benchmark_diagnostics, paths = _verified_tables(
            benchmark_root, arm
        )
        distributions = pl.concat([distributions, benchmark_distributions], how="vertical")
        diagnostics = pl.concat([diagnostics, benchmark_diagnostics], how="vertical")
        source_paths.extend([*paths, benchmark_root / "manifest.json"])

    responses = pl.read_csv(responses_path, schema_overrides={"code": pl.Utf8})
    metrics, items, summaries, contrasts = evaluate_distributions(
        distributions,
        diagnostics,
        responses,
        repetitions=args.bootstrap,
        seed=DEFAULT_SEED,
        alpha=DEFAULT_ALPHA,
        confidence=DEFAULT_CONFIDENCE,
        contrasts=CONTRASTS,
    )
    main_summary = summaries.filter(
        (pl.col("scope") == "all") & (pl.col("temperature") == PRIMARY_TEMPERATURE)
    ).sort("mean_tv")
    main_contrasts = contrasts.filter(
        (pl.col("scope") == "all") & (pl.col("temperature") == PRIMARY_TEMPERATURE)
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cell_metrics.csv": metrics,
        "item_metrics.csv": items,
        "arm_summary.csv": main_summary,
        "contrasts.csv": main_contrasts,
    }
    for name, frame in outputs.items():
        frame.write_csv(args.output_root / name)
    manifest = {
        "status": "exploratory",
        "temperature": PRIMARY_TEMPERATURE,
        "contrasts": [f"{left} - {right}" for left, right in CONTRASTS],
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": DEFAULT_SEED,
        "sources": {str(path.relative_to(REPO)): _sha256(path) for path in source_paths},
        "outputs": {
            name: _sha256(args.output_root / name) for name in outputs
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(main_summary)
    print(main_contrasts)
    print(f"neural benchmark evaluation: {args.output_root}")


if __name__ == "__main__":
    main()
