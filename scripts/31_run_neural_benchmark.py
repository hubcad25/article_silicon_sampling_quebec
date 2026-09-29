"""Fit exploratory neural benchmark N and predict pilot distributions.

N is a small nonlinear residual over the frozen utilities of statistical
benchmark S. Its architecture and optimization are fixed in ADR 0007.

Usage:
    source .venv/bin/activate
    python scripts/31_run_neural_benchmark.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.dataset import read_jsonl  # noqa: E402
from article_silicon_sampling_quebec.neural_choice import (  # noqa: E402
    NeuralChoiceConfig,
    ProjectedChoiceData,
    fit_residual_choice_network,
)

DATASET_ROOT = REPO / "data" / "datasets"
OUTPUT_ROOT = REPO / "data" / "analysis" / "neural_benchmark"
EMBEDDING_CACHE = REPO / "data" / ".cache" / "benchmark_s_embedding_cache.parquet"
STATISTICAL_ROOT = REPO / "data" / "analysis" / "statistical_benchmark"
PCA_COMPONENTS = 128
PCA_ITERATED_POWER = 3
PCA_SCALE_FLOOR = 1e-6


def _load_statistical_script():
    spec = importlib.util.spec_from_file_location(
        "run_statistical_benchmark", REPO / "scripts" / "27_run_statistical_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _load_linear_model(root: Path) -> tuple[np.ndarray, dict]:
    manifest_path = root / "manifest.json"
    model_path = root / "model.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("outputs", {}).get("model.npz")
    if expected is None or _sha256(model_path) != expected:
        raise ValueError("statistical benchmark model does not match its manifest")
    with np.load(model_path) as payload:
        coefficients = np.asarray(payload["coefficients"], dtype=np.float32)
    return coefficients, manifest


def _project(
    pca: PCA, scale: np.ndarray, features: np.ndarray
) -> np.ndarray:
    return np.asarray(pca.transform(features) / scale, dtype=np.float32)


def _choice_data(
    projected: np.ndarray,
    original: np.ndarray,
    offsets: np.ndarray,
    chosen: np.ndarray,
    coefficients: np.ndarray,
) -> ProjectedChoiceData:
    return ProjectedChoiceData(
        features=projected,
        base_utilities=np.asarray(original @ coefficients, dtype=np.float32),
        offsets=offsets,
        chosen=chosen,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--statistical-root", type=Path, default=STATISTICAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--embedding-cache", type=Path, default=EMBEDDING_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO / ".env")
    required = ["AOAI_ENDPOINT", "AOAI_KEY", "AOAI_EMBED_DEPLOYMENT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing Azure embedding settings: {', '.join(missing)}")

    statistical = _load_statistical_script()
    train_path = args.dataset_root / "c0_train_8000.jsonl"
    validation_path = args.dataset_root / "c0_validation.jsonl"
    train_texts, train_offsets, train_chosen = statistical.examples_to_texts(
        read_jsonl(train_path)
    )
    validation_texts, validation_offsets, validation_chosen = statistical.examples_to_texts(
        read_jsonl(validation_path)
    )
    test_texts, test_offsets, test_groups = statistical.prediction_texts()
    all_texts = train_texts + validation_texts + test_texts
    embedding_module = statistical._embedding_module(args.embedding_cache)
    vectors = embedding_module.embed_texts(all_texts, dry_run=args.dry_run)

    train_end = len(train_texts)
    validation_end = train_end + len(validation_texts)
    train_vectors = vectors[:train_end]
    validation_vectors = vectors[train_end:validation_end]
    test_vectors = vectors[validation_end:]
    coefficients, statistical_manifest = _load_linear_model(args.statistical_root)
    if coefficients.shape != (train_vectors.shape[1],):
        raise ValueError("statistical coefficient and embedding dimensions differ")
    expected_matrix = statistical_manifest.get("embedding_matrix_sha256")
    if expected_matrix != _array_sha256(vectors):
        raise ValueError("embedding matrix differs from the matrix used by benchmark S")

    config = NeuralChoiceConfig()
    print(f"[pca] fitting {PCA_COMPONENTS} train-only components")
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        iterated_power=PCA_ITERATED_POWER,
        random_state=config.seed,
    )
    train_projected = np.asarray(pca.fit_transform(train_vectors), dtype=np.float32)
    scale = np.maximum(train_projected.std(axis=0), PCA_SCALE_FLOOR).astype(np.float32)
    train_projected /= scale
    validation_projected = _project(pca, scale, validation_vectors)
    test_projected = _project(pca, scale, test_vectors)

    train = _choice_data(
        train_projected, train_vectors, train_offsets, train_chosen, coefficients
    )
    validation = _choice_data(
        validation_projected,
        validation_vectors,
        validation_offsets,
        validation_chosen,
        coefficients,
    )
    print("[fit] training residual neural utility model")
    fit = fit_residual_choice_network(train, validation, config)
    test = ProjectedChoiceData(
        features=test_projected,
        base_utilities=np.asarray(test_vectors @ coefficients, dtype=np.float32),
        offsets=test_offsets,
        chosen=np.zeros(len(test_offsets) - 1, dtype=np.int64),
    )
    test_probabilities = fit.model.probabilities(test)
    if len(test_groups) != 275:
        raise RuntimeError(f"expected 275 test item-cell groups, got {len(test_groups)}")

    distribution_rows = []
    diagnostic_rows = []
    for group_index, (item_idx, cell, codes) in enumerate(test_groups):
        start, stop = test_offsets[group_index:group_index + 2]
        shares = test_probabilities[start:stop]
        distribution_rows.extend({
            "arm": "N", "item_idx": item_idx, "cell": cell, "temperature": 1.0,
            "code": code, "share": float(share), "n": None,
        } for code, share in zip(codes, shares, strict=True))
        diagnostic_rows.append({
            "arm": "N", "item_idx": item_idx, "cell": cell, "temperature": 1.0,
            "transport_n": None, "effective_n": None, "invalid_n": 0,
            "invalid_rate": 0.0, "coverage_ok": True,
        })

    distributions = pl.DataFrame(distribution_rows).with_columns(pl.col("n").cast(pl.Int64))
    diagnostics = pl.DataFrame(diagnostic_rows).with_columns(
        pl.col("transport_n").cast(pl.Int64), pl.col("effective_n").cast(pl.Int64)
    )
    if distributions.select("item_idx", "cell").unique().height != 275:
        raise RuntimeError("neural distributions do not cover 275 unique item-cell groups")
    history = pl.DataFrame([
        {"epoch": epoch, "validation_nll": nll, "selected": epoch == fit.selected_epoch}
        for epoch, nll in fit.history
    ])
    fit_summary = pl.DataFrame([{
        "selected_epoch": fit.selected_epoch,
        "epochs_run": fit.epochs_run,
        "baseline_validation_nll": fit.baseline_validation_nll,
        "selected_validation_nll": fit.validation_nll,
        "selected_train_nll": fit.train_nll,
    }])

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_root.parent) as temporary_dir:
        temporary_root = Path(temporary_dir)
        distributions.write_csv(temporary_root / "distributions.csv")
        diagnostics.write_csv(temporary_root / "distribution_diagnostics.csv")
        history.write_csv(temporary_root / "training_history.csv")
        fit_summary.write_csv(temporary_root / "fit_summary.csv")
        np.savez_compressed(
            temporary_root / "model.npz",
            linear_coefficients=coefficients,
            projection_mean=np.asarray(pca.mean_, dtype=np.float32),
            projection_components=np.asarray(pca.components_, dtype=np.float32),
            projection_scale=scale,
            hidden_weights=fit.model.hidden_weights,
            hidden_bias=fit.model.hidden_bias,
            output_weights=fit.model.output_weights,
        )
        sources = [
            train_path,
            validation_path,
            args.statistical_root / "model.npz",
            args.statistical_root / "manifest.json",
            REPO / "data/split/heldout_items.json",
            REPO / "data/analysis/test_blocks.csv",
            REPO / "data/items.parquet",
            REPO / "data/strata_definition.json",
            REPO / "data/split/heldout_respondents.parquet",
            REPO / "data/crosswalks/ses_crosswalk.json",
            REPO / "docs/adr/0007-benchmark-neuronal-exploratoire.md",
        ]
        manifest = {
            "version": "1.0",
            "arm": "N",
            "status": "exploratory",
            "model": "frozen S utility plus one-hidden-layer residual MLP",
            "embedding_deployment": os.environ["AOAI_EMBED_DEPLOYMENT"],
            "embedding_api_version": embedding_module.AOAI_API_VERSION,
            "embedding_dimensions": int(train_vectors.shape[1]),
            "embedding_matrix_sha256": _array_sha256(vectors),
            "training_examples": train.n_cases,
            "validation_examples": validation.n_cases,
            "test_item_cells": len(test_groups),
            "pca": {
                "components": PCA_COMPONENTS,
                "solver": "randomized",
                "iterated_power": PCA_ITERATED_POWER,
                "scale_floor": PCA_SCALE_FLOOR,
                "fit_scope": "training alternatives only",
                "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            },
            "optimization": {
                "hidden_units": config.hidden_units,
                "activation": "tanh",
                "optimizer": "Adam",
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size_cases": config.batch_size,
                "max_epochs": config.max_epochs,
                "patience": config.patience,
                "min_delta": config.min_delta,
                "seed": config.seed,
                "selected_epoch": fit.selected_epoch,
                "epochs_run": fit.epochs_run,
                "baseline_validation_nll": fit.baseline_validation_nll,
                "selected_validation_nll": fit.validation_nll,
                "selected_train_nll": fit.train_nll,
            },
            "validation_in_final_fit": False,
            "uncertainty": "conditional on fitted N model; observed respondents and items resampled",
            "sources": {str(path.relative_to(REPO)): _sha256(path) for path in sources},
            "outputs": {
                path.name: _sha256(path)
                for path in temporary_root.iterdir()
                if path.name != "manifest.json"
            },
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(temporary_root.iterdir()):
            if path.name != "manifest.json":
                path.replace(args.output_root / path.name)
        (temporary_root / "manifest.json").replace(args.output_root / "manifest.json")

    print(
        f"selected epoch: {fit.selected_epoch}; validation NLL "
        f"{fit.baseline_validation_nll:.6f} -> {fit.validation_nll:.6f}"
    )
    print(f"neural benchmark: {args.output_root}")


if __name__ == "__main__":
    main()
