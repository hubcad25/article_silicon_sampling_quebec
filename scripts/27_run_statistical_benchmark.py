"""Fit benchmark S and predict the frozen pilot item-cell distributions.

The conditional logit is trained on the same 8,000 C0 examples as the LLM.
Each alternative is represented by a frozen Azure text-embedding-3-large
vector of ``profile + question + options + selected alternative``. The L2
penalty is selected on the existing 500-example validation set; validation
examples are never added to the training fit.

Usage:
    source .venv/bin/activate
    python scripts/27_run_statistical_benchmark.py
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
from typing import Iterable

import numpy as np
import polars as pl
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.conditional_logit import (  # noqa: E402
    ChoiceData,
    fit_conditional_logit,
    probabilities,
)
from article_silicon_sampling_quebec.dataset import read_jsonl  # noqa: E402
from article_silicon_sampling_quebec.inference import build_item_cells  # noqa: E402
from article_silicon_sampling_quebec.prompts import PromptTemplate  # noqa: E402

DATASET_ROOT = REPO / "data" / "datasets"
OUTPUT_ROOT = REPO / "data" / "analysis" / "statistical_benchmark"
DEFAULT_PENALTIES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
CHOICE_MARKER = "[SELECTED ALTERNATIVE]"


def _embedding_module(cache_path: Path):
    spec = importlib.util.spec_from_file_location(
        "build_similarity_index", REPO / "scripts" / "12_build_similarity_index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CACHE_PATH = cache_path
    return module


def _options_from_prompt(user: str) -> list[str]:
    lines = user.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Options :") + 1
    except StopIteration as exc:
        raise ValueError("target prompt has no 'Options :' block") from exc
    options = [line.strip()[2:].strip() for line in lines[start:] if line.strip().startswith("- ")]
    if len(options) < 2:
        raise ValueError("target prompt has fewer than two alternatives")
    return options


def alternative_text(system: str, user: str, option: str) -> str:
    return f"{system}\n\n{user}\n\n{CHOICE_MARKER}\n{option}"


def examples_to_texts(examples: Iterable[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts: list[str] = []
    offsets = [0]
    chosen: list[int] = []
    for example in examples:
        messages = example["messages"]
        system, user, answer = (messages[i]["content"] for i in range(3))
        options = _options_from_prompt(user)
        matches = [i for i, option in enumerate(options) if option == answer]
        if len(matches) != 1:
            raise ValueError(f"assistant answer does not identify one alternative: {answer!r}")
        texts.extend(alternative_text(system, user, option) for option in options)
        offsets.append(len(texts))
        chosen.append(matches[0])
    return texts, np.asarray(offsets, dtype=np.int64), np.asarray(chosen, dtype=np.int64)


def prediction_texts() -> tuple[list[str], np.ndarray, list[tuple[int, str, list[str]]]]:
    template = PromptTemplate(condition="C0", ses_dropout="none")
    texts: list[str] = []
    offsets = [0]
    groups: list[tuple[int, str, list[str]]] = []
    for pair in build_item_cells():
        messages = template.build_messages(pair.persona, pair.item, dimensions=pair.dimensions)
        system, user = (message["content"] for message in messages)
        labels = [option.label for option in pair.item.options]
        texts.extend(alternative_text(system, user, label) for label in labels)
        offsets.append(len(texts))
        groups.append((pair.item_idx, pair.cell, [option.code for option in pair.item.options]))
    return texts, np.asarray(offsets, dtype=np.int64), groups


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=REPO / "data" / ".cache" / "benchmark_s_embedding_cache.parquet",
        help="Persistent embedding cache (Parquet is justified by the 3,072-vector column).",
    )
    parser.add_argument("--penalties", nargs="+", type=float, default=DEFAULT_PENALTIES)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Report embedding cost without fitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO / ".env")
    required = ["AOAI_ENDPOINT", "AOAI_KEY", "AOAI_EMBED_DEPLOYMENT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing Azure embedding settings: {', '.join(missing)}")

    train_path = args.dataset_root / "c0_train_8000.jsonl"
    validation_path = args.dataset_root / "c0_validation.jsonl"
    train_texts, train_offsets, train_chosen = examples_to_texts(read_jsonl(train_path))
    validation_texts, validation_offsets, validation_chosen = examples_to_texts(
        read_jsonl(validation_path)
    )
    test_texts, test_offsets, test_groups = prediction_texts()
    all_texts = train_texts + validation_texts + test_texts
    embedding_module = _embedding_module(args.embedding_cache)
    embed = embedding_module.embed_texts
    vectors = embed(all_texts, dry_run=args.dry_run)

    train_end = len(train_texts)
    validation_end = train_end + len(validation_texts)
    train = ChoiceData(vectors[:train_end], train_offsets, train_chosen)
    validation = ChoiceData(
        vectors[train_end:validation_end], validation_offsets, validation_chosen
    )
    best, fits = fit_conditional_logit(
        train, validation, args.penalties, max_iter=args.max_iter
    )
    test_probabilities = probabilities(
        vectors[validation_end:], test_offsets, best.coefficients
    )
    if len(test_groups) != 275:
        raise RuntimeError(f"expected 275 test item-cell groups, got {len(test_groups)}")

    distribution_rows = []
    diagnostic_rows = []
    for group_index, (item_idx, cell, codes) in enumerate(test_groups):
        start, stop = test_offsets[group_index:group_index + 2]
        shares = test_probabilities[start:stop]
        distribution_rows.extend({
            "arm": "S", "item_idx": item_idx, "cell": cell, "temperature": 1.0,
            "code": code, "share": float(share), "n": None,
        } for code, share in zip(codes, shares, strict=True))
        diagnostic_rows.append({
            "arm": "S", "item_idx": item_idx, "cell": cell, "temperature": 1.0,
            "transport_n": None, "effective_n": None, "invalid_n": 0,
            "invalid_rate": 0.0, "coverage_ok": True,
        })

    distributions = pl.DataFrame(distribution_rows).with_columns(pl.col("n").cast(pl.Int64))
    diagnostics = pl.DataFrame(diagnostic_rows).with_columns(
        pl.col("transport_n").cast(pl.Int64), pl.col("effective_n").cast(pl.Int64)
    )
    if distributions.select("item_idx", "cell").unique().height != 275:
        raise RuntimeError("benchmark distributions do not cover 275 unique item-cell groups")
    if diagnostics.height != 275:
        raise RuntimeError("benchmark diagnostics do not cover 275 item-cell groups")
    tuning = pl.DataFrame([{
        "penalty": fit.penalty, "train_nll": fit.train_nll,
        "validation_nll": fit.validation_nll, "converged": fit.converged,
        "iterations": fit.iterations, "selected": fit.penalty == best.penalty,
    } for fit in fits])
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_root.parent) as temporary_dir:
        temporary_root = Path(temporary_dir)
        distributions.write_csv(temporary_root / "distributions.csv")
        diagnostics.write_csv(temporary_root / "distribution_diagnostics.csv")
        np.savez_compressed(temporary_root / "model.npz", coefficients=best.coefficients)
        tuning.write_csv(temporary_root / "tuning.csv")
        sources = [
            train_path,
            validation_path,
            REPO / "data/split/heldout_items.json",
            REPO / "data/analysis/test_blocks.csv",
            REPO / "data/items.parquet",
            REPO / "data/strata_definition.json",
            REPO / "data/split/heldout_respondents.parquet",
            REPO / "data/crosswalks/ses_crosswalk.json",
        ]
        manifest = {
            "version": "1.0",
            "arm": "S",
            "model": "L2-regularized conditional logit",
            "embedding_deployment": os.environ["AOAI_EMBED_DEPLOYMENT"],
            "embedding_api_version": embedding_module.AOAI_API_VERSION,
            "embedding_dtype": str(vectors.dtype),
            "embedding_matrix_order": "train, validation, test; source alternative order",
            "embedding_dimensions": int(train.features.shape[1]),
            "embedding_matrix_sha256": _array_sha256(vectors),
            "alternative_text": "C0 profile + target prompt + selected alternative",
            "training_examples": train.n_cases,
            "validation_examples": validation.n_cases,
            "test_item_cells": len(test_groups),
            "penalties": list(args.penalties),
            "selected_penalty": best.penalty,
            "selection_metric": "mean validation negative log-likelihood",
            "validation_in_final_fit": False,
            "uncertainty": "conditional on fitted S model; observed respondents and items resampled",
            "sources": {
                str(path.relative_to(REPO)): _sha256(path) for path in sources
            },
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
            if path.name == "manifest.json":
                continue
            path.replace(args.output_root / path.name)
        (temporary_root / "manifest.json").replace(args.output_root / "manifest.json")
    print(f"selected L2 penalty: {best.penalty:g} (validation NLL={best.validation_nll:.6f})")
    print(f"statistical benchmark: {args.output_root}")


if __name__ == "__main__":
    main()
