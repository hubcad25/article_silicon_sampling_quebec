"""Second split of the held-out respondents: a context half and an evaluation half.

Arm BS (ADR 0001, decision 7) shows the C1 model the cell's distributions on
neighbouring items computed on the **context** half only, and is evaluated on
the **eval** half only — no respondent is both context and ground truth.

The split is drawn within each survey x cell, so every cell is halved
(``n // 2`` context, the rest eval), with a seed and a per-group generator
derived from it: deterministic, and a change in one cell never reshuffles
another. Written once, then frozen like the rest of ``data/split/``.

    python scripts/26_split_heldout_halves.py            # refuses to overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.inference import HELDOUT_HALVES_PATH, HELDOUT_PATH  # noqa: E402

SEED = 20260925


def group_seed(*parts: str) -> int:
    digest = hashlib.sha256(chr(0).join((str(SEED), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build(heldout: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (survey_id, cell), group in heldout.group_by(["__survey_id", "cell"]):
        ids = sorted(str(r) for r in group["__respondent_id"])
        order = np.random.default_rng(group_seed(survey_id, cell)).permutation(len(ids))
        n_context = len(ids) // 2
        for rank, position in enumerate(order):
            rows.append({"__survey_id": survey_id, "__respondent_id": ids[position],
                         "cell": cell, "half": "context" if rank < n_context else "eval"})
    return pl.DataFrame(rows).sort("__survey_id", "cell", "__respondent_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite the frozen split.")
    args = parser.parse_args()
    if HELDOUT_HALVES_PATH.exists() and not args.force:
        raise SystemExit(f"{HELDOUT_HALVES_PATH} exists: the split is frozen")
    heldout = pl.read_parquet(HELDOUT_PATH).with_columns(pl.col("__respondent_id").cast(pl.Utf8))
    halves = build(heldout)
    if halves.height != heldout.height or halves.select(
            "__survey_id", "__respondent_id").n_unique() != heldout.height:
        raise SystemExit("split does not partition the held-out respondents")
    halves.write_csv(HELDOUT_HALVES_PATH)
    counts = halves.group_by("half").len().sort("half")
    print(f"{HELDOUT_HALVES_PATH}: {dict(counts.iter_rows())}")


if __name__ == "__main__":
    main()
