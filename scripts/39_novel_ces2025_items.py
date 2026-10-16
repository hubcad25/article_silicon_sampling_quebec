"""CES 2025 items never asked before: candidates for a second benchmark set.

For each substantive CES 2025 item, find its nearest item in the 8 training
surveys (script 36) two ways: same variable stem (cps25_X vs cps21_X / cps19_X)
and embedding cosine (same recipe as script 36). An item with no stem match and
a low max cosine is a "novel" candidate.

Output: data/analysis/ces2025_novelty.csv, one row per CES 2025 item, sorted by
max cosine (most novel first).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s36", REPO / "scripts/36_generate_benchmark_datasets.py")
s36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s36)

OUT = REPO / "data" / "analysis" / "ces2025_novelty.csv"
#: Below this cosine no prior item reads as the same question (CONTEXT_MAX_COSINE
#: is 0.95 for near-duplicates; 0.80 is a looser "same topic and form" line).
NOVEL_MAX_COSINE = 0.80


def stem(variable: str) -> str:
    return variable.split("_", 1)[1] if variable.startswith(("cps", "pes")) else variable


def main() -> None:
    train = s36.load_items()
    frames = [pl.read_parquet(p) for p in s36.ITEMS]
    ces25 = (pl.concat(frames, how="diagonal_relaxed")
             .filter((pl.col("survey_id") == "ces_2025") & ~pl.col("is_context_only").fill_null(False))
             .unique(["survey_id", "variable"], keep="last", maintain_order=True))
    both = pl.concat([train, ces25], how="diagonal_relaxed")
    vectors = s36.l2_normalize(s36.item_vectors(both))
    t, c = vectors[:train.height], vectors[train.height:]
    sims = c @ t.T
    best = sims.argmax(axis=1)

    ces_stems = {(s, stem(v)) for s, v in zip(train["survey_id"], train["variable"])
                 if s.startswith("ces_")}
    rows = []
    for i in range(ces25.height):
        var = ces25["variable"][i]
        j = int(best[i])
        rows.append({
            "variable": var,
            "question_text": ces25["question_text"][i],
            "n_options": ces25["n_options"][i],
            "n_valid": ces25["n_valid_responses"][i],
            "stem_in_ces2021": ("ces_2021", stem(var)) in ces_stems,
            "stem_in_ces2019": ("ces_2019_online", stem(var)) in ces_stems,
            "max_cosine": round(float(sims[i, j]), 3),
            "nearest_survey": train["survey_id"][j],
            "nearest_variable": train["variable"][j],
            "nearest_text": train["question_text"][j],
        })
    out = pl.DataFrame(rows).with_columns(
        novel=~pl.col("stem_in_ces2021") & ~pl.col("stem_in_ces2019")
        & (pl.col("max_cosine") < NOVEL_MAX_COSINE)).sort("max_cosine")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT)
    print(f"{out.height} CES 2025 items, {out['novel'].sum()} novel -> {OUT}")


if __name__ == "__main__":
    main()
