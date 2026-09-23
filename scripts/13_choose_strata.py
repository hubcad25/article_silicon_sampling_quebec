"""Step 1.4 — empirical choice of the strata dimensions (§3.5 of the plan).

Evaluates every admissible combination of 2, 3 and 4 candidate dimensions,
on the stacked corpus, on the Quebec-only sub-corpus and survey by survey,
and writes:

    data/strata_diagnostics.csv   every combination x every scope
    data/strata_definition.json   the retained design (written by hand after
                                  reading the diagnostics; --write-definition
                                  refreshes its embedded figures)

Usage::

    .venv/bin/python scripts/13_choose_strata.py
    .venv/bin/python scripts/13_choose_strata.py --top       # summary table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from article_silicon_sampling_quebec.corpus import strata  # noqa: E402

SCOPE_CORPUS = "CORPUS"
SCOPE_QC = "CORPUS_QC"


def build_diagnostics(sizes: tuple[int, ...] = (2, 3, 4)) -> pl.DataFrame:
    surveys = strata.perimeter_surveys()
    combos = strata.candidate_combinations(sizes)
    print(f"{len(surveys)} surveys, {len(combos)} combinations", file=sys.stderr)

    profiles = {s: strata.resolved_profiles(s) for s in surveys}
    for s in surveys:
        print(f"  resolved {s}: {profiles[s].height} respondents", file=sys.stderr)

    stacked = pl.concat(list(profiles.values()), how="vertical")
    qc_only = stacked.filter(~pl.col("__survey_id").is_in(list(strata.NATIONAL_SURVEYS)))

    rows = []
    for combo in combos:
        rows.append(strata.diagnose(stacked, combo, SCOPE_CORPUS))
        rows.append(strata.diagnose(qc_only, combo, SCOPE_QC))
        for survey_id in surveys:
            rows.append(strata.diagnose(profiles[survey_id], combo, survey_id))
    return pl.DataFrame(rows).sort(["n_dims", "dimensions", "scope"])


def summary(diag: pl.DataFrame, scope: str = SCOPE_CORPUS) -> pl.DataFrame:
    return (
        diag.filter(pl.col("scope") == scope)
        .select("n_dims", "dimensions", "n_cells_theoretical", "n_cells_nonempty",
                "pct_assigned", "pct_coarse", "pct_missing", "n_median", "n_p25",
                "n_cells_ge50", "pct_resp_in_cell_ge50", "pct_weight_in_cell_ge50")
        .sort(["n_dims", "n_median"], descending=[False, True])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(strata.DIAGNOSTICS_PATH))
    parser.add_argument("--top", action="store_true", help="print the summary table")
    args = parser.parse_args()

    diag = build_diagnostics()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    diag.write_csv(args.out)
    print(f"wrote {args.out} ({diag.height} rows)", file=sys.stderr)

    if args.top:
        pl.Config.set_tbl_rows(200)
        pl.Config.set_tbl_width_chars(220)
        for scope in (SCOPE_CORPUS, SCOPE_QC):
            print(f"\n=== {scope} ===")
            print(summary(diag, scope))


if __name__ == "__main__":
    main()
