"""Sociodemographic strata — step 1.4 of docs/plan_article.md (§2.1, §3.5, §5).

A persona is a *cell*, not a person: the unit of analysis is a combination of
canonical SES levels, and its empirical counterpart is the set of real
respondents that fall in it. This module answers two questions:

    which dimensions define a cell?   -> ``load_definition()``
    who falls in which cell?          -> ``respondent_cell()`` / ``survey_cells()``

Counting rule (the part that must not be silent)
------------------------------------------------
``SesCrosswalk`` returns a ``SesValue`` that can be in three states. They are
kept apart here instead of being merged into a cell:

* **resolved** — exactly one canonical level. The respondent enters the cell.
* **coarse** — the source code is coarser than the canonical schema, so the
  respondent belongs to a *set* of levels (``eeq_2018`` age cohorts, 1998
  education). Assigning him to one of them would invent data; spreading him
  over all of them would double-count. He is **excluded from every cell** and
  reported under ``n_coarse``.
* **missing** — non-response, unmapped code, or dimension absent from the
  survey. Excluded, reported under ``n_missing``.

``n_coarse`` and ``n_missing`` are therefore a first-class output of the
diagnostics, never a silent loss. A combination that only looks viable because
it quietly drops a third of the corpus is visible as a low ``pct_assigned``.

Weights
-------
``__weight`` is already mean-1 within each survey in the Blob Parquets, so a
weighted cell size is on the same scale as a headcount. Two weighted figures
are produced per cell: ``n_weighted`` (sum of weights) and ``n_eff``, the Kish
effective sample size ``(sum w)^2 / sum w^2`` — the honest denominator for the
sampling error of the observed distribution in §5.

No network access beyond the Blob read already performed by ``blob.read_survey``
(cached under ``data/cache/``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

from . import blob
from .ses import MISSING, SesCrosswalk, SesValue

_REPO = Path(__file__).resolve().parents[3]
DEFINITION_PATH = _REPO / "data" / "strata_definition.json"
DIAGNOSTICS_PATH = _REPO / "data" / "strata_diagnostics.csv"

#: Sentinels used in the resolved profile table. Never canonical level ids.
COARSE = "__coarse__"
UNRESOLVED = (COARSE, MISSING)

#: Candidate dimensions, in the a priori priority order of §3.5.
CANDIDATE_DIMENSIONS: tuple[str, ...] = (
    "age", "gender", "education", "region_qc", "region", "income", "language",
)

#: ``region`` and ``region_qc`` measure the same axis at two granularities;
#: a combination never holds both.
MUTUALLY_EXCLUSIVE: tuple[frozenset[str], ...] = (frozenset({"region", "region_qc"}),)

#: Surveys whose sample is national rather than Quebec-only. Relevant because
#: ``region`` only varies there, and ``region_qc`` is absent from all of them.
NATIONAL_SURVEYS: frozenset[str] = frozenset({
    "cecd_elxn_can_2011", "cecd_sante_can_usa",
    "ces_2019_online", "ces_2019_phone", "ces_2021", "ces_2025",
})


# --------------------------------------------------------------------------
# the retained definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StrataDefinition:
    """The strata design retained in ``data/strata_definition.json``."""

    dimensions: tuple[str, ...]
    min_cell_n: int
    scope: str                      # "per_survey" | "stacked"
    coarse_policy: str
    missing_policy: str
    notes: str = ""
    per_survey_dimensions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rationale: dict[str, Any] = field(default_factory=dict)

    def dimensions_for(self, survey_id: str | None = None) -> tuple[str, ...]:
        """Dimensions to use — a survey may carry a documented override."""
        if survey_id is not None and survey_id in self.per_survey_dimensions:
            return self.per_survey_dimensions[survey_id]
        return self.dimensions


@lru_cache(maxsize=1)
def load_definition(path: Path | str = DEFINITION_PATH) -> StrataDefinition:
    """Load the retained strata definition."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    retained = payload["retained"]
    return StrataDefinition(
        dimensions=tuple(retained["dimensions"]),
        min_cell_n=int(retained["min_cell_n"]),
        scope=retained["scope"],
        coarse_policy=payload["counting_rule"]["coarse"],
        missing_policy=payload["counting_rule"]["missing"],
        notes=retained.get("notes", ""),
        per_survey_dimensions={
            k: tuple(v) for k, v in retained.get("per_survey_dimensions", {}).items()
        },
        rationale=payload.get("rationale", {}),
    )


def candidate_combinations(sizes: Sequence[int] = (2, 3, 4),
                           dimensions: Sequence[str] = CANDIDATE_DIMENSIONS,
                           ) -> list[tuple[str, ...]]:
    """All admissible combinations of candidate dimensions, in priority order."""
    order = {d: i for i, d in enumerate(dimensions)}
    out: list[tuple[str, ...]] = []
    for size in sizes:
        for combo in combinations(dimensions, size):
            if any(len(excl & set(combo)) > 1 for excl in MUTUALLY_EXCLUSIVE):
                continue
            out.append(tuple(sorted(combo, key=order.__getitem__)))
    return out


# --------------------------------------------------------------------------
# respondent -> cell
# --------------------------------------------------------------------------

def resolve_value(value: SesValue | None) -> str:
    """Collapse a ``SesValue`` to a level id, ``COARSE`` or ``MISSING``."""
    if value is None:
        return MISSING
    if value.is_missing:
        return MISSING
    if value.coarse or len(value.levels) != 1:
        return COARSE
    return value.levels[0]


def respondent_cell(survey_id: str, row: Mapping[str, Any],
                    dimensions: Sequence[str] | None = None,
                    crosswalk: SesCrosswalk | None = None) -> tuple[str, ...] | None:
    """Cell of one respondent, or ``None`` when any dimension is unresolved.

    ``None`` is the deliberate answer for a coarse or missing dimension: the
    respondent contributes to no cell (see module docstring).
    """
    xw = crosswalk or _crosswalk()
    dims = tuple(dimensions) if dimensions is not None else load_definition().dimensions_for(survey_id)
    cell: list[str] = []
    for dim in dims:
        level = resolve_value(xw.apply(survey_id, dim, row))
        if level in UNRESOLVED:
            return None
        cell.append(level)
    return tuple(cell)


def cell_key(cell: Iterable[str]) -> str:
    """Stable string id of a cell, e.g. ``35_44|woman|bachelor``."""
    return "|".join(cell)


@lru_cache(maxsize=1)
def _crosswalk() -> SesCrosswalk:
    return SesCrosswalk.load()


@lru_cache(maxsize=32)
def resolved_profiles(survey_id: str) -> pl.DataFrame:
    """One row per respondent, one column per canonical dimension.

    Values are canonical level ids, or the ``COARSE`` / ``MISSING`` sentinels.
    Kept in an LRU cache: the crosswalk is applied once per survey and reused
    by every combination of dimensions.
    """
    xw = _crosswalk()
    dims = CANDIDATE_DIMENSIONS
    needed: set[str] = set()
    for dim in dims:
        needed.update(xw.source_variables(survey_id, dim))
    available = set(blob.survey_columns(survey_id))
    columns = ["__respondent_id", "__weight"] + sorted(needed & available)
    frame = blob.read_survey(survey_id, columns=[c for c in columns if c in available])

    rows = frame.to_dicts()
    data: dict[str, list] = {dim: [] for dim in dims}
    for row in rows:
        for dim in dims:
            data[dim].append(resolve_value(xw.apply(survey_id, dim, row)))

    return pl.DataFrame({
        "__survey_id": [survey_id] * len(rows),
        "__respondent_id": frame["__respondent_id"].cast(pl.Utf8),
        "__weight": frame["__weight"].cast(pl.Float64),
        **data,
    })


def corpus_profiles(survey_ids: Sequence[str] | None = None) -> pl.DataFrame:
    """Resolved profiles of the whole perimeter, stacked."""
    ids = list(survey_ids) if survey_ids is not None else perimeter_surveys()
    return pl.concat([resolved_profiles(s) for s in ids], how="vertical")


def perimeter_surveys() -> list[str]:
    """The 17 in-perimeter surveys that have a crosswalk and microdata."""
    xw = _crosswalk()
    return sorted(set(xw.survey_ids) & set(blob.available_surveys()))


# --------------------------------------------------------------------------
# cells and their sizes
# --------------------------------------------------------------------------

def cells_from_profiles(profiles: pl.DataFrame,
                        dimensions: Sequence[str]) -> pl.DataFrame:
    """Non-empty cells with their unweighted, weighted and effective sizes."""
    dims = list(dimensions)
    assigned = profiles.filter(
        pl.all_horizontal([~pl.col(d).is_in(UNRESOLVED) for d in dims])
    )
    if assigned.height == 0:
        return pl.DataFrame(schema={
            **{d: pl.Utf8 for d in dims},
            "cell": pl.Utf8, "n": pl.Int64,
            "n_weighted": pl.Float64, "n_eff": pl.Float64,
        })
    out = (
        assigned.group_by(dims)
        .agg(
            n=pl.len(),
            n_weighted=pl.col("__weight").sum(),
            _w2=(pl.col("__weight") ** 2).sum(),
        )
        .with_columns(
            n_eff=pl.when(pl.col("_w2") > 0)
            .then(pl.col("n_weighted") ** 2 / pl.col("_w2"))
            .otherwise(0.0),
            cell=pl.concat_str([pl.col(d) for d in dims], separator="|"),
        )
        .drop("_w2")
        .sort("n", descending=True)
    )
    return out.select(dims + ["cell", "n", "n_weighted", "n_eff"])


def survey_cells(survey_id: str, dimensions: Sequence[str] | None = None) -> pl.DataFrame:
    """Enumerate the non-empty cells of one survey with their counts."""
    dims = tuple(dimensions) if dimensions is not None else \
        load_definition().dimensions_for(survey_id)
    return cells_from_profiles(resolved_profiles(survey_id), dims)


def corpus_cells(dimensions: Sequence[str] | None = None,
                 survey_ids: Sequence[str] | None = None) -> pl.DataFrame:
    """Enumerate the cells of the stacked corpus."""
    dims = tuple(dimensions) if dimensions is not None else load_definition().dimensions
    return cells_from_profiles(corpus_profiles(survey_ids), dims)


def theoretical_cell_count(dimensions: Sequence[str],
                           crosswalk: SesCrosswalk | None = None) -> int:
    """Product of the canonical level counts, excluding ``missing``."""
    xw = crosswalk or _crosswalk()
    total = 1
    for dim in dimensions:
        total *= sum(1 for lv in xw.levels(dim) if lv != MISSING)
    return total


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

THRESHOLDS: tuple[int, ...] = (30, 50, 100)


def diagnose(profiles: pl.DataFrame, dimensions: Sequence[str], scope: str,
             thresholds: Sequence[int] = THRESHOLDS) -> dict[str, Any]:
    """One diagnostics row for a (scope, combination) pair."""
    dims = list(dimensions)
    total = profiles.height
    total_w = float(profiles["__weight"].sum()) if total else 0.0

    any_missing = pl.any_horizontal([pl.col(d) == MISSING for d in dims])
    any_coarse = pl.any_horizontal([pl.col(d) == COARSE for d in dims])
    flagged = profiles.select(
        _missing=any_missing,
        _coarse=any_coarse & ~any_missing,
        _w=pl.col("__weight"),
    )
    n_missing = int(flagged["_missing"].sum())
    n_coarse = int(flagged["_coarse"].sum())

    cells = cells_from_profiles(profiles, dims)
    n_assigned = int(cells["n"].sum()) if cells.height else 0
    w_assigned = float(cells["n_weighted"].sum()) if cells.height else 0.0

    row: dict[str, Any] = {
        "scope": scope,
        "n_dims": len(dims),
        "dimensions": "+".join(dims),
        "n_respondents": total,
        "n_cells_theoretical": theoretical_cell_count(dims),
        "n_cells_nonempty": cells.height,
        "n_assigned": n_assigned,
        "n_coarse": n_coarse,
        "n_missing": n_missing,
        "pct_assigned": _pct(n_assigned, total),
        "pct_coarse": _pct(n_coarse, total),
        "pct_missing": _pct(n_missing, total),
        "pct_weight_assigned": _pct(w_assigned, total_w),
    }

    for prefix, col in (("n", "n"), ("w", "n_weighted"), ("eff", "n_eff")):
        series = cells[col] if cells.height else None
        row[f"{prefix}_median"] = _q(series, 0.5)
        row[f"{prefix}_p10"] = _q(series, 0.10)
        row[f"{prefix}_p25"] = _q(series, 0.25)
        row[f"{prefix}_p75"] = _q(series, 0.75)

    for thr in thresholds:
        if cells.height:
            usable = cells.filter(pl.col("n") >= thr)
            row[f"n_cells_lt{thr}"] = cells.height - usable.height
            row[f"pct_cells_lt{thr}"] = _pct(cells.height - usable.height, cells.height)
            row[f"n_cells_ge{thr}"] = usable.height
            row[f"pct_resp_in_cell_ge{thr}"] = _pct(int(usable["n"].sum()), total)
            row[f"pct_weight_in_cell_ge{thr}"] = _pct(
                float(usable["n_weighted"].sum()), total_w)
            eff_usable = cells.filter(pl.col("n_eff") >= thr)
            row[f"n_cells_eff_ge{thr}"] = eff_usable.height
        else:
            row[f"n_cells_lt{thr}"] = 0
            row[f"pct_cells_lt{thr}"] = None
            row[f"n_cells_ge{thr}"] = 0
            row[f"pct_resp_in_cell_ge{thr}"] = 0.0
            row[f"pct_weight_in_cell_ge{thr}"] = 0.0
            row[f"n_cells_eff_ge{thr}"] = 0
    return row


def _pct(part: float, whole: float) -> float | None:
    return None if not whole else round(100.0 * part / whole, 2)


def _q(series: pl.Series | None, q: float) -> float | None:
    if series is None or series.len() == 0:
        return None
    value = series.quantile(q, interpolation="linear")
    return None if value is None else round(float(value), 1)


__all__ = [
    "CANDIDATE_DIMENSIONS", "COARSE", "DEFINITION_PATH", "DIAGNOSTICS_PATH",
    "MISSING", "NATIONAL_SURVEYS", "THRESHOLDS", "UNRESOLVED",
    "StrataDefinition", "candidate_combinations", "cell_key",
    "cells_from_profiles", "corpus_cells", "corpus_profiles", "diagnose",
    "load_definition", "perimeter_surveys", "resolve_value", "resolved_profiles",
    "respondent_cell", "survey_cells", "theoretical_cell_count",
]
