"""Tests for the strata module (step 1.4). No network access."""

from __future__ import annotations

import json

import polars as pl
import pytest

from article_silicon_sampling_quebec.corpus import strata
from article_silicon_sampling_quebec.corpus.ses import MISSING, SesValue


# --------------------------------------------------------------------------
# counting rule
# --------------------------------------------------------------------------

def test_resolve_value_keeps_the_three_states_apart():
    assert strata.resolve_value(SesValue("age", ("35_44",))) == "35_44"
    assert strata.resolve_value(SesValue("age", (MISSING,))) == MISSING
    assert strata.resolve_value(None) == MISSING
    # a coarse mapping is a set of levels: never collapsed to one of them
    coarse = SesValue("age", ("25_34", "35_44"), coarse=True)
    assert strata.resolve_value(coarse) == strata.COARSE
    # coarse flag alone is enough, even on a single level
    assert strata.resolve_value(SesValue("age", ("25_34",), coarse=True)) == strata.COARSE


def test_sentinels_are_not_canonical_levels():
    xw = strata._crosswalk()
    for dim in strata.CANDIDATE_DIMENSIONS:
        assert strata.COARSE not in xw.levels(dim)


def test_respondent_cell_returns_none_when_any_dimension_is_unresolved(monkeypatch):
    values = {
        "age": SesValue("age", ("35_44",)),
        "gender": SesValue("gender", ("woman",)),
        "education": SesValue("education", ("bachelor", "above_bachelor"), coarse=True),
    }

    class FakeXw:
        def apply(self, survey_id, dim, row):
            return values[dim]

    dims = ("age", "gender", "education")
    assert strata.respondent_cell("s", {}, dims, FakeXw()) is None
    values["education"] = SesValue("education", ("bachelor",))
    assert strata.respondent_cell("s", {}, dims, FakeXw()) == ("35_44", "woman", "bachelor")
    values["gender"] = SesValue("gender", (MISSING,))
    assert strata.respondent_cell("s", {}, dims, FakeXw()) is None


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

@pytest.fixture
def toy_profiles() -> pl.DataFrame:
    return pl.DataFrame({
        "__survey_id": ["toy"] * 6,
        "__respondent_id": [str(i) for i in range(6)],
        "__weight": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
        "age": ["35_44", "35_44", "45_54", strata.COARSE, "35_44", "45_54"],
        "gender": ["man", "woman", "man", "man", "man", MISSING],
    })


def test_cells_from_profiles_excludes_coarse_and_missing(toy_profiles):
    cells = strata.cells_from_profiles(toy_profiles, ["age", "gender"])
    assert cells["n"].sum() == 4          # rows 3 (coarse) and 5 (missing) dropped
    sizes = dict(zip(cells["cell"], cells["n"]))
    assert sizes == {"35_44|man": 2, "35_44|woman": 1, "45_54|man": 1}


def test_weighted_and_effective_sizes(toy_profiles):
    cells = strata.cells_from_profiles(toy_profiles, ["age", "gender"])
    row = cells.filter(pl.col("cell") == "45_54|man").row(0, named=True)
    assert row["n"] == 1
    assert row["n_weighted"] == pytest.approx(2.0)
    assert row["n_eff"] == pytest.approx(1.0)      # one unit, whatever its weight
    row = cells.filter(pl.col("cell") == "35_44|man").row(0, named=True)
    assert row["n_eff"] == pytest.approx(2.0)      # two equal weights


def test_empty_result_keeps_the_schema(toy_profiles):
    broken = toy_profiles.with_columns(age=pl.lit(strata.COARSE))
    cells = strata.cells_from_profiles(broken, ["age", "gender"])
    assert cells.height == 0
    assert cells.columns == ["age", "gender", "cell", "n", "n_weighted", "n_eff"]


def test_diagnose_accounts_for_every_respondent(toy_profiles):
    row = strata.diagnose(toy_profiles, ["age", "gender"], "toy")
    assert row["n_assigned"] + row["n_coarse"] + row["n_missing"] == row["n_respondents"]
    assert row["n_cells_nonempty"] == 3
    assert row["pct_assigned"] == pytest.approx(100 * 4 / 6, abs=0.01)
    # a respondent missing on one dimension and coarse on another counts once,
    # under missing
    assert row["n_missing"] == 1
    assert row["n_coarse"] == 1


# --------------------------------------------------------------------------
# combinations
# --------------------------------------------------------------------------

def test_candidate_combinations_never_pair_region_with_region_qc():
    combos = strata.candidate_combinations((2, 3, 4))
    assert all(not {"region", "region_qc"} <= set(c) for c in combos)
    assert len(combos) == 75
    assert combos[0] == ("age", "gender")   # priority order preserved


def test_theoretical_cell_count_excludes_missing():
    # 8 age brackets x 3 genders x 5 education levels
    assert strata.theoretical_cell_count(["age", "gender", "education"]) == 120


# --------------------------------------------------------------------------
# the retained definition
# --------------------------------------------------------------------------

def test_definition_loads_and_is_consistent():
    definition = strata.load_definition()
    assert definition.dimensions == ("age", "gender", "education")
    assert definition.min_cell_n == 50
    assert definition.scope == "per_survey"
    assert definition.dimensions_for("ces_2021") == ("age", "gender", "education")
    # a survey whose age is 100% coarse falls back to a definition without age
    assert "age" not in definition.dimensions_for("eeq_2018")
    assert definition.dimensions_for("cecd_elxn_qc_1998") == ("age", "gender")


def test_definition_only_names_known_dimensions_and_surveys():
    definition = strata.load_definition()
    with open(strata.DEFINITION_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    xw = strata._crosswalk()
    all_dims = set(definition.dimensions)
    for dims in definition.per_survey_dimensions.values():
        all_dims |= set(dims)
    assert all_dims <= set(strata.CANDIDATE_DIMENSIONS)
    assert set(definition.per_survey_dimensions) <= set(xw.survey_ids)
    tiers = payload["validation_tiers"]
    covered = {s for key in ("A_primary", "B_fallback", "C_age_broken")
               for s in tiers[key]["surveys"]}
    assert covered == set(xw.survey_ids)      # every survey is placed in a tier


def test_diagnostics_csv_matches_the_module():
    diag = pl.read_csv(strata.DIAGNOSTICS_PATH)
    assert set(strata.candidate_combinations((2, 3, 4))) == {
        tuple(d.split("+")) for d in diag["dimensions"].unique()
    }
    # the three states partition the respondents in every row
    assert (diag["n_assigned"] + diag["n_coarse"] + diag["n_missing"]
            == diag["n_respondents"]).all()
