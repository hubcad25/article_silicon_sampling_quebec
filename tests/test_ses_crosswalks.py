"""Tests for the SES crosswalks (step 1.3). No network access."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from article_silicon_sampling_quebec.corpus import ses as ses_mod
from article_silicon_sampling_quebec.corpus.perimeter import (
    EXCLUDED_SURVEYS,
    NORMALIZED_DIR,
)
from article_silicon_sampling_quebec.corpus.ses import (
    MISSING,
    SesCrosswalk,
    declared_levels,
    normalize_code,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def canonical() -> dict:
    return json.loads(ses_mod.CANONICAL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def crosswalk() -> dict:
    return json.loads(ses_mod.CROSSWALK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def xw() -> SesCrosswalk:
    return SesCrosswalk.load()


def perimeter_surveys() -> set[str]:
    directory = (REPO / NORMALIZED_DIR).resolve()
    if not directory.is_dir():
        pytest.skip(f"catalogue normalisé absent: {directory}")
    return {p.stem for p in directory.glob("*.json")} - set(EXCLUDED_SURVEYS)


# --------------------------------------------------------------------------
# Canonical schema
# --------------------------------------------------------------------------

def test_canonical_is_well_formed(canonical):
    assert canonical["missing_level"] == MISSING
    assert set(canonical["dimensions"]) == set(ses_mod.DIMENSIONS)
    for dim, spec in canonical["dimensions"].items():
        ids = [lv["id"] for lv in spec["levels"]]
        assert len(ids) == len(set(ids)), f"{dim}: niveaux dupliqués"
        assert MISSING in ids, f"{dim}: niveau `missing` obligatoire"
        assert len(ids) > 2, f"{dim}: schéma vide"
        for lv in spec["levels"]:
            assert lv["label_fr"] and lv["label_en"]
        assert spec["census_source"], f"{dim}: source recensement non documentée"


def test_priority_dimensions_present(canonical):
    for dim in ("age", "gender", "education", "region", "income", "language"):
        assert dim in canonical["dimensions"]


def test_age_brackets_partition_the_adult_range(canonical):
    bounds = [(lv.get("min_age"), lv.get("max_age"))
              for lv in canonical["dimensions"]["age"]["levels"]
              if lv["id"] != MISSING]
    covered = set()
    for lo, hi in bounds:
        lo = 0 if lo is None else lo
        hi = 110 if hi is None else hi
        rng = set(range(lo, hi + 1))
        assert not (covered & rng), "tranches d'âge qui se chevauchent"
        covered |= rng
    assert set(range(18, 100)) <= covered


def test_income_brackets_are_contiguous(canonical):
    levels = [lv for lv in canonical["dimensions"]["income"]["levels"]
              if lv["id"] != MISSING]
    previous_max = None
    for lv in levels:
        lo = lv.get("min_amount")
        if previous_max is not None:
            assert lo == previous_max + 1, f"trou/chevauchement avant {lv['id']}"
        previous_max = lv.get("max_amount")
    assert previous_max is None, "la dernière tranche doit être ouverte"


def test_conventions_referenced_by_mappings_exist(canonical, crosswalk):
    known = set(canonical["conventions"])
    for sid, dims in crosswalk["surveys"].items():
        for dim, spec in dims.items():
            for conv in _all_conventions(spec):
                assert conv in known, f"{sid}/{dim}: convention inconnue {conv!r}"


def _all_conventions(spec):
    out = list(spec.get("conventions", []))
    for sub in spec.get("sources", []):
        out += _all_conventions(sub)
    return out


# --------------------------------------------------------------------------
# Crosswalk structure
# --------------------------------------------------------------------------

def test_every_perimeter_survey_is_in_the_crosswalk(crosswalk):
    assert set(crosswalk["surveys"]) == perimeter_surveys()


def test_no_excluded_survey_leaks_in(crosswalk):
    assert not (set(crosswalk["surveys"]) & set(EXCLUDED_SURVEYS))


def test_dimensions_are_known(crosswalk):
    for sid, dims in crosswalk["surveys"].items():
        assert set(dims) <= set(ses_mod.DIMENSIONS), sid


def test_every_source_code_maps_to_a_declared_level(canonical, crosswalk):
    for sid, dims in crosswalk["surveys"].items():
        for dim, spec in dims.items():
            allowed = {lv["id"] for lv in canonical["dimensions"][dim]["levels"]}
            for level in declared_levels(spec):
                assert level in allowed, f"{sid}/{dim}: niveau inconnu {level!r}"


def test_categorical_maps_are_non_empty_and_keys_normalized(crosswalk):
    for sid, dims in crosswalk["surveys"].items():
        for dim, spec in _iter_specs(dims):
            if spec["kind"] != "categorical":
                continue
            assert spec["map"], f"{sid}/{dim}: mapping vide"
            for code in spec["map"]:
                assert normalize_code(code) == code, \
                    f"{sid}/{dim}: clé non normalisée {code!r}"


def test_coarse_targets_have_at_least_two_levels(crosswalk):
    for sid, dims in crosswalk["surveys"].items():
        for dim, spec in _iter_specs(dims):
            if spec["kind"] != "categorical":
                continue
            for code, target in spec["map"].items():
                if isinstance(target, dict):
                    assert len(target["to"]) >= 2, f"{sid}/{dim}/{code}"
                    assert target.get("coarse") is True
                    assert MISSING not in target["to"], \
                        f"{sid}/{dim}/{code}: `missing` ne se mélange pas"


def test_every_mapping_declares_a_variable_or_a_constant(crosswalk):
    for sid, dims in crosswalk["surveys"].items():
        for dim, spec in dims.items():
            if spec["kind"] == "constant":
                assert spec["value"]
            else:
                assert list(ses_mod._variables_of(spec)), f"{sid}/{dim}"


def test_needs_review_flag_is_boolean(crosswalk):
    for sid, dims in crosswalk["surveys"].items():
        for dim, spec in _iter_specs(dims):
            assert isinstance(spec.get("needs_review", False), bool), f"{sid}/{dim}"


def _iter_specs(dims):
    for dim, spec in dims.items():
        yield dim, spec
        for sub in spec.get("sources", []):
            yield dim, sub


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def test_normalize_code():
    assert normalize_code("09") == "9"
    assert normalize_code(9) == "9"
    assert normalize_code(9.0) == "9"
    assert normalize_code("QC") == "QC"
    assert normalize_code("") == ""
    assert normalize_code(None) is None
    assert normalize_code(float("nan")) is None


def test_categorical_roundtrip(xw):
    value = xw.apply("cecd_charte_2013_10", "age", {"qage": 3})
    assert value.level == "35_44"
    assert not value.coarse and not value.is_missing


def test_non_response_codes_map_to_missing(xw):
    assert xw.apply("cecd_charte_2013_10", "income", {"qreve": 9}).is_missing
    assert xw.apply("ces_2019_phone", "gender", {"q3": -9}).is_missing
    assert xw.apply("ces_2019_phone", "gender", {"q3": -8}).is_missing
    assert xw.apply("eeq_2012", "education", {"SCOL": 99}).is_missing
    assert xw.apply("eeq_2007", "age", {"q75": 9999}).is_missing


def test_null_and_absent_column_map_to_missing(xw):
    assert xw.apply("cecd_charte_2013_10", "age", {"qage": None}).is_missing
    assert xw.apply("cecd_charte_2013_10", "age", {}).is_missing


def test_unknown_code_is_flagged_not_silently_dropped(xw):
    value = xw.apply("cecd_charte_2013_10", "age", {"qage": 42})
    assert value.is_missing and value.unmapped and value.needs_review


def test_year_of_birth_bracketing(xw):
    # eeq_2012, terrain 2012
    assert xw.apply("eeq_2012", "age", {"AGEX": 1980}).level == "25_34"
    assert xw.apply("eeq_2012", "age", {"AGEX": 1930}).level == "75_plus"
    assert xw.apply("eeq_2012", "age", {"AGEX": 1994}).level == "18_24"


def test_yob_code_offset(xw):
    # ces_2021 : code 1 = 1920, donc code 42 = 1961 -> 60 ans en 2021
    assert xw.apply("ces_2021", "age", {"cps21_yob": 42}).level == "55_64"
    assert xw.apply("ces_2019_online", "age", {"cps19_yob": 82}).level == "18_24"


def test_age_in_years(xw):
    assert xw.apply("ces_2025", "age", {"cps25_age_in_years": 30}).level == "25_34"
    assert xw.apply("ces_2025", "age", {"cps25_age_in_years": 200}).is_missing


def test_amount_bracketing_and_outlier_guard(xw):
    assert xw.apply("ces_2019_phone", "income", {"q69": 45000}).level == "under_60k"
    assert xw.apply("ces_2019_phone", "income", {"q69": 85000}).level == "60k_100k"
    assert xw.apply("ces_2019_phone", "income", {"q69": 250000}).level == "100k_plus"
    assert xw.apply("ces_2019_phone", "income", {"q69": -9}).is_missing
    assert xw.apply("ces_2019_phone", "income", {"q69": 1e12}).is_missing
    assert xw.apply("ces_2021", "income", {"cps21_income_number": -99.0}).is_missing


def test_coalesce_falls_back_to_the_second_source(xw):
    got = xw.apply("ces_2021", "income",
                   {"cps21_income_number": None, "cps21_income_cat": 3})
    assert got.level == "under_60k" and got.variable == "cps21_income_cat"
    got = xw.apply("ces_2021", "income",
                   {"cps21_income_number": 70000.0, "cps21_income_cat": 9})
    assert got.level == "60k_100k" and got.variable == "cps21_income_number"


def test_coarse_mapping_returns_a_set(xw):
    got = xw.apply("provincial_qc_2018", "age", {"age": 3})
    assert got.coarse and got.level is None
    assert set(got.levels) == {"55_64", "65_74", "75_plus"}


def test_constant_region_for_quebec_only_surveys(xw):
    assert xw.apply("eeq_2014", "region", {}).level == "qc"


def test_zero_padded_codes_resolve(xw):
    # eeq_2007 stocke ses codes en chaînes zéro-padées ('09' = Côte-Nord)
    assert xw.apply("eeq_2007", "region_qc", {"nomx": "09"}).level == "cote_nord"
    assert xw.apply("eeq_2007", "education", {"q77": "09"}).level == "bachelor"


def test_string_codes_resolve(xw):
    assert xw.apply("cecd_sante_can_usa", "region", {"PROV": "QC"}).level == "qc"
    assert xw.apply("cecd_sante_can_usa", "region",
                    {"PROV": ""}).level == "outside_canada"


def test_profile_and_labels(xw):
    row = {"cps21_yob": 42, "cps21_genderid": 2, "cps21_education": 9,
           "cps21_province": 11, "cps21_income_number": 70000.0,
           "pes21_lang": 2}
    profile = xw.profile("ces_2021", row)
    assert profile["gender"].level == "woman"
    assert profile["education"].level == "bachelor"
    assert profile["region"].level == "qc"
    labels = xw.profile_labels("ces_2021", row)
    assert labels["gender"] == "Femme"
    assert labels["region"] == "Québec"


def test_profile_covers_every_declared_dimension(xw):
    for sid in xw.survey_ids:
        profile = xw.profile(sid, {})
        assert set(profile) == set(xw.dimensions_for(sid))
        for dim, value in profile.items():
            if xw.spec(sid, dim)["kind"] != "constant":
                assert value.is_missing


def test_all_surveys_carry_the_four_priority_dimensions(xw):
    # age, gender, education, region — sauf exceptions documentées
    missing = {sid: [d for d in ("age", "gender", "education", "region")
                     if xw.spec(sid, d) is None]
               for sid in xw.survey_ids}
    missing = {k: v for k, v in missing.items() if v}
    assert missing == {"cecd_elxn_qc_2012": ["education"]}


# --------------------------------------------------------------------------
# Coverage report
# --------------------------------------------------------------------------

def test_coverage_csv_is_complete(xw):
    path = ses_mod.CROSSWALK_DIR / "ses_coverage.csv"
    if not path.exists():
        pytest.skip("ses_coverage.csv non généré (script 11 avec accès au Blob)")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    seen = {(r["survey_id"], r["dimension"]) for r in rows}
    assert seen == {(s, d) for s in xw.survey_ids for d in ses_mod.DIMENSIONS}
    for row in rows:
        if row["source_kind"] == "none":
            continue
        assert 0.0 <= float(row["pct_non_missing"]) <= 100.0
        assert row["needs_review"] in {"yes", "no"}
        # aucun code observé dans les microdonnées ne doit rester non mappé
        assert float(row["pct_unmapped_code"]) == 0.0, row
