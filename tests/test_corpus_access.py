"""Corpus access checks — local files only, no network."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from article_silicon_sampling_quebec.corpus import catalogue
from article_silicon_sampling_quebec.corpus.perimeter import (
    EXCLUDED_SURVEYS,
    TARGET_VAR_TYPES,
    is_in_perimeter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ITEMS_PARQUET = REPO_ROOT / "data" / "items.parquet"

EXPECTED_COLUMNS = [
    "survey_id",
    "variable",
    "question_text",
    "question_text_source",
    "display_label",
    "var_type",
    "is_ordinal",
    "n_options",
    "n_options_substantive",
    "options",
    "code_map",
    "themes",
    "concepts",
    "year",
    "language",
    "n_respondents_survey",
    "n_valid_responses",
    "is_context_only",
]

catalogue_available = pytest.mark.skipif(
    not catalogue.CATALOGUE_DIR.exists(),
    reason="normalized catalogue not available locally",
)
items_built = pytest.mark.skipif(
    not ITEMS_PARQUET.exists(), reason="run scripts/10_build_item_table.py first"
)


@pytest.fixture(scope="module")
def items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS_PARQUET)


def test_excluded_surveys_are_out_of_perimeter():
    assert not any(is_in_perimeter(s) for s in EXCLUDED_SURVEYS)
    assert is_in_perimeter("eeq_2014")


@catalogue_available
def test_perimeter_keeps_seventeen_surveys():
    assert len(catalogue.catalogue_survey_ids()) == 23
    assert len(catalogue.catalogue_survey_ids(perimeter_only=True)) == 17


@catalogue_available
def test_survey_items_carry_survey_context():
    items = catalogue.survey_items("eeq_2014")
    assert items
    first = items[0]
    for key in ("survey_id", "year", "language", "n_respondents_survey", "variable"):
        assert key in first
    assert first["survey_id"] == "eeq_2014"


@catalogue_available
def test_sociodemo_items_all_flagged_and_typed():
    socio = catalogue.sociodemo_items()
    assert socio
    assert all(i["is_sociodemo"] for i in socio)
    assert all(i["sociodemo_type"] for i in socio)


@items_built
def test_item_table_schema(items):
    assert items.columns == EXPECTED_COLUMNS
    assert items.height > 0
    assert items["variable"].null_count() == 0
    assert items.select(["survey_id", "variable"]).is_duplicated().sum() == 0


@items_built
def test_item_table_respects_perimeter_and_var_types(items):
    assert set(items["survey_id"]).isdisjoint(EXCLUDED_SURVEYS)
    assert set(items["var_type"]) <= TARGET_VAR_TYPES
    assert items["n_options"].min() >= 2


@items_built
@catalogue_available
def test_no_sociodemo_item_survives(items):
    socio_keys = {
        (i["survey_id"], i["variable"]) for i in catalogue.sociodemo_items()
    }
    kept_keys = set(zip(items["survey_id"], items["variable"]))
    assert kept_keys.isdisjoint(socio_keys)


@items_built
def test_options_column_is_parseable_json(items):
    for raw, n in zip(items["options"][:200], items["n_options"][:200]):
        opts = json.loads(raw)
        assert isinstance(opts, list)
        assert len(opts) == n


# --------------------------------------------------------------------------
# Non-response accounting: the open-list bar and the refusal merge
# --------------------------------------------------------------------------

def test_substantive_count_ignores_non_response_modalities():
    """A 0-10 thermometer is 11 answers, whatever its ways of not answering."""
    from article_silicon_sampling_quebec.corpus import perimeter

    options = (
        [{"code": -9, "label": "(-9) Don't know"},
         {"code": -8, "label": "(-8) Refused"},
         {"code": -7, "label": "(-7) Skipped"}]
        + [{"code": i, "label": f"({i}) {i}"} for i in range(11)]
        + [{"code": 11, "label": "(11) Don't know enough about the party"},
           {"code": 13, "label": "(13) Refused / Prefer not to answer"}]
    )
    assert perimeter.count_substantive_options(options) == 11
    # the raw count is what used to drop every ces_2019_phone thermometer
    assert len(options) > perimeter.MAX_OPTIONS
    # a real open list keeps every one of its modalities
    countries = [{"code": i, "label": f"Country {i}"} for i in range(250)]
    assert perimeter.count_substantive_options(countries) == 250


def test_refusal_merge_collapses_refusals_and_keeps_dont_know():
    from article_silicon_sampling_quebec.corpus import perimeter

    options = [
        {"code": -9, "label": "(-9) Don't know"},
        {"code": -8, "label": "(-8) Refused"},
        {"code": -7, "label": "(-7) Skipped"},
        {"code": 1, "label": "(1) Agree"},
        {"code": 13, "label": "(13) Refused / Prefer not to answer"},
    ]
    merged, code_map = perimeter.merge_refusal_options(options, "en")
    codes = [str(o["code"]) for o in merged]
    assert codes == ["-9", "-8", "1"]
    assert code_map == {"-7": "-8", "13": "-8"}
    # "Don't know / Prefer not to answer" is a don't know, never a refusal
    combined = {"code": 16, "label": "Don't know/ Prefer not to answer"}
    assert perimeter.non_response_kind(combined) == "dont_know"
    assert not perimeter.is_refusal_option(combined)


def test_refusal_merge_is_a_no_op_with_a_single_refusal():
    from article_silicon_sampling_quebec.corpus import perimeter

    options = [{"code": 1, "label": "Yes"}, {"code": 9, "label": "Refus"}]
    merged, code_map = perimeter.merge_refusal_options(options)
    assert merged == options and code_map == {}


@items_built
def test_code_map_only_ever_points_at_a_surviving_option(items):
    """The prompt's options and the observed distribution must agree."""
    import json as _json

    from article_silicon_sampling_quebec.prompts import ItemSpec

    for row in items.iter_rows(named=True):
        spec = ItemSpec.from_row(row)
        codes = {o.code for o in spec.options}
        for raw, kept in spec.code_map:
            assert kept in codes, f"{spec.key}: {raw} -> {kept} is not an option"
            assert raw not in codes, f"{spec.key}: {raw} was merged yet still shown"
        assert _json.loads(row["code_map"] or "{}") is not None
