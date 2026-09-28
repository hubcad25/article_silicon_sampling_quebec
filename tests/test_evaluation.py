from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from article_silicon_sampling_quebec.evaluation import (
    ARM_PATHS,
    build_model_tables,
    build_observed_tables,
    read_arm_draws,
)
from article_silicon_sampling_quebec.inference import CAMPAIGNS, ItemCell, RESULT_FIELDS
from article_silicon_sampling_quebec.prompts import ItemSpec, Option, Persona


def fake_pair() -> ItemCell:
    item = ItemSpec(
        survey_id="survey", variable="q1", text="Choose one",
        options=(Option("1", "Yes"), Option("2", "No"), Option("98", "Don't know")),
        language="en", year=2020,
    )
    return ItemCell(
        item_idx=4, block="block", survey_id="survey", variable="q1", language="en",
        cell="25_34|woman", heldout_valid_n=3, item=item,
        persona=Persona(fields={"age": "25 to 34 years", "gender": "Woman"},
                        survey_id="survey", year=2020),
        dimensions=("age", "gender"),
    )


def fake_draws(arm: str, *, invalid: bool = False) -> pl.DataFrame:
    campaign_name = {
        "A": "c0-8k", "B": "c1-8k", "B0": "c1-8k", "R": "base", "BS": "c1-8k",
        "A20": "c0-20k", "B020": "c1-20k", "BR8": "c1-8k", "BR20": "c1-20k",
    }[arm]
    campaign = CAMPAIGNS[campaign_name]
    condition, context = {
        "A": ("C0", "none"), "B": ("C1", "stratum"),
        "B0": ("C1", "none"), "R": ("base", "none"),
        "BS": ("C1", "stratum_half"),
        "A20": ("C0", "none"), "B020": ("C1", "none"),
        "BR8": ("C1", "respondent_half"), "BR20": ("C1", "respondent_half"),
    }[arm]
    rows = []
    for draw_idx, code in enumerate(("1", None if invalid else "1", "2")):
        valid = code is not None
        row = dict.fromkeys(RESULT_FIELDS)
        row.update({
            "draw_key": f"4|25_34|woman|0.7|{draw_idx}",
            "arm": arm,
            "deployment": campaign.deployment,
            "model": campaign.model,
            "condition": condition,
            "context": context,
            "n_context": 1 if arm in {"B", "BS", "BR8", "BR20"} else 0,
            "temperature": 0.7,
            "item_idx": 4,
            "block": "block",
            "survey_id": "survey",
            "variable": "q1",
            "cell": "25_34|woman",
            "heldout_valid_n": 3,
            "draw_idx": draw_idx,
            "raw_response": code or "invalid",
            "matched_code": code,
            "valid": valid,
        })
        rows.append(row)
    return pl.DataFrame(rows)


def test_observed_distribution_is_weighted_and_keeps_bootstrap_rows():
    heldout = pl.DataFrame({
        "__survey_id": ["survey"] * 4,
        "__respondent_id": ["a", "b", "c", "d"],
        "__weight": [3.0, 1.0, 2.0, 10.0],
        "cell": ["25_34|woman"] * 4,
    })
    micro = pl.DataFrame({
        "__respondent_id": ["a", "b", "c", "d"],
        "q1": ["1", "2.0", "98", "not offered"],
    })

    distributions, responses = build_observed_tables(
        [fake_pair()], heldout, survey_loader=lambda _survey, _columns: micro,
    )

    assert distributions["arm"].to_list() == ["observed"] * 3
    assert distributions.schema["temperature"] == pl.Float64
    assert distributions["temperature"].null_count() == 3
    assert dict(zip(distributions["code"], distributions["share"], strict=True)) == {
        "1": 0.5, "2": 1 / 6, "98": 1 / 3,
    }
    assert distributions["n"].to_list() == [3, 3, 3]
    assert responses.select("respondent_id", "code", "weight").rows() == [
        ("a", "1", 3.0), ("b", "2", 1.0), ("c", "98", 2.0),
    ]


def test_model_distributions_include_zero_options_and_invalid_diagnostics():
    draws = {arm: fake_draws(arm, invalid=arm == "B") for arm in ARM_PATHS}

    distributions, diagnostics = build_model_tables(
        draws, [fake_pair()], temperatures=(0.7,), draws_per_group=3,
    )

    assert distributions.height == 15
    b = distributions.filter(pl.col("arm") == "B")
    assert dict(zip(b["code"], b["share"], strict=True)) == {
        "1": 0.5, "2": 0.5, "98": 0.0,
    }
    assert b["n"].to_list() == [2, 2, 2]
    b_diagnostics = diagnostics.filter(pl.col("arm") == "B").row(0, named=True)
    assert b_diagnostics["transport_n"] == 3
    assert b_diagnostics["effective_n"] == 2
    assert b_diagnostics["invalid_n"] == 1
    assert b_diagnostics["invalid_rate"] == pytest.approx(1 / 3)
    assert b_diagnostics["coverage_ok"] is True


def test_model_distributions_reject_incomplete_groups():
    draws = {arm: fake_draws(arm) for arm in ARM_PATHS}
    draws["R"] = draws["R"].head(2)

    with pytest.raises(ValueError, match="expected 3 draws, got 2"):
        build_model_tables(draws, [fake_pair()], temperatures=(0.7,), draws_per_group=3)


def test_model_distributions_accept_explicit_interim_arm_subset():
    arms = ("R", "A", "B0", "B")
    draws = {arm: fake_draws(arm) for arm in arms}

    distributions, diagnostics = build_model_tables(
        draws, [fake_pair()], temperatures=(0.7,), draws_per_group=3, arms=arms,
    )

    assert set(distributions["arm"]) == set(arms)
    assert set(diagnostics["arm"]) == set(arms)


def test_model_distributions_accept_second_brief_respondent_arm():
    distributions, diagnostics = build_model_tables(
        {"BR8": fake_draws("BR8")}, [fake_pair()],
        temperatures=(0.7,), draws_per_group=3, arms=("BR8",),
    )

    assert set(distributions["arm"]) == {"BR8"}
    assert diagnostics.row(0, named=True)["effective_n"] == 3


def test_model_distributions_reject_substituted_draw_index():
    draws = {arm: fake_draws(arm) for arm in ARM_PATHS}
    draws["R"] = draws["R"].with_columns(
        pl.when(pl.col("draw_idx") == 0).then(3).otherwise(pl.col("draw_idx")).alias("draw_idx")
    )

    with pytest.raises(ValueError, match="draw_idx coverage mismatch"):
        build_model_tables(draws, [fake_pair()], temperatures=(0.7,), draws_per_group=3)


def test_read_arm_draws_uses_the_production_layout_and_preserves_codes(tmp_path: Path):
    for arm, relative_path in ARM_PATHS.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fake_draws(arm).write_csv(path)

    frames = read_arm_draws(tmp_path)

    assert set(frames) == set(ARM_PATHS)
    assert frames["A"].schema["matched_code"] == pl.Utf8
    assert frames["A"]["matched_code"].to_list() == ["1", "1", "2"]
