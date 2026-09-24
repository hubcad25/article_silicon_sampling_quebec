"""Tests for the shared prompt template (step 3.1).

Guarantees under test, in the order the plan states them:

* §6.3 — the fine-tuned arm and the roleplay arm see the byte-identical prompt;
* §2.7 — SES dropout is reproducible at a fixed seed;
* §3.3 — response options are reproduced verbatim, never reworded or reordered;
* §2.2/§3.3 — no target leak: C2 never shows the target item's own
  distribution, C1 never shows the respondent's own answer to the target.

Everything runs offline; a socket guard makes a network call an outright error.
"""

from __future__ import annotations

import random
import socket

import pytest

from article_silicon_sampling_quebec.prompts import (
    ContextAnswer,
    ContextDistribution,
    ItemSpec,
    Option,
    Persona,
    PromptTemplate,
    CONTEXT_MAX_COSINE,
    example_rng,
    item_text,
    nearest_context_items,
    numeric_scale,
    parse_options,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - only fires on a bug
        raise AssertionError("prompts.py must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

TARGET = ItemSpec(
    survey_id="eeq_2014",
    variable="q12",
    text="Et à quel point êtes-vous satisfait(e) de la performance du gouvernement ?",
    options=(
        Option("1", "Très satisfait(e)"),
        Option("2", "Assez satisfait(e)"),
        Option("3", "Pas très satisfait(e)"),
        Option("4", "Pas du tout satisfait(e)"),
        Option("8", "Je ne sais pas"),
        Option("9", "Je préfère ne pas répondre"),
    ),
    language="fr",
    short_text="Satisfaction envers le gouvernement",
)

NEIGHBOUR = ItemSpec(
    survey_id="eeq_2014",
    variable="q07",
    text="Quel est votre intérêt pour la politique en général?",
    options=(
        Option("1", "Très intéressé(e)"),
        Option("2", "Plutôt intéressé(e)"),
        Option("3", "Pas du tout intéressé(e)"),
    ),
    language="fr",
    short_text="Intérêt pour la politique",
)

PERSONA = Persona(
    fields={
        "age": "45-54 ans",
        "gender": "Homme",
        "education": "Diplôme d'études secondaires",
        "region_qc": "Laurentides",
        "income": "Moins de 60 000 $",
    },
    survey_id="eeq_2014",
)

ALL_DIMS = ("age", "gender", "education", "region_qc", "income")


# --------------------------------------------------------------------------
# §6.3 — one template for both arms
# --------------------------------------------------------------------------

def test_prompt_is_model_agnostic():
    """No knob names a model, a chat template or a special token."""
    tpl = PromptTemplate(condition="C0")
    text = "".join(m["content"] for m in
                   tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS))
    for forbidden in ("<|", "llama", "Llama", "gpt", "assistant<"):
        assert forbidden not in text
    assert not any("model" in f for f in tpl.__dataclass_fields__)


def test_finetune_and_roleplay_arms_get_identical_prompts():
    """The FT example is the roleplay prompt plus one assistant turn."""
    tpl = PromptTemplate(condition="C1", k=6)
    ctx = [ContextAnswer.from_item(NEIGHBOUR, "2")]
    roleplay = tpl.build_messages(PERSONA, TARGET, ctx, dimensions=ALL_DIMS)
    finetune = tpl.build_example(PERSONA, TARGET, "4", ctx, dimensions=ALL_DIMS)
    assert finetune["messages"][:2] == roleplay
    assert finetune["messages"][2] == {"role": "assistant",
                                       "content": "Pas du tout satisfait(e)"}


def test_rendering_is_deterministic():
    tpl = PromptTemplate(condition="C0")
    a = tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS)
    b = tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS)
    assert a == b


def test_conditions_differ_only_by_the_context_block():
    kw = dict(k=6)
    dims = ALL_DIMS
    c0 = PromptTemplate(condition="C0", **kw).build_messages(
        PERSONA, TARGET, dimensions=dims)
    c1 = PromptTemplate(condition="C1", **kw).build_messages(
        PERSONA, TARGET, [ContextAnswer.from_item(NEIGHBOUR, "2")], dimensions=dims)
    c2 = PromptTemplate(condition="C2", **kw).build_messages(
        PERSONA, TARGET,
        [ContextDistribution(NEIGHBOUR, (("Très intéressé(e)", 0.5),
                                         ("Plutôt intéressé(e)", 0.5)), n=80)],
        dimensions=dims)
    # identical persona, identical target block, the only delta is the prefix
    assert c0[0] == c1[0] == c2[0]
    target_block = c0[1]["content"]
    assert c1[1]["content"].endswith(target_block)
    assert c2[1]["content"].endswith(target_block)


def test_population_field_is_never_dropped():
    """§3.3 — the language x population confound fix must survive dropout."""
    tpl = PromptTemplate(condition="C0", min_fields=1)
    for seed in range(50):
        system = tpl.render_persona(PERSONA, "fr", random.Random(seed))
        assert "Population : Québec" in system
    national = Persona(fields=PERSONA.fields, survey_id="ces_2021")
    assert "Population : Canada" in tpl.render_persona(national, "en", None, ALL_DIMS)


# --------------------------------------------------------------------------
# §2.7 — SES dropout
# --------------------------------------------------------------------------

def test_ses_dropout_is_reproducible_at_a_fixed_seed():
    tpl = PromptTemplate(condition="C0")
    a = tpl.keep_dimensions(ALL_DIMS, example_rng(7, "eeq_2014:q12:r001"))
    b = tpl.keep_dimensions(ALL_DIMS, example_rng(7, "eeq_2014:q12:r001"))
    assert a == b
    c = tpl.keep_dimensions(ALL_DIMS, example_rng(7, "eeq_2014:q12:r002"))
    assert isinstance(c, tuple)
    # and a different seed eventually gives something else
    assert any(tpl.keep_dimensions(ALL_DIMS, example_rng(s, "x")) != a
               for s in range(20))


def test_ses_dropout_keeps_canonical_order_and_floor():
    tpl = PromptTemplate(condition="C0", min_fields=2)
    for seed in range(100):
        kept = tpl.keep_dimensions(ALL_DIMS, random.Random(seed))
        assert 2 <= len(kept) <= len(ALL_DIMS)
        assert list(kept) == [d for d in ALL_DIMS if d in kept]


def test_uniform_subset_covers_every_richness():
    """The whole point of §2.7: every persona size must be trained on."""
    tpl = PromptTemplate(condition="C0", ses_dropout="uniform_subset", min_fields=1)
    sizes = {len(tpl.keep_dimensions(ALL_DIMS, random.Random(s)))
             for s in range(400)}
    assert sizes == set(range(1, len(ALL_DIMS) + 1))


def test_dropout_none_keeps_everything():
    tpl = PromptTemplate(condition="C0", ses_dropout="none")
    assert tpl.keep_dimensions(ALL_DIMS, random.Random(0)) == ALL_DIMS


# --------------------------------------------------------------------------
# §3.3 — verbatim options
# --------------------------------------------------------------------------

def test_options_are_verbatim_and_in_order():
    tpl = PromptTemplate(condition="C0")
    user = tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS)[1]["content"]
    positions = []
    for opt in TARGET.options:
        line = f"\n- {opt.label}\n"
        assert line in user, line
        positions.append(user.index(line))
    assert positions == sorted(positions)


def test_dont_know_and_refusal_stay_in_the_options():
    """Decision of §0: both are valid targets and stay on the menu."""
    tpl = PromptTemplate(condition="C0")
    user = tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS)[1]["content"]
    assert "\n- Je ne sais pas\n" in user
    assert "\n- Je préfère ne pas répondre\n" in user


def test_target_wording_is_verbatim_and_untouched_by_context_compaction():
    tpl = PromptTemplate(condition="C1")
    user = tpl.build_messages(PERSONA, TARGET,
                              [ContextAnswer.from_item(NEIGHBOUR, "1")],
                              dimensions=ALL_DIMS)[1]["content"]
    assert TARGET.text in user
    assert TARGET.short_text not in user  # the short label is for context only


def test_item_text_prefers_display_label_only_when_truncated():
    truncated = {"question_text": "x" * 80, "display_label": "Libellé court"}
    assert item_text(truncated) == "Libellé court"
    # 79 is the same Stata cut with a trailing space stripped, so it is
    # truncated too under the default lengths.
    also_truncated = {"question_text": "x" * 79, "display_label": "Libellé court"}
    assert item_text(also_truncated) == "Libellé court"
    intact = {"question_text": "x" * 78, "display_label": "Libellé court"}
    assert item_text(intact) == "x" * 78
    assert item_text(truncated, prefer_label_when_truncated=False) == "x" * 80


def test_parse_options_accepts_json_and_objects():
    parsed = parse_options('[{"code": 1, "label": "Oui"}]')
    assert parsed == (Option("1", "Oui"),)
    assert parse_options(list(parsed)) == parsed


# --------------------------------------------------------------------------
# leak protection
# --------------------------------------------------------------------------

def test_c1_refuses_the_target_as_its_own_context():
    tpl = PromptTemplate(condition="C1")
    with pytest.raises(ValueError, match="context leak"):
        tpl.build_messages(PERSONA, TARGET,
                           [ContextAnswer.from_item(TARGET, "4")],
                           dimensions=ALL_DIMS)


def test_c2_refuses_the_target_distribution():
    tpl = PromptTemplate(condition="C2")
    leak = ContextDistribution(TARGET, (("Très satisfait(e)", 1.0),), n=99)
    with pytest.raises(ValueError, match="context leak"):
        tpl.build_messages(PERSONA, TARGET, [leak], dimensions=ALL_DIMS)


def test_c2_prompt_holds_no_share_for_the_target_item():
    tpl = PromptTemplate(condition="C2")
    ctx = [ContextDistribution(NEIGHBOUR, (("Très intéressé(e)", 0.62),
                                           ("Plutôt intéressé(e)", 0.38)), n=120)]
    user = tpl.build_messages(PERSONA, TARGET, ctx, dimensions=ALL_DIMS)[1]["content"]
    head, _, target_block = user.partition("Question : ")
    for opt in TARGET.options:
        assert f"{opt.label} " not in head  # no "<label> 42 %" line
    assert "62 %" in head and "n=120" in head


def test_c0_ignores_any_context_passed_by_mistake():
    tpl = PromptTemplate(condition="C0")
    ctx = [ContextAnswer.from_item(NEIGHBOUR, "1")]
    with_ctx = tpl.build_messages(PERSONA, TARGET, ctx, dimensions=ALL_DIMS)
    without = tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS)
    assert with_ctx == without


def test_context_is_truncated_to_k():
    tpl = PromptTemplate(condition="C1", k=2)
    # Distinct wordings: six identical ones would be collapsed by the
    # deduplication of the block below before k ever applied.
    ctx = [ContextAnswer.from_item(
        ItemSpec(**{**NEIGHBOUR.__dict__, "variable": f"q{i:02d}",
                    "short_text": f"Intérêt pour la politique {i}"}), "1")
        for i in range(6)]
    user = tpl.build_messages(PERSONA, TARGET, ctx, dimensions=ALL_DIMS)[1]["content"]
    assert user.count("Intérêt pour la politique") == 2


# --------------------------------------------------------------------------
# retrieval policy
# --------------------------------------------------------------------------

class _FakeIndex:
    """Minimal stand-in for ``SimilarityIndex`` — no parquet, no network."""

    def __init__(self, rows):
        self.rows = rows

    def neighbors(self, key, k=None, exclude_same_survey=False):
        class _F:
            def __init__(self, rows):
                self._rows = rows

            def iter_rows(self, named=True):
                return iter(self._rows)

        return _F(self.rows)


def test_retrieval_takes_the_k_nearest_with_no_lower_threshold():
    rows = [
        {"neighbor_survey_id": "s", "neighbor_variable": "a", "cosine": 0.94},
        {"neighbor_survey_id": "s", "neighbor_variable": "self", "cosine": 1.0},
        {"neighbor_survey_id": "other", "neighbor_variable": "b", "cosine": 0.93},
        {"neighbor_survey_id": "s", "neighbor_variable": "c", "cosine": 0.61},
        {"neighbor_survey_id": "s", "neighbor_variable": "d", "cosine": 0.12},
    ]
    got = nearest_context_items(_FakeIndex(rows), ("s", "self"), k=6)
    # self excluded, cross-survey excluded, the 0.12 neighbour kept anyway
    assert [g[0][1] for g in got] == ["a", "c", "d"]
    assert got[-1][1] == pytest.approx(0.12)


# --------------------------------------------------------------------------
# anti-leak rule on the context — no quasi-duplicate of the target may be a
# context item, in any condition (see the module docstring of prompts.py)
# --------------------------------------------------------------------------

def test_context_never_includes_a_quasi_duplicate_of_the_target():
    """cps21_votechoice / cps21_v_advance at 0.967: same survey, same people."""
    rows = [
        {"neighbor_survey_id": "s", "neighbor_variable": "twin", "cosine": 0.967},
        {"neighbor_survey_id": "s", "neighbor_variable": "near", "cosine": 0.94},
        {"neighbor_survey_id": "s", "neighbor_variable": "far", "cosine": 0.55},
    ]
    got = nearest_context_items(_FakeIndex(rows), ("s", "self"), k=6)
    assert [g[0][1] for g in got] == ["near", "far"]
    assert all(cos < CONTEXT_MAX_COSINE for _, cos in got)


def test_anti_leak_cut_is_inclusive_at_the_threshold():
    rows = [
        {"neighbor_survey_id": "s", "neighbor_variable": "at", "cosine": 0.95},
        {"neighbor_survey_id": "s", "neighbor_variable": "just_under",
         "cosine": 0.9499},
    ]
    got = nearest_context_items(_FakeIndex(rows), ("s", "self"), k=6)
    assert [g[0][1] for g in got] == ["just_under"]


def test_anti_leak_cut_applies_before_k_is_counted():
    """The rule shrinks the context; it never lets a duplicate fill a slot."""
    rows = [
        {"neighbor_survey_id": "s", "neighbor_variable": f"dup{i}", "cosine": 0.99}
        for i in range(6)
    ] + [{"neighbor_survey_id": "s", "neighbor_variable": "ok", "cosine": 0.70}]
    got = nearest_context_items(_FakeIndex(rows), ("s", "self"), k=6)
    assert [g[0][1] for g in got] == ["ok"]


def test_anti_leak_cut_matches_the_quasi_duplicate_bin_boundary():
    """The two constants must never drift apart."""
    from article_silicon_sampling_quebec import split as sp

    assert CONTEXT_MAX_COSINE == sp.CONTEXT_MAX_COSINE
    quasi_lo = next(lo for name, lo, _ in sp.DISTANCE_BINS
                    if name == "quasi_duplicate")
    assert CONTEXT_MAX_COSINE == quasi_lo


def test_anti_leak_cut_can_be_lifted_only_explicitly():
    rows = [{"neighbor_survey_id": "s", "neighbor_variable": "twin",
             "cosine": 0.99}]
    assert nearest_context_items(_FakeIndex(rows), ("s", "self"), k=6) == []
    lifted = nearest_context_items(
        _FakeIndex(rows), ("s", "self"), k=6, max_cosine=1.01
    )
    assert [g[0][1] for g in lifted] == ["twin"]


def test_retrieval_respects_the_eligible_set():
    rows = [
        {"neighbor_survey_id": "s", "neighbor_variable": "a", "cosine": 0.9},
        {"neighbor_survey_id": "s", "neighbor_variable": "b", "cosine": 0.8},
    ]
    got = nearest_context_items(_FakeIndex(rows), ("s", "self"), k=6,
                                eligible=[("s", "b")])
    assert [g[0][1] for g in got] == ["b"]


# --------------------------------------------------------------------------
# bilingual symmetry
# --------------------------------------------------------------------------

def test_english_items_render_in_english():
    item = ItemSpec(
        survey_id="ces_2021", variable="cps21_x",
        text="How much do you trust the federal government?",
        options=(Option("1", "A lot"), Option("2", "Not at all")),
        language="en",
    )
    persona = Persona(fields={"age": "45 to 54 years", "gender": "Man"},
                      survey_id="ces_2021")
    msgs = PromptTemplate(condition="C0").build_messages(
        persona, item, dimensions=("age", "gender"))
    assert "You are a respondent" in msgs[0]["content"]
    assert "Population : Canada" in msgs[0]["content"]
    assert msgs[1]["content"].endswith(
        "Answer with the exact text of the chosen option only.")


def test_region_and_province_are_distinct_fields():
    """`region` (province) and `region_qc` must not both render as 'Région'."""
    tpl = PromptTemplate(condition="C0")
    persona = Persona(fields={"region_qc": "Estrie", "region": "Québec"},
                      survey_id="eeq_2007")
    system = tpl.render_persona(persona, "fr", None, ("region_qc", "region"))
    assert "Région : Estrie" in system
    assert "Province : Québec" in system


def test_truncation_lengths_are_configurable():
    row = {"question_text": "x" * 79, "display_label": "Libellé"}
    assert item_text(row, truncation_lengths=(80,)) == "x" * 79
    assert item_text(row, truncation_lengths=(79, 80)) == "Libellé"


# --------------------------------------------------------------------------
# correctif 1 — the survey year frames the interview
# --------------------------------------------------------------------------

TARGET_2012 = ItemSpec(**{**TARGET.__dict__, "survey_id": "eeq_2012", "year": 2012})


def test_the_year_of_the_survey_opens_the_persona_block():
    tpl = PromptTemplate(condition="C0")
    system = tpl.build_messages(PERSONA, TARGET_2012, dimensions=ALL_DIMS)[0]["content"]
    assert system.splitlines()[0] == (
        "Tu es un répondant à un sondage d'opinion mené en 2012."
    )
    assert system.splitlines()[1].startswith("Population : ")


def test_the_year_renders_in_english_too():
    target = ItemSpec(**{**TARGET_2012.__dict__, "language": "en"})
    system = PromptTemplate(condition="C0").build_messages(
        PERSONA, target, dimensions=ALL_DIMS)[0]["content"]
    assert system.splitlines()[0] == (
        "You are a respondent to an opinion survey conducted in 2012."
    )


def test_the_year_is_never_dropped_by_the_ses_dropout():
    """Like ``Population``, the year is the frame, not an SES dimension."""
    tpl = PromptTemplate(condition="C0", min_fields=1)
    for seed in range(200):
        system = tpl.render_persona(PERSONA, "fr",
                                    rng=random.Random(seed),
                                    year=tpl.resolve_year(PERSONA, TARGET_2012))
        assert "mené en 2012" in system
        assert "Population : " in system


def test_the_year_is_overridable_at_inference():
    """The corpus stops in 2025; the product will ask about 2026."""
    tpl = PromptTemplate(condition="C0", year_override=2026)
    system = tpl.build_messages(PERSONA, TARGET_2012, dimensions=ALL_DIMS)[0]["content"]
    assert "mené en 2026" in system
    # override > persona > item
    persona = Persona(fields=dict(PERSONA.fields), survey_id="eeq_2012", year=2018)
    assert PromptTemplate(condition="C0").resolve_year(persona, TARGET_2012) == 2018
    assert tpl.resolve_year(persona, TARGET_2012) == 2026
    assert PromptTemplate(condition="C0").resolve_year(PERSONA, TARGET_2012) == 2012


def test_an_item_without_a_year_falls_back_to_the_plain_head():
    system = PromptTemplate(condition="C0").build_messages(
        PERSONA, TARGET, dimensions=ALL_DIMS)[0]["content"]
    assert system.splitlines()[0] == "Tu es un répondant à un sondage d'opinion."


def test_from_row_carries_the_year_of_the_survey():
    spec = ItemSpec.from_row({
        "survey_id": "eeq_2012", "variable": "q1", "question_text": "Q ?",
        "display_label": "Q", "options": [{"code": "1", "label": "Oui"}],
        "language": "fr", "year": 2012,
    })
    assert spec.year == 2012


# --------------------------------------------------------------------------
# correctif 2 — a bare numeric modality keeps its scale
# --------------------------------------------------------------------------

SCALE_ITEM = ItemSpec(
    survey_id="provincial_qc_2018", variable="q6_02",
    text="À quel point es-tu en accord avec cet énoncé : j'ai l'habitude de voter.",
    options=(
        Option("1", "1- Fortement en désaccord"),
        Option("2", "2"), Option("3", "3"), Option("4", "4"),
        Option("5", "5"), Option("6", "6"),
        Option("7", "7- Fortement en accord"),
        Option("8", "Ne sais pas/Pas certain(e)"),
    ),
    language="fr", short_text="Attitude envers le vote : habitude personnelle de voter",
    year=2018,
)


def test_numeric_scale_recovers_the_anchors():
    scale = numeric_scale(SCALE_ITEM)
    assert (scale.low, scale.high) == (1, 7)
    assert scale.low_anchor == "Fortement en désaccord"
    assert scale.high_anchor == "Fortement en accord"
    assert scale.anchored


def test_a_bare_numeric_context_modality_is_put_back_on_its_scale():
    tpl = PromptTemplate(condition="C1", k=6)
    line = tpl.render_context([ContextAnswer.from_item(SCALE_ITEM, "5")], "fr")
    assert line.endswith(
        "→ 5 sur 7 (1 = Fortement en désaccord, 7 = Fortement en accord)"
    )


def test_an_anchored_modality_is_left_verbatim():
    tpl = PromptTemplate(condition="C1", k=6)
    line = tpl.render_context([ContextAnswer.from_item(SCALE_ITEM, "7")], "fr")
    assert line.endswith("→ 7- Fortement en accord")
    plain = tpl.render_context([ContextAnswer.from_item(NEIGHBOUR, "1")], "fr")
    assert plain.endswith("→ Très intéressé(e)")


def test_the_scale_annotation_speaks_the_item_language():
    item = ItemSpec(**{**SCALE_ITEM.__dict__, "language": "en", "options": (
        Option("1", "1 - greatly deteriorated"), Option("2", "2"),
        Option("3", "3 - stayed the same"), Option("4", "4"),
        Option("5", "5 - greatly improved"),
    )})
    line = PromptTemplate(condition="C1").render_context(
        [ContextAnswer.from_item(item, "4")], "en")
    assert line.endswith(
        "→ 4 out of 5 (1 = greatly deteriorated, 5 = greatly improved)"
    )


def test_a_gappy_code_list_is_not_treated_as_a_scale():
    """Three codes that are not consecutive are a coding scheme, not a scale."""
    item = ItemSpec(
        survey_id="s", variable="v", text="Q ?",
        options=(Option("1", "1"), Option("2", "2"), Option("9", "9")),
        language="fr",
    )
    assert numeric_scale(item) is None
    line = PromptTemplate(condition="C1").render_context(
        [ContextAnswer.from_item(item, "2")], "fr")
    assert line.endswith("→ 2")


def test_the_target_options_are_never_annotated():
    """Verbatim wording of the target item is non-negotiable (§3.3)."""
    user = PromptTemplate(condition="C0").build_messages(
        PERSONA, SCALE_ITEM, dimensions=ALL_DIMS)[1]["content"]
    assert "\n- 2\n" in user
    assert "\n- 1- Fortement en désaccord\n" in user
    assert "sur 7" not in user


# --------------------------------------------------------------------------
# correctif 3 — deduplication lives in the template, not in the generator
# --------------------------------------------------------------------------

TWIN = ItemSpec(
    survey_id="eeq_2014", variable="q99",
    text="Une tout autre question.",
    options=NEIGHBOUR.options, language="fr",
    # the ces_2021 defect: a generic display_label that renders as the target
    short_text=TARGET.text,
)


def test_a_context_item_rendering_as_the_target_is_dropped():
    tpl = PromptTemplate(condition="C1", k=6)
    ctx = [ContextAnswer.from_item(TWIN, "1"),
           ContextAnswer.from_item(NEIGHBOUR, "1")]
    kept = tpl.select_context(TARGET, ctx)
    assert [c.item.variable for c in kept] == ["q07"]
    user = tpl.build_messages(PERSONA, TARGET, ctx, dimensions=ALL_DIMS)[1]["content"]
    assert user.count(TARGET.text) == 1


def test_two_context_items_rendering_identically_collapse_to_one():
    tpl = PromptTemplate(condition="C1", k=6)
    ctx = [ContextAnswer.from_item(
        ItemSpec(**{**NEIGHBOUR.__dict__, "variable": f"q{i:02d}"}), "1")
        for i in range(4)]
    kept = tpl.select_context(TARGET, ctx)
    assert len(kept) == 1
    block = tpl.render_context(kept, "fr").splitlines()[1:]
    assert len(set(block)) == len(block) == 1


def test_context_selection_is_idempotent():
    """The generator applies it, then the template applies it again."""
    tpl = PromptTemplate(condition="C1", k=6)
    ctx = [ContextAnswer.from_item(TWIN, "1"),
           ContextAnswer.from_item(NEIGHBOUR, "1")]
    once = tpl.select_context(TARGET, ctx)
    assert tpl.select_context(TARGET, once) == once


def test_deduplication_can_be_turned_off_for_a_diagnostic():
    tpl = PromptTemplate(condition="C1", k=6, dedup_context=False)
    ctx = [ContextAnswer.from_item(TWIN, "1"),
           ContextAnswer.from_item(NEIGHBOUR, "1")]
    assert len(tpl.select_context(TARGET, ctx)) == 2


def test_the_hard_target_leak_still_raises_before_deduplication():
    tpl = PromptTemplate(condition="C1", k=6)
    with pytest.raises(ValueError, match="context leak"):
        tpl.select_context(TARGET, [ContextAnswer.from_item(TARGET, "1")])


# --------------------------------------------------------------------------
# text answers — the model answers with the option's text, never its code
# --------------------------------------------------------------------------

import re  # noqa: E402

from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    _TEXT,
    merge_duplicate_labels,
    normalise_answer,
)

_CODED = re.compile(r"^-?\d+\) ")

DUP_ITEM = ItemSpec(  # eeq_2018 q53 / q54, as stored in items.parquet
    survey_id="eeq_2018", variable="q53", text="Avez-vous voté ?",
    options=(Option("1", "Oui"), Option("2", "Non"), Option("3", "Je ne sais pas"),
             Option("98", "Je ne sais pas"),
             Option("99", "Je préfère ne pas répondre")),
    language="fr", code_map=(("97", "99"),),
)


def _option_lines(user: str) -> list[str]:
    return user.rsplit("\nOptions :\n", 1)[1].split("\n\n")[0].split("\n")


def test_no_code_appears_in_the_target_option_list():
    for item in (TARGET, SCALE_ITEM, DUP_ITEM):
        user = PromptTemplate(condition="C0").build_messages(
            PERSONA, item, dimensions=ALL_DIMS)[1]["content"]
        lines = _option_lines(user)
        assert lines == [f"- {o.text}" for o in item.options]
        assert not any(_CODED.match(l) for l in lines)


def test_assistant_content_is_one_of_the_rendered_labels():
    tpl = PromptTemplate(condition="C0")
    for item in (TARGET, SCALE_ITEM, DUP_ITEM):
        for opt in item.options:
            ex = tpl.build_example(PERSONA, item, opt.code, dimensions=ALL_DIMS)
            answer = ex["messages"][2]["content"]
            assert f"- {answer}" in _option_lines(ex["messages"][1]["content"])
            assert item.match_answer(answer) == opt.code


def test_an_answer_that_is_not_an_option_raises():
    with pytest.raises(ValueError):
        PromptTemplate(condition="C0").build_example(
            PERSONA, TARGET, "42", dimensions=ALL_DIMS)


def test_duplicate_labels_are_merged_onto_the_first_code():
    assert [o.code for o in DUP_ITEM.options] == ["1", "2", "3", "99"]
    assert DUP_ITEM.canonical_code("98") == "3"
    assert DUP_ITEM.canonical_code("97") == "99"   # refusal merge untouched
    assert DUP_ITEM.answer_text("98") == "Je ne sais pas"
    assert DUP_ITEM.match_answer("Je ne sais pas") == "3"
    # idempotent: rebuilding a spec from its own fields changes nothing
    assert ItemSpec(**DUP_ITEM.__dict__) == DUP_ITEM


def test_merge_is_case_and_whitespace_insensitive():
    opts, remap = merge_duplicate_labels(
        (Option("1", "Ne sais pas"), Option("2", "ne  sais PAS "), Option("3", "Oui")))
    assert [o.code for o in opts] == ["1", "3"]
    assert remap == {"2": "1"}


def test_labels_are_unique_after_merge_on_the_whole_corpus():
    from pathlib import Path

    import polars as pl

    path = Path(__file__).resolve().parents[1] / "data" / "items.parquet"
    if not path.exists():
        pytest.skip("items.parquet not available")
    for row in pl.read_parquet(path).iter_rows(named=True):
        spec = ItemSpec.from_row(row)
        keys = [normalise_answer(o.text) for o in spec.options]
        assert all(keys), spec.key
        assert len(keys) == len(set(keys)), spec.key
        for opt in spec.options:
            assert spec.match_answer(opt.text) == opt.code


@pytest.mark.parametrize("output, expected", [
    ("Pas du tout satisfait(e)", "4"),              # exact
    ("pas du tout satisfait(e)", "4"),              # case
    ("  Pas du  tout satisfait(e).\n", "4"),        # whitespace + trailing period
    ('"Je ne sais pas"', "8"),                      # quotes
    ("« Je ne sais pas »", "8"),                    # French quotes
    ("- Très satisfait(e)", "1"),                   # bullet copied from the list
    ("Ｊｅ ne sais pas", "8"),                        # NFKC
    ("4", None),                                    # a code is not an answer
    ("Pas du tout satisfait", None),                # no fuzzy match
    ("Très satisfait(e) ou assez satisfait(e)", None),
    ("", None),
    (None, None),
])
def test_match_answer(output, expected):
    assert TARGET.match_answer(output) == expected


def test_match_answer_keeps_embedded_codes_and_bare_scale_points():
    assert SCALE_ITEM.match_answer("1- Fortement en désaccord") == "1"
    assert SCALE_ITEM.match_answer("2") == "2"
    assert SCALE_ITEM.match_answer("Fortement en désaccord") is None


def test_the_instruction_is_parallel_in_both_languages():
    fr, en = _TEXT["fr"]["instruction"], _TEXT["en"]["instruction"]
    assert fr == "Réponds uniquement par le texte exact de l'option choisie."
    assert en == "Answer with the exact text of the chosen option only."
    assert set(_TEXT["fr"]) == set(_TEXT["en"])
    for lang in ("fr", "en"):
        assert "numéro" not in _TEXT[lang]["instruction"]
        assert "number" not in _TEXT[lang]["instruction"]
    item_en = ItemSpec(**{**TARGET.__dict__, "language": "en"})
    fr_user = PromptTemplate(condition="C0").build_messages(
        PERSONA, TARGET, dimensions=ALL_DIMS)[1]["content"]
    en_user = PromptTemplate(condition="C0").build_messages(
        PERSONA, item_en, dimensions=ALL_DIMS)[1]["content"]
    assert fr_user.endswith("\n\n" + fr) and en_user.endswith("\n\n" + en)
    # same layout, only the instruction differs
    assert fr_user[: -len(fr)] == en_user[: -len(en)]
