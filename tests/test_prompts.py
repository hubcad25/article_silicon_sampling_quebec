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
    assert finetune["messages"][2] == {"role": "assistant", "content": "4"}


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
        line = f"{opt.code}) {opt.label}"
        assert line in user, line
        positions.append(user.index(line))
    assert positions == sorted(positions)


def test_dont_know_and_refusal_stay_in_the_options():
    """Decision of §0: both are valid targets and stay on the menu."""
    tpl = PromptTemplate(condition="C0")
    user = tpl.build_messages(PERSONA, TARGET, dimensions=ALL_DIMS)[1]["content"]
    assert "8) Je ne sais pas" in user
    assert "9) Je préfère ne pas répondre" in user


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
    ctx = [ContextAnswer.from_item(
        ItemSpec(**{**NEIGHBOUR.__dict__, "variable": f"q{i:02d}"}), "1")
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
        "Answer with the number of the chosen option only.")


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
