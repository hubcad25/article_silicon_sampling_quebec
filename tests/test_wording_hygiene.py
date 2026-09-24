"""Tests for the render-time wording hygiene of ``prompts.py``.

Three defects the first datasets carried, each fixed at render time only:

1. context lines in the wrong language (``display_label`` summaries);
2. a variable name or questionnaire number in front of the wording;
3. mojibake (``A``` for ``À``).
"""

from __future__ import annotations

import pytest

from article_silicon_sampling_quebec.prompts import (
    CONTEXT_MAX_CHARS,
    ItemSpec,
    Option,
    build_item_specs,
    clean_wording,
    guess_language,
    has_mojibake,
    label_speaks,
    repair_mojibake,
    shorten_wording,
    strip_question_number,
    strip_variable_prefix,
)

OPTIONS = '[{"code": "1", "label": "Oui"}, {"code": "2", "label": "Non"}]'


def row(variable, text, *, label=None, language="fr", survey="s", source="questionnaire"):
    return {"survey_id": survey, "variable": variable, "question_text": text,
            "question_text_source": source, "display_label": label,
            "options": OPTIONS, "language": language, "year": 2021, "code_map": None}


# ---------------- 1. language ----------------

@pytest.mark.parametrize("text, expected", [
    ("Dans quelle mesure êtes-vous d'accord avec l'énoncé suivant ?", "fr"),
    ("How much do you trust the federal government?", "en"),
    ("L'avenir du Québec", "fr"),
    ("M. Yves-François Blanchet", None),    # capitalised proper noun: no evidence
    ("", None),
])
def test_guess_language(text, expected):
    assert guess_language(text) == expected


def test_label_speaks_requires_a_positive_match():
    assert label_speaks({"language": "en"}, "Trust in the federal government")
    assert not label_speaks({"language": "en"}, "Confiance envers le gouvernement fédéral")
    assert not label_speaks({"language": "en"}, "Blanchet")  # abstention is not a match
    assert not label_speaks({"language": "fr"}, None)


def test_context_line_comes_from_the_wording_not_a_foreign_label():
    spec = ItemSpec.from_row(row("q1", "How much do you trust the federal government?",
                                 label="Confiance envers le gouvernement fédéral",
                                 language="en"))
    assert spec.context_text == "How much do you trust the federal government?"


def test_truncated_wording_falls_back_to_a_same_language_label_only():
    cut = "x" * 80
    same = ItemSpec.from_row(row("q1", cut, label="Trust in the federal government",
                                 language="en", source="stata_label_truncated"))
    other = ItemSpec.from_row(row("q1", cut, label="Confiance envers le gouvernement",
                                  language="en", source="stata_label_truncated"))
    assert same.context_text == "Trust in the federal government"
    assert other.short_text is None


# ---------------- battery-aware shortening ----------------

def test_short_wording_is_kept_whole_and_on_one_line():
    assert shorten_wording("Êtes-vous\n  d'accord ?") == "Êtes-vous d'accord ?"


def test_long_battery_keeps_its_row_label():
    stem = "How important is it that the government protects the interests of " * 4
    out = shorten_wording(stem.strip() + " - Big Corporations")
    assert out.endswith(" - Big Corporations")
    assert len(out) <= CONTEXT_MAX_CHARS


def test_two_battery_rows_do_not_collapse():
    stem = "Dans quelle mesure êtes-vous d'accord avec chacun des énoncés suivants. " * 3
    a = shorten_wording(stem + "/ Les syndicats ont trop de pouvoir")
    b = shorten_wording(stem + "/ Les impôts sont trop élevés")
    assert a != b


def test_long_plain_wording_is_cut_on_a_sentence_or_word():
    text = "Première phrase assez longue pour compter comme une tête. " + "Mot " * 100
    out = shorten_wording(text)
    assert out == "Première phrase assez longue pour compter comme une tête."


def test_build_item_specs_restores_whole_wording_on_collision():
    head = "Please rate how the following leader performed during the campaign, overall. " * 2
    rows = [row("a", head + "They were intelligent and competent. - Blanchet", language="en"),
            row("b", head + "They were trustworthy and honest. - Blanchet", language="en")]
    specs = build_item_specs(rows)
    assert specs[("s", "a")].context_text != specs[("s", "b")].context_text


# ---------------- 2. identifiers ----------------

@pytest.mark.parametrize("variable, text, expected", [
    ("q6_02", "q6_02. [..] Êtes-vous d'accord ?", "[..] Êtes-vous d'accord ?"),
    ("p33", "p33 -- How satisfied are you?", "How satisfied are you?"),
    ("PROV1", "prov1: Quel parti ?", "Quel parti ?"),
    ("vote", "Vote déclaré au provincial", "Vote déclaré au provincial"),  # word, no punct
    ("q1", "q1.", "q1."),  # nothing left: keep
])
def test_strip_variable_prefix(variable, text, expected):
    assert strip_variable_prefix(text, variable) == expected


@pytest.mark.parametrize("text, expected", [
    ("Q13.  Dans la campagne électorale…", "Dans la campagne électorale…"),
    ("7a. S'il y avait des élections", "S'il y avait des élections"),
    ("S11.  Combien de personnes", "Combien de personnes"),
    ("1.   D'abord, pouvez-vous", "D'abord, pouvez-vous"),
    ("25. possible ou impossible", "possible ou impossible"),
    ("3) Êtes-vous", "Êtes-vous"),
    ("18 ans et plus, êtes-vous", "18 ans et plus, êtes-vous"),
    ("1.5 million de personnes", "1.5 million de personnes"),
    ("Education: Thinking about spending", "Education: Thinking about spending"),
    ("Q1.", "Q1."),
])
def test_strip_question_number(text, expected):
    assert strip_question_number(text) == expected


def test_target_and_context_render_without_identifiers():
    spec = ItemSpec(survey_id="s", variable="rts_q1",
                    text="rts_q1. Q4. Êtes-vous allé voter ?",
                    options=(Option("1", "Oui"),))
    assert spec.wording == "Êtes-vous allé voter ?"
    assert spec.context_text == "Êtes-vous allé voter ?"


# ---------------- 3. mojibake ----------------

@pytest.mark.parametrize("text, expected", [
    ("A` chaque élection", "À chaque élection"),
    ("déjà voté".encode().decode("cp1252", "replace").replace("\ufffd", "\xa0"), "déjà voté"),
    ("lâ€™avenir", "l’avenir"),
    ("déjà voté", "déjà voté"),
    ("`backtick` code", "`backtick` code"),
])
def test_repair_mojibake(text, expected):
    assert repair_mojibake(text) == expected


def test_has_mojibake_and_option_render():
    assert has_mojibake("A` chaque")
    assert not has_mojibake("À chaque")
    assert Option("1", "A` l'occasion").render() == "- À l'occasion"
    assert Option("1", "A` l'occasion").text == "À l'occasion"


def test_clean_wording_combines_all_three():
    assert clean_wording("rts_q1. A` chaque élection", "rts_q1") == "À chaque élection"
