"""Périmètre du corpus — §3.2 de docs/plan_article.md.

Constante partagée : tous les modules de la phase 1 importent d'ici plutôt que
de redéfinir la liste. Ne pas modifier sans mettre à jour le plan.
"""

from __future__ import annotations

# Sondages écartés : gouvernementaux très spécialisés (vocabulaire qui noierait
# le mix thématique) et qualitatif sans items fermés.
EXCLUDED_SURVEYS: frozenset[str] = frozenset({
    "govcan_parca_2024",
    "govcan_habit_2024",
    "govcan_06822_wave1_2024",
    "govcan_06822_wave2_2024",
    "govcan_06822_wave3_2024",
    "medaillon_organismes_qualitatif",
})

# Types d'items retenus comme cibles possibles (fermés).
TARGET_VAR_TYPES: frozenset[str] = frozenset({"single", "scale"})

# Chemin du catalogue normalisé (source de vérité des libellés), hors dépôt.
MVP_REPO = "../mvp_moteur_recherche_sondages"
NORMALIZED_DIR = f"{MVP_REPO}/ingestion/normalized"


def is_in_perimeter(survey_id: str) -> bool:
    return survey_id not in EXCLUDED_SURVEYS


# --------------------------------------------------------------------------
# Non-opinion items — items that pass every mechanical filter of the catalogue
# but cannot be an opinion target (§3.2).
#
# Three families, all removed here, upstream of everything else, so that a
# non-opinion item is neither a target NOR a retrievable context neighbour:
#
#   (a) NON_OPINION_ITEMS / NON_OPINION_PATTERNS — hand-audited in two blind
#       passes (no result was visible): paradata (polling firm, questionnaire
#       language, completion-time flag, unit-of-measure follow-ups, panel
#       recruitment, a radio-station coding field, telephone coverage),
#       respondent attributes that escaped the catalogue's ``is_sociodemo``
#       flag (country of birth, citizenship, religion, occupation, union
#       membership, urban/rural, tenure, children, household assets…), and
#       derived recodes (``typog``, a collapsed vote-intention typology nobody
#       was ever asked).
#       Attitudes *about* those topics stay in: "importance of being born in
#       Canada to be truly Canadian" is an identity item, not a birthplace.
#
#   (b) MAX_OPTIONS — items carrying more than 15 SUBSTANTIVE modalities are
#       open-list questions coded after the fact (``cps25_bornin_other`` has
#       250 country labels, ``Q65`` 77 riding names, the ``raison``/``code``
#       batteries a post-coded reason nomenclature), not closed opinion items.
#       The count excludes non-response modalities (``count_substantive_options``,
#       built on the same detector as family (c)): a 0-10 thermometer carries 11
#       answers plus 4 to 6 ways of not answering, and counting the latter used
#       to push ~24 legitimate opinion items over the bar — every ces_2019_phone
#       like/dislike thermometer (``p6``-``p17``), every left-right placement
#       (``p36``-``p42``) and the provincial vote-intention items
#       (``cps19/21/25_prov_id``, ``pes19/21_provvote``). A real open list has
#       no non-response modalities to speak of, so its count is unchanged.
#
#   (c) all-non-response items — 0-100 thermometers and numeric-knowledge
#       questions whose ONLY catalogued modalities are the non-response codes
#       (997/998/999 = "doesn't know this party" / "don't know" / "refusal").
#       As a categorical target they would ask the model to choose between
#       three ways of not answering. Detected by pattern, not by hand.
# --------------------------------------------------------------------------

import re as _re  # noqa: E402
import unicodedata as _ud  # noqa: E402

#: An item with more than this many *substantive* (non-non-response)
#: modalities is an open-list question. See ``count_substantive_options``.
MAX_OPTIONS = 15

NON_OPINION_ITEMS: frozenset[tuple[str, str]] = frozenset(
    {
        # --- paradata / questionnaire administration
        ("cecd_elxn_qc_1998", "firme_post"),
        ("cecd_elxn_qc_2007", "siglenum"),
        ("cecd_elxn_qc_2007", "radio"),
        ("cecd_elxn_qc_2007", "z1particip"),
        ("ces_2021", "cps21_inattentive"),
        ("ces_2019_phone", "q26a"),
        ("ces_2019_phone", "q26b"),
        ("eeq_2014", "LANG"),
        ("cecd_sante_can_usa", "Q1A"),
        ("cecd_sante_can_usa", "Q16B1"),
        # --- respondent attributes missed by is_sociodemo
        ("cecd_sante_can_usa", "AMERI"),
        ("cecd_sante_can_usa", "CANAD"),
        ("cecd_sante_can_usa", "HOUS"),
        ("cecd_sante_can_usa", "OCC2"),
        ("cecd_sante_can_usa", "RACE"),
        ("cecd_sante_can_usa", "RACE2"),
        ("cecd_elxn_can_2011", "p9"),
        ("cecd_elxn_can_2011", "qlanf"),
        ("cecd_elxn_qc_2007", "lusage"),
        ("cecd_elxn_can_2011", "qtras"),
        ("ces_2019_online", "cps19_bornin_canada"),
        ("ces_2019_online", "cps19_religion"),
        ("ces_2019_online", "cps19_bornin_other"),
        ("ces_2019_online", "cps19_children"),
        ("ces_2019_online", "cps19_sexuality"),
        ("ces_2019_online", "pes19_month_of_birth"),
        ("ces_2019_online", "pes19_parents_born"),
        ("ces_2019_online", "pes19_rural_urban"),
        ("ces_2019_online", "pes19_yob"),
        ("ces_2019_phone", "p49"),
        ("ces_2019_phone", "p51"),
        ("ces_2019_phone", "p55"),
        ("ces_2019_phone", "p57"),
        ("ces_2019_phone", "q1"),
        # --- factual knowledge, not opinion
        # q49 asks the respondent to NAME the federal Minister of
        # Finance; the modalities are "Correct (Bill Morneau)" / "any
        # other name" / "no, don't know". There is a right answer, so
        # the target distribution measures knowledge, not preference.
        ("ces_2019_phone", "q49"),
        ("ces_2019_phone", "q62"),
        ("ces_2019_phone", "q64"),
        ("ces_2019_phone", "q70"),
        # cps21_children ("How many children, if any, do you have?") is a
        # household attribute, exactly like cps19_children and
        # cps25_children which were already listed. Its omission was an
        # audit miss, not a decision.
        ("ces_2021", "cps21_children"),
        ("ces_2021", "cps21_bornin_canada"),
        ("ces_2021", "cps21_bornin_other"),
        ("ces_2021", "cps21_citizenship"),
        ("ces_2021", "cps21_religion"),
        ("ces_2021", "cps21_immig_status"),
        ("ces_2021", "cps21_sexuality"),
        ("ces_2021", "cps21_trans"),
        ("ces_2021", "pes21_lived"),
        ("ces_2021", "pes21_occ_cat"),
        ("ces_2021", "pes21_parents_born"),
        ("ces_2021", "pes21_rural_urban"),
        ("ces_2021", "provcode"),
        ("ces_2025", "cps25_bornin_canada"),
        ("ces_2025", "cps25_bornin_other"),
        ("ces_2025", "cps25_children"),
        ("ces_2025", "cps25_citizen_other"),
        ("ces_2025", "cps25_citizen_other2"),
        ("ces_2025", "cps25_religion"),
        ("ces_2025", "cps25_rent"),
        ("ces_2025", "cps25_rent_length"),
        ("ces_2025", "cps25_sexuality"),
        ("ces_2025", "cps25_trans"),
        ("ces_2025", "cses_module6_QD07a"),
        ("ces_2025", "pes25_disability"),
        ("ces_2025", "pes25_disabilitytype"),
        ("ces_2025", "pes25_parents_born"),
        ("ces_2025", "pes25_place_live"),
        ("eeq_2014", "QENFAN"),
        ("eeq_2018", "qparents"),
        ("provincial_qc_2012", "SD2B"),
        ("provincial_qc_2018", "d6"),
        ("provincial_qc_2018", "qa"),
        # --- derived recode, not a question
        ("cecd_elxn_qc_1998", "typog"),
        # ------------------------------------------------------------------
        # Second audit pass (22 September), on the final draw. Same three
        # families, applied item by item to the 60 test items and then swept
        # across the whole corpus for their twins.
        # ------------------------------------------------------------------
        # --- factual knowledge: the item has a right answer, so its target
        #     distribution measures who knows it, not who prefers what. q49
        #     (name the federal finance minister) was already listed; these
        #     are the same question in the other surveys, plus the rest of
        #     the knowledge batteries.
        ("ces_2019_phone", "q48"),            # name the Premier of your province
        ("ces_2021", "cps21_finmin_name"),
        ("ces_2021", "cps21_govgen_name"),
        ("ces_2025", "cps25_finmin_name"),
        ("ces_2025", "cps25_govgen_name"),
        ("eeq_2012", "Q63"),                  # who has authority over education
        ("eeq_2012", "Q64"),                  # who leads the CAQ
        ("provincial_qc_2012", "Q40"),        # true/false on the electoral system
        # "What share of health spending is public?" with four numeric
        # brackets and a don't know: an exam question, not an opinion.
        ("cecd_sante_can_usa", "Q52A1"),
        ("cecd_sante_can_usa", "Q52A2"),
        # --- respondent attributes the is_sociodemo flag missed
        ("cecd_sante_can_usa", "Q4"),         # serious/chronic illness
        ("cecd_sante_can_usa", "Q28E"),       # do you have a family doctor
        ("ces_2019_online", "pes19_health"),  # self-rated health, like Q4:
        ("ces_2021", "pes21_health"),         # a condition, not an opinion
        # The social-network battery describes the respondent's FRIENDS —
        # how many, how educated, how ethnically different. That is the
        # respondent's milieu, a demographic fact about their circle, and
        # its per-stratum distribution is demography, not opinion.
        ("ces_2019_online", "pes19_socnet1"),
        ("ces_2019_online", "pes10_socnet3"),
        ("ces_2019_online", "pes19_socnet2_1"),
        ("ces_2019_online", "pes19_socnet2_2"),
        ("ces_2019_online", "pes19_socnet2_3"),
        ("eeq_2018", "q67"),                  # year of arrival in Canada
        ("cecd_sante_can_usa", "Q29"),        # private health insurance
        ("cecd_sante_can_usa", "Q37"),        # spell without insurance coverage
        ("ces_2019_online", "cps19_fed_member"),   # dues-paying party member
        ("ces_2019_online", "cps19_prov_member"),  # flagged by the previous
        ("ces_2019_online", "cps19_union"),        # audit and never applied
        ("ces_2019_online", "pes19_partymember"),
        ("ces_2019_phone", "p31"),
        ("ces_2021", "cps21_union"),
        ("ces_2021", "pes21_partymember"),
        ("ces_2025", "cps25_union"),
        ("ces_2025", "pes25_partymember"),
        ("ces_2025", "cses_module6_Q27d"),    # had COVID in the household
        # --- paradata: attention checks. cps21_inattentive (the derived
        #     flag) was listed; the trap question itself was not. "We are
        #     more interested in making sure you're doing the survey
        #     carefully, so please just select the colour brown here."
        ("ces_2021", "cps21_attcheck"),
        ("ces_2025", "cps25_attcheck"),
        # --- paradata: what the election administration did, not opinion
        ("ces_2019_online", "pes19_emb_register"),  # got a card in the mail
        ("ces_2019_online", "pes19_emb_card"),      # was the card correct
        ("ces_2021", "pes21_emb_register"),
        ("ces_2021", "pes21_emb_card"),
        ("ces_2025", "pes25_emb_register"),
        # --- derived recodes and unlabelled duplicates of another variable.
        #     Same status as typog: nobody was ever asked these, they carry
        #     no display_label, and their question_text is an internal note
        #     ("VOTE AU SONDAGE", "Intention de vote provinciale + relance
        #     recodée"). As near-perfect twins of the real item they also
        #     inflate the quasi_duplicate bin.
        ("cecd_charte_2013_10", "vote1"),
        ("cecd_charte_2013_10", "vote3"),
        ("cecd_charte_2013_10", "intvoteprov"),
        ("cecd_charte_2013_10", "intvoteprov2"),
        ("cecd_elxn_qc_1998", "r3_createc"),
        ("cecd_elxn_qc_1998", "r4_createc"),
        ("cecd_elxn_qc_1998", "intvote2"),
        ("cecd_elxn_qc_1998", "q3post2"),
        ("cecd_elxn_qc_1998", "allervo2"),
        ("cecd_elxn_qc_1998", "q3post3"),
    }
)

#: Household asset, property and mortgage batteries — what a household owns.
#: Same status as income: persona material, never a target. Kept as patterns
#: because they are whole batteries.
NON_OPINION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("eeq_2012", r"^Q101[A-G]$"),
    ("eeq_2014", r"^Q(59|60)[A-G]$"),
    ("eeq_2018", r"^(q63[ab]_\d+|q64[ab]_\d+|q65[ab])$"),
    # Knowledge batteries — every slot has a right answer.
    #   provincial_qc_2012 Q10A-E: match a leader's PHOTO to a party. The
    #     photo is not in the catalogue, so the item is also unanswerable
    #     from the text, and three of the five slots carry the wrong option
    #     list (the same three names for every party).
    #   eeq_2018 q12_1-4: "what post did <person> hold last year", five
    #     mutually exclusive offices.
    #   eeq_2018 q15_1-4: match a campaign promise to the party that made it.
    ("provincial_qc_2012", r"^Q10[A-E]$"),
    ("eeq_2018", r"^q12_[1-4]$"),
    ("eeq_2018", r"^q15_[1-4]$"),
    # Which kind of health plan covers you — a household attribute, exactly
    # like Q29 above.
    ("cecd_sante_can_usa", r"^Q30[A-E]$"),
)

#: Sentinel numeric codes used for non-response across the corpus. They are
#: authoritative: ``(-7) Skipped`` carries a label the vocabulary below does
#: not match, and 187 options of ces_2019_phone were being counted as answers
#: because of it.
NON_RESPONSE_CODES: frozenset[str] = frozenset(
    {"997", "998", "999", "996", "-9", "-8", "-7"}
)

#: The "don't know" subset of the sentinels (997 = "doesn't know this party",
#: 998 = "don't know", -9 = "don't know"). Everything else in
#: ``NON_RESPONSE_CODES`` is a refusal or a skip.
DONT_KNOW_CODES: frozenset[str] = frozenset({"997", "998", "-9"})

#: "Don't know" vocabulary — the subset of non-response that is a genuine
#: state of opinion and a valid target (plan §0, decision of 21 September).
#: Checked FIRST, so a combined modality such as "Don't know / Prefer not to
#: answer" counts as a don't know and is never merged away.
DONT_KNOW_LABEL = _re.compile(
    r"(ne sai[ts] pas"
    r"|ne (le |la |le/la |les )?conna[it]"
    r"|don'?t know|do not know"
    r"|\bnsp\b)"
)

#: Non-response vocabulary, matched on an accent-folded, lower-cased label.
#: Superset of ``DONT_KNOW_LABEL``: what it matches and ``DONT_KNOW_LABEL``
#: does not is a refusal.
NON_RESPONSE_LABEL = _re.compile(
    r"(ne sai[ts] pas"
    r"|ne (le |la |le/la |les )?conna[it]"
    r"|refus"
    r"|pas de reponse"
    r"|prefere ne pas repondre"
    r"|aucune reponse"
    r"|don'?t know|do not know"
    r"|no answer|no response"
    r"|prefer not to"
    r"|not asked"
    r"|sans objet|non applicable"
    r"|\bnsp\b"
    r"|missing"
    r"|skipped"
    r"|declined)"
)

#: Label carried by the single merged refusal modality, per survey language.
#: Used only when the merged group has no usable label of its own.
REFUSAL_MERGED_LABEL: dict[str, str] = {
    "fr": "Refus / préfère ne pas répondre",
    "en": "Refused / prefer not to answer",
}


def _fold(text: str | None) -> str:
    folded = _ud.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return " ".join(folded.lower().split())


def non_response_kind(option: dict) -> str | None:
    """``"dont_know"``, ``"refusal"``, or ``None`` for a substantive answer.

    The label wins when it says something; the sentinel code decides
    otherwise. "Don't know" is tested first: a modality that offers both
    ("Don't know / Prefer not to answer", the CES house style) is a don't
    know, because collapsing it into the refusal bucket would delete the
    don't-know mass the design keeps as a target.
    """
    label = _fold(option.get("label"))
    if label:
        if DONT_KNOW_LABEL.search(label):
            return "dont_know"
        if NON_RESPONSE_LABEL.search(label):
            return "refusal"
    code = str(option.get("code"))
    if code in DONT_KNOW_CODES:
        return "dont_know"
    if code in NON_RESPONSE_CODES:
        return "refusal"
    return None


def is_non_response_option(option: dict) -> bool:
    """True when a modality is a non-response code rather than an answer."""
    return non_response_kind(option) is not None


def is_refusal_option(option: dict) -> bool:
    """True for refusal / "prefer not to answer" / NA / skipped modalities.

    Deliberately excludes "don't know": that one is a real state of opinion
    and a valid target (plan §0), so it stays its own modality.
    """
    return non_response_kind(option) == "refusal"


def all_options_non_response(options) -> bool:
    """True when EVERY catalogued modality of an item is a non-response."""
    options = list(options or [])
    return bool(options) and all(is_non_response_option(o) for o in options)


def count_substantive_options(options) -> int:
    """Number of modalities that are an actual answer — family (b)'s counter.

    A 0-10 thermometer has 11, whatever the number of ways it offers of not
    answering; an open list of 250 countries still has 250.
    """
    return sum(1 for o in (options or []) if not is_non_response_option(o))


def merge_refusal_options(options, language: str | None = None):
    """Collapse every refusal modality of an item into a single one.

    Refusals are instrument behaviour, not opinion: an item that offers
    "Refused", "Prefer not to answer" and "Skipped" as three separate codes
    asks the model to guess *how* a respondent declined. They become one
    modality, keeping the first refusal's code so the training target stays a
    code the microdata actually contains. "Don't know" is untouched.

    Returns ``(merged_options, code_map)``; ``code_map`` sends every absorbed
    raw code to the surviving one and is the ONLY correct way to fold the
    observed microdata onto the modalities shown in the prompt. Applying one
    without the other compares two different partitions of the same item.
    """
    options = [dict(o) for o in (options or [])]
    refusals = [o for o in options if is_refusal_option(o)]
    if len(refusals) < 2:
        return options, {}

    keeper = refusals[0]
    keep_code = str(keeper.get("code"))
    label = (keeper.get("label") or "").strip()
    if not label:
        label = REFUSAL_MERGED_LABEL.get(language or "fr", REFUSAL_MERGED_LABEL["fr"])
    code_map = {
        str(o.get("code")): keep_code for o in refusals if str(o.get("code")) != keep_code
    }

    merged = []
    for option in options:
        code = str(option.get("code"))
        if code == keep_code:
            merged.append({**option, "label": label})
        elif code in code_map:
            continue
        else:
            merged.append(option)
    return merged, code_map


def is_listed_non_opinion(survey_id: str, variable: str) -> bool:
    """Family (a): the hand-audited list and the battery patterns."""
    if (survey_id, variable) in NON_OPINION_ITEMS:
        return True
    return any(
        survey == survey_id and _re.fullmatch(pattern, variable)
        for survey, pattern in NON_OPINION_PATTERNS
    )


# --------------------------------------------------------------------------
# (d) Unresolved Qualtrics piping — the catalogued stem still carries the
# placeholder instead of the text the respondent actually saw.
#
# Two reasons this is a corpus filter and not a cosmetic one:
#
#   * the placeholder often IS the load-bearing word. "How much should the
#     federal government spend on [Field-justice_law]?" has no object;
#     "Immigrants [Field-immigincrease_exp_text] in my area." has no verb;
#     "How strongly [Field-pid_en][QID39-ChoiceTextEntryValue-7] do you
#     feel?" is not a sentence. A model cannot answer them and a human
#     reading the pre-registered list cannot check them.
#   * the EMBEDDING was computed on that broken text, so the item sits at a
#     meaningless place on the distance axis — the axis the whole paper is
#     about. Dropping the item is the only way to keep the axis honest;
#     keeping it as a context neighbour would inject the placeholder into a
#     prompt verbatim.
#
# A milder variant exists (``... under ${e://Field/premier}?``) where the
# stem still reads. The rule does not try to tell them apart: 46 items out
# of ~1900, no way to recover the substitution offline, and a mechanical
# rule is auditable where a hand list of "still readable enough" is not.
# --------------------------------------------------------------------------

#: Qualtrics / Stata piping placeholders seen in this corpus.
PIPING_PATTERN = _re.compile(
    r"(\$\{[^}]*\}"          # ${e://Field/x}, ${q://QID1/...}
    r"|\[Field-[^\]]*\]"      # [Field-justice_law]
    r"|\[QID\d+[^\]]*\]"      # [QID39-ChoiceTextEntryValue-7]
    r"|<<?Q\d+[A-Za-z]*>>?"   # <Q70>, <<Q52Party>>
    r"|\\?\[ins[eé]r[^\]]*\\?\]"   # \[insérer réponse de Q92\]
    r"|\\?\[insert[^\]]*\\?\])"
)


def has_unresolved_piping(question_text: str | None) -> bool:
    """True when a stem still shows a questionnaire placeholder."""
    return bool(PIPING_PATTERN.search(question_text or ""))


# --------------------------------------------------------------------------
# Context-only items — reported behaviour and personal experience
# (decision of 22 September)
#
# "Did you vote", "did you donate to a party", "did you sign a petition",
# "did you volunteer for a campaign", "did you watch the leaders' debate":
# the respondent reports an ACT, not an attitude. Two things follow.
#
#   * It is not opinion. The design predicts what a stratum thinks; a
#     turnout rate is not a thought.
#   * The observed distribution is itself wrong. Turnout, debate viewing and
#     petition signing are massively over-reported in surveys, so the
#     empirical counterpart we would score against is biased by an unknown
#     amount that has nothing to do with the model. Scoring a prediction
#     against it measures the respondent's self-flattery, not the method.
#
# So they are **usable as context, never as a target** — neither in training
# nor in test. That is a weaker filter than NON_OPINION_ITEMS on purpose:
# knowing that a respondent voted and signed two petitions is genuinely
# informative about their politics, and a C1 prompt should be allowed to say
# so. They therefore stay in ``data/items.parquet`` (and in the similarity
# index, so they remain retrievable) but carry ``is_context_only = True``,
# and ``scripts/14_build_split.py`` bars them from the candidate pool.
#
# The line drawn, and it is drawn narrowly: a **discrete act the respondent
# performed** (or that was done to them, for "did a party contact you").
# Deliberately NOT in the family:
#   * stated vote CHOICE ("which party did you vote for") — a direction, and
#     the central quantity of electoral opinion research. Its bias is recall
#     and bandwagon, not over-reporting of an act. Kept as a target.
#   * willingness and intention ("how ready would you be to donate to a
#     party", eeq_2018 q32_*) — a disposition, which is an attitude.
#   * attention and interest scales ("how much attention did you pay to the
#     campaign", provincial_qc_2012 PQ9*) — intensity, not an act.
#   * attitudes ABOUT participation ("low turnout weakens democracy").
# --------------------------------------------------------------------------

CONTEXT_ONLY_ITEMS: frozenset[tuple[str, str]] = frozenset(
    {
        # --- turnout, general elections and the 1995 referendum
        ("cecd_elxn_can_2011", "z2"),
        ("cecd_elxn_qc_1998", "q1post"),
        ("cecd_elxn_qc_2007", "voteoui"),   # "êtes-vous allé voter", despite the name
        ("cecd_elxn_qc_2012", "participation"),
        ("ces_2019_online", "cps19_turnout_2015"),
        ("ces_2019_online", "pes19_turnout2019"),
        ("ces_2019_online", "pes19_turnout2019_v2"),
        ("ces_2019_phone", "p2"),
        ("ces_2019_phone", "q59"),
        ("ces_2021", "cps21_turnout_2019"),
        ("ces_2021", "pes21_turnout2021"),
        ("ces_2025", "cps25_turnout_2021"),
        ("ces_2025", "pes25_turnout2025"),
        ("eeq_2007", "q11"),
        ("eeq_2008", "q11"),
        ("eeq_2012", "Q21"),
        ("eeq_2012", "Q22"),
        ("eeq_2012", "Q47"),
        ("eeq_2014", "Q2"),
        ("eeq_2014", "Q16"),
        ("eeq_2018", "q5"),
        ("eeq_2018", "q23"),
        ("provincial_qc_2012", "Q6"),
        ("provincial_qc_2012", "PQ5_2"),
        # --- how the vote was cast, and registering to vote
        ("ces_2019_online", "pes19_howvote"),
        ("ces_2019_online", "pes19_emb_register2"),
        ("ces_2019_online", "pes19_emb_reg_how"),
        ("ces_2021", "cps21_howvote2"),
        ("ces_2021", "pes21_howvote"),
        ("ces_2021", "pes21_emb_register2"),
        ("ces_2021", "pes21_emb_reg_how"),
        ("ces_2025", "cps25_howvote2"),
        ("ces_2025", "pes25_howvote"),
        ("eeq_2018", "q6a"),
        ("provincial_qc_2012", "PQ5C"),
        # --- donating, volunteering, spoiling a ballot
        ("ces_2019_online", "cps19_volunteer"),
        ("ces_2019_online", "cps19_fed_donate"),
        ("ces_2019_online", "cps19_spoil"),
        ("ces_2019_phone", "p30"),
        ("ces_2019_phone", "q45"),
        ("ces_2021", "cps21_volunteer"),
        ("ces_2021", "cps21_spoil"),
        ("ces_2025", "cps25_volunteer"),
        ("eeq_2018", "q53"),
        ("eeq_2018", "q54"),
        ("eeq_2018", "q55a"),
        ("eeq_2018", "q55b"),
        # --- exposure acts: watched the debate, read or heard a poll
        ("cecd_elxn_can_2011", "luentend"),
        ("cecd_elxn_qc_2007", "luentend"),
        ("cecd_elxn_qc_2007", "z2q2"),
        ("cecd_elxn_qc_2012", "luentend"),
        ("ces_2019_online", "cps19_debate_en"),
        ("ces_2019_online", "cps19_debate_fr"),
        ("ces_2019_phone", "q77"),
        ("ces_2019_phone", "q77eng"),
        ("ces_2019_phone", "q77fr"),
        ("ces_2021", "cps21_debate_en"),
        ("ces_2021", "cps21_debate_fr"),
        ("ces_2021", "cps21_debate_fr2"),
        ("ces_2025", "cps25_debate_en"),
        ("ces_2025", "cps25_debate_fr"),
        ("provincial_qc_2012", "PQ11"),
        ("provincial_qc_2012", "PQ12"),
        ("provincial_qc_2018", "rts_q5"),
        # --- contacted by a party, discussed politics
        ("ces_2019_online", "pes19_contact1"),
        ("ces_2019_online", "pes19_disagreed"),
        ("ces_2021", "pes21_contact1"),
        ("ces_2025", "pes25_contact1"),
        # --- personal experience of the health system. Same logic one step
        #     away from politics: "did you ever pay out of pocket", "did you
        #     have trouble paying a medical bill" report an event, not a
        #     view of it. They say a lot about a respondent — good context —
        #     and nothing about what a stratum thinks.
        ("cecd_sante_can_usa", "Q23"),
        ("cecd_sante_can_usa", "Q39"),
    }
)

#: Whole participation batteries, kept as patterns because every slot of
#: them is the same reported act with a different verb.
CONTEXT_ONLY_PATTERNS: tuple[tuple[str, str], ...] = (
    # "how many times have you done these things over the past 12 months"
    ("ces_2019_online", r"^pes19_partic\d_\d$"),
    ("ces_2021", r"^pes21_partic\d_\d$"),
    ("ces_2025", r"^pes25_partic\d_\d$"),
    ("ces_2019_phone", r"^p29_[a-c]$"),
    # "avez-vous fait cette activité" / "avez-vous discuté de l'élection avec"
    ("eeq_2018", r"^q3[01]_\d$"),
    # campaign media use, active support, and non-electoral political action
    ("provincial_qc_2012", r"^PQ(10[A-F]|15[A-D]|18[A-D])$"),
    # "diriez-vous avoir regardé le débat / le face-à-face"
    ("cecd_elxn_qc_2012", r"^vudebat"),
    # "in your dealing with private health care organizations, have you had
    # any difficulties…" — four slots, four reported experiences.
    ("cecd_sante_can_usa", r"^Q38[A-D]$"),
)


def is_context_only(survey_id: str, variable: str) -> bool:
    """True when an item may be injected as context but never be a target."""
    if (survey_id, variable) in CONTEXT_ONLY_ITEMS:
        return True
    return any(
        survey == survey_id and _re.match(pattern, variable) is not None
        for survey, pattern in CONTEXT_ONLY_PATTERNS
    )
