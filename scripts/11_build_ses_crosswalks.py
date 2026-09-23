"""Step 1.3 — SES crosswalks (see docs/plan_article.md §3.4, §3.5, §2.1).

Authoring source of truth for three artefacts under data/crosswalks/:

  ses_canonical.json   canonical level schema (~5 dimensions), census-measurable
  ses_crosswalk.json   per-survey mapping RAW code -> canonical level(s)
  ses_coverage.csv     per survey x dimension coverage report

The canonical levels are chosen so that every one of them can be produced from
Canadian census / Statistics Canada tables, because they are used to
post-stratify to the real population and to define the product's input
vocabulary. Survey levels that are finer than the canonical schema are
collapsed; survey levels that are coarser map to a SET of canonical levels and
are flagged `coarse` rather than being given invented granularity.

Usage:
    .venv/bin/python scripts/11_build_ses_crosswalks.py            # JSON + coverage (reads Blob)
    .venv/bin/python scripts/11_build_ses_crosswalks.py --no-data  # JSON only, no network
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.corpus.perimeter import (  # noqa: E402
    EXCLUDED_SURVEYS,
    NORMALIZED_DIR,
)
from article_silicon_sampling_quebec.corpus import ses as ses_mod  # noqa: E402

OUT_DIR = REPO / "data" / "crosswalks"
VERSION = "1.0"

# --------------------------------------------------------------------------
# Canonical schema
# --------------------------------------------------------------------------

CONVENTIONS = {
    "missing_explicit": (
        "Tout code de non-réponse (refus, ne sait pas, -9/-8/-7, 98/99, 9999, "
        "valeur hors domaine) est mappé vers le niveau `missing`. Aucune "
        "imputation silencieuse."
    ),
    "highest_completed_credential": (
        "Scolarité : on retient le plus haut diplôme COMPLÉTÉ, comme le "
        "recensement (« plus haut certificat, diplôme ou grade »). Un niveau "
        "entamé sans diplôme ne compte pas."
    ),
    "qc_admission_prerequisite": (
        "Au Québec, le DES est requis pour le cégep et le DEC pour "
        "l'université. « Cégep non complété » implique donc un DES "
        "(-> high_school) et « université non complétée » un DEC "
        "(-> college_trades). Appliqué uniquement aux sondages québécois."
    ),
    "some_university_outside_qc": (
        "Hors Québec, « some university » n'implique aucun diplôme "
        "postsecondaire : mappé vers l'ensemble coarse "
        "{high_school, college_trades}."
    ),
    "continuous_bracketing": (
        "Année de naissance -> âge = année du terrain - année de naissance, "
        "puis découpage sur les bornes canoniques. Revenu en dollars -> "
        "découpage sur les bornes canoniques du revenu de ménage avant impôt."
    ),
    "coarse_partial_surjection": (
        "Quand un code source est plus grossier que le canonique, il mappe "
        "vers un ENSEMBLE de niveaux canoniques (`coarse: true`). La perte "
        "est documentée, pas comblée."
    ),
}

AGE_LEVELS = [
    ("under_18", "Moins de 18 ans", "Under 18", None, 17),
    ("18_24", "18-24 ans", "18 to 24 years", 18, 24),
    ("25_34", "25-34 ans", "25 to 34 years", 25, 34),
    ("35_44", "35-44 ans", "35 to 44 years", 35, 44),
    ("45_54", "45-54 ans", "45 to 54 years", 45, 54),
    ("55_64", "55-64 ans", "55 to 64 years", 55, 64),
    ("65_74", "65-74 ans", "65 to 74 years", 65, 74),
    ("75_plus", "75 ans et plus", "75 years and over", 75, None),
]

ADULT_AGE = ["18_24", "25_34", "35_44", "45_54", "55_64", "65_74", "75_plus"]

EDU_LEVELS = [
    ("no_diploma", "Aucun diplôme", "No certificate, diploma or degree"),
    ("high_school", "Diplôme d'études secondaires", "High school diploma or equivalent"),
    ("college_trades", "Postsecondaire inférieur au baccalauréat (métiers, cégep/collège, certificat universitaire)",
     "Postsecondary certificate or diploma below bachelor"),
    ("bachelor", "Baccalauréat", "Bachelor's degree"),
    ("above_bachelor", "Diplôme supérieur au baccalauréat", "Certificate, diploma or degree above bachelor"),
]

PROVINCES = [
    ("nl", "Terre-Neuve-et-Labrador", "Newfoundland and Labrador"),
    ("pe", "Île-du-Prince-Édouard", "Prince Edward Island"),
    ("ns", "Nouvelle-Écosse", "Nova Scotia"),
    ("nb", "Nouveau-Brunswick", "New Brunswick"),
    ("qc", "Québec", "Quebec"),
    ("on", "Ontario", "Ontario"),
    ("mb", "Manitoba", "Manitoba"),
    ("sk", "Saskatchewan", "Saskatchewan"),
    ("ab", "Alberta", "Alberta"),
    ("bc", "Colombie-Britannique", "British Columbia"),
    ("yt", "Yukon", "Yukon"),
    ("nt", "Territoires du Nord-Ouest", "Northwest Territories"),
    ("nu", "Nunavut", "Nunavut"),
    ("outside_canada", "Hors Canada", "Outside Canada"),
]

# 17 régions administratives du Québec (codes officiels ISQ en commentaire).
RA_QC = [
    ("bas_saint_laurent", "Bas-Saint-Laurent", 1),
    ("saguenay_lac_saint_jean", "Saguenay–Lac-Saint-Jean", 2),
    ("capitale_nationale", "Capitale-Nationale", 3),
    ("mauricie", "Mauricie", 4),
    ("estrie", "Estrie", 5),
    ("montreal", "Montréal", 6),
    ("outaouais", "Outaouais", 7),
    ("abitibi_temiscamingue", "Abitibi-Témiscamingue", 8),
    ("cote_nord", "Côte-Nord", 9),
    ("nord_du_quebec", "Nord-du-Québec", 10),
    ("gaspesie_iles", "Gaspésie–Îles-de-la-Madeleine", 11),
    ("chaudiere_appalaches", "Chaudière-Appalaches", 12),
    ("laval", "Laval", 13),
    ("lanaudiere", "Lanaudière", 14),
    ("laurentides", "Laurentides", 15),
    ("monteregie", "Montérégie", 16),
    ("centre_du_quebec", "Centre-du-Québec", 17),
]
RA_ALL = [r[0] for r in RA_QC]

INCOME_LEVELS = [
    ("under_60k", "Moins de 60 000 $", "Under $60,000", None, 59999),
    ("60k_100k", "60 000 $ à 99 999 $", "$60,000 to $99,999", 60000, 99999),
    ("100k_plus", "100 000 $ et plus", "$100,000 and over", 100000, None),
]

LANG_LEVELS = [
    ("french", "Français", "French"),
    ("english", "Anglais", "English"),
    ("other", "Autre langue", "Other language"),
    ("multiple", "Réponses multiples", "Multiple responses"),
]


def _lvl(lid, fr, en, **extra):
    d = {"id": lid, "label_fr": fr, "label_en": en}
    d.update({k: v for k, v in extra.items() if v is not None})
    return d


MISSING = "missing"


def build_canonical() -> dict:
    return {
        "version": VERSION,
        "generated_by": "scripts/11_build_ses_crosswalks.py",
        "plan_reference": "docs/plan_article.md §3.4, §3.5",
        "missing_level": MISSING,
        "design_rule": (
            "Un niveau canonique n'est retenu que s'il est mesurable dans les "
            "données du recensement canadien / de Statistique Canada."
        ),
        "conventions": CONVENTIONS,
        "dimensions": {
            "age": {
                "label_fr": "Âge",
                "ordered": True,
                "census_source": "Recensement, Âge (groupes quinquennaux agrégés)",
                "levels": [
                    _lvl(i, fr, en, min_age=lo, max_age=hi)
                    for i, fr, en, lo, hi in AGE_LEVELS
                ] + [_lvl(MISSING, "Non déclaré", "Not stated")],
            },
            "gender": {
                "label_fr": "Genre",
                "ordered": False,
                "census_source": "Recensement 2021, Genre (homme+, femme+, personnes de diverses identités de genre)",
                "levels": [
                    _lvl("man", "Homme", "Man"),
                    _lvl("woman", "Femme", "Woman"),
                    _lvl("non_binary_or_other", "Non binaire ou autre genre", "Non-binary or another gender"),
                    _lvl(MISSING, "Non déclaré", "Not stated"),
                ],
            },
            "education": {
                "label_fr": "Scolarité",
                "ordered": True,
                "census_source": "Recensement, Plus haut certificat, diplôme ou grade",
                "levels": [_lvl(i, fr, en) for i, fr, en in EDU_LEVELS]
                + [_lvl(MISSING, "Non déclaré", "Not stated")],
            },
            "region": {
                "label_fr": "Province ou territoire",
                "ordered": False,
                "census_source": "Recensement, Province ou territoire de résidence",
                "levels": [_lvl(i, fr, en) for i, fr, en in PROVINCES]
                + [_lvl(MISSING, "Non déclaré", "Not stated")],
            },
            "region_qc": {
                "label_fr": "Région administrative du Québec",
                "ordered": False,
                "census_source": "Recensement / ISQ, Régions administratives du Québec (17)",
                "applies_when": "region == 'qc'",
                "levels": [
                    _lvl(i, fr, fr, isq_code=f"{c:02d}") for i, fr, c in RA_QC
                ] + [_lvl(MISSING, "Non déclaré", "Not stated")],
            },
            "income": {
                "label_fr": "Revenu du ménage avant impôt",
                "ordered": True,
                "census_source": "Recensement, Revenu total du ménage avant impôt",
                "granularity_note": (
                    "Trois niveaux seulement : les bornes des sondages sont "
                    "incompatibles (CES : 30/60/90/110/150/200 k$ ; sondages "
                    "québécois : 20/40/60/80/100/150 k$ ; EEQ 2012-2014 : "
                    "8/16/24/40/56/72/88/104 k$). 60 k$ et 100 k$ sont les "
                    "seules bornes (quasi) communes ; découper plus finement "
                    "produirait des cellules non comparables."
                ),
                "levels": [
                    _lvl(i, fr, en, min_amount=lo, max_amount=hi)
                    for i, fr, en, lo, hi in INCOME_LEVELS
                ] + [_lvl(MISSING, "Non déclaré", "Not stated")],
            },
            "language": {
                "label_fr": "Langue maternelle",
                "ordered": False,
                "census_source": "Recensement, Langue maternelle",
                "concept_note": (
                    "Concept cible = langue maternelle / première langue "
                    "apprise. Les sondages n'offrant que la langue parlée à la "
                    "maison sont marqués `concept: language_home`."
                ),
                "levels": [_lvl(i, fr, en) for i, fr, en in LANG_LEVELS]
                + [_lvl(MISSING, "Non déclaré", "Not stated")],
            },
        },
    }


# --------------------------------------------------------------------------
# Small authoring helpers
# --------------------------------------------------------------------------

def cat(variable, mapping, *, needs_review=False, notes=None, conventions=None,
        source_label=None, concept=None):
    return {
        "kind": "categorical",
        "variable": variable,
        "map": {str(k): v for k, v in mapping.items()},
        "needs_review": needs_review,
        "notes": notes or [],
        "conventions": conventions or [],
        **({"source_label": source_label} if source_label else {}),
        **({"concept": concept} if concept else {}),
    }


def coarse(levels, note=None, needs_review=False):
    d = {"to": list(levels), "coarse": True}
    if note:
        d["note"] = note
    if needs_review:
        d["needs_review"] = True
    return d


def const(value, notes=None):
    return {"kind": "constant", "value": value, "needs_review": False,
            "notes": notes or [], "conventions": []}


def yob(variable, reference_year, missing_values=(), *, code_offset=None,
        notes=None, needs_review=False):
    d = {
        "kind": "year_of_birth",
        "variable": variable,
        "reference_year": reference_year,
        "missing_values": [str(m) for m in missing_values],
        "valid_year_range": [1900, reference_year],
        "needs_review": needs_review,
        "notes": notes or [],
        "conventions": ["continuous_bracketing"],
    }
    if code_offset is not None:
        d["kind"] = "yob_code"
        d["code_offset"] = code_offset
    return d


def age_years(variable, missing_values=(), notes=None, needs_review=False):
    return {
        "kind": "age_years",
        "variable": variable,
        "missing_values": [str(m) for m in missing_values],
        "valid_age_range": [16, 110],
        "needs_review": needs_review,
        "notes": notes or [],
        "conventions": ["continuous_bracketing"],
    }


def amount(variable, missing_values=(), *, max_plausible=1_000_000,
           notes=None, needs_review=False):
    return {
        "kind": "amount",
        "variable": variable,
        "missing_values": [str(m) for m in missing_values],
        "max_plausible": max_plausible,
        "min_plausible": 0,
        "needs_review": needs_review,
        "notes": notes or [],
        "conventions": ["continuous_bracketing"],
    }


# --------------------------------------------------------------------------
# Reusable code schemes
# --------------------------------------------------------------------------

# CROP / CECD : 7 tranches d'âge fermées.
CROP_AGE7 = {1: "18_24", 2: "25_34", 3: "35_44", 4: "45_54", 5: "55_64",
             6: "65_74", 7: "75_plus"}

# EEQ : CLAGE / q0age, codes 2..8.
EEQ_AGE = {2: "18_24", 3: "25_34", 4: "35_44", 5: "45_54", 6: "55_64",
           7: "65_74", 8: "75_plus", 99: MISSING, 98: MISSING}

# CROP / CECD : scolarité 10 niveaux (charte 2013, can 2011).
CROP_EDU10 = {
    1: "no_diploma", 2: "no_diploma", 3: "no_diploma",
    4: "high_school", 5: "high_school",
    6: "college_trades", 7: "college_trades",
    8: "bachelor", 9: "bachelor", 10: "above_bachelor",
    9999: MISSING,
}

# CES (cps19/cps21/cps25_education, ces_2019_phone q61) : 11 niveaux + DK.
CES_EDU = {
    1: "no_diploma", 2: "no_diploma", 3: "no_diploma", 4: "no_diploma",
    5: "high_school",
    6: "high_school",
    7: "college_trades",
    8: coarse(["high_school", "college_trades"],
              "« Some university » : aucun diplôme postsecondaire garanti."),
    9: "bachelor", 10: "above_bachelor", 11: "above_bachelor",
    12: MISSING,
}

CES_PROV13 = {1: "ab", 2: "bc", 3: "mb", 4: "nb", 5: "nl", 6: "nt", 7: "ns",
              8: "nu", 9: "on", 10: "pe", 11: "qc", 12: "sk", 13: "yt"}

CES_LANG_HOME = {1: "english", 2: "french", 18: MISSING}
CES_LANG_HOME.update({c: "other" for c in range(3, 18)})

# CES employment/income catégoriel (cps19_income_cat, cps21_income_cat, cps25_income)
CES_INCOME_CAT = {
    1: "under_60k",          # No income
    2: "under_60k",          # $1 - $30,000
    3: "under_60k",          # $30,001 - $60,000
    4: "60k_100k",           # $60,001 - $90,000
    5: coarse(["60k_100k", "100k_plus"],
              "Tranche 90 001-110 000 $ à cheval sur la borne canonique 100 k$."),
    6: "100k_plus", 7: "100k_plus", 8: "100k_plus",
    9: MISSING,
}

EEQ_EDU11 = {  # eeq_2007 (codes zéro-padés), eeq_2008
    1: "no_diploma", 2: "no_diploma", 3: "no_diploma", 4: "no_diploma",
    5: "high_school",
    6: "high_school",
    7: "college_trades",
    8: "college_trades",
    9: "bachelor", 10: "above_bachelor", 11: "above_bachelor",
    98: MISSING, 99: MISSING,
}

EEQ_INCOME10 = {  # eeq_2007 / eeq_2008 : <20k ... >100k par tranche de 10k
    1: "under_60k", 2: "under_60k", 3: "under_60k", 4: "under_60k",
    5: "under_60k", 6: "60k_100k", 7: "60k_100k", 8: "60k_100k",
    9: "60k_100k", 10: "100k_plus", 98: MISSING, 99: MISSING,
}

EEQ_INCOME9 = {  # eeq_2012 / eeq_2014 : 8/16/24/40/56/72/88/104 k$
    1: "under_60k", 2: "under_60k", 3: "under_60k", 4: "under_60k",
    5: "under_60k", 6: "60k_100k", 7: "60k_100k", 8: "60k_100k",
    9: "100k_plus", 98: MISSING, 99: MISSING,
}

EEQ_LANG_HOME = {  # q80 / Q107 / Q66 / q70 : 1=Anglais, 2=Français
    1: "english", 2: "french", 96: "other", 98: MISSING, 99: MISSING,
}
EEQ_LANG_HOME.update({c: "other" for c in range(3, 17)})

QC_RA17 = {i: RA_ALL[i - 1] for i in range(1, 18)}
# Variante utilisée par cecd_sante_can_usa (QQC) : 10 et 11 inversés.
QC_RA17_SANTE = dict(QC_RA17)
QC_RA17_SANTE[10] = "gaspesie_iles"
QC_RA17_SANTE[11] = "nord_du_quebec"

RMR_MTL_RING = ["laval", "lanaudiere", "laurentides", "monteregie"]
RMR_QC = ["capitale_nationale", "chaudiere_appalaches"]
REST_OF_QC_WITHOUT_MTL_CMA = [r for r in RA_ALL if r not in (["montreal"] + RMR_MTL_RING)]
REST_OF_QC_WITHOUT_MTL_AND_QC_CMA = [
    r for r in RA_ALL if r not in (["montreal"] + RMR_MTL_RING + RMR_QC)
]


# --------------------------------------------------------------------------
# Per-survey crosswalks (17 sondages du périmètre)
# --------------------------------------------------------------------------

QC_ONLY_NOTE = ["Échantillon québécois : la province est une constante du sondage."]


def build_crosswalk() -> dict:
    s: dict[str, dict] = {}

    # ---------------- CROP / CECD ----------------
    s["cecd_charte_2013_10"] = {
        "age": cat("qage", CROP_AGE7),
        "gender": cat("sexe", {1: "man", 2: "woman"}),
        "education": cat("qetud", CROP_EDU10,
                         conventions=["highest_completed_credential",
                                      "qc_admission_prerequisite"]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("reg", {
            1: "montreal",
            2: "monteregie",
            3: coarse(["laval", "lanaudiere", "laurentides"],
                      "« Montréal reste RMR rive-nord »."),
            4: coarse(RMR_QC, "RMR de Québec = Capitale-Nationale + partie de Chaudière-Appalaches."),
            5: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA, "« Reste du Québec »."),
        }),
        "income": cat("qreve", {1: "under_60k", 2: "under_60k", 3: "under_60k",
                                4: "60k_100k", 5: "60k_100k",
                                6: "100k_plus", 7: "100k_plus", 9: MISSING}),
        "language": cat("qlanm", {1: "french", 2: "english", 6: "other"},
                        notes=["qlanm = langue maternelle (qlanf = langue d'usage, non retenue)."]),
    }

    s["cecd_elxn_can_2011"] = {
        "age": cat("qage", CROP_AGE7),
        "gender": cat("sexe", {1: "man", 2: "woman"}),
        "education": cat("qetud", CROP_EDU10,
                         conventions=["highest_completed_credential",
                                      "qc_admission_prerequisite"]),
        "region": const("qc", QC_ONLY_NOTE + [
            "Sondage fédéral 2011 mais échantillon et variable `reg` strictement québécois."]),
        "region_qc": cat("reg", {
            1: "montreal",
            2: coarse(RMR_MTL_RING, "« Montréal reste RMR »."),
            3: coarse(RMR_QC),
            4: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA),
        }),
        "income": cat("qreve", {1: "under_60k", 2: "under_60k", 3: "under_60k",
                                4: "60k_100k", 5: "60k_100k",
                                6: "100k_plus", 7: "100k_plus", 9: MISSING}),
        "language": cat("qlanm", {1: "french", 2: "english", 6: "other"}),
    }

    s["cecd_elxn_qc_1998"] = {
        "age": cat("age", {
            1: "18_24", 2: "25_34", 3: "35_44", 4: "45_54", 5: "55_64",
            6: coarse(["65_74", "75_plus"],
                      "Code 6 observé dans les microdonnées (n=234) mais ABSENT du catalogue "
                      "(qui ne déclare que 5 tranches, 18-24 à 55-64). Lu comme « 65 ans et plus ».",
                      needs_review=True),
        }, needs_review=True,
           notes=["Divergence catalogue/microdonnées sur le code 6."]),
        "gender": cat("sexe_post", {1: "man", 2: "woman"}),
        "education": cat("scol", {
            1: "no_diploma",
            2: coarse(["high_school", "college_trades"],
                      "« 10-15 ans » de scolarité : secondaire complété jusqu'au cégep."),
            3: coarse(["bachelor", "above_bachelor"], "« univ. + »."),
        }, needs_review=True,
           notes=["Scolarité en 3 niveaux seulement : mapping surjectif partiel, "
                  "aucune granularité canonique atteignable."],
           conventions=["coarse_partial_surjection"]),
        "region": const("qc", QC_ONLY_NOTE),
    }

    s["cecd_elxn_qc_2007"] = {
        "age": cat("age", {1: "18_24", 2: "25_34", 3: "35_44", 4: "45_54",
                           5: "55_64",
                           6: coarse(["65_74", "75_plus"], "« 65 ans et plus »."),
                           9: MISSING}),
        "gender": cat("sexe", {1: "man", 2: "woman"},
                      notes=["`genre_post` écarté : 779/2442 valeurs nulles."]),
        "education": cat("scol", {
            1: "no_diploma",
            2: coarse(["no_diploma", "high_school"],
                      "« 8 à 12 années (secondaire) » : inclut les non-diplômés."),
            3: "college_trades",
            4: coarse(["bachelor", "above_bachelor"], "« 16 années ou plus (Université) »."),
            9: MISSING,
        }, needs_review=True,
           notes=["Scolarité mesurée en années d'études, pas en diplômes."],
           conventions=["coarse_partial_surjection"]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("reg2", {
            1: "abitibi_temiscamingue", 2: "bas_saint_laurent",
            3: "chaudiere_appalaches", 4: "chaudiere_appalaches",
            5: "cote_nord", 6: "estrie", 7: "gaspesie_iles",
            8: "lanaudiere", 9: "laurentides", 10: "mauricie",
            11: "monteregie", 12: "lanaudiere", 13: "laurentides",
            14: "laval", 15: "monteregie", 16: "montreal",
            17: "nord_du_quebec", 18: "outaouais",
            19: "capitale_nationale", 20: "capitale_nationale",
            21: "saguenay_lac_saint_jean", 22: "centre_du_quebec",
        }, notes=["`reg2` (22 sous-régions RA x RMR) retenue plutôt que `reg` "
                  "(3 groupes) : elle se replie exactement sur les 17 RA.",
                  "`reg2_pst` écartée : 2050/2442 valeurs nulles."]),
        "income": cat("revenu", {1: "under_60k", 2: "under_60k", 3: "under_60k",
                                 4: "60k_100k",
                                 5: coarse(["60k_100k", "100k_plus"],
                                           "Tranche ouverte « 80 000 $ et plus »."),
                                 9: MISSING}),
        "language": cat("lmat", {1: "french", 2: "english", 6: "other", 9: MISSING}),
    }

    s["cecd_elxn_qc_2012"] = {
        "age": cat("age", {1: "18_24", 2: "25_34", 3: "35_44", 4: "45_54",
                           5: "55_64",
                           6: coarse(["65_74", "75_plus"], "« 65+ »."),}),
        "gender": cat("sexe", {1: "man", 2: "woman"}),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("reg", {
            1: "montreal",
            2: coarse(RMR_MTL_RING, "« Rest of Montreal CMA »."),
            3: coarse(RMR_QC, "« Quebec CMA »."),
            4: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA, "« Rest of Quebec »."),
        }),
        "language": cat("lmat", {1: "french", 2: "english", 96: "other"}),
    }

    s["cecd_sante_can_usa"] = {
        "age": yob("BIRTH", 2011, missing_values=[9999]),
        "gender": cat("SEX", {1: "woman", 2: "man"},
                      notes=["Attention : codage inversé par rapport aux autres "
                             "sondages (1=Female, 2=Male) — conforme au catalogue."]),
        "education": cat("SCOL", {
            1: "no_diploma", 2: "no_diploma", 3: "no_diploma",
            4: "high_school",
            5: "high_school",
            6: "college_trades",
            7: coarse(["high_school", "college_trades"],
                      "« Less than university completed » : diplôme postsecondaire incertain.",
                      needs_review=True),
            8: coarse(["bachelor", "above_bachelor"],
                      "« University completed » : ne distingue pas le baccalauréat des cycles supérieurs."),
            9: MISSING,
        }, needs_review=True,
           conventions=["highest_completed_credential", "some_university_outside_qc",
                        "coarse_partial_surjection"],
           notes=["Échelle anglophone en 9 niveaux « Less than X / X completed »."]),
        "region": cat("PROV", {
            "AB": "ab", "BC": "bc", "MB": "mb", "NB": "nb", "NL": "nl",
            "NS": "ns", "NT": "nt", "NU": "nu", "ON": "on", "PE": "pe",
            "QC": "qc", "SK": "sk", "YK": "yt",
            "": "outside_canada",
        }, notes=["Chaîne vide observée pour 3542/7064 répondants : ce sont les "
                  "répondants américains (variable STATE renseignée). Non déclarée au catalogue.",
                  "Code catalogue 'YK' (et non 'YT') pour le Yukon."]),
        "region_qc": cat("QQC", QC_RA17_SANTE,
                         notes=["ATTENTION : QQC inverse les codes 10 et 11 par rapport "
                                "aux EEQ (ici 10=Gaspésie, 11=Nord-du-Québec).",
                                "Renseignée seulement pour les 946 répondants du Québec."]),
        "income": cat("REVEN", {1: "under_60k", 2: "under_60k", 3: "under_60k",
                                4: "under_60k", 5: "under_60k",
                                6: "60k_100k", 7: "60k_100k", 8: "60k_100k",
                                9: "60k_100k",
                                10: "100k_plus", 11: "100k_plus", 12: "100k_plus",
                                99: MISSING}),
        "language": cat("MTONG", {1: "english", 2: "french", 3: "other", 4: "other",
                                  5: "other", 6: "other", 7: "other", 96: "other"},
                        notes=["`LANG` (EN/FR) est la langue d'administration du "
                               "questionnaire, pas un attribut du répondant : écartée "
                               "malgré son sociodemo_type=language au catalogue."]),
    }

    # ---------------- CES ----------------
    s["ces_2019_online"] = {
        "age": yob("cps19_yob", 2019, code_offset=1919,
                   notes=["Le catalogue déclare `cps19_yob` continue ; les microdonnées "
                          "contiennent en fait un CODE 1..82. année = 1919 + code.",
                          "Vérifié : 2019 - (1919 + cps19_yob) == cps19_age pour les "
                          "37 822 répondants (cps19_age existe au Parquet mais n'est pas "
                          "marquée is_sociodemo au catalogue)."]),
        "gender": cat("cps19_gender", {1: "man", 2: "woman", 3: "non_binary_or_other"},
                      notes=["Aucun libellé au catalogue (var_type=continuous). Codes "
                             "alignés sur cps21_genderid (1=A man, 2=A woman, 3=autre), "
                             "même libellé de question."]),
        "education": cat("cps19_education", CES_EDU,
                         conventions=["highest_completed_credential",
                                      "some_university_outside_qc"],
                         notes=["Aucun libellé au catalogue ; 12 codes observés, identiques "
                                "au schéma cps21_education (même wording de question)."]),
        "region": cat("cps19_province", {k + 13: v for k, v in {
            1: "ab", 2: "bc", 3: "mb", 4: "nb", 5: "nl", 6: "nt", 7: "ns",
            8: "nu", 9: "on", 10: "pe", 11: "qc", 12: "sk", 13: "yt"}.items()},
            needs_review=True,
            notes=["Codes observés 14..26, sans libellé au catalogue. Décalage de +13 "
                   "sur l'ordre alphabétique anglais utilisé par pes19_province (1..13) ; "
                   "les effectifs le confirment (22=ON 14 808, 24=QC 8 399, 14=AB 4 481).",
                   "À valider sur le codebook officiel du CES 2019 avant post-stratification."]),
        "income": {
            "kind": "coalesce",
            "needs_review": False,
            "notes": ["cps19_income_number (montant, 25 556 non nuls) puis "
                      "cps19_income_cat (tranches, posée en repli). Valeurs aberrantes "
                      "observées jusqu'à 6.7e60 -> `missing`."],
            "conventions": ["continuous_bracketing"],
            "sources": [
                amount("cps19_income_number", max_plausible=10_000_000),
                cat("cps19_income_cat", CES_INCOME_CAT),
            ],
        },
        "language": cat("pes19_lang", {68: "english", 69: "french", 85: MISSING,
                                       **{c: "other" for c in range(70, 85)}},
                        concept="language_home", needs_review=True,
                        notes=["Seule variable de langue du sondage, posée à la vague "
                               "post-électorale : 27 482/37 822 valeurs nulles (couverture 27 %).",
                               "Concept = langue parlée à la maison, pas langue maternelle."]),
    }

    s["ces_2019_phone"] = {
        "age": yob("q2", 2019, missing_values=[-9, -8, -7],
                   notes=["Le catalogue ne déclare que -9/-8/-7 ; les microdonnées "
                          "contiennent l'ANNÉE brute (1919..2001)."]),
        "gender": cat("q3", {1: "man", 2: "woman", 3: "non_binary_or_other",
                             -9: MISSING, -8: MISSING, -7: MISSING}),
        "education": cat("q61", {**CES_EDU, -9: MISSING, -8: MISSING, -7: MISSING},
                         conventions=["highest_completed_credential",
                                      "some_university_outside_qc"]),
        "region": cat("q4", {1: "nl", 2: "pe", 3: "ns", 4: "nb", 5: "qc", 6: "on",
                             7: "mb", 8: "sk", 9: "ab", 10: "bc", 11: "nt",
                             12: "yt", 13: "nu",
                             -9: MISSING, -8: MISSING, -7: MISSING},
                      notes=["Ordre géographique est-ouest, différent de l'ordre "
                             "alphabétique des CES en ligne."]),
        "income": amount("q69", missing_values=[-9, -8, -7], max_plausible=10_000_000,
                         notes=["Le catalogue ne déclare que -9/-8/-7 ; les microdonnées "
                                "contiennent le MONTANT en dollars (0..2 120 000)."]),
        "language": cat("q67", {1: "english", 4: "french",
                                **{c: "other" for c in list(range(2, 4)) + list(range(5, 32))},
                                -9: MISSING, -8: MISSING, -7: MISSING},
                        notes=["Langue maternelle (« first language you learned »)."]),
    }

    s["ces_2021"] = {
        "age": yob("cps21_yob", 2021, code_offset=1919,
                   notes=["Code 1..91 = 1920..2010 (catalogue). Recoupé avec "
                          "cps21_yob_2 (codage inverse, 1=2003) : 0 désaccord sur "
                          "20 968 lignes, et égalité exacte avec cps21_age."]),
        "gender": cat("cps21_genderid", {1: "man", 2: "woman",
                                         3: "non_binary_or_other",
                                         4: "non_binary_or_other"}),
        "education": cat("cps21_education", CES_EDU,
                         conventions=["highest_completed_credential",
                                      "some_university_outside_qc"]),
        "region": cat("cps21_province", CES_PROV13),
        "income": {
            "kind": "coalesce",
            "needs_review": False,
            "notes": ["cps21_income_number puis cps21_income_cat. Sentinelle -99 "
                      "(642 cas) NON déclarée au catalogue -> `missing` ; valeurs "
                      "aberrantes jusqu'à 1.1e30 -> `missing`."],
            "conventions": ["continuous_bracketing"],
            "sources": [
                amount("cps21_income_number", missing_values=[-99],
                       max_plausible=10_000_000),
                cat("cps21_income_cat", CES_INCOME_CAT),
            ],
        },
        "language": cat("pes21_lang", CES_LANG_HOME, concept="language_home",
                        needs_review=True,
                        notes=["Vague post-électorale seulement : 5 899/20 968 nulles "
                               "(couverture 72 %).",
                               "Concept = langue parlée à la maison."]),
    }

    s["ces_2025"] = {
        "age": age_years("cps25_age_in_years",
                         notes=["Âge en années directement (18..96). cps25_yob "
                                "(code 1..91 = 1920..2010) donne le même âge à ±1 an "
                                "près selon la date d'anniversaire."]),
        "gender": cat("cps25_genderid", {1: "man", 2: "woman",
                                         3: "non_binary_or_other",
                                         4: "non_binary_or_other"}),
        "education": cat("cps25_education", CES_EDU,
                         conventions=["highest_completed_credential",
                                      "some_university_outside_qc"]),
        "region": cat("cps25_province", CES_PROV13),
        "income": cat("cps25_income", CES_INCOME_CAT),
        "language": cat("pes25_lang", CES_LANG_HOME, concept="language_home",
                        needs_review=True,
                        notes=["Vague post-électorale seulement : 6 105/20 180 nulles "
                               "(couverture 70 %).",
                               "Concept = langue parlée à la maison."]),
    }

    # ---------------- EEQ ----------------
    s["eeq_2007"] = {
        "age": yob("q75", 2007, missing_values=[9999]),
        "gender": cat("q76", {1: "man", 2: "woman"}),
        "education": cat("q77", EEQ_EDU11,
                         conventions=["highest_completed_credential",
                                      "qc_admission_prerequisite"]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("nomx", {
            1: "bas_saint_laurent", 2: "saguenay_lac_saint_jean",
            3: "capitale_nationale", 4: "mauricie", 5: "estrie",
            6: "montreal", 7: "outaouais", 8: "abitibi_temiscamingue",
            9: "cote_nord", 11: "gaspesie_iles", 12: "chaudiere_appalaches",
            13: "laval", 14: "lanaudiere", 15: "laurentides",
            16: "monteregie", 17: "centre_du_quebec",
            24: "lanaudiere", 25: "laurentides", 26: "monteregie",
            32: "chaudiere_appalaches", 33: "capitale_nationale",
        }, notes=["21 sous-groupes = 16 RA éclatées en volets RMR / hors-RMR ; "
                  "recollées sur la RA. Aucun code Nord-du-Québec."]),
        "income": cat("q78", EEQ_INCOME10),
        "language": cat("q80", EEQ_LANG_HOME, concept="language_home",
                        needs_review=True,
                        notes=["`langu` (langue maternelle) écartée : codes 3,4,5,6,7 "
                               "observés dans les microdonnées alors que le catalogue ne "
                               "déclare que 1=Français, 2=Anglais, 9=Nsp/Refus.",
                               "Repli sur q80 = langue parlée à la maison (concept différent)."]),
    }

    s["eeq_2008"] = {
        "age": cat("q0age", EEQ_AGE,
                   notes=["`q0age` (tranches, 0 valeur nulle) retenue plutôt que "
                          "`q75` (année de naissance, 26 nulles)."]),
        "gender": cat("q76", {1: "man", 2: "woman"}),
        "education": cat("q77", EEQ_EDU11,
                         conventions=["highest_completed_credential",
                                      "qc_admission_prerequisite"]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("reg", {
            1: coarse(["montreal"] + RMR_MTL_RING, "« Mtl RMR »."),
            2: coarse(RMR_QC, "« Qc RMR »."),
            3: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA,
                      "« Est » : découpage non défini au catalogue.", needs_review=True),
            4: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA,
                      "« Centre » : découpage non défini au catalogue.", needs_review=True),
            5: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA,
                      "« Ouest » : découpage non défini au catalogue.", needs_review=True),
        }, needs_review=True,
           notes=["Les groupes Est / Centre / Ouest ne sont définis nulle part : "
                  "impossible de les rattacher à des RA sans le codebook."]),
        "income": cat("q78", EEQ_INCOME10),
        "language": cat("langu", {1: "french", 2: "english", 3: "other", 9: MISSING}),
    }

    s["eeq_2012"] = {
        "age": yob("AGEX", 2012, missing_values=[9999]),
        "gender": cat("SEXE", {1: "man", 2: "woman"}),
        "education": cat("SCOL", {
            1: "no_diploma", 2: "no_diploma", 3: "no_diploma", 4: "no_diploma",
            5: "high_school",
            6: "high_school",
            7: "college_trades", 8: "college_trades", 9: "college_trades",
            10: "college_trades",
            11: "bachelor", 12: "above_bachelor",
            98: MISSING, 99: MISSING,
        }, conventions=["highest_completed_credential", "qc_admission_prerequisite"],
           notes=["Code 10 « Certificat ou diplôme », placé entre « Université non "
                  "complétée » et « Baccalauréat » : lu comme certificat universitaire "
                  "inférieur au baccalauréat."]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("Q0QC", QC_RA17),
        "income": cat("REVEN", EEQ_INCOME9),
        "language": cat("LANGU", {1: "french", 2: "english", 3: "other", 9: MISSING}),
    }

    s["eeq_2014"] = {
        "age": cat("CLAGE", EEQ_AGE,
                   notes=["`CLAGE` (tranches) retenue ; `QAGE` (année de naissance) "
                          "donne le même découpage."]),
        "gender": cat("QSEXE", {1: "man", 2: "woman"}),
        "education": cat("QSCOL", {
            1: "no_diploma", 2: "no_diploma", 3: "no_diploma", 4: "no_diploma",
            5: "high_school",
            6: "high_school",
            7: "college_trades", 8: "college_trades", 9: "college_trades",
            10: "bachelor", 11: "above_bachelor",
            99: MISSING,
        }, conventions=["highest_completed_credential", "qc_admission_prerequisite"]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("QREGION", QC_RA17,
                         notes=["`REGIO` (MTL RMR / QC RMR / autres) écartée : plus grossière."]),
        "income": cat("Q57", EEQ_INCOME9),
        "language": cat("QLANG", {
            1: "french", 2: "english", 3: "other",
            4: "multiple", 5: "multiple", 6: "multiple",
            8: MISSING, 9: MISSING,
        }),
    }

    s["eeq_2018"] = {
        "age": cat("age", {
            0: "under_18",
            1: coarse(["under_18", "18_24"], "« Z (nés après 1999) » : 18 ans ou moins en 2018.",
                      needs_review=True),
            2: coarse(["18_24", "25_34", "35_44"], "« Milléniaux (1980-1999) » : 19-38 ans en 2018.",
                      needs_review=True),
            3: coarse(["35_44", "45_54", "55_64"], "« X (1960-1979) » : 39-58 ans en 2018.",
                      needs_review=True),
            4: coarse(["55_64", "65_74"], "« Baby-boomers (1945-1959) » : 59-73 ans en 2018.",
                      needs_review=True),
            5: coarse(["65_74", "75_plus"], "« Pré-baby-boomers (avant 1945) » : 74 ans et plus.",
                      needs_review=True),
        }, needs_review=True,
           notes=["Seule variable d'âge exploitable : des cohortes générationnelles, "
                  "transversales aux tranches canoniques. Aucune cellule d'âge propre.",
                  "`agenum` (âge en années) écartée : 3 031/3 072 valeurs nulles."],
           conventions=["coarse_partial_surjection"]),
        "gender": cat("qsexe", {1: "man", 2: "woman"}),
        "education": cat("qscol", {
            1: "no_diploma", 2: "high_school", 3: "college_trades",
            4: "college_trades", 5: "college_trades", 6: "bachelor",
            7: "above_bachelor", 8: "above_bachelor", 9: "above_bachelor",
            10: "no_diploma", 11: "no_diploma", 12: "no_diploma",
            13: "high_school",
            14: "high_school",
            15: "college_trades",
            16: coarse(["bachelor", "above_bachelor"], "« Université - Diplôme universitaire ou grade »."),
            98: MISSING, 99: MISSING,
        }, needs_review=True,
           conventions=["highest_completed_credential", "qc_admission_prerequisite"],
           notes=["La variable superpose DEUX échelles incompatibles : 1-9 (échelle "
                  "recensement « plus haut diplôme ») et 10-16 (échelle EEQ historique "
                  "« complété / non complété »). Les deux blocs sont peuplés "
                  "simultanément (code 14 n=589, code 8 n=457), ce qui n'est pas "
                  "interprétable sans le codebook.",
                  "Arbitrage requis : soit on ne retient qu'un bloc, soit on obtient "
                  "la règle de recodage d'origine."]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("q0qc", QC_RA17,
                         notes=["`regio` (3 groupes) écartée : plus grossière."]),
        "income": {
            "kind": "coalesce",
            "needs_review": True,
            "notes": ["q57 (sous-groupe fin, 2 177 non nuls) puis q56 (4 groupes larges).",
                      "Code 97 de q56 observé (n=478) et ABSENT du catalogue -> `missing`. "
                      "C'est très probablement le renvoi vers q57 ; à confirmer."],
            "conventions": ["continuous_bracketing", "coarse_partial_surjection"],
            "sources": [
                cat("q57", {1: "under_60k", 2: "under_60k", 3: "under_60k",
                            4: "60k_100k", 5: "60k_100k",
                            6: "100k_plus", 7: "100k_plus",
                            98: MISSING, 99: MISSING}),
                cat("q56", {1: "under_60k", 2: "under_60k", 3: "60k_100k",
                            4: coarse(["60k_100k", "100k_plus"],
                                      "« 90 000 $ ou plus » : à cheval sur la borne 100 k$."),
                            97: MISSING, 98: MISSING, 99: MISSING}),
            ],
        },
        "language": cat("qlangue", {1: "french", 2: "english", 96: "other",
                                    98: MISSING, 99: MISSING}),
    }

    # ---------------- Panels provinciaux ----------------
    s["provincial_qc_2012"] = {
        "age": yob("YOB", 2012, missing_values=[9999]),
        "gender": cat("GEND", {1: "man", 2: "woman"}),
        "education": cat("SD4", {
            1: "no_diploma", 2: "no_diploma", 3: "no_diploma", 4: "no_diploma",
            5: "high_school",
            6: "high_school",
            7: "college_trades",
            8: "college_trades",
            9: "bachelor", 10: "above_bachelor",
            11: MISSING,
        }, needs_review=True,
           conventions=["highest_completed_credential", "qc_admission_prerequisite"],
           notes=["Code 11 observé (n=20) et ABSENT du catalogue (qui s'arrête à 10) "
                  "-> `missing` en attendant arbitrage."]),
        "region": const("qc", QC_ONLY_NOTE + [
            "SD2A « Résidez-vous au Québec ? » vaut 1 (oui) pour les 990 répondants."]),
        "income": cat("SD5", {
            1: "under_60k", 2: "under_60k", 3: "under_60k", 4: "under_60k",
            5: "under_60k",
            6: "60k_100k", 7: "60k_100k", 8: "60k_100k", 9: "60k_100k",
            10: "100k_plus", 11: "100k_plus",
            12: MISSING, 98: MISSING, 99: MISSING,
        }, needs_review=True,
           notes=["Code 12 observé (n=100) et ABSENT du catalogue (qui s'arrête à "
                  "11 = « 120 000 $ et plus ») -> `missing`. Codes 98/99 non déclarés "
                  "non plus mais sans ambiguïté (NSP / refus)."]),
        "language": cat("MLANG", {1: "english", 2: "french", 3: "other"},
                        notes=["`SD6F` écartée : code 88 observé (n=18) absent du catalogue."]),
    }

    s["provincial_qc_2018"] = {
        "age": cat("age", {
            1: coarse(["18_24", "25_34"], "« 18-34 »."),
            2: coarse(["35_44", "45_54"], "« 35-54 »."),
            3: coarse(["55_64", "65_74", "75_plus"], "« 55+ »."),
            4: MISSING,
        }, needs_review=True,
           conventions=["coarse_partial_surjection"],
           notes=["Âge en 3 tranches seulement ; aucune cellule canonique atteignable."]),
        "gender": cat("sexfix", {1: "man", 2: "woman"}),
        "education": cat("d3", {
            1: "no_diploma", 2: "no_diploma", 3: "high_school",
            4: "college_trades", 5: "college_trades", 6: "college_trades",
            7: "bachelor", 8: "above_bachelor", 9: MISSING,
        }, conventions=["highest_completed_credential"],
           notes=["Échelle déjà calquée sur les catégories du recensement : "
                  "mapping bijectif à la collapse près."]),
        "region": const("qc", QC_ONLY_NOTE),
        "region_qc": cat("fsa_tabl", {
            1: "montreal",
            2: coarse(RMR_MTL_RING, "« Couronne » : couronnes nord et sud de Montréal.",
                      needs_review=True),
            3: coarse(RMR_QC, "« Région de Québec »."),
            4: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA,
                      "« Régions du Sud » : découpage non défini au catalogue.",
                      needs_review=True),
            5: coarse(REST_OF_QC_WITHOUT_MTL_AND_QC_CMA,
                      "« Régions du Nord » : découpage non défini au catalogue.",
                      needs_review=True),
        }, needs_review=True,
           notes=["Regroupement bâti sur les FSA ; la frontière Sud/Nord n'est pas "
                  "documentée et ne correspond à aucun découpage administratif connu."]),
        "income": cat("d5", {1: "under_60k", 2: "under_60k", 3: "under_60k",
                             4: "60k_100k", 5: "60k_100k",
                             6: "100k_plus", 7: "100k_plus", 8: MISSING}),
        "language": cat("s1", {1: "french", 2: "english", 3: "other", 4: MISSING}),
    }

    return {
        "version": VERSION,
        "generated_by": "scripts/11_build_ses_crosswalks.py",
        "canonical": "data/crosswalks/ses_canonical.json",
        "n_surveys": len(s),
        "surveys": s,
    }


# --------------------------------------------------------------------------
# Coverage report (needs the microdata)
# --------------------------------------------------------------------------

COVERAGE_COLUMNS = [
    "survey_id", "dimension", "source_variable", "source_kind", "concept",
    "n_respondents", "n_source_levels", "n_source_levels_unmapped",
    "n_canonical_levels_reached", "n_canonical_levels_total",
    "pct_non_missing", "pct_coarse", "pct_unmapped_code",
    "needs_review", "notes",
]


def _download_parquets(survey_ids, cache_dir: Path) -> dict[str, Path]:
    """Pull one Parquet per survey from the Blob (container survey-responses)."""
    from azure.storage.blob import BlobServiceClient
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    account = os.environ["AZURE_STORAGE_ACCOUNT"]
    key = os.environ["AZURE_STORAGE_KEY"]
    client = BlobServiceClient(f"https://{account}.blob.core.windows.net",
                               credential=key).get_container_client("survey-responses")
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sid in survey_ids:
        dest = cache_dir / f"{sid}.parquet"
        if not dest.exists():
            dest.write_bytes(client.download_blob(f"{sid}.parquet").readall())
        paths[sid] = dest
    return paths


def build_coverage(xw, paths: dict[str, Path]) -> list[dict]:
    import polars as pl

    rows: list[dict] = []
    for sid in sorted(xw.survey_ids):
        frame = pl.read_parquet(paths[sid])
        records = frame.to_dicts()
        n = len(records)
        for dim in ses_mod.DIMENSIONS:
            spec = xw.spec(sid, dim)
            if spec is None:
                rows.append({
                    "survey_id": sid, "dimension": dim, "source_variable": "",
                    "source_kind": "none", "concept": "", "n_respondents": n,
                    "n_source_levels": 0, "n_source_levels_unmapped": 0,
                    "n_canonical_levels_reached": 0,
                    "n_canonical_levels_total": len(xw.levels(dim)) - 1,
                    "pct_non_missing": 0.0, "pct_coarse": 0.0,
                    "pct_unmapped_code": 0.0, "needs_review": "",
                    "notes": "aucune variable ne porte cette dimension",
                })
                continue
            variables = list(xw.source_variables(sid, dim))
            source_levels: set[str] = set()
            for var in variables:
                if var in frame.columns:
                    source_levels |= {
                        c for c in (ses_mod.normalize_code(v)
                                    for v in frame[var].unique().to_list())
                        if c is not None
                    }
            reached: set[str] = set()
            unmapped_codes: set[str] = set()
            n_ok = n_coarse = n_unmapped = 0
            for rec in records:
                val = xw.apply(sid, dim, rec)
                if val.unmapped:
                    n_unmapped += 1
                    unmapped_codes.add(str(ses_mod.normalize_code(val.raw)))
                if val.is_missing:
                    continue
                n_ok += 1
                reached |= set(val.levels)
                if val.coarse:
                    n_coarse += 1
            notes = list(spec.get("notes", []))
            if unmapped_codes:
                notes.append("codes non mappés observés : "
                             + ", ".join(sorted(unmapped_codes)[:10]))
            rows.append({
                "survey_id": sid, "dimension": dim,
                "source_variable": "+".join(variables) or "(constante)",
                "source_kind": spec["kind"],
                "concept": spec.get("concept", ""),
                "n_respondents": n,
                "n_source_levels": len(source_levels),
                "n_source_levels_unmapped": len(unmapped_codes),
                "n_canonical_levels_reached": len(reached),
                "n_canonical_levels_total": len(xw.levels(dim)) - 1,
                "pct_non_missing": round(100 * n_ok / n, 2) if n else 0.0,
                "pct_coarse": round(100 * n_coarse / n, 2) if n else 0.0,
                "pct_unmapped_code": round(100 * n_unmapped / n, 2) if n else 0.0,
                "needs_review": "yes" if spec.get("needs_review") else "no",
                "notes": " | ".join(notes),
            })
    return rows


# --------------------------------------------------------------------------

def perimeter_surveys() -> list[str]:
    directory = (REPO / NORMALIZED_DIR).resolve()
    ids = sorted(p.stem for p in directory.glob("*.json"))
    return [i for i in ids if i not in EXCLUDED_SURVEYS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-data", action="store_true",
                        help="write the JSON only, skip the Blob-backed coverage report")
    parser.add_argument("--cache-dir",
                        default=os.environ.get("SES_PARQUET_CACHE")
                        or str(Path(tempfile.gettempdir()) / "ses_parquet_cache"),
                        help="cache local des Parquet — hors dépôt, aucune microdonnée dupliquée ici")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = build_canonical()
    crosswalk = build_crosswalk()

    expected = set(perimeter_surveys())
    got = set(crosswalk["surveys"])
    if expected != got:
        raise SystemExit(f"périmètre non couvert : manquants={sorted(expected - got)} "
                         f"en trop={sorted(got - expected)}")

    (OUT_DIR / "ses_canonical.json").write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "ses_crosswalk.json").write_text(
        json.dumps(crosswalk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"écrit {OUT_DIR/'ses_canonical.json'}")
    print(f"écrit {OUT_DIR/'ses_crosswalk.json'} ({len(got)} sondages)")

    if args.no_data:
        return 0

    xw = ses_mod.SesCrosswalk.load(OUT_DIR / "ses_canonical.json",
                                   OUT_DIR / "ses_crosswalk.json")
    paths = _download_parquets(sorted(got), Path(args.cache_dir))
    rows = build_coverage(xw, paths)
    with open(OUT_DIR / "ses_coverage.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COVERAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"écrit {OUT_DIR/'ses_coverage.csv'} ({len(rows)} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
