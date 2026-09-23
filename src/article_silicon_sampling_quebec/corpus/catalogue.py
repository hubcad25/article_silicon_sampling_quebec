"""Catalogue access — question metadata from the normalized JSON files.

Source of truth is ``../mvp_moteur_recherche_sondages/ingestion/normalized/*.json``;
the Azure AI Search index ``survey-questions`` is a projection of it and is better
kept for semantic retrieval. For batch work we read the JSON directly.

Join key with the microdata: ``variable`` == Parquet column name.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from article_silicon_sampling_quebec.corpus.perimeter import (
    NORMALIZED_DIR,
    is_in_perimeter,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOGUE_DIR = (REPO_ROOT / NORMALIZED_DIR).resolve()

# Question fields carried through; absent keys are filled with the default below.
_QUESTION_DEFAULTS: dict[str, Any] = {
    "variable": None,
    "question_text": None,
    "display_label": None,
    "response_options": [],
    "var_type": None,
    "text_kind": None,
    "is_ordinal": None,
    "is_sociodemo": False,
    "sociodemo_type": None,
    "concepts": [],
    "themes": [],
}


def catalogue_survey_ids(perimeter_only: bool = False) -> list[str]:
    """Survey ids present in the normalized catalogue."""
    ids = sorted(p.stem for p in CATALOGUE_DIR.glob("*.json"))
    return [s for s in ids if is_in_perimeter(s)] if perimeter_only else ids


@lru_cache(maxsize=None)
def _load(survey_id: str) -> dict[str, Any]:
    path = CATALOGUE_DIR / f"{survey_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No normalized catalogue for survey '{survey_id}'")
    return json.loads(path.read_text(encoding="utf-8"))


def survey_meta(survey_id: str) -> dict[str, Any]:
    """Survey-level metadata: name, year, pollster, language, n_respondents, tags."""
    return dict(_load(survey_id)["survey"])


def survey_items(survey_id: str) -> list[dict[str, Any]]:
    """All catalogue items of a survey, as flat dicts with a stable key set.

    Survey-level context is carried on each item (``survey_id``, ``year``,
    ``language``, ``n_respondents_survey``) so items from several surveys can be
    concatenated directly.
    """
    doc = _load(survey_id)
    meta = doc["survey"]
    items = []
    for q in doc["questions"]:
        item = {k: q.get(k, default) for k, default in _QUESTION_DEFAULTS.items()}
        item.update(
            survey_id=meta["survey_id"],
            year=meta.get("year"),
            language=meta.get("language"),
            n_respondents_survey=meta.get("n_respondents"),
        )
        items.append(item)
    return items


def all_items(perimeter_only: bool = True) -> list[dict[str, Any]]:
    """Catalogue items across surveys."""
    items: list[dict[str, Any]] = []
    for survey_id in catalogue_survey_ids(perimeter_only=perimeter_only):
        items.extend(survey_items(survey_id))
    return items


def sociodemo_items(
    survey_id: str | None = None, perimeter_only: bool = True
) -> list[dict[str, Any]]:
    """Sociodemographic items, with their ``sociodemo_type``.

    These are the persona-building blocks (§3.4), never targets.
    """
    pool = (
        survey_items(survey_id)
        if survey_id is not None
        else all_items(perimeter_only=perimeter_only)
    )
    return [i for i in pool if i.get("is_sociodemo")]
