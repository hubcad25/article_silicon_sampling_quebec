"""Build the frozen inference set of the CES 2025 benchmark (plan §4-§5).

For each of the 15 cells (5 regions x 3 ages in 2025), 1,000 Democracy Checkup
2024 respondents are drawn with replacement, with probability proportional to
``dc24_quota_weight``. Each profile answers the ten CES 2025 items, one request
per item, rendered with the training template (``PromptTemplate``, condition
C1, k = 7, no SES dropout, survey year 2025):

    system  persona: population, age band in 2025, gender, education, province,
            household income, first language
    user    the respondent's own answers to the seven anchor items, then the
            CES 2025 item and its options, in the respondent's language

Writes benchmark/frozen/:
    requests.jsonl   one line per request: request_id, profile_id, cell,
                     language, item, messages, options (text -> CES code)
    profiles.csv     one row per profile, with its anchor answers
    manifest.json    seeds, counts, sha256 of every output

    .venv/bin/python -m benchmark.build
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

from . import AGES, CELLS, FROZEN, REGION_OF, REGIONS, REPO, TARGETS, age_band

sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec import dataset as ds  # noqa: E402
from article_silicon_sampling_quebec.corpus.ses import SesCrosswalk  # noqa: E402
from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    ContextAnswer,
    Persona,
    PromptTemplate,
    build_item_specs,
)

SEED = 20261001
PROFILES_PER_CELL = 1_000
YEAR = 2025
SOURCE = "dc_2024"
#: The seven anchors of the plan, in its order (data/anchor_map/anchor_map.csv).
ANCHORS = ("dc24_lr_self_1", "dc24_party_id", "dc24_inequality_gap", "dc24_pos_bilingualis",
           "dc24_trust2", "dc24_ties_us", "dc24_pos_equal_1")
EXTRA_FRENCH = REPO / "data" / "extra_french_wording.json"

ds.RESPONSE_LANGUAGE[SOURCE] = ("UserLanguage", frozenset({"FR-CA"}))


def specs() -> tuple[ds.SpecSet, ds.SpecSet]:
    items = pl.read_parquet(REPO / "data" / "items.parquet")
    ces = items.filter((pl.col("survey_id") == "ces_2025")
                       & pl.col("variable").is_in([f"cps25_{t}" for t in TARGETS]))
    extra = pl.read_parquet(REPO / "data" / "items_extra.parquet").filter(
        (pl.col("survey_id") == SOURCE) & pl.col("variable").is_in(ANCHORS))
    if ces.height != len(TARGETS) or extra.height != len(ANCHORS):
        raise SystemExit(f"missing items: {ces.height} targets, {extra.height} anchors")
    targets = ds.SpecSet(build_item_specs(ces.iter_rows(named=True)),
                         {"fr": build_item_specs(ds.french_item_rows(ces))})
    anchors = ds.SpecSet(build_item_specs(extra.iter_rows(named=True)),
                         {"fr": build_item_specs(ds.french_item_rows(extra, EXTRA_FRENCH))})
    return targets, anchors


def draw_profiles(panel: ds.SurveyPanel, crosswalk: SesCrosswalk) -> pd.DataFrame:
    """Respondents by cell, then the weighted draw with replacement."""
    raw = panel.raw.select("__respondent_id", "__weight", "dc24_age_in_years", "dc24_province",
                           "UserLanguage").to_pandas().rename(
        columns={"__respondent_id": "respondent_id", "__weight": "weight"})
    regions = []
    for _, row in raw.iterrows():
        value = crosswalk.apply(SOURCE, "region", {"dc24_province": row.dc24_province})
        regions.append(REGION_OF.get(value.levels[0]))
    raw["region"] = regions
    raw["age_2025"] = pd.to_numeric(raw.dc24_age_in_years, errors="coerce") + 1
    raw["age_group"] = [age_band(a) for a in raw.age_2025]
    raw["row"] = np.arange(len(raw))
    rng = np.random.default_rng(SEED)
    draws = []
    for region in REGIONS:
        for age in AGES:
            pool = raw[(raw.region == region) & (raw.age_group == age) & (raw.weight > 0)]
            weights = pool.weight.to_numpy() / pool.weight.sum()
            picked = pool.iloc[rng.choice(len(pool), size=PROFILES_PER_CELL, replace=True, p=weights)]
            draws.append(picked.assign(cell=f"{region}|{age}", pool_size=len(pool)))
    profiles = pd.concat(draws, ignore_index=True)
    profiles.insert(0, "profile_id", [f"p{i:05d}" for i in range(len(profiles))])
    return profiles


def persona(panel: ds.SurveyPanel, crosswalk: SesCrosswalk, row: int, age_2025: float,
            language: str) -> Persona:
    """The training persona, with the age band of 2025 instead of 2024."""
    source = dict(panel._ses.row(row, named=True)) if panel._ses is not None else {}
    source["dc24_age_in_years"] = age_2025
    labels = crosswalk.profile_labels(SOURCE, source, lang=language)
    return Persona(fields={k: v for k, v in labels.items() if v and v != ds.MISSING},
                   survey_id=SOURCE, year=YEAR)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    targets, anchors = specs()
    crosswalk = SesCrosswalk.load()
    panel = ds.SurveyPanel(SOURCE, crosswalk, frozenset())
    profiles = draw_profiles(panel, crosswalk)
    template = PromptTemplate(condition="C1", k=len(ANCHORS), ses_dropout="none", year_override=YEAR)

    FROZEN.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    with (FROZEN / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for p in profiles.itertuples():
            language = "fr" if p.UserLanguage == "FR-CA" else "en"
            who = persona(panel, crosswalk, p.row, p.age_2025, language)
            context, answers = [], {}
            for variable in ANCHORS:
                spec = anchors.for_language((SOURCE, variable), language)
                code = panel.answer(p.row, spec) if spec is not None else None
                label = spec.option_label(code) if code is not None else None
                answers[variable] = code
                if label:
                    context.append(ContextAnswer(item=spec, code=code, label=label))
            rows.append({"profile_id": p.profile_id, "cell": p.cell, "respondent_id": p.respondent_id,
                         "weight": p.weight, "language": language, "age_2025": p.age_2025,
                         **{f"persona_{k}": v for k, v in who.fields.items()},
                         "n_anchors": len(context), **{f"anchor_{v}": c for v, c in answers.items()}})
            for target in TARGETS:
                item = targets.for_language(("ces_2025", f"cps25_{target}"), language)
                messages = template.build_messages(who, item, context, dimensions=list(who.fields))
                record = {"request_id": f"{p.profile_id}:{target}", "profile_id": p.profile_id,
                          "cell": p.cell, "language": language, "item": target, "messages": messages,
                          "options": {o.text: o.code for o in item.options}}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    pd.DataFrame(rows).to_csv(FROZEN / "profiles.csv", index=False)

    manifest = {
        "generated_by": "benchmark/build.py", "plan": "docs/ces2025_benchmark_framework.md",
        "seed": SEED, "source_survey": SOURCE, "profiles_per_cell": PROFILES_PER_CELL,
        "cells": list(CELLS), "targets": list(TARGETS), "anchors": list(ANCHORS),
        "template": {"condition": "C1", "k": len(ANCHORS), "ses_dropout": "none", "year": YEAR},
        "n_profiles": len(profiles), "n_requests": n,
        "pool_sizes": profiles.groupby("cell").pool_size.first().to_dict(),
        "share_french": round(float(np.mean([r["language"] == "fr" for r in rows])), 4),
        "mean_anchors": round(float(np.mean([r["n_anchors"] for r in rows])), 3),
        "sha256": {name: sha256(FROZEN / name) for name in ("requests.jsonl", "profiles.csv")},
    }
    (FROZEN / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_profiles", "n_requests", "share_french",
                                                "mean_anchors", "pool_sizes")}, indent=1))


if __name__ == "__main__":
    main()
