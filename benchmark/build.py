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

from . import (AGES, CELLS, CONTEXT, FROZEN, REGION_OF, REGIONS, REPO, SET, TARGET_VARS, TARGETS,
               age_band)

sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec import dataset as ds  # noqa: E402
from article_silicon_sampling_quebec.corpus.ses import SesCrosswalk  # noqa: E402
from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    CONTEXT_MAX_COSINE,
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
#: BENCH_CONTEXT=rich: the DC 2024 core of the B datasets, then the N_NEAR
#: DC 2024 items nearest the target -- the training policy of scripts/40.
RICH_MANIFEST = REPO / "data" / "datasets_ces2025_rich" / "manifest.json"
N_NEAR = 6
if CONTEXT == "rich":
    ANCHORS = tuple(json.loads(RICH_MANIFEST.read_text())["dc_2024_core"])

ds.RESPONSE_LANGUAGE[SOURCE] = ("UserLanguage", frozenset({"FR-CA"}))


def specs() -> tuple[ds.SpecSet, ds.SpecSet, dict[str, list[str]]]:
    items = pl.read_parquet(REPO / "data" / "items.parquet")
    ces = items.filter((pl.col("survey_id") == "ces_2025")
                       & pl.col("variable").is_in(TARGET_VARS))
    extra = pl.read_parquet(REPO / "data" / "items_extra.parquet").filter(
        (pl.col("survey_id") == SOURCE) & pl.col("variable").is_in(ANCHORS))
    if ces.height != len(TARGETS) or extra.height != len(ANCHORS):
        raise SystemExit(f"missing items: {ces.height} targets, {extra.height} anchors")
    if CONTEXT == "rich":  # every screened DC 2024 item can be a nearest neighbour
        extra = s36().load_items().filter(pl.col("survey_id") == SOURCE)
    targets = ds.SpecSet(build_item_specs(ces.iter_rows(named=True)),
                         {"fr": build_item_specs(ds.french_item_rows(ces))})
    anchors = ds.SpecSet(build_item_specs(extra.iter_rows(named=True)),
                         {"fr": build_item_specs(ds.french_item_rows(extra, EXTRA_FRENCH))})
    return targets, anchors, context_variables(ces, extra)


def s36():
    """scripts/36, for the item screen and the embedding recipe of the datasets."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s36", REPO / "scripts/36_generate_benchmark_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_variables(targets_frame: pl.DataFrame, pool: pl.DataFrame) -> dict[str, list[str]]:
    """Per target: the context items, the same for every profile.

    fix: the anchors. rich: the core minus near-duplicates of the target, then
    the N_NEAR nearest non-core items below CONTEXT_MAX_COSINE (scripts/40).
    An item worded exactly like the target is dropped too: DC 2024 repeats some
    CES questions verbatim (demsat, pos_jobs) at a cosine under 0.95, and its
    answer would be the respondent's own answer to the target, a year earlier.
    """
    if CONTEXT != "rich":
        return {t: list(ANCHORS) for t in TARGETS}
    module = s36()
    both = pl.concat([targets_frame, pool], how="diagonal_relaxed")
    vectors = module.l2_normalize(module.item_vectors(both))
    tv, pv = vectors[:targets_frame.height], vectors[targets_frame.height:]
    pool_vars = list(pool["variable"])
    def norm(text):
        return " ".join(str(text).lower().split())
    pool_text = dict(zip(pool_vars, map(norm, pool["question_text"])))
    out = {}
    for i, variable in enumerate(targets_frame["variable"]):
        same = {v for v, t in pool_text.items() if t == norm(targets_frame["question_text"][i])}
        sims = np.where(np.isin(pool_vars, list(same)), 1.0, pv @ tv[i])
        cos = dict(zip(pool_vars, sims))
        core = [v for v in ANCHORS if cos.get(v, 0) < CONTEXT_MAX_COSINE]
        near = [pool_vars[j] for j in np.argsort(-sims)
                if pool_vars[j] not in ANCHORS and sims[j] < CONTEXT_MAX_COSINE][:N_NEAR]
        out[variable.split("_", 1)[1]] = core + near
    return out


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
    targets, anchors, contexts = specs()
    crosswalk = SesCrosswalk.load()
    panel = ds.SurveyPanel(SOURCE, crosswalk, frozenset())
    profiles = draw_profiles(panel, crosswalk)
    k = max(len(v) for v in contexts.values())
    template = PromptTemplate(condition="C1", k=k, ses_dropout="none", year_override=YEAR)

    FROZEN.mkdir(parents=True, exist_ok=True)
    rows, n, n_shown = [], 0, 0
    with (FROZEN / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for p in profiles.itertuples():
            language = "fr" if p.UserLanguage == "FR-CA" else "en"
            who = persona(panel, crosswalk, p.row, p.age_2025, language)
            given, answers = {}, {}
            for variable in dict.fromkeys(v for vs in contexts.values() for v in vs):
                spec = anchors.for_language((SOURCE, variable), language)
                code = panel.answer(p.row, spec) if spec is not None else None
                label = spec.option_label(code) if code is not None else None
                if variable in ANCHORS:
                    answers[variable] = code
                if label:
                    given[variable] = ContextAnswer(item=spec, code=code, label=label)
            context = [given[v] for v in ANCHORS if v in given]
            rows.append({"profile_id": p.profile_id, "cell": p.cell, "respondent_id": p.respondent_id,
                         "weight": p.weight, "language": language, "age_2025": p.age_2025,
                         **{f"persona_{k}": v for k, v in who.fields.items()},
                         "n_anchors": len(context), **{f"anchor_{v}": c for v, c in answers.items()}})
            for target, variable in zip(TARGETS, TARGET_VARS):
                item = targets.for_language(("ces_2025", variable), language)
                shown = [given[v] for v in contexts[target] if v in given]
                n_shown += len(shown)
                messages = template.build_messages(who, item, shown, dimensions=list(who.fields))
                record = {"request_id": f"{p.profile_id}:{target}", "profile_id": p.profile_id,
                          "cell": p.cell, "language": language, "item": target, "messages": messages,
                          "options": {o.text: o.code for o in item.options}}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    pd.DataFrame(rows).to_csv(FROZEN / "profiles.csv", index=False)

    manifest = {
        "generated_by": "benchmark/build.py", "plan": "docs/ces2025_benchmark_framework.md",
        "seed": SEED, "source_survey": SOURCE, "profiles_per_cell": PROFILES_PER_CELL,
        "cells": list(CELLS), "target_set": SET, "targets": list(TARGET_VARS), "anchors": list(ANCHORS),
        "context": CONTEXT, "context_by_target": contexts,
        "template": {"condition": "C1", "k": k, "ses_dropout": "none", "year": YEAR},
        "n_profiles": len(profiles), "n_requests": n,
        "pool_sizes": profiles.groupby("cell").pool_size.first().to_dict(),
        "share_french": round(float(np.mean([r["language"] == "fr" for r in rows])), 4),
        "mean_anchors": round(float(np.mean([r["n_anchors"] for r in rows])), 3),
        "mean_context_shown": round(n_shown / n, 3),
        "sha256": {name: sha256(FROZEN / name) for name in ("requests.jsonl", "profiles.csv")},
    }
    (FROZEN / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_profiles", "n_requests", "share_french",
                                                "mean_anchors", "mean_context_shown")}, indent=1))


if __name__ == "__main__":
    main()
