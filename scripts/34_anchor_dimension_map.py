"""Propose, for every training survey, which variable measures each anchor dimension.

The "varied" training context uses a fixed set of items covering the same nine
dimensions as the inference anchors of the CES 2025 benchmark
(docs/ces2025_benchmark_framework.md, §4). Each survey measures a dimension with
its own variable; this script ranks candidates by cosine similarity to the
Democracy Checkup 2024 anchor item and writes a table for manual validation.

Candidates: items of the current corpus (data/items.parquet, CES 2025 excluded)
and closed items of the Democracy Checkup 2019-2024 files in data/raw/dc_*.

Outputs (data/anchor_map/):
    candidates.csv   top 5 candidates per survey x dimension
    proposed.csv     one proposed variable per survey x dimension (to validate)
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from article_silicon_sampling_quebec.corpus.similarity import build_embed_text, l2_normalize  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
_spec = importlib.util.spec_from_file_location("sim12", REPO_ROOT / "scripts/12_build_similarity_index.py")
sim12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim12)

OUT = REPO_ROOT / "data/anchor_map"
DC_YEARS = range(2019, 2025)
TOP_K = 5
#: Below this cosine the best candidate is proposed as "none".
MIN_COSINE = 0.55

#: The nine anchor dimensions, keyed to their Democracy Checkup 2024 item.
DIMENSIONS = {
    "ideology": "dc24_lr_self_1",
    "partisanship": "dc24_party_id",
    "redistribution": "dc24_inequality_gap",
    "moral_traditionalism": "dc24_pos_family_val",
    "language_identity": "dc24_pos_bilingualis",
    "social_trust": "dc24_trust2",
    "canada_us": "dc24_ties_us",
    "secularism": "dc24_pos_relig_sym",
    "equal_rights": "dc24_pos_equal_1",
}

#: DC variables that describe the respondent rather than an opinion.
SES_PATTERN = re.compile(
    r"age|gender|province|educ|income|lang|weight|date|duration|second|responseid|^id$|postal|born|birth|"
    r"marital|child|kids|religion|employ|occup|union|sector|ethni|racial|indigen|sexuality|"
    r"citizen|region|urban|rural|household|_text$|attention|consent|finished|progress|status",
    re.I,
)
QUERY_SUFFIX = re.compile(r"\s*-\s*Selected Choice$")


def dc_items(year: int) -> pd.DataFrame:
    """Closed opinion items of one Democracy Checkup file."""
    path = REPO_ROOT / f"data/raw/dc_{year}/dc_{year}.dta"
    reader = pd.io.stata.StataReader(path)
    var_labels, value_labels = reader.variable_labels(), reader.value_labels()
    frame = pd.read_stata(path, convert_categoricals=False)
    rows = []
    for var in frame.columns:
        label = QUERY_SUFFIX.sub("", var_labels.get(var) or "").strip()
        if not label or SES_PATTERN.search(var):
            continue
        values = pd.to_numeric(frame[var], errors="coerce")
        valid = values[values >= 0]
        options = {k: v for k, v in value_labels.get(var, {}).items() if k >= 0}
        slider = not options and valid.size and valid.max() <= 100 and (valid % 1 == 0).all()
        if not (2 <= len(options) <= 12 or slider):
            continue
        rows.append({
            "survey_id": f"dc_{year}", "variable": var, "question_text": label,
            "options": " | ".join(str(v) for v in options.values()),
            "pct_valid": round(100 * valid.size / len(frame), 1),
        })
    return pd.DataFrame(rows)


def corpus_items() -> pd.DataFrame:
    """Current corpus items, CES 2025 excluded."""
    items = pl.read_parquet(REPO_ROOT / "data/items.parquet").filter(pl.col("survey_id") != "ces_2025")
    return pd.DataFrame({
        "survey_id": items["survey_id"], "variable": items["variable"],
        "question_text": items["question_text"],
        "options": [" | ".join(o.get("label", "") for o in json.loads(opts or "[]"))
                    for opts in items["options"].to_list()],
        "pct_valid": (100 * items["n_valid_responses"] / items["n_respondents_survey"]).round(1),
    })


def main() -> None:
    dc = pd.concat([dc_items(y) for y in DC_YEARS], ignore_index=True)
    candidates = pd.concat([corpus_items(), dc], ignore_index=True)

    # Corpus items reuse their production vectors; only DC items are embedded here.
    stored = pl.read_parquet(REPO_ROOT / "data/item_embeddings.parquet")
    stored = {(s, v): e for s, v, e in zip(stored["survey_id"], stored["variable"], stored["embedding"])}
    candidates = candidates[[k in stored or s.startswith("dc_")
                             for k, s in zip(zip(candidates.survey_id, candidates.variable),
                                             candidates.survey_id)]].reset_index(drop=True)
    is_dc = candidates.survey_id.str.startswith("dc_").to_numpy()
    dc_texts = [build_embed_text(t, option_labels=o.split(" | ") if o else None)
                for t, o in zip(candidates.question_text[is_dc], candidates.options[is_dc])]
    vectors = np.zeros((len(candidates), 3072), dtype=np.float32)
    vectors[is_dc] = sim12.embed_texts(dc_texts)
    vectors[~is_dc] = np.asarray([stored[k] for k in zip(candidates.survey_id[~is_dc],
                                                         candidates.variable[~is_dc])])
    vectors = l2_normalize(vectors)
    anchors = dc[dc.survey_id == "dc_2024"].set_index("variable")
    query_texts = [build_embed_text(anchors.loc[v, "question_text"],
                                    option_labels=anchors.loc[v, "options"].split(" | ") or None)
                   for v in DIMENSIONS.values()]
    queries = l2_normalize(sim12.embed_texts(query_texts))
    cosines = vectors @ queries.T

    ranked, proposed = [], []
    for survey, block in candidates.groupby("survey_id", sort=True):
        idx = block.index.to_numpy()
        for d, (dimension, anchor) in enumerate(DIMENSIONS.items()):
            order = idx[np.argsort(-cosines[idx, d])][:TOP_K]
            for rank, i in enumerate(order, 1):
                ranked.append({"survey_id": survey, "dimension": dimension, "rank": rank,
                               "variable": candidates.variable[i], "cosine": round(float(cosines[i, d]), 3),
                               "pct_valid": candidates.pct_valid[i],
                               "question_text": candidates.question_text[i][:160],
                               "options": candidates.options[i][:160]})
            best = order[0]
            exact = survey == "dc_2024" and candidates.variable[best] == anchor
            ok = exact or cosines[best, d] >= MIN_COSINE
            proposed.append({"survey_id": survey, "dimension": dimension,
                             "variable": candidates.variable[best] if ok else "",
                             "cosine": round(float(cosines[best, d]), 3),
                             "pct_valid": candidates.pct_valid[best],
                             "question_text": candidates.question_text[best][:160] if ok else "",
                             "validated": ""})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_csv(OUT / "candidates.csv", index=False)
    proposed = pd.DataFrame(proposed)
    proposed.to_csv(OUT / "proposed.csv", index=False)
    coverage = proposed.assign(found=proposed.variable.ne("")).pivot(
        index="survey_id", columns="dimension", values="found")
    print(coverage.sum(axis=1).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
