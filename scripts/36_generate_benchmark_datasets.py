"""Training datasets for the CES 2025 benchmark: two context arms, three sizes.

Plan: docs/ces2025_benchmark_framework.md. Every example is one real
respondent answering one item, with the persona and seven of their own answers
to other items of the same survey as context. The two arms render the **same
pairs** (same respondent, same target, same SES dropout); only the choice of
the context items differs:

    SEM  the 7 items of the survey nearest the target (embedding cosine)
    FIX  the survey's items for the seven anchor dimensions of the benchmark
         (data/anchor_map/anchor_map.csv): ideology, partisanship,
         redistribution, language and identity, social trust, Canada-US
         relations, equal rights

Both arms drop the target itself and any item at cosine >= 0.95 of it (the
anti-leak rule of prompts.nearest_context_items).

Corpus: eight federal surveys, 2019-2024, and nothing collected in 2025 —
CES 2019 online, CES 2021, Democracy Checkup 2019-2024 (ingested by
scripts/35_ingest_democracy_checkup.py). No held-out respondents or items: the
evaluation data is CES 2025, which is not in the corpus.

Sizes are nested: the 20k file is the first 20k lines of the 50k file, itself
the first 50k lines of the 100k file. The validation pairs are disjoint from
every training pair.

Writes to data/datasets_ces2025/:
    {sem,fix}_train_{20000,50000,100000}.jsonl   {sem,fix}_validation.jsonl
    pairs.csv   manifest.json

Run::

    .venv/bin/python scripts/36_generate_benchmark_datasets.py --show 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec import dataset as ds  # noqa: E402
from article_silicon_sampling_quebec.corpus.ses import SesCrosswalk  # noqa: E402
from article_silicon_sampling_quebec.corpus.similarity import (  # noqa: E402
    SimilarityIndex,
    build_embed_text,
    l2_normalize,
)
from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    CONTEXT_MAX_COSINE,
    PromptTemplate,
    build_item_specs,
    nearest_context_items,
    parse_options,
)
from article_silicon_sampling_quebec.split import Split, sha256_file, today  # noqa: E402

load_dotenv(REPO / ".env")

SURVEYS = ("ces_2019_online", "ces_2021", "dc_2019", "dc_2020", "dc_2021", "dc_2022",
           "dc_2023", "dc_2024")
SEED = 20260930
#: Seven, not nine: the secularism and moral-traditionalism anchors fail the
#: Foundry hate/fairness data check (scripts/38), so both arms get seven.
K = 7
TRAIN_SIZES = (20_000, 50_000, 100_000)
VALIDATION_SIZE = 500
#: DC waves hold 60-100 items for ~12 500 pairs each: the cap must let an
#: item be answered by a few hundred respondents.
MAX_PAIRS_PER_ITEM = 400
N_THEMES = 12
ARMS = ("sem", "fix")

OUT = REPO / "data" / "datasets_ces2025"
ANCHOR_MAP = REPO / "data" / "anchor_map" / "anchor_map.csv"
#: Items whose rendering scores this hate severity or more (Azure Content
#: Safety, 0-7) in either language are dropped as targets and as context:
#: Foundry rejects a training file with too many such lines (scripts/38).
HATE_SCREEN = REPO / "data" / "content_safety" / "item_hate.csv"
MAX_HATE_SEVERITY = 4
ITEMS = [REPO / "data" / "items.parquet", REPO / "data" / "items_extra.parquet"]
FRENCH = [ds.FRENCH_WORDING_PATH, REPO / "data" / "extra_french_wording.json"]
DIMENSIONS = ("ideology", "partisanship", "redistribution", "language_identity",
              "social_trust", "canada_us", "equal_rights")

#: Interview language of the DC waves; CES entries already in dataset.py.
ds.RESPONSE_LANGUAGE.update({
    "dc_2019": ("Q_Language", frozenset({"FR-CA"})),
    **{f"dc_{y}": ("UserLanguage", frozenset({"FR-CA"})) for y in range(2020, 2025)},
})


def load_items() -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in ITEMS]
    items = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("survey_id").is_in(SURVEYS))
    items = items.unique(["survey_id", "variable"], keep="last", maintain_order=True)
    screened = pl.read_csv(HATE_SCREEN).filter(
        pl.max_horizontal("target_severity", "context_severity") >= MAX_HATE_SEVERITY)
    items = items.join(screened.select("survey_id", "variable").unique(),
                       on=["survey_id", "variable"], how="anti")
    # Qualtrics appends " - Selected Choice" to items with a write-in option.
    return items.with_columns(pl.col("question_text").str.replace(r"\s*-\s*Selected Choice$", ""))


def item_vectors(items: pl.DataFrame) -> np.ndarray:
    """Production vectors where they exist, embeddings of the same recipe otherwise."""
    spec = importlib.util.spec_from_file_location("sim12", REPO / "scripts/12_build_similarity_index.py")
    sim12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim12)
    stored = pl.read_parquet(REPO / "data" / "item_embeddings.parquet")
    stored = {(s, v): e for s, v, e in zip(stored["survey_id"], stored["variable"], stored["embedding"])}
    keys = list(zip(items["survey_id"], items["variable"]))
    missing = [i for i, k in enumerate(keys) if k not in stored]
    texts = [build_embed_text(items["question_text"][i], items["display_label"][i],
                              [o.text for o in parse_options(items["options"][i])])
             for i in missing]
    fresh = sim12.embed_texts(texts) if texts else np.zeros((0, 3072))
    vectors = np.zeros((len(keys), 3072), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in stored:
            vectors[i] = stored[k]
    vectors[missing] = fresh
    return l2_normalize(vectors)


def same_survey_index(keys: list[tuple[str, str]], vectors: np.ndarray, k: int = 50) -> SimilarityIndex:
    rows = []
    surveys = np.array([s for s, _ in keys])
    for survey in SURVEYS:
        idx = np.flatnonzero(surveys == survey)
        sims = vectors[idx] @ vectors[idx].T
        for a, i in enumerate(idx):
            order = [j for j in np.argsort(-sims[a]) if idx[j] != i][:k]
            for rank, j in enumerate(order, 1):
                rows.append((keys[i][0], keys[i][1], rank, keys[idx[j]][0], keys[idx[j]][1],
                             float(sims[a, j]), True))
    return SimilarityIndex(pl.DataFrame(rows, schema=["survey_id", "variable", "rank",
                                                      "neighbor_survey_id", "neighbor_variable",
                                                      "cosine", "same_survey"], orient="row"))


def fixed_context(key, anchors, position, vectors) -> tuple:
    """The survey's anchor items, in dimension order, minus the target and its near-duplicates."""
    out = []
    for anchor in anchors.get(key[0], []):
        if anchor == key or anchor not in position:
            continue
        cosine = float(vectors[position[key]] @ vectors[position[anchor]])
        if cosine < CONTEXT_MAX_COSINE:
            out.append((anchor, cosine))
    return tuple(out)


def build():
    items = load_items()
    keys = list(zip(items["survey_id"], items["variable"]))
    position = {k: i for i, k in enumerate(keys)}
    vectors = item_vectors(items)
    index = same_survey_index(keys, vectors)

    french = []
    for path in FRENCH:
        french += ds.french_item_rows(items, path)
    specs = ds.SpecSet(build_item_specs(items.iter_rows(named=True)),
                       {"fr": build_item_specs(french)})
    split = Split(items=frozenset(), respondents={}, manifest={})

    targets = split.training_items(items).sort("survey_id", "variable")
    tkeys = [k for k in zip(targets["survey_id"], targets["variable"])
             if specs[k].options and specs[k].text]
    matrix = type("E", (), {"vector": lambda self, k: vectors[position[k]]})()
    themes = ds.theme_labels(tkeys, matrix, N_THEMES, SEED)

    crosswalk = SesCrosswalk.load()
    panels = {s: ds.SurveyPanel(s, crosswalk, frozenset()) for s in SURVEYS}

    capacity = {k: min(int(panels[k[0]].eligible_rows(specs[k], specs).size), MAX_PAIRS_PER_ITEM)
                for k in tkeys}
    frame = pl.DataFrame({"survey_id": [k[0] for k in tkeys], "variable": [k[1] for k in tkeys],
                          "theme": [themes[k] for k in tkeys]})
    total = max(TRAIN_SIZES) + VALIDATION_SIZE
    quotas = ds.allocate_pairs(frame, capacity, total, levels=("survey_id", "theme"))
    print(f"targets {len(tkeys)} · capacity {sum(capacity.values())} · "
          f"allocated {sum(quotas.values())} / {total}")
    pairs = ds.sample_pairs(quotas, panels, specs, themes, SEED)
    rng = random.Random(f"{SEED}:shuffle")
    rng.shuffle(pairs)

    anchor_map = pd.read_csv(ANCHOR_MAP)
    anchor_map = anchor_map[anchor_map.survey_id.isin(SURVEYS) & anchor_map.dimension.isin(DIMENSIONS)]
    anchor_map["order"] = anchor_map.dimension.map({d: i for i, d in enumerate(DIMENSIONS)})
    anchors = {s: [(s, v) for v in g.sort_values("order").variable]
               for s, g in anchor_map.groupby("survey_id")}
    semantic = {k: tuple(nearest_context_items(index, k, k=K, same_survey_only=True))
                for k in {p.key for p in pairs}}
    arms = {
        "sem": [replace(p, context=semantic[p.key]) for p in pairs],
        "fix": [replace(p, context=fixed_context(p.key, anchors, position, vectors)) for p in pairs],
    }
    return arms, panels, specs, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    arms, panels, specs, split = build()
    template = PromptTemplate(condition="C1", k=K)
    OUT.mkdir(parents=True, exist_ok=True)
    written, audits, meta_rows = {}, {}, []
    for arm, pairs in arms.items():
        rendered = [ds.render(p, "C1", panels[p.survey_id], specs, {"C1": template}, SEED)
                    for p in pairs]
        examples = [r["example"] for r in rendered]
        written[f"{arm}_validation.jsonl"] = ds.write_jsonl(OUT / f"{arm}_validation.jsonl",
                                                             examples[:VALIDATION_SIZE])
        train = examples[VALIDATION_SIZE:]
        for size in TRAIN_SIZES:
            name = f"{arm}_train_{size}.jsonl"
            written[name] = ds.write_jsonl(OUT / name, train[:size])
        meta = pl.DataFrame([{
            "survey_id": p.survey_id, "variable": p.variable, "respondent_id": p.respondent_id,
            "language": p.language, "answer_code": p.answer_code,
            "context_keys": json.dumps([list(k) for k, _ in p.context]),
            "context_used": json.dumps(r["context_used"]),
            "context_cosines": json.dumps([c for _, c in p.context]),
        } for p, r in zip(pairs, rendered, strict=True)])
        problems = ds.audit_examples(examples, meta, split, specs, "C1")
        audits[arm] = {k: len(v) for k, v in problems.items() if v}
        hard = {k: v for k, v in audits[arm].items() if k not in ds.AUDIT_WARNINGS}
        n_ctx = [r["n_context"] for r in rendered]
        print(f"\n[{arm}] audit violations: {hard or 'none'} · warnings: "
              f"{ {k: v for k, v in audits[arm].items() if k in ds.AUDIT_WARNINGS} }")
        print(f"[{arm}] context items: mean {np.mean(n_ctx):.2f} · "
              f"distribution {dict(sorted(Counter(n_ctx).items()))}")
        for i, (p, r) in enumerate(zip(pairs, rendered, strict=True)):
            if arm == "sem":
                meta_rows.append({"split": "validation" if i < VALIDATION_SIZE else "train",
                                  "row_index": i if i < VALIDATION_SIZE else i - VALIDATION_SIZE,
                                  "survey_id": p.survey_id, "variable": p.variable,
                                  "respondent_id": p.respondent_id, "language": p.language,
                                  "answer_code": p.answer_code,
                                  "answer_label": r["example"]["messages"][-1]["content"]})
            meta_rows_i = meta_rows[i]
            meta_rows_i[f"n_context_{arm}"] = r["n_context"]
            meta_rows_i[f"context_{arm}"] = json.dumps(r["context_used"])
        if args.show:
            for p, r in list(zip(pairs, rendered))[:args.show]:
                print(f"\n=== {arm} · {p.survey_id}/{p.variable} · {p.language}")
                for m in r["example"]["messages"]:
                    print(f"--- {m['role']}\n{m['content']}")

    pairs_frame = pl.DataFrame(meta_rows)
    pairs_frame.write_csv(OUT / "pairs.csv")
    composition = (pairs_frame.filter(pl.col("split") == "train")
                   .group_by("survey_id", "language").len().sort("survey_id", "language"))
    print("\n", composition)
    manifest = {
        "generated_at": today(), "generated_by": "scripts/36_generate_benchmark_datasets.py",
        "plan_reference": "docs/ces2025_benchmark_framework.md",
        "seed": SEED, "k_context": K, "arms": list(ARMS), "surveys": list(SURVEYS),
        "train_sizes": list(TRAIN_SIZES), "validation_size": VALIDATION_SIZE,
        "max_pairs_per_item": MAX_PAIRS_PER_ITEM, "context_max_cosine": CONTEXT_MAX_COSINE,
        "sampling_rule": "Equal-share water-filling down survey -> theme -> item; respondents "
                         "drawn without replacement within an item; pairs shuffled; sizes nested.",
        "files": written, "audit": audits,
        "n_distinct_targets": pairs_frame.select("survey_id", "variable").unique().height,
        "output_hashes": {n: sha256_file(OUT / n) for n in sorted(written)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(written)} files to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
