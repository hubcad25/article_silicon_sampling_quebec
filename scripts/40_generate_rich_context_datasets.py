"""Fine-tuning datasets with a richer context per respondent (synthese 2026-10-14, §2B).

Same pipeline as scripts/36 (8 surveys, C1 template, water-filling sampling,
hate screen, anti-leak cosine 0.95), one new arm:

    RICH  a fixed core of up to 12 themes (the 7 FIX dimensions of
          data/anchor_map/anchor_map.csv + 5 new ones), then the 6 items of the
          survey nearest the target (embedding cosine) that are not core items:
          up to 18 answers of the respondent.

New themes are mapped per survey like scripts/34: the item of that survey
closest (cosine) to the Democracy Checkup 2024 key item, kept at cosine >=
MIN_COSINE unless overridden in CORE_OVERRIDES. A survey without the theme
skips it. Items about the same thing as a benchmark target (immigration level,
environment vs jobs, satisfaction with democracy, ...) are kept out of the core
so the core gives no benchmark item its own 2024 answer.

Writes data/datasets_ces2025_rich/:
    rich_validation.jsonl, rich_train_{50000,100000,150000}.jsonl (nested)
    pairs.csv, core_map.csv, manifest.json

    .venv/bin/python scripts/40_generate_rich_context_datasets.py [--show 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s36", REPO / "scripts/36_generate_benchmark_datasets.py")
s36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s36)
ds, Split, today, sha256_file = s36.ds, s36.Split, s36.today, s36.sha256_file

SEED = 20261016
N_NEAR = 6
#: Themes added to the 7 FIX dimensions, keyed to their Democracy Checkup 2024 item.
NEW_THEMES = {
    "financial_situation": "dc24_ladder_current_1",
    "government_satisfaction": "dc24_fed_gov_satisfa",
    "external_efficacy": "dc24_pos_govt_care",
    "market_role": "dc24_pos_govt_jobs",
    "religiosity": "dc24_religion_import",
}
THEMES = tuple(s36.DIMENSIONS) + tuple(NEW_THEMES)
K = len(THEMES) + N_NEAR
MIN_COSINE = 0.60
#: Manual corrections to the automatic mapping: (survey, theme) -> variable or None.
CORE_OVERRIDES: dict[tuple[str, str], str | None] = {
    # Best match is satisfaction with democracy: a benchmark target, not this theme.
    ("dc_2019", "government_satisfaction"): None,
    # Best match is the expected financial situation in 5 years, not the current one.
    ("dc_2022", "financial_situation"): None,
}
TRAIN_SIZES = (50_000, 100_000, 150_000)
VALIDATION_SIZE = 500
MAX_PAIRS_PER_ITEM = 600
OUT = REPO / "data" / "datasets_ces2025_rich"


def core_map(items: pl.DataFrame, position: dict, vectors: np.ndarray) -> pd.DataFrame:
    """One row per survey x theme: the FIX anchors, then the best match of each new theme."""
    fix = pd.read_csv(s36.ANCHOR_MAP)
    fix = fix[fix.survey_id.isin(s36.SURVEYS) & fix.dimension.isin(s36.DIMENSIONS)]
    rows = [{"survey_id": r.survey_id, "theme": r.dimension, "variable": r.variable,
             "cosine_to_dc24": None, "source": "anchor_map"}
            for r in fix.itertuples() if (r.survey_id, r.variable) in position]
    taken = {(r["survey_id"], r["variable"]) for r in rows}
    surveys = np.array(items["survey_id"].to_list())
    variables = items["variable"].to_list()
    for theme, key_var in NEW_THEMES.items():
        key = ("dc_2024", key_var)
        if key not in position:
            raise SystemExit(f"{key} missing from the screened items")
        for survey in s36.SURVEYS:
            if (survey, theme) in CORE_OVERRIDES:
                var, cos, source = CORE_OVERRIDES[(survey, theme)], None, "override"
            else:
                idx = [i for i in np.flatnonzero(surveys == survey)
                       if (survey, variables[i]) not in taken]
                sims = vectors[idx] @ vectors[position[key]]
                best = int(np.argmax(sims))
                var, cos, source = variables[idx[best]], float(sims[best]), "cosine"
                if cos < MIN_COSINE:
                    var = None
            if var is None:
                continue
            taken.add((survey, var))
            rows.append({"survey_id": survey, "theme": theme, "variable": var,
                         "cosine_to_dc24": None if cos is None else round(cos, 3), "source": source})
    out = pd.DataFrame(rows)
    out["order"] = out.theme.map({t: i for i, t in enumerate(THEMES)})
    return out.sort_values(["survey_id", "order"]).drop(columns="order").reset_index(drop=True)


def rich_context(key, core, nearest, position, vectors) -> tuple:
    """The survey's core items (theme order) then the nearest non-core items."""
    fixed = s36.fixed_context(key, core, position, vectors)
    core_keys = set(core.get(key[0], []))
    near = [(k, c) for k, c in nearest[key] if k not in core_keys][:N_NEAR]
    return fixed + tuple(near)


def build():
    items = s36.load_items()
    keys = list(zip(items["survey_id"], items["variable"]))
    position = {k: i for i, k in enumerate(keys)}
    vectors = s36.item_vectors(items)
    index = s36.same_survey_index(keys, vectors)

    cmap = core_map(items, position, vectors)
    core = {s: [(s, v) for v in g.variable] for s, g in cmap.groupby("survey_id", sort=False)}

    french = []
    for path in s36.FRENCH:
        french += ds.french_item_rows(items, path)
    specs = ds.SpecSet(s36.build_item_specs(items.iter_rows(named=True)),
                       {"fr": s36.build_item_specs(french)})
    split = Split(items=frozenset(), respondents={}, manifest={})
    targets = split.training_items(items).sort("survey_id", "variable")
    tkeys = [k for k in zip(targets["survey_id"], targets["variable"])
             if specs[k].options and specs[k].text]
    matrix = type("E", (), {"vector": lambda self, k: vectors[position[k]]})()
    themes = ds.theme_labels(tkeys, matrix, s36.N_THEMES, SEED)

    crosswalk = s36.SesCrosswalk.load()
    panels = {s: ds.SurveyPanel(s, crosswalk, frozenset()) for s in s36.SURVEYS}
    capacity = {k: min(int(panels[k[0]].eligible_rows(specs[k], specs).size), MAX_PAIRS_PER_ITEM)
                for k in tkeys}
    frame = pl.DataFrame({"survey_id": [k[0] for k in tkeys], "variable": [k[1] for k in tkeys],
                          "theme": [themes[k] for k in tkeys]})
    total = max(TRAIN_SIZES) + VALIDATION_SIZE
    quotas = ds.allocate_pairs(frame, capacity, total, levels=("survey_id", "theme"))
    print(f"targets {len(tkeys)} · capacity {sum(capacity.values())} · "
          f"allocated {sum(quotas.values())} / {total}")
    pairs = ds.sample_pairs(quotas, panels, specs, themes, SEED)
    random.Random(f"{SEED}:shuffle").shuffle(pairs)

    nearest = {k: s36.nearest_context_items(index, k, k=N_NEAR + len(THEMES), same_survey_only=True)
               for k in {p.key for p in pairs}}
    rich = [replace(p, context=rich_context(p.key, core, nearest, position, vectors)) for p in pairs]
    return rich, panels, specs, split, cmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    pairs, panels, specs, split, cmap = build()
    print(cmap.pivot(index="theme", columns="survey_id", values="variable")
          .reindex(list(THEMES)).fillna("—").to_string())
    template = s36.PromptTemplate(condition="C1", k=K)
    OUT.mkdir(parents=True, exist_ok=True)
    cmap.to_csv(OUT / "core_map.csv", index=False)

    rendered = [ds.render(p, "C1", panels[p.survey_id], specs, {"C1": template}, SEED) for p in pairs]
    examples = [r["example"] for r in rendered]
    written = {"rich_validation.jsonl": ds.write_jsonl(OUT / "rich_validation.jsonl",
                                                       examples[:VALIDATION_SIZE])}
    train = examples[VALIDATION_SIZE:]
    for size in TRAIN_SIZES:
        written[f"rich_train_{size}.jsonl"] = ds.write_jsonl(OUT / f"rich_train_{size}.jsonl",
                                                             train[:size])
    meta = pl.DataFrame([{
        "split": "validation" if i < VALIDATION_SIZE else "train",
        "row_index": i if i < VALIDATION_SIZE else i - VALIDATION_SIZE,
        "survey_id": p.survey_id, "variable": p.variable, "respondent_id": p.respondent_id,
        "language": p.language, "answer_code": p.answer_code,
        "answer_label": r["example"]["messages"][-1]["content"],
        "n_context": r["n_context"], "context_keys": json.dumps([list(k) for k, _ in p.context]),
        "context_used": json.dumps(r["context_used"]),
        "context_cosines": json.dumps([c for _, c in p.context]),
    } for i, (p, r) in enumerate(zip(pairs, rendered, strict=True))])
    problems = ds.audit_examples(examples, meta, split, specs, "C1")
    audit = {k: len(v) for k, v in problems.items() if v}
    hard = {k: v for k, v in audit.items() if k not in ds.AUDIT_WARNINGS}
    n_ctx = [r["n_context"] for r in rendered]
    print(f"\naudit violations: {hard or 'none'} · warnings: "
          f"{ {k: v for k, v in audit.items() if k in ds.AUDIT_WARNINGS} }")
    print(f"context items: mean {np.mean(n_ctx):.2f} · distribution {dict(sorted(Counter(n_ctx).items()))}")
    meta.write_csv(OUT / "pairs.csv")
    if args.show:
        for p, r in list(zip(pairs, rendered))[:args.show]:
            print(f"\n=== {p.survey_id}/{p.variable} · {p.language}")
            for m in r["example"]["messages"]:
                print(f"--- {m['role']}\n{m['content']}")

    manifest = {
        "generated_at": today(), "generated_by": "scripts/40_generate_rich_context_datasets.py",
        "plan_reference": "docs/synthese_2026-10-14.md §2B",
        "seed": SEED, "k_context": K, "n_core_themes": len(THEMES), "n_nearest": N_NEAR,
        "themes": list(THEMES), "new_theme_keys_dc24": NEW_THEMES,
        "dc_2024_core": cmap[cmap.survey_id == "dc_2024"].variable.tolist(),
        "arms": ["rich"], "surveys": list(s36.SURVEYS),
        "train_sizes": list(TRAIN_SIZES), "validation_size": VALIDATION_SIZE,
        "max_pairs_per_item": MAX_PAIRS_PER_ITEM, "context_max_cosine": s36.CONTEXT_MAX_COSINE,
        "max_hate_severity": s36.MAX_HATE_SEVERITY,
        "sampling_rule": "Equal-share water-filling down survey -> theme -> item; respondents "
                         "drawn without replacement within an item; pairs shuffled; sizes nested.",
        "files": written, "audit": audit,
        "context_mean": round(float(np.mean(n_ctx)), 2),
        "n_distinct_targets": meta.select("survey_id", "variable").unique().height,
        "output_hashes": {n: sha256_file(OUT / n) for n in sorted(written)},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(written)} files to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
