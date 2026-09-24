"""Step 3.2 — generate the C0 / C1 fine-tuning datasets.

Writes, under ``data/datasets/``:

    c0_train_8000.jsonl   c0_train_20000.jsonl   c0_validation.jsonl
    c1_train_8000.jsonl   c1_train_20000.jsonl   c1_validation.jsonl
    pairs.parquet         pairs.csv              composition.csv
    token_report.csv      manifest.json

``pairs.parquet`` / ``pairs.csv`` are line-aligned with the JSONL files: row
``i < validation_size`` is line ``i`` of ``cX_validation.jsonl``, the following
rows are the training lines in order (``row_index`` = line number in the
training file, the same in the 8 000 and 20 000 files since one is a prefix of
the other). The assistant turn of every line is ``answer_label``.

The JSONL files are in Azure AI Foundry chat format and uploadable as is. The
8 000-pair file is the first 8 000 lines of the 20 000-pair one, and C0 and C1
render the **same** pairs, so neither the duration contrast nor the condition
contrast is confounded by which respondents were drawn.

Run::

    .venv/bin/python scripts/16_generate_dataset.py --show 1

Everything reproducible from ``dataset.SEED``: rerunning overwrites the files
with byte-identical content. Token figures are measured with the exact Llama-3
tokenizer through ``scripts/15_prompt_report.py`` — never estimated.

No network: microdata is read from ``data/cache/`` only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec import dataset as ds  # noqa: E402
from article_silicon_sampling_quebec.corpus import similarity  # noqa: E402
from article_silicon_sampling_quebec.corpus.blob import CACHE_DIR  # noqa: E402
from article_silicon_sampling_quebec.corpus.ses import SesCrosswalk  # noqa: E402
from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    PromptTemplate,
    build_item_specs,
    nearest_context_items,
)
from article_silicon_sampling_quebec.split import (  # noqa: E402
    HELDOUT_ITEMS_PATH,
    MANIFEST_PATH,
    input_hashes,
    load_split,
    sha256_file,
    today,
)

ITEMS_PATH = REPO / "data" / "items.parquet"
SIMILARITY_PATH = REPO / "data" / "item_similarity.parquet"
EMBEDDINGS_PATH = REPO / "data" / "item_embeddings.parquet"

#: Azure serverless fine-tuning price used for the budget line (§0: the 70B
#: training price is unpublished, the Qwen tariff is the stand-in).
USD_PER_M_TOKENS = 5.50


def _prompt_report():
    """Import ``scripts/15_prompt_report.py`` for its exact tokenizer.

    Step 3.1 owns the tokenizer plumbing and is frozen; re-implementing the
    count here would be one more place for the two to drift apart.
    """
    spec = importlib.util.spec_from_file_location(
        "prompt_report", REPO / "scripts" / "15_prompt_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(cfg: ds.GeneratorConfig, verbose: bool = True):
    items = pl.read_parquet(ITEMS_PATH)
    cached = {p.stem for p in CACHE_DIR.glob("*.parquet")}
    split = load_split()
    index = similarity.load_index(SIMILARITY_PATH)
    embeddings = similarity.load_embeddings(EMBEDDINGS_PATH)
    crosswalk = SesCrosswalk.load()

    # Every item may be rendered as context (context-only items included:
    # reported behaviour is informative *about* a respondent) except the
    # held-out ones — Split.context_pool, applied here as the eligible set.
    context_pool = [
        (r["survey_id"], r["variable"])
        for r in items.iter_rows(named=True)
        if not split.is_test_item((r["survey_id"], r["variable"]))
        and r["survey_id"] in cached
    ]
    # Context lines are resolved over the whole table (collisions between
    # sibling items are invisible to a per-row build): build_item_specs.
    # CES items also exist in French, for the respondents who answered in
    # French: resolved the same way, over the French table only.
    french_rows = ds.french_item_rows(items)
    specs = ds.SpecSet(build_item_specs(items.iter_rows(named=True)),
                       {"fr": build_item_specs(french_rows)})
    if verbose:
        print(f"French CES versions: {len(french_rows)} items")

    targets = split.training_items(items).filter(pl.col("survey_id").is_in(list(cached)))
    targets = targets.filter(
        pl.col("survey_id").is_in(list(cached))
    ).sort("survey_id", "variable")
    keys = list(zip(targets["survey_id"], targets["variable"], strict=True))
    # Drop the handful of items whose wording or option list is empty: they
    # cannot be rendered as a question at all.
    keys = [k for k in keys if specs[k].options and specs[k].text]
    themes = ds.theme_labels(keys, embeddings, cfg.n_theme_clusters, cfg.seed)

    if verbose:
        print(f"targets: {len(keys)} · context pool: {len(context_pool)} · "
              f"test items: {len(split.items)}")

    panels: dict[str, ds.SurveyPanel] = {}
    for survey_id in sorted({k[0] for k in keys}):
        t0 = time.time()
        panels[survey_id] = ds.SurveyPanel(
            survey_id, crosswalk, split.respondents.get(survey_id, frozenset())
        )
        if verbose:
            panel = panels[survey_id]
            print(f"  {survey_id:<22} {len(panel.rids):>6} respondents, "
                  f"{panel.training_rows.size:>6} training "
                  f"({time.time() - t0:.1f}s)")

    # Allocation unit = item x prompt language. A CES item and its French
    # version are two units, so the language level balances the prompts the
    # model actually reads, not the catalogue's language tag.
    units = [(k, specs[k].language) for k in keys]
    units += [(k, "fr") for k in keys
              if specs[k].language != "fr" and specs.for_language(k, "fr") is not None]
    units.sort()
    capacity = {
        (k[0], k[1], lang): min(
            int(panels[k[0]].eligible_rows(specs[k], specs, language=lang).size),
            cfg.max_pairs_per_item)
        for k, lang in units
    }
    frame = pl.DataFrame(
        {
            "survey_id": [k[0] for k, _ in units],
            "variable": [k[1] for k, _ in units],
            "language": [lang for _, lang in units],
            "theme": [themes[k] for k, _ in units],
        }
    )
    quotas = ds.allocate_pairs(frame, capacity, cfg.total_pairs,
                               key_columns=("survey_id", "variable", "language"))
    if verbose:
        usable = sum(capacity.values())
        print(f"capacity {usable} pairs · allocated {sum(quotas.values())} "
              f"over {len(quotas)} item x language units")

    pairs = ds.sample_pairs(quotas, panels, specs, themes, cfg.seed)

    # Retrieval once per target item, not once per pair: it is a property of
    # the item. Test items are barred from the eligible set (double exclusion).
    neighbours = {
        key: tuple(nearest_context_items(index, key, k=cfg.k,
                                         same_survey_only=True,
                                         eligible=context_pool))
        for key in sorted({p.key for p in pairs})
    }
    pairs = [replace(p, context=neighbours[p.key]) for p in pairs]

    rng = random.Random(f"{cfg.seed}:shuffle")
    rng.shuffle(pairs)
    return pairs, panels, specs, split, themes


# --------------------------------------------------------------------------
# render + write
# --------------------------------------------------------------------------


def render_all(pairs, panels, specs, cfg):
    templates = {
        cond: PromptTemplate(condition=cond, k=cfg.k, **cfg.template_kwargs)
        for cond in cfg.conditions
    }
    out = {cond: [] for cond in cfg.conditions}
    for pair in pairs:
        for cond in cfg.conditions:
            out[cond].append(
                ds.render(pair, cond, panels[pair.survey_id], specs, templates,
                          cfg.seed)
            )
    return out


def metadata_frame(pairs, rendered, cfg) -> pl.DataFrame:
    cond = cfg.conditions[-1]
    n_val = cfg.validation_size
    rows = []
    for i, (pair, rec) in enumerate(zip(pairs, rendered[cond], strict=True)):
        rows.append({
            "split": "validation" if i < n_val else "train",
            "row_index": i if i < n_val else i - n_val,
            "survey_id": pair.survey_id,
            "variable": pair.variable,
            "respondent_id": pair.respondent_id,
            "language": pair.language,
            "theme": pair.theme,
            "answer_code": pair.answer_code,
            "answer_label": rec["example"]["messages"][-1]["content"],
            "n_context": rec["n_context"],
            "n_ses_fields": rec["n_ses_fields"],
            "context_keys": json.dumps(rec["context_keys"]),
            "context_used": json.dumps(rec["context_used"]),
            "context_cosines": json.dumps(rec["context_cosines"]),
            "max_context_cosine": max(rec["context_cosines"], default=None),
        })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

PCTS = [(0.10, "p10"), (0.50, "p50"), (0.90, "p90"), (0.99, "p99")]


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(int(round(q * (len(values) - 1))), len(values) - 1)]


def token_rows(counts, meta, condition, fname, n):
    subset = counts[:n]
    langs = meta["language"].to_list()[:n]
    rows = []
    for lang in ("all", "fr", "en"):
        vals = [c for c, l in zip(subset, langs, strict=True)
                if lang == "all" or l == lang]
        if not vals:
            continue
        total = sum(vals)
        row = {
            "file": fname,
            "condition": condition,
            "language": lang,
            "n_examples": len(vals),
            "mean_tokens": round(statistics.fmean(vals), 1),
            "sd_tokens": round(statistics.pstdev(vals), 1),
            "min_tokens": min(vals),
            "max_tokens": max(vals),
            "total_tokens": total,
            "usd_at_5.50_per_M": round(total / 1e6 * USD_PER_M_TOKENS, 2),
        }
        for q, name in PCTS:
            row[name] = quantile(vals, q)
        rows.append(row)
    return rows


def composition_rows(meta: pl.DataFrame, n: int, label: str):
    sub = meta.head(n)
    rows = []
    for axis in ("survey_id", "language", "theme"):
        # ties broken on the value: group_by order is not stable across runs
        counts = sub.group_by(axis).len().sort(["len", axis],
                                               descending=[True, False])
        for value, count in zip(counts[axis], counts["len"], strict=True):
            rows.append({
                "file": label, "axis": axis, "value": value,
                "n_pairs": int(count),
                "share": round(count / sub.height, 4),
            })
    rows.append({"file": label, "axis": "n_items", "value": "distinct_targets",
                 "n_pairs": sub.select("survey_id", "variable").unique().height,
                 "share": None})
    rows.append({"file": label, "axis": "n_respondents",
                 "value": "distinct_respondents",
                 "n_pairs": sub.select("survey_id", "respondent_id").unique().height,
                 "share": None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=ds.SEED)
    ap.add_argument("--k", type=int, default=ds.K_CONTEXT)
    ap.add_argument("--max-per-item", type=int, default=ds.MAX_PAIRS_PER_ITEM)
    ap.add_argument("--clusters", type=int, default=ds.N_THEME_CLUSTERS)
    ap.add_argument("--validation", type=int, default=ds.VALIDATION_SIZE)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(ds.TRAIN_SIZES))
    ap.add_argument("--out", type=Path, default=ds.DATASET_DIR)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many rendered examples per condition")
    args = ap.parse_args()

    cfg = ds.GeneratorConfig(
        seed=args.seed, k=args.k, train_sizes=tuple(sorted(args.sizes)),
        validation_size=args.validation, max_pairs_per_item=args.max_per_item,
        n_theme_clusters=args.clusters, out_dir=args.out,
    )
    t0 = time.time()
    pairs, panels, specs, split, themes = build(cfg)
    print(f"{len(pairs)} pairs drawn in {time.time() - t0:.1f}s")

    rendered = render_all(pairs, panels, specs, cfg)
    meta = metadata_frame(pairs, rendered, cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta.write_parquet(cfg.out_dir / "pairs.parquet")
    # CSV companion (AGENTS.md data-format policy); list columns stay JSON strings.
    meta.write_csv(cfg.out_dir / "pairs.csv")

    n_val = cfg.validation_size
    report = _prompt_report()
    tok = report.load_tokenizer()
    count_messages = report.count_messages
    print(f"tokenizer: {tok.name} ({tok.method})")

    written, token_report, composition = {}, [], []
    for cond in cfg.conditions:
        examples = [r["example"] for r in rendered[cond]]
        counts = [count_messages(tok, e["messages"])["total"] for e in examples]

        val_path = cfg.out_dir / f"{cond.lower()}_validation.jsonl"
        ds.write_jsonl(val_path, examples[:n_val])
        written[val_path.name] = n_val
        token_report += token_rows(counts[:n_val], meta.head(n_val), cond,
                                   val_path.name, n_val)

        for size in cfg.train_sizes:
            path = cfg.out_dir / f"{cond.lower()}_train_{size}.jsonl"
            ds.write_jsonl(path, examples[n_val:n_val + size])
            written[path.name] = min(size, len(examples) - n_val)
            token_report += token_rows(counts[n_val:], meta.slice(n_val), cond,
                                       path.name, size)
            if cond == cfg.conditions[0]:
                composition += composition_rows(meta.slice(n_val), size,
                                                f"train_{size}")
        if cond == cfg.conditions[0]:
            composition += composition_rows(meta, n_val, "validation")

    # -- audit, on the written files -------------------------------------
    print("\naudit (rules of §2.2/§4, checked on the generated JSONL)")
    audit_summary = {}
    for cond in cfg.conditions:
        for name in written:
            if not name.startswith(cond.lower()):
                continue
            path = cfg.out_dir / name
            examples = ds.read_jsonl(path)
            offset = 0 if "validation" in name else n_val
            slice_meta = meta.slice(offset, len(examples))
            problems = ds.audit_examples(examples, slice_meta, split, specs, cond)
            bad = {k: len(v) for k, v in problems.items()
                   if v and k not in ds.AUDIT_WARNINGS}
            warn = {k: len(v) for k, v in problems.items()
                    if v and k in ds.AUDIT_WARNINGS}
            audit_summary[name] = {"violations": bad, "warnings": warn}
            flag = "OK" if not bad else f"VIOLATIONS {bad}"
            if warn:
                flag += f"  (warnings: {warn})"
            print(f"  {name:<26} {len(examples):>6} lines  {flag}")
            for key, values in problems.items():
                if values and key not in ds.AUDIT_WARNINGS:
                    print(f"      {key}: {values[:3]}")

    # -- artefacts --------------------------------------------------------
    with (cfg.out_dir / "token_report.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(token_report[0]))
        writer.writeheader()
        writer.writerows(token_report)
    with (cfg.out_dir / "composition.csv").open("w", newline="",
                                                encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(composition[0]))
        writer.writeheader()
        writer.writerows(composition)

    ctx = meta["n_context"].to_list()
    manifest = {
        "version": "1.0",
        "generated_at": today(),
        "generated_by": "scripts/16_generate_dataset.py",
        "plan_reference": "docs/plan_article.md §2.2, §2.5, §3.2 (phase 3.2)",
        "seed": cfg.seed,
        "conditions": list(cfg.conditions),
        "k_context": cfg.k,
        "train_sizes": list(cfg.train_sizes),
        "validation_size": cfg.validation_size,
        "max_pairs_per_item": cfg.max_pairs_per_item,
        "n_theme_clusters": cfg.n_theme_clusters,
        "theme_proxy": (
            "items.parquet carries no themes (empty for all 1778 rows), so the "
            "theme balancing axis is a spherical k-means over "
            "data/item_embeddings.parquet, seeded with the generation seed."
        ),
        "sampling_rule": (
            "Equal-share water-filling down language -> survey -> theme -> item, "
            "capacity of an item = min(eligible training respondents, "
            f"{cfg.max_pairs_per_item}); respondents drawn without replacement "
            "inside each item; all pairs shuffled with random.Random(f'{seed}:"
            "shuffle'). The 8k file is the first 8k lines of the 20k file, the "
            "validation pairs are the first ones drawn and are disjoint from "
            "every training pair, and C0/C1 render the same pairs with the same "
            "SES-dropout draw."
        ),
        "files": written,
        "audit": audit_summary,
        "n_distinct_targets": meta.select("survey_id", "variable").unique().height,
        "n_distinct_respondents": meta.select("survey_id",
                                              "respondent_id").unique().height,
        "mean_context_items_c1": round(float(np.mean(ctx)), 3),
        "share_examples_with_full_k": round(float(np.mean(np.array(ctx) == cfg.k)), 4),
        "share_examples_without_context": round(float(np.mean(np.array(ctx) == 0)), 4),
        "input_hashes": input_hashes([
            ITEMS_PATH, SIMILARITY_PATH, EMBEDDINGS_PATH,
            HELDOUT_ITEMS_PATH, MANIFEST_PATH,
        ]),
        "output_hashes": {
            name: sha256_file(cfg.out_dir / name) for name in sorted(written)
        },
    }
    (cfg.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # -- console summary --------------------------------------------------
    print("\nfile                        cond  lang     n   mean    p50    p90"
          "     total tok     USD")
    for row in token_report:
        if row["language"] != "all":
            continue
        print(f"{row['file']:<27} {row['condition']:<5} {row['language']:<4} "
              f"{row['n_examples']:>5} {row['mean_tokens']:>6.0f} "
              f"{row['p50']:>6} {row['p90']:>6} {row['total_tokens']:>12,} "
              f"{row['usd_at_5.50_per_M']:>7.2f}")

    print("\ncontext items actually obtained in C1 (k=6 requested):")
    hist = Counter(ctx)
    for n in sorted(hist):
        print(f"  {n} item(s): {hist[n]:>6} pairs ({hist[n] / len(ctx):6.1%})")
    print(f"  mean = {np.mean(ctx):.2f}")

    if args.show:
        for cond in cfg.conditions:
            shown = 0
            for pair, rec in zip(pairs, rendered[cond], strict=True):
                if shown >= args.show:
                    break
                if cond == "C1" and rec["n_context"] < cfg.k:
                    continue
                shown += 1
                print("\n" + "=" * 78)
                print(f"### {cond} · {pair.survey_id}/{pair.variable} · "
                      f"{pair.language} · respondent {pair.respondent_id} · "
                      f"{rec['n_context']} context items")
                for msg in rec["example"]["messages"]:
                    print(f"--- {msg['role']} ---")
                    print(msg["content"])

    print(f"\nwrote {len(written)} JSONL files to {cfg.out_dir}")


if __name__ == "__main__":
    main()
