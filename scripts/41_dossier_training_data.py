"""C trained by example: dossier exercises from the 8 training surveys.

Every eligible item of the training surveys becomes one exercise, in the exact
prompt format of benchmark/dossier.py (so a model trained on it is scored with
``python -m benchmark.dossier run/score --deployment <name>`` unchanged):

    user       the item, its options, and a dossier of similar questions from
               the OTHER training surveys (never the item's own survey, never
               the item asked again: same stem or cosine >= 0.80), plus the
               fixed Democracy Checkup 2024 attitude block
    assistant  the item's real weighted distribution, nationally and in the 15
               cells (region x age at that survey), as the JSON dossier.py parses

Anti-leak: an item at cosine >= 0.80, same stem or same wording as any of the
18 benchmark items (SETS main + novel) is no exercise; nor is an item that close
to one of the three attitude items of the fixed block.

Small cells: a cell with n < MIN_N respondents is shrunk toward the national
distribution, (n * p_cell + (MIN_N - n) * p_nat) / MIN_N, so an empty cell gets
the national figure and a cell of 25 keeps most of its own shares. Every cell
stays in the reply, as the benchmark asks for all 15.

Augmentation: VARIANTS dossiers per item. Variant 0 is the benchmark policy
(top K neighbours, similarity order); the others draw 5-8 neighbours among the
top POOL, the last one in shuffled order. Validation (~10% of items, variant 0
only) holds out whole topic clusters (k-means on item embeddings).

    .venv/bin/python scripts/41_dossier_training_data.py            # build
    .venv/bin/python scripts/41_dossier_training_data.py --audit    # + RAI screen of 1,000 lines

Writes data/datasets_dossier/: dossier_train_<N>.jsonl (full and nested half),
dossier_validation.jsonl, dossier_rft_{train,validation}.jsonl (the same
exercises for reinforcement fine-tuning: messages without the reply, plus
target / codes / item for benchmark/dossier_grader.py), exercises.csv, manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from benchmark import CELLS, SETS
from benchmark import dossier as dz

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "datasets_dossier"
SEED = 20261017
VARIANTS = 3
POOL = 12
MIN_N = 30
MIN_NATIONAL_N = 500
N_CLUSTERS = 40
VALIDATION_SHARE = 0.10
#: Unverified Foundry limit for Llama 3.3 fine-tuning; every example must fit.
MAX_TOKENS = 8192
TOKENIZER = next((Path.home() / ".cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct")
                 .glob("snapshots/*/tokenizer.json"), None)
SURVEY_YEAR = {"ces_2019_online": 2019, "ces_2021": 2021, **{f"dc_{y}": y for y in range(2019, 2025)}}


def key_stem(variable: str) -> str:
    """Variable without its wave prefix: cps21_imm, pes19_x, dc22_trust2 -> imm, x, trust2."""
    return re.sub(r"^(cps|pes)\d\d_|^dc\d\d_", "", variable)


def norm(text) -> str:
    return " ".join(str(text or "").lower().split())


def setting(survey: str) -> str:
    year = SURVEY_YEAR[survey]
    when = (f"during the {year} federal election campaign" if survey.startswith("ces")
            else f"in {year}")
    return f"{dz.SURVEY_NAMES[survey]}, an online survey of Canadian adults (18+, ten provinces), {when}"


def bench_rows() -> pl.DataFrame:
    items = pl.read_parquet(REPO / "data" / "items.parquet")
    names = [v for s in SETS.values() for v in s]
    return items.filter((pl.col("survey_id") == "ces_2025") & pl.col("variable").is_in(names))


def shrink(shares: np.ndarray | None, n: int, national: np.ndarray) -> np.ndarray:
    if shares is None or n == 0:
        return national.copy()
    if n >= MIN_N:
        return shares
    return (n * shares + (MIN_N - n) * national) / MIN_N


def rounded(p: np.ndarray, codes: list[str]) -> dict[str, float]:
    r = np.round(p / p.sum(), 3)
    r[int(np.argmax(r))] += round(1 - r.sum(), 3)
    return {c: round(float(v), 3) for c, v in zip(codes, r)}


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        labels = np.argmax(x @ centres.T, axis=1)  # unit vectors: cosine
        for c in range(k):
            members = x[labels == c]
            if len(members):
                v = members.mean(0)
                centres[c] = v / np.linalg.norm(v)
    return labels


def build() -> dict:
    rng = random.Random(SEED)
    train = dz.s36.load_items()
    bench = bench_rows()
    both = pl.concat([train, bench], how="diagonal_relaxed")
    vectors = dz.s36.l2_normalize(dz.s36.item_vectors(both))
    tv, bv = vectors[:train.height], vectors[train.height:]
    keys = list(zip(train["survey_id"], train["variable"]))
    texts = [norm(t) for t in train["question_text"]]
    stems = [key_stem(v) for v in train["variable"]]
    attitude_idx = [keys.index(("dc_2024", v)) for _, v in dz.ATTITUDES]

    # Anti-leak against the 18 benchmark items and the attitude block.
    to_bench = tv @ bv.T
    bench_stems = [key_stem(v.replace("cps25_", "cps21_").replace("pes25_", "pes21_"))
                   for v in bench["variable"]]
    bench_texts = [norm(t) for t in bench["question_text"]]
    leak = {}
    for j in range(train.height):
        hits = [bench["variable"][b] for b in range(bench.height)
                if to_bench[j, b] >= dz.SAME_ITEM_COSINE or stems[j] == bench_stems[b]
                or texts[j] == bench_texts[b]]
        if hits:
            leak[j] = hits
    near_attitude = {j for j in range(train.height)
                     if max(float(tv[j] @ tv[a]) for a in attitude_idx) >= dz.SAME_ITEM_COSINE}

    crosswalk = dz.SesCrosswalk.load()
    past = {s: dz.Past(s, crosswalk) for s in dz.s36.SURVEYS}
    extra = pl.concat([pl.read_parquet(p) for p in dz.s36.ITEMS], how="diagonal_relaxed")
    profile = dz.attitudes(past["dc_2024"], extra)

    tables: dict[int, dict] = {}
    def table(j: int):
        if j not in tables:
            row = train.row(j, named=True)
            options = dz.parse_options(row["options"])
            tables[j] = (options, past[row["survey_id"]].table(row["variable"], [o.code for o in options]))
        return tables[j]

    def candidates(j: int) -> list[int]:
        """Neighbours of j by the dossier.py rules, minus j's own survey and j asked again."""
        sims = tv @ tv[j]
        out: list[int] = []
        for i in map(int, np.argsort(-sims)):
            if keys[i][0] == keys[j][0] or sims[i] >= dz.SAME_ITEM_COSINE:
                continue
            if stems[i] == stems[j] or texts[i] == texts[j]:
                continue
            if not (2 <= (train["n_options"][i] or 0) <= 11):
                continue
            if any(float(tv[i] @ tv[p]) >= dz.NEIGHBOUR_DUP_COSINE for p in out):
                continue
            if table(i)[1]["national"] is None:
                continue
            out.append(i)
            if len(out) == POOL:
                break
        return out

    exercises, dropped = [], {"leak": 0, "attitude": 0, "options": 0, "small": 0}
    for j in range(train.height):
        if j in leak:
            dropped["leak"] += 1
            continue
        if j in near_attitude:
            dropped["attitude"] += 1
            continue
        if not (2 <= (train["n_options"][j] or 0) <= 11):
            dropped["options"] += 1
            continue
        options, tab = table(j)
        n_total = sum(n for _, n in tab["cells"].values())
        if tab["national"] is None or n_total < MIN_NATIONAL_N or len(options) < 2:
            dropped["small"] += 1
            continue
        exercises.append(j)
    print(f"{len(exercises)} exercises · dropped {dropped}", flush=True)

    # Topic-cluster split.
    labels = kmeans(tv[exercises], N_CLUSTERS, SEED)
    order = list(range(N_CLUSTERS))
    rng.shuffle(order)
    held, count = set(), 0
    for c in order:
        if count >= VALIDATION_SHARE * len(exercises):
            break
        held.add(c)
        count += int((labels == c).sum())
    split = {j: "validation" if labels[e] in held else "train" for e, j in enumerate(exercises)}

    tok = None
    if TOKENIZER is not None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER))
    def n_tokens(messages) -> int:
        text = "".join(m["content"] for m in messages)
        return (len(tok.encode(text).ids) if tok else len(text) // 4) + 8 * len(messages)

    lines = {"train": [], "validation": []}
    rows = []
    for e, j in enumerate(exercises):
        row = train.row(j, named=True)
        options, tab = table(j)
        codes = [o.code for o in options]
        national = tab["national"]
        reply = {"national": rounded(national, codes),
                 "cells": {c: rounded(shrink(*tab["cells"][c], national), codes) for c in CELLS}}
        answer = json.dumps(reply, separators=(",", ":"))
        pool = candidates(j)
        n_variants = 1 if split[j] == "validation" else VARIANTS
        for v in range(n_variants):
            if v == 0:
                picked = pool[:dz.K]
            else:
                picked = sorted(rng.sample(pool, min(len(pool), rng.randint(5, 8))),
                                key=lambda i: -float(tv[i] @ tv[j]))
                if v == VARIANTS - 1:
                    rng.shuffle(picked)
            blocks = []
            for n, i in enumerate(picked, 1):
                nb = train.row(i, named=True)
                nb_options, nb_tab = table(i)
                blocks.append(dz.render_question(n, nb["survey_id"], float(tv[i] @ tv[j]),
                                                 nb["question_text"], nb["display_label"],
                                                 nb_options, nb_tab))
            messages = dz.prompt(row["question_text"], row["display_label"], options, blocks,
                                 profile, setting=setting(row["survey_id"]))
            messages.append({"role": "assistant", "content": answer})
            tokens = n_tokens(messages)
            while tokens > MAX_TOKENS and blocks:  # drop the least similar neighbour
                blocks.pop()
                messages = dz.prompt(row["question_text"], row["display_label"], options, blocks,
                                     profile, setting=setting(row["survey_id"]))
                messages.append({"role": "assistant", "content": answer})
                tokens = n_tokens(messages)
            rft = {"messages": messages[:-1], "target": reply["cells"], "codes": codes,
                   "item": f"{keys[j][0]}:{keys[j][1]}"}
            lines[split[j]].append((j, json.dumps({"messages": messages}, ensure_ascii=False),
                                    json.dumps(rft, ensure_ascii=False)))
            rows.append({"survey_id": keys[j][0], "variable": keys[j][1], "split": split[j],
                         "variant": v, "n_neighbours": len(blocks), "tokens": tokens,
                         "cluster": int(labels[e]),
                         "small_cells": sum(n < MIN_N for _, n in tab["cells"].values())})
        if e % 100 == 0:
            print(f"  {e}/{len(exercises)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rng.shuffle(lines["train"])
    full = [l for _, l, _ in lines["train"]]
    # Nested half: whole items, so the half keeps all variants of the items it has.
    items_order = list(dict.fromkeys(j for j, _, _ in lines["train"]))
    half_items = set(items_order[: len(items_order) // 2])
    half = [l for j, l, _ in lines["train"] if j in half_items]
    files = {f"dossier_train_{len(full)}.jsonl": full, f"dossier_train_{len(half)}.jsonl": half,
             "dossier_validation.jsonl": [l for _, l, _ in lines["validation"]],
             # Same exercises for reinforcement fine-tuning (o4-mini): prompt only,
             # the grader (benchmark/dossier_grader.py) reads target and codes.
             "dossier_rft_train.jsonl": [r for _, _, r in lines["train"]],
             "dossier_rft_validation.jsonl": [r for _, _, r in lines["validation"]]}
    for name, content in files.items():
        (OUT / name).write_text("\n".join(content) + "\n", encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exercises.csv", index=False)

    leak_by_bench: dict[str, int] = {}
    for hits in leak.values():
        for h in hits:
            leak_by_bench[h] = leak_by_bench.get(h, 0) + 1
    manifest = {
        "generated_at": date.today().isoformat(), "generated_by": "scripts/41_dossier_training_data.py",
        "seed": SEED, "prompt": "benchmark/dossier.py (prompt, render_question, attitudes)",
        "surveys": list(dz.s36.SURVEYS), "variants_train": VARIANTS, "pool": POOL, "k_variant0": dz.K,
        "min_cell_n_shrink": MIN_N, "min_national_n": MIN_NATIONAL_N,
        "n_exercises": len(exercises),
        "n_items": {s: int(frame[frame.variant == 0].split.eq(s).sum()) for s in ("train", "validation")},
        "files": {k: len(v) for k, v in files.items()},
        "dropped": dropped, "leak_by_benchmark_item": dict(sorted(leak_by_bench.items())),
        "validation_clusters": sorted(int(c) for c in held),
        "tokens": {"tokenizer": "Llama 3 (3.1-8B tokenizer.json)" if tok else "chars/4",
                   "median": int(frame.tokens.median()), "p95": int(frame.tokens.quantile(0.95)),
                   "max": int(frame.tokens.max()), "cap": MAX_TOKENS},
        "neighbours_mean": round(float(frame.n_neighbours.mean()), 2),
        "sha256": {k: hashlib.sha256((OUT / k).read_bytes()).hexdigest() for k in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_exercises", "n_items", "files", "dropped",
                                                "leak_by_benchmark_item", "tokens")}, indent=1))
    return manifest


def audit(n: int = 1000) -> dict:
    """Hate severity of n training lines, each split into Content Safety-sized chunks."""
    spec = importlib.util.spec_from_file_location("s38", REPO / "scripts/38_content_safety_screen.py")
    s38 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s38)
    path = max((p for p in OUT.glob("dossier_train_*.jsonl") if p.stem.rsplit("_", 1)[1].isdigit()),
               key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    lines = path.read_text(encoding="utf-8").splitlines()
    sample = random.Random(0).sample(lines, min(n, len(lines)))
    chunks = []
    for line in sample:
        text = "\n".join(m["content"] for m in json.loads(line)["messages"])
        parts, current = [], ""
        for para in text.split("\n\n"):
            if len(current) + len(para) > 8500:
                parts.append(current)
                current = ""
            current += para + "\n\n"
        parts.append(current)
        chunks.append(parts)
    scores = s38.score_all({p for parts in chunks for p in parts})
    values = [max(scores[p] for p in parts) for parts in chunks]
    result = {f">={cut}": sum(v >= cut for v in values) for cut in (2, 4, 6)} | {"n": len(values)}
    manifest = json.loads((OUT / "manifest.json").read_text())
    manifest["rai_audit"] = result
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("RAI audit (max hate severity per line):", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="only the RAI screen of built files")
    args = parser.parse_args()
    audit() if args.audit else build()
