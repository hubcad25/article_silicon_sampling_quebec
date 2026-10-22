"""C, step 1: from a dossier to a distribution, zero-shot (docs/synthese_2026-10-14.md §2C).

Instead of 15,000 fake respondents, one request per target item: an analyst
model reads a dossier built only from the 8 training surveys and writes the
share of each answer in each of the 15 cells. No training, no CES 2025 answer.

The dossier of an item:
    - its wording and options (English, CES 2025);
    - the K most similar past questions (embedding cosine, vectors of scripts/36),
      the item itself in earlier waves excluded: same variable stem in CES 2019 /
      2021, or cosine >= 0.80. Main-set items are thus treated as new;
    - for each, the national weighted distribution and each cell's gap from it
      (cells = region x age at that survey);
    - three attitude summaries by cell (left-right, social trust,
      redistribution) from Democracy Checkup 2024.

Recipe note (docs/plan_prochaine_note_C.md): ``--breakdowns all`` writes one
request per breakdown (BREAKDOWNS: the 15 cells, the four margins, two
crossings), ``--news`` adds the item's dated news summaries (V5), and
BENCH_DOSSIER_VARIANT=<name> writes to results_dossier/<set>_<name>/.

    BENCH_SET=main  python -m benchmark.dossier build [--breakdowns all] [--news]
    BENCH_SET=main  python -m benchmark.dossier run [--deployment gpt-5.5] [--samples 3]
    BENCH_SET=main  python -m benchmark.dossier score

Writes benchmark/results_dossier/<set>/ (dossiers.jsonl, responses.jsonl,
tv_by_cell.csv, by_item.csv) and one row per set in results_dossier/summary.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl

from . import AGES, ASKED_ONLY, CELLS, REGION_OF, REGIONS, REPO, ROOT, SET, TARGET_VARS, TARGETS

sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec import dataset as ds  # noqa: E402
from article_silicon_sampling_quebec.corpus.ses import SesCrosswalk  # noqa: E402
from article_silicon_sampling_quebec.foundry import FoundryChat  # noqa: E402
from article_silicon_sampling_quebec.prompts import parse_options  # noqa: E402

_spec = importlib.util.spec_from_file_location("s36", REPO / "scripts" / "36_generate_benchmark_datasets.py")
s36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s36)

OUT = ROOT / "results_dossier"
#: A recipe variant writes its own folder: results_dossier/<set>[_<variant>]/.
VARIANT = os.environ.get("BENCH_DOSSIER_VARIANT", "")
HERE = OUT / (f"{SET}_{VARIANT}" if VARIANT else SET)
DEFAULT_DEPLOYMENT = "gpt-5.5"


def tagged(name: str, deployment: str) -> Path:
    """Per-deployment output file; the first model (gpt-5.5) keeps the plain names."""
    if deployment == DEFAULT_DEPLOYMENT:
        return HERE / name
    stem, ext = name.rsplit(".", 1)
    return HERE / f"{stem}_{deployment}.{ext}"
K = 8
#: A past question this close to the target is the target asked again. 0.90
#: let through the same wording under another name (pes21_envirojob 0.84,
#: dc_2019 immigration_level 0.86, pes21_dem_sat 0.89); other spending items
#: sit at 0.70-0.77 and stay.
SAME_ITEM_COSINE = 0.80
#: Two neighbours this close are the same question in two waves: keep one.
NEIGHBOUR_DUP_COSINE = 0.95
MAX_OPTIONS_FULL = 7
MIN_CELL_N = 20
SURVEY_NAMES = {"ces_2019_online": "CES 2019 (online)", "ces_2021": "CES 2021",
                **{f"dc_{y}": f"Democracy Checkup {y}" for y in range(2019, 2025)}}
#: Canonical age levels -> benchmark age group.
AGE_GROUP = {"18_24": "18-34", "25_34": "18-34", "35_44": "35-54", "45_54": "35-54",
             "55_64": "55+", "65_74": "55+", "75_plus": "55+"}
#: Subgroup dimensions beyond the 15 cells: canonical crosswalk level -> benchmark group.
GENDER_GROUP = {"man": "men", "woman": "women"}
EDUCATION_GROUP = {"no_diploma": "high school or less", "high_school": "high school or less",
                   "college_trades": "college or trades", "bachelor": "university",
                   "above_bachelor": "university"}
DIMENSIONS = {"region": REGIONS, "age": AGES, "gender": ("men", "women"),
              "education": ("high school or less", "college or trades", "university")}
DIMENSION_TEXT = {"region": "region (BC; Prairies = AB, SK, MB; ON; QC; Atlantic = NB, NL, NS, PE)",
                  "age": "age in years (18-34, 35-54, 55+)", "gender": "gender (men, women)",
                  "education": "education (high school or less, college or trades, university)"}
#: One request per breakdown. "region_age" is the 15 cells of the benchmark; the
#: others give the margins and crossings the full grid is raked from.
BREAKDOWNS = {"region_age": ("region", "age"), "region": ("region",), "age": ("age",),
              "gender": ("gender",), "education": ("education",),
              "age_gender": ("age", "gender"), "region_gender": ("region", "gender")}
NEWS_THEMES = REPO / "data" / "analysis" / "item_themes_v5.csv"
NEWS_SUMMARIES = REPO / "data" / "analysis" / "news_summaries_v5.csv"


def groups(breakdown: str) -> tuple[str, ...]:
    """Subgroup labels of a breakdown, "region|age" style; region_age gives CELLS."""
    labels = [""]
    for dim in BREAKDOWNS[breakdown]:
        labels = [f"{l}|{v}" if l else v for l in labels for v in DIMENSIONS[dim]]
    return tuple(labels)


ATTITUDES = (("Left-right self-placement (0 = left, 10 = right)", "dc24_lr_self_1"),
             ("Social trust", "dc24_trust2"),
             ("Redistribution (gap between rich and poor)", "dc24_inequality_gap"))
DK = re.compile(r"know|not sure|prefer not|refus|sais pas|préf", re.I)

ds.RESPONSE_LANGUAGE.update({
    "dc_2019": ("Q_Language", frozenset({"FR-CA"})),
    **{f"dc_{y}": ("UserLanguage", frozenset({"FR-CA"})) for y in range(2020, 2025)},
})


def stem(variable: str) -> str:
    return variable.split("_", 1)[1] if variable.startswith(("cps", "pes")) else variable


# -- past surveys ---------------------------------------------------------

class Past:
    """One training survey: per respondent, weight and subgroup levels."""

    def __init__(self, survey: str, crosswalk: SesCrosswalk):
        self.panel = ds.SurveyPanel(survey, crosswalk, frozenset())
        source = self.panel._ses.to_dicts() if self.panel._ses is not None else []
        maps = {"region": REGION_OF, "age": AGE_GROUP, "gender": GENDER_GROUP,
                "education": EDUCATION_GROUP}
        self.levels = {dim: np.array([m.get(crosswalk.apply(survey, dim, row).level or "")
                                      for row in source], dtype=object)
                       for dim, m in maps.items()}
        self._labels: dict[str, np.ndarray] = {}
        self.cells = self.labels("region_age")
        self.weights = self.panel.raw["__weight"].cast(pl.Float64).fill_null(0).to_numpy()

    def labels(self, breakdown: str) -> np.ndarray:
        """Each respondent's group in `breakdown` (None when a level is missing)."""
        if breakdown not in self._labels:
            dims = [self.levels[d] for d in BREAKDOWNS[breakdown]]
            self._labels[breakdown] = np.array(
                ["|".join(v) if all(v) else None for v in zip(*dims)], dtype=object)
        return self._labels[breakdown]

    def table(self, variable: str, codes: list[str], breakdown: str = "region_age") -> dict:
        """National shares and per-group shares (with n) over `codes`."""
        values = np.array(self.panel.codes(variable), dtype=object)
        index = {c: i for i, c in enumerate(codes)}
        idx = np.array([index.get(v, -1) for v in values])
        keep = idx >= 0
        def share(mask):
            w = np.bincount(idx[mask], self.weights[mask], minlength=len(codes))
            return w / w.sum() if w.sum() > 0 else None
        out = {"national": share(keep), "cells": {}}
        labels = self.labels(breakdown)
        for group in groups(breakdown):
            mask = keep & (labels == group)
            out["cells"][group] = (share(mask), int(mask.sum()))
        return out


def mean_index(shares: np.ndarray, substantive: list[int]) -> float:
    s = shares[substantive]
    return float((s * (np.arange(len(s)) + 1)).sum() / s.sum()) if s.sum() > 0 else float("nan")


def render_question(n: int, survey: str, cosine: float, text: str, topic: str | None, options,
                    table: dict) -> str:
    labels = [o.label for o in options]
    width = max(15, *map(len, table["cells"]))
    lines = [f"### Q{n}. {SURVEY_NAMES[survey]} (similarity {cosine:.2f})", f"\"{text}\"",
             *([f"Topic: {topic}"] if topic else []),
             "Options: " + "; ".join(f"[{i + 1}] {l}" for i, l in enumerate(labels)),
             "National %: " + " | ".join(f"[{i + 1}] {100 * p:.0f}" for i, p in enumerate(table["national"]))]
    if len(labels) <= MAX_OPTIONS_FULL:
        lines.append("Gap from national, percentage points, per option [1..%d]:" % len(labels))
        for cell in table["cells"]:
            shares, n_cell = table["cells"][cell]
            if shares is None or n_cell < MIN_CELL_N:
                lines.append(f"  {cell:<{width}} (n={n_cell}, too few)")
                continue
            gaps = " ".join(f"{100 * (p - q):+.0f}" for p, q in zip(shares, table["national"]))
            lines.append(f"  {cell:<{width}} (n={n_cell}): {gaps}")
    else:
        substantive = [i for i, l in enumerate(labels) if not DK.search(l)]
        national = mean_index(table["national"], substantive)
        lines.append(f"Mean position on [1..{len(substantive)}] (non-answers excluded): national "
                     f"{national:.2f}; gap from national by subgroup:")
        for cell in table["cells"]:
            shares, n_cell = table["cells"][cell]
            if shares is None or n_cell < MIN_CELL_N:
                lines.append(f"  {cell:<{width}} (n={n_cell}, too few)")
                continue
            lines.append(f"  {cell:<{width}} (n={n_cell}): {mean_index(shares, substantive) - national:+.2f}")
    return "\n".join(lines)


# -- build ----------------------------------------------------------------

def target_rows() -> pl.DataFrame:
    items = pl.read_parquet(REPO / "data" / "items.parquet")
    rows = items.filter((pl.col("survey_id") == "ces_2025") & pl.col("variable").is_in(TARGET_VARS))
    order = {v: i for i, v in enumerate(TARGET_VARS)}
    return rows.with_columns(pl.col("variable").replace_strict(order).alias("_o")).sort("_o").drop("_o")


def neighbours(train: pl.DataFrame, sims: np.ndarray, vectors: np.ndarray, target: str) -> list[int]:
    picked: list[int] = []
    for j in map(int, np.argsort(-sims)):
        survey, variable = train["survey_id"][j], train["variable"][j]
        if sims[j] >= SAME_ITEM_COSINE:
            continue
        if survey in ("ces_2019_online", "ces_2021") and stem(variable) == stem(target):
            continue
        if not (2 <= (train["n_options"][j] or 0) <= 11):
            continue
        if any(float(vectors[j] @ vectors[p]) >= NEIGHBOUR_DUP_COSINE for p in picked):
            continue
        picked.append(int(j))
        if len(picked) == K:
            break
    return picked


def attitudes(past: Past, items: pl.DataFrame, breakdown: str = "region_age") -> str:
    lines = ["## Attitude profile of each subgroup (Democracy Checkup 2024)",
             "Mean position, gap from the national mean (positive = higher on the scale):"]
    width = max(15, *map(len, groups(breakdown)))
    header = "  " + f"{'':<{width}}" + "".join(f"{name.split(' (')[0][:22]:>24}" for name, _ in ATTITUDES)
    blocks, nationals = [], []
    for name, variable in ATTITUDES:
        row = items.filter((pl.col("survey_id") == "dc_2024") & (pl.col("variable") == variable)).row(0, named=True)
        options = parse_options(row["options"])
        codes = [o.code for o in options]
        substantive = [i for i, o in enumerate(options) if not DK.search(o.label)]
        table = past.table(variable, codes, breakdown)
        national = mean_index(table["national"], substantive)
        nationals.append(f"{name}: national {national:.2f} on [1..{len(substantive)}] "
                         f"({options[substantive[0]].label} .. {options[substantive[-1]].label})")
        blocks.append({c: mean_index(s, substantive) - national if s is not None else float("nan")
                       for c, (s, _) in table["cells"].items()})
    lines += ["  " + n for n in nationals] + [header]
    for cell in groups(breakdown):
        lines.append("  " + f"{cell:<{width}}" + "".join(f"{b[cell]:>+24.2f}" for b in blocks))
    return "\n".join(lines)


def news_for(survey: str, variable: str) -> list[str]:
    """The dated news summaries of an item's themes (data/analysis, V5), at most two."""
    themes = pl.read_csv(NEWS_THEMES, infer_schema_length=0).filter(
        (pl.col("survey_id") == survey) & (pl.col("variable") == variable))
    if themes.is_empty():
        return []
    wanted = [t for t in themes.select("theme1", "theme2").row(0) if t and t != "0"]
    summaries = pl.read_csv(NEWS_SUMMARIES, infer_schema_length=0).filter(pl.col("survey_id") == survey)
    by_theme = dict(zip(summaries["theme"], summaries["summary"]))
    return [by_theme[t] for t in wanted if t in by_theme]


def build(breakdowns: tuple[str, ...] = ("region_age",), news: bool = False) -> None:
    train = s36.load_items()
    targets = target_rows()
    if targets.height != len(TARGET_VARS):
        raise SystemExit(f"{targets.height} of {len(TARGET_VARS)} targets found")
    both = pl.concat([train, targets], how="diagonal_relaxed")
    vectors = s36.l2_normalize(s36.item_vectors(both))
    tv_, cv = vectors[:train.height], vectors[train.height:]
    crosswalk = SesCrosswalk.load()
    surveys: dict[str, Past] = {}
    def past(survey: str) -> Past:
        if survey not in surveys:
            print(f"  loading {survey}", flush=True)
            surveys[survey] = Past(survey, crosswalk)
        return surveys[survey]
    extra = pl.concat([pl.read_parquet(p) for p in s36.ITEMS], how="diagonal_relaxed")
    profiles = {b: attitudes(past("dc_2024"), extra, b) for b in breakdowns}

    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "dossiers.jsonl").open("w", encoding="utf-8") as out:
        for i, (short, variable) in enumerate(zip(TARGETS, TARGET_VARS)):
            row = targets.row(i, named=True)
            options = parse_options(row["options"])
            picked = neighbours(train, cv[i] @ tv_.T, tv_, variable)
            context = news_for("ces_2025", variable) if news else None
            for breakdown in breakdowns:
                blocks, used = [], []
                for n, j in enumerate(picked, 1):
                    nb = train.row(j, named=True)
                    nb_options = parse_options(nb["options"])
                    table = past(nb["survey_id"]).table(nb["variable"], [o.code for o in nb_options],
                                                        breakdown)
                    if table["national"] is None:
                        continue
                    cos = float(cv[i] @ tv_[j])
                    blocks.append(render_question(n, nb["survey_id"], cos, nb["question_text"],
                                                  nb["display_label"], nb_options, table))
                    used.append({"survey_id": nb["survey_id"], "variable": nb["variable"],
                                 "cosine": round(cos, 3)})
                messages = prompt(row["question_text"], row["display_label"], options, blocks,
                                  profiles[breakdown], breakdown=breakdown, news=context,
                                  asked=ASKED_ONLY.get(variable, "").split(" (")[0] or None)
                out.write(json.dumps({"item": short, "variable": variable, "breakdown": breakdown,
                                      "codes": [o.code for o in options], "neighbours": used,
                                      "news": len(context or []), "messages": messages},
                                     ensure_ascii=False) + "\n")
                print(f"{short} [{breakdown}]: {len(used)} neighbours, max cosine "
                      f"{max((u['cosine'] for u in used), default=0):.2f}, {len(context or [])} news, "
                      f"~{sum(len(m['content']) for m in messages) // 4} tokens", flush=True)


SYSTEM = ("You are an expert analyst of Canadian public opinion. You estimate how Canadians answer "
          "survey questions, using evidence from earlier surveys.")


#: Where the new question was asked. Training exercises (scripts/41) name their own survey.
SETTING = ("an online survey of Canadian adults (18+, ten provinces) during the April 2025 "
           "federal election campaign")


def subgroups_text(breakdown: str) -> str:
    dims = BREAKDOWNS[breakdown]
    written = f', written "{"|".join(dims)}"' if len(dims) > 1 else ""
    return f"{len(groups(breakdown))} subgroups: " + " x ".join(DIMENSION_TEXT[d] for d in dims) + written + "."


def prompt(text: str, topic: str | None, options, blocks: list[str], profile: str,
           setting: str = SETTING, breakdown: str = "region_age",
           news: list[str] | None = None, asked: str | None = None) -> list[dict]:
    """The dossier request. region_age without news is the prompt of the first round."""
    codes = [o.code for o in options]
    labels = groups(breakdown)
    cells = ", ".join(labels)
    news_block = ("## News context (facts dated before the survey)\n" + "\n\n".join(news) + "\n\n"
                  if news else "")
    user = f"""A new question was asked in {setting}. No earlier survey asked it. Estimate how each subgroup answered.

## The new question
"{text}"{f"{chr(10)}Topic: {topic}" if topic else ""}
Options (code: label):
{chr(10).join(f"- {o.code}: {o.label}" for o in options)}{f"{chr(10)}Asked only of: {asked}. Shares are among them." if asked else ""}

{news_block}## Subgroups
{subgroups_text(breakdown)}

## Evidence: similar questions from earlier surveys (2019-2024)
For each: the national answer distribution, then how far each subgroup sat from the national figure. Sample sizes are unweighted respondents in the subgroup.

{chr(10).join(chr(10) + b for b in blocks)}

{profile}

## Your task
For each of the {len(labels)} subgroups, give the share of respondents choosing each option, including "don't know" or "prefer not to answer" options, which real respondents do choose. Shares are proportions between 0 and 1 and sum to 1 within each subgroup. Use the evidence: think about the likely national distribution first, then how each subgroup departs from it.

Return only a JSON object, no comment:
{{"national": {{{", ".join(f'"{c}": p' for c in codes)}}},
 "cells": {{"{labels[0]}": {{{", ".join(f'"{c}": p' for c in codes)}}}, ... one entry for each of: {cells}}}}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# -- run ------------------------------------------------------------------

def parse(reply: str, codes: list[str], labels: tuple[str, ...] = CELLS) -> dict[str, np.ndarray] | None:
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", reply, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    out = {}
    for cell in labels:
        probs = (data.get("cells") or {}).get(cell)
        if not isinstance(probs, dict):
            return None
        vec = np.array([max(float(probs.get(c, 0) or 0), 0.0) for c in codes])
        if vec.sum() <= 0:
            return None
        out[cell] = vec / vec.sum()
    return out


def run(deployment: str, samples: int) -> None:
    chat = FoundryChat(deployment=deployment, timeout=600, api_version="2025-04-01-preview")
    dossiers = [json.loads(l) for l in (HERE / "dossiers.jsonl").read_text(encoding="utf-8").splitlines()]
    path = tagged("responses.jsonl", deployment)
    done = set()
    if path.exists():
        done = {(r["item"], r.get("breakdown", "region_age"), r["sample"])
                for r in map(json.loads, path.read_text().splitlines()) if r.get("valid")}
    jobs = [(d, s) for d in dossiers for s in range(samples)
            if (d["item"], d.get("breakdown", "region_age"), s) not in done]
    print(f"{deployment}: {len(jobs)} requests", flush=True)

    def one(job):
        d, s = job
        if not deployment.startswith("bench-"):  # OpenAI models (gpt-*, o4-mini); bench-* = our Llamas
            body = {"messages": d["messages"], "max_completion_tokens": 32000,
                    "response_format": {"type": "json_object"}}
        else:  # open-weight chat models: plain max_tokens, JSON asked in the prompt
            body = {"messages": d["messages"], "max_tokens": 4096, "temperature": 1.0}
        reply, usage, error = "", {}, ""
        for attempt in range(6):
            try:
                data = chat._post(body)
                choices = data.get("choices") or []
                reply = (choices[0].get("message", {}).get("content") or "") if choices else ""
                usage = data.get("usage", {})
                error = "" if choices else "no choice (filtered?)"
                if reply:
                    break
            except Exception as e:  # noqa: BLE001 - logged and retried
                error = str(e)[:300]
                body_text = getattr(e, "read", lambda: b"")()
                if body_text:
                    error += " " + body_text[:300].decode(errors="replace")
            time.sleep(min(2 ** attempt * 5, 120))
        breakdown = d.get("breakdown", "region_age")
        valid = parse(reply, d["codes"], groups(breakdown)) is not None
        return {"item": d["item"], "breakdown": breakdown, "sample": s, "deployment": deployment,
                "valid": valid,
                "raw": reply, "error": error,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "prompt_tokens": usage.get("prompt_tokens")}

    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(6) as pool:
        for record in pool.map(one, jobs):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {record['item']} [{record['breakdown']}] #{record['sample']}: valid={record['valid']} {record['error'][:120]}",
                  flush=True)


# -- score ----------------------------------------------------------------

def score(deployment: str = DEFAULT_DEPLOYMENT) -> None:
    from . import score as sc

    # The 15 cells only; the other breakdowns are scored by the rich-grid measures.
    dossiers = {d["item"]: d for d in map(json.loads, (HERE / "dossiers.jsonl").read_text().splitlines())
                if d.get("breakdown", "region_age") == "region_age"}
    responses = tagged("responses.jsonl", deployment)
    records = [r for r in map(json.loads, responses.read_text().splitlines())
               if r["valid"] and r.get("breakdown", "region_age") == "region_age"]
    opts = sc.options()
    frame = sc.human_data()
    human, _ = sc.human_panel(frame, opts)
    floor = sc.human_floor(human, opts, np.random.default_rng(sc.SEED))
    cell_w = frame.groupby("cell").w.sum().reindex(list(CELLS)).to_numpy()
    cell_w = cell_w / cell_w.sum()

    rows, items = [], []
    for j, t in enumerate(TARGETS):
        codes = dossiers[t]["codes"]
        if codes != opts[t]:
            raise SystemExit(f"{t}: dossier codes {codes} != benchmark codes {opts[t]}")
        draws = [parse(r["raw"], codes) for r in records if r["item"] == t]
        if not draws:
            continue
        model = {c: np.mean([d[c] for d in draws], axis=0) for c in CELLS}
        k = len(codes)
        h = {c: human.dist(c, j, k) for c in CELLS}
        h_nat = sum(w * h[c] for w, c in zip(cell_w, CELLS))
        m_nat = sum(w * model[c] for w, c in zip(cell_w, CELLS))
        tvs = []
        for g, c in enumerate(CELLS):
            value = sc.tv(h[c], model[c])
            tvs.append(value)
            rows.append({"item": t, "cell": c, "tv": round(value, 4), "floor": round(floor[g, j], 4)})
        items.append({"item": t, "n_samples": len(draws), "mean_tv": round(float(np.mean(tvs)), 4),
                      "national_tv": round(sc.tv(h_nat, m_nat), 4),
                      "floor": round(float(floor[:, j].mean()), 4)})
    pd.DataFrame(rows).to_csv(tagged("tv_by_cell.csv", deployment), index=False)
    by_item = pd.DataFrame(items)
    by_item.to_csv(tagged("by_item.csv", deployment), index=False)
    deployment = records[0]["deployment"] if records else ""
    summary = {"set": SET, "model": f"dossier-zeroshot-{deployment}", "n_items": len(items),
               "mean_tv": round(float(by_item.mean_tv.mean()), 4),
               "mean_national_tv": round(float(by_item.national_tv.mean()), 4),
               "floor": round(float(by_item.floor.mean()), 4),
               "invalid_rate": round(1 - len(records) / max(1, len(responses.read_text().splitlines())), 4)}
    path = OUT / "summary.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[(old["set"] != SET) | (old["model"] != summary["model"])]
    pd.concat([old, pd.DataFrame([summary])]).to_csv(path, index=False)
    print(by_item.to_string(index=False))
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--breakdowns", default="region_age",
                   help="comma-separated names of BREAKDOWNS, or 'all'")
    p.add_argument("--news", action="store_true", help="add the V5 news summaries")
    p = sub.add_parser("run")
    p.add_argument("--deployment", default="gpt-5.5")
    p.add_argument("--samples", type=int, default=3)
    p = sub.add_parser("score")
    p.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    args = parser.parse_args()
    if args.cmd == "build":
        names = tuple(BREAKDOWNS) if args.breakdowns == "all" else tuple(args.breakdowns.split(","))
        build(names, args.news)
    elif args.cmd == "run":
        run(args.deployment, args.samples)
    else:
        score(args.deployment)


if __name__ == "__main__":
    main()
