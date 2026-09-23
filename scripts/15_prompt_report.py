"""Step 3.1 deliverable — measured token cost of the shared prompt template.

Renders real examples (real items, real respondents, real stratum
distributions with the §3.3 double masking) for C0 / C1 / C2 and tokenises
them with the Llama-3 tokenizer, then writes the distribution of tokens per
example to ``data/prompt_token_report.csv``.

Nothing here is an estimate: every number comes from an actually rendered
prompt. Run::

    .venv/bin/python scripts/15_prompt_report.py --items 300 --respondents 3

Tokenizer. Llama-3.1 / 3.2 / 3.3 all ship the *same* 128 256-token tokenizer;
the local ``unsloth/Meta-Llama-3.1-8B-Instruct`` snapshot is therefore exact
for ``Llama-3.3-70B-Instruct``, not an approximation. If no tokenizer file can
be found, the script falls back to a characters/token ratio measured on this
corpus and labels every row ``method=chars``.

No network: microdata is read from ``data/cache/`` only.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.corpus import similarity, strata  # noqa: E402
from article_silicon_sampling_quebec.corpus.blob import CACHE_DIR, read_survey  # noqa: E402
from article_silicon_sampling_quebec.corpus.ses import MISSING, SesCrosswalk  # noqa: E402
from article_silicon_sampling_quebec.corpus.strata import UNRESOLVED, cell_key  # noqa: E402
from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    ContextAnswer,
    ContextDistribution,
    ItemSpec,
    Persona,
    PromptTemplate,
    example_rng,
    nearest_context_items,
)

ITEMS_PATH = REPO / "data" / "items.parquet"
OUT_PATH = REPO / "data" / "prompt_token_report.csv"

#: Llama-3 chat wrapper, counted explicitly so the figures are training-ready.
_HEADER = "<|start_header_id|>{role}<|end_header_id|>\n\n"
_EOT = "<|eot_id|>"
_BOS = "<|begin_of_text|>"

TOKENIZER_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct",
    Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct",
    Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct",
]


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

class CharFallback:
    """Last resort: characters / ratio. Reported as such, never as a token count."""

    name = "chars@3.6"
    method = "chars"
    ratio = 3.6

    def count(self, text: str) -> int:
        return round(len(text) / self.ratio)


class HFTokenizer:
    method = "tokenizer"

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path))
        self.name = path.parents[2].name.replace("models--", "").replace("--", "/")

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def load_tokenizer():
    for root in TOKENIZER_CANDIDATES:
        for path in sorted(root.glob("snapshots/*/tokenizer.json")):
            try:
                return HFTokenizer(path)
            except Exception:  # pragma: no cover - corrupt snapshot
                continue
    return CharFallback()


def count_messages(tok, messages) -> dict[str, int]:
    """Token counts per role plus the chat-template overhead."""
    per_role = {}
    total = tok.count(_BOS)
    for msg in messages:
        body = tok.count(msg["content"])
        overhead = tok.count(_HEADER.format(role=msg["role"])) + tok.count(_EOT)
        per_role[msg["role"]] = body
        total += body + overhead
    return {"total": total, **{f"tok_{k}": v for k, v in per_role.items()}}


# --------------------------------------------------------------------------
# corpus plumbing (offline)
# --------------------------------------------------------------------------

def cached_surveys() -> set[str]:
    return {p.stem for p in CACHE_DIR.glob("*.parquet")}


class SurveyData:
    """Per-survey microdata, resolved cells and personas, loaded once."""

    def __init__(self, survey_id: str, crosswalk: SesCrosswalk):
        self.survey_id = survey_id
        self.crosswalk = crosswalk
        self.raw = read_survey(survey_id)
        profiles = strata.resolved_profiles(survey_id)
        dims = strata.load_definition().dimensions_for(survey_id)
        keys = []
        for row in profiles.iter_rows(named=True):
            levels = [row[d] for d in dims]
            keys.append(None if any(l in UNRESOLVED for l in levels)
                        else cell_key(levels))
        self.cell_by_rid = dict(zip(profiles["__respondent_id"], keys, strict=True))
        self.weight_by_rid = dict(
            zip(profiles["__respondent_id"], profiles["__weight"], strict=True)
        )
        self.rids = [str(r) for r in self.raw["__respondent_id"]]
        self._row_index = {rid: i for i, rid in enumerate(self.rids)}
        self._columns = set(self.raw.columns)
        self._codes: dict[str, list[str | None]] = {}
        self._cell_counts: dict[str, dict[str, dict[str, float]]] = {}
        self._cells = [self.cell_by_rid.get(r) for r in self.rids]
        self._weights = [self.weight_by_rid.get(r, 1.0) for r in self.rids]

    def row(self, rid: str) -> dict:
        return self.raw.row(self._row_index[rid], named=True)

    def persona(self, rid: str, survey_language: str) -> Persona:
        lang = "en" if survey_language == "en" else "fr"
        labels = self.crosswalk.profile_labels(self.survey_id, self.row(rid), lang=lang)
        labels = {k: v for k, v in labels.items() if v and v != MISSING}
        return Persona(fields=labels, survey_id=self.survey_id)

    def codes(self, variable: str) -> list[str | None]:
        """Whole column as normalised codes, materialised once per variable.

        A wide CES Parquet has thousands of columns, so ``DataFrame.row(named=True)``
        per lookup dominates the runtime; going column-wise removes it.
        """
        if variable not in self._codes:
            if variable not in self._columns:
                self._codes[variable] = [None] * len(self.rids)
            else:
                self._codes[variable] = [normalise_codes(
                    None if v is None else str(v).strip())
                    for v in self.raw[variable]]
        return self._codes[variable]

    def answer(self, rid: str, variable: str):
        idx = self._row_index.get(rid)
        return None if idx is None else self.codes(variable)[idx]

    def cell_counts(self, variable: str, cell: str) -> dict[str, float]:
        """Weighted counts of each code, per cell, computed once per variable."""
        table = self._cell_counts.get(variable)
        if table is None:
            table = {}
            for code, cell_id, weight in zip(self.codes(variable), self._cells,
                                             self._weights, strict=True):
                if code is None or cell_id is None:
                    continue
                bucket = table.setdefault(cell_id, {})
                bucket[code] = bucket.get(code, 0.0) + weight
            self._cell_counts[variable] = table
        return table.get(cell, {})


def normalise_codes(value: str | None) -> str | None:
    """Raw Parquet values are floats for numeric codes: ``3.0`` -> ``3``."""
    if value is None:
        return None
    try:
        f = float(value)
    except ValueError:
        return value
    return str(int(f)) if f.is_integer() else str(f)


def stratum_distribution(data: SurveyData, item: ItemSpec, cell: str,
                         exclude_rid: str) -> ContextDistribution | None:
    """C2 unit with the §3.3 double masking.

    The respondent is subtracted from his own cell's counts (one O(1)
    subtraction on the cached cell counts), and the target item never reaches
    this function — the caller only passes context items.
    """
    # Fold the raw codes onto the modalities the prompt actually offers: the
    # refusal merge must hit the observed distribution exactly as it hits the
    # option list, or the two are different partitions of the item.
    counts: dict[str, float] = {}
    for code, weight in data.cell_counts(item.variable, cell).items():
        canonical = item.canonical_code(code)
        counts[canonical] = counts.get(canonical, 0.0) + weight
    own = item.canonical_code(normalise_codes(data.answer(exclude_rid, item.variable)))
    if own is not None and own in counts:
        counts[own] -= data.weight_by_rid.get(exclude_rid, 1.0)
        if counts[own] <= 0:
            counts.pop(own)
    valid = {opt.code: counts.get(opt.code, 0.0) for opt in item.options}
    total = sum(valid.values())
    if total <= 0:
        return None
    shares = tuple((opt.label, valid[opt.code] / total)
                   for opt in item.options if valid[opt.code] > 0)
    return ContextDistribution(item=item, shares=shares, n=int(round(total)))


# --------------------------------------------------------------------------
# sampling and rendering
# --------------------------------------------------------------------------

def build_records(n_items: int, n_respondents: int, seed: int,
                  k: int, compact_context: bool, template_kwargs: dict):
    items = pl.read_parquet(ITEMS_PATH)
    items = items.filter(pl.col("survey_id").is_in(list(cached_surveys())))
    index = similarity.load_index()
    crosswalk = SesCrosswalk.load()

    by_key = {(r["survey_id"], r["variable"]): r for r in items.iter_rows(named=True)}
    rng = random.Random(seed)
    # Stratified by language: the report is read per language, so each side
    # needs enough items regardless of the 46/54 corpus split.
    sides = []
    for lang in ("fr", "en"):
        side = sorted(k for k in by_key if by_key[k]["language"] == lang)
        rng.shuffle(side)
        sides.append(side)
    keys = [k for pair in zip(*sides) for k in pair]
    keys += [k for side in sides for k in side[min(len(s) for s in sides):]]

    templates = {c: PromptTemplate(condition=c, k=k, **template_kwargs)
                 for c in ("C0", "C1", "C2")}
    surveys: dict[str, SurveyData] = {}
    records = []
    used = 0

    per_lang = {"fr": 0, "en": 0}
    cap = n_items // 2
    for key in keys:
        if used >= n_items:
            break
        row = by_key[key]
        if per_lang.get(row["language"], 0) >= cap:
            continue
        survey_id = row["survey_id"]
        target = ItemSpec.from_row(row)
        if not target.options or not target.text:
            continue
        if not compact_context:
            target = ItemSpec(**{**target.__dict__, "short_text": None})
        if survey_id not in surveys:
            try:
                surveys[survey_id] = SurveyData(survey_id, crosswalk)
            except Exception as exc:  # pragma: no cover
                print(f"  skip {survey_id}: {exc}")
                surveys[survey_id] = None
        data = surveys[survey_id]
        if data is None:
            continue

        neighbours = nearest_context_items(index, key, k=k, same_survey_only=True)
        ctx_items = []
        for nkey, cos in neighbours:
            nrow = by_key.get(nkey)
            if nrow is None:
                continue
            spec = ItemSpec.from_row(nrow)
            if not compact_context:
                spec = ItemSpec(**{**spec.__dict__, "short_text": None})
            ctx_items.append((spec, cos))

        # respondents who actually answered the target item and sit in a cell
        pool = [rid for rid in data.rids
                if data.cell_by_rid.get(rid)
                and target.option_label(normalise_codes(data.answer(rid, target.variable)))]
        if not pool:
            continue
        rng.shuffle(pool)
        picked = pool[:n_respondents]
        used += 1
        per_lang[row["language"]] = per_lang.get(row["language"], 0) + 1

        for rid in picked:
            cell = data.cell_by_rid[rid]
            answer = target.canonical_code(
                normalise_codes(data.answer(rid, target.variable))
            )
            persona = data.persona(rid, target.language)
            example_id = f"{survey_id}:{target.variable}:{rid}"
            # same dropout draw across conditions: the C0/C1/C2 delta is the
            # context block and nothing else.
            dims = templates["C0"].keep_dimensions(
                list(persona.fields), example_rng(seed, example_id)
            )

            c1 = []
            for spec, _ in ctx_items:
                code = spec.canonical_code(
                    normalise_codes(data.answer(rid, spec.variable))
                )
                label = spec.option_label(code) if code else None
                if label:
                    c1.append(ContextAnswer(item=spec, code=code, label=label))
            c2 = []
            for spec, _ in ctx_items:
                dist = stratum_distribution(data, spec, cell, rid)
                if dist:
                    c2.append(dist)

            for cond, ctx in (("C0", ()), ("C1", c1), ("C2", c2)):
                messages = templates[cond].build_example(
                    persona, target, answer, ctx, dimensions=dims
                )["messages"]
                records.append({
                    "condition": cond,
                    "language": target.language,
                    "survey_id": survey_id,
                    "variable": target.variable,
                    "n_context": len(ctx),
                    "n_ses_fields": len(dims),
                    "messages": messages,
                })
    return records


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

PCTS = [(0.10, "p10"), (0.25, "p25"), (0.50, "p50"),
        (0.75, "p75"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")]


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    i = min(int(round(q * (len(values) - 1))), len(values) - 1)
    return values[i]


def summarise(rows, group, tok, variant):
    values = [r["total"] for r in rows]
    out = {
        "variant": variant,
        "tokenizer": tok.name,
        "method": tok.method,
        "condition": group[0],
        "language": group[1],
        "n_examples": len(values),
        "mean": round(statistics.fmean(values), 1),
        "sd": round(statistics.pstdev(values), 1) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "mean_system": round(statistics.fmean([r["tok_system"] for r in rows]), 1),
        "mean_user": round(statistics.fmean([r["tok_user"] for r in rows]), 1),
        "mean_context_items": round(
            statistics.fmean([r["n_context"] for r in rows]), 2),
        "mean_ses_fields": round(
            statistics.fmean([r["n_ses_fields"] for r in rows]), 2),
    }
    for q, name in PCTS:
        out[name] = quantile(values, q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--respondents", type=int, default=3)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many rendered examples per condition")
    args = ap.parse_args()

    tok = load_tokenizer()
    print(f"tokenizer: {tok.name} ({tok.method})")
    if tok.method == "chars":
        print("  WARNING: no tokenizer file found — figures are characters/3.6, "
              "not tokens. Treat them as +/- 15 %.")

    variants = {
        # default: context items rendered by their short label, target verbatim
        "default": dict(compact=True, kwargs={}),
        # every context wording verbatim — the expensive end of the design
        "verbatim_context": dict(compact=False, kwargs={}),
        # C2 capped at the 4 heaviest modalities — the cheap end
        "c2_top4": dict(compact=True, kwargs={"max_options_in_context": 4}),
    }

    all_rows = []
    shown = {c: 0 for c in ("C0", "C1", "C2")}
    for variant, cfg in variants.items():
        print(f"building variant {variant} …")
        records = build_records(args.items, args.respondents, args.seed, args.k,
                                cfg["compact"], cfg["kwargs"])
        counted = []
        for rec in records:
            counted.append({**rec, **count_messages(tok, rec["messages"])})
        for cond in ("C0", "C1", "C2"):
            for lang in ("fr", "en", "all"):
                subset = [r for r in counted if r["condition"] == cond
                          and (lang == "all" or r["language"] == lang)]
                if subset:
                    all_rows.append(summarise(subset, (cond, lang), tok, variant))
        if variant == "default" and args.show:
            for rec in counted:
                cond = rec["condition"]
                if shown[cond] >= args.show or rec["n_context"] < args.k and cond != "C0":
                    continue
                shown[cond] += 1
                print("\n" + "=" * 78)
                print(f"### {cond} · {rec['survey_id']}/{rec['variable']} · "
                      f"{rec['language']} · {rec['total']} tokens")
                for msg in rec["messages"]:
                    print(f"--- {msg['role']} ---")
                    print(msg["content"])

    fields = list(all_rows[0])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {OUT_PATH} ({len(all_rows)} rows)")

    print("\nvariant           cond lang    n   mean     sd    p50    p90    max")
    for r in all_rows:
        print(f"{r['variant']:<17} {r['condition']:<4} {r['language']:<4} "
              f"{r['n_examples']:>4} {r['mean']:>6.0f} {r['sd']:>6.0f} "
              f"{r['p50']:>6} {r['p90']:>6} {r['max']:>6}")


if __name__ == "__main__":
    main()
