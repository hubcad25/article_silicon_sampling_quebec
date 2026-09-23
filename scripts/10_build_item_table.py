"""Build the item table — one row per item eligible as a target (§3.2, step 1.2).

Filters, in order:
  1. survey in perimeter (17 surveys)
  2. not sociodemo (those build personas, never targets)
  3. var_type in TARGET_VAR_TYPES (closed items: single / scale)
  4. usable response_options (>= 2 distinct codes) and not an open/verbatim item
  5. hand-listed exclusions (EXCLUDED_ITEMS) — items that pass every mechanical
     filter but are not valid targets
  6. non-opinion items (corpus.perimeter, three families): the hand-audited
     paradata / respondent-attribute / derived-recode list, items with more
     than MAX_OPTIONS *substantive* modalities (open-list questions coded after
     the fact — non-response modalities do not count, so a 0-10 thermometer is
     11 options and not 17), and items whose every catalogued modality is a
     non-response code
  7. unresolved Qualtrics/Stata piping in the stem (``has_unresolved_piping``):
     the catalogued text still shows ``[Field-justice_law]`` or
     ``${e://Field/premier}`` instead of what the respondent read, so both the
     prompt and the EMBEDDING are built on broken text
  8. non-empty in the microdata (n_valid_responses > 0)

Context-only items (reported behaviour: voted, donated, signed, volunteered,
watched the debate) are NOT dropped — they are flagged ``is_context_only``.
They stay retrievable as C1 context, but ``scripts/14_build_split.py`` bars
them from the target pool, at training as well as at test. See
``corpus.perimeter.CONTEXT_ONLY_ITEMS`` for why.

Refusal merge. Refusal / "prefer not to answer" / NA / skipped modalities are
instrument behaviour, not opinion; an item offering three of them asks the
model to guess *how* a respondent declined. They are collapsed into one
modality (``perimeter.merge_refusal_options``) and the ``code_map`` column
records raw code -> surviving code, so the observed distributions used for
validation can be folded onto exactly the modalities the prompt shows. "Don't
know" is kept as its own modality: it is a real state of opinion and a valid
target (plan §0).

Filter 6 lives here rather than downstream on purpose: an item that is not an
opinion target must also never be reachable as a C1/C2 context neighbour, and
the similarity index is built from this table.

``question_text`` is capped at 80 characters for the Stata-sourced surveys —
Stata's own limit on a variable label — so the sub-items of a matrix battery
share a byte-identical stem and the wording, which *is* the object of the
study, is cut mid-sentence. ``scripts/09_extract_ces_full_wording.py`` recovers
the real wording of the four CES from the questionnaires themselves (the 2021
Qualtrics ``.qsf``, the codebook / CATI PDFs for the others) into
``data/ces_full_wording.json``; this script applies it as an **override**
wherever it exists at high or medium confidence, and records where each
wording came from in ``question_text_source``. Nothing is invented: a variable
absent from that table keeps its Stata text and the ``display_label``
fallback, and is reported if it still looks cut.

``display_label`` does not exist in the normalized JSON: it is authored at
ingestion and lives only in the Azure AI Search index. We pull it from there
(same helper as scripts/12_build_similarity_index.py). It remains the fallback
for the items no source covers.

``n_valid_responses`` is counted from the microdata: respondents with a non-null
answer to that item. Outputs data/items.parquet (options serialized as JSON) and
a data/items.csv companion for inspection.

    .venv/bin/python scripts/10_build_item_table.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from article_silicon_sampling_quebec.corpus import blob, catalogue  # noqa: E402
from article_silicon_sampling_quebec.corpus.perimeter import (  # noqa: E402
    MAX_OPTIONS,
    TARGET_VAR_TYPES,
    all_options_non_response,
    count_substantive_options,
    has_unresolved_piping,
    is_context_only,
    is_in_perimeter,
    is_listed_non_opinion,
    merge_refusal_options,
)

MIN_OPTIONS = 2
#: Lengths at which a Stata-sourced ``question_text`` is a truncated label
#: (80 = the hard cap, 79 = the cap after a trailing space was stripped).
#: Mirrors ``prompts.TRUNCATION_LENGTHS``. Kept only to REPORT the items no
#: recovery covers — it is no longer the gate on applying one. It was, and it
#: silently left 119 CES items on a truncated stem, cps19_pos_energy ("…help
#: Canada's energy sector, includin", 78 characters) among them: Stata drops
#: the partial word, so a cut label is very often shorter than the cap.
TRUNCATION_LENGTHS = (79, 80)
#: Characters a finished question ends on. A short stem ending on anything
#: else is a label that was cut.
SENTENCE_ENDINGS = ("?", ".", "!", ":", ")", "»", '"', "…", "’", "'")
#: A stem at most this long that does not end like a sentence is truncated.
TRUNCATION_MAX_LEN = 80
#: Confidences of ``data/ces_full_wording.json`` trusted enough to override.
APPLIED_CONFIDENCES = frozenset({"high", "medium"})
FULL_WORDING = REPO_ROOT / "data" / "ces_full_wording.json"
OUT_PARQUET = REPO_ROOT / "data" / "items.parquet"
OUT_CSV = REPO_ROOT / "data" / "items.csv"

# Items that survive every mechanical filter but are not valid targets.
# eeq_2012 Q82D_M1..M5 are the five "mention" columns of ONE multi-response
# question (same stem, same 7 options, cosine 1.0 between them). They are typed
# `single` by mistake; none of the five carries the question's distribution
# (M2..M5 are conditional on having given 2+ answers, hence the collapsing n),
# and the pipeline targets a distribution over a single categorical variable, so
# they cannot be collapsed into one item here either. Dropped.
EXCLUDED_ITEMS: frozenset[tuple[str, str]] = frozenset(
    {("eeq_2012", f"Q82D_M{i}") for i in range(1, 6)}
)

SCHEMA = {
    "survey_id": pl.Utf8,
    "variable": pl.Utf8,
    "question_text": pl.Utf8,
    "question_text_source": pl.Utf8,
    "display_label": pl.Utf8,
    "var_type": pl.Utf8,
    "is_ordinal": pl.Boolean,
    "n_options": pl.Int32,
    "n_options_substantive": pl.Int32,
    "options": pl.Utf8,  # JSON list of {code, label}, refusals merged
    "code_map": pl.Utf8,  # JSON dict raw code -> surviving code
    "themes": pl.Utf8,  # JSON list
    "concepts": pl.Utf8,  # JSON list
    "year": pl.Int32,
    "language": pl.Utf8,
    "n_respondents_survey": pl.Int32,
    "n_valid_responses": pl.Int32,
    "is_context_only": pl.Boolean,
}


def full_wording() -> dict[tuple[str, str], dict]:
    """The recovered CES wording, keyed by (survey_id, variable).

    Built offline by scripts/09_extract_ces_full_wording.py. Missing file =
    no override, and the report says so.
    """
    if not FULL_WORDING.exists():
        return {}
    table = json.loads(FULL_WORDING.read_text(encoding="utf-8"))
    return {
        (survey_id, variable): entry
        for survey_id, entries in table.items()
        for variable, entry in entries.items()
        if entry.get("confidence") in APPLIED_CONFIDENCES
    }


def looks_truncated(text: str) -> bool:
    """True when a stem was cut by Stata's 80-character label limit."""
    stripped = (text or "").strip()
    return bool(stripped) and len(stripped) <= TRUNCATION_MAX_LEN and not (
        stripped.endswith(SENTENCE_ENDINGS)
    )


def resolve_wording(survey_id: str, item: dict, wording: dict) -> tuple[str, str]:
    """Final ``(question_text, question_text_source)`` of one catalogue item.

    The questionnaire is the source of truth for the wording, and the wording
    IS the object of study, so a high/medium-confidence recovery is applied
    whatever the length of the Stata label. Gating it on the label being
    exactly 79-80 characters was a bug: Stata drops the partial word, so a cut
    label is usually shorter than the cap, and 119 CES items — cps19_pos_energy
    ("…help Canada's energy sector, includin", 78 characters) among them — kept
    a stem cut mid-sentence while their full wording sat unused in
    data/ces_full_wording.json.
    """
    text = item["question_text"] or ""
    recovered = wording.get((survey_id, item["variable"]))
    if recovered:
        return recovered["question_text_full"], recovered["source"]
    if looks_truncated(text):
        return text, "stata_label_truncated"
    return text, "stata_label"


def usable_options(item: dict) -> bool:
    opts = item.get("response_options") or []
    codes = {o.get("code") for o in opts if o.get("code") is not None}
    return len(codes) >= MIN_OPTIONS


def is_open(item: dict) -> bool:
    return item.get("var_type") == "open" or item.get("text_kind") == "open"


def display_labels() -> dict[tuple[str, str], str]:
    """`display_label` per (survey_id, variable), from the AI Search catalogue.

    Reuses the helper in scripts/12_build_similarity_index.py rather than
    duplicating the query. Best-effort: an empty dict just means the column
    stays null, and the caller warns.
    """
    import importlib.util

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    path = REPO_ROOT / "scripts" / "12_build_similarity_index.py"
    spec = importlib.util.spec_from_file_location("_similarity_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalogue_docs = module._catalogue_enrichment() or {}
    return {
        key: value["display_label"]
        for key, value in catalogue_docs.items()
        if value.get("display_label")
    }


def count_valid(survey_id: str, variables: list[str]) -> dict[str, int]:
    """Non-null respondent count per variable, straight from the Parquet."""
    if not variables:
        return {}
    present = set(blob.survey_columns(survey_id))
    known = [v for v in variables if v in present]
    counts = {v: None for v in variables}
    if known:
        con = blob.connect([survey_id])
        select = ", ".join(f'COUNT("{v}")' for v in known)
        row = con.execute(f'SELECT {select} FROM "{survey_id}"').fetchone()
        con.close()
        counts.update(dict(zip(known, row)))
    return counts


def main() -> None:
    all_survey_ids = catalogue.catalogue_survey_ids()
    perimeter_ids = [s for s in all_survey_ids if is_in_perimeter(s)]
    with_microdata = set(blob.available_surveys())

    drops: dict[str, Counter] = defaultdict(Counter)
    kept_by_survey: dict[str, list[dict]] = {}
    orphan_surveys = [s for s in perimeter_ids if s not in with_microdata]
    extra_microdata = sorted(with_microdata - set(all_survey_ids))

    wording = full_wording()
    if not wording:
        print("WARNING: data/ces_full_wording.json missing — CES items keep "
              "their 80-character Stata labels\n")

    for survey_id in perimeter_ids:
        kept = []
        for item in catalogue.survey_items(survey_id):
            drops[survey_id]["total"] += 1
            if item["is_sociodemo"]:
                drops[survey_id]["sociodemo"] += 1
                continue
            if is_open(item):
                drops[survey_id]["open"] += 1
                continue
            if item["var_type"] not in TARGET_VAR_TYPES:
                drops[survey_id]["var_type"] += 1
                continue
            if not usable_options(item):
                drops[survey_id]["no_options"] += 1
                continue
            if (survey_id, item["variable"]) in EXCLUDED_ITEMS:
                drops[survey_id]["excluded_item"] += 1
                continue
            if is_listed_non_opinion(survey_id, item["variable"]):
                drops[survey_id]["non_opinion"] += 1
                continue
            if count_substantive_options(item["response_options"]) > MAX_OPTIONS:
                drops[survey_id]["open_list"] += 1
                continue
            if all_options_non_response(item["response_options"]):
                drops[survey_id]["all_nr"] += 1
                continue
            # Resolve the wording FIRST: the recovered .qsf / codebook text
            # is the one that will be embedded and shown, and it is the one
            # that carries the placeholders — the truncated Stata label often
            # stops before them.
            item["question_text_resolved"], item["question_text_source"] = (
                resolve_wording(survey_id, item, wording)
            )
            if has_unresolved_piping(item["question_text_resolved"]):
                drops[survey_id]["piped"] += 1
                continue
            kept.append(item)
        kept_by_survey[survey_id] = kept

    labels = display_labels()
    if not labels:
        print("WARNING: no display_label from the catalogue index — "
              "battery sub-items will be indistinguishable\n")

    rows = []
    for survey_id, items in kept_by_survey.items():
        valid = (
            count_valid(survey_id, [i["variable"] for i in items])
            if survey_id in with_microdata
            else {}
        )
        for item in items:
            text = item["question_text_resolved"]
            source = item["question_text_source"]
            raw_opts = item["response_options"] or []
            opts, code_map = merge_refusal_options(raw_opts, item.get("language"))
            rows.append(
                {
                    "survey_id": survey_id,
                    "variable": item["variable"],
                    "question_text": text,
                    "question_text_source": source,
                    "display_label": (
                        item["display_label"]
                        or labels.get((survey_id, item["variable"]))
                    ),
                    "var_type": item["var_type"],
                    "is_ordinal": item["is_ordinal"],
                    "n_options": len(opts),
                    "n_options_substantive": count_substantive_options(opts),
                    "options": json.dumps(opts, ensure_ascii=False),
                    "code_map": json.dumps(code_map, ensure_ascii=False),
                    "themes": json.dumps(item["themes"] or [], ensure_ascii=False),
                    "concepts": json.dumps(item["concepts"] or [], ensure_ascii=False),
                    "year": item["year"],
                    "language": item["language"],
                    "n_respondents_survey": item["n_respondents_survey"],
                    "n_valid_responses": valid.get(item["variable"]),
                    "is_context_only": is_context_only(survey_id, item["variable"]),
                }
            )

    df = pl.DataFrame(rows, schema=SCHEMA)

    # Items with no answer at all in the microdata carry no distribution to
    # predict. They pass every catalogue-side filter, so they can only be caught
    # here, once n_valid_responses is known.
    empty = df.filter(pl.col("n_valid_responses") == 0)
    for row in empty.iter_rows(named=True):
        drops[row["survey_id"]]["empty"] += 1
    df = df.filter(pl.col("n_valid_responses") != 0)
    for survey_id, items in kept_by_survey.items():
        drops[survey_id]["kept"] = len(items) - drops[survey_id]["empty"]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PARQUET)
    df.write_csv(OUT_CSV)

    report(df, drops, perimeter_ids, orphan_surveys, extra_microdata)


def report(df, drops, perimeter_ids, orphan_surveys, extra_microdata) -> None:
    print(f"surveys in perimeter : {len(perimeter_ids)}")
    print(f"catalogue items      : {sum(d['total'] for d in drops.values())}")
    print(f"items kept           : {df.height}")
    print()

    print("per-survey funnel (total -> kept)")
    header = (
        f"{'survey_id':<32}{'total':>7}{'sociodemo':>11}{'open':>7}{'var_type':>10}"
        f"{'no_opts':>9}{'excl':>6}{'nonopin':>9}{'openlist':>10}{'allNR':>7}"
        f"{'piped':>7}{'empty':>7}{'kept':>7}"
    )
    print(header)
    print("-" * len(header))
    for survey_id in perimeter_ids:
        d = drops[survey_id]
        print(
            f"{survey_id:<32}{d['total']:>7}{d['sociodemo']:>11}{d['open']:>7}"
            f"{d['var_type']:>10}{d['no_options']:>9}{d['excluded_item']:>6}"
            f"{d['non_opinion']:>9}{d['open_list']:>10}{d['all_nr']:>7}"
            f"{d['piped']:>7}{d['empty']:>7}{d['kept']:>7}"
        )
    agg = Counter()
    for d in drops.values():
        agg.update(d)
    print("-" * len(header))
    print(
        f"{'TOTAL':<32}{agg['total']:>7}{agg['sociodemo']:>11}{agg['open']:>7}"
        f"{agg['var_type']:>10}{agg['no_options']:>9}{agg['excluded_item']:>6}"
        f"{agg['non_opinion']:>9}{agg['open_list']:>10}{agg['all_nr']:>7}"
        f"{agg['piped']:>7}{agg['empty']:>7}{agg['kept']:>7}"
    )
    print()

    nv = df["n_valid_responses"].drop_nulls()
    print("n_valid_responses:")
    print(
        f"  min={nv.min()}  p25={nv.quantile(0.25):.0f}  median={nv.median():.0f}"
        f"  p75={nv.quantile(0.75):.0f}  max={nv.max()}"
    )
    for threshold in (50, 100, 500):
        print(f"  items with < {threshold}: {(nv < threshold).sum()}")
    print(f"  items with no microdata count: {df['n_valid_responses'].null_count()}")
    print()

    print("question_text provenance")
    print(
        df.group_by("question_text_source")
        .agg(pl.len().alias("items"))
        .sort("items", descending=True)
    )
    still_cut = df.filter(pl.col("question_text_source") == "stata_label_truncated")
    print(f"  items still on a truncated label (display_label fallback): "
          f"{still_cut.height}")
    for row in still_cut.iter_rows(named=True):
        print(f"    {row['survey_id']}/{row['variable']}: {row['question_text']}")
    print()

    missing_label = df["display_label"].null_count()
    print(f"display_label: {df.height - missing_label}/{df.height} resolved "
          f"({missing_label} missing)")
    print()

    n_ctx = int(df["is_context_only"].sum())
    print(f"context-only items (retrievable, never a target): {n_ctx}")
    print()

    print("items and respondents by language")
    print(
        df.group_by("language")
        .agg(pl.len().alias("items"), pl.col("survey_id").n_unique().alias("surveys"))
        .sort("items", descending=True)
    )
    print()

    if orphan_surveys:
        print(f"PERIMETER SURVEYS WITHOUT MICRODATA: {orphan_surveys}")
    else:
        print("all perimeter surveys have microdata in the Blob")
    if extra_microdata:
        print(f"MICRODATA WITHOUT CATALOGUE ENTRY: {extra_microdata}")
    print(f"\nwrote {OUT_PARQUET} and {OUT_CSV}")


if __name__ == "__main__":
    main()
