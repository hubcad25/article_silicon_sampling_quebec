"""Build and freeze the pre-registered split — step 2.1/2.2 of the plan.

Produces, under ``data/split/``:

    heldout_items.json        the test items, their distance, their labels,
                              and the >= 0.97 pairs to review by hand
    heldout_respondents.parquet  the held-out respondents and their cell
    split_manifest.json       seed, date, parameters, input hashes, counts
    split_diagnostics.csv     one row per test item, plus the candidate pool

Deterministic: same inputs + same seed -> byte-identical items and manifest
(the ``generated_at`` field aside). No network — microdata is read from the
local cache under ``data/cache/``.

    .venv/bin/python scripts/14_build_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from article_silicon_sampling_quebec import split as sp  # noqa: E402
from article_silicon_sampling_quebec.corpus import blob, strata  # noqa: E402
from article_silicon_sampling_quebec.corpus.similarity import (  # noqa: E402
    load_embeddings,
    load_index,
)
from article_silicon_sampling_quebec.prompts import (  # noqa: E402
    nearest_context_items,
)

CONTEXT_K = 6

CONTEXT_LEAK_RULE = (
    "Anti-leak rule on C1/C2 context retrieval, enforced in "
    "prompts.nearest_context_items: no item at cosine >= "
    f"{sp.CONTEXT_MAX_COSINE} of the target may be injected as a context item, in any "
    "condition, at training or at inference. Without it the training twin of a "
    "test item is a legal context for the same respondent and the model is "
    "handed the answer (ces_2021 cps21_votechoice / cps21_v_advance sit at "
    "0.967, same survey, same respondents). It is a property of the retrieval "
    "policy, not of the split: it applies to every target, test or training."
)

ITEMS_PATH = REPO_ROOT / "data" / "items.parquet"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "item_embeddings.parquet"
SIMILARITY_PATH = REPO_ROOT / "data" / "item_similarity.parquet"
STRATA_PATH = REPO_ROOT / "data" / "strata_definition.json"


def label_of(row: dict) -> str:
    """Human-readable label of an item: display_label wins, question_text backs it."""
    label = (row.get("display_label") or "").strip()
    text = (row.get("question_text") or "").strip()
    if label and text and label.lower() not in text.lower():
        return f"{label} — {text}"
    return label or text


# --------------------------------------------------------------------------
# 1. respondent split
# --------------------------------------------------------------------------


def build_respondent_split(rng: np.random.Generator) -> tuple[pl.DataFrame, list[dict]]:
    definition = strata.load_definition()
    held_parts: list[pl.DataFrame] = []
    rows: list[dict] = []
    for survey_id in sorted(strata.perimeter_surveys()):
        profiles = strata.resolved_profiles(survey_id)
        dims = definition.dimensions_for(survey_id)
        held = sp.split_survey_respondents(
            profiles, dims, strata.UNRESOLVED, rng
        )
        held_parts.append(held)
        rows.append(
            {
                "survey_id": survey_id,
                "dimensions": "+".join(dims),
                "n_respondents": profiles.height,
                "holdout_fraction": sp.holdout_fraction(profiles.height),
                "n_heldout": held.height,
                "n_training": profiles.height - held.height,
                "n_heldout_cells": held["cell"].n_unique(),
            }
        )
    return pl.concat(held_parts, how="vertical"), rows


# --------------------------------------------------------------------------
# 2. validability of every item against the held-out respondents
# --------------------------------------------------------------------------


def build_validability(items: pl.DataFrame, heldout: pl.DataFrame) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for survey_id in sorted(items["survey_id"].unique().to_list()):
        variables = items.filter(pl.col("survey_id") == survey_id)[
            "variable"
        ].to_list()
        present = [v for v in variables if v in set(blob.survey_columns(survey_id))]
        micro = blob.read_survey(
            survey_id, columns=["__respondent_id", *present]
        ).with_columns(pl.col("__respondent_id").cast(pl.Utf8))
        held = heldout.filter(pl.col("__survey_id") == survey_id)
        counts = sp.estimable_cells(held, micro, variables)
        parts.append(counts.with_columns(survey_id=pl.lit(survey_id)))
    return pl.concat(parts, how="vertical").select(
        "survey_id",
        "variable",
        pl.col(f"n_cells_ge{sp.CELL_N_CONTRAST}").alias("n_cells_ge30"),
        pl.col(f"n_cells_ge{sp.CELL_N_ABSOLUTE}").alias("n_cells_ge100"),
    )


# --------------------------------------------------------------------------
# 2b. cost of the anti-leak rule on context retrieval
# --------------------------------------------------------------------------


def context_leak_effect(items: pl.DataFrame, test_keys: set[tuple[str, str]]) -> dict:
    """How many items lose context because of the >= 0.95 cut.

    Measured on the real retrieval policy, with the real eligibility (context
    items come from the training corpus only — double exclusion), so the
    numbers are the ones the generator will actually see.
    """
    index = load_index(SIMILARITY_PATH)
    eligible = {
        (s_, v_)
        for s_, v_ in zip(items["survey_id"], items["variable"], strict=True)
        if (s_, v_) not in test_keys
    }
    under_with = under_without = dropped = 0
    affected: list[dict] = []
    for s_, v_ in zip(items["survey_id"], items["variable"], strict=True):
        key = (s_, v_)
        free = nearest_context_items(
            index, key, k=CONTEXT_K, eligible=eligible, max_cosine=1.01
        )
        cut = nearest_context_items(index, key, k=CONTEXT_K, eligible=eligible)
        if len(free) < CONTEXT_K:
            under_without += 1
        if len(cut) < CONTEXT_K:
            under_with += 1
        if len(cut) < len(free):
            dropped += 1
            if len(cut) < CONTEXT_K <= len(free):
                affected.append(
                    {
                        "survey_id": s_,
                        "variable": v_,
                        "n_context_without_rule": len(free),
                        "n_context_with_rule": len(cut),
                        "is_test_item": key in test_keys,
                    }
                )
    return {
        "k": CONTEXT_K,
        "max_cosine": sp.CONTEXT_MAX_COSINE,
        "n_items": items.height,
        "n_items_under_k_without_rule": under_without,
        "n_items_under_k_with_rule": under_with,
        "n_items_pushed_under_k_by_rule": len(affected),
        "n_items_losing_at_least_one_neighbour": dropped,
        "items_pushed_under_k": affected,
    }


# --------------------------------------------------------------------------
# 2c. the human sign-off file on the >= 0.97 pairs
# --------------------------------------------------------------------------


def write_signoff_template(pairs: list[dict]) -> None:
    """Refresh data/split/review_pairs_signed_off.json, preserving verdicts.

    The mechanism that produced the 0.97 list worked last time; what failed
    was that nobody read the list before the freeze. So the list is now a
    file a human has to edit, and tests/test_split.py fails while any pair is
    unsigned. A pair already signed off keeps its verdict when the split is
    rebuilt and the same two questions come back; everything else is written
    unsigned, which is a loud failure and not a silent one.
    """
    existing: dict[str, dict] = {}
    if sp.REVIEW_SIGNOFF_PATH.exists():
        with open(sp.REVIEW_SIGNOFF_PATH, encoding="utf-8") as handle:
            for entry in json.load(handle).get("pairs", []):
                existing[sp.pair_signature(entry)] = entry

    entries = []
    for pair in pairs:
        signature = sp.pair_signature(pair)
        previous = existing.get(signature, {})
        entries.append(
            {
                **pair,
                "signature": signature,
                "signed_off": bool(previous.get("signed_off", False)),
                "verdict": previous.get("verdict", ""),
                "reviewer": previous.get("reviewer", ""),
                "reviewed_on": previous.get("reviewed_on", ""),
            }
        )

    payload = {
        "version": "1.0",
        "generated_at": sp.today(),
        "generated_by": "scripts/14_build_split.py",
        "threshold": sp.REVIEW_COSINE,
        "instructions": (
            "One entry per pair at cosine >= "
            f"{sp.REVIEW_COSINE} involving a test item. Read both labels. The "
            "cosine is blind to negation and to a flipped referent: two items "
            "at 0.987 in this corpus asked about the share of CANADIANS versus "
            "AMERICANS who were worried. For each pair, set signed_off to true "
            "and write a one-line verdict saying whether the two questions are "
            "really the same question (the quasi_duplicate bin is legitimate) "
            "or not (the test item is in the wrong distance bin and belongs in "
            "split.TEST_ITEM_EXCLUSIONS). Fill reviewer and reviewed_on. "
            "tests/test_split.py::test_every_review_pair_is_signed_off fails "
            "until every entry here is signed off — that failure is the lock, "
            "not a regression."
        ),
        "n_pairs": len(entries),
        "n_signed_off": sum(1 for e in entries if e["signed_off"]),
        "pairs": entries,
    }
    with open(sp.REVIEW_SIGNOFF_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"review sign-off file: {payload['n_signed_off']}/{payload['n_pairs']} "
        f"pairs signed -> {sp.REVIEW_SIGNOFF_PATH}"
    )


# --------------------------------------------------------------------------
# 3. selection
# --------------------------------------------------------------------------


def main() -> int:
    sp.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sp.SEED)

    items = pl.read_parquet(ITEMS_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"items: {items.height} · embeddings: {len(embeddings)}")

    heldout_resp, survey_rows = build_respondent_split(rng)
    print(
        f"held-out respondents: {heldout_resp.height} "
        f"over {heldout_resp['__survey_id'].n_unique()} surveys"
    )

    validability = build_validability(items, heldout_resp)
    table = items.join(validability, on=["survey_id", "variable"], how="left")

    # Cosine matrix over the whole perimeter, in items.parquet row order.
    order = np.array(
        [
            embeddings.position((s, v))
            for s, v in zip(table["survey_id"], table["variable"], strict=True)
        ]
    )
    matrix = embeddings.matrix[order]
    cosines = (matrix @ matrix.T).astype(np.float32)

    # Non-opinion items are gone upstream (scripts/10_build_item_table.py).
    # Two bars are left here. sp.TEST_ITEM_EXCLUSIONS: items that train but may
    # not be test items. is_context_only: reported behaviour, which may be
    # retrieved as context but is never a target, at test or at training.
    context_only = (
        table["is_context_only"].fill_null(False).to_list()
        if "is_context_only" in table.columns
        else [False] * table.height
    )
    # A pre-registered test item has to be readable by the human who signs
    # the list off, and by the model: no display_label means the prompt
    # shows the raw stem alone, and in this corpus those stems are the
    # elliptical ones ("25. possible ou impossible ... d) Que vous votiez
    # pour un autre parti"). Six items, all cecd_elxn_qc_2007. They still
    # train; they just cannot be the questions we publish.
    no_label = [
        not (label or "").strip() for label in table["display_label"].to_list()
    ]
    excluded_mask = [
        sp.is_excluded_test_item(s_, v_) or c or nl
        for s_, v_, c, nl in zip(
            table["survey_id"], table["variable"], context_only, no_label,
            strict=True,
        )
    ]
    excluded = sorted(
        (s_, v_)
        for s_, v_ in zip(table["survey_id"], table["variable"], strict=True)
        if sp.is_excluded_test_item(s_, v_)
    )
    n_context_only = sum(1 for c in context_only if c)
    print(f"barred from the test set by design: {len(excluded)} -> {excluded}")
    print(
        f"barred from every target pool as context-only (reported "
        f"behaviour): {n_context_only}"
    )
    print(f"barred from the test set for want of a display_label: {sum(no_label)}")

    # Provisional bins: distance to the full corpus. Used only to form the
    # candidate pool; the published bins come from the recomputation below.
    provisional, _ = sp.distance_to_training(cosines, test_positions=[])
    candidates = table.with_columns(
        cos_provisional=pl.Series(provisional),
        distance_bin=pl.Series(sp.assign_bins(provisional)),
    ).with_columns(
        eligible=(pl.col("n_cells_ge30").fill_null(0) >= sp.MIN_ESTIMABLE_CELLS)
        & ~pl.Series("_excluded", excluded_mask)
    )
    print(f"eligible candidates: {int(candidates['eligible'].sum())}")

    selection = sp.select_test_items(candidates, cosines, rng)
    if selection.shortfall:
        print(f"WARNING shortfall in design cells: {selection.shortfall}")
    positions = selection.positions
    print(f"selected: {len(positions)} test items")

    # Distance to the FINAL training corpus. The pairwise separation constraint
    # makes this exact for every bin above 'isolated' (see split.py docstring).
    cos_train, nearest = sp.distance_to_training(cosines, positions)

    same_survey = np.equal.outer(
        np.asarray(table["survey_id"].to_list()),
        np.asarray(table["survey_id"].to_list()),
    )
    cross = np.array(cosines, dtype=np.float32, copy=True)
    np.fill_diagonal(cross, -np.inf)
    cross[:, positions] = -np.inf
    cross[same_survey] = -np.inf
    cos_train_cross = cross.max(axis=1)

    rows = table.to_dicts()
    labels = [
        {"survey_id": r["survey_id"], "variable": r["variable"], "label": label_of(r)}
        for r in rows
    ]

    entries = []
    for i in positions:
        row = rows[i]
        j = int(nearest[i])
        # Round first, bin second: the published bin must follow the published
        # cosine, not a full-precision value nobody can see.
        cos = round(float(cos_train[i]), 6)
        entries.append(
            {
                "survey_id": row["survey_id"],
                "variable": row["variable"],
                "language": row["language"],
                "year": row["year"],
                "question_text": row["question_text"],
                "display_label": row["display_label"],
                "themes": json.loads(row["themes"] or "[]"),
                "n_options": row["n_options"],
                "is_ordinal": row["is_ordinal"],
                "n_valid_responses": row["n_valid_responses"],
                "distance_bin": sp.bin_of(cos),
                "cos_to_train": cos,
                "distance_to_train": round(1.0 - cos, 6),
                "cos_to_corpus": round(float(provisional[i]), 6),
                "cos_to_train_cross_survey": round(float(cos_train_cross[i]), 6),
                "nearest_train_survey_id": rows[j]["survey_id"],
                "nearest_train_variable": rows[j]["variable"],
                "nearest_train_label": labels[j]["label"],
                "strata_dimensions": list(
                    strata.load_definition().dimensions_for(row["survey_id"])
                ),
                "n_heldout_cells_ge30": int(row["n_cells_ge30"] or 0),
                "n_heldout_cells_ge100": int(row["n_cells_ge100"] or 0),
            }
        )
    entries.sort(key=lambda e: (e["distance_bin"], e["language"], e["survey_id"]))

    test_keys = {(e["survey_id"], e["variable"]) for e in entries}
    leak_effect = context_leak_effect(items, test_keys)
    print(
        "anti-leak rule: "
        f"{leak_effect['n_items_under_k_without_rule']} items under k={CONTEXT_K} "
        f"without it, {leak_effect['n_items_under_k_with_rule']} with it "
        f"(+{leak_effect['n_items_pushed_under_k_by_rule']})"
    )

    pairs = sp.review_pairs(cosines, labels, positions)
    pairs_wide = sp.review_pairs(
        cosines, labels, positions, threshold=sp.REVIEW_COSINE_WIDE
    )

    # ---------------------------------------------------------------- outputs
    payload = {
        "version": "1.0",
        "generated_at": sp.today(),
        "seed": sp.SEED,
        "plan_reference": "docs/plan_article.md §4 (phase 2)",
        "n_items": len(entries),
        "selection_rule": sp.SELECTION_RULE,
        "circularity_note": (
            "Distances are measured against the final training corpus. Test "
            f"items are pairwise below {sp.TEST_PAIR_MAX_COSINE}, so any "
            "neighbour at or above that cosine is necessarily a training item: "
            "cos_to_train equals cos_to_corpus exactly for every bin above "
            "'isolated', and an isolated item can only move further away."
        ),
        "items": entries,
        "excluded_test_items": [
            {"survey_id": s_, "variable": v_} for s_, v_ in sorted(excluded)
        ],
        "context_leak_rule": CONTEXT_LEAK_RULE,
        "review_pairs_threshold": sp.REVIEW_COSINE,
        "review_pairs": pairs,
        "review_pairs_wide_threshold": sp.REVIEW_COSINE_WIDE,
        "review_pairs_wide": pairs_wide,
    }
    with open(sp.HELDOUT_ITEMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    write_signoff_template(pairs)

    heldout_resp.sort(["__survey_id", "__respondent_id"]).write_parquet(
        sp.HELDOUT_RESPONDENTS_PATH
    )

    # diagnostics: one row per test item, plus the size of every design cell
    diag = pl.DataFrame(
        [
            {
                "kind": "test_item",
                "survey_id": e["survey_id"],
                "variable": e["variable"],
                "language": e["language"],
                "distance_bin": e["distance_bin"],
                "cos_to_train": e["cos_to_train"],
                "cos_to_corpus": e["cos_to_corpus"],
                "cos_to_train_cross_survey": e["cos_to_train_cross_survey"],
                "n_valid_responses": e["n_valid_responses"],
                "n_heldout_cells_ge30": e["n_heldout_cells_ge30"],
                "n_heldout_cells_ge100": e["n_heldout_cells_ge100"],
                "n_candidates_in_cell": None,
                "label": label_of(
                    {
                        "display_label": e["display_label"],
                        "question_text": e["question_text"],
                    }
                ),
            }
            for e in entries
        ]
    ).with_columns(pl.col("n_candidates_in_cell").cast(pl.Int64))
    # Sorted: polars' group_by does not promise a row order, and an unsorted
    # tail is the one thing that made two runs of this script differ.
    pool = (
        candidates.filter("eligible")
        .group_by(["distance_bin", "language"])
        .agg(pl.len().alias("n_candidates_in_cell"))
        .sort(["distance_bin", "language"])
    )
    pool_rows = pl.DataFrame(
        [
            {
                "kind": "candidate_pool",
                "survey_id": None,
                "variable": None,
                "language": r["language"],
                "distance_bin": r["distance_bin"],
                "cos_to_train": None,
                "cos_to_corpus": None,
                "cos_to_train_cross_survey": None,
                "n_valid_responses": None,
                "n_heldout_cells_ge30": None,
                "n_heldout_cells_ge100": None,
                "n_candidates_in_cell": r["n_candidates_in_cell"],
                "label": None,
            }
            for r in pool.to_dicts()
        ],
    )
    pl.concat([diag, pool_rows], how="diagonal_relaxed").select(
        diag.columns
    ).write_csv(sp.DIAGNOSTICS_PATH)

    by_bin = {
        name: sum(1 for e in entries if e["distance_bin"] == name)
        for name in sp.BIN_NAMES
    }
    by_lang = {
        lang: sum(1 for e in entries if e["language"] == lang) for lang in sp.LANGUAGES
    }
    by_bin_lang = {
        f"{name}|{lang}": sum(
            1
            for e in entries
            if e["distance_bin"] == name and e["language"] == lang
        )
        for name, lang in sp.design_cells()
    }
    by_survey: dict[str, int] = {}
    for e in entries:
        by_survey[e["survey_id"]] = by_survey.get(e["survey_id"], 0) + 1

    manifest = {
        "version": "1.0",
        "generated_at": sp.today(),
        "seed": sp.SEED,
        "plan_reference": "docs/plan_article.md §4 (phase 2)",
        "generated_by": "scripts/14_build_split.py",
        "input_hashes": sp.input_hashes(
            [ITEMS_PATH, SIMILARITY_PATH, EMBEDDINGS_PATH, STRATA_PATH]
        ),
        "parameters": {
            "distance_bins": [
                {"name": n, "cos_lo": lo, "cos_hi": hi} for n, lo, hi in sp.DISTANCE_BINS
            ],
            "languages": list(sp.LANGUAGES),
            "items_per_design_cell": sp.ITEMS_PER_DESIGN_CELL,
            "max_items_per_survey": sp.MAX_ITEMS_PER_SURVEY,
            "test_pair_max_cosine": sp.TEST_PAIR_MAX_COSINE,
            "min_estimable_cells": sp.MIN_ESTIMABLE_CELLS,
            "cell_n_contrast": sp.CELL_N_CONTRAST,
            "cell_n_absolute": sp.CELL_N_ABSOLUTE,
            "large_survey_n": sp.LARGE_SURVEY_N,
            "holdout_fraction_large": sp.HOLDOUT_FRACTION_LARGE,
            "holdout_fraction_small": sp.HOLDOUT_FRACTION_SMALL,
            "review_cosine": sp.REVIEW_COSINE,
            "review_cosine_wide": sp.REVIEW_COSINE_WIDE,
            "n_excluded_test_items": len(excluded),
            "n_context_only_items": n_context_only,
            "context_max_cosine": sp.CONTEXT_MAX_COSINE,
        },
        "item_selection_rule": sp.SELECTION_RULE,
        "context_leak_rule": CONTEXT_LEAK_RULE,
        "context_leak_effect": leak_effect,
        "respondent_selection_rule": sp.RESPONDENT_RULE,
        "counts": {
            "n_corpus_items": items.height,
            "n_eligible_candidates": int(candidates["eligible"].sum()),
            "n_test_items": len(entries),
            "n_training_items": items.height - len(entries),
            "n_training_targets": items.height - len(entries) - n_context_only,
            "n_context_only_items": n_context_only,
            "n_items_without_display_label": sum(no_label),
            "by_distance_bin": by_bin,
            "by_language": by_lang,
            "by_design_cell": by_bin_lang,
            "by_survey": by_survey,
            "n_items_with_cells_ge30": sum(
                1 for e in entries if e["n_heldout_cells_ge30"] > 0
            ),
            "n_items_with_cells_ge100": sum(
                1 for e in entries if e["n_heldout_cells_ge100"] > 0
            ),
            "median_heldout_cells_ge30": float(
                np.median([e["n_heldout_cells_ge30"] for e in entries])
            ),
            "median_heldout_cells_ge100": float(
                np.median([e["n_heldout_cells_ge100"] for e in entries])
            ),
            "n_review_pairs": len(pairs),
            "n_review_pairs_wide": len(pairs_wide),
            "shortfall": {f"{k[0]}|{k[1]}": v for k, v in selection.shortfall.items()},
            "n_respondents_total": sum(r["n_respondents"] for r in survey_rows),
            "n_respondents_heldout": heldout_resp.height,
        },
        "by_survey_respondents": survey_rows,
        "circularity": {
            "problem": (
                "The distance of a test item to the training corpus depends on "
                "which items are in training, hence on the split itself."
            ),
            "method": (
                "Constraint, not iteration: any two test items sit pairwise below "
                f"cosine {sp.TEST_PAIR_MAX_COSINE}, the 'isolated' threshold. A "
                "neighbour at or above 0.70 of a test item is therefore always a "
                "training item, so the nearest-training-neighbour cosine equals "
                "the nearest-corpus-neighbour cosine exactly for every bin above "
                "'isolated'; an isolated item can only lose a neighbour and move "
                "further away, staying isolated. Published distances are still "
                "recomputed exactly against the final training corpus, from "
                "data/item_embeddings.parquet (full 2070x2070 cosine matrix, not "
                "the k=50 index, which truncation could otherwise bias)."
            ),
            "max_abs_shift_cos": round(
                float(
                    np.max(
                        np.abs(
                            provisional[positions] - cos_train[positions]
                        )
                    )
                ),
                6,
            ),
            "n_items_changing_bin": sum(
                1
                for i in positions
                if sp.bin_of(round(float(provisional[i]), 6))
                != sp.bin_of(round(float(cos_train[i]), 6))
            ),
        },
    }
    with open(sp.MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"review pairs >= {sp.REVIEW_COSINE}: {len(pairs)}")
    print(f"review pairs >= {sp.REVIEW_COSINE_WIDE}: {len(pairs_wide)}")
    print(f"wrote {sp.SPLIT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
