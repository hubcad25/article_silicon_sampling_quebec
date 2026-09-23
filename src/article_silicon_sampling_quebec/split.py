"""Pre-registered split — phase 2 of docs/plan_article.md (§4, §2.1, §2.3, §3.5).

This module is the methodological lock of the project. Once
``data/split/`` is frozen and committed, nothing in the training corpus nor in
the list of test items may change.

Three rules, all enforced here
------------------------------
**Double exclusion.** A test item is never a training target *and* never
injected as a context item. The retrieval pool at inference time is the
training corpus only (``context_pool``). Letting a test item back in as a
neighbour of a training target would train on the test data through the back
door — the single most expensive mistake available here.

**Factorial items x respondents.** Test items are scored against respondents
that are themselves held out. An unseen item answered by already-seen
respondents is an easier test than the one we want to report.

**Distance stratification, measured against the *training* corpus.** The
x-axis of the central figure is the cosine of a test item to its nearest
neighbour *in training*, which depends on the split itself. The circularity is
removed by construction rather than by iteration: selected test items are
forced to sit pairwise below ``TEST_PAIR_MAX_COSINE`` (= the "isolated"
threshold, 0.70). Any neighbour at cosine >= 0.70 of a test item is therefore
guaranteed to be a training item, so for every bin above "isolated" the
distance to the training corpus equals the distance to the full corpus
*exactly*. Only "isolated" items can move, and only further away — they stay
isolated. Distances published in the artefacts are always recomputed against
the final training corpus.

Non-opinion items (paradata, respondent attributes, factual-knowledge
questions, open lists, all-non-response thermometers, stems still carrying an
unresolved Qualtrics placeholder) are filtered upstream, in
``scripts/10_build_item_table.py`` via ``corpus.perimeter``, so they are
neither targets nor retrievable context neighbours. This module used to carry
a stopgap copy of that list; it is gone.

**Context-only items** are the in-between case, added on 22 September:
reported behaviour (voted, donated, signed a petition, watched the debate) is
not an attitude, and its observed distribution is itself inflated by
over-reporting, so it may never be a target — at test OR at training. It is
still useful *about* a respondent, so it stays in the corpus and in the
retrieval index and is only barred from the target pool, by
``perimeter.is_context_only`` and the ``is_context_only`` column of
``items.parquet``. See ``Split.training_items``.

Nothing here touches the network: the embeddings, the item table and the
similarity index are read from ``data/``; microdata comes from the local
``data/cache/`` parquets through ``corpus.blob``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ItemKey = tuple[str, str]  # (survey_id, variable)

_REPO = Path(__file__).resolve().parents[2]
SPLIT_DIR = _REPO / "data" / "split"
HELDOUT_ITEMS_PATH = SPLIT_DIR / "heldout_items.json"
HELDOUT_RESPONDENTS_PATH = SPLIT_DIR / "heldout_respondents.parquet"
MANIFEST_PATH = SPLIT_DIR / "split_manifest.json"
DIAGNOSTICS_PATH = SPLIT_DIR / "split_diagnostics.csv"
#: Human sign-off on the >= REVIEW_COSINE pairs. Filled in by a person, never
#: by a script; ``tests/test_split.py`` fails until it covers every pair.
REVIEW_SIGNOFF_PATH = SPLIT_DIR / "review_pairs_signed_off.json"

# --------------------------------------------------------------------------
# Frozen parameters — every number the selection depends on lives here.
# --------------------------------------------------------------------------

#: Single random seed of the whole split. Changing it invalidates the freeze.
SEED = 20260922

#: Distance bins on the cosine to the nearest training item, low to high.
#: The 0.95 / 0.70 anchors come from the hand validation of step 1.5; the two
#: interior cuts split the remaining mass roughly in half.
DISTANCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("isolated", 0.00, 0.70),
    ("far", 0.70, 0.775),
    ("moderate", 0.775, 0.85),
    ("near", 0.85, 0.95),
    ("quasi_duplicate", 0.95, 1.01),
)
BIN_NAMES: tuple[str, ...] = tuple(name for name, _, _ in DISTANCE_BINS)

#: Language strata. Mean cosine is 0.775 within-language against 0.633
#: across, so semantic distance partly *is* language: an unbalanced test set
#: would confound the two on the distance-performance curve.
LANGUAGES: tuple[str, ...] = ("en", "fr")

#: Balanced factorial design: 5 distance bins x 2 languages x 6 items = 60.
#: 4 per cell (40 items) is too thin: item difficulty is the dominant source of
#: dispersion (§5, paired-by-item bootstrap), so a cell of 4 cannot separate a
#: condition effect from one hard item.
ITEMS_PER_DESIGN_CELL = 6

#: No survey may dominate the test set. Raised from 6 to 7 when
#: ITEMS_PER_DESIGN_CELL went from 4 to 6: all English items come from 5
#: surveys and each language now needs 30 of them, so a cap of 6 leaves the
#: quasi_duplicate|en cell 2 items short. 7 is the smallest value with no
#: shortfall; it still keeps every survey under 12% of the test set.
MAX_ITEMS_PER_SURVEY = 7

#: Pairwise separation inside the test set — see the module docstring. Equal to
#: the "isolated" threshold, which is what makes the distance stable.
TEST_PAIR_MAX_COSINE = 0.70

#: Respondent holdout. Training pairs are sampled (~1e5 out of ~1e7 available
#: respondent x item pairs), so respondents are not the scarce resource;
#: the size of the held-out cell is. Small surveys therefore give more.
LARGE_SURVEY_N = 10_000
HOLDOUT_FRACTION_LARGE = 0.25
HOLDOUT_FRACTION_SMALL = 0.50

#: Cell sizes, counted among held-out respondents *with a valid answer to the
#: item*. 30 = contrasts between conditions (the observed cell's error is
#: common to both arms and cancels in the paired bootstrap); 100 = absolute
#: precision claims and the flattening diagnostic (§3.5).
CELL_N_CONTRAST = 30
CELL_N_ABSOLUTE = 100

#: An item with fewer than this many estimable cells cannot carry a by-strata
#: comparison and is not eligible for the test set.
MIN_ESTIMABLE_CELLS = 8

#: Pairs at or above this cosine involving a test item go to human review:
#: the cosine is blind to negation and to a flipped referent.
REVIEW_COSINE = 0.97

#: Items that are eligible targets and stay in TRAINING, but are barred from
#: the TEST set by a documented decision. Not a quality filter — a design one.
#:
#:   eeq_2014/Q24A, Q24B and Q24C are the three slots of ONE ranking battery
#:   ("rank these three constitutional options: first, second, third
#:   choice"). Each sits at ~0.96-0.97 of the other two, which lands whichever
#:   one is drawn in the quasi_duplicate bin. That is a battery artefact, not
#:   the "the same question was asked in an earlier survey" case the bin is
#:   meant to measure, and it would make the easy end of the coverage curve
#:   look easier than it is. Barring only Q24C (the first pass of this audit)
#:   did not hold: the next draw took Q24A and reproduced the artefact
#:   exactly. The bar is on the battery, so all three slots carry it.
#:
#:   ces_2021/pes21_partic2_2 sits at cosine 1.000 of ten training items — the
#:   other slots of the same "indicate how often you have done each of the
#:   following" battery (pes21_partic1_1 … pes21_partic3_4). All twelve slots
#:   share a byte-identical embedding text: ces_2021 stores the 80-character
#:   truncated stem and a single generic display_label ("Activités de
#:   participation et d'engagement politique") for every slot, so the embedding
#:   never sees which activity is being asked about. The 1.000 is an artefact of
#:   the catalogue, not a property of the questions — same failure mode as
#:   eeq_2014/Q24C — and it would put a fake ceiling on the easy end of the
#:   coverage curve. (The ces_2019_online and ces_2025 twins of this battery do
#:   carry per-slot display labels and are unaffected.) Since 22 September the
#:   whole partic battery is also context-only — reported behaviour — so this
#:   entry is now belt and braces; it is kept because the battery-artefact
#:   argument stands on its own and would survive that decision being revisited.
#:
#: The genuine quasi-duplicates are deliberately kept: eeq_2008/q22 against
#: eeq_2007/q22, and eeq_2018/q20a against q20b, are the same question asked
#: of different samples — exactly what this bin is for.
TEST_ITEM_EXCLUSIONS: frozenset[ItemKey] = frozenset(
    {
        # Battery ranking slots: not a re-asked question, just another slot of a
        # ranking whose other slots are in training.
        ("eeq_2014", "Q24A"),
        ("eeq_2014", "Q24B"),
        ("eeq_2014", "Q24C"),
        # Check-all-that-apply slot: byte-identical embedding text across the
        # 12 slots of the battery.
        ("ces_2021", "pes21_partic2_2"),
        # Reviewed by hand before the freeze (2026-09-22):
        # Recall of a vote cast four years earlier. Current-election vote choice
        # stays a valid target (it reports a just-performed act and is the
        # central variable of electoral research); recall at four years carries
        # documented bandwagon and misremembering bias, so the observed
        # distribution is itself biased.
        ("cecd_elxn_qc_1998", "vote94"),
        # "Which party is best at addressing this issue?" — "this issue" refers
        # to the respondent's answer to the previous question, which the prompt
        # never shows. Same failure as unresolved piping, but with no
        # placeholder syntax to catch it mechanically.
        ("ces_2019_phone", "q8"),
        # question_text byte-identical to its training twin pes21_internetrisk1
        # ("Which statement comes closest to your own view?") with the same
        # generic battery display_label; only the options differ (voting online
        # vs registering online). Same mode as pes21_partic2_2.
        ("ces_2021", "pes21_internetrisk2"),
    }
)


def is_excluded_test_item(survey_id: str, variable: str) -> bool:
    """True when an item may train but may not be a test item."""
    return (survey_id, variable) in TEST_ITEM_EXCLUSIONS


#: Anti-leak rule on the context (C1/C2). No item at or above this cosine to
#: the target may ever be injected as a context item — see
#: ``prompts.nearest_context_items``. Without it the training twin of a test
#: item is a legal context for the same respondent, and the model is handed
#: the answer: cps21_votechoice and cps21_v_advance sit at 0.967 in the same
#: survey, over the same respondents. Same value as the quasi_duplicate cut,
#: on purpose: a context item is allowed to be near, never a duplicate.
CONTEXT_MAX_COSINE = 0.95


#: Second, wider review list: every training neighbour of a test item above
#: this cosine. The 0.97 list is short by construction (only the
#: quasi-duplicate bin can populate it), and the bin boundary that actually
#: decides the x-axis sits at 0.95 and 0.85.
REVIEW_COSINE_WIDE = 0.90

SELECTION_RULE = (
    "Eligible candidates = items of data/items.parquet (already purged upstream "
    "by scripts/10_build_item_table.py of paradata, respondent attributes, "
    "factual-knowledge questions, derived recodes, open-list questions, "
    "all-non-response items and stems still carrying an unresolved Qualtrics "
    "placeholder), minus TEST_ITEM_EXCLUSIONS, minus the items flagged "
    "is_context_only (reported behaviour: never a target, at test or at "
    "training), minus the items with no display_label (their stem alone is "
    "elliptical and unreadable), "
    "whose origin survey has "
    f">= {MIN_ESTIMABLE_CELLS} strata cells holding >= {CELL_N_CONTRAST} held-out "
    "respondents with a valid answer to that item, cells being defined by "
    "data/strata_definition.json (per-survey dimensions). Candidates are crossed "
    f"into {len(DISTANCE_BINS)} distance bins x {len(LANGUAGES)} languages; "
    f"{ITEMS_PER_DESIGN_CELL} items are drawn uniformly at random per design cell "
    f"with numpy.random.default_rng(SEED={SEED}), subject to two constraints: at "
    f"most {MAX_ITEMS_PER_SURVEY} items per survey, and pairwise cosine between any "
    f"two test items < {TEST_PAIR_MAX_COSINE}. Design cells are visited in a fixed "
    "order (bins low to high, 'en' then 'fr'); a candidate violating a constraint "
    "is skipped, never swapped in later. Distance bins are assigned on the cosine "
    "to the nearest item of the FINAL training corpus, recomputed exactly from "
    "data/item_embeddings.parquet after selection."
)

RESPONDENT_RULE = (
    "Per survey, respondents that resolve to a strata cell are ranked by a seeded "
    "uniform draw within their cell and the first floor(frac * cell_n) are held "
    f"out, with frac = {HOLDOUT_FRACTION_LARGE} for surveys of >= {LARGE_SURVEY_N} "
    f"respondents and {HOLDOUT_FRACTION_SMALL} otherwise. Respondents that resolve "
    "to no cell (coarse or missing on any dimension) stay in training: they can "
    "never appear in an evaluation, so holding them out would only cost training "
    "data and leaks nothing."
)


# --------------------------------------------------------------------------
# Distance bins
# --------------------------------------------------------------------------


def bin_of(cosine: float) -> str:
    """Name of the distance bin holding `cosine`."""
    for name, lo, hi in DISTANCE_BINS:
        if lo <= cosine < hi:
            return name
    raise ValueError(f"cosine out of range: {cosine}")


def assign_bins(cosines: Iterable[float]) -> list[str]:
    return [bin_of(float(c)) for c in cosines]


def design_cells() -> list[tuple[str, str]]:
    """The (distance bin, language) design cells, in selection order."""
    return [(name, lang) for name in BIN_NAMES for lang in LANGUAGES]


# --------------------------------------------------------------------------
# Respondent split
# --------------------------------------------------------------------------


def holdout_fraction(n_respondents: int) -> float:
    """Share of respondents held out in a survey of that size."""
    return (
        HOLDOUT_FRACTION_LARGE
        if n_respondents >= LARGE_SURVEY_N
        else HOLDOUT_FRACTION_SMALL
    )


def split_survey_respondents(
    profiles: pl.DataFrame,
    dimensions: Sequence[str],
    unresolved: Sequence[str],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Held-out respondents of one survey, stratified by cell.

    `profiles` is one row per respondent with one column per canonical SES
    dimension (``corpus.strata.resolved_profiles``). Returns the held-out rows
    only, with their ``cell``; everyone else trains.
    """
    dims = list(dimensions)
    frame = profiles.sort("__respondent_id")
    assigned = frame.filter(
        pl.all_horizontal([~pl.col(d).is_in(list(unresolved)) for d in dims])
    ).with_columns(cell=pl.concat_str([pl.col(d) for d in dims], separator="|"))
    if assigned.height == 0:
        return assigned.head(0).select(
            "__survey_id", "__respondent_id", "__weight", "cell"
        )

    frac = holdout_fraction(frame.height)
    draw = pl.Series("_u", rng.random(assigned.height))
    held = (
        assigned.with_columns(draw)
        .with_columns(
            _rank=pl.col("_u").rank("ordinal").over("cell"),
            _cell_n=pl.len().over("cell"),
        )
        .filter(pl.col("_rank") <= (pl.col("_cell_n") * frac).floor())
    )
    return held.select("__survey_id", "__respondent_id", "__weight", "cell")


# --------------------------------------------------------------------------
# Validability of an item against the held-out respondents
# --------------------------------------------------------------------------


def estimable_cells(
    heldout: pl.DataFrame,
    microdata: pl.DataFrame,
    variables: Sequence[str],
    thresholds: Sequence[int] = (CELL_N_CONTRAST, CELL_N_ABSOLUTE),
) -> pl.DataFrame:
    """For each variable, how many held-out cells are large enough to score.

    `heldout` needs ``__respondent_id`` and ``cell``; `microdata` needs
    ``__respondent_id`` plus the variables. A response counts when it is
    non-null, the same definition as ``n_valid_responses`` in items.parquet.
    """
    known = [v for v in variables if v in microdata.columns]
    empty = {
        "variable": list(variables),
        **{f"n_cells_ge{t}": [0] * len(variables) for t in thresholds},
    }
    if not known or heldout.height == 0:
        return pl.DataFrame(empty)

    joined = heldout.select("__respondent_id", "cell").join(
        microdata.select(["__respondent_id", *known]),
        on="__respondent_id",
        how="inner",
    )
    per_cell = joined.group_by("cell").agg(
        [pl.col(v).count().alias(v) for v in known]
    )
    rows = []
    for variable in variables:
        row: dict[str, Any] = {"variable": variable}
        for threshold in thresholds:
            row[f"n_cells_ge{threshold}"] = (
                int((per_cell[variable] >= threshold).sum())
                if variable in known
                else 0
            )
        rows.append(row)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# Test-item selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """Outcome of one selection pass."""

    positions: list[int]  # row indices into the candidate table
    shortfall: dict[tuple[str, str], int]  # design cells left under target


def select_test_items(
    candidates: pl.DataFrame,
    pairwise: np.ndarray,
    rng: np.random.Generator,
    per_cell: int = ITEMS_PER_DESIGN_CELL,
    max_per_survey: int = MAX_ITEMS_PER_SURVEY,
    max_pair_cosine: float = TEST_PAIR_MAX_COSINE,
) -> Selection:
    """Draw the test items, stratified on distance bin x language.

    `candidates` holds ``survey_id``, ``language``, ``distance_bin`` and
    ``eligible``; `pairwise` is the candidate-by-candidate cosine matrix, in
    the same row order. Pure and deterministic given `rng`.
    """
    survey = candidates["survey_id"].to_list()
    language = candidates["language"].to_list()
    dbin = candidates["distance_bin"].to_list()
    eligible = candidates["eligible"].to_list()

    chosen: list[int] = []
    per_survey: dict[str, int] = {}
    shortfall: dict[tuple[str, str], int] = {}

    for name, lang in design_cells():
        pool = [
            i
            for i in range(candidates.height)
            if eligible[i] and dbin[i] == name and language[i] == lang
        ]
        rng.shuffle(pool)
        taken = 0
        for i in pool:
            if taken >= per_cell:
                break
            if per_survey.get(survey[i], 0) >= max_per_survey:
                continue
            if chosen and float(pairwise[i, chosen].max()) >= max_pair_cosine:
                continue
            chosen.append(i)
            per_survey[survey[i]] = per_survey.get(survey[i], 0) + 1
            taken += 1
        if taken < per_cell:
            shortfall[(name, lang)] = per_cell - taken

    return Selection(positions=sorted(chosen), shortfall=shortfall)


def distance_to_training(
    cosines: np.ndarray,
    test_positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest training neighbour of every item, self and test items excluded.

    `cosines` is the full item-by-item cosine matrix. Returns
    (max cosine, argmax index); items whose every neighbour is held out get
    -1 and a cosine of -inf, which cannot happen for a 2 000-item corpus but
    is not silently rounded away either.
    """
    matrix = np.array(cosines, dtype=np.float32, copy=True)
    np.fill_diagonal(matrix, -np.inf)
    matrix[:, list(test_positions)] = -np.inf
    best = matrix.argmax(axis=1)
    value = matrix[np.arange(matrix.shape[0]), best]
    best = np.where(np.isfinite(value), best, -1)
    return value, best


# --------------------------------------------------------------------------
# Double exclusion — the retrieval pool at inference time
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """The frozen split, as the rest of the pipeline sees it."""

    items: frozenset[ItemKey]
    respondents: dict[str, frozenset[str]]
    manifest: dict[str, Any]

    # -- items -------------------------------------------------------------

    def is_test_item(self, key: ItemKey) -> bool:
        return key in self.items

    def _key_frame(self, columns: tuple[str, str]) -> pl.DataFrame:
        survey, variable = columns
        return pl.DataFrame(
            {
                survey: [k[0] for k in sorted(self.items)],
                variable: [k[1] for k in sorted(self.items)],
            },
            schema={survey: pl.Utf8, variable: pl.Utf8},
        )

    def training_items(self, items: pl.DataFrame) -> pl.DataFrame:
        """The training TARGETS: `items` minus the test items, minus the
        context-only ones.

        Reported behaviour ("did you vote", "did you sign a petition") is
        never a target — not at test, and not at training either, because
        the observed distribution it would be trained on is itself inflated
        by over-reporting. It stays in `items` and in the similarity index
        so that ``context_pool`` can still retrieve it as C1 context.
        """
        kept = items.join(
            self._key_frame(("survey_id", "variable")),
            on=["survey_id", "variable"],
            how="anti",
        )
        if "is_context_only" in kept.columns:
            kept = kept.filter(~pl.col("is_context_only").fill_null(False))
        return kept

    def context_pool(self, neighbours: pl.DataFrame) -> pl.DataFrame:
        """Drop every held-out item from a retrieval result.

        The one function standing between the design and a back-door leak:
        C1 and C2 inject neighbours into the prompt, and a held-out item
        reached as a neighbour would be training data. Expects the neighbour
        columns of ``corpus.similarity.SimilarityIndex``.
        """
        return neighbours.join(
            self._key_frame(("neighbor_survey_id", "neighbor_variable")),
            on=["neighbor_survey_id", "neighbor_variable"],
            how="anti",
        )

    # -- respondents -------------------------------------------------------

    def is_test_respondent(self, survey_id: str, respondent_id: str) -> bool:
        return respondent_id in self.respondents.get(survey_id, frozenset())

    def training_respondents(
        self, survey_id: str, respondent_ids: Iterable[str]
    ) -> list[str]:
        held = self.respondents.get(survey_id, frozenset())
        return [r for r in respondent_ids if r not in held]


def load_split(
    items_path: Path | str = HELDOUT_ITEMS_PATH,
    respondents_path: Path | str = HELDOUT_RESPONDENTS_PATH,
    manifest_path: Path | str = MANIFEST_PATH,
) -> Split:
    """Load the frozen split from ``data/split/``."""
    with open(items_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    keys = frozenset(
        (entry["survey_id"], entry["variable"]) for entry in payload["items"]
    )
    frame = pl.read_parquet(respondents_path)
    grouped = frame.group_by("__survey_id").agg(pl.col("__respondent_id"))
    respondents = {
        survey: frozenset(ids)
        for survey, ids in zip(
            grouped["__survey_id"], grouped["__respondent_id"].to_list(), strict=True
        )
    }
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return Split(items=keys, respondents=respondents, manifest=manifest)


# --------------------------------------------------------------------------
# Manual review — the cosine is blind to negation and to a flipped referent
# --------------------------------------------------------------------------


def review_pairs(
    cosines: np.ndarray,
    labels: Sequence[dict[str, Any]],
    test_positions: Sequence[int],
    threshold: float = REVIEW_COSINE,
) -> list[dict[str, Any]]:
    """Every pair at >= `threshold` with a test item on one side.

    Two items at 0.987 in this corpus asked about the share of *Canadians*
    versus *Americans* who were worried: a near-duplicate by cosine and a
    different question. Such a pair puts a test item in the wrong distance
    bin, so the list is reviewed by hand before the freeze.
    """
    out: list[dict[str, Any]] = []
    test = set(test_positions)
    for i in sorted(test):
        row = np.asarray(cosines[i], dtype=np.float32).copy()
        row[i] = -np.inf
        for j in np.flatnonzero(row >= threshold):
            out.append(
                {
                    "test_survey_id": labels[i]["survey_id"],
                    "test_variable": labels[i]["variable"],
                    "test_label": labels[i]["label"],
                    "other_survey_id": labels[int(j)]["survey_id"],
                    "other_variable": labels[int(j)]["variable"],
                    "other_label": labels[int(j)]["label"],
                    "other_side": "test" if int(j) in test else "train",
                    "cosine": round(float(row[int(j)]), 4),
                }
            )
    return sorted(out, key=lambda r: -r["cosine"])


def pair_signature(pair: dict[str, Any]) -> str:
    """Stable identity of a review pair, order-independent on the two sides.

    The cosine is deliberately NOT part of it: a sign-off is a statement
    about two questions, and it should survive the fourth decimal moving.
    """
    left = f"{pair['test_survey_id']}/{pair['test_variable']}"
    right = f"{pair['other_survey_id']}/{pair['other_variable']}"
    return "::".join(sorted((left, right)))


def signed_off_pairs(path: Path | str = REVIEW_SIGNOFF_PATH) -> set[str]:
    """Signatures a human has actually signed off in ``path``.

    A missing file, or an entry whose ``signed_off`` is not exactly ``True``,
    counts as unsigned. There is no way to sign off by omission.
    """
    path = Path(path)
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        pair_signature(entry)
        for entry in payload.get("pairs", [])
        if entry.get("signed_off") is True
    }


# --------------------------------------------------------------------------
# Freeze helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(paths: Sequence[Path | str]) -> dict[str, str]:
    """sha256 of every input the split depends on, keyed by repo-relative path."""
    out = {}
    for path in paths:
        path = Path(path)
        try:
            key = str(path.resolve().relative_to(_REPO))
        except ValueError:
            key = str(path)
        out[key] = sha256_file(path)
    return out


def today() -> str:
    return date.today().isoformat()


__all__ = [
    "BIN_NAMES", "CELL_N_ABSOLUTE", "CELL_N_CONTRAST", "DIAGNOSTICS_PATH",
    "CONTEXT_MAX_COSINE", "DISTANCE_BINS", "HELDOUT_ITEMS_PATH",
    "HELDOUT_RESPONDENTS_PATH",
    "HOLDOUT_FRACTION_LARGE", "HOLDOUT_FRACTION_SMALL", "ITEMS_PER_DESIGN_CELL",
    "LANGUAGES", "LARGE_SURVEY_N", "MANIFEST_PATH", "MAX_ITEMS_PER_SURVEY",
    "MIN_ESTIMABLE_CELLS", "RESPONDENT_RULE", "REVIEW_COSINE",
    "REVIEW_COSINE_WIDE", "REVIEW_SIGNOFF_PATH", "SEED", "TEST_ITEM_EXCLUSIONS",
    "pair_signature", "signed_off_pairs",
    "is_excluded_test_item",
    "SELECTION_RULE", "SPLIT_DIR", "TEST_PAIR_MAX_COSINE", "Selection", "Split",
    "assign_bins", "bin_of", "design_cells", "distance_to_training",
    "estimable_cells", "holdout_fraction", "input_hashes", "load_split",
    "review_pairs", "select_test_items", "sha256_file",
    "split_survey_respondents", "today",
]
