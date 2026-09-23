"""Tests for the pre-registered split (phase 2). No network access.

The first two are the ones that matter: a test item must never be a training
target, and must never be reachable as a context item. Everything else in the
design survives a bug; those two do not.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from article_silicon_sampling_quebec import split as sp
from article_silicon_sampling_quebec.corpus import similarity

ITEMS_PATH = sp._REPO / "data" / "items.parquet"
EMBEDDINGS_PATH = sp._REPO / "data" / "item_embeddings.parquet"
SIMILARITY_PATH = sp._REPO / "data" / "item_similarity.parquet"

split_frozen = pytest.mark.skipif(
    not sp.HELDOUT_ITEMS_PATH.exists(),
    reason="run scripts/14_build_split.py first",
)


@pytest.fixture(scope="module")
def frozen() -> sp.Split:
    return sp.load_split()


@pytest.fixture(scope="module")
def items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS_PATH)


@pytest.fixture(scope="module")
def payload() -> dict:
    with open(sp.HELDOUT_ITEMS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(sp.MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# double exclusion
# --------------------------------------------------------------------------

@split_frozen
def test_no_test_item_is_a_training_target(frozen, items):
    training = frozen.training_items(items)
    n_context_only = int(items["is_context_only"].fill_null(False).sum())
    assert training.height == items.height - len(frozen.items) - n_context_only
    keys = set(zip(training["survey_id"], training["variable"], strict=True))
    assert keys.isdisjoint(frozen.items)


@split_frozen
def test_no_test_item_is_reachable_as_a_context_item(frozen):
    """The back door: a held-out item retrieved as a neighbour for C1/C2.

    Checked on the whole materialised k-NN index, not on a sample: every
    neighbour of every item must survive `context_pool` only if it trains.
    """
    index = similarity.load_index(SIMILARITY_PATH)
    pool = frozen.context_pool(index.frame)
    reached = set(
        zip(pool["neighbor_survey_id"], pool["neighbor_variable"], strict=True)
    )
    assert reached.isdisjoint(frozen.items)
    # and the filter actually removed something, i.e. the test is not vacuous
    assert pool.height < index.frame.height


@split_frozen
def test_context_pool_still_leaves_neighbours_for_every_test_item(frozen):
    """Double exclusion must not starve C2: a test item needs a retrieval set."""
    index = similarity.load_index(SIMILARITY_PATH)
    for key in sorted(frozen.items):
        allowed = frozen.context_pool(index.neighbors(key))
        assert allowed.height >= 6, f"{key} has fewer than k=6 training neighbours"


def test_context_pool_drops_a_held_out_neighbour():
    split = sp.Split(
        items=frozenset({("s1", "held")}),
        respondents={},
        manifest={},
    )
    neighbours = pl.DataFrame(
        {
            "neighbor_survey_id": ["s1", "s1"],
            "neighbor_variable": ["held", "kept"],
            "cosine": [0.99, 0.50],
        }
    )
    out = split.context_pool(neighbours)
    assert out["neighbor_variable"].to_list() == ["kept"]


# --------------------------------------------------------------------------
# factorial split: respondents
# --------------------------------------------------------------------------

@split_frozen
def test_respondent_sets_are_disjoint(frozen):
    frame = pl.read_parquet(sp.HELDOUT_RESPONDENTS_PATH)
    # no respondent held out twice, inside a survey or across surveys
    assert frame.select("__survey_id", "__respondent_id").n_unique() == frame.height
    for survey_id, held in frozen.respondents.items():
        kept = frozen.training_respondents(survey_id, sorted(held) + ["__absent__"])
        assert kept == ["__absent__"]


@split_frozen
def test_every_held_out_respondent_has_a_cell(frozen):
    frame = pl.read_parquet(sp.HELDOUT_RESPONDENTS_PATH)
    assert frame["cell"].null_count() == 0
    assert (frame["cell"].str.len_chars() > 0).all()


def test_split_survey_respondents_is_stratified_and_deterministic():
    profiles = pl.DataFrame(
        {
            "__survey_id": ["s"] * 40,
            "__respondent_id": [f"r{i:03d}" for i in range(40)],
            "__weight": [1.0] * 40,
            "age": ["25_34"] * 20 + ["35_44"] * 20,
            "gender": ["man", "woman"] * 20,
        }
    )
    dims = ("age", "gender")
    first = sp.split_survey_respondents(
        profiles, dims, sp_unresolved(), np.random.default_rng(7)
    )
    again = sp.split_survey_respondents(
        profiles, dims, sp_unresolved(), np.random.default_rng(7)
    )
    assert first.to_dicts() == again.to_dicts()
    # small survey -> half of every cell, and every cell is represented
    assert first.height == 20
    assert first["cell"].n_unique() == 4
    assert first.group_by("cell").len()["len"].to_list() == [5, 5, 5, 5]


def test_unresolved_respondents_are_never_held_out():
    profiles = pl.DataFrame(
        {
            "__survey_id": ["s"] * 4,
            "__respondent_id": ["a", "b", "c", "d"],
            "__weight": [1.0] * 4,
            "age": ["25_34", "__coarse__", "missing", "25_34"],
            "gender": ["man", "man", "man", "man"],
        }
    )
    held = sp.split_survey_respondents(
        profiles, ("age", "gender"), ("__coarse__", "missing"),
        np.random.default_rng(1),
    )
    assert set(held["__respondent_id"]) <= {"a", "d"}


def sp_unresolved() -> tuple[str, ...]:
    return ("__coarse__", "missing")


def test_holdout_fraction_follows_survey_size():
    assert sp.holdout_fraction(sp.LARGE_SURVEY_N) == sp.HOLDOUT_FRACTION_LARGE
    assert sp.holdout_fraction(sp.LARGE_SURVEY_N - 1) == sp.HOLDOUT_FRACTION_SMALL


# --------------------------------------------------------------------------
# distance axis
# --------------------------------------------------------------------------

def test_bins_tile_the_cosine_range():
    assert sp.bin_of(0.0) == "isolated"
    assert sp.bin_of(0.6999) == "isolated"
    assert sp.bin_of(0.70) == "far"
    assert sp.bin_of(0.95) == "quasi_duplicate"
    assert sp.bin_of(1.0) == "quasi_duplicate"
    with pytest.raises(ValueError):
        sp.bin_of(1.5)


def test_distance_to_training_ignores_self_and_held_out_items():
    cosines = np.array(
        [
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ],
        dtype=np.float32,
    )
    value, best = sp.distance_to_training(cosines, test_positions=[1])
    # item 0's best neighbour was the held-out item 1; it falls back to item 2
    assert best[0] == 2
    assert value[0] == pytest.approx(0.3)
    # a held-out item still gets its distance to the training corpus
    assert best[1] == 0
    assert value[1] == pytest.approx(0.9)


@split_frozen
def test_strata_coverage_is_the_one_announced(payload, manifest):
    entries = payload["items"]
    assert len(entries) == manifest["counts"]["n_test_items"]
    for name, lang in sp.design_cells():
        count = sum(
            1
            for e in entries
            if e["distance_bin"] == name and e["language"] == lang
        )
        assert count == sp.ITEMS_PER_DESIGN_CELL, (name, lang, count)
    for lang in sp.LANGUAGES:
        assert sum(1 for e in entries if e["language"] == lang) == (
            sp.ITEMS_PER_DESIGN_CELL * len(sp.BIN_NAMES)
        )


@split_frozen
def test_published_bin_matches_the_published_cosine(payload):
    for entry in payload["items"]:
        assert sp.bin_of(entry["cos_to_train"]) == entry["distance_bin"]
        assert entry["distance_to_train"] == pytest.approx(
            1.0 - entry["cos_to_train"], abs=1e-6
        )


@split_frozen
def test_circularity_is_neutralised_by_the_separation_constraint(payload):
    """Outside the isolated bin, removing the test items changes nothing.

    That equality is the whole argument: the published distance is the distance
    to the *training* corpus, and it did not have to be iterated to get there.
    """
    for entry in payload["items"]:
        if entry["distance_bin"] == "isolated":
            assert entry["cos_to_train"] <= entry["cos_to_corpus"] + 1e-6
        else:
            assert entry["cos_to_train"] == pytest.approx(
                entry["cos_to_corpus"], abs=1e-6
            )


@split_frozen
def test_test_items_are_pairwise_separated(frozen):
    embeddings = similarity.load_embeddings(EMBEDDINGS_PATH)
    keys = sorted(frozen.items)
    matrix = np.stack([embeddings.vector(k) for k in keys])
    cosines = matrix @ matrix.T
    np.fill_diagonal(cosines, 0.0)
    assert float(cosines.max()) < sp.TEST_PAIR_MAX_COSINE


# --------------------------------------------------------------------------
# selection rule
# --------------------------------------------------------------------------

def _toy_candidates(n_per_cell: int = 10) -> tuple[pl.DataFrame, np.ndarray]:
    rows = []
    for name in sp.BIN_NAMES:
        for lang in sp.LANGUAGES:
            for i in range(n_per_cell):
                rows.append(
                    {
                        "survey_id": f"s{name}_{lang}_{i}",
                        "variable": f"{name}_{lang}_{i}",
                        "language": lang,
                        "distance_bin": name,
                        "eligible": True,
                    }
                )
    frame = pl.DataFrame(rows)
    return frame, np.zeros((frame.height, frame.height), dtype=np.float32)


def test_selection_is_reproducible_at_a_fixed_seed():
    frame, pairwise = _toy_candidates()
    first = sp.select_test_items(frame, pairwise, np.random.default_rng(sp.SEED))
    again = sp.select_test_items(frame, pairwise, np.random.default_rng(sp.SEED))
    other = sp.select_test_items(frame, pairwise, np.random.default_rng(sp.SEED + 1))
    assert first.positions == again.positions
    assert first.positions != other.positions
    assert len(first.positions) == sp.ITEMS_PER_DESIGN_CELL * len(sp.design_cells())
    assert first.shortfall == {}


def test_selection_never_takes_an_ineligible_item():
    frame, pairwise = _toy_candidates()
    frame = frame.with_columns(
        eligible=pl.col("distance_bin") != "isolated"
    )
    out = sp.select_test_items(frame, pairwise, np.random.default_rng(sp.SEED))
    taken = frame[out.positions]
    assert "isolated" not in taken["distance_bin"].to_list()
    assert out.shortfall == {
        ("isolated", "en"): sp.ITEMS_PER_DESIGN_CELL,
        ("isolated", "fr"): sp.ITEMS_PER_DESIGN_CELL,
    }


def test_selection_respects_the_pairwise_separation():
    frame, _ = _toy_candidates()
    pairwise = np.full((frame.height, frame.height), 0.99, dtype=np.float32)
    out = sp.select_test_items(frame, pairwise, np.random.default_rng(sp.SEED))
    # everything is a near-duplicate of everything: only one item can be taken
    assert len(out.positions) == 1


def test_selection_respects_the_per_survey_cap():
    frame, pairwise = _toy_candidates()
    frame = frame.with_columns(survey_id=pl.lit("only"))
    out = sp.select_test_items(frame, pairwise, np.random.default_rng(sp.SEED))
    assert len(out.positions) == sp.MAX_ITEMS_PER_SURVEY


# --------------------------------------------------------------------------
# validability
# --------------------------------------------------------------------------

def test_estimable_cells_counts_valid_answers_per_cell():
    heldout = pl.DataFrame(
        {
            "__respondent_id": [f"r{i}" for i in range(6)],
            "cell": ["a", "a", "a", "b", "b", "b"],
        }
    )
    micro = pl.DataFrame(
        {
            "__respondent_id": [f"r{i}" for i in range(6)],
            "q1": [1, 2, 3, 1, None, None],
            "q2": [None, None, None, None, None, None],
        }
    )
    out = sp.estimable_cells(heldout, micro, ["q1", "q2", "absent"], thresholds=(2,))
    counts = dict(zip(out["variable"], out["n_cells_ge2"], strict=True))
    assert counts == {"q1": 1, "q2": 0, "absent": 0}


@split_frozen
def test_every_test_item_clears_the_validability_floor(payload):
    for entry in payload["items"]:
        assert entry["n_heldout_cells_ge30"] >= sp.MIN_ESTIMABLE_CELLS


@split_frozen
def test_no_barred_item_made_it_into_the_test_set(frozen):
    for survey_id, variable in frozen.items:
        assert not sp.is_excluded_test_item(survey_id, variable)


@split_frozen
def test_barred_test_items_are_still_training_items(frozen):
    """TEST_ITEM_EXCLUSIONS is a design bar, not a corpus filter."""
    items = pl.read_parquet(sp._REPO / "data" / "items.parquet")
    corpus = set(zip(items["survey_id"], items["variable"], strict=True))
    for key in sp.TEST_ITEM_EXCLUSIONS:
        assert key in corpus, f"{key} is no longer in the corpus at all"
        assert key not in frozen.items


@split_frozen
def test_non_opinion_items_are_gone_from_the_corpus_upstream(frozen):
    """The stopgap list left split.py: the filter now lives in step 1.2."""
    from article_silicon_sampling_quebec.corpus import perimeter

    items = pl.read_parquet(sp._REPO / "data" / "items.parquet")
    for survey_id, variable, options in zip(
        items["survey_id"], items["variable"], items["options"], strict=True,
    ):
        parsed = json.loads(options)
        assert not perimeter.is_listed_non_opinion(survey_id, variable)
        # The open-list bar counts substantive modalities only: a 0-10
        # thermometer is 11 answers plus its ways of not answering.
        assert perimeter.count_substantive_options(parsed) <= perimeter.MAX_OPTIONS
        assert not perimeter.all_options_non_response(parsed)


@split_frozen
def test_manifest_records_the_context_anti_leak_rule(manifest):
    assert manifest["parameters"]["context_max_cosine"] == sp.CONTEXT_MAX_COSINE
    effect = manifest["context_leak_effect"]
    assert effect["max_cosine"] == sp.CONTEXT_MAX_COSINE
    # The rule can only cost neighbours, never add any.
    assert (
        effect["n_items_under_k_with_rule"]
        >= effect["n_items_under_k_without_rule"]
    )


# --------------------------------------------------------------------------
# the freeze itself
# --------------------------------------------------------------------------

@split_frozen
def test_manifest_pins_the_seed_the_parameters_and_the_inputs(manifest):
    assert manifest["seed"] == sp.SEED
    params = manifest["parameters"]
    assert params["items_per_design_cell"] == sp.ITEMS_PER_DESIGN_CELL
    assert params["test_pair_max_cosine"] == sp.TEST_PAIR_MAX_COSINE
    assert params["min_estimable_cells"] == sp.MIN_ESTIMABLE_CELLS
    assert [b["name"] for b in params["distance_bins"]] == list(sp.BIN_NAMES)
    assert manifest["item_selection_rule"] == sp.SELECTION_RULE
    assert manifest["respondent_selection_rule"] == sp.RESPONDENT_RULE


@split_frozen
def test_manifest_pins_the_inputs(manifest):
    """The freeze is only a freeze if its inputs are byte-identical."""
    for path, digest in manifest["input_hashes"].items():
        assert sp.sha256_file(sp._REPO / path) == digest, f"{path} changed since freeze"


@split_frozen
def test_review_pairs_are_the_ones_a_human_must_read(payload):
    threshold = payload["review_pairs_threshold"]
    for pair in payload["review_pairs"]:
        assert pair["cosine"] >= threshold
        assert pair["test_label"] and pair["other_label"]
        assert (pair["test_survey_id"], pair["test_variable"]) != (
            pair["other_survey_id"],
            pair["other_variable"],
        )


def test_review_pairs_finds_a_high_cosine_pair():
    cosines = np.array(
        [[1.0, 0.99, 0.2], [0.99, 1.0, 0.1], [0.2, 0.1, 1.0]], dtype=np.float32
    )
    labels = [
        {"survey_id": "s", "variable": f"v{i}", "label": f"L{i}"} for i in range(3)
    ]
    pairs = sp.review_pairs(cosines, labels, [0], threshold=0.97)
    assert len(pairs) == 1
    assert pairs[0]["other_variable"] == "v1"
    assert pairs[0]["other_side"] == "train"


@split_frozen
def test_every_review_pair_is_signed_off(payload):
    """The lock on the >= 0.97 list: a human has to have read it.

    Last time the mechanism found the artefacts and nobody read the list
    before the freeze. So this test fails — loudly, and on purpose — while
    any pair in ``data/split/review_pairs_signed_off.json`` is unsigned. It
    is not a regression; it is the review that has not happened yet. Sign the
    file off and it goes green.
    """
    pairs = payload["review_pairs"]
    signed = sp.signed_off_pairs()
    missing = [
        sp.pair_signature(pair)
        for pair in pairs
        if sp.pair_signature(pair) not in signed
    ]
    assert not missing, (
        f"{len(missing)} of {len(pairs)} review pairs at cosine >= "
        f"{sp.REVIEW_COSINE} are not signed off in "
        f"{sp.REVIEW_SIGNOFF_PATH.name}: {missing}"
    )


@split_frozen
def test_the_signoff_file_covers_exactly_the_current_pairs(payload):
    """A sign-off from a previous draw must not silently cover a new pair."""
    assert sp.REVIEW_SIGNOFF_PATH.exists(), (
        "run scripts/14_build_split.py: it writes the sign-off template"
    )
    with open(sp.REVIEW_SIGNOFF_PATH, encoding="utf-8") as handle:
        stored = json.load(handle)
    listed = {sp.pair_signature(entry) for entry in stored["pairs"]}
    current = {sp.pair_signature(pair) for pair in payload["review_pairs"]}
    assert listed == current
    assert stored["threshold"] == payload["review_pairs_threshold"]


def test_signed_off_pairs_ignores_everything_but_an_explicit_true(tmp_path):
    path = tmp_path / "signoff.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"test_survey_id": "s", "test_variable": "a",
                     "other_survey_id": "s", "other_variable": "b",
                     "signed_off": True},
                    {"test_survey_id": "s", "test_variable": "c",
                     "other_survey_id": "s", "other_variable": "d",
                     "signed_off": "yes"},
                    {"test_survey_id": "s", "test_variable": "e",
                     "other_survey_id": "s", "other_variable": "f"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert sp.signed_off_pairs(path) == {"s/a::s/b"}
    assert sp.signed_off_pairs(tmp_path / "absent.json") == set()


def test_pair_signature_does_not_depend_on_the_side_or_the_cosine():
    left = {"test_survey_id": "s1", "test_variable": "a",
            "other_survey_id": "s2", "other_variable": "b", "cosine": 0.99}
    right = {"test_survey_id": "s2", "test_variable": "b",
             "other_survey_id": "s1", "other_variable": "a", "cosine": 0.97}
    assert sp.pair_signature(left) == sp.pair_signature(right)


# --------------------------------------------------------------------------
# context-only items (reported behaviour)
# --------------------------------------------------------------------------

@split_frozen
def test_no_test_item_is_a_reported_behaviour(frozen):
    """Turnout, donations, petitions: never a target, at test or training."""
    from article_silicon_sampling_quebec.corpus import perimeter

    for survey_id, variable in frozen.items:
        assert not perimeter.is_context_only(survey_id, variable), (
            f"{survey_id}/{variable} is reported behaviour, not an opinion"
        )


@split_frozen
def test_context_only_items_are_barred_from_training_targets(frozen, items):
    assert "is_context_only" in items.columns
    flagged = items.filter(pl.col("is_context_only"))
    assert flagged.height > 0
    targets = frozen.training_items(items)
    keys = set(zip(targets["survey_id"], targets["variable"], strict=True))
    for survey_id, variable in zip(
        flagged["survey_id"], flagged["variable"], strict=True
    ):
        assert (survey_id, variable) not in keys


@split_frozen
def test_context_only_items_are_still_retrievable_as_context(frozen, items):
    """They are barred as targets, not deleted: C1 must still reach them."""
    index = similarity.load_index(SIMILARITY_PATH)
    pool = frozen.context_pool(index.frame)
    reached = set(
        zip(pool["neighbor_survey_id"], pool["neighbor_variable"], strict=True)
    )
    flagged = set(
        zip(
            items.filter(pl.col("is_context_only"))["survey_id"],
            items.filter(pl.col("is_context_only"))["variable"],
            strict=True,
        )
    )
    assert reached & flagged


def test_is_context_only_matches_the_batteries_and_nothing_else():
    from article_silicon_sampling_quebec.corpus import perimeter

    assert perimeter.is_context_only("eeq_2008", "q11")            # did you vote
    assert perimeter.is_context_only("ces_2021", "pes21_partic3_4")  # battery
    assert perimeter.is_context_only("eeq_2018", "q31_4")          # battery
    # vote CHOICE and willingness stay targets — see the docstring
    assert not perimeter.is_context_only("eeq_2008", "q12a")
    assert not perimeter.is_context_only("eeq_2018", "q32_3")
    assert not perimeter.is_context_only("eeq_2018", "q20a")


# --------------------------------------------------------------------------
# upstream filters added in the second audit pass
# --------------------------------------------------------------------------

@split_frozen
def test_no_item_carries_an_unresolved_piping_placeholder(items):
    """A stem showing ${e://Field/x} was embedded on broken text."""
    from article_silicon_sampling_quebec.corpus import perimeter

    offenders = [
        f"{s}/{v}"
        for s, v, text in zip(
            items["survey_id"], items["variable"], items["question_text"],
            strict=True,
        )
        if perimeter.has_unresolved_piping(text)
    ]
    assert not offenders, offenders


def test_has_unresolved_piping_knows_the_four_placeholder_dialects():
    from article_silicon_sampling_quebec.corpus import perimeter

    for text in (
        "How satisfied are you with ${e://Field/premier}?",
        "How much should the government spend on [Field-justice_law]?",
        "How strongly [QID39-ChoiceTextEntryValue-7] do you feel?",
        "Vous sentez-vous très proche du <Q70>?",
    ):
        assert perimeter.has_unresolved_piping(text)
    assert not perimeter.has_unresolved_piping("Should abortion be banned?")
    assert not perimeter.has_unresolved_piping(None)
