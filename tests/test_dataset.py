"""Tests for the dataset generator (step 3.2). No network access.

Two halves.

*Unit* tests pin the pure logic: the water-filling allocation, the theme
clustering, the respondent draw, the context assembly.

*Data* tests re-open the JSONL files that ``scripts/16_generate_dataset.py``
actually wrote and re-derive every exclusion rule from the frozen split and
the similarity index. They deliberately do **not** call the generator: a rule
enforced only by the code that produced the file is a rule nobody checked.
They skip when ``data/datasets/`` has not been generated yet.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from article_silicon_sampling_quebec import dataset as ds
from article_silicon_sampling_quebec import split as sp
from article_silicon_sampling_quebec.corpus import similarity
from article_silicon_sampling_quebec.prompts import (
    CONTEXT_MAX_COSINE,
    ItemSpec,
    Option,
    Persona,
    PromptTemplate,
)

ITEMS_PATH = sp._REPO / "data" / "items.parquet"
DATASETS = ds.DATASET_DIR

generated = pytest.mark.skipif(
    not (DATASETS / "manifest.json").exists(),
    reason="run scripts/16_generate_dataset.py first",
)


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------


def test_equal_allocation_is_equal_when_capacity_allows():
    got = ds.equal_allocation({"a": 100, "b": 100, "c": 100}, 30)
    assert got == {"a": 10, "b": 10, "c": 10}


def test_equal_allocation_never_exceeds_capacity_and_refills_siblings():
    got = ds.equal_allocation({"small": 2, "big": 1000, "mid": 20}, 100)
    assert got["small"] == 2
    assert sum(got.values()) == 100
    assert all(got[k] <= cap for k, cap in {"small": 2, "big": 1000, "mid": 20}.items())
    # what the small group could not absorb went to the others, not lost
    assert got["big"] > got["mid"] >= 20


def test_equal_allocation_caps_at_total_capacity():
    got = ds.equal_allocation({"a": 3, "b": 4}, 1000)
    assert got == {"a": 3, "b": 4}


def test_equal_allocation_is_deterministic():
    caps = {f"g{i}": (i * 7) % 23 for i in range(40)}
    assert ds.equal_allocation(caps, 137) == ds.equal_allocation(caps, 137)


def test_allocate_pairs_balances_the_levels_before_the_items():
    # One survey with many items, one with few: equal shares per survey, then
    # the small survey is capped by its own capacity.
    frame = pl.DataFrame({
        "survey_id": ["big"] * 10 + ["small"] * 2,
        "variable": [f"v{i}" for i in range(10)] + ["a", "b"],
        "language": ["fr"] * 12,
        "theme": ["t0"] * 12,
    })
    capacity = {("big", f"v{i}"): 100 for i in range(10)}
    capacity |= {("small", "a"): 10, ("small", "b"): 10}
    quotas = ds.allocate_pairs(frame, capacity, 120, levels=("language", "survey_id"))
    assert sum(quotas.values()) == 120
    assert sum(v for k, v in quotas.items() if k[0] == "small") == 20
    assert sum(v for k, v in quotas.items() if k[0] == "big") == 100
    assert all(quotas[k] <= capacity[k] for k in quotas)


def test_allocate_pairs_keeps_the_two_languages_even():
    frame = pl.DataFrame({
        "survey_id": ["a"] * 4 + ["b"] * 4,
        "variable": [f"v{i}" for i in range(8)],
        "language": ["fr"] * 4 + ["en"] * 4,
        "theme": ["t0"] * 8,
    })
    capacity = {(r["survey_id"], r["variable"]): 500 for r in frame.iter_rows(named=True)}
    quotas = ds.allocate_pairs(frame, capacity, 200)
    per_lang = {"fr": 0, "en": 0}
    for row in frame.iter_rows(named=True):
        per_lang[row["language"]] += quotas.get((row["survey_id"], row["variable"]), 0)
    assert per_lang == {"fr": 100, "en": 100}


# --------------------------------------------------------------------------
# theme proxy
# --------------------------------------------------------------------------


def test_spherical_kmeans_is_deterministic_and_separates_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, size=(30, 8)) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
    b = rng.normal(0, 0.01, size=(30, 8)) + np.array([0, 1, 0, 0, 0, 0, 0, 0])
    matrix = np.vstack([a, b])
    first = ds.spherical_kmeans(matrix, 2, seed=1)
    assert np.array_equal(first, ds.spherical_kmeans(matrix, 2, seed=1))
    assert len(set(first[:30])) == 1 and len(set(first[30:])) == 1
    assert first[0] != first[30]


def test_spherical_kmeans_never_returns_an_empty_cluster():
    rng = np.random.default_rng(3)
    labels = ds.spherical_kmeans(rng.normal(size=(60, 16)), 5, seed=7)
    assert set(labels) == set(range(5))


def test_stable_seed_survives_a_new_interpreter_with_another_hash_salt():
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from article_silicon_sampling_quebec.dataset import _stable_seed;"
        "print(_stable_seed(20260923, 'ces_2025', 'q1'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=sp._REPO, capture_output=True, text=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == ds._stable_seed(20260923, "ces_2025", "q1")


# --------------------------------------------------------------------------
# sampling and context, on a synthetic panel
# --------------------------------------------------------------------------


class FakePanel:
    """Minimal stand-in for ``SurveyPanel``: three items, six respondents."""

    def __init__(self):
        self.rids = [str(i) for i in range(6)]
        self.answers = {
            "target": ["1", "2", "1", None, "2", "1"],
            "near": ["1", None, "2", "1", "1", "2"],
            "twin": ["2", "2", "2", "2", "2", "2"],
        }
        self.training_rows = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # 5 is held out

    def eligible_rows(self, item):
        offered = {o.code for o in item.options}
        return np.array(
            [i for i in self.training_rows
             if self.answers[item.variable][i] in offered], dtype=np.int64
        )

    def answer(self, row, item):
        return item.canonical_code(self.answers[item.variable][row])

    def persona(self, row, language):
        return Persona(fields={"age": "35-44"}, survey_id="s")


def _spec(variable, text, short=None):
    return ItemSpec(
        survey_id="s", variable=variable, text=text,
        options=(Option("1", "Yes"), Option("2", "No")),
        language="fr", short_text=short,
    )


@pytest.fixture
def fake():
    specs = {
        ("s", "target"): _spec("target", "Question cible"),
        ("s", "near"): _spec("near", "Un voisin", short="Un voisin"),
        # same rendered label as the target: the catalogue defect of ces_2021
        ("s", "twin"): _spec("twin", "Autre chose", short="Question cible"),
    }
    return FakePanel(), specs


def test_sample_pairs_draws_distinct_respondents_and_skips_non_answers(fake):
    panel, specs = fake
    pairs = ds.sample_pairs({("s", "target"): 10}, {"s": panel}, specs,
                            {("s", "target"): "t0"}, seed=1)
    rids = [p.respondent_id for p in pairs]
    assert len(rids) == len(set(rids))
    # respondent 3 never answered the target, respondent 5 is held out
    assert set(rids) == {"0", "1", "2", "4"}
    assert all(p.answer_code in {"1", "2"} for p in pairs)


def test_sample_pairs_is_reproducible_and_respects_the_quota(fake):
    panel, specs = fake
    args = ({("s", "target"): 2}, {"s": panel}, specs, {("s", "target"): "t0"})
    first = ds.sample_pairs(*args, seed=4)
    assert len(first) == 2
    assert [p.respondent_id for p in first] == [
        p.respondent_id for p in ds.sample_pairs(*args, seed=4)
    ]


def test_context_answers_drops_the_target_and_duplicate_renderings(fake):
    panel, specs = fake
    pair = ds.Pair(
        survey_id="s", variable="target", respondent_id="0", row=0,
        language="fr", theme="t0", answer_code="1",
        context=((("s", "twin"), 0.93), (("s", "near"), 0.80)),
    )
    got = ds.context_answers(pair, panel, specs)
    # "twin" renders byte-identically to the target's wording -> dropped
    assert [c.item.variable for c in got] == ["near"]
    assert got[0].label == "Yes"


def test_context_answers_skips_items_the_respondent_did_not_answer(fake):
    panel, specs = fake
    pair = ds.Pair(
        survey_id="s", variable="target", respondent_id="1", row=1,
        language="fr", theme="t0", answer_code="2",
        context=((("s", "near"), 0.80),),
    )
    assert ds.context_answers(pair, panel, specs) == []


def test_render_shares_the_ses_dropout_across_conditions(fake):
    panel, specs = fake
    templates = {c: PromptTemplate(condition=c, k=6) for c in ("C0", "C1")}
    pair = ds.Pair(
        survey_id="s", variable="target", respondent_id="0", row=0,
        language="fr", theme="t0", answer_code="1",
        context=((("s", "near"), 0.80),),
    )
    c0 = ds.render(pair, "C0", panel, specs, templates, seed=9)
    c1 = ds.render(pair, "C1", panel, specs, templates, seed=9)
    assert c0["example"]["messages"][0] == c1["example"]["messages"][0]
    assert c0["n_context"] == 0 and c1["n_context"] == 1
    # the assistant turn is the option's text, not its code
    assert c0["example"]["messages"][2]["content"] == "Yes"
    assert c1["example"]["messages"][2]["content"] == "Yes"


def test_audit_catches_a_planted_leak(fake):
    panel, specs = fake
    split = sp.Split(items=frozenset({("s", "near")}), respondents={},
                     manifest={})
    meta = pl.DataFrame([{
        "survey_id": "s", "variable": "target", "respondent_id": "0",
        "answer_code": "1",
        "context_keys": json.dumps([["s", "near"]]),
        "context_used": json.dumps([["s", "near"]]),
        "context_cosines": json.dumps([0.99]),
    }])
    example = {"messages": [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "Tes réponses :\n- Un voisin → Yes\n\nQuestion : x"},
        {"role": "assistant", "content": "1"},
    ]}
    problems = ds.audit_examples([example], meta, split, specs, "C1")
    assert problems["context_above_max_cosine"]
    assert problems["context_is_test_item"]
    # a code where the option text is expected, and options listed with codes
    assert problems["answer_not_an_option"]
    assert problems["answer_does_not_match_code"]
    assert problems["target_option_has_code"]


def test_audit_accepts_a_clean_text_answer(fake):
    panel, specs = fake
    templates = {"C0": PromptTemplate(condition="C0", k=6)}
    pair = ds.Pair(survey_id="s", variable="target", respondent_id="0", row=0,
                   language="fr", theme="t0", answer_code="2")
    rec = ds.render(pair, "C0", panel, specs, templates, seed=9)
    meta = pl.DataFrame([{
        "survey_id": "s", "variable": "target", "respondent_id": "0",
        "answer_code": "2", "context_keys": "[]", "context_used": "[]",
        "context_cosines": "[]",
    }])
    split = sp.Split(items=frozenset(), respondents={}, manifest={})
    problems = ds.audit_examples([rec["example"]], meta, split, specs, "C0")
    assert not any(problems.values()), problems
    assert rec["example"]["messages"][2]["content"] == "No"


# --------------------------------------------------------------------------
# the generated files
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(DATASETS / "manifest.json", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def meta() -> pl.DataFrame:
    return pl.read_parquet(DATASETS / "pairs.parquet")


@pytest.fixture(scope="module")
def items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS_PATH)


@pytest.fixture(scope="module")
def frozen() -> sp.Split:
    return sp.load_split()


@pytest.fixture(scope="module")
def specs(items) -> dict:
    return {
        (r["survey_id"], r["variable"]): ItemSpec.from_row(r)
        for r in items.iter_rows(named=True)
    }


def _lines(name: str) -> list[dict]:
    return ds.read_jsonl(DATASETS / name)


@generated
def test_every_file_is_valid_chat_jsonl(manifest):
    for name, expected in manifest["files"].items():
        rows = _lines(name)
        assert len(rows) == expected, name
        for row in rows[:200]:
            roles = [m["role"] for m in row["messages"]]
            assert roles == ["system", "user", "assistant"], name
            assert set(row) == {"messages"}
            assert all(m["content"].strip() for m in row["messages"])


@generated
def test_targets_are_training_targets_only(manifest, meta, items, frozen):
    training = frozen.training_items(items)
    allowed = set(zip(training["survey_id"], training["variable"], strict=True))
    drawn = set(zip(meta["survey_id"], meta["variable"], strict=True))
    assert drawn <= allowed
    # explicit restatement of the two exclusions the plan names
    assert not drawn & set(frozen.items)
    context_only = {
        (r["survey_id"], r["variable"]) for r in items.iter_rows(named=True)
        if r["is_context_only"]
    }
    assert not drawn & context_only


@generated
def test_no_held_out_respondent_ever_appears(meta, frozen):
    for survey, rid in zip(meta["survey_id"], meta["respondent_id"], strict=True):
        assert not frozen.is_test_respondent(survey, rid)


@generated
def test_context_never_leaks_the_target(meta, frozen):
    """Re-derived from the produced rows, not from nearest_context_items."""
    for row in meta.iter_rows(named=True):
        key = (row["survey_id"], row["variable"])
        used = [tuple(k) for k in json.loads(row["context_used"])]
        assert key not in used
        assert not any(frozen.is_test_item(k) for k in used)
        assert len(used) == len(set(used))


@generated
def test_no_context_item_sits_at_or_above_the_anti_leak_cut(meta):
    index = similarity.load_index(sp._REPO / "data" / "item_similarity.parquet")
    checked = 0
    for key in {(r["survey_id"], r["variable"]) for r in meta.iter_rows(named=True)}:
        frame = index.neighbors(key, k=None)
        cos = dict(zip(
            list(zip(frame["neighbor_survey_id"], frame["neighbor_variable"],
                     strict=True)),
            frame["cosine"].to_list(), strict=True,
        ))
        rows = meta.filter((pl.col("survey_id") == key[0])
                           & (pl.col("variable") == key[1]))
        for row in rows.iter_rows(named=True):
            for nkey in [tuple(k) for k in json.loads(row["context_used"])]:
                assert cos.get(nkey, 0.0) < CONTEXT_MAX_COSINE, (key, nkey)
                checked += 1
    assert checked > 10_000


@generated
def test_target_answer_is_always_one_of_the_rendered_options(meta, specs):
    rows = _lines("c1_train_20000.jsonl")
    for example, row in zip(rows, meta.slice(ds.VALIDATION_SIZE).to_dicts(),
                            strict=False):
        item = specs[(row["survey_id"], row["variable"])]
        answer = example["messages"][2]["content"]
        listed = example["messages"][1]["content"].rsplit("\nOptions :\n", 1)[1]
        assert answer in {o.text for o in item.options}
        assert f"\n- {answer}\n" in f"\n{listed}\n"
        assert answer == row["answer_label"]
        assert item.match_answer(answer) == row["answer_code"]


@generated
def test_no_target_option_carries_a_code():
    import re
    coded = re.compile(r"^-?\d+\) ")
    for name in ("c0_validation.jsonl", "c1_train_20000.jsonl"):
        for example in _lines(name):
            listed = (example["messages"][1]["content"]
                      .rsplit("\nOptions :\n", 1)[1].split("\n\n")[0])
            lines = listed.split("\n")
            assert all(l.startswith("- ") for l in lines), name
            assert not any(coded.match(l) for l in lines), name


@generated
def test_pairs_csv_is_line_aligned_with_the_jsonl(meta):
    csv_meta = pl.read_csv(DATASETS / "pairs.csv", infer_schema_length=0)
    assert csv_meta.height == meta.height
    for col in ("split", "row_index", "survey_id", "variable", "language",
                "theme", "respondent_id", "answer_code", "answer_label",
                "n_context", "n_ses_fields", "max_context_cosine"):
        assert col in csv_meta.columns, col
    val = csv_meta.filter(pl.col("split") == "validation")
    train = csv_meta.filter(pl.col("split") == "train")
    assert val.height == ds.VALIDATION_SIZE
    assert val["row_index"].to_list() == [str(i) for i in range(val.height)]
    assert train["row_index"].to_list() == [str(i) for i in range(train.height)]
    for name, frame in (("c1_validation.jsonl", val),
                        ("c0_train_8000.jsonl", train)):
        for example, label in zip(_lines(name), frame["answer_label"].to_list(),
                                  strict=False):
            assert example["messages"][2]["content"] == label


@generated
def test_c0_carries_no_context_block():
    for row in _lines("c0_train_20000.jsonl"):
        user = row["messages"][1]["content"]
        assert user.startswith("Question : ")
        # the only "- " lines are the target's options
        assert "\n- " not in user.split("\nOptions :\n")[0]


@generated
def test_the_8k_file_is_the_prefix_of_the_20k_file():
    for cond in ("c0", "c1"):
        small = (DATASETS / f"{cond}_train_8000.jsonl").read_bytes()
        big = (DATASETS / f"{cond}_train_20000.jsonl").read_bytes()
        assert big.startswith(small)


@generated
def test_validation_is_disjoint_from_training(meta):
    val = set(map(tuple, meta.head(ds.VALIDATION_SIZE)
                  .select("survey_id", "variable", "respondent_id").rows()))
    train = set(map(tuple, meta.slice(ds.VALIDATION_SIZE)
                    .select("survey_id", "variable", "respondent_id").rows()))
    assert val and not (val & train)


@generated
def test_no_pair_is_drawn_twice(meta):
    assert meta.select("survey_id", "variable", "respondent_id").unique().height \
        == meta.height


@generated
def test_c0_and_c1_render_the_same_pairs_in_the_same_order():
    c0 = _lines("c0_train_8000.jsonl")
    c1 = _lines("c1_train_8000.jsonl")
    assert len(c0) == len(c1)
    for a, b in zip(c0, c1, strict=True):
        assert a["messages"][0] == b["messages"][0]       # same persona, same dropout
        assert a["messages"][2] == b["messages"][2]       # same target answer
        # C1's user turn is C0's, preceded by the context block
        assert b["messages"][1]["content"].endswith(a["messages"][1]["content"])


@generated
def test_the_sample_is_balanced_not_uniform(meta):
    """Uniform sampling would put >70 % of the pairs in the three big CES."""
    shares = (meta.group_by("survey_id").len()
              .with_columns(share=pl.col("len") / meta.height))
    assert shares["share"].max() < 0.12
    langs = meta.group_by("language").len()
    assert abs(langs["len"][0] - langs["len"][1]) / meta.height < 0.02
    assert meta["theme"].n_unique() >= 8


@generated
def test_the_files_still_hash_to_the_manifest(manifest):
    for name, digest in manifest["output_hashes"].items():
        assert sp.sha256_file(DATASETS / name) == digest, name


@generated
def test_the_manifest_pins_the_frozen_split_inputs(manifest):
    with open(sp.MANIFEST_PATH, encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    for path, digest in split_manifest["input_hashes"].items():
        if path in manifest["input_hashes"]:
            assert manifest["input_hashes"][path] == digest, path
