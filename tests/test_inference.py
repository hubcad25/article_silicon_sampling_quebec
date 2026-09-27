from __future__ import annotations

from csv import DictReader
from dataclasses import replace
import json
from pathlib import Path

import pytest

from article_silicon_sampling_quebec.inference import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EXPECTED_ITEM_CELL_PAIRS,
    DrawTask,
    ItemCell,
    RunSettings,
    StallError,
    attach_stratum_context,
    build_item_cells,
    cell_distribution,
    ensure_manifest,
    load_checkpoint,
    make_tasks,
    run,
)
from article_silicon_sampling_quebec.prompts import ContextDistribution, ItemSpec, Option, Persona


class FakeChat:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.retries = 0
        self.throttled = 0
        self.calls = []

    def complete(self, messages, **settings):
        self.calls.append((messages, settings))
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def pair() -> ItemCell:
    item = ItemSpec(
        survey_id="survey", variable="q1", text="Choose one",
        options=(Option("1", "Yes"), Option("2", "No")), language="en", year=2020,
    )
    return ItemCell(
        item_idx=7, block="block", survey_id="survey", variable="q1", language="en",
        cell="25_34|woman", heldout_valid_n=31, item=item,
        persona=Persona(fields={"age": "25 to 34 years", "gender": "Woman"},
                        survey_id="survey", year=2020),
        dimensions=("age", "gender"),
    )


def manifest_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ("pilot_csv", "heldout_respondents", "items", "strata_definition"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    split = tmp_path / "split_manifest"
    split.write_text(json.dumps({"input_hashes": {"frozen": "hash"}}), encoding="utf-8")
    paths["split_manifest"] = split
    return paths


def test_frozen_pilot_has_exact_item_cell_coverage_and_persona_dimensions():
    pairs = build_item_cells()
    assert RunSettings().model == DEFAULT_MODEL
    assert len(pairs) == EXPECTED_ITEM_CELL_PAIRS
    assert len({p.item_idx for p in pairs}) == 12
    assert len(make_tasks(pairs, RunSettings())) == 110_000
    for p in pairs:
        assert tuple(p.persona.fields) == p.dimensions
        assert len(p.cell.split("|")) == len(p.dimensions)
        assert p.heldout_valid_n >= 30


def test_successful_invalid_completion_is_checkpointed_and_resume_skips_it(tmp_path):
    output = tmp_path / "run.csv"
    settings = RunSettings(temperatures=(0.7,), draws=2, workers=1,
                           limit_item_cell_pairs=1)
    first = FakeChat(["not an option", "Yes"])
    summary = run(settings, output, chat=first, pairs=[pair()],
                  manifest_paths=manifest_files(tmp_path))

    records = load_checkpoint(output.with_suffix(".jsonl"))
    assert len(records) == 2
    assert sum(record["valid"] for record in records.values()) == 1
    assert summary["effective_n"] == 1
    assert first.calls[0][1] == {"temperature": 0.7, "max_tokens": DEFAULT_MAX_TOKENS,
                                 "top_p": 1.0}
    system = first.calls[0][0][0]["content"]
    assert "Age : 25 to 34 years" in system and "Gender : Woman" in system
    assert all("context" not in message["content"].lower() for message in first.calls[0][0])
    assert records[next(iter(records))]["heldout_valid_n"] == 31

    resumed = FakeChat([])
    run(settings, output, chat=resumed, pairs=[pair()],
        manifest_paths=manifest_files(tmp_path))
    assert resumed.calls == []


def test_transport_failure_is_not_completed_and_is_retried_on_resume(tmp_path):
    output = tmp_path / "failed.csv"
    settings = RunSettings(temperatures=(1.0,), draws=1, workers=1,
                           limit_item_cell_pairs=1)
    paths = manifest_files(tmp_path)
    with pytest.raises(RuntimeError, match="remains pending"):
        run(settings, output, chat=FakeChat([OSError("down")]), pairs=[pair()],
            manifest_paths=paths)
    assert load_checkpoint(output.with_suffix(".jsonl")) == {}
    with output.with_suffix(".diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        diagnostics = list(DictReader(handle))
    assert diagnostics[0]["transport_n"] == "0"
    assert diagnostics[0]["coverage_ok"] == "False"

    summary = run(settings, output, chat=FakeChat(["No"]), pairs=[pair()],
                  manifest_paths=paths)
    assert summary["transport_n"] == summary["effective_n"] == 1


def test_concurrent_completions_are_persisted_once(tmp_path):
    output = tmp_path / "concurrent.csv"
    settings = RunSettings(temperatures=(0.3, 1.3), draws=4, workers=3,
                           limit_item_cell_pairs=1)
    chat = FakeChat(["Yes"] * 8)

    summary = run(settings, output, chat=chat, pairs=[pair()],
                  manifest_paths=manifest_files(tmp_path))
    records = load_checkpoint(output.with_suffix(".jsonl"))

    assert len(chat.calls) == 8
    assert len(records) == len(set(records)) == 8
    assert summary["tasks"] == summary["transport_n"] == summary["effective_n"] == 8


def test_checkpoint_repairs_partial_tail_and_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_bytes(b'{"draw_key":"a","valid":true}\n{"draw_key"')
    assert set(load_checkpoint(path)) == {"a"}
    assert path.read_bytes().endswith(b"\n")

    path.write_text('{"draw_key":"a"}\n{"draw_key":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_checkpoint(path)


def test_manifest_refuses_incompatible_resume(tmp_path):
    path = tmp_path / "manifest.json"
    base = {
        "schema_version": "1", "scientific_contract": "adr", "input_hashes": {},
        "frozen_split_input_hashes": {}, "settings": {"draws": 1}, "expected": {},
    }
    ensure_manifest(path, base)
    ensure_manifest(path, {**base, "created_at": "later"})
    with pytest.raises(ValueError, match="incompatible"):
        ensure_manifest(path, {**base, "settings": {"draws": 2}})


def test_draw_key_is_stable():
    task = DrawTask(pair(), 1.0, 4)
    assert task.key == "7|25_34|woman|1|4"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temperatures": (0.7, 0.7)}, "unique"),
        ({"temperatures": (float("nan"),)}, "finite"),
        ({"top_p": 0}, "top_p"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"arm": "C2"}, "arm"),
    ],
)
def test_settings_reject_values_that_make_calls_or_keys_unsafe(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RunSettings(**kwargs)


def neighbour() -> ItemSpec:
    return ItemSpec(
        survey_id="survey", variable="q2", text="Another question",
        options=(Option("1", "Agree"), Option("2", "Disagree")), language="en", year=2020,
    )


def test_arms_map_to_model_condition_and_context():
    assert (RunSettings(arm="A").condition, RunSettings(arm="A").context) == ("C0", "none")
    assert (RunSettings(arm="B").condition, RunSettings(arm="B").context) == ("C1", "stratum")
    assert (RunSettings(arm="B0").condition, RunSettings(arm="B0").context) == ("C1", "none")


def test_cell_distribution_is_weighted_and_ignores_invalid_codes():
    dist = cell_distribution(neighbour(), ["1", "2.0", None, "9"], [3.0, 1.0, 5.0, 5.0])
    assert dist.n == 2
    assert dist.shares == (("Agree", 0.75), ("Disagree", 0.25))
    assert cell_distribution(neighbour(), [None, "9"], [1.0, 1.0]) is None


def test_arm_b_renders_stratum_context_and_a_does_not(tmp_path):
    dist = ContextDistribution(item=neighbour(), shares=(("Agree", 0.75), ("Disagree", 0.25)),
                               n=40)
    with_context = replace(pair(), context=(dist,), context_cosines=(0.8,))
    for arm, expected in (("B", True), ("B0", False), ("A", False)):
        chat = FakeChat(["Yes"])
        settings = RunSettings(arm=arm, temperatures=(0.7,), draws=1, workers=1,
                               limit_item_cell_pairs=1)
        run(settings, tmp_path / f"{arm}.csv", chat=chat, pairs=[with_context],
            manifest_paths=manifest_files(tmp_path))
        user = chat.calls[0][0][-1]["content"]
        assert ("Another question : Agree 75 %, Disagree 25 % (n=40)" in user) is expected
        record = next(iter(load_checkpoint(tmp_path / f"{arm}.jsonl").values()))
        assert record["arm"] == arm and record["n_context"] == int(expected)
    assert (tmp_path / "B.context.csv").exists() and not (tmp_path / "A.context.csv").exists()


def test_stratum_context_uses_heldout_cell_and_never_the_target(tmp_path):
    pairs = build_item_cells()
    with_context = attach_stratum_context(pairs)
    assert [(p.item_idx, p.cell) for p in with_context] == [(p.item_idx, p.cell) for p in pairs]
    for p in with_context:
        assert len(p.context) <= 6 and len(p.context) == len(p.context_cosines)
        for dist, cosine in zip(p.context, p.context_cosines):
            assert dist.item.key != p.item.key and dist.item.survey_id == p.survey_id
            assert cosine < 0.95 and dist.n >= 10
            assert abs(sum(share for _, share in dist.shares) - 1) < 1e-9


def test_stalled_run_raises_instead_of_hanging(tmp_path):
    import threading

    gate = threading.Event()

    class HangingChat(FakeChat):
        def complete(self, messages, **settings):
            gate.wait(5)
            return "Yes"

    settings = RunSettings(temperatures=(0.7,), draws=1, workers=1, limit_item_cell_pairs=1)
    with pytest.raises(StallError):
        run(settings, tmp_path / "stall.csv", chat=HangingChat([]), pairs=[pair()],
            manifest_paths=manifest_files(tmp_path), stall_seconds=0.2, progress_every=0.05)
    gate.set()
    progress = json.loads((tmp_path / "stall.progress.json").read_text())
    assert progress["state"] == "failed" and "StallError" in progress["error"]


def test_arm_bs_uses_only_the_context_half(tmp_path):
    import polars as pl

    from article_silicon_sampling_quebec.inference import HELDOUT_HALVES_PATH, HELDOUT_PATH

    assert (RunSettings(arm="BS").condition, RunSettings(arm="BS").context) == ("C1", "stratum_half")
    halves = pl.read_csv(HELDOUT_HALVES_PATH, schema_overrides={"__respondent_id": pl.Utf8})
    held = pl.read_parquet(HELDOUT_PATH)
    assert halves.height == held.height and set(halves["half"]) == {"context", "eval"}
    per_cell = halves.group_by("__survey_id", "cell").agg(
        (pl.col("half") == "context").sum().alias("c"), pl.len().alias("n"))
    assert (per_cell["c"] == per_cell["n"] // 2).all()

    pairs = build_item_cells()[:20]
    all_eval = tmp_path / "halves.csv"
    halves.with_columns(pl.lit("eval").alias("half")).write_csv(all_eval)
    assert all(not p.context for p in attach_stratum_context(pairs, halves_path=all_eval))
    half = attach_stratum_context(pairs, halves_path=HELDOUT_HALVES_PATH)
    full = {(p.item_idx, p.cell, d.item.key): d.n for p in attach_stratum_context(pairs)
            for d in p.context}
    for p in half:
        for d in p.context:
            assert d.n <= full.get((p.item_idx, p.cell, d.item.key), d.n)
