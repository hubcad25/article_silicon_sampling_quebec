from __future__ import annotations

from csv import DictReader
import json
from pathlib import Path

import pytest

from article_silicon_sampling_quebec.c0_inference import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EXPECTED_ITEM_CELL_PAIRS,
    DrawTask,
    ItemCell,
    RunSettings,
    build_item_cells,
    ensure_manifest,
    load_checkpoint,
    make_tasks,
    run,
)
from article_silicon_sampling_quebec.prompts import ItemSpec, Option, Persona


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
    ],
)
def test_settings_reject_values_that_make_calls_or_keys_unsafe(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RunSettings(**kwargs)
