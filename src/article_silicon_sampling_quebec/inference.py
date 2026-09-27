"""Resumable raw inference for the frozen ADR 0001 C0 pilot.

JSONL is the authoritative checkpoint: one successful transport call is one
fsynced line.  The CSV is an atomic, regenerated companion for analysis.  A
transport failure is never written, so its stable draw key remains pending on
resume; an unparseable final JSONL line is treated as an interrupted append and
removed, while corruption anywhere else is refused.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import polars as pl

from .corpus import blob, strata
from .corpus.ses import SesCrosswalk
from .foundry import FoundryChat
from .prompts import ItemSpec, Persona, PromptTemplate, build_item_specs
from .split import CELL_N_CONTRAST, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = REPO_ROOT / "data" / "analysis" / "test_blocks.csv"
ITEMS_PATH = REPO_ROOT / "data" / "items.parquet"
HELDOUT_PATH = REPO_ROOT / "data" / "split" / "heldout_respondents.parquet"
SPLIT_MANIFEST_PATH = REPO_ROOT / "data" / "split" / "split_manifest.json"
STRATA_PATH = REPO_ROOT / "data" / "strata_definition.json"

DEFAULT_TEMPERATURES = (0.3, 0.7, 1.0, 1.3)
DEFAULT_DRAWS = 100
DEFAULT_DEPLOYMENT = "c0-8k-txt"
DEFAULT_MODEL = "Llama-3.3-70B-Instruct-9.ft-0413559e867f42a59860f17f6153597b-c0-8k-txt"
DEFAULT_WORKERS = 3
# Measured with the repository's Llama-3 tokenizer: the longest of the 68
# rendered labels on the 12 pilot items is 19 tokens. 32 leaves headroom for
# tokenisation/service variation without inviting a long free-form response.
DEFAULT_MAX_TOKENS = 32
EXPECTED_PILOT_ITEMS = 12
EXPECTED_ITEM_CELL_PAIRS = 275

RESULT_FIELDS = (
    "draw_key", "deployment", "model", "condition", "temperature", "top_p",
    "max_tokens", "item_idx", "block", "survey_id", "variable", "language",
    "cell", "heldout_valid_n", "draw_idx", "raw_response", "matched_code", "valid",
    "started_at", "completed_at", "latency_seconds", "client_retries_total",
    "client_throttled_total",
)


@dataclass(frozen=True)
class ItemCell:
    item_idx: int
    block: str
    survey_id: str
    variable: str
    language: str
    cell: str
    heldout_valid_n: int
    item: ItemSpec
    persona: Persona
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class DrawTask:
    pair: ItemCell
    temperature: float
    draw_idx: int

    @property
    def key(self) -> str:
        p = self.pair
        temperature = format(self.temperature, ".12g")
        return f"{p.item_idx}|{p.cell}|{temperature}|{self.draw_idx}"


@dataclass(frozen=True)
class RunSettings:
    deployment: str = DEFAULT_DEPLOYMENT
    model: str = DEFAULT_MODEL
    condition: str = "C0"
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    draws: int = DEFAULT_DRAWS
    top_p: float = 1.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    workers: int = DEFAULT_WORKERS
    limit_item_cell_pairs: int | None = None

    def __post_init__(self) -> None:
        if self.condition != "C0":
            raise ValueError("this harness only supports condition C0")
        if not self.temperatures or self.draws < 1 or self.workers < 1:
            raise ValueError("temperatures, draws, and workers must be non-empty/positive")
        if any(not math.isfinite(value) or value < 0 for value in self.temperatures):
            raise ValueError("temperatures must be finite and non-negative")
        if len(set(self.temperatures)) != len(self.temperatures):
            raise ValueError("temperatures must be unique")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not self.deployment or not self.model:
            raise ValueError("deployment and model must be non-empty")
        if self.limit_item_cell_pairs is not None and self.limit_item_cell_pairs < 1:
            raise ValueError("limit_item_cell_pairs must be positive")

    @property
    def full_mode(self) -> bool:
        return (
            self.temperatures == DEFAULT_TEMPERATURES
            and self.draws == DEFAULT_DRAWS
            and self.limit_item_cell_pairs is None
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pilot_rows(path: Path) -> list[dict[str, Any]]:
    frame = pl.read_csv(path)
    required = {"item_idx", "block", "pilot", "survey_id", "variable", "language",
                "cells_ge30"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pilot CSV is missing columns: {sorted(missing)}")
    rows = frame.filter(pl.col("pilot") == True).sort("item_idx").to_dicts()  # noqa: E712
    if len(rows) != EXPECTED_PILOT_ITEMS:
        raise ValueError(f"expected exactly {EXPECTED_PILOT_ITEMS} pilot items, got {len(rows)}")
    if len({(r["survey_id"], r["variable"]) for r in rows}) != len(rows):
        raise ValueError("pilot CSV contains duplicate item keys")
    return rows


def build_item_cells(
    pilot_path: Path = PILOT_PATH,
    items_path: Path = ITEMS_PATH,
    heldout_path: Path = HELDOUT_PATH,
    *,
    survey_loader: Callable[[str, list[str] | None], pl.DataFrame] = blob.read_survey,
    definition: strata.StrataDefinition | None = None,
    crosswalk: SesCrosswalk | None = None,
) -> list[ItemCell]:
    """Build and gate the 275 frozen item-cell pairs, without network calls."""
    pilot = _pilot_rows(Path(pilot_path))
    items = pl.read_parquet(items_path)
    specs = build_item_specs(items.iter_rows(named=True))
    heldout = pl.read_parquet(heldout_path).with_columns(
        pl.col("__respondent_id").cast(pl.Utf8)
    )
    definition = definition or strata.load_definition()
    crosswalk = crosswalk or SesCrosswalk.load()
    pairs: list[ItemCell] = []

    for survey_id in sorted({str(r["survey_id"]) for r in pilot}):
        survey_rows = [r for r in pilot if r["survey_id"] == survey_id]
        variables = [str(r["variable"]) for r in survey_rows]
        micro = survey_loader(survey_id, ["__respondent_id", *variables]).with_columns(
            pl.col("__respondent_id").cast(pl.Utf8)
        )
        frozen = heldout.filter(pl.col("__survey_id") == survey_id).select(
            "__respondent_id", "cell"
        )
        joined = frozen.join(micro, on="__respondent_id", how="inner")
        dims = definition.dimensions_for(survey_id)

        for row in survey_rows:
            key = (survey_id, str(row["variable"]))
            if key not in specs:
                raise ValueError(f"pilot item absent from items.parquet: {key}")
            item = specs[key]
            if item.language != row["language"]:
                raise ValueError(f"language mismatch for {key}: {row['language']} vs {item.language}")
            counts = (
                joined.group_by("cell")
                .agg(pl.col(key[1]).count().alias("n_valid"))
                .filter(pl.col("n_valid") >= CELL_N_CONTRAST)
                .sort("cell")
            )
            expected = int(row["cells_ge30"])
            if counts.height != expected:
                raise ValueError(
                    f"frozen cell count changed for {key}: expected {expected}, got {counts.height}"
                )
            for count_row in counts.iter_rows(named=True):
                cell = count_row["cell"]
                levels = str(cell).split("|")
                if len(levels) != len(dims):
                    raise ValueError(f"cell {cell!r} does not encode dimensions {dims} for {key}")
                lang = item.language if item.language in {"fr", "en"} else "fr"
                fields = {
                    dim: crosswalk.label(dim, level, lang)
                    for dim, level in zip(dims, levels, strict=True)
                }
                pairs.append(ItemCell(
                    item_idx=int(row["item_idx"]), block=str(row["block"]),
                    survey_id=survey_id, variable=key[1], language=item.language,
                    cell=str(cell), heldout_valid_n=int(count_row["n_valid"]), item=item,
                    persona=Persona(fields=fields, survey_id=survey_id, year=item.year),
                    dimensions=tuple(dims),
                ))

    pairs.sort(key=lambda p: (p.item_idx, p.cell))
    if len(pairs) != EXPECTED_ITEM_CELL_PAIRS:
        raise ValueError(
            f"expected exactly {EXPECTED_ITEM_CELL_PAIRS} pilot item-cell pairs, got {len(pairs)}"
        )
    return pairs


def make_tasks(pairs: Sequence[ItemCell], settings: RunSettings) -> list[DrawTask]:
    selected = list(pairs[:settings.limit_item_cell_pairs])
    return [
        DrawTask(pair, temperature, draw_idx)
        for pair in selected
        for temperature in settings.temperatures
        for draw_idx in range(settings.draws)
    ]


def _jsonl_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".jsonl")


def _manifest_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".manifest.json")


def _diagnostics_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".diagnostics.csv")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(settings: RunSettings, n_pairs: int, paths: Mapping[str, Path] | None = None,
                   ) -> dict[str, Any]:
    paths = paths or {
        "pilot_csv": PILOT_PATH,
        "heldout_respondents": HELDOUT_PATH,
        "split_manifest": SPLIT_MANIFEST_PATH,
        "items": ITEMS_PATH,
        "strata_definition": STRATA_PATH,
    }
    split_manifest = json.loads(Path(paths["split_manifest"]).read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "scientific_contract": "docs/adr/0001-sous-ensemble-pilote-blocs-thematiques.md",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "frozen_split_input_hashes": split_manifest.get("input_hashes", {}),
        "settings": asdict(settings),
        "expected": {
            "pilot_items": EXPECTED_PILOT_ITEMS,
            "frozen_item_cell_pairs": EXPECTED_ITEM_CELL_PAIRS,
            "selected_item_cell_pairs": n_pairs,
            "tasks": n_pairs * len(settings.temperatures) * settings.draws,
        },
    }


def _compatible_manifest(existing: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    keys = ("schema_version", "scientific_contract", "input_hashes",
            "frozen_split_input_hashes", "settings", "expected")
    # The in-memory settings contain tuples; JSON necessarily reads them back as lists.
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    return all(canonical(existing.get(key)) == canonical(wanted.get(key)) for key in keys)


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not _compatible_manifest(existing, manifest):
            raise ValueError(f"refusing incompatible resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, manifest)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Read unique records; repair only a partial final append."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: dict[str, dict[str, Any]] = {}
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index != len(lines) - 1:
                raise ValueError(f"corrupt checkpoint line {index + 1}: {path}") from exc
            break
        key = record.get("draw_key")
        if not key or key in records:
            raise ValueError(f"missing or duplicate draw key on line {index + 1}: {key!r}")
        records[key] = record
        valid_bytes += len(line)
    if valid_bytes != len(raw):
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    elif raw and not raw.endswith(b"\n"):
        # A complete JSON object can reach disk just before its newline.  Keep
        # the valid record and restore the delimiter before the next append.
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def append_checkpoint(handle, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_csv_companion(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = sorted(records, key=lambda r: str(r["draw_key"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp = Path(handle.name)
    temp.replace(path)


def write_diagnostics(path: Path, tasks: Sequence[DrawTask],
                      records: Mapping[str, Mapping[str, Any]]) -> None:
    groups: dict[tuple[int, str, float], list[Mapping[str, Any]]] = {}
    for task in tasks:
        key = (task.pair.item_idx, task.pair.cell, task.temperature)
        groups.setdefault(key, [])
        if task.key in records:
            groups[key].append(records[task.key])
    rows = []
    expected_per_group = len({t.draw_idx for t in tasks}) if tasks else 0
    for (item_idx, cell, temperature), group in sorted(groups.items()):
        valid = sum(bool(r["valid"]) for r in group)
        rows.append({
            "item_idx": item_idx, "cell": cell, "temperature": temperature,
            "transport_n": len(group), "effective_n": valid,
            "invalid_n": len(group) - valid,
            "invalid_rate": ((len(group) - valid) / len(group)) if group else None,
            "expected_transport_n": expected_per_group,
            "coverage_ok": len(group) == expected_per_group,
        })
    fields = ("item_idx", "cell", "temperature", "transport_n", "effective_n",
              "invalid_n", "invalid_rate", "expected_transport_n", "coverage_ok")
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temp = Path(handle.name)
    temp.replace(path)


def validate_coverage(tasks: Sequence[DrawTask], records: Mapping[str, Mapping[str, Any]],
                      *, full_mode: bool) -> None:
    expected = {task.key for task in tasks}
    actual = set(records)
    if actual != expected:
        raise RuntimeError(
            f"task coverage mismatch: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )
    if full_mode:
        counts: dict[tuple[int, str, float], int] = {}
        for task in tasks:
            key = (task.pair.item_idx, task.pair.cell, task.temperature)
            counts[key] = counts.get(key, 0) + 1
        bad = {key: n for key, n in counts.items() if n != DEFAULT_DRAWS}
        if len(counts) != EXPECTED_ITEM_CELL_PAIRS * len(DEFAULT_TEMPERATURES) or bad:
            raise RuntimeError(f"full-mode coverage gate failed: {bad}")


def _complete(chat: FoundryChat, template: PromptTemplate, settings: RunSettings,
              task: DrawTask) -> dict[str, Any]:
    started = utc_now()
    before = time.monotonic()
    pair = task.pair
    messages = template.build_messages(
        pair.persona, pair.item, dimensions=pair.dimensions
    )
    raw = chat.complete(messages, temperature=task.temperature,
                        max_tokens=settings.max_tokens, top_p=settings.top_p)
    matched = pair.item.match_answer(raw)
    return {
        "draw_key": task.key, "deployment": settings.deployment,
        "model": settings.model, "condition": settings.condition,
        "temperature": task.temperature, "top_p": settings.top_p,
        "max_tokens": settings.max_tokens, "item_idx": pair.item_idx,
        "block": pair.block, "survey_id": pair.survey_id,
        "variable": pair.variable, "language": pair.language, "cell": pair.cell,
        "heldout_valid_n": pair.heldout_valid_n,
        "draw_idx": task.draw_idx, "raw_response": raw,
        "matched_code": matched, "valid": matched is not None,
        "started_at": started, "completed_at": utc_now(),
        "latency_seconds": round(time.monotonic() - before, 6),
        "client_retries_total": chat.retries,
        "client_throttled_total": chat.throttled,
    }


def run(settings: RunSettings, output_csv: Path, *,
        chat: FoundryChat | None = None, pairs: Sequence[ItemCell] | None = None,
        manifest_paths: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """Run pending draws, stopping after draining in-flight work on failure."""
    output_csv = Path(output_csv)
    if output_csv.suffix.lower() != ".csv":
        raise ValueError("output path must end in .csv")
    all_pairs = list(pairs) if pairs is not None else build_item_cells()
    selected_pairs = all_pairs[:settings.limit_item_cell_pairs]
    tasks = make_tasks(all_pairs, settings)
    manifest = build_manifest(settings, len(selected_pairs), manifest_paths)
    manifest_path = _manifest_path(output_csv)
    checkpoint_path = _jsonl_path(output_csv)
    diagnostics_path = _diagnostics_path(output_csv)
    ensure_manifest(manifest_path, manifest)
    records = load_checkpoint(checkpoint_path)
    expected_keys = {task.key for task in tasks}
    unexpected = set(records) - expected_keys
    if unexpected:
        raise ValueError(f"checkpoint has {len(unexpected)} keys outside this run")
    pending = [task for task in tasks if task.key not in records]
    if pending and chat is None:
        chat = FoundryChat(settings.deployment)
    template = PromptTemplate(condition="C0", ses_dropout="none")
    first_error: BaseException | None = None

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                iterator = iter(pending)
                active: dict[Future, DrawTask] = {}
                for _ in range(settings.workers):
                    task = next(iterator, None)
                    if task is not None:
                        active[pool.submit(_complete, chat, template, settings, task)] = task
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = active.pop(future)
                        try:
                            record = future.result()
                        except Exception as exc:
                            first_error = first_error or exc
                            continue
                        if record["draw_key"] in records:
                            raise RuntimeError(f"duplicate completed draw key: {task.key}")
                        append_checkpoint(handle, record)
                        records[task.key] = record
                    if first_error is None:
                        for _ in done:
                            replacement = next(iterator, None)
                            if replacement is not None:
                                active[pool.submit(
                                    _complete, chat, template, settings, replacement
                                )] = replacement
    finally:
        write_csv_companion(output_csv, records.values())
        write_diagnostics(diagnostics_path, tasks, records)

    if first_error is not None:
        raise RuntimeError("transport failure; failed draw remains pending for resume") from first_error
    validate_coverage(tasks, records, full_mode=settings.full_mode)
    valid = sum(bool(record["valid"]) for record in records.values())
    return {
        "tasks": len(tasks), "transport_n": len(records), "effective_n": valid,
        "invalid_n": len(records) - valid,
        "invalid_rate": (len(records) - valid) / len(records) if records else 0.0,
        "checkpoint": str(checkpoint_path), "csv": str(output_csv),
        "manifest": str(manifest_path), "diagnostics": str(diagnostics_path),
    }


__all__ = [
    "DEFAULT_DEPLOYMENT", "DEFAULT_DRAWS", "DEFAULT_MAX_TOKENS", "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURES", "DEFAULT_WORKERS", "DrawTask",
    "EXPECTED_ITEM_CELL_PAIRS", "EXPECTED_PILOT_ITEMS", "ItemCell", "RunSettings",
    "append_checkpoint", "build_item_cells", "build_manifest", "ensure_manifest",
    "load_checkpoint", "make_tasks", "run", "validate_coverage",
    "write_csv_companion", "write_diagnostics",
]
