"""Resumable raw inference for the frozen ADR 0001 pilot, one arm per output.

Arms (ADR 0001, « Stratégie d'inférence ») — same 275 item x cell pairs, same
draw keys, temperatures and draws; only the model and the context block differ:

====  ==========  ==================================================================
A     C0 model    persona only
B     C1 model    persona + the cell's observed distributions on the k nearest items
B0    C1 model    persona only (separates the fine-tune effect from the context one)
R     base model  persona only — the roleplay reference, same prompt, no fine-tune
BS    C1 model    as B, but distributions from the *context half* of the held-out
                  respondents only; evaluated on the other half (ADR 0001, decision 7)
====  ==========  ==================================================================

B's distributions are computed on the **held-out** respondents of the cell —
never shown in training — with the retrieval policy used at training
(``prompts.nearest_context_items``: same survey, cosine < 0.95, test items
barred) and a floor of ``context_min_n`` valid answers per context item.

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
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import polars as pl

from .corpus import blob, similarity, strata
from .corpus.ses import SesCrosswalk
from .dataset import normalise_code
from .foundry import FoundryChat
from .prompts import (
    ContextDistribution, ItemSpec, Persona, PromptTemplate, build_item_specs,
    nearest_context_items,
)
from .split import CELL_N_CONTRAST, HELDOUT_ITEMS_PATH, load_split, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = REPO_ROOT / "data" / "analysis" / "test_blocks.csv"
ITEMS_PATH = REPO_ROOT / "data" / "items.parquet"
HELDOUT_PATH = REPO_ROOT / "data" / "split" / "heldout_respondents.parquet"
#: Second split of the held-out respondents (arm BS): context half / eval half.
HELDOUT_HALVES_PATH = REPO_ROOT / "data" / "split" / "heldout_halves.csv"
SPLIT_MANIFEST_PATH = REPO_ROOT / "data" / "split" / "split_manifest.json"
STRATA_PATH = REPO_ROOT / "data" / "strata_definition.json"
SIMILARITY_PATH = REPO_ROOT / "data" / "item_similarity.parquet"

DEFAULT_TEMPERATURES = (0.3, 0.7, 1.0, 1.3)
DEFAULT_DRAWS = 100
DEFAULT_DEPLOYMENT = "c0-8k-txt"
DEFAULT_MODEL = "Llama-3.3-70B-Instruct-9.ft-0413559e867f42a59860f17f6153597b-c0-8k-txt"
DEFAULT_WORKERS = 3
DEFAULT_CONTEXT_K = 6
#: A context line backed by fewer valid held-out answers than this is dropped.
DEFAULT_CONTEXT_MIN_N = 10
#: Candidates retrieved before the n floor and the dedup; the index keeps 50.
CONTEXT_CANDIDATES = 50
# Measured with the repository's Llama-3 tokenizer: the longest of the 68
# rendered labels on the 12 pilot items is 19 tokens. 32 leaves headroom for
# tokenisation/service variation without inviting a long free-form response.
DEFAULT_MAX_TOKENS = 32
EXPECTED_PILOT_ITEMS = 12
EXPECTED_ITEM_CELL_PAIRS = 275

@dataclass(frozen=True)
class Arm:
    model_condition: str  # what the checkpoint was trained on
    context: str          # "none" or "stratum"


ARMS: dict[str, Arm] = {
    "A": Arm(model_condition="C0", context="none"),
    "B": Arm(model_condition="C1", context="stratum"),
    "B0": Arm(model_condition="C1", context="none"),
    "R": Arm(model_condition="base", context="none"),
    "BS": Arm(model_condition="C1", context="stratum_half"),
}


@dataclass(frozen=True)
class Campaign:
    """One fine-tuned model, one deployment, the arms it serves."""

    name: str
    model: str | None  # None until the fine-tune has succeeded
    deployment: str
    arms: tuple[str, ...]
    sku: str = "GlobalStandard"
    capacity: int = 1000
    version: str = "1"


# The Llama-3.3-70B fine-tune quota is 1 000 units per SKU on the account:
# GlobalStandard and DataZoneStandard are counted separately, so a C0 and a C1
# campaign can run at full rate side by side on different SKUs.
CAMPAIGNS: dict[str, Campaign] = {c.name: c for c in (
    Campaign("c0-8k", DEFAULT_MODEL, "c0-8k-txt", ("A",)),
    Campaign("c1-8k",
             "Llama-3.3-70B-Instruct-9.ft-5dfdb814460746c69cf0ed69177d8608-c1-8k-txt",
             "c1-8k-txt", ("B", "B0", "BS"), sku="DataZoneStandard"),
    Campaign("c0-20k", None, "c0-20k-txt", ("A",)),
    Campaign("c1-20k", None, "c1-20k-txt", ("B", "B0", "BS"), sku="DataZoneStandard"),
    # The base the fine-tunes were trained from ("…-Instruct-9.ft-…"): version 9.
    # Its own quota: 250 units, GlobalStandard.
    Campaign("base", "Llama-3.3-70B-Instruct", "llama33-70b-base", ("R",),
             capacity=250, version="9"),
)}

RESULT_FIELDS = (
    "draw_key", "arm", "deployment", "model", "condition", "context", "n_context", "temperature", "top_p",
    "max_tokens", "item_idx", "block", "survey_id", "variable", "language",
    "cell", "heldout_valid_n", "draw_idx", "raw_response", "matched_code", "valid",
    "started_at", "completed_at", "latency_seconds", "client_retries_total",
    "client_throttled_total", "client_filtered_total",
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
    #: Arm B only: the rendered context, in order (empty in A and B0).
    context: tuple[ContextDistribution, ...] = ()
    context_cosines: tuple[float, ...] = ()


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
    arm: str = "A"
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES
    draws: int = DEFAULT_DRAWS
    top_p: float = 1.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    workers: int = DEFAULT_WORKERS
    limit_item_cell_pairs: int | None = None
    context_k: int = DEFAULT_CONTEXT_K
    context_min_n: int = DEFAULT_CONTEXT_MIN_N

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {sorted(ARMS)}")
        if self.context_k < 1 or self.context_min_n < 1:
            raise ValueError("context_k and context_min_n must be positive")
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
    def condition(self) -> str:
        return ARMS[self.arm].model_condition

    @property
    def context(self) -> str:
        return ARMS[self.arm].context

    def template(self) -> PromptTemplate:
        """A and B0 render no context; B renders the cell's distributions."""
        if self.context != "none":
            return PromptTemplate(condition="C2", ses_dropout="none", k=self.context_k)
        return PromptTemplate(condition="C0", ses_dropout="none")

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


def cell_distribution(item: ItemSpec, codes: Sequence[Any], weights: Sequence[float],
                      ) -> ContextDistribution | None:
    """Weighted shares of `item`'s options among one cell's valid answers."""
    offered = [opt.code for opt in item.options]
    totals = dict.fromkeys(offered, 0.0)
    n = 0
    for raw, weight in zip(codes, weights, strict=True):
        code = item.canonical_code(normalise_code(raw))
        if code in totals:
            totals[code] += float(weight)
            n += 1
    mass = sum(totals.values())
    if n == 0 or mass <= 0:
        return None
    shares = tuple((opt.label, totals[opt.code] / mass) for opt in item.options)
    return ContextDistribution(item=item, shares=shares, n=n)


def attach_stratum_context(
    pairs: Sequence[ItemCell],
    *,
    k: int = DEFAULT_CONTEXT_K,
    min_n: int = DEFAULT_CONTEXT_MIN_N,
    items_path: Path = ITEMS_PATH,
    heldout_path: Path = HELDOUT_PATH,
    similarity_path: Path = SIMILARITY_PATH,
    heldout_items_path: Path = HELDOUT_ITEMS_PATH,
    survey_loader: Callable[[str, list[str] | None], pl.DataFrame] = blob.read_survey,
    halves_path: Path | None = None,
) -> list[ItemCell]:
    """Arms B / BS: each pair gets its cell's observed distributions on the k nearest items.

    Retrieval is the training policy (same survey, cosine < 0.95, test items
    barred). Distributions come from the held-out respondents of the cell,
    weighted by ``__weight``; a neighbour with fewer than `min_n` valid answers
    in the cell is skipped, and the template's dedup/leak check runs before
    the cut to `k`. Fewer than `k` lines — none, for an item whose survey has
    no neighbour in the index — is a legitimate outcome, recorded as is.

    With `halves_path` (arm BS), only the respondents of the **context** half
    feed the distributions; the eval half never reaches the prompt.
    """
    items = pl.read_parquet(items_path)
    specs = build_item_specs(items.iter_rows(named=True))
    split = load_split(items_path=heldout_items_path)
    pool = [key for key in specs if not split.is_test_item(key)]
    index = similarity.load_index(similarity_path)
    heldout = pl.read_parquet(heldout_path).with_columns(
        pl.col("__respondent_id").cast(pl.Utf8)
    )
    if halves_path is not None:
        context_half = pl.read_csv(halves_path, schema_overrides={"__respondent_id": pl.Utf8}) \
            .filter(pl.col("half") == "context").select("__survey_id", "__respondent_id")
        heldout = heldout.join(context_half, on=["__survey_id", "__respondent_id"], how="semi")
    template = PromptTemplate(condition="C2", ses_dropout="none", k=k)

    by_item: dict[tuple[str, str], list[ItemCell]] = {}
    for p in pairs:
        by_item.setdefault((p.survey_id, p.variable), []).append(p)

    out: dict[tuple[int, str], ItemCell] = {}
    for key, item_pairs in sorted(by_item.items()):
        candidates = [
            (nkey, cosine)
            for nkey, cosine in nearest_context_items(
                index, key, k=CONTEXT_CANDIDATES, same_survey_only=True, eligible=pool)
            if nkey in specs and specs[nkey].options
        ]
        if candidates:
            micro = survey_loader(
                key[0], ["__respondent_id", *[n[1] for n, _ in candidates]]
            ).with_columns(pl.col("__respondent_id").cast(pl.Utf8))
            joined = heldout.filter(pl.col("__survey_id") == key[0]).join(
                micro, on="__respondent_id", how="inner")
        for p in item_pairs:
            dists, cosines = [], {}
            if candidates:
                cell = joined.filter(pl.col("cell") == p.cell)
                weights = cell["__weight"].to_list()
                for nkey, cosine in candidates:
                    dist = cell_distribution(specs[nkey], cell[nkey[1]].to_list(), weights)
                    if dist is not None and dist.n >= min_n:
                        dists.append(dist)
                        cosines[nkey] = cosine
            kept = template.select_context(p.item, dists)
            out[(p.item_idx, p.cell)] = replace(
                p, context=tuple(kept),
                context_cosines=tuple(cosines[d.item.key] for d in kept),
            )
    return [out[(p.item_idx, p.cell)] for p in pairs]


def write_context_table(path: Path, pairs: Sequence[ItemCell]) -> None:
    """Sidecar: exactly what arm B shows, per pair — the audit trail of §3.3."""
    rows = []
    for p in pairs:
        for rank, (dist, cosine) in enumerate(zip(p.context, p.context_cosines,
                                                  strict=True), start=1):
            rows.append({
                "item_idx": p.item_idx, "cell": p.cell, "rank": rank,
                "context_survey_id": dist.item.survey_id,
                "context_variable": dist.item.variable,
                "cosine": round(float(cosine), 6), "n": dist.n,
                "shares": json.dumps([[label, round(share, 6)] for label, share in dist.shares],
                                     ensure_ascii=False),
            })
    fields = ("item_idx", "cell", "rank", "context_survey_id", "context_variable",
              "cosine", "n", "shares")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temp = Path(handle.name)
    temp.replace(path)


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


def _progress_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".progress.json")


def _context_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".context.csv")


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
        "item_similarity": SIMILARITY_PATH,
        "heldout_items": HELDOUT_ITEMS_PATH,
        **({"heldout_halves": HELDOUT_HALVES_PATH}
           if settings.context == "stratum_half" else {}),
    }
    split_manifest = json.loads(Path(paths["split_manifest"]).read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "scientific_contract": "docs/adr/0001-sous-ensemble-pilote-blocs-thematiques.md",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "frozen_split_input_hashes": split_manifest.get("input_hashes", {}),
        "settings": {**asdict(settings), "condition": settings.condition,
                     "context": settings.context},
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
    context = pair.context if settings.context != "none" else ()
    messages = template.build_messages(
        pair.persona, pair.item, context, dimensions=pair.dimensions
    )
    raw = chat.complete(messages, temperature=task.temperature,
                        max_tokens=settings.max_tokens, top_p=settings.top_p)
    matched = pair.item.match_answer(raw)
    return {
        "draw_key": task.key, "arm": settings.arm, "deployment": settings.deployment,
        "model": settings.model, "condition": settings.condition,
        "context": settings.context, "n_context": len(context),
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
        "client_filtered_total": getattr(chat, "filtered", 0),
    }


class StallError(RuntimeError):
    """No draw completed for ``stall_seconds``: the run is stuck, not slow."""


def _atomic_json_quiet(path: Path, value: Mapping[str, Any]) -> None:
    try:
        _atomic_json(path, value)
    except OSError:  # progress is advisory; never let it kill the run
        pass


def run(settings: RunSettings, output_csv: Path, *,
        chat: FoundryChat | None = None, pairs: Sequence[ItemCell] | None = None,
        manifest_paths: Mapping[str, Path] | None = None,
        stall_seconds: float = 600.0, progress_every: float = 30.0) -> dict[str, Any]:
    """Run pending draws, stopping after draining in-flight work on failure.

    Writes ``<output>.progress.json`` every `progress_every` seconds. If no
    draw completes for `stall_seconds` the run raises :class:`StallError`
    instead of hanging; worker threads possibly still blocked in a call are
    abandoned (the caller is expected to exit the process).
    """
    output_csv = Path(output_csv)
    if output_csv.suffix.lower() != ".csv":
        raise ValueError("output path must end in .csv")
    all_pairs = list(pairs) if pairs is not None else build_item_cells()
    if pairs is None and settings.context != "none":
        all_pairs = attach_stratum_context(
            all_pairs, k=settings.context_k, min_n=settings.context_min_n,
            halves_path=HELDOUT_HALVES_PATH if settings.context == "stratum_half" else None)
    selected_pairs = all_pairs[:settings.limit_item_cell_pairs]
    tasks = make_tasks(all_pairs, settings)
    manifest = build_manifest(settings, len(selected_pairs), manifest_paths)
    manifest_path = _manifest_path(output_csv)
    checkpoint_path = _jsonl_path(output_csv)
    diagnostics_path = _diagnostics_path(output_csv)
    progress_path = _progress_path(output_csv)
    ensure_manifest(manifest_path, manifest)
    if settings.context != "none":
        write_context_table(_context_path(output_csv), selected_pairs)
    records = load_checkpoint(checkpoint_path)
    expected_keys = {task.key for task in tasks}
    unexpected = set(records) - expected_keys
    if unexpected:
        raise ValueError(f"checkpoint has {len(unexpected)} keys outside this run")
    pending = [task for task in tasks if task.key not in records]
    if pending and chat is None:
        chat = FoundryChat(settings.deployment)
    template = settings.template()
    first_error: BaseException | None = None
    recent: deque[float] = deque()
    started = last_done = time.monotonic()

    def progress(state: str, error: BaseException | None = None) -> None:
        now = time.monotonic()
        while recent and now - recent[0] > 300:
            recent.popleft()
        valid = sum(bool(r["valid"]) for r in records.values())
        _atomic_json_quiet(progress_path, {
            "arm": settings.arm, "deployment": settings.deployment, "state": state,
            "tasks": len(tasks), "done": len(records), "valid": valid,
            "invalid": len(records) - valid, "pending": len(tasks) - len(records),
            "per_minute_last_5min": round(
                len(recent) / (max(min(now - started, 300.0), 1.0) / 60), 1),
            "seconds_since_last_draw": round(now - last_done, 1),
            "throttled_total": getattr(chat, "throttled", 0),
            "retries_total": getattr(chat, "retries", 0),
            "filtered_total": getattr(chat, "filtered", 0),
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "updated_at": utc_now(),
        })

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=settings.workers)
    try:
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            iterator = iter(pending)
            active: dict[Future, DrawTask] = {}
            for _ in range(settings.workers):
                task = next(iterator, None)
                if task is not None:
                    active[pool.submit(_complete, chat, template, settings, task)] = task
            progress("running")
            last_progress = time.monotonic()
            while active:
                done, _ = wait(active, timeout=progress_every, return_when=FIRST_COMPLETED)
                now = time.monotonic()
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
                    recent.append(now)
                    last_done = now
                if first_error is None and now - last_done > stall_seconds:
                    first_error = StallError(
                        f"no draw completed in {stall_seconds:.0f}s "
                        f"({len(active)} calls in flight)")
                    break
                if first_error is None:
                    for _ in done:
                        replacement = next(iterator, None)
                        if replacement is not None:
                            active[pool.submit(
                                _complete, chat, template, settings, replacement
                            )] = replacement
                if now - last_progress >= progress_every:
                    progress("running")
                    last_progress = now
    finally:
        pool.shutdown(wait=first_error is None, cancel_futures=True)
        write_csv_companion(output_csv, records.values())
        write_diagnostics(diagnostics_path, tasks, records)
        progress("failed" if first_error else "completed", first_error)

    if isinstance(first_error, StallError):
        raise first_error
    if first_error is not None:
        raise RuntimeError("transport failure; failed draw remains pending for resume") from first_error
    validate_coverage(tasks, records, full_mode=settings.full_mode)
    valid = sum(bool(record["valid"]) for record in records.values())
    return {
        "arm": settings.arm, "tasks": len(tasks), "transport_n": len(records),
        "effective_n": valid, "invalid_n": len(records) - valid,
        "invalid_rate": (len(records) - valid) / len(records) if records else 0.0,
        "checkpoint": str(checkpoint_path), "csv": str(output_csv),
        "manifest": str(manifest_path), "diagnostics": str(diagnostics_path),
    }


__all__ = [
    "ARMS", "CAMPAIGNS", "HELDOUT_HALVES_PATH", "Arm", "Campaign", "StallError",
    "attach_stratum_context", "cell_distribution", "write_context_table",
    "DEFAULT_DEPLOYMENT", "DEFAULT_DRAWS", "DEFAULT_MAX_TOKENS", "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURES", "DEFAULT_WORKERS", "DrawTask",
    "EXPECTED_ITEM_CELL_PAIRS", "EXPECTED_PILOT_ITEMS", "ItemCell", "RunSettings",
    "append_checkpoint", "build_item_cells", "build_manifest", "ensure_manifest",
    "load_checkpoint", "make_tasks", "run", "validate_coverage",
    "write_csv_companion", "write_diagnostics",
]
