"""Prepare observed and model distributions for the frozen inference pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import math
import tempfile

import polars as pl

from article_silicon_sampling_quebec.corpus import blob
from article_silicon_sampling_quebec.dataset import normalise_code
from article_silicon_sampling_quebec.inference import (
    ARMS,
    CAMPAIGNS,
    DEFAULT_DRAWS,
    DEFAULT_TEMPERATURES,
    HELDOUT_HALVES_PATH,
    HELDOUT_PATH,
    ItemCell,
    build_item_cells,
)


REPO = Path(__file__).resolve().parents[2]
INFERENCE_ROOT = REPO / "data" / "analysis" / "inference"
ANALYSIS_ROOT = REPO / "data" / "analysis"
ARM_PATHS = {
    "A": Path("c0-8k/A.csv"),
    "B": Path("c1-8k/B.csv"),
    "B0": Path("c1-8k/B0.csv"),
    "R": Path("base/R.csv"),
    "BS": Path("c1-8k/BS.csv"),
}
ARM_CAMPAIGNS = {
    "A": "c0-8k", "B": "c1-8k", "B0": "c1-8k", "R": "base", "BS": "c1-8k",
}
SECOND_BRIEF_ARM_PATHS = {
    "A20": Path("c0-20k/A20.csv"),
    "B020": Path("c1-20k/B020.csv"),
    "BR8": Path("c1-8k/BR8.csv"),
    "BR20": Path("c1-20k/BR20.csv"),
}
SECOND_BRIEF_ARM_CAMPAIGNS = {
    "A20": "c0-20k", "B020": "c1-20k", "BR8": "c1-8k", "BR20": "c1-20k",
}
ALL_ARM_PATHS = {**ARM_PATHS, **SECOND_BRIEF_ARM_PATHS}
ALL_ARM_CAMPAIGNS = {**ARM_CAMPAIGNS, **SECOND_BRIEF_ARM_CAMPAIGNS}

DISTRIBUTION_COLUMNS = ("arm", "item_idx", "cell", "temperature", "code", "share", "n")
DIAGNOSTIC_COLUMNS = (
    "arm", "item_idx", "cell", "temperature", "transport_n", "effective_n",
    "invalid_n", "invalid_rate", "coverage_ok",
)
OBSERVED_RESPONSE_COLUMNS = ("item_idx", "cell", "respondent_id", "code", "weight")


def _pair_index(pairs: Sequence[ItemCell]) -> dict[tuple[int, str], ItemCell]:
    index = {(pair.item_idx, pair.cell): pair for pair in pairs}
    if len(index) != len(pairs):
        raise ValueError("item-cell pairs are not unique")
    return index


def build_observed_tables(
    pairs: Sequence[ItemCell],
    heldout: pl.DataFrame,
    *,
    survey_loader: Callable[[str, list[str] | None], pl.DataFrame] = blob.read_survey,
    validate_pair_n: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return weighted truth distributions and respondent-level bootstrap rows."""
    required = {"__survey_id", "__respondent_id", "__weight", "cell"}
    missing = required - set(heldout.columns)
    if missing:
        raise ValueError(f"held-out respondents are missing columns: {sorted(missing)}")
    heldout = heldout.with_columns(pl.col("__respondent_id").cast(pl.Utf8))
    distributions: list[dict] = []
    responses: list[dict] = []

    by_survey: dict[str, list[ItemCell]] = {}
    for pair in pairs:
        by_survey.setdefault(pair.survey_id, []).append(pair)

    for survey_id, survey_pairs in sorted(by_survey.items()):
        variables = sorted({pair.variable for pair in survey_pairs})
        micro = survey_loader(survey_id, ["__respondent_id", *variables]).with_columns(
            pl.col("__respondent_id").cast(pl.Utf8)
        )
        survey_heldout = heldout.filter(pl.col("__survey_id") == survey_id)
        for pair in sorted(survey_pairs, key=lambda value: (value.item_idx, value.cell)):
            cell = survey_heldout.filter(pl.col("cell") == pair.cell).join(
                micro.select("__respondent_id", pair.variable),
                on="__respondent_id",
                how="inner",
                validate="1:1",
            )
            offered = [option.code for option in pair.item.options]
            totals = dict.fromkeys(offered, 0.0)
            valid_rows: list[dict] = []
            for respondent_id, raw, weight in cell.select(
                "__respondent_id", pair.variable, "__weight"
            ).iter_rows():
                code = pair.item.canonical_code(normalise_code(raw))
                if code not in totals:
                    continue
                numeric_weight = float(weight)
                if not math.isfinite(numeric_weight) or numeric_weight < 0:
                    raise ValueError(
                        f"invalid weight for item {pair.item_idx}, cell {pair.cell}: {weight!r}"
                    )
                totals[code] += numeric_weight
                valid_rows.append({
                    "item_idx": pair.item_idx,
                    "cell": pair.cell,
                    "respondent_id": str(respondent_id),
                    "code": code,
                    "weight": numeric_weight,
                })
            n = len(valid_rows)
            if validate_pair_n and n != pair.heldout_valid_n:
                raise ValueError(
                    f"observed valid n changed for item {pair.item_idx}, cell {pair.cell}: "
                    f"expected {pair.heldout_valid_n}, got {n}"
                )
            mass = sum(totals.values())
            if mass <= 0:
                raise ValueError(
                    f"observed weight mass is zero for item {pair.item_idx}, cell {pair.cell}"
                )
            responses.extend(valid_rows)
            distributions.extend({
                "arm": "observed",
                "item_idx": pair.item_idx,
                "cell": pair.cell,
                "temperature": None,
                "code": code,
                "share": totals[code] / mass,
                "n": n,
            } for code in offered)

    return (
        pl.DataFrame(distributions).with_columns(
            pl.col("temperature").cast(pl.Float64)
        ).select(DISTRIBUTION_COLUMNS).sort("item_idx", "cell", "code"),
        pl.DataFrame(responses).select(OBSERVED_RESPONSE_COLUMNS).sort(
            "item_idx", "cell", "respondent_id"
        ),
    )


def build_model_tables(
    draws_by_arm: Mapping[str, pl.DataFrame],
    pairs: Sequence[ItemCell],
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    draws_per_group: int = DEFAULT_DRAWS,
    arms: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validate complete draw files and return distributions plus diagnostics."""
    expected_arms = set(arms) if arms is not None else set(ARM_PATHS)
    unknown_arms = expected_arms - set(ALL_ARM_PATHS)
    if unknown_arms:
        raise ValueError(f"unknown arms: {sorted(unknown_arms)}")
    if set(draws_by_arm) != expected_arms:
        raise ValueError(
            f"expected arms {sorted(expected_arms)}, got {sorted(draws_by_arm)}"
        )
    if draws_per_group < 1 or not temperatures:
        raise ValueError("draws_per_group and temperatures must be positive/non-empty")

    pairs_by_key = _pair_index(pairs)
    expected_groups = {
        (item_idx, cell, float(temperature))
        for item_idx, cell in pairs_by_key
        for temperature in temperatures
    }
    distribution_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    required = {
        "draw_key", "arm", "deployment", "model", "condition", "context", "n_context",
        "temperature", "item_idx", "block", "survey_id", "variable", "cell",
        "heldout_valid_n", "draw_idx", "matched_code", "valid",
    }

    for expected_arm, source in sorted(draws_by_arm.items()):
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"arm {expected_arm} is missing columns: {sorted(missing)}")
        frame = source.with_columns(
            pl.col("arm").cast(pl.Utf8),
            pl.col("deployment").cast(pl.Utf8),
            pl.col("model").cast(pl.Utf8),
            pl.col("condition").cast(pl.Utf8),
            pl.col("context").cast(pl.Utf8),
            pl.col("cell").cast(pl.Utf8),
            pl.col("matched_code").cast(pl.Utf8),
            pl.col("temperature").cast(pl.Float64),
            pl.col("item_idx").cast(pl.Int64),
            pl.col("heldout_valid_n").cast(pl.Int64),
            pl.col("draw_idx").cast(pl.Int64),
            pl.col("n_context").cast(pl.Int64),
            pl.col("valid").cast(pl.Boolean),
        )
        actual_arms = set(frame["arm"].unique().to_list())
        if actual_arms != {expected_arm}:
            raise ValueError(f"{expected_arm}.csv contains arm values {sorted(actual_arms)}")
        campaign = CAMPAIGNS[ALL_ARM_CAMPAIGNS[expected_arm]]
        expected_metadata = {
            "deployment": campaign.deployment,
            "model": campaign.model,
            "condition": ARMS[expected_arm].model_condition,
            "context": ARMS[expected_arm].context,
        }
        for column, expected in expected_metadata.items():
            actual = set(frame[column].unique().to_list())
            if actual != {expected}:
                raise ValueError(
                    f"arm {expected_arm} has {column} values {sorted(actual)}, expected {expected!r}"
                )
        has_context = ARMS[expected_arm].context != "none"
        if not has_context and set(frame["n_context"].unique().to_list()) != {0}:
            raise ValueError(f"arm {expected_arm} must have n_context = 0")
        if has_context and frame.filter(
            pl.col("n_context").is_null()
            | (pl.col("n_context") < 0)
            | (pl.col("n_context") > 6)
        ).height:
            raise ValueError(f"arm {expected_arm} has n_context outside 0..6")
        if frame["draw_key"].n_unique() != frame.height:
            raise ValueError(f"arm {expected_arm} contains duplicate draw_key values")
        inconsistent = frame.filter(
            pl.col("valid").is_null()
            | (pl.col("valid") != pl.col("matched_code").is_not_null())
        )
        if inconsistent.height:
            raise ValueError(f"arm {expected_arm} has inconsistent valid/matched_code values")

        actual_groups = {
            (int(item_idx), str(cell), float(temperature))
            for item_idx, cell, temperature in frame.select(
                "item_idx", "cell", "temperature"
            ).unique().iter_rows()
        }
        missing_groups = expected_groups - actual_groups
        extra_groups = actual_groups - expected_groups
        if missing_groups or extra_groups:
            raise ValueError(
                f"arm {expected_arm} group coverage mismatch: "
                f"{len(missing_groups)} missing, {len(extra_groups)} unexpected"
            )

        for (item_idx, cell, temperature), group in frame.group_by(
            "item_idx", "cell", "temperature", maintain_order=False
        ):
            key = (int(item_idx), str(cell))
            pair = pairs_by_key[key]
            temperature = float(temperature)
            pair_metadata = {
                "block": pair.block,
                "survey_id": pair.survey_id,
                "variable": pair.variable,
            }
            for column, expected in pair_metadata.items():
                actual = set(group[column].unique().to_list())
                if actual != {expected}:
                    raise ValueError(
                        f"arm {expected_arm}, item {item_idx}, cell {cell}: "
                        f"{column} values {sorted(actual)}, expected {expected!r}"
                    )
            if group.height != draws_per_group:
                raise ValueError(
                    f"arm {expected_arm}, item {item_idx}, cell {cell}, "
                    f"temperature {temperature:g}: expected {draws_per_group} draws, "
                    f"got {group.height}"
                )
            actual_draw_indices = set(group["draw_idx"].to_list())
            expected_draw_indices = set(range(draws_per_group))
            if actual_draw_indices != expected_draw_indices:
                raise ValueError(
                    f"arm {expected_arm}, item {item_idx}, cell {cell}, "
                    f"temperature {temperature:g}: draw_idx coverage mismatch"
                )
            expected_keys = {
                f"{item_idx}|{cell}|{format(temperature, '.12g')}|{draw_idx}"
                for draw_idx in expected_draw_indices
            }
            if set(group["draw_key"].to_list()) != expected_keys:
                raise ValueError(
                    f"arm {expected_arm}, item {item_idx}, cell {cell}, "
                    f"temperature {temperature:g}: draw_key values do not match the draw metadata"
                )
            heldout_ns = set(group["heldout_valid_n"].to_list())
            if heldout_ns != {pair.heldout_valid_n}:
                raise ValueError(
                    f"arm {expected_arm}, item {item_idx}, cell {cell}: "
                    f"heldout_valid_n is {sorted(heldout_ns)}, expected {pair.heldout_valid_n}"
                )
            offered = [option.code for option in pair.item.options]
            valid_codes = group.filter(pl.col("valid"))["matched_code"].to_list()
            unexpected_codes = set(valid_codes) - set(offered)
            if unexpected_codes:
                raise ValueError(
                    f"arm {expected_arm}, item {item_idx}, cell {cell}: "
                    f"codes not offered by the item: {sorted(unexpected_codes)}"
                )
            counts = {code: 0 for code in offered}
            for code in valid_codes:
                counts[code] += 1
            effective_n = len(valid_codes)
            invalid_n = group.height - effective_n
            diagnostic_rows.append({
                "arm": expected_arm,
                "item_idx": int(item_idx),
                "cell": str(cell),
                "temperature": temperature,
                "transport_n": group.height,
                "effective_n": effective_n,
                "invalid_n": invalid_n,
                "invalid_rate": invalid_n / group.height,
                "coverage_ok": group.height == draws_per_group,
            })
            distribution_rows.extend({
                "arm": expected_arm,
                "item_idx": int(item_idx),
                "cell": str(cell),
                "temperature": temperature,
                "code": code,
                "share": counts[code] / effective_n if effective_n else None,
                "n": effective_n,
            } for code in offered)

    sort_columns = ["arm", "item_idx", "cell", "temperature", "code"]
    return (
        pl.DataFrame(distribution_rows).select(DISTRIBUTION_COLUMNS).sort(sort_columns),
        pl.DataFrame(diagnostic_rows).select(DIAGNOSTIC_COLUMNS).sort(sort_columns[:-1]),
    )


def read_arm_draws(
    inference_root: Path = INFERENCE_ROOT,
    *,
    arms: Sequence[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Read selected production arm CSVs without inferring option codes as numbers."""
    selected = list(arms) if arms is not None else list(ARM_PATHS)
    unknown_arms = set(selected) - set(ALL_ARM_PATHS)
    if unknown_arms:
        raise ValueError(f"unknown arms: {sorted(unknown_arms)}")
    frames = {}
    for arm in selected:
        relative_path = ALL_ARM_PATHS[arm]
        path = Path(inference_root) / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing inference output for arm {arm}: {path}")
        frames[arm] = pl.read_csv(
            path,
            schema_overrides={"arm": pl.Utf8, "cell": pl.Utf8, "matched_code": pl.Utf8},
            null_values={"matched_code": ""},
        )
    return frames


def write_distribution_outputs(
    *,
    inference_root: Path = INFERENCE_ROOT,
    output_root: Path = ANALYSIS_ROOT,
    heldout_path: Path = HELDOUT_PATH,
    halves_path: Path = HELDOUT_HALVES_PATH,
    pairs: Sequence[ItemCell] | None = None,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    draws_per_group: int = DEFAULT_DRAWS,
    arms: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Build and write all analysis-ready distribution inputs."""
    pairs = list(pairs) if pairs is not None else build_item_cells()
    heldout = pl.read_parquet(heldout_path).with_columns(
        pl.col("__respondent_id").cast(pl.Utf8)
    )
    halves = pl.read_csv(
        halves_path,
        schema_overrides={"__respondent_id": pl.Utf8},
    ).filter(pl.col("half") == "eval")
    eval_heldout = heldout.join(
        halves.select("__survey_id", "__respondent_id", "cell"),
        on=["__survey_id", "__respondent_id", "cell"],
        how="inner",
        validate="1:1",
    )
    if eval_heldout.height != halves.height:
        raise ValueError("heldout_halves eval rows do not match held-out respondents")
    observed, responses = build_observed_tables(
        pairs, eval_heldout, validate_pair_n=False,
    )
    model, diagnostics = build_model_tables(
        read_arm_draws(inference_root, arms=arms), pairs,
        temperatures=temperatures, draws_per_group=draws_per_group, arms=arms,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "distributions": output_root / "distributions.csv",
        "diagnostics": output_root / "distribution_diagnostics.csv",
        "observed_responses": output_root / "observed_responses.csv",
    }
    _write_csv_atomic(pl.concat([observed, model], how="vertical"), outputs["distributions"])
    _write_csv_atomic(diagnostics, outputs["diagnostics"])
    _write_csv_atomic(responses, outputs["observed_responses"])
    return outputs


def _write_csv_atomic(frame: pl.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_csv(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ALL_ARM_PATHS", "ANALYSIS_ROOT", "ARM_PATHS", "INFERENCE_ROOT",
    "SECOND_BRIEF_ARM_PATHS", "build_model_tables",
    "build_observed_tables", "read_arm_draws", "write_distribution_outputs",
]
