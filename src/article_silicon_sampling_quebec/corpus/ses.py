"""Apply the SES crosswalks — step 1.3 of docs/plan_article.md (§3.4, §2.1).

Turns a raw respondent row (a Parquet row, raw-first) into a canonical
sociodemographic profile:

    >>> xw = SesCrosswalk.load()
    >>> xw.profile("ces_2021", {"cps21_yob": 42, "cps21_genderid": 2})["age"].level
    '58_64'  # illustrative

Design rules enforced here:
  * every non-response code maps to the explicit `missing` level, never imputed;
  * a source code coarser than the canonical schema yields a SET of levels
    (`coarse=True`) rather than an invented single level;
  * unknown codes (present in the microdata but absent from the crosswalk) are
    reported as `missing` with `unmapped=True` so they surface in QA instead of
    silently disappearing.

No network access: reads the two JSON files under data/crosswalks/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[3]
CROSSWALK_DIR = _REPO / "data" / "crosswalks"
CANONICAL_PATH = CROSSWALK_DIR / "ses_canonical.json"
CROSSWALK_PATH = CROSSWALK_DIR / "ses_crosswalk.json"

MISSING = "missing"

#: Canonical dimensions, ordered as they should appear in a persona.
DIMENSIONS: tuple[str, ...] = (
    "age", "gender", "education", "region", "region_qc", "income", "language",
)


def normalize_code(value: Any) -> str | None:
    """Normalize a raw code to the string form used as crosswalk key.

    Integers, zero-padded strings and integral floats all collapse to the same
    key ("09", 9, 9.0 -> "9"). Non-numeric strings keep their case ("QC"), the
    empty string stays "" (a real category in cecd_sante_can_usa). Nulls and
    NaN return None.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return str(int(text))
    except ValueError:
        return text


def _as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return None if (isinstance(value, float) and math.isnan(value)) else float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class SesValue:
    """One canonical dimension for one respondent."""

    dimension: str
    levels: tuple[str, ...]
    coarse: bool = False
    raw: Any = None
    variable: str | None = None
    needs_review: bool = False
    unmapped: bool = False

    @property
    def level(self) -> str | None:
        """The single canonical level, or None when the mapping is coarse."""
        return self.levels[0] if len(self.levels) == 1 else None

    @property
    def is_missing(self) -> bool:
        return self.levels == (MISSING,)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.level or "|".join(self.levels)


def _missing(dimension: str, raw: Any = None, variable: str | None = None,
             needs_review: bool = False, unmapped: bool = False) -> SesValue:
    return SesValue(dimension, (MISSING,), False, raw, variable, needs_review, unmapped)


class SesCrosswalk:
    """Canonical schema + per-survey mappings, with an apply() front end."""

    def __init__(self, canonical: Mapping[str, Any], crosswalk: Mapping[str, Any]):
        self.canonical = canonical
        self.crosswalk = crosswalk
        self._surveys: dict[str, dict] = dict(crosswalk["surveys"])
        self._levels: dict[str, tuple[str, ...]] = {
            dim: tuple(lv["id"] for lv in spec["levels"])
            for dim, spec in canonical["dimensions"].items()
        }
        self._bounds: dict[str, list[tuple[str, float, float]]] = {}
        for dim, key_lo, key_hi in (("age", "min_age", "max_age"),
                                    ("income", "min_amount", "max_amount")):
            bounds = []
            for lv in canonical["dimensions"][dim]["levels"]:
                if lv["id"] == MISSING:
                    continue
                lo = lv.get(key_lo, -math.inf)
                hi = lv.get(key_hi, math.inf)
                bounds.append((lv["id"], float(lo), float(hi)))
            self._bounds[dim] = bounds

    # ---------------- loading ----------------

    @classmethod
    def load(cls, canonical_path: Path | str = CANONICAL_PATH,
             crosswalk_path: Path | str = CROSSWALK_PATH) -> "SesCrosswalk":
        with open(canonical_path, encoding="utf-8") as fh:
            canonical = json.load(fh)
        with open(crosswalk_path, encoding="utf-8") as fh:
            crosswalk = json.load(fh)
        return cls(canonical, crosswalk)

    # ---------------- introspection ----------------

    @property
    def survey_ids(self) -> tuple[str, ...]:
        return tuple(self._surveys)

    def levels(self, dimension: str) -> tuple[str, ...]:
        return self._levels[dimension]

    def dimensions_for(self, survey_id: str) -> tuple[str, ...]:
        spec = self._surveys[survey_id]
        return tuple(d for d in DIMENSIONS if d in spec)

    def spec(self, survey_id: str, dimension: str) -> dict | None:
        return self._surveys[survey_id].get(dimension)

    def source_variables(self, survey_id: str, dimension: str) -> tuple[str, ...]:
        return tuple(_variables_of(self.spec(survey_id, dimension) or {}))

    def label(self, dimension: str, level: str, lang: str = "fr") -> str:
        for lv in self.canonical["dimensions"][dimension]["levels"]:
            if lv["id"] == level:
                return lv.get(f"label_{lang}", lv["id"])
        raise KeyError(f"{dimension}/{level}")

    # ---------------- application ----------------

    def apply(self, survey_id: str, dimension: str, row: Mapping[str, Any]) -> SesValue:
        spec = self.spec(survey_id, dimension)
        if spec is None:
            return _missing(dimension)
        return self._apply_spec(dimension, spec, row)

    def profile(self, survey_id: str, row: Mapping[str, Any]) -> dict[str, SesValue]:
        """Canonical profile of one respondent, dimension -> SesValue."""
        return {dim: self.apply(survey_id, dim, row)
                for dim in self.dimensions_for(survey_id)}

    def profile_labels(self, survey_id: str, row: Mapping[str, Any],
                       lang: str = "fr", include_missing: bool = False) -> dict[str, str]:
        """Human-readable persona rendering (the §3.4 'gabarit canonique')."""
        out: dict[str, str] = {}
        for dim, val in self.profile(survey_id, row).items():
            if val.is_missing and not include_missing:
                continue
            out[dim] = " / ".join(self.label(dim, lv, lang) for lv in val.levels)
        return out

    # ---------------- kind dispatch ----------------

    def _apply_spec(self, dimension: str, spec: Mapping[str, Any],
                    row: Mapping[str, Any]) -> SesValue:
        kind = spec["kind"]
        review = bool(spec.get("needs_review", False))
        if kind == "constant":
            return SesValue(dimension, (spec["value"],), False, None, None, review)
        if kind == "coalesce":
            last = _missing(dimension, needs_review=review)
            for sub in spec["sources"]:
                got = self._apply_spec(dimension, sub, row)
                if not got.is_missing:
                    return SesValue(dimension, got.levels, got.coarse, got.raw,
                                    got.variable, review or got.needs_review,
                                    got.unmapped)
                last = got
            return SesValue(dimension, (MISSING,), False, last.raw, last.variable,
                            review or last.needs_review, last.unmapped)

        variable = spec["variable"]
        raw = row.get(variable)
        if kind == "categorical":
            return self._apply_categorical(dimension, spec, variable, raw, review)
        if kind in ("year_of_birth", "yob_code"):
            return self._apply_year(dimension, spec, variable, raw, review, kind)
        if kind == "age_years":
            return self._apply_age_years(dimension, spec, variable, raw, review)
        if kind == "amount":
            return self._apply_amount(dimension, spec, variable, raw, review)
        raise ValueError(f"unknown mapping kind: {kind!r}")

    def _apply_categorical(self, dimension, spec, variable, raw, review) -> SesValue:
        code = normalize_code(raw)
        if code is None:
            return _missing(dimension, raw, variable, review)
        target = spec["map"].get(code)
        if target is None:
            return _missing(dimension, raw, variable, needs_review=True, unmapped=True)
        if isinstance(target, str):
            return SesValue(dimension, (target,), False, raw, variable, review)
        levels = tuple(target["to"])
        return SesValue(dimension, levels, bool(target.get("coarse", False)), raw,
                        variable, review or bool(target.get("needs_review", False)))

    def _apply_year(self, dimension, spec, variable, raw, review, kind) -> SesValue:
        code = normalize_code(raw)
        if code is None or code in spec.get("missing_values", []):
            return _missing(dimension, raw, variable, review)
        number = _as_number(raw)
        if number is None:
            return _missing(dimension, raw, variable, needs_review=True, unmapped=True)
        year = int(number) + int(spec["code_offset"]) if kind == "yob_code" else int(number)
        lo, hi = spec.get("valid_year_range", [1900, spec["reference_year"]])
        if not (lo <= year <= hi):
            return _missing(dimension, raw, variable, needs_review=True, unmapped=True)
        return self._bracket("age", int(spec["reference_year"]) - year, dimension,
                             raw, variable, review)

    def _apply_age_years(self, dimension, spec, variable, raw, review) -> SesValue:
        code = normalize_code(raw)
        if code is None or code in spec.get("missing_values", []):
            return _missing(dimension, raw, variable, review)
        number = _as_number(raw)
        lo, hi = spec.get("valid_age_range", [0, 120])
        if number is None or not (lo <= number <= hi):
            return _missing(dimension, raw, variable, needs_review=True, unmapped=True)
        return self._bracket("age", number, dimension, raw, variable, review)

    def _apply_amount(self, dimension, spec, variable, raw, review) -> SesValue:
        code = normalize_code(raw)
        if code is None or code in spec.get("missing_values", []):
            return _missing(dimension, raw, variable, review)
        number = _as_number(raw)
        if number is None or not (spec.get("min_plausible", 0) <= number
                                  <= spec.get("max_plausible", math.inf)):
            return _missing(dimension, raw, variable, review)
        return self._bracket("income", number, dimension, raw, variable, review)

    def _bracket(self, scale: str, value: float, dimension: str, raw, variable,
                 review) -> SesValue:
        for level, lo, hi in self._bounds[scale]:
            if lo <= value <= hi:
                return SesValue(dimension, (level,), False, raw, variable, review)
        return _missing(dimension, raw, variable, needs_review=True, unmapped=True)


def _variables_of(spec: Mapping[str, Any]) -> Iterable[str]:
    if spec.get("kind") == "coalesce":
        for sub in spec["sources"]:
            yield from _variables_of(sub)
    elif "variable" in spec:
        yield spec["variable"]


def declared_levels(spec: Mapping[str, Any]) -> set[str]:
    """All canonical levels a mapping can emit (helper for tests/QA)."""
    out: set[str] = set()
    kind = spec.get("kind")
    if kind == "constant":
        return {spec["value"]}
    if kind == "coalesce":
        for sub in spec["sources"]:
            out |= declared_levels(sub)
        out.add(MISSING)
        return out
    if kind == "categorical":
        for target in spec["map"].values():
            if isinstance(target, str):
                out.add(target)
            else:
                out |= set(target["to"])
        out.add(MISSING)
        return out
    out.add(MISSING)
    return out


__all__ = [
    "CANONICAL_PATH", "CROSSWALK_PATH", "CROSSWALK_DIR", "DIMENSIONS", "MISSING",
    "SesCrosswalk", "SesValue", "declared_levels", "normalize_code",
]
