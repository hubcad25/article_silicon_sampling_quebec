"""Score model runs against CES 2025 (plan §3, §6, §7).

    python -m benchmark.score ces25-sem-20k ces25-fix-20k llama33-70b-base [--smoke]

Human side: every CES 2025 respondent aged 18+ in the ten provinces with a
positive ``cps25_weight_general_all``; for each item, the weighted shares of
the recorded answers (DK included, not-asked excluded) in each of the 15 cells.
Model side: shares of the valid answers (invalid removed, rate reported).

Writes benchmark/results/:
    tv_by_cell.csv   model x cell x item: TV, human n, n_eff, valid model n, invalid rate
    summary.csv      per model: mean of the 150 TVs (headline) with 95% CI,
                     cell-weighted mean, mean population TV, cells better than
                     the modal baseline
    contrasts.csv    every pair of models: difference of headline, paired 95% CI

References, scored like models: ``human-floor`` (TV between two random halves
of each cell, mean of 200 splits) and ``modal-baseline`` (the national
weighted mode of each item, for everyone).
Bootstrap: 2,000 replicates, basic (reverse-percentile) intervals; human respondents resampled within cells with
their weights; model profiles resampled within cells, the same profile ids for
every model, so contrasts are paired.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

from . import CELLS, FROZEN, REGION_OF, REPO, RESULTS, RUNS, TARGETS, age_band

sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.corpus.blob import read_survey  # noqa: E402

SEED = 20261002
N_BOOT = int(__import__("os").environ.get("BENCH_N_BOOT", 2_000))
N_SPLITS = 200
#: cps25_province codes (the crosswalk's order) -> canonical province.
PROVINCES = {1: "ab", 2: "bc", 3: "mb", 4: "nb", 5: "nl", 7: "ns", 9: "on", 10: "pe", 11: "qc", 12: "sk"}


def options() -> dict[str, list[str]]:
    """Option codes of each item, from the frozen requests (identical across languages)."""
    out = {}
    with (FROZEN / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            out.setdefault(record["item"], sorted(set(record["options"].values()), key=int))
            if len(out) == len(TARGETS):
                return out
    return out


def human_data() -> pd.DataFrame:
    columns = ["cps25_province", "cps25_age_in_years", "cps25_weight_general_all"]
    frame = read_survey("ces_2025", columns + [f"cps25_{t}" for t in TARGETS]).to_pandas()
    frame["region"] = frame.cps25_province.map(PROVINCES).map(REGION_OF)
    frame["age_group"] = [age_band(a) for a in frame.cps25_age_in_years]
    frame = frame[frame.region.notna() & frame.age_group.notna()
                  & (frame.cps25_weight_general_all > 0)].copy()
    frame["cell"] = frame.region + "|" + frame.age_group
    frame["w"] = frame.cps25_weight_general_all
    return frame


def shares(codes: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Weighted shares over option indices 0..k-1; codes < 0 are ignored."""
    keep = codes >= 0
    total = np.bincount(codes[keep], weights[keep], minlength=k)
    mass = total.sum()
    return total / mass if mass > 0 else np.full(k, np.nan)


def basic_ci(estimate: float, replicates: np.ndarray) -> tuple[float, float]:
    """Basic (reverse-percentile) 95% interval: 2*estimate - quantiles.

    TV is biased upward under resampling (a resample of a cell has fewer
    distinct units, so its distribution is noisier). Percentile intervals sit
    above the estimate, sometimes entirely; the basic interval removes that
    shift.
    """
    lo, hi = np.nanpercentile(replicates, [2.5, 97.5])
    return 2 * estimate - hi, 2 * estimate - lo


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


class Panel:
    """Per cell, a matrix of option indices (rows = units, columns = items) and weights."""

    def __init__(self, codes: dict[str, np.ndarray], weights: dict[str, np.ndarray]):
        self.codes, self.weights = codes, weights

    def dist(self, cell: str, j: int, k: int, rows: np.ndarray | None = None) -> np.ndarray:
        codes, weights = self.codes[cell][:, j], self.weights[cell]
        if rows is not None:
            codes, weights = codes[rows], weights[rows]
        return shares(codes, weights, k)


def index_codes(values, opts: list[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(opts)}
    return np.array([lookup.get(str(int(v)) if pd.notna(v) and v != "" else None, -1)
                     for v in values], dtype=np.int64)


def human_panel(frame: pd.DataFrame, opts: dict[str, list[str]]) -> tuple[Panel, dict]:
    codes, weights, audit = {}, {}, {}
    for cell in CELLS:
        block = frame[frame.cell == cell]
        codes[cell] = np.column_stack([index_codes(block[f"cps25_{t}"], opts[t]) for t in TARGETS])
        weights[cell] = block.w.to_numpy(float)
        audit[cell] = len(block)
    for t in TARGETS:
        odd = set(frame[f"cps25_{t}"].dropna().astype(int).astype(str)) - set(opts[t])
        if odd:
            print(f"  warning: human codes outside the options of {t}: {sorted(odd)}")
    return Panel(codes, weights), audit


def model_panel(name: str, profiles: pd.DataFrame, opts: dict[str, list[str]]) -> tuple[Panel, pd.DataFrame]:
    responses = pd.read_csv(RUNS / name / "responses.csv", dtype={"code": str}, keep_default_na=False)
    wide = responses.pivot(index="profile_id", columns="item", values="code")
    codes, weights = {}, {}
    for cell in CELLS:
        ids = profiles.loc[profiles.cell == cell, "profile_id"]
        block = wide.reindex(ids)
        codes[cell] = np.column_stack([index_codes(pd.to_numeric(block[t], errors="coerce"), opts[t])
                                       for t in TARGETS])
        weights[cell] = np.ones(len(ids))
    invalid = responses.assign(invalid=~responses.valid.astype(str).eq("True")).groupby(
        ["cell", "item"]).invalid.mean()
    return Panel(codes, weights), invalid


def modal_panel(human: Panel, profiles: pd.DataFrame, opts: dict[str, list[str]]) -> Panel:
    modes = []
    for j, t in enumerate(TARGETS):
        k = len(opts[t])
        total = sum(np.bincount(human.codes[c][:, j][human.codes[c][:, j] >= 0],
                                human.weights[c][human.codes[c][:, j] >= 0], minlength=k) for c in CELLS)
        modes.append(int(np.argmax(total)))
    codes = {c: np.tile(modes, ((profiles.cell == c).sum(), 1)) for c in CELLS}
    return Panel(codes, {c: np.ones(len(codes[c])) for c in CELLS})


def cell_tvs(human: Panel, model: Panel, opts, hrows=None, mrows=None) -> np.ndarray:
    out = np.empty((len(CELLS), len(TARGETS)))
    for g, cell in enumerate(CELLS):
        for j, t in enumerate(TARGETS):
            k = len(opts[t])
            out[g, j] = tv(human.dist(cell, j, k, None if hrows is None else hrows[g]),
                           model.dist(cell, j, k, None if mrows is None else mrows[g]))
    return out


def human_floor(human: Panel, opts, rng) -> np.ndarray:
    total = np.zeros((len(CELLS), len(TARGETS)))
    for _ in range(N_SPLITS):
        for g, cell in enumerate(CELLS):
            order = rng.permutation(len(human.weights[cell]))
            a, b = order[: len(order) // 2], order[len(order) // 2:]
            for j, t in enumerate(TARGETS):
                k = len(opts[t])
                total[g, j] += tv(human.dist(cell, j, k, a), human.dist(cell, j, k, b))
    return total / N_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--smoke", action="store_true", help="score the <model>-smoke runs")
    args = parser.parse_args()
    runs = [m + ("-smoke" if args.smoke else "") for m in args.models]

    rng = np.random.default_rng(SEED)
    opts = options()
    frame = human_data()
    human, cell_n = human_panel(frame, opts)
    profiles = pd.read_csv(FROZEN / "profiles.csv")
    if args.smoke:
        answered = set(pd.read_csv(RUNS / runs[0] / "responses.csv").profile_id)
        profiles = profiles[profiles.profile_id.isin(answered)]
    cell_share = frame.groupby("cell").w.sum().reindex(list(CELLS))
    cell_share = (cell_share / cell_share.sum()).to_numpy()

    panels, invalid = {}, {}
    for name in runs:
        panels[name], invalid[name] = model_panel(name, profiles, opts)
    panels["modal-baseline"] = modal_panel(human, profiles, opts)

    point = {name: cell_tvs(human, panel, opts) for name, panel in panels.items()}
    point["human-floor"] = human_floor(human, opts, rng)

    # Paired bootstrap of the headline (mean of 150 TVs).
    hsize = {c: len(human.weights[c]) for c in CELLS}
    msize = {c: (profiles.cell == c).sum() for c in CELLS}
    boot = {name: np.empty(N_BOOT) for name in panels}
    for b in range(N_BOOT):
        hrows = [rng.integers(0, hsize[c], hsize[c]) for c in CELLS]
        mrows = [rng.integers(0, msize[c], msize[c]) for c in CELLS]
        for name, panel in panels.items():
            boot[name][b] = np.nanmean(cell_tvs(human, panel, opts, hrows, mrows))

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, values in point.items():
        for g, cell in enumerate(CELLS):
            block = frame[frame.cell == cell]
            for j, t in enumerate(TARGETS):
                asked = block[f"cps25_{t}"].notna()
                w = block.w[asked]
                rows.append({"model": name, "cell": cell, "item": t, "tv": round(values[g, j], 4),
                             "human_n": int(asked.sum()),
                             "human_n_eff": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
                             "model_valid_n": int((panels[name].codes[cell][:, j] >= 0).sum())
                             if name in panels else None,
                             "invalid_rate": round(float(invalid[name].get((cell, t), np.nan)), 4)
                             if name in invalid else None})
    pd.DataFrame(rows).to_csv(RESULTS / "tv_by_cell.csv", index=False)

    modal = point["modal-baseline"]
    summary = []
    for name, values in point.items():
        lo, hi = (basic_ci(float(np.nanmean(values)), boot[name]) if name in boot
                  else (np.nan, np.nan))
        summary.append({"model": name, "mean_tv_150": round(float(np.nanmean(values)), 4),
                        "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                        "cell_weighted_tv": round(float(np.nansum(values.mean(axis=1) * cell_share)), 4),
                        "cells_better_than_modal": int((values < modal - 1e-12).sum()),
                        **{f"tv_{t}": round(float(values[:, j].mean()), 4) for j, t in enumerate(TARGETS)}})
    summary = pd.DataFrame(summary).sort_values("mean_tv_150")
    summary.to_csv(RESULTS / "summary.csv", index=False)

    contrasts = []
    for a, b in itertools.combinations(panels, 2):
        estimate = float(np.nanmean(point[a]) - np.nanmean(point[b]))
        lo, hi = basic_ci(estimate, boot[a] - boot[b])
        contrasts.append({"model_a": a, "model_b": b, "diff": round(estimate, 4),
                          "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)})
    pd.DataFrame(contrasts).to_csv(RESULTS / "contrasts.csv", index=False)
    print(summary[["model", "mean_tv_150", "ci_low", "ci_high", "cells_better_than_modal"]].to_string(index=False))


if __name__ == "__main__":
    main()
