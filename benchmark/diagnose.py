"""Diagnostics on scored runs: strata means, national level, subgroup differences.

    python -m benchmark.diagnose ces25-fix-50k ces25-sem-50k ... llama33-70b-base

Reads the runs and CES 2025 like ``score``; writes benchmark/results/:
    strata.csv          per model: mean TV by region and by age group (from tv_by_cell.csv)
    national.csv        per model x item: TV between the model's and CES 2025's national
                        distribution (cells pooled with their CES 2025 weight share)
    subgroup_signal.csv per model: how well the model reproduces the gaps between cells.
                        A gap is a cell's share of an option minus the national share
                        (human gaps from CES 2025, model gaps from the model's own
                        national share). slope = OLS slope of model gaps on human gaps
                        (1 = full size, 0 = no difference between cells), corr = their
                        correlation, also restricted to region and age margins.
                        hybrid_tv = mean TV when each cell gets the CES 2025
                        national distribution plus the model's gap for that cell
                        (hybrid_x2_tv: the gap doubled).

Also scores ``national-baseline``: every cell gets the CES 2025 national
distribution. Like ``modal-baseline`` it is picked from the evaluation data:
a diagnostic, not a forecast.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import AGES, CELLS, FROZEN, REGIONS, RESULTS, TARGETS
from .score import cell_tvs, human_data, human_panel, model_panel, options, tv


def distributions(panel, opts) -> dict[str, list[np.ndarray]]:
    return {c: [panel.dist(c, j, len(opts[t])) for j, t in enumerate(TARGETS)] for c in CELLS}


def pooled(dists, share, cells) -> list[np.ndarray]:
    weights = np.array([share[c] for c in cells])
    weights = weights / weights.sum()
    return [sum(w * dists[c][j] for w, c in zip(weights, cells)) for j in range(len(TARGETS))]


def gaps(dists, share, level) -> np.ndarray:
    """Gaps of each margin (cell, region or age group) from the national share, flattened."""
    groups = {"cell": {c: [c] for c in CELLS},
              "region": {r: [c for c in CELLS if c.startswith(r + "|")] for r in REGIONS},
              "age": {a: [c for c in CELLS if c.endswith("|" + a)] for a in AGES}}[level]
    nat = pooled(dists, share, CELLS)
    return np.concatenate([np.concatenate([d - n for d, n in zip(pooled(dists, share, members), nat)])
                           for members in groups.values()])


def hybrid(h_nat: np.ndarray, m_cell: np.ndarray, m_nat: np.ndarray, k: float) -> np.ndarray:
    """CES 2025 national share plus k times the model's gap for this cell, clipped and renormalized."""
    q = np.clip(h_nat + k * (m_cell - m_nat), 0, None)
    return q / q.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    opts = options()
    frame = human_data()
    human, _ = human_panel(frame, opts)
    profiles = pd.read_csv(FROZEN / "profiles.csv")
    share = (frame.groupby("cell").w.sum() / frame.w.sum()).to_dict()
    hd = distributions(human, opts)
    h_nat = pooled(hd, share, CELLS)

    by_cell = pd.read_csv(RESULTS / "tv_by_cell.csv")
    by_cell[["region", "age"]] = by_cell.cell.str.split("|", expand=True)
    strata = pd.concat([by_cell.groupby(["model", k]).tv.mean().unstack() for k in ("region", "age")], axis=1)
    strata = strata[list(REGIONS) + list(AGES)]

    nat_tv = np.array([[tv(hd[c][j], h_nat[j]) for j in range(len(TARGETS))] for c in CELLS])
    strata.loc["national-baseline"] = [nat_tv[[g for g, c in enumerate(CELLS) if c.split("|")[i] == v]].mean()
                                       for i, values in ((0, REGIONS), (1, AGES)) for v in values]
    strata.insert(0, "mean", by_cell.groupby("model").tv.mean())
    strata.loc["national-baseline", "mean"] = nat_tv.mean()
    strata.round(4).to_csv(RESULTS / "strata.csv")

    national, signal = [], []
    for name in args.models:
        panel, _ = model_panel(name, profiles, opts)
        md = distributions(panel, opts)
        m_nat = pooled(md, share, CELLS)
        for j, t in enumerate(TARGETS):
            national.append({"model": name, "item": t, "national_tv": round(tv(h_nat[j], m_nat[j]), 4)})
        row = {"model": name}
        for level in ("cell", "region", "age"):
            h, m = gaps(hd, share, level), gaps(md, share, level)
            row[f"slope_{level}"] = round(float(np.polyfit(h, m, 1)[0]), 3)
            row[f"corr_{level}"] = round(float(np.corrcoef(h, m)[0, 1]), 3)
        for label, k in (("hybrid_tv", 1.0), ("hybrid_x2_tv", 2.0)):
            row[label] = round(float(np.mean([[tv(hd[c][j], hybrid(h_nat[j], md[c][j], m_nat[j], k))
                                                for j in range(len(TARGETS))] for c in CELLS])), 4)
        row["human_gap_sd"] = round(float(gaps(hd, share, "cell").std()), 4)
        row["model_gap_sd"] = round(float(gaps(md, share, "cell").std()), 4)
        signal.append(row)
        print(name, row)
    national = pd.DataFrame(national)
    national.to_csv(RESULTS / "national.csv", index=False)
    pd.DataFrame(signal).to_csv(RESULTS / "subgroup_signal.csv", index=False)
    print(strata.round(3).to_string())
    print(national.pivot(index="item", columns="model", values="national_tv").round(3).to_string())
    print("national-baseline mean TV", round(nat_tv.mean(), 4), "by item",
          dict(zip(TARGETS, nat_tv.mean(0).round(3))))


if __name__ == "__main__":
    main()
