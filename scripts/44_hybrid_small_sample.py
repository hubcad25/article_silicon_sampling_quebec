"""Small biased sample + dossier model C (docs/test_hybride_petit_echantillon.md).

The CES 2025 full sample is treated as truth. We repeatedly draw small real
samples, estimate the 15 region-by-age cells, and compare the sample alone,
model C alone, and two hybrids. No model is called.

    source .venv/bin/activate
    python scripts/44_hybrid_small_sample.py

Writes ``data/analysis/hybrid_small_sample.csv``. The CSV is deliberately long:
one row per series, Trump-question scope, sampling mechanism, sample size,
repetition, method, and (for the Dirichlet hybrid) prior strength k.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmark import CELLS, REGION_OF, SETS, age_band  # noqa: E402
from benchmark.dossier import parse as parse_dossier  # noqa: E402
from article_silicon_sampling_quebec.corpus.blob import read_survey  # noqa: E402

OUT = REPO / "data" / "analysis" / "hybrid_small_sample.csv"
MODEL = "bench-ces25-dossier-1k-s1"
SEED = 20261019
REPETITIONS = 100
HEADLINE_NS = (50, 100, 200, 400)
# Extra points only make the equivalent-size calibration less likely to end in
# an uninformative ">400". They remain in the CSV but not in the headline table.
CALIBRATION_NS = (800, 1600)
ALL_NS = HEADLINE_NS + CALIBRATION_NS
KS = (5, 10, 20, 40)
TRUMP = frozenset({"tarrifs_1", "tarrifs_2"})
INTEREST = "cps25_interest_gen_1"
BIAS_CUTOFF = 8
BIAS_MULTIPLIER = 2.0

PROVINCES = {
    1: "ab", 2: "bc", 3: "mb", 4: "nb", 5: "nl", 7: "ns",
    9: "on", 10: "pe", 11: "qc", 12: "sk",
}


def option_codes(set_name: str) -> dict[str, list[str]]:
    """Read option codes from the frozen requests without loading model runs."""
    folder = "frozen" if set_name == "main" else "frozen_novel"
    out: dict[str, list[str]] = {}
    with (REPO / "benchmark" / folder / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            item = record["item"]
            out.setdefault(item, sorted(set(record["options"].values()), key=int))
            if len(out) == len(SETS[set_name]):
                break
    return out


def index_codes(values: pd.Series, options: list[str]) -> np.ndarray:
    lookup = {code: i for i, code in enumerate(options)}
    return np.array([
        lookup.get(str(int(value)) if pd.notna(value) and value != "" else None, -1)
        for value in values
    ], dtype=np.int16)


def shares(codes: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray | None:
    keep = codes >= 0
    if not keep.any():
        return None
    mass = np.bincount(codes[keep], weights=weights[keep], minlength=k)
    return mass / mass.sum() if mass.sum() > 0 else None


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def load_frame() -> pd.DataFrame:
    variables = [v for values in SETS.values() for v in values]
    columns = ["cps25_province", "cps25_age_in_years", "cps25_weight_general_all", INTEREST,
               *variables]
    frame = read_survey("ces_2025", columns).to_pandas()
    frame["region"] = frame.cps25_province.map(PROVINCES).map(REGION_OF)
    frame["age_group"] = [age_band(age) for age in frame.cps25_age_in_years]
    frame = frame[
        frame.region.notna()
        & frame.age_group.notna()
        & (frame.cps25_weight_general_all > 0)
    ].copy()
    frame["cell"] = frame.region + "|" + frame.age_group
    frame["w"] = frame.cps25_weight_general_all.astype(float)
    return frame.reset_index(drop=True)


def load_model_c(set_name: str, options: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Mean the three valid C draws into item x cell x option arrays."""
    path = (REPO / "benchmark" / "results_dossier" / set_name
            / f"responses_{MODEL}.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    out: dict[str, np.ndarray] = {}
    for item, codes in options.items():
        draws = [parse_dossier(r["raw"], codes) for r in records
                 if r.get("item") == item and r.get("valid")]
        draws = [draw for draw in draws if draw is not None]
        if not draws:
            raise RuntimeError(f"No valid C draw for {set_name}/{item}")
        out[item] = np.stack([
            np.mean([draw[cell] for draw in draws], axis=0) for cell in CELLS
        ])
    return out


def balanced_stratified_sample(
    rng: np.random.Generator, cell_rows: list[np.ndarray], n: int,
) -> np.ndarray:
    """Allocate n as evenly as possible over the 15 cells, then sample within cells."""
    counts = np.full(len(CELLS), n // len(CELLS), dtype=int)
    counts[: n % len(CELLS)] += 1
    return np.concatenate([
        rng.choice(rows, size=count, replace=False) for rows, count in zip(cell_rows, counts)
    ])


def draw_sample(
    rng: np.random.Generator,
    mechanism: str,
    n: int,
    population_n: int,
    bias_prob: np.ndarray,
    cell_rows: list[np.ndarray],
) -> np.ndarray:
    if mechanism == "random":
        return rng.choice(population_n, size=n, replace=False)
    if mechanism == "biased":
        return rng.choice(population_n, size=n, replace=False, p=bias_prob)
    if mechanism == "stratified":
        return balanced_stratified_sample(rng, cell_rows, n)
    raise ValueError(mechanism)


def rake_recentered(initial: np.ndarray, cell_weights: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Project recentered C cells onto valid rows with the requested national margin.

    The additive recentering already has the right row and weighted-column sums,
    but may contain negatives. Clipping followed by iterative proportional fitting
    restores non-negativity while preserving both sets of margins.
    """
    positive_columns = target > 0
    mass = np.maximum(initial, 1e-12) * cell_weights[:, None]
    mass[:, ~positive_columns] = 0.0
    for _ in range(1000):
        row_sum = mass.sum(axis=1)
        mass *= (cell_weights / row_sum)[:, None]
        col_sum = mass.sum(axis=0)
        mass[:, positive_columns] *= (target[positive_columns] / col_sum[positive_columns])[None, :]
        if (np.max(np.abs(mass.sum(axis=1) - cell_weights)) < 1e-12
                and np.max(np.abs(mass.sum(axis=0) - target)) < 1e-12):
            break
    result = mass / cell_weights[:, None]
    if not (np.allclose(result.sum(axis=1), 1, atol=1e-9)
            and np.allclose((result * cell_weights[:, None]).sum(axis=0), target, atol=1e-9)):
        raise RuntimeError("Level-plus-gaps projection did not converge")
    return result


def item_estimates(
    sample: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
    cell_index: np.ndarray,
    cell_weights: np.ndarray,
    model: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Return sample-only cells, Bayesian cells by k, and level-plus-gaps cells."""
    k_options = model.shape[1]
    national = shares(codes[sample], weights[sample], k_options)
    if national is None:  # Defensive only; no benchmark item reaches this branch in the simulation.
        national = (model * cell_weights[:, None]).sum(axis=0)

    observed = np.empty_like(model)
    valid_n = np.zeros(len(CELLS), dtype=int)
    for g in range(len(CELLS)):
        rows = sample[cell_index[sample] == g]
        dist = shares(codes[rows], weights[rows], k_options)
        observed[g] = national if dist is None else dist
        valid_n[g] = int((codes[rows] >= 0).sum())

    # Population-cell post-stratification, including the national fallback for empty cells.
    observed_national = (observed * cell_weights[:, None]).sum(axis=0)
    observed_national /= observed_national.sum()

    bayes = {
        prior_k: (prior_k * model + valid_n[:, None] * observed)
        / (prior_k + valid_n[:, None])
        for prior_k in KS
    }
    model_national = (model * cell_weights[:, None]).sum(axis=0)
    initial = model + observed_national - model_national
    level_gaps = rake_recentered(initial, cell_weights, observed_national)
    return observed, bayes, level_gaps


def score_cells(truth: dict[str, np.ndarray], estimates: dict[str, np.ndarray], items: list[str]) -> float:
    return float(np.mean([
        tv(truth[item][g], estimates[item][g])
        for item in items for g in range(len(CELLS))
    ]))


def add_rows(
    rows: list[dict], set_name: str, mechanism: str, n: int, repetition: int,
    truth: dict[str, np.ndarray], sample_est: dict[str, np.ndarray],
    model_est: dict[str, np.ndarray], bayes_est: dict[int, dict[str, np.ndarray]],
    gap_est: dict[str, np.ndarray],
) -> None:
    scopes = {"all": list(model_est)}
    if set_name == "novel":
        scopes["without_trump"] = [item for item in model_est if item not in TRUMP]
    base = {"series": set_name, "mechanism": mechanism, "n": n, "repetition": repetition}
    for scope, items in scopes.items():
        common = {**base, "question_scope": scope}
        rows.append({**common, "method": "sample_only", "k": None,
                     "tv": score_cells(truth, sample_est, items)})
        rows.append({**common, "method": "c_only", "k": None,
                     "tv": score_cells(truth, model_est, items)})
        rows.append({**common, "method": "level_plus_gaps", "k": None,
                     "tv": score_cells(truth, gap_est, items)})
        for prior_k in KS:
            rows.append({**common, "method": "bayesian", "k": prior_k,
                         "tv": score_cells(truth, bayes_est[prior_k], items)})


def mark_selected_k(result: pd.DataFrame) -> pd.DataFrame:
    """Choose k on repeated questions separately for each mechanism and n."""
    main = result[(result.series == "main") & (result.method == "bayesian")]
    means = main.groupby(["mechanism", "n", "k"], as_index=False).tv.mean()
    chosen = means.loc[means.groupby(["mechanism", "n"]).tv.idxmin(), ["mechanism", "n", "k"]]
    selected = {(row.mechanism, int(row.n)): int(row.k) for row in chosen.itertuples()}
    result["selected_on_main"] = pd.array([pd.NA] * len(result), dtype="boolean")
    bayes_rows = result.method == "bayesian"
    result.loc[bayes_rows, "selected_on_main"] = [
        int(k) == selected[(mechanism, int(n))]
        for mechanism, n, k in result.loc[bayes_rows, ["mechanism", "n", "k"]].itertuples(index=False)
    ]
    print("Selected k on repeated questions:")
    print(chosen.pivot(index="mechanism", columns="n", values="k").to_string())
    return result


def run() -> pd.DataFrame:
    frame = load_frame()
    weights = frame.w.to_numpy(float)
    cell_lookup = {cell: i for i, cell in enumerate(CELLS)}
    cell_index = frame.cell.map(cell_lookup).to_numpy(int)
    cell_rows = [np.flatnonzero(cell_index == g) for g in range(len(CELLS))]
    cell_weights = np.array([weights[cell_index == g].sum() for g in range(len(CELLS))])
    cell_weights /= cell_weights.sum()

    high_interest = pd.to_numeric(frame[INTEREST], errors="coerce").fillna(-99).to_numpy() >= BIAS_CUTOFF
    bias_prob = np.where(high_interest, BIAS_MULTIPLIER, 1.0)
    bias_prob /= bias_prob.sum()

    set_data = {}
    for set_name, variables in SETS.items():
        options = option_codes(set_name)
        items = [variable.split("_", 1)[1] for variable in variables]
        code_arrays = {
            item: index_codes(frame[variable], options[item])
            for item, variable in zip(items, variables)
        }
        truth = {}
        for item in items:
            truth[item] = np.stack([
                shares(code_arrays[item][cell_index == g], weights[cell_index == g], len(options[item]))
                for g in range(len(CELLS))
            ])
        set_data[set_name] = (items, code_arrays, truth, load_model_c(set_name, options))

    rows: list[dict] = []
    rng = np.random.default_rng(SEED)
    for mechanism in ("random", "biased", "stratified"):
        for n in ALL_NS:
            for repetition in range(REPETITIONS):
                sample = draw_sample(rng, mechanism, n, len(frame), bias_prob, cell_rows)
                for set_name, (items, code_arrays, truth, model) in set_data.items():
                    sample_est, gap_est = {}, {}
                    bayes_est = {prior_k: {} for prior_k in KS}
                    for item in items:
                        observed, bayes, level_gaps = item_estimates(
                            sample, code_arrays[item], weights, cell_index, cell_weights, model[item]
                        )
                        sample_est[item] = observed
                        gap_est[item] = level_gaps
                        for prior_k in KS:
                            bayes_est[prior_k][item] = bayes[prior_k]
                    add_rows(rows, set_name, mechanism, n, repetition, truth, sample_est,
                             model, bayes_est, gap_est)
            print(f"{mechanism:>10} n={n}: {REPETITIONS} repetitions", flush=True)

    result = mark_selected_k(pd.DataFrame(rows))
    result["tv"] = result.tv.round(6)
    result["seed"] = SEED
    result["bias_definition"] = (
        f"without replacement; inclusion propensity x{BIAS_MULTIPLIER:g} if {INTEREST}>={BIAS_CUTOFF}"
    )
    result = result[["series", "question_scope", "mechanism", "n", "repetition", "method", "k",
                     "selected_on_main", "tv", "seed", "bias_definition"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"Wrote {len(result):,} rows to {OUT.relative_to(REPO)}")
    return result


if __name__ == "__main__":
    run()
