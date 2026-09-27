"""Generate clearly labelled fake inference campaigns with the production CSV schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.evaluation import ARM_PATHS, build_observed_tables  # noqa: E402
from article_silicon_sampling_quebec.inference import (  # noqa: E402
    ARMS,
    CAMPAIGNS,
    DEFAULT_DRAWS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURES,
    HELDOUT_PATH,
    RESULT_FIELDS,
    build_item_cells,
)


ARM_BLEND = {"A": 0.15, "B": 0.08, "B0": 0.22, "R": 0.35, "BS": 0.1}
ARM_INVALID_RATE = {"A": 0.01, "B": 0.015, "B0": 0.02, "R": 0.04, "BS": 0.015}
ARM_CAMPAIGNS = {
    "A": "c0-8k", "B": "c1-8k", "B0": "c1-8k", "R": "base", "BS": "c1-8k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "data" / "analysis" / "fake" / "inference")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=20_260_924)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.draws < 1:
        raise SystemExit("--draws must be positive")
    rng = np.random.default_rng(args.seed)
    pairs = build_item_cells()
    observed, _ = build_observed_tables(pairs, pl.read_parquet(HELDOUT_PATH))
    observed_by_pair = {
        (int(item_idx), str(cell)): group.sort("code")
        for (item_idx, cell), group in observed.group_by("item_idx", "cell")
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    warning = "SYNTHETIC DATA FOR PIPELINE TESTING ONLY. DO NOT REPORT AS RESULTS.\n"
    (args.output_root / "FAKE_DATA.txt").write_text(warning, encoding="utf-8")
    (args.output_root.parent / "FAKE_DATA.txt").write_text(warning, encoding="utf-8")

    for arm, relative_path in ARM_PATHS.items():
        output = args.output_root / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        campaign = CAMPAIGNS[ARM_CAMPAIGNS[arm]]
        diagnostics = []
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for pair in pairs:
                truth = observed_by_pair[(pair.item_idx, pair.cell)]
                options = truth["code"].to_list()
                labels = {option.code: option.label for option in pair.item.options}
                p = truth["share"].to_numpy()
                base = (1 - ARM_BLEND[arm]) * p + ARM_BLEND[arm] / len(p)
                for temperature in DEFAULT_TEMPERATURES:
                    synthetic_p = np.power(base, 1 / temperature)
                    synthetic_p /= synthetic_p.sum()
                    invalid_rate = min(
                        ARM_INVALID_RATE[arm] + 0.005 * (pair.item_idx % 3), 0.2
                    )
                    valid_n = 0
                    for draw_idx in range(args.draws):
                        invalid = rng.random() < invalid_rate
                        code = None if invalid else str(rng.choice(options, p=synthetic_p))
                        valid_n += not invalid
                        temperature_key = format(temperature, ".12g")
                        writer.writerow({
                            "draw_key": f"{pair.item_idx}|{pair.cell}|{temperature_key}|{draw_idx}",
                            "arm": arm,
                            "deployment": campaign.deployment,
                            "model": campaign.model,
                            "condition": ARMS[arm].model_condition,
                            "context": ARMS[arm].context,
                            "n_context": 0 if arm not in {"B", "BS"} or pair.item_idx == 7 else 6,
                            "temperature": temperature,
                            "top_p": 1.0,
                            "max_tokens": DEFAULT_MAX_TOKENS,
                            "item_idx": pair.item_idx,
                            "block": pair.block,
                            "survey_id": pair.survey_id,
                            "variable": pair.variable,
                            "language": pair.language,
                            "cell": pair.cell,
                            "heldout_valid_n": pair.heldout_valid_n,
                            "draw_idx": draw_idx,
                            "raw_response": "FAKE INVALID" if invalid else labels[code],
                            "matched_code": code,
                            "valid": not invalid,
                            "started_at": "FAKE",
                            "completed_at": "FAKE",
                            "latency_seconds": 0,
                            "client_retries_total": 0,
                            "client_throttled_total": 0,
                        })
                    diagnostics.append({
                        "item_idx": pair.item_idx, "cell": pair.cell,
                        "temperature": temperature, "transport_n": args.draws,
                        "effective_n": valid_n, "invalid_n": args.draws - valid_n,
                        "invalid_rate": (args.draws - valid_n) / args.draws,
                        "expected_transport_n": args.draws, "coverage_ok": True,
                    })
        pl.DataFrame(diagnostics).write_csv(output.with_suffix(".diagnostics.csv"))
        output.with_suffix(".manifest.json").write_text(json.dumps({
            "synthetic": True,
            "warning": "FOR PIPELINE TESTING ONLY - DO NOT REPORT",
            "settings": {
                "arm": arm, "draws": args.draws,
                "temperatures": list(DEFAULT_TEMPERATURES), "seed": args.seed,
            },
        }, indent=2) + "\n", encoding="utf-8")
        print(f"{arm}: {output}")


if __name__ == "__main__":
    main()
