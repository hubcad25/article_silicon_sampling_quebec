"""Build analysis-ready distributions after all five inference arms are fetched.

Usage:
    .venv/bin/python scripts/21_build_distributions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.evaluation import (  # noqa: E402
    ANALYSIS_ROOT,
    INFERENCE_ROOT,
    write_distribution_outputs,
)
from article_silicon_sampling_quebec.inference import DEFAULT_DRAWS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, default=INFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument(
        "--arms", nargs="+", choices=("R", "A", "B0", "B", "BS"),
        help="Optional subset for an explicitly interim analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_distribution_outputs(
        inference_root=args.inference_root,
        output_root=args.output_root,
        draws_per_group=args.draws,
        arms=args.arms,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
