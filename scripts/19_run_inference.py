"""Run the frozen ADR 0001 C0 inference pilot through FoundryChat."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.c0_inference import (  # noqa: E402
    DEFAULT_DEPLOYMENT,
    DEFAULT_DRAWS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURES,
    DEFAULT_WORKERS,
    RunSettings,
    run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "data" / "analysis" / "c0_inference.csv")
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit-item-cell-pairs", type=int)
    parser.add_argument(
        "--delete-deployment-after",
        action="store_true",
        help="Delete the Azure deployment when the run exits, including after failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO / ".env")
    settings = RunSettings(
        deployment=args.deployment, model=args.model,
        temperatures=tuple(args.temperatures), draws=args.draws,
        workers=args.workers, max_tokens=args.max_tokens, top_p=args.top_p,
        limit_item_cell_pairs=args.limit_item_cell_pairs,
    )
    try:
        summary = run(settings, args.output)
    finally:
        if args.delete_deployment_after:
            subprocess.run(
                [
                    "az", "cognitiveservices", "account", "deployment", "delete",
                    "-g", "rg-opubliq-sondages", "-n", "info-4552-resource",
                    "--deployment-name", args.deployment,
                ],
                check=True,
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
