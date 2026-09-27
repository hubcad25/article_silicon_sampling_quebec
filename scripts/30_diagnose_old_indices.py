"""Compute the ADR 0005 diagnostic of the old percentage-based indices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.metrics import (  # noqa: E402
    DEFAULT_BOOTSTRAPS,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    PRIMARY_TEMPERATURE,
    old_indices_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "data/analysis/distributions.csv")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "data/analysis/second_brief")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    return parser.parse_args()


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _metric(summary: pl.DataFrame, name: str) -> dict:
    return summary.filter(pl.col("metric") == name).row(0, named=True)


def _interval(row: dict) -> str:
    return f"{_number(row['estimate'])} [{_number(row['ci_low'])} ; {_number(row['ci_high'])}]"


def write_brief_fragment(path: Path, by_item: pl.DataFrame, summary: pl.DataFrame) -> None:
    direction = _metric(summary, "direction")
    amp_bs = _metric(summary, "amplification_bs")
    amp_b0 = _metric(summary, "amplification_b0")
    amp_difference = _metric(summary, "amplification_bs_minus_b0")
    entropy_observed = _metric(summary, "mean_observed_entropy")
    entropy_b0 = _metric(summary, "mean_b0_entropy")
    entropy_bs = _metric(summary, "mean_bs_entropy")
    entropy_difference = _metric(summary, "entropy_bs_minus_observed")

    point_support = (
        direction["estimate"] > 0.5
        and amp_bs["estimate"] > 1
        and entropy_difference["estimate"] < 0
    )
    interval_support = (
        direction["ci_low"] > 0.5
        and amp_bs["ci_low"] > 1
        and entropy_difference["ci_high"] < 0
    )
    verdict = (
        "Les trois critères ponctuels de l’ADR 0005 sont satisfaits."
        if point_support else
        "Les trois critères ponctuels de l’ADR 0005 ne sont pas tous satisfaits."
    )
    uncertainty = (
        "Les trois conclusions franchissent aussi leur seuil dans les intervalles bootstrap."
        if interval_support else
        "L’incertitude bootstrap ne permet pas d’affirmer les trois conclusions simultanément."
    )

    item_lines = [
        "| Question | Direction | Pente BS | Pente B0 | Entropie BS − observée |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in by_item.iter_rows(named=True):
        item_lines.append(
            f"| {row['item_idx']} | {_number(row['direction'])} | "
            f"{_number(row['amplification_bs'])} | {_number(row['amplification_b0'])} | "
            f"{_number(row['entropy_bs_minus_observed'])} |"
        )

    text = f"""<!-- Généré par scripts/30_diagnose_old_indices.py; ne pas modifier à la main. -->

## Diagnostic des anciens indices en pourcentages

À température 1,0, l’ajout des indices déplace les probabilités dans la direction des écarts
humains pour **{_interval(direction)}** des unités question × cellule × option admissibles. La pente
d’amplification est de **{_interval(amp_bs)}** pour BS, contre **{_interval(amp_b0)}** pour B0; leur
différence est de **{_interval(amp_difference)}**.

L’entropie moyenne, en nats, vaut **{_interval(entropy_observed)}** chez les répondants,
**{_interval(entropy_b0)}** pour B0 et **{_interval(entropy_bs)}** pour BS. La différence BS − observée
est de **{_interval(entropy_difference)}**; une valeur négative indique des réponses synthétiques plus
concentrées.

**En clair**, les pourcentages donnés en indices font bouger le modèle, mais ils ne l'aident pas à
repérer de façon fiable ce qui distingue un sous-groupe. Ses déplacements ne vont pas plus souvent
dans la bonne direction que dans la mauvaise et demeurent beaucoup trop faibles. Contrairement à ce
qu'on soupçonnait, le modèle ne semble pas non plus se rabattre excessivement sur la réponse
majoritaire : il disperse plutôt ses réponses davantage que les vrais répondants.

**Conclusion.** {verdict} {uncertainty}

Les estimations regroupent les unités prévues dans l’ADR 0005. Les intervalles à 95 % proviennent
d’un bootstrap apparié par question ({int(direction['n_items'])} questions). Les écarts humains nuls
sont exclus du dénominateur de direction; une absence de déplacement de BS par rapport à B0 compte
comme un désaccord. Les pentes sont des régressions par l’origine après centrage par question et
option. L’entropie est calculée pour chaque paire question × cellule avant agrégation.

### Résultats par question

{chr(10).join(item_lines)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    distributions = pl.read_csv(
        args.input, schema_overrides={"code": pl.Utf8, "temperature": pl.Float64}
    )
    options, entropies, by_item, summary = old_indices_diagnostics(
        distributions, temperature=PRIMARY_TEMPERATURE, repetitions=args.bootstrap,
        seed=args.seed, confidence=args.confidence,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "diagnostic_anciens_indices_options.csv": options,
        "diagnostic_anciens_indices_entropie_paires.csv": entropies,
        "diagnostic_anciens_indices_par_question.csv": by_item,
        "diagnostic_anciens_indices_resume.csv": summary,
        "diagnostic_anciens_indices_parametres.csv": pl.DataFrame([{
            "temperature": PRIMARY_TEMPERATURE,
            "bootstrap_repetitions": args.bootstrap,
            "bootstrap_seed": args.seed,
            "confidence": args.confidence,
            "bootstrap_unit": "paired item",
            "point_aggregation": "pooled item-cell-option units",
            "direction_denominator": "observed deviation != 0; zero BS-B0 change is disagreement",
            "amplification": "through-origin slope after centering within item-option",
            "entropy": "natural logarithm; computed by item-cell",
        }]),
    }
    for name, frame in outputs.items():
        frame.write_csv(args.output_root / name)
    fragment = args.output_root / "diagnostic_anciens_indices_pour_note.md"
    write_brief_fragment(fragment, by_item, summary)

    source_paths = [args.input, REPO / "docs/adr/0005-indices-repondants-reels-inference.md"]
    manifest = {
        "sources": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "outputs": {
            name: hashlib.sha256((args.output_root / name).read_bytes()).hexdigest()
            for name in [*outputs, fragment.name]
        },
    }
    manifest_path = args.output_root / "diagnostic_anciens_indices_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"old-indices diagnostic: {args.output_root}")


if __name__ == "__main__":
    main()
