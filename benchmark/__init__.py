"""CES 2025 benchmark: any model -> 150 subgroup TVs (docs/ces2025_benchmark_framework.md).

    python -m benchmark.build                     # once: the frozen inference set
    python -m benchmark.run --model NAME          # one model's answers, resumable
    python -m benchmark.score --model NAME ...    # TVs, references, bootstrap

``build`` renders every prompt with the training template, so ``run`` only
moves messages to a backend and ``score`` only reads answers and CES 2025.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
FROZEN = ROOT / "frozen"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

REGIONS = ("BC", "Prairies", "ON", "QC", "Atlantic")
AGES = ("18-34", "35-54", "55+")
CELLS = tuple(f"{r}|{a}" for r in REGIONS for a in AGES)
TARGETS = ("spend_afford_h", "spend_nation_c", "spend_defence", "spend_rec_indi", "pos_life",
           "pos_energy", "pos_jobs", "imm", "demsat", "own_fin_retro")
#: Canonical region codes of the SES crosswalk -> benchmark region.
REGION_OF = {"bc": "BC", "ab": "Prairies", "sk": "Prairies", "mb": "Prairies", "on": "ON",
             "qc": "QC", "nb": "Atlantic", "nl": "Atlantic", "ns": "Atlantic", "pe": "Atlantic"}


def age_band(age: float) -> str | None:
    if age is None or age < 18:
        return None
    return "18-34" if age < 35 else "35-54" if age < 55 else "55+"
