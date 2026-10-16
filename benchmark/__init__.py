"""CES 2025 benchmark: any model -> 150 subgroup TVs (docs/ces2025_benchmark_framework.md).

    python -m benchmark.build                     # once: the frozen inference set
    python -m benchmark.run --model NAME          # one model's answers, resumable
    python -m benchmark.score --model NAME ...    # TVs, references, bootstrap

``build`` renders every prompt with the training template, so ``run`` only
moves messages to a backend and ``score`` only reads answers and CES 2025.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
#: Target set: "main" = the ten items of the plan (all asked in CES 2021);
#: "novel" = eight CES 2025 items never asked in the training surveys
#: (scripts/39_novel_ces2025_items.py, chosen 2026-10-16). Each set has its own
#: frozen/, runs/ and results/ folders.
SET = os.environ.get("BENCH_SET", "main")
SETS = {
    "main": ("cps25_spend_afford_h", "cps25_spend_nation_c", "cps25_spend_defence",
             "cps25_spend_rec_indi", "cps25_pos_life", "cps25_pos_energy", "cps25_pos_jobs",
             "cps25_imm", "cps25_demsat", "cps25_own_fin_retro"),
    "novel": ("cps25_tarrifs_1", "cps25_tarrifs_2", "cps25_foreign_pol_protect",
              "cps25_safe_injection", "cps25_student_protes", "cps25_resident_2f",
              "cps25_rent_security", "pes25_decent_housing"),
}
TARGET_VARS = SETS[SET]
#: Context given to each profile: "fix" = the 7 anchors of the plan (A models);
#: "rich" = 12 core themes + the 6 DC 2024 items nearest the target (B models,
#: scripts/40). Folders get a "_rich" suffix.
CONTEXT = os.environ.get("BENCH_CONTEXT", "fix")
SUFFIX = ("" if SET == "main" else f"_{SET}") + ("" if CONTEXT == "fix" else f"_{CONTEXT}")
#: Overridable so a cloud container reads the frozen set and writes runs on the share.
FROZEN = Path(os.environ.get("BENCH_FROZEN", ROOT / f"frozen{SUFFIX}"))
RUNS = Path(os.environ.get("BENCH_RUNS", ROOT / f"runs{SUFFIX}"))
RESULTS = ROOT / f"results{SUFFIX}"

REGIONS = ("BC", "Prairies", "ON", "QC", "Atlantic")
AGES = ("18-34", "35-54", "55+")
CELLS = tuple(f"{r}|{a}" for r in REGIONS for a in AGES)
#: Short item names (variable without its cps25_/pes25_ prefix).
TARGETS = tuple(v.split("_", 1)[1] for v in TARGET_VARS)
#: Canonical region codes of the SES crosswalk -> benchmark region.
REGION_OF = {"bc": "BC", "ab": "Prairies", "sk": "Prairies", "mb": "Prairies", "on": "ON",
             "qc": "QC", "nb": "Atlantic", "nl": "Atlantic", "ns": "Atlantic", "pe": "Atlantic"}


def age_band(age: float) -> str | None:
    if age is None or age < 18:
        return None
    return "18-34" if age < 35 else "35-54" if age < 55 else "55+"
