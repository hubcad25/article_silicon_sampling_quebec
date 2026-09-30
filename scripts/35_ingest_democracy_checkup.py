"""Ingest the Democracy Checkup 2019-2024 into the local corpus format.

The CES 2025 benchmark (docs/ces2025_benchmark_framework.md) trains on eight
federal surveys: CES 2019 online, CES 2021 and the Democracy Checkup (DC)
2019-2024. The DC waves are not in the shared corpus, so this script builds,
from the Borealis files in ``data/raw/dc_YYYY/`` (``dc_YYYY.dta`` and the
codebook PDF):

    data/cache/dc_YYYY.parquet        microdata, same layout as the corpus
                                      (raw columns + __respondent_id,
                                      __survey_id, __weight)
    data/items_extra.parquet          items.parquet-shaped rows, English
    data/extra_french_wording.json    French versions, ces_french_wording.json
                                      layout
    data/crosswalks/ses_crosswalk.json  one SES entry per DC wave (updated)

Wording comes from the codebook, never from the ``.dta`` variable labels: those
are cut at 80 characters and, in 2020-2021, replaced by summaries ("Wealth
gap"). Each codebook entry is ``variable wording`` followed by ``o label
(code)`` lines, once in English and once in French. An item is kept only if
every observed non-negative code is one of its options.

The CES left-right self-placement (a slider, absent from the catalogue) is
added to ``items_extra.parquet`` too: it is an anchor dimension.

Run::

    .venv/bin/python scripts/35_ingest_democracy_checkup.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from article_silicon_sampling_quebec.corpus.blob import CACHE_DIR  # noqa: E402
from article_silicon_sampling_quebec.prompts import guess_language  # noqa: E402

YEARS = range(2019, 2025)
RAW = REPO / "data" / "raw"
ITEMS_OUT = REPO / "data" / "items_extra.parquet"
FRENCH_OUT = REPO / "data" / "extra_french_wording.json"
CROSSWALK = REPO / "data" / "crosswalks" / "ses_crosswalk.json"

OPTION = re.compile(r"^\s*o\s+(.*?)\s*\((-?\d+)\)(?:\s*(?::|->)\s*\S+|\s*_+)?\s*$")
FOOTER = re.compile(r"^\s*(Page \d+ of \d+|\d+)\s*$")
LAYOUT = re.compile(r"\s{6,}|&nbsp")
SKIP_VARIABLE = re.compile(
    r"_DO_|_TEXT$|_t_|timing|consent|ResponseId|Status|IPAddress|Progress|Duration|Finished|"
    r"RecordedDate|StartDate|EndDate|UserLanguage|Q_Language|weight|_yob|^yob|age_in_years|"
    r"province|gender|education|income|language_\d|^language$|attention|postal",
    re.I,
)
#: Qualtrics piping and layout residue: an item carrying any of it is dropped,
#: never shown to the model half-rendered.
ARTEFACT = re.compile(r"\$\{|e://|Display This|Selected Choice|\[Display Order\]|_TEXT\b", re.I)
#: Survey-administration variables that parse like items.
ADMIN_VARIABLE = re.compile(r"^(dc\d\d_)?(wave|feedback|feeback)", re.I)
DK_LABEL = re.compile(r"don.?t know|prefer not|ne sais pas|préfère ne pas", re.I)

LR_ITEMS = {
    # survey, variable, year, respondent-language column not needed: catalogue items.
    ("ces_2019_online", "cps19_lr_scale_bef_1", 2019),
    ("ces_2021", "cps21_lr_scale_bef_1", 2021),
}
LR_EN = "In politics, people sometimes talk of left and right. Where would you place yourself on this scale?"
LR_FR = "En politique, on parle parfois de gauche et de droite. Où vous placeriez-vous sur cette échelle?"


def slider_options(left: str, right: str) -> list[dict]:
    return [{"code": i, "label": f"{i} - {left}" if i == 0 else f"{i} - {right}" if i == 10 else str(i)}
            for i in range(11)]


# --------------------------------------------------------------------------
# Codebook
# --------------------------------------------------------------------------


def codebook_text(year: int) -> list[str]:
    path = RAW / f"dc_{year}" / "codebook.txt"
    if not path.exists():
        subprocess.run(["pdftotext", "-layout", str(path.with_suffix(".pdf")), str(path)], check=True)
    return path.read_text(encoding="utf-8").splitlines()


def parse_codebook(lines: list[str], variables: set[str]) -> dict[str, list[dict]]:
    """Every entry of the codebook, in order, grouped by variable."""
    entries: dict[str, list[dict]] = {}
    current = None
    for line in lines:
        head = line.split(None, 1)
        if head and head[0] in variables and not line.startswith(" "):
            current = {"text": [head[1]] if len(head) > 1 else [], "options": [],
                       "layout": [], "closed": False}
            entries.setdefault(head[0], []).append(current)
            continue
        if current is None:
            continue
        match = OPTION.match(line)
        if match:
            current["options"].append((int(match.group(2)), " ".join(match.group(1).split())))
            current["closed"] = True
            continue
        if current["closed"] or FOOTER.match(line):
            continue
        if LAYOUT.search(line.strip()) or re.fullmatch(r"[\d\s]+", line.strip() or "x"):
            if line.strip():
                current["layout"].append(line.strip())
            continue
        if line.strip():
            current["text"].append(line.strip())
    for blocks in entries.values():
        for block in blocks:
            block["text"] = " ".join(" ".join(block["text"]).split())
    return entries


def endpoints(layout: list[str]) -> tuple[str, str] | None:
    """``Left ... Right`` from a slider's layout lines."""
    for line in layout:
        words = [w for w in re.split(r"\s{3,}", line)
                 if w and not re.fullmatch(r"[\d\s]+", w) and not DK_LABEL.search(w)
                 and not re.search(r"know|sais", w, re.I)]
        if len(words) == 2 and "&nbsp" not in line:
            return words[0], words[1]
    return None


# --------------------------------------------------------------------------
# One wave
# --------------------------------------------------------------------------


def ingest(year: int) -> tuple[list[dict], dict]:
    survey = f"dc_{year}"
    frame = pd.read_stata(RAW / survey / f"{survey}.dta", convert_categoricals=False)
    entries = parse_codebook(codebook_text(year), set(frame.columns))

    items, french = [], {}
    for variable, blocks in entries.items():
        if SKIP_VARIABLE.search(variable) or ADMIN_VARIABLE.search(variable) or not blocks[0]["text"]:
            continue
        values = pd.to_numeric(frame[variable], errors="coerce")
        observed = {int(v) for v in values.dropna().unique() if v >= 0 and float(v).is_integer()}
        if not observed:
            continue
        english = blocks[0]
        fr_block = next((b for b in blocks[1:] if guess_language(b["text"]) == "fr"), None)
        options = [{"code": c, "label": l} for c, l in english["options"]]
        fr_options = {str(c): l for c, l in fr_block["options"]} if fr_block else {}
        if not options and observed <= set(range(11)) and (ends := endpoints(english["layout"])):
            options = slider_options(*ends)
            fr_ends = endpoints(fr_block["layout"]) if fr_block else None
            fr_options = ({str(o["code"]): o["label"] for o in slider_options(*fr_ends)}
                          if fr_ends else {})
        texts = [english["text"], *(o["label"] for o in options),
                 fr_block["text"] if fr_block else "", *fr_options.values()]
        if any(ARTEFACT.search(t) for t in texts):
            continue
        codes = {o["code"] for o in options}
        if len(options) < 2 or len(codes) != len(options) or not observed <= codes:
            continue
        items.append({
            "survey_id": survey, "variable": variable, "question_text": english["text"],
            "question_text_source": "codebook_pdf", "display_label": None, "var_type": "single",
            "is_ordinal": False, "n_options": len(options),
            "n_options_substantive": sum(not DK_LABEL.search(o["label"]) for o in options),
            "options": json.dumps(options, ensure_ascii=False), "code_map": "{}",
            "themes": "[]", "concepts": "[]", "year": year, "language": "en",
            "n_respondents_survey": len(frame),
            "n_valid_responses": int(values.isin(list(codes)).sum()),
            "is_context_only": False,
        })
        complete = fr_block is not None and set(fr_options) == {str(c) for c in codes}
        french[variable] = {"status": "complete" if complete else "incomplete",
                            "question_text_fr": fr_block["text"] if fr_block else "",
                            "options_fr": fr_options, "source": "codebook_pdf", "notes": ""}

    micro = frame.copy()
    for column in micro.columns:
        if micro[column].dtype == object:
            micro[column] = micro[column].astype("string")
    weight = "quota_weight" if year == 2019 else f"dc{year % 100}_quota_weight"
    micro["__respondent_id"] = micro["ResponseId"].astype("string")
    micro["__survey_id"] = survey
    micro["__weight"] = pd.to_numeric(micro[weight], errors="coerce")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(micro).write_parquet(CACHE_DIR / f"{survey}.parquet")
    print(f"{survey}: {len(frame)} respondents · {len(entries)} codebook entries · "
          f"{len(items)} items · {sum(v['status'] == 'complete' for v in french.values())} in French")
    return items, french


# --------------------------------------------------------------------------
# SES crosswalk
# --------------------------------------------------------------------------


EDUCATION = {"1": "no_diploma", "2": "no_diploma", "3": "no_diploma", "4": "no_diploma",
             "5": "high_school", "6": "high_school", "7": "college_trades",
             "8": {"to": ["high_school", "college_trades"], "coarse": True,
                   "note": "« Some university » : aucun diplôme postsecondaire garanti."},
             "9": "bachelor", "10": "above_bachelor", "11": "above_bachelor", "12": "missing"}
PROVINCES = ["ab", "bc", "mb", "nb", "nl", "nt", "ns", "nu", "on", "pe", "qc", "sk", "yt"]
INCOME_8 = {"1": "under_60k", "2": "under_60k", "3": "under_60k", "4": "60k_100k",
            "5": {"to": ["60k_100k", "100k_plus"], "coarse": True,
                  "note": "Tranche 90 001-110 000 $ à cheval sur la borne canonique 100 k$."},
            "6": "100k_plus", "7": "100k_plus", "8": "100k_plus", "9": "missing"}


def categorical(variable: str, mapping: dict) -> dict:
    return {"kind": "categorical", "variable": variable, "map": mapping,
            "needs_review": False, "notes": [], "conventions": []}


def crosswalk_entry(year: int) -> dict:
    p = "" if year == 2019 else f"dc{year % 100}_"
    gender = "gender" if year in (2019,) else (f"{p}gender" if year in (2020, 2021) else f"{p}genderid")
    entry = {
        "gender": categorical(gender, {"1": "man", "2": "woman", "3": "non_binary_or_other",
                                       "4": "non_binary_or_other"}),
        "education": {**categorical(f"{p}education", EDUCATION),
                      "conventions": ["highest_completed_credential", "some_university_outside_qc"]},
    }
    offset = 13 if year == 2019 else 0
    entry["region"] = categorical(f"{p}province",
                                  {str(i + 1 + offset): prov for i, prov in enumerate(PROVINCES)})
    if year == 2020:
        entry["age"] = {"kind": "yob_code", "variable": "dc20_yob", "reference_year": 2020,
                        "missing_values": [], "valid_year_range": [1900, 2002], "code_offset": 1919,
                        "needs_review": False, "notes": [], "conventions": ["continuous_bracketing"]}
    else:
        entry["age"] = {"kind": "age_years", "variable": "age_in_years" if year in (2019, 2021)
                        else f"{p}age_in_years", "valid_age_range": [18, 110],
                        "needs_review": False, "notes": [], "conventions": ["continuous_bracketing"]}
    if year == 2019:
        entry["income"] = {"kind": "coalesce", "needs_review": False, "notes": [], "conventions": [],
                           "sources": [categorical("income_category_july_aug_wave", {
                               "1": "under_60k", "2": "under_60k", "3": "under_60k", "4": "60k_100k",
                               "5": INCOME_8["5"], "6": "100k_plus", "7": "100k_plus",
                               "8": "100k_plus", "9": "missing"})]}
        entry["language"] = categorical("language", {"68": "english", "69": "french",
                                                     **{str(c): "other" for c in range(70, 90)}})
    else:
        if year != 2020:
            entry["income"] = categorical(f"{p}income_category", INCOME_8)
        entry["language"] = {"kind": "coalesce", "needs_review": False, "notes": [
            "Cases à cocher : français d'abord, puis anglais, puis autre."], "conventions": [],
            "sources": [categorical(f"{p}language_2", {"1": "french"}),
                        categorical(f"{p}language_1", {"1": "english"}),
                        categorical(f"{p}language_3", {"1": "other"})]}
    return entry


# --------------------------------------------------------------------------


def lr_rows() -> tuple[list[dict], dict]:
    rows, french = [], {}
    for survey, variable, year in sorted(LR_ITEMS):
        options = slider_options("Left", "Right")
        rows.append({
            "survey_id": survey, "variable": variable, "question_text": LR_EN,
            "question_text_source": "codebook_pdf", "display_label": None, "var_type": "single",
            "is_ordinal": False, "n_options": 11, "n_options_substantive": 11,
            "options": json.dumps(options), "code_map": "{}", "themes": "[]", "concepts": "[]",
            "year": year, "language": "en", "n_respondents_survey": None,
            "n_valid_responses": None, "is_context_only": False,
        })
        french.setdefault(survey, {})[variable] = {
            "status": "complete", "question_text_fr": LR_FR, "source": "dc_codebook", "notes": "",
            "options_fr": {str(o["code"]): o["label"] for o in slider_options("Gauche", "Droite")}}
    return rows, french


def main() -> None:
    all_items, all_french = lr_rows()
    for year in YEARS:
        items, french = ingest(year)
        all_items += items
        all_french[f"dc_{year}"] = french
    pl.DataFrame(all_items, infer_schema_length=None).write_parquet(ITEMS_OUT)
    FRENCH_OUT.write_text(json.dumps(all_french, ensure_ascii=False, indent=1), encoding="utf-8")

    crosswalk = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    for year in YEARS:
        crosswalk["surveys"][f"dc_{year}"] = crosswalk_entry(year)
    crosswalk["n_surveys"] = len(crosswalk["surveys"])
    CROSSWALK.write_text(json.dumps(crosswalk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(all_items)} items -> {ITEMS_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
