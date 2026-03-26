#!/usr/bin/env python3
"""
Prepare CES 2021 data for silicon sampling pipeline.

Run AFTER: 01_build_thematic_domains.py

Processing steps:
1. Filter Quebec respondents only (cps21_province == 11)
2. Select relevant variables (SES + attitude questions)
3. Transform numeric values to text labels via Stata value labels
4. Add thematic_domain column to questions metadata

Outputs:
    data/processed/questions.parquet   — one row per question (variable metadata)
    data/processed/respondents.parquet — respondents × variables matrix (text labels)
"""

import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl
from pandas.io.stata import StataReader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_stata_data(path: str) -> tuple[pd.DataFrame, dict, dict]:
    """Load Stata file with value labels."""
    logger.info(f"Loading {path}...")
    reader = StataReader(path, convert_categoricals=False)
    df = reader.read()
    var_labels = reader.variable_labels()
    value_labels = reader.value_labels()
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    # Fix encoding: Stata files may have latin-1 encoded strings
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            df[col] = df[col].apply(
                lambda x: x.encode("latin-1").decode("utf-8", errors="ignore")
                if isinstance(x, str) else x
            )
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    # Fix value_labels encoding (latin-1 → UTF-8)
    for var_name, labels_dict in value_labels.items():
        try:
            fixed = {}
            for key, lstr in labels_dict.items():
                if isinstance(lstr, str):
                    try:
                        fixed[key] = lstr.encode("latin-1").decode("utf-8", errors="ignore")
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        fixed[key] = lstr
                else:
                    fixed[key] = lstr
            value_labels[var_name] = fixed
        except Exception:
            pass

    return df, var_labels, value_labels


def load_codebook_jsonl(path: str) -> tuple[dict, dict]:
    """Load question metadata from JSONL codebook, returning EN and FR dicts separately.

    The JSONL has two entries per variable: English first, French second.
    Heuristic: an entry is French if its label or question contains accented characters
    or if it's the second entry seen for that variable.
    """
    logger.info(f"Loading codebook from {path}...")
    codebook_en: dict = {}
    codebook_fr: dict = {}
    seen: dict = {}  # var -> first entry already stored
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            var = item.get("variable")
            if not var:
                continue
            if var not in seen:
                seen[var] = item
                codebook_en[var] = item
            else:
                # Second entry for this variable → French version
                codebook_fr[var] = item
    logger.info(
        f"Loaded {len(codebook_en)} EN + {len(codebook_fr)} FR question entries"
    )
    return codebook_en, codebook_fr


def load_thematic_domains(path: str) -> dict:
    """Load thematic domain mapping from JSON file."""
    logger.info(f"Loading thematic domains from {path}...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} thematic domain mappings")
    return data


def filter_quebec(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Quebec respondents only (cps21_province == 11)."""
    initial = len(df)
    df_qc = df[df["cps21_province"] == 11].copy()
    logger.info(
        f"Filtered to Quebec: {len(df_qc)} of {initial} "
        f"({100 * len(df_qc) / initial:.1f}%)"
    )
    return df_qc


def select_relevant_variables(
    df: pd.DataFrame, codebook: dict, thematic_domains: dict
) -> list[str]:
    """
    Select relevant variables for the analysis.

    Categories:
    - SES: age (yob), province, education, gender, language
    - Attitude questions: all variables in thematic_domains.json (train + test + _drop)
      plus any CPS21 variables found in the codebook not already covered.
    PES21 questions excluded to focus on core CPS21 survey.
    """
    ses_vars = [
        "cps21_yob",        # Year of birth (converted to age)
        "cps21_province",
        "cps21_education",
        "cps21_genderid",
        "cps21_language_1",
    ]
    selected = [v for v in ses_vars if v in df.columns]
    selected_set = set(selected)

    # Always include all variables explicitly classified in thematic_domains.json
    for var in thematic_domains:
        if var in df.columns and var not in selected_set:
            selected.append(var)
            selected_set.add(var)

    # Also include any remaining CPS21 codebook variables not already covered
    exclude_patterns = {
        "_t", "_timing", "captcha", "TEXT", "Duration", "Date",
        "ResponseId", "StartDate", "EndDate", "RecordedDate", "_DO_", "_ADO_",
    }
    exclude_vars = {"cps21_consent", "cps21_citizenship"}

    for var in codebook:
        if var.startswith("pes21"):
            continue
        if var not in df.columns:
            continue
        if var in selected_set:
            continue
        if any(p in var for p in exclude_patterns):
            continue
        if var in exclude_vars:
            continue
        if var.endswith("_TEXT"):
            continue
        selected.append(var)
        selected_set.add(var)

    logger.info(f"Selected {len(selected)} variables ({len(ses_vars)} SES + attitude questions)")
    return selected


def apply_value_labels(df: pd.DataFrame, value_labels: dict, var: str) -> pd.Series:
    """Convert numeric column to text labels. Unmapped values stay as strings."""
    if var not in value_labels:
        return df[var].astype(str)
    return df[var].map(value_labels[var]).fillna(df[var].astype(str))


def compute_age(df: pd.DataFrame) -> pd.Series:
    """Compute age from year of birth. Survey conducted in 2021."""
    yob = pd.to_numeric(df["cps21_yob"], errors="coerce").astype("Int64")
    return (2021 - yob).astype(str)


def prepare_respondents(
    df: pd.DataFrame, selected_vars: list[str], value_labels: dict
) -> pd.DataFrame:
    """Build respondents matrix with text labels."""
    logger.info("Transforming values to text labels...")
    respondents = pd.DataFrame(index=df.index)

    rename = {
        "cps21_yob": ("age", lambda: compute_age(df)),
        "cps21_language_1": ("language", lambda: apply_value_labels(df, value_labels, "cps21_language_1")),
        "cps21_province": ("province", lambda: apply_value_labels(df, value_labels, "cps21_province")),
        "cps21_education": ("education", lambda: apply_value_labels(df, value_labels, "cps21_education")),
        "cps21_genderid": ("gender", lambda: apply_value_labels(df, value_labels, "cps21_genderid")),
    }

    # Survey interface language (FR-CA or EN) — used to select prompt language
    if "UserLanguage" in df.columns:
        respondents["survey_language"] = df["UserLanguage"].astype(str)
    else:
        respondents["survey_language"] = "EN"

    for var in selected_vars:
        if var in rename:
            col_name, fn = rename[var]
            respondents[col_name] = fn()
        else:
            respondents[var] = apply_value_labels(df, value_labels, var)

    logger.info(f"Respondents matrix: {len(respondents)} × {len(respondents.columns)}")
    return respondents


TEST_DOMAINS = {"vote_choice", "national_identity"}
DROP_DOMAINS = {"_drop"}


def get_split(domain: str | None) -> str:
    """Map a thematic domain to train/test/drop."""
    if domain in DROP_DOMAINS:
        return "drop"
    if domain in TEST_DOMAINS:
        return "test"
    return "train"


def prepare_questions(
    selected_vars: list[str],
    var_labels: dict,
    value_labels: dict,
    codebook_en: dict,
    codebook_fr: dict,
    thematic_domains: dict,
) -> pd.DataFrame:
    """Build questions metadata table with thematic domain, split, and FR columns."""
    logger.info("Preparing questions metadata...")

    col_rename = {
        "cps21_language_1": "language",
        "cps21_province": "province",
        "cps21_education": "education",
        "cps21_genderid": "gender",
    }

    rows = []
    for var in selected_vars:
        if var == "cps21_yob":
            continue  # replaced by age, not an attitude question

        label = var_labels.get(var, "")
        question_text = codebook_en.get(var, {}).get("question", "")

        options = codebook_en.get(var, {}).get("options", [])
        if not options and var in value_labels:
            options = [
                f"{k}: {v}" for k, v in sorted(value_labels[var].items())
            ]

        # French versions
        label_fr = codebook_fr.get(var, {}).get("label", "")
        question_fr = codebook_fr.get(var, {}).get("question", "")
        options_fr = codebook_fr.get(var, {}).get("options", [])

        domain_entry = thematic_domains.get(var, {})
        domain = domain_entry.get("domain") if isinstance(domain_entry, dict) else None
        split = get_split(domain)

        rows.append({
            "variable_name": var,
            "column_name": col_rename.get(var, var),
            "label": label,
            "question": question_text,
            "options": json.dumps(options, ensure_ascii=False) if options else "",
            "label_fr": label_fr,
            "question_fr": question_fr,
            "options_fr": json.dumps(options_fr, ensure_ascii=False) if options_fr else "",
            "thematic_domain": domain,
            "split": split,
        })

    df_q = pd.DataFrame(rows)

    counts = df_q["split"].value_counts()
    logger.info(
        f"Questions table: {len(df_q)} rows "
        f"(train={counts.get('train', 0)}, "
        f"test={counts.get('test', 0)}, "
        f"drop={counts.get('drop', 0)})"
    )
    return df_q


def main() -> None:
    data_dir = Path("data")
    src_dir = Path("src/article_silicon_sampling_quebec")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df, var_labels, value_labels = load_stata_data(
        str(data_dir / "raw" / "ces_2021" / "ces_2021.dta")
    )
    codebook_en, codebook_fr = load_codebook_jsonl(
        str(data_dir / "raw" / "ces_2021" / "ces_2021_codebook_questions.jsonl")
    )
    thematic_domains = load_thematic_domains(
        str(src_dir / "thematic_domains.json")
    )

    df_qc = filter_quebec(df)
    selected_vars = select_relevant_variables(df_qc, codebook_en, thematic_domains)
    respondents = prepare_respondents(df_qc, selected_vars, value_labels)
    questions = prepare_questions(
        selected_vars, var_labels, value_labels,
        codebook_en, codebook_fr, thematic_domains,
    )

    # Exclude _drop variables from respondents matrix
    keep_cols = questions.loc[questions["split"] != "drop", "column_name"].tolist()
    # Always keep SES columns and survey metadata (no split assignment)
    ses_cols = ["survey_language", "age", "province", "education", "gender", "language"]
    keep_cols = [c for c in respondents.columns if c in ses_cols or c in keep_cols]
    respondents = respondents[keep_cols]
    logger.info(f"Respondents matrix after dropping _drop vars: {len(respondents)} × {len(respondents.columns)}")

    # Save
    pl.from_pandas(questions).write_parquet(str(processed_dir / "questions.parquet"))
    questions.to_csv(str(processed_dir / "questions.csv"), index=False, encoding="utf-8")
    logger.info(f"Saved questions → {processed_dir / 'questions.parquet'}")

    pl.from_pandas(respondents).write_parquet(str(processed_dir / "respondents.parquet"))
    respondents.to_csv(str(processed_dir / "respondents.csv"), index=False, encoding="utf-8")
    logger.info(f"Saved respondents → {processed_dir / 'respondents.parquet'}")

    logger.info("✓ Data preparation complete")


if __name__ == "__main__":
    main()
