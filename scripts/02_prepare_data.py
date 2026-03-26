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


def load_codebook_jsonl(path: str) -> dict:
    """Load question metadata from JSONL codebook."""
    logger.info(f"Loading codebook from {path}...")
    codebook: dict = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            var = item.get("variable")
            if var and var not in codebook:
                codebook[var] = item
    logger.info(f"Loaded {len(codebook)} question entries")
    return codebook


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


def select_relevant_variables(df: pd.DataFrame, codebook: dict) -> list[str]:
    """
    Select relevant variables for the analysis.

    Categories:
    - SES: age (yob), province, education, gender, language
    - Attitude questions: CPS21 only (exclude timing/admin/TEXT/captcha/display-order)
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
        if any(p in var for p in exclude_patterns):
            continue
        if var in exclude_vars:
            continue
        if var.endswith("_TEXT"):
            continue
        selected.append(var)

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

    for var in selected_vars:
        if var in rename:
            col_name, fn = rename[var]
            respondents[col_name] = fn()
        else:
            respondents[var] = apply_value_labels(df, value_labels, var)

    logger.info(f"Respondents matrix: {len(respondents)} × {len(respondents.columns)}")
    return respondents


def prepare_questions(
    selected_vars: list[str],
    var_labels: dict,
    value_labels: dict,
    codebook: dict,
    thematic_domains: dict,
) -> pd.DataFrame:
    """Build questions metadata table with thematic domain column."""
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
        question_text = codebook.get(var, {}).get("question", "")

        options = codebook.get(var, {}).get("options", [])
        if not options and var in value_labels:
            options = [
                f"{k}: {v}" for k, v in sorted(value_labels[var].items())
            ]

        domain_entry = thematic_domains.get(var, {})
        domain = domain_entry.get("domain") if isinstance(domain_entry, dict) else None

        rows.append({
            "variable_name": var,
            "column_name": col_rename.get(var, var),
            "label": label,
            "question": question_text,
            "options": json.dumps(options) if options else "",
            "thematic_domain": domain,
        })

    df_q = pd.DataFrame(rows)
    logger.info(f"Questions table: {len(df_q)} rows")
    return df_q


def main() -> None:
    data_dir = Path("data")
    src_dir = Path("src/article_silicon_sampling_quebec")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df, var_labels, value_labels = load_stata_data(
        str(data_dir / "raw" / "ces_2021" / "ces_2021.dta")
    )
    codebook = load_codebook_jsonl(
        str(data_dir / "raw" / "ces_2021" / "ces_2021_codebook_questions.jsonl")
    )
    thematic_domains = load_thematic_domains(
        str(src_dir / "thematic_domains.json")
    )

    df_qc = filter_quebec(df)
    selected_vars = select_relevant_variables(df_qc, codebook)
    respondents = prepare_respondents(df_qc, selected_vars, value_labels)
    questions = prepare_questions(selected_vars, var_labels, value_labels, codebook, thematic_domains)

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
