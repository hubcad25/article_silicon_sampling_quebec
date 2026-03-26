#!/usr/bin/env python3
"""
Prepare CES 2021 data for silicon sampling pipeline.

Processing steps:
1. Filter Quebec respondents only (cps21_province == 11)
2. Select relevant variables (SES + attitude questions)
3. Transform numeric values to text labels via Stata value labels
4. Output: respondents.parquet (respondents × variables) + questions.parquet (metadata)
"""

import json
from pathlib import Path
import logging

import pandas as pd
import polars as pl
from pandas.io.stata import StataReader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_stata_data(path: str) -> tuple[pd.DataFrame, dict]:
    """Load Stata file with value labels."""
    logger.info(f"Loading {path}...")
    reader = StataReader(path, convert_categoricals=False)
    df = reader.read()
    var_labels = reader.variable_labels()
    value_labels = reader.value_labels()
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    
    # Fix encoding issues: Stata files may have latin-1 encoded strings
    # Convert to proper UTF-8 representation
    for col in df.select_dtypes(include=['object', 'string']).columns:
        try:
            # Try to decode as latin-1 and re-encode as UTF-8
            df[col] = df[col].apply(
                lambda x: x.encode('latin-1').decode('utf-8', errors='ignore')
                if isinstance(x, str) else x
            )
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If it fails, leave as is
            pass
    
    # Also fix value_labels encoding (latin-1 → UTF-8)
    for var_name, labels_dict in value_labels.items():
        try:
            fixed_labels = {}
            for key, label_str in labels_dict.items():
                if isinstance(label_str, str):
                    # Try latin-1 to UTF-8 conversion
                    try:
                        fixed_label = label_str.encode('latin-1').decode('utf-8', errors='ignore')
                        fixed_labels[key] = fixed_label
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        fixed_labels[key] = label_str
                else:
                    fixed_labels[key] = label_str
            value_labels[var_name] = fixed_labels
        except Exception:
            # If anything fails, leave as is
            pass
    
    return df, var_labels, value_labels


def load_codebook_jsonl(path: str) -> dict:
    """Load question metadata from JSONL codebook."""
    logger.info(f"Loading codebook from {path}...")
    codebook = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            var = item.get("variable")
            if var:
                # Store first occurrence (avoid duplicates for bilingual entries)
                if var not in codebook:
                    codebook[var] = item
    logger.info(f"Loaded {len(codebook)} question entries")
    return codebook


def filter_quebec(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Quebec respondents only (cps21_province == 11)."""
    initial_count = len(df)
    df_qc = df[df["cps21_province"] == 11].copy()
    logger.info(
        f"Filtered to Quebec: {len(df_qc)} of {initial_count} "
        f"({100*len(df_qc)/initial_count:.1f}%)"
    )
    return df_qc


def select_relevant_variables(df: pd.DataFrame, codebook: dict) -> list[str]:
    """
    Select relevant variables for the analysis.
    
    Categories:
    - SES: age (yob), province, education, gender, language
    - Attitude questions: CPS21 survey questions only (exclude timing/admin/TEXT/captcha)
    Note: PES21 questions excluded to focus on core CPS21 survey
    """
    selected = []
    
    # SES variables (hardcoded keys, CPS21 only)
    ses_vars = [
        "cps21_yob",  # Year of birth (will compute age)
        "cps21_province",  # Province
        "cps21_education",  # Education level
        "cps21_genderid",  # Gender identity
        "cps21_language_1",  # First language
    ]
    selected.extend([v for v in ses_vars if v in df.columns])
    
    # Attitude questions from codebook (CPS21 only)
    # Exclude: timing (_t suffix), admin, captcha, TEXT (open-ended), response metadata
    exclude_patterns = {"_t", "_timing", "captcha", "TEXT", "Duration", "Date", "ResponseId", "StartDate", "EndDate", "RecordedDate"}
    exclude_vars = {"cps21_consent", "cps21_citizenship"}  # Skip consent/citizenship
    
    for var in codebook.keys():
        # Skip PES21 questions (keep CPS21 only)
        if var.startswith("pes21"):
            continue
        # Skip if doesn't exist in data
        if var not in df.columns:
            continue
        # Skip if matches exclusion patterns
        if any(pattern in var for pattern in exclude_patterns):
            continue
        # Skip explicit exclusions
        if var in exclude_vars:
            continue
        # Skip if it's a TEXT field (open-ended)
        if var.endswith("_TEXT"):
            continue
        selected.append(var)
    
    logger.info(f"Selected {len(selected)} variables: {len(ses_vars)} SES + CPS21 attitude questions")
    return selected


def apply_value_labels(
    df: pd.DataFrame, value_labels: dict, var: str
) -> pd.Series:
    """
    Convert numeric column to text labels using Stata value labels.
    Missing/unmapped values remain as strings of the number.
    """
    if var not in value_labels:
        # No value labels for this variable, keep as is
        return df[var].astype(str)
    
    label_map = value_labels[var]
    # Map numeric to text, leaving unmapped values as strings
    return df[var].map(label_map).fillna(df[var].astype(str))


def compute_age(df: pd.DataFrame) -> pd.Series:
    """Compute age from year of birth (cps21_yob). Survey was conducted in 2021."""
    if "cps21_yob" not in df.columns:
        return pd.Series(index=df.index, dtype=object)
    # Convert to int64 first to avoid overflow with int8 Stata type
    yob = pd.to_numeric(df["cps21_yob"], errors="coerce").astype("Int64")
    age = 2021 - yob
    return age.astype(str)


def prepare_respondents(
    df: pd.DataFrame, selected_vars: list[str], value_labels: dict
) -> pd.DataFrame:
    """Transform selected variables to text labels."""
    logger.info("Transforming values to text labels...")
    
    respondents = pd.DataFrame(index=df.index)
    
    for var in selected_vars:
        if var == "cps21_yob":
            # Replace YOB with computed age
            respondents["age"] = compute_age(df)
        elif var == "cps21_language_1":
            # Rename for clarity
            respondents["language"] = apply_value_labels(df, value_labels, var)
        elif var == "cps21_province":
            respondents["province"] = apply_value_labels(df, value_labels, var)
        elif var == "cps21_education":
            respondents["education"] = apply_value_labels(df, value_labels, var)
        elif var == "cps21_genderid":
            respondents["gender"] = apply_value_labels(df, value_labels, var)
        else:
            # Attitude questions: keep original variable name
            respondents[var] = apply_value_labels(df, value_labels, var)
    
    logger.info(
        f"Created respondents matrix: {len(respondents)} × {len(respondents.columns)}"
    )
    return respondents


def prepare_questions(
    selected_vars: list[str], var_labels: dict, value_labels: dict, codebook: dict
) -> pd.DataFrame:
    """Prepare questions metadata."""
    logger.info("Preparing questions metadata...")
    
    questions = []
    for var in selected_vars:
        if var == "cps21_yob":
            # Skip YOB (replaced by age)
            continue
        
        # Get variable label
        label = var_labels.get(var, "")
        
        # Get question text from codebook
        question = codebook.get(var, {}).get("question", "")
        
        # Get answer options from codebook or value labels
        options = []
        if var in codebook:
            cb_options = codebook[var].get("options", [])
            if cb_options:
                options = cb_options
        
        if not options and var in value_labels:
            # Fallback to value labels as options
            labels_dict = value_labels[var]
            options = [
                f"{v}: {label_text}"
                for v, label_text in sorted(labels_dict.items())
            ]
        
        # Serialize options as JSON string for CSV export
        import json
        options_json = json.dumps(options) if options else ""
        
        # Determine renamed column name
        if var == "cps21_language_1":
            col_name = "language"
        elif var == "cps21_province":
            col_name = "province"
        elif var == "cps21_education":
            col_name = "education"
        elif var == "cps21_genderid":
            col_name = "gender"
        else:
            col_name = var
        
        questions.append({
            "variable_name": var,
            "column_name": col_name,
            "label": label,
            "question": question,
            "options": options_json,
        })
    
    df_questions = pd.DataFrame(questions)
    logger.info(f"Created questions matrix: {len(df_questions)} rows")
    return df_questions


def main():
    """Main pipeline."""
    # Paths
    data_dir = Path("data")
    raw_ces = data_dir / "raw" / "ces_2021" / "ces_2021.dta"
    codebook_path = data_dir / "raw" / "ces_2021" / "ces_2021_codebook_questions.jsonl"
    processed_dir = data_dir / "processed"
    
    # Create output directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, var_labels, value_labels = load_stata_data(str(raw_ces))
    codebook = load_codebook_jsonl(str(codebook_path))
    
    # Process
    df_qc = filter_quebec(df)
    selected_vars = select_relevant_variables(df_qc, codebook)
    respondents = prepare_respondents(df_qc, selected_vars, value_labels)
    questions = prepare_questions(selected_vars, var_labels, value_labels, codebook)
    
    # Save
    respondents_path = processed_dir / "respondents.parquet"
    questions_path = processed_dir / "questions.parquet"
    
    # Save respondents (parquet + CSV)
    respondents_pl = pl.from_pandas(respondents)
    respondents_pl.write_parquet(str(respondents_path))
    logger.info(f"Saved respondents → {respondents_path}")
    
    # Also save as CSV for ease of use (UTF-8 encoded via pandas)
    respondents_csv_path = processed_dir / "respondents.csv"
    respondents.to_csv(str(respondents_csv_path), index=False, encoding="utf-8")
    logger.info(f"Saved respondents → {respondents_csv_path}")
    
    questions_pl = pl.from_pandas(questions)
    questions_pl.write_parquet(str(questions_path))
    logger.info(f"Saved questions → {questions_path}")
    
    # Also save as CSV for ease of use (UTF-8 encoded via pandas)
    questions_csv_path = processed_dir / "questions.csv"
    questions.to_csv(str(questions_csv_path), index=False, encoding="utf-8")
    logger.info(f"Saved questions → {questions_csv_path}")
    
    logger.info("✓ Data preparation complete")


if __name__ == "__main__":
    main()
