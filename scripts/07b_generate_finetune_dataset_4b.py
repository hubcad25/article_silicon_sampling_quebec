#!/usr/bin/env python3
"""
Generate SFT dataset for condition 4B.

For each TRAIN respondent, generate exactly 12 examples (one per test question).
Context: all 108 train question responses as full SES profile.
Target: the respondent's answer to each test question.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

MISSING_STRINGS = {"", "null", "none", "nan"}

SES_FIELDS: list[tuple[str, str, str]] = [
    ("age", "Birth year", "Annee de naissance"),
    ("gender", "Gender", "Genre"),
    ("education", "Education", "Scolarite"),
    ("province", "Province", "Province"),
    ("language", "Language", "Langue"),
    ("voted_2019", "Voted in 2019", "A vote en 2019"),
    ("riding", "Riding", "Circonscription"),
]

FR_VALUE_OVERRIDES = {
    "yes": "oui",
    "no": "non",
    "French": "Francais",
    "English": "Anglais",
    "Other": "Autre",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate finetuning dataset for condition 4B"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Questions parquet with split and bilingual labels",
    )
    parser.add_argument(
        "--respondents",
        type=Path,
        default=Path("data/processed/respondents.parquet"),
        help="Respondents parquet with survey responses and respondent_split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/finetune_train_4b.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--shuffle-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle context questions order per respondent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return None
    return text


def pick_lang(row: dict[str, Any], is_french: bool) -> str:
    if is_french:
        return (
            normalize_text(row.get("label_fr"))
            or normalize_text(row.get("question_fr"))
            or normalize_text(row.get("label"))
            or normalize_text(row.get("question"))
            or ""
        )
    return (
        normalize_text(row.get("label"))
        or normalize_text(row.get("question"))
        or normalize_text(row.get("label_fr"))
        or normalize_text(row.get("question_fr"))
        or ""
    )


def strip_option_code(opt: str) -> str:
    cleaned = re.sub(r"^\s*\d+\s*:\s*", "", opt).strip()
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", cleaned).strip()
    return cleaned


def parse_options_list(options_json: Any) -> list[str]:
    options_text = normalize_text(options_json)
    if not options_text:
        return []
    try:
        options = json.loads(options_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(options, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in options:
        opt = normalize_text(raw)
        if not opt or "_TEXT" in opt:
            continue
        cleaned = strip_option_code(opt)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def build_en_to_fr_value_map(questions: pl.DataFrame) -> dict[str, str]:
    value_map: dict[str, str] = {}
    for row in questions.select(["options", "options_fr"]).to_dicts():
        en_opts = parse_options_list(row.get("options"))
        fr_opts = parse_options_list(row.get("options_fr"))
        if len(en_opts) != len(fr_opts):
            continue
        for en, fr in zip(en_opts, fr_opts):
            if en and fr:
                value_map[en] = fr
    return value_map


def localize_value(value: str, is_french: bool, en_to_fr: dict[str, str]) -> str:
    if not is_french:
        return value
    if value in FR_VALUE_OVERRIDES:
        return FR_VALUE_OVERRIDES[value]
    return en_to_fr.get(value, value)


def build_ses_lines(
    respondent: dict[str, Any],
    is_french: bool,
    en_to_fr: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for col, label_en, label_fr in SES_FIELDS:
        value = normalize_text(respondent.get(col))
        if not value:
            continue
        value = localize_value(value, is_french, en_to_fr)
        label = label_fr if is_french else label_en
        lines.append(f"- {label}: {value}")
    return lines


def build_context_lines(
    train_data: list[dict],
    respondent: dict[str, Any],
    is_french: bool,
    en_to_fr: dict[str, str],
    shuffle: bool,
    seed: int,
    respondent_id: int,
) -> list[str]:
    context_items: list[tuple[str, str]] = []
    for item in train_data:
        col = item["col"]
        raw_value = respondent.get(col)
        value = normalize_text(raw_value)
        if not value:
            continue
        localized_value = localize_value(value, is_french, en_to_fr)
        prompt = item["prompt_fr"] if is_french else item["prompt_en"]
        context_items.append((prompt, localized_value))

    if shuffle:
        random.Random(seed + respondent_id).shuffle(context_items)

    lines: list[str] = []
    for prompt, value in context_items:
        lines.append(f"Q: {prompt} R: {value}")
    return lines


def build_input_text(
    ses_lines: list[str],
    context_lines: list[str],
    target_prompt: str,
    is_french: bool,
    target_choices: list[str] | None,
) -> str:
    ses_block = "\n".join(ses_lines)
    body = "\n".join(context_lines)

    if is_french:
        ses_prefix = f"Profil SES:\n{ses_block}\n\n" if ses_block else ""
        choices_block = f"Choix: {' / '.join(target_choices)}\n" if target_choices else ""
        return (
            "Voici des reponses du meme repondant au sondage.\n"
            f"{ses_prefix}"
            f"{body}\n\n"
            "Question cible:\n"
            f"Q: {target_prompt}\n"
            f"{choices_block}"
            "R:"
        )

    ses_prefix = f"SES profile:\n{ses_block}\n\n" if ses_block else ""
    choices_block = f"Choices: {' / '.join(target_choices)}\n" if target_choices else ""
    return (
        "Here are responses from the same survey respondent.\n"
        f"{ses_prefix}"
        f"{body}\n\n"
        "Target question:\n"
        f"Q: {target_prompt}\n"
        f"{choices_block}"
        "R:"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Loading questions: %s", args.questions)
    questions = pl.read_parquet(args.questions)

    logger.info("Loading respondents: %s", args.respondents)
    respondents = pl.read_parquet(args.respondents)

    # Filter to train respondents only (condition 4B specific)
    if "respondent_split" not in respondents.columns:
        raise SystemExit("ERROR: respondents.parquet missing 'respondent_split' column")
    train_respondents = respondents.filter(pl.col("respondent_split") == "train")
    logger.info(
        "Filtered to train respondents: %d of %d",
        len(train_respondents),
        len(respondents),
    )

    # Build EN<->FR value map
    en_to_fr = {}
    for row in questions.select(["options", "options_fr"]).to_dicts():
        en_opts = parse_options_list(row.get("options"))
        fr_opts = parse_options_list(row.get("options_fr"))
        if len(en_opts) != len(fr_opts):
            continue
        for en, fr in zip(en_opts, fr_opts):
            if en and fr:
                en_to_fr[en] = fr
    logger.info("Value localization pairs (EN->FR): %d", len(en_to_fr))

    # Split questions
    train_q = questions.filter(pl.col("split") == "train").sort("variable_name")
    test_q = questions.filter(pl.col("split") == "test").sort("variable_name")

    logger.info(
        "Train questions: %d, Test questions: %d",
        len(train_q),
        len(test_q),
    )

    # Precompute train question data: column_name, prompt (EN/FR)
    train_data: list[dict] = []
    for row in train_q.to_dicts():
        var_name = row["variable_name"]
        column_name = row["column_name"]
        prompt_en = row["label"] or row["question"] or ""
        prompt_fr = row["label_fr"] or row["question_fr"] or ""
        train_data.append({
            "var": var_name,
            "col": column_name,
            "prompt_en": prompt_en.strip(),
            "prompt_fr": prompt_fr.strip(),
        })

    # Precompute test question data: target prompts and choices
    test_data: list[dict] = []
    for row in test_q.to_dicts():
        var_name = row["variable_name"]
        column_name = row["column_name"]
        prompt_en = row["label"] or row["question"] or ""
        prompt_fr = row["label_fr"] or row["question_fr"] or ""
        choices_en = parse_options_list(row.get("options"))
        choices_fr = parse_options_list(row.get("options_fr"))
        test_data.append({
            "var": var_name,
            "col": column_name,
            "prompt_en": prompt_en.strip(),
            "prompt_fr": prompt_fr.strip(),
            "choices_en": choices_en,
            "choices_fr": choices_fr,
        })

    # Generate examples
    examples: list[dict[str, str]] = []
    rows = train_respondents.with_row_index("respondent_id").to_dicts()
    for respondent in rows:
        survey_lang = normalize_text(respondent.get("survey_language")) or "EN"
        is_french = survey_lang.upper().startswith("FR")
        respondent_id = int(respondent.get("respondent_id", 0))

        ses_lines = build_ses_lines(respondent, is_french, en_to_fr)

        # Build context lines from all train questions (same for all test targets)
        ctx_lines = build_context_lines(
            train_data, respondent, is_french, en_to_fr,
            args.shuffle_context, args.seed, respondent_id,
        )
        if not ctx_lines:
            continue

        # For each test question, create an example
        for test_item in test_data:
            target_col = test_item["col"]
            raw_value = respondent.get(target_col)
            value = normalize_text(raw_value)
            if not value:
                continue  # Skip missing answers

            target_value = localize_value(value, is_french, en_to_fr)
            target_prompt = test_item["prompt_fr"] if is_french else test_item["prompt_en"]
            target_choices = test_item["choices_fr"] if is_french else test_item["choices_en"]

            input_text = build_input_text(
                ses_lines,
                ctx_lines,
                target_prompt,
                is_french,
                target_choices,
            )

            examples.append({
                "input": input_text,
                "output": target_value,
            })

    logger.info(
        "Generated %d examples from %d train respondents × %d test questions",
        len(examples),
        len(rows),
        len(test_data),
    )

    if not examples:
        raise SystemExit("No finetuning examples generated")

    # Save
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved -> %s", output_path)


if __name__ == "__main__":
    main()