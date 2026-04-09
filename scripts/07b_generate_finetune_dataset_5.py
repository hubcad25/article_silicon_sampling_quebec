#!/usr/bin/env python3
"""
Generate SFT dataset for conditions 5A/5B (respondent generalization).

For each TRAIN respondent, generate one example per virtual target (VD):
  1. vote_intention  — coalesce of cps21_votechoice / cps21_vote_unlikely / cps21_vote_lean
  2. not_vote_for    — battery of cps21_not_vote_for_1..5, output = comma-joined positive hits
  3. cps21_quebec_sov
  4. cps21_fed_id
  5. cps21_prov_id   — null → "None" (no provincial party identification)
  6. cps21_2nd_choice

Context: n_ctx randomly sampled train question responses + SES profile.
Target: the respondent's answer to each virtual VD.

Use --n_ctx 10 for condition 5A, --n_ctx 15 for condition 5B.
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

# Virtual target definitions — these replace the raw test questions with
# merged/derived targets that better reflect the underlying constructs.

# vote_intention: coalesce of the 3 mutually exclusive routing questions
VOTE_INTENTION_COLS = ["cps21_votechoice", "cps21_vote_unlikely", "cps21_vote_lean"]
VOTE_INTENTION_PROMPT_EN = "Which party do you intend to vote for?"
VOTE_INTENTION_PROMPT_FR = "Pour quel parti avez-vous l'intention de voter?"

# not_vote_for: select-all battery — collect party names where column is non-null
NOT_VOTE_FOR_COLS = [
    "cps21_not_vote_for_1",  # Liberal Party
    "cps21_not_vote_for_2",  # Conservative Party
    "cps21_not_vote_for_3",  # NDP
    "cps21_not_vote_for_4",  # Bloc Québécois
    "cps21_not_vote_for_5",  # Green Party
]
NOT_VOTE_FOR_PARTY_EN = ["Liberal Party", "Conservative Party", "NDP", "Bloc Québécois", "Green Party"]
NOT_VOTE_FOR_PARTY_FR = ["Parti libéral", "Parti conservateur", "NPD", "Bloc québécois", "Parti vert"]
NOT_VOTE_FOR_PROMPT_EN = "Parties that you would absolutely not vote for?"
NOT_VOTE_FOR_PROMPT_FR = "Y a-t-il un ou des partis pour lesquels vous ne voteriez jamais?"

# raw test questions passed through unchanged (single-column)
RAW_TEST_VARS = {"cps21_quebec_sov", "cps21_fed_id", "cps21_prov_id", "cps21_2nd_choice"}

# prov_id: fill missing with "None" (no provincial ID)
PROV_ID_NONE_EN = "None"
PROV_ID_NONE_FR = "Aucun"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate finetuning dataset for conditions 5A/5B (respondent generalization)"
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
        default=Path("data/processed/finetune_train_5a.jsonl"),
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

    # Filter to train respondents only (conditions 5A/5B: respondent generalization)
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

    # Build virtual target definitions
    # Each entry: {kind, prompt_en, prompt_fr, choices_en, choices_fr, ...}
    # kind = "raw" | "vote_intention" | "not_vote_for" | "prov_id"
    virtual_targets: list[dict] = []

    # 1. vote_intention (merged routing questions)
    # Choices: union of options from the 3 source questions
    vote_choices_en: list[str] = []
    vote_choices_fr: list[str] = []
    seen_en: set[str] = set()
    seen_fr: set[str] = set()
    for var in VOTE_INTENTION_COLS:
        row = test_q.filter(pl.col("variable_name") == var)
        if row.is_empty():
            continue
        r = row.to_dicts()[0]
        for opt in parse_options_list(r.get("options")):
            if opt not in seen_en:
                seen_en.add(opt)
                vote_choices_en.append(opt)
        for opt in parse_options_list(r.get("options_fr")):
            if opt not in seen_fr:
                seen_fr.add(opt)
                vote_choices_fr.append(opt)
    virtual_targets.append({
        "kind": "vote_intention",
        "prompt_en": VOTE_INTENTION_PROMPT_EN,
        "prompt_fr": VOTE_INTENTION_PROMPT_FR,
        "choices_en": vote_choices_en,
        "choices_fr": vote_choices_fr,
    })

    # 2. not_vote_for (select-all battery → enumerated output)
    virtual_targets.append({
        "kind": "not_vote_for",
        "prompt_en": NOT_VOTE_FOR_PROMPT_EN,
        "prompt_fr": NOT_VOTE_FOR_PROMPT_FR,
        "choices_en": NOT_VOTE_FOR_PARTY_EN,
        "choices_fr": NOT_VOTE_FOR_PARTY_FR,
    })

    # 3. raw single-column test questions
    for row in test_q.filter(pl.col("variable_name").is_in(list(RAW_TEST_VARS))).to_dicts():
        prompt_en = row["label"] or row["question"] or ""
        prompt_fr = row["label_fr"] or row["question_fr"] or ""
        virtual_targets.append({
            "kind": "prov_id" if row["variable_name"] == "cps21_prov_id" else "raw",
            "col": row["column_name"],
            "prompt_en": prompt_en.strip(),
            "prompt_fr": prompt_fr.strip(),
            "choices_en": parse_options_list(row.get("options")),
            "choices_fr": parse_options_list(row.get("options_fr")),
        })

    logger.info("Virtual targets: %d", len(virtual_targets))

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

        for target in virtual_targets:
            kind = target["kind"]
            target_prompt = target["prompt_fr"] if is_french else target["prompt_en"]
            target_choices = target["choices_fr"] if is_french else target["choices_en"]

            # --- resolve output value ---
            if kind == "vote_intention":
                # coalesce across the 3 routing columns
                target_value = None
                for col in VOTE_INTENTION_COLS:
                    v = normalize_text(respondent.get(col))
                    if v:
                        target_value = localize_value(v, is_french, en_to_fr)
                        break
                if not target_value:
                    continue  # respondent has no vote intention answer

            elif kind == "not_vote_for":
                # collect party names where the binary column is non-null
                parties = NOT_VOTE_FOR_PARTY_FR if is_french else NOT_VOTE_FOR_PARTY_EN
                selected = [
                    parties[i]
                    for i, col in enumerate(NOT_VOTE_FOR_COLS)
                    if normalize_text(respondent.get(col))
                ]
                if not selected:
                    continue  # respondent didn't exclude any party
                target_value = ", ".join(selected)

            elif kind == "prov_id":
                v = normalize_text(respondent.get(target["col"]))
                if v:
                    target_value = localize_value(v, is_french, en_to_fr)
                else:
                    target_value = PROV_ID_NONE_FR if is_french else PROV_ID_NONE_EN

            else:  # raw
                v = normalize_text(respondent.get(target["col"]))
                if not v:
                    continue
                target_value = localize_value(v, is_french, en_to_fr)

            input_text = build_input_text(
                ses_lines,
                ctx_lines,
                target_prompt,
                is_french,
                target_choices,
            )
            examples.append({"input": input_text, "output": target_value})

    logger.info(
        "Generated %d examples from %d train respondents × %d virtual targets",
        len(examples),
        len(rows),
        len(virtual_targets),
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