#!/usr/bin/env python3
"""
Unified SFT data generation script.

Supports two generalization targets:
- q: Question generalization (all respondents, individual train questions as targets).
- r: Respondent generalization (train respondents only, virtual derived variables as targets).

Usage:
    python scripts/generate_sft_data.py --target q --n-ctx 10
    python scripts/generate_sft_data.py --target r --n-ctx 15 --limit-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

MISSING_STRINGS = {"", "null", "none", "nan"}
GROUP_SUFFIX_RE = re.compile(r"_+\d+$")

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

# Virtual target definitions for respondent generalization (target 'r')
VOTE_INTENTION_COLS = ["cps21_votechoice", "cps21_vote_unlikely", "cps21_vote_lean"]
VOTE_INTENTION_PROMPT_EN = "Which party do you intend to vote for?"
VOTE_INTENTION_PROMPT_FR = "Pour quel parti avez-vous l'intention de voter?"

NOT_VOTE_FOR_COLS = [
    "cps21_not_vote_for_1",
    "cps21_not_vote_for_2",
    "cps21_not_vote_for_3",
    "cps21_not_vote_for_4",
    "cps21_not_vote_for_5",
]
NOT_VOTE_FOR_PARTY_EN = [
    "Liberal Party",
    "Conservative Party",
    "NDP",
    "Bloc Québécois",
    "Green Party",
]
NOT_VOTE_FOR_PARTY_FR = [
    "Parti libéral",
    "Parti conservateur",
    "NPD",
    "Bloc québécois",
    "Parti vert",
]
NOT_VOTE_FOR_PROMPT_EN = "Parties that you would absolutely not vote for?"
NOT_VOTE_FOR_PROMPT_FR = (
    "Y a-t-il un ou des partis pour lesquels vous ne voteriez jamais?"
)

RAW_TEST_VARS = {
    "cps21_quebec_sov",
    "cps21_fed_id",
    "cps21_prov_id",
    "cps21_2nd_choice",
}
PROV_ID_NONE_EN = "None"
PROV_ID_NONE_FR = "Aucun"


@dataclass
class QuestionItem:
    variable_name: str
    column_name: str
    parent_key: str
    label_en: str
    label_fr: str
    option_single_en: str | None
    option_single_fr: str | None
    options_en: list[str]
    options_fr: list[str]


@dataclass
class QuestionGroup:
    parent_key: str
    kind: str
    items: list[QuestionItem]
    prompt_en: str
    prompt_fr: str
    options_en: list[str]
    options_fr: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified SFT dataset")
    parser.add_argument(
        "--target",
        choices=["q", "r"],
        required=True,
        help="q (Question generalization) or r (Respondent generalization)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        required=True,
        help="Number of context question groups (10, 15, 25, 50)",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit total number of generated examples (for testing)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Path to questions.parquet",
    )
    parser.add_argument(
        "--respondents",
        type=Path,
        default=Path("data/processed/respondents.parquet"),
        help="Path to respondents.parquet",
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


def parent_var(var_name: str) -> str:
    return GROUP_SUFFIX_RE.sub("", var_name)


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


def parse_option_pairs(options_json: Any) -> dict[str, str]:
    options_text = normalize_text(options_json)
    if not options_text:
        return {}
    try:
        options = json.loads(options_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(options, list):
        return {}
    out: dict[str, str] = {}
    for raw in options:
        opt = normalize_text(raw)
        if not opt:
            continue
        m_colon = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", opt)
        if m_colon:
            code, label = m_colon.group(1), m_colon.group(2)
            out[code] = label.strip()
            continue
        m_paren = re.match(r"^\s*(.+?)\s*\((\d+)\)\s*$", opt)
        if m_paren:
            label, code = m_paren.group(1), m_paren.group(2)
            out[code] = label.strip()
    return out


def build_en_to_fr_value_map(questions: pl.DataFrame) -> dict[str, str]:
    value_map: dict[str, str] = {}
    for row in questions.select(["options", "options_fr"]).to_dicts():
        en_pairs = parse_option_pairs(row.get("options"))
        fr_pairs = parse_option_pairs(row.get("options_fr"))
        if not en_pairs or not fr_pairs:
            # Fallback for simple lists without codes
            en_opts = parse_options_list(row.get("options"))
            fr_opts = parse_options_list(row.get("options_fr"))
            if len(en_opts) == len(fr_opts):
                for en, fr in zip(en_opts, fr_opts):
                    if en and fr:
                        value_map.setdefault(en, fr)
            continue
        for code, en_label in en_pairs.items():
            fr_label = fr_pairs.get(code)
            if fr_label:
                value_map.setdefault(en_label, fr_label)
    return value_map


def localize_value(value: str, is_french: bool, en_to_fr: dict[str, str]) -> str:
    if not is_french:
        return value
    if value in FR_VALUE_OVERRIDES:
        return FR_VALUE_OVERRIDES[value]
    return en_to_fr.get(value, value)


def split_child_label(raw_label: str) -> tuple[str, str | None]:
    label = raw_label.strip()
    if " - " in label:
        parent, child = label.rsplit(" - ", 1)
        parent = parent.strip()
        child = child.strip()
        if parent and child:
            return parent, child
    return label, None


def is_usable_child_label(label: str | None) -> bool:
    if not label:
        return False
    cleaned = label.strip()
    if len(cleaned) < 2:
        return False
    if re.fullmatch(r"[A-Za-z]?\s*\(\d+\)", cleaned):
        return False
    if cleaned.lower() in {"e (6)", "dk", "na"}:
        return False
    return True


def infer_group_kind(items: list[QuestionItem], respondents: pl.DataFrame) -> str:
    if len(items) == 1:
        return "simple"
    all_single = True
    for item in items:
        # Check if values are binary/multi-select style (presence implies selection)
        # In this dataset, select_all often has null for unselected and a string for selected.
        values = (
            respondents.select(
                pl.col(item.column_name).cast(pl.Utf8).drop_nulls().alias("v")
            )
            .to_series()
            .to_list()
        )
        unique_vals = {v.strip() for v in values if normalize_text(v)}
        if len(unique_vals) > 1:
            all_single = False
            break
    return "select_all" if all_single else "battery"


def build_groups(
    questions: pl.DataFrame, respondents: pl.DataFrame, split: str = "train"
) -> list[QuestionGroup]:
    available_cols = set(respondents.columns)
    rows = questions.filter(pl.col("split") == split).to_dicts()
    items: list[QuestionItem] = []
    for row in rows:
        column_name = row.get("column_name")
        variable_name = row.get("variable_name")
        if not column_name or not variable_name or column_name not in available_cols:
            continue
        items.append(
            QuestionItem(
                variable_name=variable_name,
                column_name=column_name,
                parent_key=parent_var(variable_name),
                label_en=pick_lang(row, is_french=False),
                label_fr=pick_lang(row, is_french=True),
                option_single_en=None,  # Not used in current logic but kept for compat
                option_single_fr=None,
                options_en=parse_options_list(row.get("options")),
                options_fr=parse_options_list(row.get("options_fr")),
            )
        )
    grouped: dict[str, list[QuestionItem]] = {}
    for item in items:
        grouped.setdefault(item.parent_key, []).append(item)
    groups: list[QuestionGroup] = []
    for p_key in sorted(grouped):
        group_items = sorted(grouped[p_key], key=lambda x: x.variable_name)
        kind = infer_group_kind(group_items, respondents)
        parent_en, _ = split_child_label(group_items[0].label_en)
        parent_fr, _ = split_child_label(group_items[0].label_fr)
        groups.append(
            QuestionGroup(
                parent_key=p_key,
                kind=kind,
                items=group_items,
                prompt_en=parent_en,
                prompt_fr=parent_fr,
                options_en=group_items[0].options_en,
                options_fr=group_items[0].options_fr,
            )
        )
    return groups


def child_label(item: QuestionItem, is_french: bool, fallback_idx: int) -> str:
    _, child_fr = split_child_label(item.label_fr)
    _, child_en = split_child_label(item.label_en)
    if is_french and is_usable_child_label(child_fr):
        return child_fr  # type: ignore
    if is_usable_child_label(child_en):
        return child_en  # type: ignore
    if is_usable_child_label(child_fr):
        return child_fr  # type: ignore
    suffix = item.variable_name.split("_")[-1]
    return f"item_{suffix}" if suffix.isdigit() else f"item_{fallback_idx}"


def format_group_answer(
    group: QuestionGroup,
    respondent: dict[str, Any],
    is_french: bool,
    en_to_fr: dict[str, str],
) -> str | None:
    if group.kind == "simple":
        raw = respondent.get(group.items[0].column_name)
        value = normalize_text(raw)
        return localize_value(value, is_french, en_to_fr) if value else None
    if group.kind == "select_all":
        selected: list[str] = []
        for idx, item in enumerate(group.items, start=1):
            value = normalize_text(respondent.get(item.column_name))
            if value:
                label = child_label(item, is_french, idx)
                selected.append(label)
        return ", ".join(selected) if selected else None
    battery_parts: list[str] = []
    for idx, item in enumerate(group.items, start=1):
        label = child_label(item, is_french, idx)
        value = normalize_text(respondent.get(item.column_name))
        if value:
            value = localize_value(value, is_french, en_to_fr)
            battery_parts.append(f"{label}: {value}")
    return " / ".join(battery_parts) if battery_parts else None


def build_ses_lines(
    respondent: dict[str, Any], is_french: bool, en_to_fr: dict[str, str]
) -> list[str]:
    lines: list[str] = []
    for col, label_en, label_fr in SES_FIELDS:
        value = normalize_text(respondent.get(col))
        if value:
            value = localize_value(value, is_french, en_to_fr)
            label = label_fr if is_french else label_en
            lines.append(f"- {label}: {value}")
    return lines


def build_input_text(
    ses_lines: list[str],
    context_lines: list[str],
    target_prompt: str,
    is_french: bool,
    choices_line: str | None = None,
) -> str:
    ses_block = "\n".join(ses_lines)
    body = "\n".join(context_lines)
    if is_french:
        ses_prefix = f"Profil SES:\n{ses_block}\n\n" if ses_block else ""
        choices_block = f"{choices_line}\n" if choices_line else ""
        return (
            "Voici des reponses du meme repondant au sondage.\n"
            f"{ses_prefix}{body}\n\n"
            "Question cible:\n"
            f"Q: {target_prompt}\n"
            f"{choices_block}R:"
        )
    ses_prefix = f"SES profile:\n{ses_block}\n\n" if ses_block else ""
    choices_block = f"{choices_line}\n" if choices_line else ""
    return (
        "Here are responses from the same survey respondent.\n"
        f"{ses_prefix}{body}\n\n"
        "Target question:\n"
        f"Q: {target_prompt}\n"
        f"{choices_block}R:"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Loading data...")
    questions = pl.read_parquet(args.questions)
    respondents = pl.read_parquet(args.respondents)

    if args.target == "r":
        before = len(respondents)
        respondents = respondents.filter(pl.col("respondent_split") == "train")
        logger.info(
            "Target 'r': Filtered to train respondents (%d -> %d)",
            before,
            len(respondents),
        )

    en_to_fr = build_en_to_fr_value_map(questions)
    groups = build_groups(questions, respondents, split="train")
    logger.info("Train question groups: %d", len(groups))

    # Pre-render context answers for all respondents
    rows = respondents.with_row_index("respondent_id").to_dicts()
    examples: list[dict[str, str]] = []

    # Virtual targets for 'r'
    test_q = questions.filter(pl.col("split") == "test")
    virtual_targets = []
    if args.target == "r":
        # 1. vote_intention
        vote_choices_en, vote_choices_fr = [], []
        seen_en, seen_fr = set(), set()
        for var in VOTE_INTENTION_COLS:
            q_row = test_q.filter(pl.col("variable_name") == var)
            if not q_row.is_empty():
                r = q_row.to_dicts()[0]
                for opt in parse_options_list(r.get("options")):
                    if opt not in seen_en:
                        seen_en.add(opt)
                        vote_choices_en.append(opt)
                for opt in parse_options_list(r.get("options_fr")):
                    if opt not in seen_fr:
                        seen_fr.add(opt)
                        vote_choices_fr.append(opt)
        virtual_targets.append(
            {
                "kind": "vote_intention",
                "prompt_en": VOTE_INTENTION_PROMPT_EN,
                "prompt_fr": VOTE_INTENTION_PROMPT_FR,
                "choices_en": vote_choices_en,
                "choices_fr": vote_choices_fr,
            }
        )
        # 2. not_vote_for
        virtual_targets.append(
            {
                "kind": "not_vote_for",
                "prompt_en": NOT_VOTE_FOR_PROMPT_EN,
                "prompt_fr": NOT_VOTE_FOR_PROMPT_FR,
                "choices_en": NOT_VOTE_FOR_PARTY_EN,
                "choices_fr": NOT_VOTE_FOR_PARTY_FR,
            }
        )
        # 3. Raw test vars
        for row in test_q.filter(
            pl.col("variable_name").is_in(list(RAW_TEST_VARS))
        ).to_dicts():
            virtual_targets.append(
                {
                    "kind": "prov_id"
                    if row["variable_name"] == "cps21_prov_id"
                    else "raw",
                    "col": row["column_name"],
                    "prompt_en": (row["label"] or row["question"] or "").strip(),
                    "prompt_fr": (row["label_fr"] or row["question_fr"] or "").strip(),
                    "choices_en": parse_options_list(row.get("options")),
                    "choices_fr": parse_options_list(row.get("options_fr")),
                }
            )

    for respondent in rows:
        survey_lang = normalize_text(respondent.get("survey_language")) or "EN"
        is_french = survey_lang.upper().startswith("FR")
        ses_lines = build_ses_lines(respondent, is_french, en_to_fr)
        respondent_id = int(respondent.get("respondent_id", 0))

        # Render all available train groups for this respondent
        rendered_groups: dict[str, tuple[str, str, list[str]]] = {}
        for group in groups:
            answer = format_group_answer(group, respondent, is_french, en_to_fr)
            if answer:
                prompt = group.prompt_fr if is_french else group.prompt_en
                choices = group.options_fr if is_french else group.options_en
                rendered_groups[group.parent_key] = (prompt, answer, choices)

        if not rendered_groups:
            continue

        if args.target == "q":
            keys = sorted(rendered_groups.keys())
            # For each group as a target
            for target_idx, target_key in enumerate(keys):
                target_prompt, target_answer, target_choices = rendered_groups[
                    target_key
                ]
                context_keys = [k for k in keys if k != target_key]
                random.Random(args.seed + respondent_id * 100 + target_idx).shuffle(
                    context_keys
                )
                context_keys = context_keys[: args.n_ctx]
                if not context_keys:
                    continue

                context_lines = [
                    f"Q: {rendered_groups[k][0]} R: {rendered_groups[k][1]}"
                    for k in context_keys
                ]
                choices_text = " / ".join(target_choices)
                choices_line = (
                    (
                        f"Choix: {choices_text}"
                        if is_french
                        else f"Choices: {choices_text}"
                    )
                    if target_choices
                    else None
                )

                examples.append(
                    {
                        "input": build_input_text(
                            ses_lines,
                            context_lines,
                            target_prompt,
                            is_french,
                            choices_line,
                        ),
                        "output": target_answer,
                    }
                )
        else:  # target 'r'
            keys = sorted(rendered_groups.keys())
            random.Random(args.seed + respondent_id).shuffle(keys)
            context_keys = keys[: args.n_ctx]
            if not context_keys:
                continue
            context_lines = [
                f"Q: {rendered_groups[k][0]} R: {rendered_groups[k][1]}"
                for k in context_keys
            ]

            for target in virtual_targets:
                kind = target["kind"]
                t_prompt = target["prompt_fr"] if is_french else target["prompt_en"]
                t_choices = target["choices_fr"] if is_french else target["choices_en"]
                t_value = None

                if kind == "vote_intention":
                    for col in VOTE_INTENTION_COLS:
                        v = normalize_text(respondent.get(col))
                        if v:
                            t_value = localize_value(v, is_french, en_to_fr)
                            break
                elif kind == "not_vote_for":
                    parties = (
                        NOT_VOTE_FOR_PARTY_FR if is_french else NOT_VOTE_FOR_PARTY_EN
                    )
                    sel = [
                        parties[i]
                        for i, col in enumerate(NOT_VOTE_FOR_COLS)
                        if normalize_text(respondent.get(col))
                    ]
                    if sel:
                        t_value = ", ".join(sel)
                elif kind == "prov_id":
                    v = normalize_text(respondent.get(target["col"]))
                    t_value = (
                        localize_value(v, is_french, en_to_fr)
                        if v
                        else (PROV_ID_NONE_FR if is_french else PROV_ID_NONE_EN)
                    )
                else:  # raw
                    v = normalize_text(respondent.get(target["col"]))
                    if v:
                        t_value = localize_value(v, is_french, en_to_fr)

                if t_value:
                    choices_text = " / ".join(t_choices)
                    choices_line = (
                        (
                            f"Choix: {choices_text}"
                            if is_french
                            else f"Choices: {choices_text}"
                        )
                        if t_choices
                        else None
                    )
                    examples.append(
                        {
                            "input": build_input_text(
                                ses_lines,
                                context_lines,
                                t_prompt,
                                is_french,
                                choices_line,
                            ),
                            "output": t_value,
                        }
                    )

    if args.limit_samples and len(examples) > args.limit_samples:
        random.Random(args.seed).shuffle(examples)
        examples = examples[: args.limit_samples]

    output_path = Path(f"data/processed/sft_{args.target}_{args.n_ctx}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Saved %d examples -> %s", len(examples), output_path)


if __name__ == "__main__":
    main()
