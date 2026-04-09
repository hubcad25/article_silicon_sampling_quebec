#!/usr/bin/env python3
"""
Generate SFT dataset from CES train-split questions.

Input:
    data/processed/questions.parquet
    data/processed/respondents.parquet

Output:
    data/processed/finetune_train.jsonl

Each JSONL row is an (input, output) pair built for one respondent and one
target train question (grouped by parent variable when needed).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate finetuning input/output pairs from train split"
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
        help="Respondents parquet with survey responses",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/finetune_train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle question order per respondent (default: enabled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic shuffling",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help=(
            "Number of context questions per example (randomly sampled after shuffle). "
            "None = use all available context questions (original behaviour)."
        ),
    )
    parser.add_argument(
        "--train-respondents-only",
        action="store_true",
        default=False,
        help=(
            "Only include respondents whose respondent_split == 'train'. "
            "Use for conditions 5A/5B (respondent generalization)."
        ),
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


def parse_single_option(options_json: Any) -> str | None:
    options_text = normalize_text(options_json)
    if not options_text:
        return None
    try:
        options = json.loads(options_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(options, list) or len(options) != 1:
        return None
    opt = normalize_text(options[0])
    if not opt:
        return None
    return re.sub(r"^\s*\d+\s*:\s*", "", opt).strip() or None


def strip_option_code(opt: str) -> str:
    """Remove numeric codes from option labels: '1: Label' or 'Label (1)' -> 'Label'."""
    # Prefix format: "1: Label"
    cleaned = re.sub(r"^\s*\d+\s*:\s*", "", opt).strip()
    # Suffix format: "Label (1)"
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
            continue
        for code, en_label in en_pairs.items():
            fr_label = fr_pairs.get(code)
            if not fr_label:
                continue
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


def infer_group_kind(items: list[QuestionItem], respondents: pl.DataFrame) -> str:
    if len(items) == 1:
        return "simple"

    all_single = True
    for item in items:
        values = (
            respondents
            .select(pl.col(item.column_name).cast(pl.Utf8).drop_nulls().alias("v"))
            .to_series()
            .to_list()
        )
        unique_vals = {v.strip() for v in values if normalize_text(v)}
        if len(unique_vals) > 1:
            all_single = False
            break

    return "select_all" if all_single else "battery"


def build_groups(questions: pl.DataFrame, respondents: pl.DataFrame) -> list[QuestionGroup]:
    available_cols = set(respondents.columns)
    rows = questions.filter(pl.col("split") == "train").to_dicts()

    items: list[QuestionItem] = []
    for row in rows:
        column_name = row.get("column_name")
        variable_name = row.get("variable_name")
        if not column_name or not variable_name:
            continue
        if column_name not in available_cols:
            continue
        items.append(
            QuestionItem(
                variable_name=variable_name,
                column_name=column_name,
                parent_key=parent_var(variable_name),
                label_en=pick_lang(row, is_french=False),
                label_fr=pick_lang(row, is_french=True),
                option_single_en=parse_single_option(row.get("options")),
                option_single_fr=parse_single_option(row.get("options_fr")),
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

        options_en = group_items[0].options_en
        options_fr = group_items[0].options_fr

        groups.append(
            QuestionGroup(
                parent_key=p_key,
                kind=kind,
                items=group_items,
                prompt_en=parent_en,
                prompt_fr=parent_fr,
                options_en=options_en,
                options_fr=options_fr,
            )
        )

    return groups


def child_label(item: QuestionItem, is_french: bool, fallback_idx: int) -> str:
    _, child_fr = split_child_label(item.label_fr)
    _, child_en = split_child_label(item.label_en)

    if is_french and is_usable_child_label(child_fr):
        return child_fr  # type: ignore[return-value]
    if is_usable_child_label(child_en):
        return child_en  # type: ignore[return-value]
    if is_usable_child_label(child_fr):
        return child_fr  # type: ignore[return-value]

    option_single = item.option_single_fr if is_french else item.option_single_en
    if is_usable_child_label(option_single):
        return option_single

    suffix = item.variable_name.split("_")[-1]
    if suffix.isdigit():
        return f"item_{suffix}"
    return f"item_{fallback_idx}"


def format_group_answer(
    group: QuestionGroup,
    respondent: dict[str, Any],
    is_french: bool,
    en_to_fr: dict[str, str],
) -> str | None:
    if group.kind == "simple":
        raw = respondent.get(group.items[0].column_name)
        value = normalize_text(raw)
        if not value:
            return None
        return localize_value(value, is_french, en_to_fr)

    if group.kind == "select_all":
        selected: list[str] = []
        for idx, item in enumerate(group.items, start=1):
            value = normalize_text(respondent.get(item.column_name))
            if value:
                if is_french:
                    selected.append(child_label(item, is_french, idx))
                else:
                    selected.append(localize_value(value, is_french, en_to_fr))
        if not selected:
            return None
        return ", ".join(selected)

    battery_parts: list[str] = []
    for idx, item in enumerate(group.items, start=1):
        label = child_label(item, is_french, idx)
        value = normalize_text(respondent.get(item.column_name))
        if not value:
            continue
        value = localize_value(value, is_french, en_to_fr)
        battery_parts.append(f"{label}: {value}")
    if not battery_parts:
        return None
    return " / ".join(battery_parts)


def build_input_text(
    ses_lines: list[str],
    context_lines: list[str],
    target_prompt: str,
    is_french: bool,
    choices_line: str | None = None,
) -> str:
    ses_block = "\n".join(ses_lines)
    if is_french:
        body = "\n".join(context_lines)
        ses_prefix = f"Profil SES:\n{ses_block}\n\n" if ses_block else ""
        choices_block = f"{choices_line}\n" if choices_line else ""
        return (
            "Voici des reponses du meme repondant au sondage.\n"
            f"{ses_prefix}"
            f"{body}\n\n"
            "Question cible:\n"
            f"Q: {target_prompt}\n"
            f"{choices_block}"
            "R:"
        )

    body = "\n".join(context_lines)
    ses_prefix = f"SES profile:\n{ses_block}\n\n" if ses_block else ""
    choices_block = f"{choices_line}\n" if choices_line else ""
    return (
        "Here are responses from the same survey respondent.\n"
        f"{ses_prefix}"
        f"{body}\n\n"
        "Target question:\n"
        f"Q: {target_prompt}\n"
        f"{choices_block}"
        "R:"
    )


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


def generate_examples(
    groups: list[QuestionGroup],
    respondents: pl.DataFrame,
    en_to_fr: dict[str, str],
    shuffle: bool,
    seed: int,
    n_ctx: int | None = None,
) -> list[dict[str, str]]:
    rows = respondents.with_row_index("respondent_id").to_dicts()
    examples: list[dict[str, str]] = []

    for respondent in rows:
        survey_lang = normalize_text(respondent.get("survey_language")) or "EN"
        is_french = survey_lang.upper().startswith("FR")
        ses_lines = build_ses_lines(respondent, is_french, en_to_fr)

        rendered: dict[str, tuple[str, str, list[str]]] = {}
        for group in groups:
            answer = format_group_answer(group, respondent, is_french, en_to_fr)
            if answer is None:
                continue
            prompt = group.prompt_fr if is_french else group.prompt_en
            choices = group.options_fr if is_french else group.options_en
            rendered[group.parent_key] = (prompt, answer, choices)

        if not rendered:
            continue

        respondent_id = int(respondent.get("respondent_id", 0))
        ordered = [g.parent_key for g in groups if g.parent_key in rendered]
        if shuffle:
            random.Random(seed + respondent_id).shuffle(ordered)

        for target_idx, target_key in enumerate(ordered):
            target_prompt, target_answer, target_choices = rendered[target_key]

            context_keys = [ctx_key for ctx_key in ordered if ctx_key != target_key]
            if shuffle:
                random.Random(seed + respondent_id * 1009 + target_idx).shuffle(context_keys)

            # Truncate context to n_ctx questions if specified
            if n_ctx is not None:
                context_keys = context_keys[:n_ctx]

            context_lines = []
            for ctx_key in context_keys:
                ctx_prompt, ctx_answer, _ = rendered[ctx_key]
                context_lines.append(f"Q: {ctx_prompt} R: {ctx_answer}")

            if not context_lines:
                continue

            choices_line = None
            if target_choices:
                choices_text = " / ".join(target_choices)
                if is_french:
                    choices_line = f"Choix: {choices_text}"
                else:
                    choices_line = f"Choices: {choices_text}"

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

    return examples


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Loading questions: %s", args.questions)
    questions = pl.read_parquet(args.questions)
    logger.info("Loading respondents: %s", args.respondents)
    respondents = pl.read_parquet(args.respondents)

    if args.train_respondents_only:
        if "respondent_split" not in respondents.columns:
            raise SystemExit(
                "Column 'respondent_split' not found in respondents. "
                "Run 02_prepare_data.py to add it before using --train-respondents-only."
            )
        before = len(respondents)
        respondents = respondents.filter(pl.col("respondent_split") == "train")
        logger.info(
            "Filtered to train respondents only: %d -> %d", before, len(respondents)
        )

    en_to_fr = build_en_to_fr_value_map(questions)
    logger.info("Value localization pairs (EN->FR): %d", len(en_to_fr))

    groups = build_groups(questions, respondents)
    n_simple = sum(1 for g in groups if g.kind == "simple")
    n_select = sum(1 for g in groups if g.kind == "select_all")
    n_battery = sum(1 for g in groups if g.kind == "battery")
    logger.info(
        "Train question groups: %d (simple=%d, select_all=%d, battery=%d)",
        len(groups),
        n_simple,
        n_select,
        n_battery,
    )
    if args.n_ctx is not None:
        logger.info("Context truncated to n_ctx=%d questions per example", args.n_ctx)

    examples = generate_examples(
        groups,
        respondents,
        en_to_fr,
        shuffle=args.shuffle,
        seed=args.seed,
        n_ctx=args.n_ctx,
    )
    if not examples:
        raise SystemExit("No finetuning examples generated")

    write_jsonl(args.output, examples)
    logger.info("Saved %d examples -> %s", len(examples), args.output)


if __name__ == "__main__":
    main()
