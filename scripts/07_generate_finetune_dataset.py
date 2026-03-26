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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

MISSING_STRINGS = {"", "null", "none", "nan"}
GROUP_SUFFIX_RE = re.compile(r"_+\d+$")


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


@dataclass
class QuestionGroup:
    parent_key: str
    kind: str
    items: list[QuestionItem]
    prompt_en: str
    prompt_fr: str


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
) -> str | None:
    if group.kind == "simple":
        raw = respondent.get(group.items[0].column_name)
        return normalize_text(raw)

    if group.kind == "select_all":
        selected: list[str] = []
        for idx, item in enumerate(group.items, start=1):
            value = normalize_text(respondent.get(item.column_name))
            if value:
                selected.append(value or child_label(item, is_french, idx))
        if not selected:
            return None
        return ", ".join(selected)

    battery_parts: list[str] = []
    for idx, item in enumerate(group.items, start=1):
        label = child_label(item, is_french, idx)
        value = normalize_text(respondent.get(item.column_name))
        if not value:
            continue
        battery_parts.append(f"{label}: {value}")
    if not battery_parts:
        return None
    return " / ".join(battery_parts)


def build_input_text(
    context_lines: list[str],
    target_prompt: str,
    is_french: bool,
) -> str:
    if is_french:
        body = "\n".join(context_lines)
        return (
            "Voici des reponses du meme repondant au sondage.\n"
            f"{body}\n\n"
            "Question cible:\n"
            f"Q: {target_prompt}\n"
            "R:"
        )

    body = "\n".join(context_lines)
    return (
        "Here are responses from the same survey respondent.\n"
        f"{body}\n\n"
        "Target question:\n"
        f"Q: {target_prompt}\n"
        "R:"
    )


def generate_examples(groups: list[QuestionGroup], respondents: pl.DataFrame) -> list[dict[str, str]]:
    rows = respondents.with_row_index("respondent_id").to_dicts()
    examples: list[dict[str, str]] = []

    for respondent in rows:
        survey_lang = normalize_text(respondent.get("survey_language")) or "EN"
        is_french = survey_lang.upper().startswith("FR")

        rendered: dict[str, tuple[str, str]] = {}
        for group in groups:
            answer = format_group_answer(group, respondent, is_french)
            if answer is None:
                continue
            prompt = group.prompt_fr if is_french else group.prompt_en
            rendered[group.parent_key] = (prompt, answer)

        if not rendered:
            continue

        ordered = [g.parent_key for g in groups if g.parent_key in rendered]
        for target_key in ordered:
            target_prompt, target_answer = rendered[target_key]
            context_lines = []
            for ctx_key in ordered:
                if ctx_key == target_key:
                    continue
                ctx_prompt, ctx_answer = rendered[ctx_key]
                context_lines.append(f"Q: {ctx_prompt} R: {ctx_answer}")

            if not context_lines:
                continue

            examples.append(
                {
                    "input": build_input_text(context_lines, target_prompt, is_french),
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

    examples = generate_examples(groups, respondents)
    if not examples:
        raise SystemExit("No finetuning examples generated")

    write_jsonl(args.output, examples)
    logger.info("Saved %d examples -> %s", len(examples), args.output)


if __name__ == "__main__":
    main()
