#!/usr/bin/env python3
"""
Simulate Condition 4A: fine-tuned LLM with respondent context (question generalization, n_ctx=10).

For each respondent, we build a prompt in the same format used for SFT:
  - SES profile lines
  - n_ctx randomly sampled train-split question answers from the same respondent
  - one held-out test question as the target

The prompt is sent to the Hugging Face inference endpoint hosting the
fine-tuned model (condition 4A).

Outputs:
    data/results/condition4a_samples.parquet
    data/results/condition4a_samples.csv

Checkpoint:
    data/results/condition4a_samples.checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_ENDPOINT = (
    "https://n9yxdvjrydaws7a6.us-east-1.aws.endpoints.huggingface.cloud"
)
DEFAULT_OUTPUT = Path("data/results/condition4a_samples.parquet")
DEFAULT_CHECKPOINT = Path("data/results/condition4a_samples.checkpoint.jsonl")
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_NEW_TOKENS = 12
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRIES = 3

MISSING_STRINGS = {"", "null", "none", "nan"}
GROUP_SUFFIX_RE = re.compile(r"_+\d+$")

SES_FIELDS: list[tuple[str, str, str]] = [
    ("age", "Birth year", "Annee de naissance"),
    ("gender", "Gender", "Genre"),
    ("education", "Education", "Scolarite"),
    ("province", "Province", "Province"),
    ("language", "Language", "Langue"),
    ("voted_2019", "Voted in 2019", "A vote en 2019"),
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
        description="Simulate condition 4A (question gen, n_ctx=10) using the fine-tuned HF endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Respondents parquet",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Hugging Face inference endpoint URL",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (defaults to HF_TOKEN/HF_API_KEY/HUGGINGFACEHUB_API_TOKEN)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Final parquet output path",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path (defaults to output with .csv)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="JSONL checkpoint path used for resume",
    )
    parser.add_argument(
        "--max-respondents",
        type=int,
        default=None,
        help="Limit the number of respondents for quick testing",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit the number of test questions for quick testing",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Concurrent request workers",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max new tokens to generate per answer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retry attempts for transient failures",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush progress to the checkpoint every N completed rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic shuffling",
    )
    parser.add_argument(
        "--shuffle-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the order of train-context questions per respondent",
    )
    parser.add_argument(
        "--max-context-questions",
        type=int,
        default=30,
        help="Maximum number of train questions to include in context to avoid context window explosion",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the checkpoint when it exists",
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


def render_target_prompt(question_row: dict[str, Any], is_french: bool) -> str:
    question_text = pick_lang(question_row, is_french=is_french)
    return f"Q: {question_text}"


def extract_answer_text(response_text: str | None, options: list[str]) -> str | None:
    if not response_text:
        return None

    text = response_text.strip()
    for opt in options:
        if opt and opt in text:
            return opt

    first_line = text.splitlines()[0].strip()
    if first_line:
        return first_line
    return text or None


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
            respondents.select(pl.col(item.column_name).cast(pl.Utf8).drop_nulls().alias("v")).to_series().to_list()
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
        return child_fr
    if is_usable_child_label(child_en):
        return child_en
    if is_usable_child_label(child_fr):
        return child_fr

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
    body = "\n".join(context_lines)
    if is_french:
        ses_prefix = f"Profil SES:\n{ses_block}\n\n" if ses_block else ""
        choices_block = f"{choices_line}\n" if choices_line else ""
        return (
            "Voici des reponses du meme repondant au sondage.\n"
            f"{ses_prefix}"
            f"{body}\n\n"
            "Question cible:\n"
            f"{target_prompt}\n"
            f"{choices_block}"
            "R:"
        )

    ses_prefix = f"SES profile:\n{ses_block}\n\n" if ses_block else ""
    choices_block = f"{choices_line}\n" if choices_line else ""
    return (
        "Here are responses from the same survey respondent.\n"
        f"{ses_prefix}"
        f"{body}\n\n"
        "Target question:\n"
        f"{target_prompt}\n"
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


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_generation(prompt: str, generated: str) -> str:
    text = generated.strip()
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    # Le fine-tuning n'ayant pas de vrai stop token, le modèle génère parfois la question suivante ("\nQ: ...")
    # On coupe à la première ligne pour ne garder que la réponse.
    text = text.split("\n")[0].strip()
    return text


def call_endpoint(
    session: requests.Session,
    endpoint: str,
    token: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        },
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "generated_text" in first:
                        return clean_generation(prompt, str(first["generated_text"]))
                    if "text" in first:
                        return clean_generation(prompt, str(first["text"]))

            if isinstance(data, dict):
                if "generated_text" in data:
                    return clean_generation(prompt, str(data["generated_text"]))
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if isinstance(choice, dict):
                        if "text" in choice:
                            return clean_generation(prompt, str(choice["text"]))
                        message = choice.get("message")
                        if isinstance(message, dict) and "content" in message:
                            return clean_generation(prompt, str(message["content"]))

            raise ValueError(f"Unexpected endpoint response: {data!r}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                sleep_for = 1.5 ** attempt
                logger.warning("Request failed (%s). Retrying in %.1fs", exc, sleep_for)
                time.sleep(sleep_for)
                continue
            break

    raise RuntimeError(f"Endpoint call failed after {retries + 1} attempts: {last_error}")


def build_tasks(
    respondents: pl.DataFrame,
    questions: pl.DataFrame,
    groups: list[QuestionGroup],
    en_to_fr: dict[str, str],
    max_respondents: int | None,
    max_questions: int | None,
    shuffle_context: bool,
    seed: int,
    max_context_questions: int | None,
) -> list[dict[str, Any]]:
    test_questions = questions.filter(pl.col("split") == "test").to_dicts()
    if max_questions is not None:
        test_questions = test_questions[:max_questions]

    question_order = {row["variable_name"]: idx for idx, row in enumerate(test_questions)}

    train_groups = groups
    rows = respondents.with_row_index("respondent_id").to_dicts()
    if max_respondents is not None:
        rows = rows[:max_respondents]

    tasks: list[dict[str, Any]] = []
    for respondent in rows:
        survey_lang = normalize_text(respondent.get("survey_language")) or "EN"
        is_french = survey_lang.upper().startswith("FR")
        ses_lines = build_ses_lines(respondent, is_french, en_to_fr)

        rendered: dict[str, tuple[str, str]] = {}
        for group in train_groups:
            answer = format_group_answer(group, respondent, is_french, en_to_fr)
            if answer is None:
                continue
            prompt = group.prompt_fr if is_french else group.prompt_en
            rendered[group.parent_key] = (prompt, answer)

        respondent_id = int(respondent.get("respondent_id", 0))
        ordered = [g.parent_key for g in train_groups if g.parent_key in rendered]
        if shuffle_context:
            random.Random(seed + respondent_id).shuffle(ordered)

        for q_row in test_questions:
            var_name = q_row["variable_name"]
            target_text = pick_lang(q_row, is_french)
            target_prompt = render_target_prompt(q_row, is_french)
            options = parse_options_list(q_row.get("options_fr" if is_french else "options"))
            context_keys = ordered.copy()
            if shuffle_context:
                random.Random(seed + respondent_id * 1009 + question_order[var_name]).shuffle(
                    context_keys
                )
            if max_context_questions is not None:
                context_keys = context_keys[:max_context_questions]

            context_lines = []
            for ctx_key in context_keys:
                ctx_prompt, ctx_answer = rendered[ctx_key]
                context_lines.append(f"Q: {ctx_prompt} R: {ctx_answer}")

            if not context_lines:
                continue

            choices_line = None
            if options:
                choices_text = " / ".join(options)
                if is_french:
                    choices_line = f"Choix: {choices_text}"
                else:
                    choices_line = f"Choices: {choices_text}"

            tasks.append(
                {
                    "respondent_id": respondent_id,
                    "question_index": question_order[var_name],
                    "question": var_name,
                    "question_text": target_text,
                    "choices": options,
                    "is_french": is_french,
                    "input": build_input_text(ses_lines, context_lines, target_prompt, is_french, choices_line),
                    "survey_language": survey_lang,
                }
            )

    return tasks


def run_task(
    task: dict[str, Any],
    endpoint: str,
    token: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    session = requests.Session()
    try:
        response = call_endpoint(
            session=session,
            endpoint=endpoint,
            token=token,
            prompt=task["input"],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
            retries=retries,
        )
        task = dict(task)
        task.update(
            {
                "response_text": response,
                "predicted_answer": extract_answer_text(response, task.get("choices", [])),
                "status": "ok",
                "endpoint": endpoint,
            }
        )
        return task
    except Exception as exc:  # noqa: BLE001
        task = dict(task)
        task.update(
            {
                "response_text": None,
                "predicted_answer": None,
                "status": "error",
                "error": str(exc),
                "endpoint": endpoint,
            }
        )
        return task
    finally:
        session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    token = (
        args.token
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_API_KEY")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise SystemExit("Missing HF token. Set HF_TOKEN, HF_API_KEY, or HUGGINGFACEHUB_API_TOKEN.")

    logger.info("Loading questions: %s", args.questions)
    questions = pl.read_parquet(args.questions)
    logger.info("Loading respondents: %s", args.respondents)
    respondents = pl.read_parquet(args.respondents)
    en_to_fr = build_en_to_fr_value_map(questions)
    logger.info("Value localization pairs (EN->FR): %d", len(en_to_fr))

    groups = build_groups(questions, respondents)
    logger.info("Train groups: %d", len(groups))

    tasks = build_tasks(
        respondents=respondents,
        questions=questions,
        groups=groups,
        en_to_fr=en_to_fr,
        max_respondents=args.max_respondents,
        max_questions=args.max_questions,
        shuffle_context=args.shuffle_context,
        seed=args.seed,
        max_context_questions=args.max_context_questions,
    )

    if not tasks:
        raise SystemExit("No tasks generated")

    logger.info("Generated %d tasks", len(tasks))
    if args.max_respondents is not None or args.max_questions is not None:
        logger.info(
            "Testing mode: respondents=%s questions=%s",
            args.max_respondents if args.max_respondents is not None else "all",
            args.max_questions if args.max_questions is not None else "all",
        )

    completed_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()

    if args.resume:
        for row in load_checkpoint(args.checkpoint):
            key = (int(row["respondent_id"]), str(row["question"]))
            seen_keys.add(key)
            completed_rows.append(row)
        if seen_keys:
            logger.info("Resuming from checkpoint: %d rows already done (use --no-resume to restart)", len(seen_keys))

    pending_tasks = [
        task for task in tasks if (task["respondent_id"], task["question"]) not in seen_keys
    ]
    logger.info("Pending tasks: %d", len(pending_tasks))

    new_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(
                run_task,
                task,
                args.endpoint,
                token,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
                args.timeout,
                args.retries,
            ): task
            for task in pending_tasks
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            row = future.result()
            new_rows.append(row)
            if len(new_rows) >= args.checkpoint_every:
                append_checkpoint(args.checkpoint, new_rows)
                completed_rows.extend(new_rows)
                logger.info("Checkpointed %d rows", len(completed_rows))
                new_rows = []

            if idx % 25 == 0 or idx == len(future_map):
                logger.info("Progress: %d/%d", idx, len(future_map))

    if new_rows:
        append_checkpoint(args.checkpoint, new_rows)
        completed_rows.extend(new_rows)
        logger.info("Checkpointed %d rows", len(completed_rows))

    if not completed_rows:
        raise SystemExit("No completed rows")

    results = (
        pl.DataFrame(completed_rows)
        .unique(subset=["respondent_id", "question"], keep="last")
        .sort(["respondent_id", "question_index"])
    )
    ok_count = results.filter(pl.col("status") == "ok").shape[0]
    err_count = results.filter(pl.col("status") == "error").shape[0]
    logger.info("Completed rows: ok=%d error=%d", ok_count, err_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.write_parquet(args.output)
    logger.info("Saved parquet -> %s", args.output)

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    results.with_columns(
        pl.col("choices").list.join(" | ")
    ).write_csv(csv_output)
    logger.info("Saved csv -> %s", csv_output)


if __name__ == "__main__":
    main()
