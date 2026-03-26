#!/usr/bin/env python3
"""
Simulate Condition 2: LLM + respondent demographics (no RAG).

For each test question, sample N respondent demographic profiles from the
Quebec CES 2021 respondents table, then prompt the LLM with:
  - the respondent profile (age, province, education, gender, language)
  - the target survey question + options

Output:
    data/results/condition2_samples.parquet
        - question: variable name
        - sample_idx: which sample (0 to N-1)
        - respondent_id: sampled respondent row id
        - demographic profile columns
        - simulated_response: the LLM response
        - model_name: model used
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import litellm
import polars as pl
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = "openrouter/meta-llama/llama-3.1-70b-instruct"
DEFAULT_N_SAMPLES = 500
DEFAULT_TEMPERATURE = 0.7
DEMOGRAPHIC_COLUMNS = ["age", "province", "education", "gender", "language"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate Condition 2 (LLM + demographics) on test questions"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Questions parquet with metadata",
    )
    parser.add_argument(
        "--respondents",
        type=Path,
        default=Path("data/processed/respondents.parquet"),
        help="Respondents parquet with demographic columns",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LLM model (LiteLLM format)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of demographic-profile samples per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (0=deterministic, 1=high variance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for profile sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/condition2_samples.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per response",
    )
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if value is None:
        return "Not specified"
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return "Not specified"
    return text


def load_test_questions(questions_path: Path) -> pl.DataFrame:
    logger.info("Loading questions from %s...", questions_path)
    df = pl.read_parquet(questions_path)
    test_q = df.filter(pl.col("split") == "test")
    logger.info("Loaded %d test questions", test_q.height)
    return test_q


def load_respondent_profiles(respondents_path: Path) -> list[dict[str, Any]]:
    logger.info("Loading respondents from %s...", respondents_path)
    respondents = pl.read_parquet(respondents_path).with_row_index("respondent_id")

    required = ["respondent_id", "survey_language", *DEMOGRAPHIC_COLUMNS]
    missing = [col for col in required if col not in respondents.columns]
    if missing:
        raise SystemExit(f"Missing required respondent columns: {missing}")

    profiles = (
        respondents
        .select(required)
        .to_dicts()
    )
    logger.info("Loaded %d respondent profiles", len(profiles))
    return profiles


def parse_options(options_json: str) -> list[str]:
    if not options_json:
        return []
    try:
        options = json.loads(options_json)
    except json.JSONDecodeError:
        return []
    return [opt for opt in options if "_TEXT" not in str(opt)]


def build_prompt(question_row: dict[str, Any], profile: dict[str, Any]) -> tuple[str, str]:
    survey_language = normalize_value(profile.get("survey_language"))
    is_french = survey_language.upper().startswith("FR")

    if is_french:
        label = question_row.get("label_fr") or question_row.get("label") or ""
        question_text = question_row.get("question_fr") or question_row.get("question") or ""
        options = parse_options(question_row.get("options_fr") or question_row.get("options") or "")
    else:
        label = question_row.get("label") or ""
        question_text = question_row.get("question") or ""
        options = parse_options(question_row.get("options") or "")

    profile_lines = [
        f"Age: {normalize_value(profile.get('age'))}",
        f"Province: {normalize_value(profile.get('province'))}",
        f"Education: {normalize_value(profile.get('education'))}",
        f"Gender: {normalize_value(profile.get('gender'))}",
        f"Primary language: {normalize_value(profile.get('language'))}",
    ]

    if is_french:
        system = (
            "Tu participes a un sondage politique canadien (CES 2021). "
            "En te basant sur le profil socio-demographique fourni, reponds a la "
            "question suivante comme cette personne repondrait. "
            "Reponds avec exactement une des options fournies, mot pour mot. "
            "Rien d'autre."
        )
        user_msg = "Profil socio-demographique:\n"
        user_msg += "\n".join(profile_lines) + "\n\n"
        user_msg += f"Question: {label or question_text}\n\n"
        if label and question_text and "Display This Choice" not in question_text:
            user_msg += f"{question_text}\n\n"
        user_msg += "Options:\n"
        for opt in options:
            user_msg += f"- {opt}\n"
        user_msg += "\nTa reponse:"
    else:
        system = (
            "You are participating in a Canadian federal election survey (CES 2021). "
            "Using the respondent demographic profile below, answer the question "
            "as that person would. Reply with exactly one of the provided options, "
            "word for word. Nothing else."
        )
        user_msg = "Respondent profile:\n"
        user_msg += "\n".join(profile_lines) + "\n\n"
        user_msg += f"Question: {label or question_text}\n\n"
        if label and question_text and "Display This Choice" not in question_text:
            user_msg += f"{question_text}\n\n"
        user_msg += "Options:\n"
        for opt in options:
            user_msg += f"- {opt}\n"
        user_msg += "\nYour answer:"

    return system, user_msg


def call_llm(
    system: str,
    user_msg: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str | None:
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY in environment or .env file. "
            "Set it via: export OPENROUTER_API_KEY=... or add to .env"
        )
    litellm.api_key = api_key

    test_questions = load_test_questions(args.questions).to_dicts()
    profiles = load_respondent_profiles(args.respondents)
    rng = random.Random(args.seed)

    logger.info("Model: %s", args.model)
    logger.info("Samples per question: %d", args.n_samples)
    logger.info("Temperature: %s", args.temperature)
    logger.info("Total API calls: %d", len(test_questions) * args.n_samples)

    samples: list[dict[str, Any]] = []

    for q_idx, question_row in enumerate(test_questions):
        var_name = question_row["variable_name"]
        logger.info("[%d/%d] %s", q_idx + 1, len(test_questions), var_name)

        for sample_idx in range(args.n_samples):
            profile = profiles[rng.randrange(len(profiles))]
            system, user_msg = build_prompt(question_row, profile)
            response = call_llm(
                system=system,
                user_msg=user_msg,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            if response is None:
                logger.warning("  Sample %d failed", sample_idx)
                continue

            samples.append(
                {
                    "question": var_name,
                    "sample_idx": sample_idx,
                    "respondent_id": profile.get("respondent_id"),
                    "survey_language": normalize_value(profile.get("survey_language")),
                    "age": normalize_value(profile.get("age")),
                    "province": normalize_value(profile.get("province")),
                    "education": normalize_value(profile.get("education")),
                    "gender": normalize_value(profile.get("gender")),
                    "language": normalize_value(profile.get("language")),
                    "simulated_response": response,
                    "model_name": args.model,
                }
            )

        collected = len([s for s in samples if s["question"] == var_name])
        logger.info("  Collected %d samples", collected)

    if not samples:
        raise SystemExit("No samples collected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df = pl.DataFrame(samples)
    results_df.write_parquet(args.output)
    logger.info("Saved %d samples to %s", len(samples), args.output)

    csv_output = args.output.with_suffix(".csv")
    results_df.write_csv(csv_output)
    logger.info("Saved %d samples to %s", len(samples), csv_output)


if __name__ == "__main__":
    main()
