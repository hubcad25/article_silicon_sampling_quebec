#!/usr/bin/env python3
"""
Simulate Condition 1: Cold LLM (no context).

The LLM is asked the survey question with no information about the respondent.
We draw N samples from the same prompt to obtain a simulated distribution.

Output:
    data/results/condition1_samples.parquet
        - question: variable name
        - sample_idx: which sample (0 to N-1)
        - simulated_response: the LLM's response
        - model_name: which model was used
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import litellm
import polars as pl
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv()

DEFAULT_MODEL = "openrouter/meta-llama/llama-3.1-70b-instruct"
DEFAULT_N_SAMPLES = 500
DEFAULT_TEMPERATURE = 0.7  # > 0 for stochasticity in cold baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate cold LLM baseline (no context) on test questions"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/questions.parquet"),
        help="Questions parquet with metadata",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LLM model (LiteLLM format, e.g. 'openrouter/meta-llama/llama-3.1-70b-instruct')",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of samples per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (0=deterministic, 1=high variance)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/condition1_samples.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per response",
    )
    return parser.parse_args()


def load_test_questions(questions_path: Path) -> pl.DataFrame:
    """Load test split questions only."""
    logger.info(f"Loading questions from {questions_path}...")
    df = pl.read_parquet(questions_path)
    test_q = df.filter(pl.col("split") == "test")
    logger.info(f"Loaded {test_q.shape[0]} test questions")
    return test_q


def build_prompt(
    question_row: dict[str, Any], language: str = "EN"
) -> tuple[str, str]:
    """Build system and user prompts for cold baseline.

    Condition 1 = no SES context, just the question and options.
    Language determines prompt language (EN or FR-CA).
    """
    var_name = question_row["variable_name"]

    # Choose language-specific fields
    if language == "FR-CA":
        label = question_row.get("label_fr") or question_row.get("label") or ""
        question_text = (
            question_row.get("question_fr") or question_row.get("question") or ""
        )
        options_json = question_row.get("options_fr") or question_row.get("options") or ""
    else:  # EN
        label = question_row.get("label") or ""
        question_text = question_row.get("question") or ""
        options_json = question_row.get("options") or ""

    # Parse options
    options = []
    if options_json:
        try:
            options = json.loads(options_json)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse options for {var_name}")

    # Filter out TEXT placeholders (open-ended responses we can't simulate)
    options = [opt for opt in options if "_TEXT" not in opt]

    # Build prompt text
    if language == "FR-CA":
        system = (
            "Tu participes à un sondage politique canadien (CES 2021). "
            "Réponds à la question suivante comme un résidant du Québec le ferait. "
            "Réponds avec exactement une des options fournies, mot pour mot. Rien d'autre."
        )
        user_msg = f"Question: {label or question_text}\n\n"
        if label and question_text and "Display This Choice" not in question_text:
            user_msg += f"{question_text}\n\n"
        user_msg += "Options:\n"
        for opt in options:
            user_msg += f"- {opt}\n"
        user_msg += "\nTa réponse:"
    else:  # EN
        system = (
            "You are participating in a Canadian federal election survey (CES 2021). "
            "Answer the following survey question as a Quebec resident would. "
            "Reply with exactly one of the provided options, word for word. Nothing else."
        )
        user_msg = f"Question: {label or question_text}\n\n"
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
    """Call LLM via LiteLLM and return response text."""
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
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    args = parse_args()

    # Verify API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY in environment or .env file. "
            "Set it via: export OPENROUTER_API_KEY=... or add to .env"
        )
    litellm.api_key = api_key

    # Load questions
    test_q = load_test_questions(args.questions)
    questions = test_q.to_dicts()

    logger.info(f"Model: {args.model}")
    logger.info(f"Samples per question: {args.n_samples}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Total API calls: {len(questions) * args.n_samples}")
    logger.info("")

    # Collect samples
    samples = []
    for q_idx, q_row in enumerate(questions):
        var_name = q_row["variable_name"]
        logger.info(f"[{q_idx + 1}/{len(questions)}] {var_name}")

        # Condition 1 = cold baseline, no SES context
        # Use English for simplicity in baseline (can extend to multilingual later)
        system, user_msg = build_prompt(q_row, language="EN")

        for sample_idx in range(args.n_samples):
            response = call_llm(
                system,
                user_msg,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if response:
                samples.append({
                    "question": var_name,
                    "sample_idx": sample_idx,
                    "simulated_response": response,
                    "model_name": args.model,
                })
            else:
                logger.warning(f"  Sample {sample_idx}: failed")

        logger.info(f"  ✓ {len([s for s in samples if s['question'] == var_name])} samples collected")

    if not samples:
        raise SystemExit("No samples collected")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df = pl.DataFrame(samples)
    results_df.write_parquet(args.output)
    logger.info(f"Saved {len(samples)} samples → {args.output}")


if __name__ == "__main__":
    main()
