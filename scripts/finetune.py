#!/usr/bin/env python3
"""
Fine-tune Llama-3.1-8B-Instruct on finetune_train.jsonl with LoRA/QLoRA.

Condition 4 of the silicon sampling experiment: SFT on CES 2021 train-domain
questions, evaluated on held-out test-domain questions.

Training data format (flat input/output):
    {"input": "Voici des reponses ...\n\nQuestion cible:\nQ: ...\nR:", "output": "..."}

The model is trained to complete the input by generating the output (answer).
Loss is computed only on the output tokens (via SFTConfig completion_only_loss=True,
which replaces the deprecated DataCollatorForCompletionOnlyLM from trl < 0.9).

Usage:
    # Dry-run to validate data loading and config
    python scripts/finetune.py --dry_run

    # Full training on GPU (LoRA, full precision)
    python scripts/finetune.py --output_dir data/models/lora_condition4

    # QLoRA (4-bit, for consumer GPUs < 40 GB VRAM)
    python scripts/finetune.py --use_4bit --output_dir data/models/lora_condition4

    # Push LoRA weights to HuggingFace
    python scripts/finetune.py --hf_repo your-org/lora-condition4

GPU requirements:
    - LoRA full precision: ~20 GB VRAM (A100 40GB or equivalent)
    - QLoRA NF4:           ~10 GB VRAM (RTX 3090 / A10 / etc.)

Install finetuning dependencies before running:
    pip install -e ".[finetune]"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATA = "data/processed/finetune_train.jsonl"
DEFAULT_OUTPUT_DIR = "data/models/lora_condition4"
DEFAULT_EVAL_SPLIT = 0.05  # 5% of data held out for eval loss monitoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B-Instruct on CES survey data (condition 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(DEFAULT_DATA),
        help="Path to finetune_train.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory to save LoRA weights and tokenizer",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model ID (or local path)",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use QLoRA NF4 quantization (for GPUs < 40 GB VRAM)",
    )

    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout"
    )

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-device training batch size"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Learning rate (AdamW)"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=2048, help="Maximum sequence length in tokens"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Fraction of steps used for LR warmup",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override epochs: stop after this many steps (-1 = use --epochs)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Subsample train dataset to this many samples (-1 = use all)",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=DEFAULT_EVAL_SPLIT,
        help="Fraction of data to use for eval loss monitoring",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Logging / checkpointing
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log metrics every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate on eval split every N steps",
    )

    # HuggingFace Hub
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repo ID to push LoRA weights (e.g. your-org/lora-condition4)",
    )

    # Dry-run
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate data loading and config without training. "
            "Runs on CPU, no GPU required."
        ),
    )

    # Smoke test: use tiny eval split to keep end-of-run evaluation fast
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Truncate eval dataset to 100 samples for fast end-to-end validation.",
    )

    return parser.parse_args()


def check_finetune_deps() -> None:
    """Fail early with a clear message if finetuning packages are missing."""
    missing = []
    for pkg in ("peft", "trl", "accelerate", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # bitsandbytes is only needed for QLoRA — checked later
    if missing:
        logger.error(
            "Missing finetuning dependencies: %s\n"
            "Install them with:\n"
            '    pip install -e ".[finetune]"',
            ", ".join(missing),
        )
        sys.exit(1)


def load_dataset_splits(data_path: Path, eval_split: float, seed: int, smoke_test: bool = False):
    """Load the pre-tokenized dataset from HuggingFace (pushed by tokenize_dataset.py).

    The tokenized dataset is pre-split into train/test on HF — no tokenization
    or splitting needed here. This skips the expensive tokenization step on the GPU pod.

    If the tokenized dataset is not available, falls back to raw JSONL with on-the-fly
    tokenization (slow — run tokenize_dataset.py first to avoid this).
    """
    import os
    from datasets import load_dataset

    hf_tokenized_id = "hubcad25/article_silicon_sampling_quebec_tokenized"
    hf_raw_id = "hubcad25/article_silicon_sampling_quebec_data"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

    try:
        logger.info("Loading pre-tokenized dataset from HF: %s ...", hf_tokenized_id)
        ds = load_dataset(hf_tokenized_id, token=hf_token)
        train_ds = ds["train"]
        eval_ds = ds["test"]
        if smoke_test:
            eval_ds = eval_ds.select(range(100))
        logger.info("Train: %d samples | Eval: %d samples%s", len(train_ds), len(eval_ds), " (smoke_test)" if smoke_test else "")
        return train_ds, eval_ds

    except Exception as e:
        logger.warning("Could not load tokenized dataset (%s). Falling back to raw JSONL + on-the-fly tokenization.", e)
        logger.warning("Run scripts/tokenize_dataset.py locally first to avoid this slow path.")

    # Fallback: raw JSONL
    if data_path.exists():
        logger.info("Loading raw dataset from local file %s ...", data_path)
        ds = load_dataset("json", data_files=str(data_path), split="train")
    else:
        logger.info("Loading raw dataset from HF: %s ...", hf_raw_id)
        ds = load_dataset(hf_raw_id, data_files="finetune_train.jsonl", split="train", token=hf_token)

    logger.info("Total samples: %d", len(ds))

    ds = ds.map(lambda x: {"text": x["input"] + x["output"]}, remove_columns=["input", "output"])
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=eval_split, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info("Train: %d samples | Eval: %d samples", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def format_sample(sample: dict) -> str:
    """Concatenate input and output into a single string for causal LM training.

    The DataCollatorForCompletionOnlyLM will find RESPONSE_TEMPLATE ("\nR:")
    and mask all tokens before the last occurrence — only the final answer
    tokens contribute to the loss.
    """
    return sample["input"] + sample["output"]


def build_model_and_tokenizer(args: argparse.Namespace):
    """Load base model and tokenizer, apply QLoRA quantization if requested."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Llama tokenizer has no pad token by default; use eos as pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config (QLoRA)
    bnb_config = None
    if args.use_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            logger.error(
                "bitsandbytes is required for --use_4bit.\n"
                'Install with: pip install -e ".[finetune]"'
            )
            sys.exit(1)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA NF4 quantization enabled")

    logger.info("Loading base model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    )

    # Required for gradient checkpointing + PEFT compatibility
    model.config.use_cache = False
    model.enable_input_require_grads()

    return model, tokenizer


def build_lora_config(args: argparse.Namespace):
    """Build LoRA config targeting all linear projection layers in Llama."""
    from peft import LoraConfig, TaskType

    # These are the standard target modules for Llama-3 architecture.
    # Targeting all projection layers (not just q/v) gives better performance.
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_training_args(args: argparse.Namespace):
    """Build SFTConfig (trl >= 0.9 replacement for TrainingArguments + SFTTrainer config)."""
    from trl import SFTConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)

    return SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,          # Use bfloat16 (A100/H100). Change to fp16=True on older GPUs.
        optim="paged_adamw_32bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",   # Disable wandb/tensorboard by default; enable manually if needed
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT-specific: packing (faster than completion_only_loss, loss on all tokens)
        dataset_text_field="text",
        packing=True,
        max_length=args.max_seq_len,
        warmup_steps=int(0.03 * (303126 / (args.batch_size * args.grad_accum))),
    )


def print_dry_run_summary(args: argparse.Namespace, train_ds, eval_ds) -> None:
    """Print a summary of data and config for dry-run validation."""
    from datasets import Dataset

    print("\n" + "=" * 60)
    print("DRY-RUN SUMMARY")
    print("=" * 60)
    print(f"Model:           {args.model}")
    print(f"Data:            {args.data}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Use 4-bit:       {args.use_4bit}")
    print()
    print(f"Train samples:   {len(train_ds)}")
    print(f"Eval samples:    {len(eval_ds)}")
    print()
    print("LoRA config:")
    print(f"  r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  target_modules: q/k/v/o/gate/up/down_proj")
    print()
    print("Training config:")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    effective_batch = args.batch_size * args.grad_accum
    print(f"  effective_batch_size={effective_batch}")
    print(f"  lr={args.lr}, warmup_ratio={args.warmup_ratio}")
    print(f"  max_seq_len={args.max_seq_len}")
    print(f"  max_steps={args.max_steps} (-1 = use epochs)")
    print()
    print("Sample (train[0]):")
    sample = train_ds[0]
    formatted = format_sample(sample)
    print(f"  input  ({len(sample['input'])} chars): {sample['input'][:120]!r}...")
    print(f"  output ({len(sample['output'])} chars): {sample['output']!r}")
    print(f"  formatted ({len(formatted)} chars total)")
    print()
    if args.hf_repo:
        print(f"HF repo:         {args.hf_repo}")
    print("=" * 60)
    print("Dry-run complete. No training performed.")
    print("=" * 60 + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Check finetuning deps (peft, trl, accelerate, datasets)
    check_finetune_deps()

    # Load dataset
    train_ds, eval_ds = load_dataset_splits(args.data, args.eval_split, args.seed, smoke_test=args.smoke_test)

    # Subsample train if requested
    if args.max_train_samples > 0 and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
        logger.info("Subsampled train to %d samples", args.max_train_samples)

    # Dry-run: print summary and exit without touching a GPU
    if args.dry_run:
        print_dry_run_summary(args, train_ds, eval_ds)
        sys.exit(0)

    # --- From here: GPU required ---

    from trl import SFTTrainer

    # Load model + tokenizer
    model, tokenizer = build_model_and_tokenizer(args)

    # LoRA config (passed directly to SFTTrainer — no need to call get_peft_model manually)
    lora_config = build_lora_config(args)

    # SFTConfig: TrainingArguments + SFT-specific options (completion_only_loss, max_length)
    training_args = build_training_args(args)

    # SFTTrainer handles LoRA application, tokenization, and completion-only loss internally
    # dataset_text_field="text" set in SFTConfig — no formatting_func needed
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log training metrics
    logger.info("Training complete.")
    metrics = train_result.metrics
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    # Save LoRA weights + tokenizer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s ...", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Model saved.")

    # Save training metrics to CSV for manuscript workflows
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "condition4_train_metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    logger.info("Training metrics saved to %s", metrics_path)

    # Push to HuggingFace Hub if requested
    if args.hf_repo:
        logger.info("Pushing LoRA weights to HuggingFace: %s ...", args.hf_repo)
        model.push_to_hub(args.hf_repo, private=False)
        tokenizer.push_to_hub(args.hf_repo, private=False)
        logger.info("Pushed to https://huggingface.co/%s", args.hf_repo)


if __name__ == "__main__":
    main()
