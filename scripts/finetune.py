#!/usr/bin/env python3
"""
Generic SFT training script supporting Qwen and Llama via Unsloth.
Optimized for Calcul Canada (Alliance) environment.

Features:
- Auto-resourcing: Detects GPU VRAM and adjusts batch size/accumulation.
- Unified Model Mapping: supports 0.5b, 1b, 8b, 70b.
- Structured Output: Saves to data/models/sft_{target}_{size}_ctx{n_ctx}/
- Offline Ready: Designed to use pre-cached models from HF_HOME.
- Tensorboard Logging: Default for CC monitoring.

Usage:
    # Dry-run to check paths and config
    python scripts/finetune.py --target q --model_size 1b --n_ctx 10 --dry_run

    # Full training
    python scripts/finetune.py --target q --model_size 1b --n_ctx 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# --- Configuration & Mapping ---

MODEL_MAP = {
    "0.5b": "unsloth/Qwen2.5-0.5B-bnb-4bit",
    "1b": "unsloth/Llama-3.2-1B-bnb-4bit",
    "8b": "unsloth/Llama-3.1-8B-bnb-4bit",
    "70b": "unsloth/Llama-3.1-70B-bnb-4bit",
}

# Default training config
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
TARGET_BATCH_SIZE = 128  # Global effective batch size

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic SFT Engine")
    
    # Core Matrix Params
    parser.add_argument("--target", type=str, choices=["q", "r"], help="Target condition: q (question) or r (respondent)")
    parser.add_argument("--model_size", type=str, choices=["0.5b", "1b", "8b", "70b"], default="1b")
    parser.add_argument("--n_ctx", type=int, default=10, help="Context size (number of previous questions)")
    
    # Overrides
    parser.add_argument("--model", type=str, default=None, help="Explicit model ID (overrides model_size mapping)")
    parser.add_argument("--data", type=Path, default=None, help="Path to JSONL training data")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for model weights")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=None, help="Manual local batch size (auto-detected if None)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Manual grad accumulation (auto-detected if None)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length (auto-adjusted if None)")
    parser.add_argument("--seed", type=int, default=42)
    
    # Infrastructure
    parser.add_argument("--dry_run", action="store_true", help="Print summary and exit without loading GPU")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization (Unsloth default)")
    parser.add_argument("--hf_repo", type=str, default=None, help="HF repo to push to (optional)")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Logging tool (tensorboard, none)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark steps and exit")
    parser.add_argument("--smoke_test", action="store_true", help="Run with tiny data and 1 epoch")

    args = parser.parse_args()

    # 1. Infer Model
    if args.model is None:
        args.model = MODEL_MAP.get(args.model_size)

    # 2. Infer Data Path
    if args.data is None:
        # Try a few common patterns based on existing files
        patterns = [
            f"data/processed/sft_{args.target}_{args.n_ctx}.jsonl",
            f"data/processed/finetune_train_{args.target}_ctx{args.n_ctx}.jsonl",
            f"data/processed/finetune_train_{args.target}.jsonl", # Fallback
        ]
        for p in patterns:
            if Path(p).exists():
                args.data = Path(p)
                break
        
        if args.data is None:
            # If still None, default to the most likely one even if it doesn't exist (will fail later with clear error)
            args.data = Path(f"data/processed/sft_{args.target}_{args.n_ctx}.jsonl")

    # 3. Infer Output Dir
    if args.output_dir is None:
        args.output_dir = Path(f"data/models/sft_{args.target}_{args.model_size}_ctx{args.n_ctx}")

    # 4. Infer Max Seq Len if not provided
    if args.max_seq_len is None:
        # Heuristic: n_ctx=10 -> ~2048, n_ctx=15 -> ~3072
        args.max_seq_len = 2048 if args.n_ctx <= 10 else 3072

    return args

def check_finetune_deps(dry_run: bool = False):
    """Verify that required libraries are installed."""
    if dry_run:
        # Skip GPU-heavy imports during dry-run if no GPU is present
        try:
            import transformers
            import datasets
            return
        except ImportError:
            pass

    try:
        import unsloth
        import trl
        import transformers
        import peft
        import datasets
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Please run 'pip install -e \".[finetune]\"'")
        sys.exit(1)

def load_dataset_splits(data_path: Path, seed: int, smoke_test: bool = False):
    """Load JSONL data and split into train/eval."""
    from datasets import load_dataset
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    if smoke_test:
        dataset = dataset.select(range(min(len(dataset), 50)))
        
    # Split: 95% train, 5% eval
    ds_split = dataset.train_test_split(test_size=0.05, seed=seed)
    return ds_split["train"], ds_split["test"]

def format_sample(sample: dict[str, Any]) -> str:
    """Format a sample for printing."""
    inp = sample.get("input", "")
    out = sample.get("output", "")
    return f"INPUT:\n{inp[:200]}...\n\nOUTPUT:\n{out}"

def get_vram_config(model_size: str):
    """Auto-detect GPU VRAM and return optimal batch/accum."""
    if not torch.cuda.is_available():
        return 1, 128 # Fallback for dry-run
        
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"Detected GPU with {vram_gb:.1f} GB VRAM")
    
    # Heuristics for Unsloth 4-bit
    if vram_gb > 70: # A100 80GB
        batch_size = 8 if "70b" not in model_size else 2
    elif vram_gb > 30: # A100 40GB or RTX 3090/4090
        batch_size = 4 if "70b" not in model_size else 1
    else: # V100 16/32GB or smaller
        batch_size = 2 if "8b" in model_size else 4
        
    grad_accum = max(1, TARGET_BATCH_SIZE // batch_size)
    return batch_size, grad_accum

def build_model_and_tokenizer(args: argparse.Namespace):
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None, # Auto
        load_in_4bit=args.use_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    
    return model, tokenizer

def build_training_args(args: argparse.Namespace, batch_size: int, grad_accum: int):
    from trl import SFTConfig
    from unsloth import is_bfloat16_supported
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    return SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        seed=args.seed,
        report_to=args.report_to,
        dataset_text_field="text", # Unsloth handles formatting via a mapping usually
        max_seq_length=args.max_seq_len,
        packing=False,
    )

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    args = parse_args()
    check_finetune_deps(dry_run=args.dry_run)
    
    # 1. Load Data
    train_ds, eval_ds = load_dataset_splits(args.data, args.seed, args.smoke_test)
    logger.info(f"Dataset loaded: {len(train_ds)} train, {len(eval_ds)} eval")
    
    # 2. Auto-resourcing
    batch_size, grad_accum = get_vram_config(args.model_size)
    if args.batch_size: batch_size = args.batch_size
    if args.grad_accum: grad_accum = args.grad_accum
    
    logger.info(f"Config: batch_size={batch_size}, grad_accum={grad_accum} (Effective: {batch_size*grad_accum})")

    # 3. Dry-Run Summary
    if args.dry_run:
        print("\n" + "="*40)
        print("DRY RUN SUMMARY")
        print("="*40)
        print(f"Target:      {args.target}")
        print(f"Model:       {args.model}")
        print(f"Data:        {args.data}")
        print(f"Output:      {args.output_dir}")
        print(f"Max Seq Len: {args.max_seq_len}")
        print(f"Batch/Accum: {batch_size}/{grad_accum}")
        print("\nSAMPLE DATA:")
        print(format_sample(train_ds[0]))
        print("="*40 + "\n")
        return

    # 4. Load Model (GPU Required)
    from unsloth import FastLanguageModel
    model, tokenizer = build_model_and_tokenizer(args)
    
    # Formatting for Unsloth/TRL
    # CES data has 'input' and 'output'
    def format_prompts(examples):
        texts = []
        for i, o in zip(examples["input"], examples["output"]):
            # Ensure there is a space or newline after R: if not present
            # and append EOS
            prompt = i
            if not prompt.endswith(" ") and not prompt.endswith("\n"):
                prompt += " "
            texts.append(prompt + o + tokenizer.eos_token)
        return {"text": texts}
    
    train_ds = train_ds.map(format_prompts, batched=True)
    eval_ds = eval_ds.map(format_prompts, batched=True)

    # 5. Training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    training_args = build_training_args(args, batch_size, grad_accum)
    
    # Response template for completion-only loss
    # Our prompt ends with "R:" or "R: "
    response_template = "R:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        data_collator=collator,
        args=training_args,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # 6. Save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    if args.hf_repo:
        logger.info(f"Pushing to HF Hub: {args.hf_repo}")
        model.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)

if __name__ == "__main__":
    main()
