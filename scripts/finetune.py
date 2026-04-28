#!/usr/bin/env python3
"""
Fine-tune Llama-3.1-8B on CES survey data with LoRA/QLoRA.

Shared training script for all SFT conditions (4A, 4B, 5A, 5B).
Only --data, --output_dir, and --hf_repo differ across conditions.

Conditions:
    4A  question generalization, n_ctx=10  (all respondents)
    4B  question generalization, n_ctx=15  (all respondents)
    5A  respondent generalization, n_ctx=10 (train respondents only)
    5B  respondent generalization, n_ctx=15 (train respondents only)

Training data format (flat input/output):
    {"input": "Voici des reponses ...\n\nQuestion cible:\nQ: ...\nR:", "output": "..."}

The model is trained to complete the input by generating the output (answer).
Loss is computed only on the output tokens (via SFTConfig completion_only_loss=True,
which replaces the deprecated DataCollatorForCompletionOnlyLM from trl < 0.9).

Usage:
    # Dry-run to validate data loading and config
    python scripts/finetune.py --dry_run

    # Full training on GPU (LoRA, full precision)
    python scripts/finetune.py --output_dir data/models/lora_condition4a

    # QLoRA (4-bit, for consumer GPUs < 40 GB VRAM)
    python scripts/finetune.py --use_4bit --output_dir data/models/lora_condition4a

    # Push LoRA weights to HuggingFace
    python scripts/finetune.py --hf_repo your-org/lora-condition4a

GPU requirements:
    - LoRA full precision: ~20 GB VRAM (A100 40GB or equivalent)
    - QLoRA NF4:           ~10 GB VRAM (RTX 3090 / A10 / etc.)

Install finetuning dependencies before running:
    pip install -e ".[finetune]"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_DATA = "data/processed/finetune_train_q_nctx10.jsonl"
DEFAULT_OUTPUT_DIR = "data/models/sft_q_1b_ctx10"

def get_model_family(model_id: str) -> str:
    if "qwen" in model_id.lower():
        return "qwen"
    return "llama"

def build_model_and_tokenizer(args: argparse.Namespace):
    """Load base model and tokenizer via Unsloth, apply QLoRA quantization if requested."""
    import torch
    from unsloth import FastLanguageModel

    logger.info("Loading model: %s", args.model)
    
    # Auto-adjust sequence length for large contexts
    # n_ctx=50 + SES + Prompt can reach ~1500-2000 tokens
    max_seq_length = args.max_seq_len
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        dtype=None, # Auto detection
        load_in_4bit=args.use_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    return model, tokenizer



def build_training_args(args: argparse.Namespace):
    """Build SFTConfig (trl >= 0.9 replacement for TrainingArguments + SFTTrainer config)."""
    from trl import SFTConfig
    from unsloth import is_bfloat16_supported

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
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,  # Keep only last 2 checkpoints to save disk space
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        seed=args.seed,
        report_to=args.report_to or "none",
        dataloader_num_workers=0,  # dataset is in RAM after tokenization; workers add overhead
        dataset_text_field="text",
        packing=False,  # Packing causes batch_size mismatch in loss with Unsloth+trl 0.24
        max_length=args.max_seq_len,
        warmup_steps=int(0.03 * (303126 / (args.batch_size * args.grad_accum))),
    )


def print_dry_run_summary(args: argparse.Namespace, train_ds, eval_ds) -> None:
    """Print a summary of data and config for dry-run validation."""

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
    print("  target_modules: q/k/v/o/gate/up/down_proj")
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
    if "input" in sample and "output" in sample:
        formatted = format_sample(sample)
        print(f"  input  ({len(sample['input'])} chars): {sample['input'][:120]!r}...")
        print(f"  output ({len(sample['output'])} chars): {sample['output']!r}")
        print(f"  formatted ({len(formatted)} chars total)")
    elif "text" in sample:
        print(f"  text ({len(sample['text'])} chars): {sample['text'][:120]!r}...")
    else:
        print(f"  keys: {list(sample.keys())}")
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

    # Load dataset (from cache if available)
    train_ds, eval_ds = load_dataset_splits(
        args.data, args.eval_split, args.seed,
        smoke_test=args.smoke_test,
        tokenized_cache=args.tokenized_cache,
    )

    # Subsample train if requested
    if args.max_train_samples > 0 and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
        logger.info("Subsampled train to %d samples", args.max_train_samples)

    # Dry-run: print summary and exit without touching a GPU
    if args.dry_run:
        print_dry_run_summary(args, train_ds, eval_ds)
        sys.exit(0)

    # --- From here: GPU required ---

    # Unsloth MUST be imported before trl/transformers/peft to apply its patches.
    # Importing it here (before SFTTrainer) satisfies that requirement even though
    # the model isn't loaded yet.
    from unsloth import FastLanguageModel  # noqa: F401 — side-effect import

    from transformers import TrainerCallback
    from trl import SFTTrainer

    class PushToHubCallback(TrainerCallback):
        """Push LoRA adapter to HuggingFace Hub every N steps to survive pod interruption."""
        def __init__(self, repo_id: str, every_n_steps: int = 50):
            self.repo_id = repo_id
            self.every_n_steps = every_n_steps

        def on_save(self, args, state, control, **kwargs):
            if state.global_step % self.every_n_steps == 0:
                try:
                    kwargs["model"].push_to_hub(self.repo_id, commit_message=f"checkpoint step {state.global_step}")
                    logger.info("Pushed checkpoint at step %d to %s", state.global_step, self.repo_id)
                except Exception as e:
                    logger.warning("Failed to push checkpoint at step %d: %s", state.global_step, e)

    # Load model + tokenizer
    model, tokenizer = build_model_and_tokenizer(args)

    # SFTConfig: TrainingArguments + SFT-specific options (completion_only_loss, max_length)
    training_args = build_training_args(args)

    # SFTTrainer handles tokenization, and completion-only loss internally
    callbacks = []
    if args.hf_repo:
        callbacks.append(PushToHubCallback(repo_id=args.hf_repo, every_n_steps=args.save_steps))

    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            total = state.max_steps
            if step % 10 == 0 or step == 1:
                print(f"Step {step}/{total} ({100*step//total}%)")

    callbacks.append(ProgressCallback())

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    # Benchmark: measure s/step on 20 warmup + 30 timed steps, then exit
    if args.benchmark:
        import time
        from transformers import TrainerCallback

        WARMUP_STEPS = 20
        TIMED_STEPS = 30

        class BenchmarkCallback(TrainerCallback):
            def __init__(self):
                self.step_times: list[float] = []
                self._t: float | None = None

            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step >= WARMUP_STEPS:
                    self._t = time.perf_counter()

            def on_step_end(self, args, state, control, **kwargs):
                if self._t is not None:
                    self.step_times.append(time.perf_counter() - self._t)
                    self._t = None
                if state.global_step >= WARMUP_STEPS + TIMED_STEPS:
                    control.should_training_stop = True

        bench_cb = BenchmarkCallback()
        trainer.add_callback(bench_cb)

        logger.info("Benchmark mode: %d warmup + %d timed steps ...", WARMUP_STEPS, TIMED_STEPS)
        trainer.train()

        times = bench_cb.step_times
        if times:
            avg = sum(times) / len(times)
            mn, mx = min(times), max(times)
            logger.info("Benchmark results (%d steps): avg=%.2fs  min=%.2fs  max=%.2fs", len(times), avg, mn, mx)

            # Extrapolate to full run
            train_samples = len(train_ds)
            effective_batch = args.batch_size * args.grad_accum
            steps_per_epoch = train_samples // effective_batch
            total_steps = steps_per_epoch * args.epochs
            checkpoint_saves = total_steps // args.save_steps
            checkpoint_overhead = checkpoint_saves * 5  # ~5s per LoRA checkpoint save

            print("\n" + "=" * 60)
            print("BENCHMARK EXTRAPOLATION")
            print("=" * 60)
            print(f"Measured:          {avg:.2f}s/step ({mn:.2f}-{mx:.2f}s)")
            print(f"Train samples:     {train_samples:,}")
            print(f"Steps/epoch:       {steps_per_epoch:,}")
            print(f"Total steps:       {total_steps:,}  ({args.epochs} epochs)")
            print(f"Checkpoint saves:  {checkpoint_saves}  (every {args.save_steps} steps, ~5s each)")
            total_sec = total_steps * avg + checkpoint_overhead
            print(f"Estimated time:    {total_sec/3600:.1f}h  ({total_sec/60:.0f}min)")
            print("=" * 60 + "\n")
        else:
            logger.warning("No timed steps recorded — increase dataset size or timed steps.")
        sys.exit(0)

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
    condition_tag = args.output_dir.name.replace("lora_", "")  # e.g. "condition4a"
    metrics_path = results_dir / f"{condition_tag}_train_metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
    logger.info("Training metrics saved to %s", metrics_path)

    # Push LoRA adapter to HuggingFace Hub
    if args.hf_repo:
        logger.info("Pushing LoRA weights to HuggingFace: %s ...", args.hf_repo)
        model.push_to_hub(args.hf_repo, private=False)
        tokenizer.push_to_hub(args.hf_repo, private=False)
        logger.info("Pushed to https://huggingface.co/%s", args.hf_repo)

    # Merge LoRA into base model and push merged weights
    if args.hf_repo_merged:
        logger.info("Merging LoRA into base model and pushing to HuggingFace: %s ...", args.hf_repo_merged)
        model.push_to_hub_merged(args.hf_repo_merged, tokenizer, save_method="merged_16bit", private=False)
        logger.info("Merged model pushed to https://huggingface.co/%s", args.hf_repo_merged)


if __name__ == "__main__":
    main()
