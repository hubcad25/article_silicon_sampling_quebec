"""
Launch a fine-tuning job on Modal for any SFT condition.

All conditions share the same Unsloth/LoRA setup and finetune.py script.
Only the input dataset, output paths, and HF repo differ.

Usage:
    # Condition 4A (question generalisation)
    modal run scripts/modal_finetune.py --condition 4a

    # Condition 4B (respondent generalisation)
    modal run scripts/modal_finetune.py --condition 4b

    # Smoke-test (1 epoch, 20 samples, pushes to a -smoke-test repo)
    modal run scripts/modal_finetune.py --condition 4a --smoke-test

    # Override HF repo explicitly (overrides the per-condition default)
    modal run scripts/modal_finetune.py --condition 4a --hf-repo org/my-model
"""

import os
import subprocess

import modal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Per-condition configuration
# ---------------------------------------------------------------------------
CONDITION_CONFIG = {
    "4a": {
        "local_data": "data/processed/finetune_train.jsonl",
        "volume_data": "finetune_train_4a.jsonl",
        "output_dir": "/data/models/lora_condition4a",
        "hf_repo": "hubcad25/llama-3.1-8b-quebec-lora-condition4a",
        "hf_repo_merged": "hubcad25/llama-3.1-8b-quebec-condition4a",
    },
    "4b": {
        "local_data": "data/processed/finetune_train_4b.jsonl",
        "volume_data": "finetune_train_4b.jsonl",
        "output_dir": "/data/models/lora_condition4b",
        "hf_repo": "hubcad25/llama-3.1-8b-quebec-lora-condition4b",
        "hf_repo_merged": "hubcad25/llama-3.1-8b-quebec-condition4b",
    },
}

# ---------------------------------------------------------------------------
# Modal image — shared across all conditions
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "huggingface_hub")
    .run_commands(
        # Install unsloth; pin torchao to a version compatible with torch 2.6
        "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
        "pip install 'torchao==0.8.0'",
    )
    .add_local_file(local_path="scripts/finetune.py", remote_path="/root/finetune.py")
)

app = modal.App("finetune-silicon-sampling")

volume = modal.Volume.from_name("finetune-data", create_if_missing=True)

hf_token = os.environ.get("HF_TOKEN", "")
secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token})] if hf_token else []


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    volumes={"/data": volume},
    secrets=secrets,
)
def run_finetuning(
    volume_data: str,
    output_dir: str,
    hf_repo: str,
    hf_repo_merged: str,
    smoke_test: bool = False,
) -> None:
    print(f"Fine-tuning | data={volume_data} output={output_dir} repo={hf_repo} merged={hf_repo_merged} smoke={smoke_test}")
    cmd = [
        "python", "/root/finetune.py",
        "--data", f"/data/{volume_data}",
        "--output_dir", output_dir,
        "--epochs", "1" if smoke_test else "3",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--batch_size", "4",
        "--grad_accum", "8",
        "--lr", "2e-4",
        "--max_seq_len", "4096",
        "--hf_repo", hf_repo,
        "--hf_repo_merged", hf_repo_merged,
    ]
    if smoke_test:
        cmd.extend(["--max_train_samples", "20", "--smoke_test"])
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    condition: str = "4a",
    hf_repo: str = "",
    smoke_test: bool = False,
) -> None:
    """
    Args:
        condition:  Which SFT condition to train (4a, 4b, ...).
        hf_repo:    Override the default HF model repo for this condition.
        smoke_test: 1 epoch / 20 samples, pushes to <hf_repo>-smoke-test.
    """
    if condition not in CONDITION_CONFIG:
        raise SystemExit(f"Unknown condition '{condition}'. Valid: {list(CONDITION_CONFIG)}")

    cfg = CONDITION_CONFIG[condition]
    local_data = cfg["local_data"]
    volume_data = cfg["volume_data"]
    output_dir = cfg["output_dir"]
    repo = hf_repo or cfg["hf_repo"]
    repo_merged = cfg["hf_repo_merged"]

    if smoke_test:
        repo = repo + "-smoke-test"
        repo_merged = repo_merged + "-smoke-test"
        print(f"Smoke-test mode: LoRA -> {repo}, merged -> {repo_merged}")

    # Upload dataset to Modal volume
    if os.path.exists(local_data):
        print(f"Uploading {local_data} -> volume:/{volume_data} ...")
        subprocess.run(
            ["modal", "volume", "put", "-f", "finetune-data", local_data, volume_data],
            check=True,
        )
        print("Upload complete.")
    else:
        print(f"Warning: {local_data} not found locally — assuming it is already in the volume.")

    print(f"Launching condition {condition} fine-tuning on Modal -> LoRA: {repo}, merged: {repo_merged}")
    run_finetuning.remote(volume_data, output_dir, repo, repo_merged, smoke_test)
