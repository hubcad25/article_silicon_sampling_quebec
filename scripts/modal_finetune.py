"""
Launch a fine-tuning job on Modal for any SFT condition.

All conditions share the same Unsloth/LoRA setup and finetune.py script.
Only the input dataset, output paths, and HF repo differ.

Condition naming:
    4a  — question generalization, n_ctx=10  (new questions, known respondents)
    4b  — question generalization, n_ctx=15
    5a  — respondent generalization, n_ctx=10 (known questions, new respondents)
    5b  — respondent generalization, n_ctx=15
    6   — respondent gen + RAG at inference (extension)

Usage:
    # Pre-download base model into the volume (one-time, ~5 min)
    modal run scripts/modal_finetune.py --prepare

    # Condition 4A (question gen, n_ctx=10)
    modal run scripts/modal_finetune.py --condition 4a

    # Condition 5A (respondent gen, n_ctx=10)
    modal run scripts/modal_finetune.py --condition 5a

    # Smoke-test (1 epoch, 20 samples, pushes to a -smoke-test repo)
    modal run scripts/modal_finetune.py --condition 4a --smoke-test

    # Benchmark (measure s/step, then exit)
    modal run scripts/modal_finetune.py --condition 4a --benchmark

    # Override HF repo explicitly
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
# Conditions 4A/4B: question generalization (all respondents, n_ctx=10/15)
# Conditions 5A/5B: respondent generalization (train respondents only, n_ctx=10/15)
CONDITION_CONFIG = {
    "4a": {
        "local_data": "data/processed/finetune_train_4a.jsonl",
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
    "5a": {
        "local_data": "data/processed/finetune_train_5a.jsonl",
        "volume_data": "finetune_train_5a.jsonl",
        "output_dir": "/data/models/lora_condition5a",
        "hf_repo": "hubcad25/llama-3.1-8b-quebec-lora-condition5a",
        "hf_repo_merged": "hubcad25/llama-3.1-8b-quebec-condition5a",
    },
    "5b": {
        "local_data": "data/processed/finetune_train_5b.jsonl",
        "volume_data": "finetune_train_5b.jsonl",
        "output_dir": "/data/models/lora_condition5b",
        "hf_repo": "hubcad25/llama-3.1-8b-quebec-lora-condition5b",
        "hf_repo_merged": "hubcad25/llama-3.1-8b-quebec-condition5b",
    },
}

# ---------------------------------------------------------------------------
# Modal image — shared across all conditions
# ---------------------------------------------------------------------------
image = (
    # Use a CUDA base image and install unsloth via pip.
    # unsloth/unsloth:latest is a "studio" image (supervisord/Jupyter/SSH entrypoint)
    # which conflicts with Modal's function runner — it launches a Jupyter server
    # instead of executing the remote function.
    #
    # Single pip_install call so the resolver sees all constraints at once.
    # torch cu124 wheel served via --extra-index-url so pip can find it alongside PyPI.
    # Validate locally before changing pins:
    #   pip install --dry-run torch==2.6.0 unsloth trl transformers accelerate peft \
    #     datasets bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu124
    # Use debian_slim + unsloth CUDA variant (same approach as Modal's official example).
    # unsloth[cu128-torch270] installs pre-compiled kernels for CUDA 12.8 + torch 2.7.0.
    # This is critical — plain `unsloth` without the CUDA variant misses optimized kernels.
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "trl==0.19.1",
        "transformers==4.54.0",
        "accelerate==1.9.0",
        "peft==0.16.0",
        "datasets==3.6.0",
        "huggingface_hub==0.34.2",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
    )
    .add_local_file(local_path="scripts/finetune.py", remote_path="/root/finetune.py")
)

app = modal.App("finetune-silicon-sampling")

volume = modal.Volume.from_name("finetune-data", create_if_missing=True)

hf_token = os.environ.get("HF_TOKEN", "")
secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token})] if hf_token else []

# Paths inside the Modal volume
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"
DEFAULT_GPU = "A100-80GB"
VOLUME_HF_CACHE = "/data/hf_cache"  # HF_HOME inside the volume — survives across runs


# ---------------------------------------------------------------------------
# Prepare: download base model into volume (one-time setup)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/data": volume},
    secrets=secrets,
)
def download_base_model() -> None:
    """Download Llama-3.1-8B from HuggingFace into the Modal volume.

    Run once with: modal run scripts/modal_finetune.py --prepare
    Subsequent fine-tuning runs will load the model from the volume (~5s)
    instead of downloading from HF (~60s).
    """
    import os
    from pathlib import Path
    from huggingface_hub import snapshot_download

    import os
    from pathlib import Path

    # Point HF cache to the Modal volume so the download persists
    os.environ["HF_HOME"] = VOLUME_HF_CACHE

    # Check if model is already in the HF cache
    cache_dir = Path(VOLUME_HF_CACHE) / "hub"
    model_slug = BASE_MODEL_ID.replace("/", "--")
    cached = any(cache_dir.glob(f"models--{model_slug}")) if cache_dir.exists() else False

    if cached:
        print(f"Model already cached in {VOLUME_HF_CACHE} — skipping download.")
        return

    print(f"Downloading {BASE_MODEL_ID} -> HF cache at {VOLUME_HF_CACHE} ...")
    snapshot_download(
        repo_id=BASE_MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.pt", "original/*"],  # skip pytorch bin, keep safetensors
    )
    volume.commit()
    print(f"Model cached at {VOLUME_HF_CACHE}.")


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    volumes={"/data": volume},
    secrets=secrets,
)
def run_finetuning(
    volume_data: str,
    output_dir: str,
    hf_repo: str,
    hf_repo_merged: str,
    base_model: str,
    smoke_test: bool = False,
    benchmark: bool = False,
    max_train_samples: int = -1,
) -> None:
    import os
    from pathlib import Path

    print(f"Fine-tuning | data={volume_data} output={output_dir} model={base_model} repo={hf_repo} merged={hf_repo_merged} smoke={smoke_test} benchmark={benchmark}")

    # Point HF_HOME to the volume so the cached model is found automatically
    os.environ["HF_HOME"] = VOLUME_HF_CACHE

    print(f"HF_HOME={VOLUME_HF_CACHE} (model will load from volume cache if present)")

    # Use cached tokenized dataset if available
    tokenized_cache = f"/data/tokenized/{volume_data.replace('.jsonl', '')}"

    cmd = [
        "python", "/root/finetune.py",
        "--data", f"/data/{volume_data}",
        "--model", base_model,
        "--output_dir", output_dir,
        "--epochs", "1" if (smoke_test or benchmark) else "3",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--batch_size", "4",
        "--grad_accum", "8",
        "--lr", "2e-4",
        "--max_seq_len", "2560",  # p99 of actual token lengths is ~2107; 2560 covers all samples
        "--hf_repo", hf_repo,
        "--hf_repo_merged", hf_repo_merged,
        "--tokenized_cache", tokenized_cache,
    ]
    if smoke_test:
        cmd.extend(["--max_train_samples", "20", "--smoke_test"])
    if benchmark:
        cmd.append("--benchmark")
    if max_train_samples > 0:
        cmd.extend(["--max_train_samples", str(max_train_samples)])
    env = os.environ.copy()
    env["HF_HOME"] = VOLUME_HF_CACHE
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    condition: str = "4a",
    hf_repo: str = "",
    base_model: str = "",
    smoke_test: bool = False,
    benchmark: bool = False,
    max_train_samples: int = -1,
    prepare: bool = False,
) -> None:
    """
    Args:
        condition:          Which SFT condition to train (4a, 4b, 5a, 5b).
        hf_repo:            Override the default HF model repo for this condition.
        base_model:         Override base model (e.g. meta-llama/Llama-3.2-3B-Instruct).
        smoke_test:         1 epoch / 20 samples, pushes to <hf_repo>-smoke-test.
        benchmark:          20 warmup + 30 timed steps, prints s/step estimate, exits.
        max_train_samples:  Subsample train set to this many samples (-1 = all).
        prepare:            Download base model into volume (one-time setup), then exit.
    """
    if prepare:
        print(f"Downloading {BASE_MODEL_ID} into Modal volume ...")
        download_base_model.remote()
        print("Done. Base model is now cached in the volume.")
        return

    model_id = base_model or BASE_MODEL_ID

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

    if benchmark:
        print(f"Benchmark mode: 20 warmup + 30 timed steps, will exit after reporting s/step.")

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

    print(f"Launching condition {condition} fine-tuning on Modal -> model={model_id} LoRA={repo}")
    run_finetuning.remote(volume_data, output_dir, repo, repo_merged, model_id, smoke_test, benchmark, max_train_samples)
