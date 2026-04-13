import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata = {
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}

nb.cells = [
    new_markdown_cell(
        "# Fine-tune on CES Survey Data (QLoRA, Colab)\n\n"
        "**Runtime → Change runtime type → GPU**\n\n"
        "Edit the CONFIG cell below to change model, condition, and training params.\n\n"
        "**To pull latest changes** after initial clone:\n"
        "```python\n"
        "!git -C article_silicon_sampling_quebec pull origin main\n"
        "```"
    ),
    new_code_cell(
        "# ============================================================\n"
        "# CONFIG — edit these before running\n"
        "# ============================================================\n\n"
        "# Your HF token (https://huggingface.co/settings/tokens)\n"
        'HF_TOKEN = ""  # <-- SET YOUR TOKEN HERE\n\n'
        "# Your W&B API key (https://wandb.ai/authorize) — leave empty to disable\n"
        'WANDB_API_KEY = ""  # <-- SET YOUR KEY HERE\n\n'
        "# Model options (BASE models — not Instruct):\n"
        "#   Qwen/Qwen2.5-0.5B                    (~3GB VRAM, RECOMMENDED for publication)\n"
        "#   Qwen/Qwen2.5-1.5B                    (~5GB VRAM)\n"
        "#   Qwen/Qwen2.5-3B                      (~8GB VRAM)\n"
        "#   meta-llama/Llama-3.2-1B              (~4GB VRAM)\n"
        "#   meta-llama/Llama-3.2-3B              (~8GB VRAM)\n"
        'MODEL_NAME = "Qwen/Qwen2.5-0.5B"\n\n'
        "# Dataset condition (from huggingface.co/datasets/hubcad25/article_silicon_sampling_quebec_data)\n"
        "# Options: finetune_train_4a.jsonl, 4b.jsonl, 5a.jsonl, 5b.jsonl\n"
        'DATASET_FILE = "finetune_train_4a.jsonl"\n\n'
        "# Training params\n"
        "EPOCHS = 1          # 1 for smoke test, 3 for full run\n"
        "MAX_TRAIN_SAMPLES = 5000  # Remove/comment for full dataset\n"
        'HF_REPO = "hubcad25/qwen-0.5b-lora-4a"  # Output HF repo\n'
        "# ============================================================\n"
        "import os\n"
        'os.environ["HF_TOKEN"] = HF_TOKEN\n'
        "if WANDB_API_KEY:\n"
        '    os.environ["WANDB_API_KEY"] = WANDB_API_KEY\n'
        "print(f\"Model: {MODEL_NAME}\")\n"
        "print(f\"Dataset: {DATASET_FILE}\")\n"
        "print(f\"Epochs: {EPOCHS}, max_train_samples: {MAX_TRAIN_SAMPLES or 'ALL'}\")\n"
        "print(f\"W&B: {'enabled' if WANDB_API_KEY else 'disabled'}\")"
    ),
    new_code_cell(
        "# Install dependencies\n"
        "!pip install -q unsloth trl transformers accelerate peft datasets huggingface_hub wandb\n"
        "!pip install -q polars\n\n"
        "# Login to W&B if key is set\n"
        "if WANDB_API_KEY:\n"
        "    !wandb login $WANDB_API_KEY"
    ),
    new_code_cell(
        "# Clone repo + download dataset\n"
        "!git clone https://github.com/hubcad25/article_silicon_sampling_quebec.git\n"
        "import os\n"
        'os.chdir("article_silicon_sampling_quebec")\n\n'
        "from huggingface_hub import hf_hub_download\n"
        "dataset_local = hf_hub_download(\n"
        '    repo_id="hubcad25/article_silicon_sampling_quebec_data",\n'
        "    filename=DATASET_FILE,\n"
        "    token=HF_TOKEN,\n"
        '    repo_type="dataset",\n'
        ")\n"
        'print(f"Dataset: {dataset_local}")\n'
        "!wc -l {dataset_local}"
    ),
    new_code_cell(
        "# Run fine-tuning\n"
        'OUTPUT_DIR = "/content/model_output"\n'
        "import subprocess\n\n"
        "cmd = [\n"
        '    "python", "scripts/finetune.py",\n'
        "    \"--data\", dataset_local,\n"
        "    \"--model\", MODEL_NAME,\n"
        '    "--output_dir", OUTPUT_DIR,\n'
        "    \"--use_4bit\",\n"
        "    \"--epochs\", str(EPOCHS),\n"
        '    "--batch_size", "4",\n'
        '    "--grad_accum", "8",\n'
        '    "--lr", "2e-4",\n'
        '    "--max_seq_len", "2048",\n'
        "    \"--hf_repo\", HF_REPO,\n"
        "]\n"
        "if MAX_TRAIN_SAMPLES:\n"
        "    cmd.extend([\"--max_train_samples\", str(MAX_TRAIN_SAMPLES)])\n\n"
        "if WANDB_API_KEY:\n"
        "    cmd.append(\"--report_to\")\n"
        "    cmd.append(\"wandb\")\n\n"
        "print(f\"Running: {' '.join(cmd[:8])} ...\")\n"
        "env = os.environ.copy()\n"
        "result = subprocess.run(cmd, env=env, capture_output=True, text=True)\n"
        "if result.returncode == 0:\n"
        "    print(\"Done. LoRA pushed to HF.\")\n"
        "else:\n"
        "    print(f\"Failed (exit {result.returncode})\")\n"
        "    print(\"=== STDERR ===\")\n"
        "    print(result.stderr[-4000:])\n"
        "    print(\"=== STDOUT ===\")\n"
        "    print(result.stdout[-2000:])"
    ),
    new_markdown_cell(
        "---\n\n"
        "## Monitoring (run while fine-tuning is running)\n\n"
        "Run this cell in a **new cell** to monitor progress."
    ),
    new_code_cell(
        "# Monitor GPU usage while fine-tuning runs\n"
        "!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader\n\n"
        "# Check for checkpoint files (shows if training is progressing)\n"
        "import os, glob\n"
        "if os.path.exists(\"/content/model_output\"):\n"
        "    checkpoints = sorted(glob.glob(\"/content/model_output/checkpoint-*\"))\n"
        "    print(f\"Checkpoints: {len(checkpoints)}\")\n"
        "    if checkpoints:\n"
        "        print(f\"Latest: {checkpoints[-1]}\")\n"
        "else:\n"
        "    print(\"No output yet — still initializing\")"
    ),
]

nbformat.validate(nb)
with open("notebooks/finetune_colab.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Notebook written successfully")
