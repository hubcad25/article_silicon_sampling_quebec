import os
import subprocess

from dotenv import load_dotenv
import modal

# Load environment variables (e.g., HF_TOKEN from .env)
load_dotenv()

app = modal.App("finetune-condition4")

# (1) Environnement GPU: Unsloth et dépendances
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.4.0", "huggingface_hub")
    .run_commands(
        "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
        "pip install --no-deps trl peft accelerate bitsandbytes datasets flash-attn xformers"
    )
    # (2) Ajout du script de fine-tuning directement dans l'image
    .add_local_file(local_path="scripts/finetune.py", remote_path="/root/finetune.py")
)

# Volume Modal pour stocker les données et le modèle
volume = modal.Volume.from_name("finetune-data", create_if_missing=True)

# Configuration du secret HF_TOKEN (depuis .env local)
hf_token = os.environ.get("HF_TOKEN", "")
secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token})] if hf_token else []

@app.function(
    image=image,
    gpu="A100",  # On peut remplacer par "H100" si désiré
    timeout=86400, # 24h max
    volumes={"/data": volume},
    secrets=secrets
)
def run_finetuning(hf_repo: str):
    print("Début du fine-tuning sur Modal...")
    
    # (3) Lance le fine-tuning avec les bons paramètres et (4) Push sur HF
    cmd = [
        "python", "/root/finetune.py",
        "--data", "/data/finetune_train.jsonl",
        "--output_dir", "/data/models/lora_condition4",
        "--epochs", "3",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--batch_size", "4",
        "--grad_accum", "8",
        "--lr", "2e-4",
        "--max_seq_len", "4096",
        "--hf_repo", hf_repo
    ]
    
    # Lancement du sous-processus de training
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(
    hf_repo: str = "hubcad25/llama-3.1-8b-quebec-lora-condition4"
):
    """
    Point d'entrée local.
    Usage: modal run scripts/modal_finetune.py
    """
    local_data_path = "data/processed/finetune_train.jsonl"
    
    # (2) Uploade finetune_train.jsonl dans le volume via la CLI Modal
    if os.path.exists(local_data_path):
        print(f"Upload de {local_data_path} vers le volume Modal 'finetune-data'...")
        # Lancement de la commande modal volume put
        subprocess.run(["modal", "volume", "put", "finetune-data", local_data_path, "finetune_train.jsonl"], check=True)
        print("Upload terminé.")
    else:
        print(f"Attention: {local_data_path} non trouvé localement.")
        print("Veuillez vous assurer que le fichier de données est présent dans le volume, ou qu'il sera téléchargé par finetune.py.")
        
    print(f"Lancement du job de fine-tuning sur Modal (push prévu sur {hf_repo})...")
    run_finetuning.remote(hf_repo)
