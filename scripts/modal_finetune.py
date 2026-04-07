import os
import subprocess

import modal

# On charge dotenv localement SEULEMENT (pas dans le conteneur Modal)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = modal.App("finetune-condition4")

# (1) Environnement GPU: Unsloth et dépendances
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.4.0", "huggingface_hub")
    .run_commands(
        # The torchao package requires torch > 2.4.0 in its recent versions.
        # Transformers quantizers fails to import if torchao errors out.
        # We also need transformers and trl specifically pinned or unsloth handles it,
        # but let's install unsloth first.
        "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
        # Install the rest, but force an older torchao or just rely on unsloth dependencies
        "pip install --no-deps trl peft accelerate bitsandbytes datasets",
        "pip install torchao==0.6.1" # Fix for torch.int1 error in transformers/quantizers
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
def run_finetuning(hf_repo: str, smoke_test: bool = False):
    print(f"Début du fine-tuning sur Modal (smoke_test={smoke_test})...")
    
    # (3) Lance le fine-tuning avec les bons paramètres et (4) Push sur HF
    cmd = [
        "python", "/root/finetune.py",
        "--data", "/data/finetune_train.jsonl",
        "--output_dir", "/data/models/lora_condition4",
        "--epochs", "3" if not smoke_test else "1",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--batch_size", "4",
        "--grad_accum", "8",
        "--lr", "2e-4",
        "--max_seq_len", "4096",
        "--hf_repo", hf_repo
    ]
    
    if smoke_test:
        print("Mode smoke test activé : entraînement limité à 20 échantillons.")
        cmd.extend(["--max_train_samples", "20", "--smoke_test"])
    
    # Lancement du sous-processus de training
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(
    hf_repo: str = "hubcad25/llama-3.1-8b-quebec-lora-condition4",
    smoke_test: bool = False
):
    """
    Point d'entrée local.
    Usage normal : modal run scripts/modal_finetune.py
    Usage test   : modal run scripts/modal_finetune.py --smoke-test
    """
    local_data_path = "data/processed/finetune_train.jsonl"
    
    if smoke_test:
        # On renomme le dépôt HF pour ne pas polluer le vrai modèle
        hf_repo = hf_repo + "-smoke-test"
        print(f"ATTENTION: Mode smoke_test activé. Le modèle sera poussé sur: {hf_repo}")
    
    # (2) Uploade finetune_train.jsonl dans le volume via la CLI Modal
    if os.path.exists(local_data_path):
        print(f"Upload de {local_data_path} vers le volume Modal 'finetune-data'...")
        # Ajout du flag --force (-f) pour écraser le fichier s'il est déjà présent
        subprocess.run(["modal", "volume", "put", "-f", "finetune-data", local_data_path, "finetune_train.jsonl"], check=True)
        print("Upload terminé.")
    else:
        print(f"Attention: {local_data_path} non trouvé localement.")
        print("Veuillez vous assurer que le fichier de données est présent dans le volume, ou qu'il sera téléchargé par finetune.py.")
        
    print(f"Lancement du job de fine-tuning sur Modal (push prévu sur {hf_repo})...")
    run_finetuning.remote(hf_repo, smoke_test)
