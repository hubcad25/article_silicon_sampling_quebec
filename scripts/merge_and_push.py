#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and push to HuggingFace.

Usage:
    HF_TOKEN=hf_xxx python scripts/merge_and_push.py

Requirements:
    pip install transformers peft accelerate huggingface_hub safetensors
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HF_TOKEN = os.environ["HF_TOKEN"]
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_REPO  = "hubcad25/lora_condition4"
OUTPUT_REPO = "hubcad25/lora_condition4_merged"

print("=== 1/4 Chargement du modele de base ===")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    token=HF_TOKEN,
)

print("=== 2/4 Chargement du tokenizer ===")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

print("=== 3/4 Application du LoRA et merge ===")
model = PeftModel.from_pretrained(
    model,
    LORA_REPO,
    token=HF_TOKEN,
)
model = model.merge_and_unload()
print("Merge terminé.")

print("=== 4/4 Push vers HuggingFace ===")
model.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN, private=True)
print(f"Done — https://huggingface.co/{OUTPUT_REPO}")
