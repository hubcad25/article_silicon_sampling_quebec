#!/usr/bin/env python3
"""Profile a single training step to identify bottlenecks."""
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

print("Loading dataset sample...")
t0 = time.time()
ds = load_dataset("hubcad25/article_silicon_sampling_quebec_tokenized", token=token, split="train[:32]")
print(f"  Dataset loaded in {time.time()-t0:.1f}s")
print(f"  Sample length: {len(ds[0]['input_ids'])} tokens")

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, token=token)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("\nLoading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    token=token,
)
model.config.use_cache = False
print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Build a batch of 16
from torch.nn.utils.rnn import pad_sequence
ids = [torch.tensor(ds[i]['input_ids'][:2048]) for i in range(16)]
masks = [torch.tensor(ds[i]['attention_mask'][:2048]) for i in range(16)]
input_ids = pad_sequence(ids, batch_first=True, padding_value=tok.pad_token_id).cuda()
attention_mask = pad_sequence(masks, batch_first=True, padding_value=0).cuda()
labels = input_ids.clone()
print(f"\nBatch shape: {input_ids.shape}")

# Warmup
print("\nWarmup forward pass...")
with torch.no_grad():
    _ = model(input_ids=input_ids[:1], attention_mask=attention_mask[:1])

# Profile forward
print("\nForward pass (batch=16)...")
torch.cuda.synchronize()
t0 = time.time()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
torch.cuda.synchronize()
fwd_time = time.time() - t0
print(f"  Forward: {fwd_time:.2f}s | loss={out.loss.item():.3f}")

# Profile backward
print("Backward pass...")
torch.cuda.synchronize()
t0 = time.time()
out.loss.backward()
torch.cuda.synchronize()
bwd_time = time.time() - t0
print(f"  Backward: {bwd_time:.2f}s")

print(f"\nTotal per step (fwd+bwd): {fwd_time+bwd_time:.2f}s")
print(f"GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
