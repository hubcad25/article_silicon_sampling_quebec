#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def generate():
    n_ctx_list = [10, 15, 25, 50]
    modes = [
        ("q", False), # Question Gen: all respondents
        ("r", True),  # Respondent Gen: train respondents only
    ]
    
    script = "scripts/07_generate_finetune_dataset.py"
    
    for mode_name, train_only in modes:
        for n in n_ctx_list:
            output = f"data/processed/finetune_train_{mode_name}_nctx{n}.jsonl"
            cmd = [
                sys.executable, script,
                "--n-ctx", str(n),
                "--output", output
            ]
            if train_only:
                cmd.append("--train-respondents-only")
            
            print(f"Generating: {output} ...")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    generate()
