#!/bin/bash
# jobs/sft_runner.sh - Unified Slurm template for 2x4x4 SFT Matrix
#
# Logic:
# 1. Sources .env for ACCOUNT and base paths.
# 2. Uses TARGET, SIZE, CTX from environment (passed via command line or .env).
# 3. Sets Slurm resources dynamically based on SIZE.

# Source configuration
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Validation of mandatory matrix variables
if [ -z "$TARGET" ] || [ -z "$SIZE" ] || [ -z "$CTX" ]; then
    echo "Error: TARGET, SIZE, and CTX must be set (in .env or as environment variables)."
    echo "Example: TARGET=q SIZE=8b CTX=15 sbatch jobs/sft_runner.sh"
    exit 1
fi

# Resource profiling (Narval/Beluga optimized)
case $SIZE in
    "0.5b"|"1b")
        GPU="a100:1"
        MEM="32G"
        TIME="03:00:00"
        CPUS=8
        ;;
    "8b")
        GPU="a100:1"
        MEM="64G"
        TIME="08:00:00"
        CPUS=12
        ;;
    "70b")
        GPU="a100:1" 
        MEM="128G"
        TIME="24:00:00"
        CPUS=16
        ;;
    *)
        GPU="a100:1"
        MEM="32G"
        TIME="03:00:00"
        CPUS=8
        ;;
esac

# If running as a submission wrapper
if [[ "$1" == "--submit" ]]; then
    sbatch --account=${SLURM_ACCOUNT} \
           --gres=gpu:${GPU} \
           --cpus-per-task=${CPUS} \
           --mem=${MEM} \
           --time=${TIME} \
           --job-name="sft_${TARGET}_${SIZE}_ctx${CTX}" \
           --output="logs/sft_%j_${TARGET}_${SIZE}.log" \
           "$0"
    exit 0
fi

# Execution on Compute Node
module load python/3.10 cuda/12.1 2>/dev/null || echo "Running outside Slurm/Narval environment"

export HF_HOME=${HF_HOME:-"/scratch/$USER/hf_cache"}
mkdir -p "$HF_HOME"

source .venv/bin/activate

python scripts/finetune.py \
    --target "$TARGET" \
    --model_size "$SIZE" \
    --n_ctx "$CTX" \
    --report_to tensorboard
