#!/bin/bash
#SBATCH --job-name=sft_quebec
#SBATCH --output=logs/sft_%j.log
#SBATCH --error=logs/sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=def-youruser # TODO: Change this to your account

# Usage: sbatch scripts/submit_sft.sh <target> <model_size> <n_ctx>
# Example: sbatch scripts/submit_sft.sh q 1b 10

set -e

TARGET=$1
SIZE=$2
CTX=$3

if [[ -z "$TARGET" || -z "$SIZE" || -z "$CTX" ]]; then
    echo "Usage: sbatch scripts/submit_sft.sh <target> <model_size> <n_ctx>"
    exit 1
fi

# Load environment
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting SFT for target=$TARGET, size=$SIZE, ctx=$CTX"

# Run training
# Note: HF_HOME is used from environment if set, otherwise defaults to ~/.cache/huggingface
python scripts/finetune.py \
    --target "$TARGET" \
    --model_size "$SIZE" \
    --n_ctx "$CTX" \
    --report_to tensorboard

echo "SFT completed successfully."
