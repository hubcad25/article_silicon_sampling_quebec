#!/bin/bash
# jobs/sft_runner.sh - Unified SFT Pipeline: Data Gen + Slurm Submission
# Usage:
#   TARGET=q SIZE=0.5b CTX=10 bash jobs/sft_runner.sh --submit

# 1. Load configuration from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 2. Validation
if [ -z "$TARGET" ] || [ -z "$SIZE" ] || [ -z "$CTX" ]; then
    echo "Error: TARGET, SIZE, and CTX must be set."
    exit 1
fi

DATA_FILE="data/processed/sft_${TARGET}_${CTX}.jsonl"

# 3. Data Generation (if missing)
if [ ! -f "$DATA_FILE" ]; then
    echo "Data file $DATA_FILE not found. Generating..."
    source .venv/bin/activate
    python scripts/generate_sft_data.py --target "$TARGET" --n-ctx "$CTX"
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed."
        exit 1
    fi
fi

# 4. Resource Profiling
if [[ "$SMOKE" == "1" ]]; then
    GPU="a100:1"; MEM="16G"; TIME="00:30:00"; CPUS=4
    SMOKE_FLAG="--smoke_test"
    JOB_PREFIX="smoke_"
else
    case $SIZE in
        "0.5b"|"1b") GPU="a100:1"; MEM="32G"; TIME="03:00:00"; CPUS=8 ;;
        "8b")        GPU="a100:1"; MEM="64G"; TIME="08:00:00"; CPUS=12 ;;
        "70b")       GPU="a100:1"; MEM="128G"; TIME="24:00:00"; CPUS=16 ;;
        *)           GPU="a100:1"; MEM="32G"; TIME="03:00:00"; CPUS=8 ;;
    esac
    SMOKE_FLAG=""
    JOB_PREFIX=""
fi

# 5. Handle Submission
if [[ "$1" == "--submit" ]]; then
    mkdir -p logs
    echo "Submitting SFT job: TARGET=$TARGET, SIZE=$SIZE, CTX=$CTX (SMOKE=$SMOKE)"
    sbatch --account=${SLURM_ACCOUNT} \
           --gres=gpu:${GPU} \
           --cpus-per-task=${CPUS} \
           --mem=${MEM} \
           --time=${TIME} \
           --job-name="${JOB_PREFIX}sft_${TARGET}_${SIZE}_ctx${CTX}" \
           --output="logs/sft_%j_${TARGET}_${SIZE}.log" \
           "$0"
    exit 0
fi

# 6. Execution (Compute Node)
echo "Running SFT Job... (SMOKE=$SMOKE)"
module load python/3.10 cuda/12.1 2>/dev/null
export HF_HOME=${HF_HOME:-"/scratch/$USER/hf_cache"}
mkdir -p "$HF_HOME"

source .venv/bin/activate
python scripts/finetune.py \
    --target "$TARGET" \
    --model_size "$SIZE" \
    --n_ctx "$CTX" \
    --report_to tensorboard \
    $SMOKE_FLAG
