#!/bin/bash
# scripts/setup_narval.sh - One-click environment setup for Narval

echo "--- Starting Narval Environment Setup ---"

# 1. Load required modules
echo "Loading Python and CUDA modules..."
module load python/3.10 cuda/12.1

# 2. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python -m venv .venv
else
    echo ".venv already exists."
fi

# 3. Activate and install dependencies
echo "Activating environment and installing requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Success
echo "--- Setup Complete! ---"
echo "You can now run your jobs using: jobs/sft_runner.sh"
