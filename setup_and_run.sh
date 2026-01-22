#!/usr/bin/env bash

# Exit immediately on error, treat unset variables as errors, and catch pipeline errors.
set -euo pipefail

# Move to the repository root (directory where this script lives).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# Upgrade pip to avoid old resolver issues.
python -m pip install --upgrade pip

# Install project dependencies from requirements.txt (PyTorch is assumed preinstalled).
python -m pip install -r requirements.txt

# Install the local Mamba package in editable mode (required by the project).
python -m pip install -e mamba

# Install a small helper library used to download OneDrive artifacts.
python -m pip install onedrivedownloader

# Verify that CUDA is visible to PyTorch and print basic GPU info.
python - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
PY

# Run a minimal text-to-motion inference using the helper Python script.
python run_inference.py
