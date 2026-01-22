#!/usr/bin/env bash

# Exit immediately on error, treat unset variables as errors, and catch pipeline errors.
set -euo pipefail

# Move to the repository root (directory where this script lives).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# Pick a Python binary (prefer python3.10 to match the project baseline).
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python"
  fi
fi

# Upgrade pip to avoid old resolver issues.
"${PYTHON_BIN}" -m pip install --upgrade pip

# Ensure PyTorch is installed before building torch-dependent packages.
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(1)
PY
if [[ $? -ne 0 ]]; then
  # Install PyTorch 1.12.x with CUDA wheels (adjust if your Paperspace image differs).
  "${PYTHON_BIN}" -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
fi

# Install project dependencies from requirements.txt after PyTorch is present.
"${PYTHON_BIN}" -m pip install -r requirements.txt

# Install the local Mamba package in editable mode (required by the project).
"${PYTHON_BIN}" -m pip install -e mamba

# Install a small helper library used to download OneDrive artifacts.
"${PYTHON_BIN}" -m pip install onedrivedownloader

# Verify that CUDA is visible to PyTorch and print basic GPU info.
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
PY

# Run a minimal text-to-motion inference using the helper Python script.
python run_inference.py
