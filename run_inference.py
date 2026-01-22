"""Minimal text-to-motion inference runner for Light-T2M."""

from __future__ import annotations

# Standard library imports for file paths, subprocess calls, and error handling.
import subprocess
import sys
from pathlib import Path

# Third-party import for CUDA availability checks.
import torch

# Optional helper for downloading OneDrive artifacts.
from onedrivedownloader import download as onedrive_download

# Define repository-relative paths to avoid hardcoded absolute paths.
REPO_ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = REPO_ROOT / "downloads"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "hml3d.ckpt"
DEPS_GLOVE_DIR = REPO_ROOT / "deps" / "glove"
DEPS_T2M_DIR = REPO_ROOT / "deps" / "t2m_guo"
DATA_STATS_DIR = REPO_ROOT / "data" / "HumanML3D"

# OneDrive URLs from the project README for dependencies and pretrained checkpoints.
DEPS_URL = "https://1drv.ms/u/s!ApyE_Lf3PFl2i4NcE8mgVUN3oX9nTQ?e=345HR5"
CKPT_URL = "https://1drv.ms/u/s!ApyE_Lf3PFl2i4Nb_QxAif-rcumPlg?e=O82IX1"


def download_if_missing(url: str, zip_name: str, expected_paths: list[Path]) -> None:
    """Download and unzip OneDrive artifacts if expected paths are missing."""

    # Check whether all expected paths already exist.
    if all(path.exists() for path in expected_paths):
        print("Artifacts already present, skipping download.")
        return

    # Create the downloads directory if needed.
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOADS_DIR / zip_name

    # Download and unzip the artifacts into the repository root.
    print(f"Downloading {url} to {zip_path}...")
    onedrive_download(url, filename=str(zip_path), unzip=True, unzip_path=str(REPO_ROOT))

    # Validate that the expected paths are now present.
    missing = [path for path in expected_paths if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Expected artifacts missing after download: {missing_str}")


def ensure_required_assets() -> None:
    """Ensure dependencies and checkpoints are available for inference."""

    # Download dependency archives (e.g., GloVe and T2M evaluator files) if missing.
    download_if_missing(DEPS_URL, "deps.zip", [DEPS_GLOVE_DIR, DEPS_T2M_DIR])

    # Download pretrained checkpoints if missing.
    download_if_missing(CKPT_URL, "checkpoints.zip", [CHECKPOINT_PATH])

    # Check that dataset statistics are available for mean/std normalization.
    mean_path = DATA_STATS_DIR / "Mean.npy"
    std_path = DATA_STATS_DIR / "Std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            "Missing dataset statistics. Please place Mean.npy and Std.npy under "
            f"{DATA_STATS_DIR} before running inference. You can download the "
            "HumanML3D dataset (which contains these files) following the "
            "instructions in the repository README."
        )


def run_sample_motion() -> None:
    """Run the sample_motion script with a small, reproducible configuration."""

    # Choose device based on CUDA availability.
    device = "0" if torch.cuda.is_available() else "cpu"

    # Build the command with Hydra overrides for a minimal inference run.
    command = [
        sys.executable,
        "src/sample_motion.py",
        f"ckpt_path={CHECKPOINT_PATH}",
        f"data_dir={DATA_STATS_DIR}",
        "save_path=./visual_datas",
        "sample_name=quick_start",
        "text=A person walks forward and waves their right hand.",
        "length=60",
        "repeats=1",
        f"device={device}",
        "model.guidance_scale=4",
        "model.noise_scheduler.prediction_type=sample",
    ]

    # Execute the command and stream output to the console.
    print("Running inference:", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    """Main entry point that prepares assets and runs inference."""

    # Ensure that required assets are present (download if necessary).
    ensure_required_assets()

    # Run the minimal sample motion generation.
    run_sample_motion()


if __name__ == "__main__":
    # Execute the main function when run as a script.
    main()
