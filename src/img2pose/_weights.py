"""Internal weight management for img2pose.

This module handles downloading and caching pre-trained model weights.
"""

import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import torch

# Model registry with Google Drive file IDs
# The download is a ZIP containing: models/img2pose_v1.pth, pose_mean.npy, pose_stddev.npy
MODEL_REGISTRY = {
    "img2pose_v1": {
        "gdrive_id": "1OvnZ7OUQFg2bAgFADhT7UnCkSaXst10O",
        "zip_filename": "img2pose_v1.zip",
        "model_path_in_zip": "models/img2pose_v1.pth",
        "pose_mean_in_zip": "models/WIDER_train_pose_mean_v1.npy",
        "pose_stddev_in_zip": "models/WIDER_train_pose_stddev_v1.npy",
    },
}

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "img2pose"


def get_cache_dir() -> Path:
    """Get the cache directory for model weights.

    Returns the path from IMG2POSE_CACHE environment variable if set,
    otherwise uses ~/.cache/img2pose/
    """
    cache_dir = os.environ.get("IMG2POSE_CACHE", str(DEFAULT_CACHE_DIR))
    return Path(cache_dir)


def _download_from_gdrive(file_id: str, destination: Path, show_progress: bool = True) -> None:
    """Download a file from Google Drive.

    Uses gdown library which handles Google Drive's confirmation pages
    and large file downloads properly.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download model weights from Google Drive. "
            "Install it with: pip install gdown"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(destination), quiet=not show_progress)
    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download from Google Drive (id={file_id}): {e}")


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP file to the specified directory."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def get_model_path(
    model_name: str = "img2pose_v1",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Get path to model weights, downloading if necessary.

    Args:
        model_name: Name of the model in the registry
        cache_dir: Custom cache directory (default: ~/.cache/img2pose/)
        force_download: Force re-download even if cached

    Returns:
        Path to the model weights file

    Raises:
        ValueError: If model_name is not in registry
        RuntimeError: If download fails
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model_info = MODEL_REGISTRY[model_name]
    cache_dir = cache_dir or get_cache_dir()

    # Path to extracted model file
    model_path = cache_dir / model_info["model_path_in_zip"]
    zip_path = cache_dir / model_info["zip_filename"]

    # Check if already cached
    if model_path.exists() and not force_download:
        return model_path

    # Download the ZIP from Google Drive
    print(f"Downloading {model_name}...")
    _download_from_gdrive(model_info["gdrive_id"], zip_path)

    # Extract the ZIP
    print(f"Extracting to {cache_dir}...")
    _extract_zip(zip_path, cache_dir)

    # Clean up ZIP file
    zip_path.unlink()

    if not model_path.exists():
        raise RuntimeError(
            f"Expected model file not found after extraction: {model_path}"
        )

    return model_path


def get_pose_stats_paths(
    model_name: str = "img2pose_v1",
    cache_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Get paths to pose mean and stddev files.

    Downloads model package if not already cached.

    Args:
        model_name: Name of the model in the registry
        cache_dir: Custom cache directory

    Returns:
        Tuple of (pose_mean_path, pose_stddev_path)
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    # Ensure model is downloaded (this also extracts pose stats)
    get_model_path(model_name, cache_dir)

    model_info = MODEL_REGISTRY[model_name]
    cache_dir = cache_dir or get_cache_dir()

    pose_mean_path = cache_dir / model_info["pose_mean_in_zip"]
    pose_stddev_path = cache_dir / model_info["pose_stddev_in_zip"]

    return pose_mean_path, pose_stddev_path


def _strip_module_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix from state dict keys.

    This handles checkpoints saved from DataParallel-wrapped models.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Strip "module." (7 chars)
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_weights(
    model_name: str = "img2pose_v1",
    cache_dir: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    """Load model weights from cache or download.

    Args:
        model_name: Name of the model in the registry
        cache_dir: Custom cache directory
        device: Device to load weights to

    Returns:
        Dictionary containing model state dict under "fpn_model" key
    """
    model_path = get_model_path(model_name, cache_dir)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "fpn_model" not in checkpoint:
        raise RuntimeError(
            f"Invalid checkpoint format: missing 'fpn_model' key. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    # Strip "module." prefix from keys (handles DataParallel checkpoints)
    checkpoint["fpn_model"] = _strip_module_prefix(checkpoint["fpn_model"])

    return checkpoint


def load_weights_from_path(
    path: str,
    device: str = "cpu",
) -> dict:
    """Load model weights from a local path.

    Args:
        path: Path to the model weights file
        device: Device to load weights to

    Returns:
        Dictionary containing model state dict under "fpn_model" key
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "fpn_model" not in checkpoint:
        raise RuntimeError(
            f"Invalid checkpoint format: missing 'fpn_model' key. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    # Strip "module." prefix from keys (handles DataParallel checkpoints)
    checkpoint["fpn_model"] = _strip_module_prefix(checkpoint["fpn_model"])

    return checkpoint
