"""Internal weight management for img2pose.

This module handles downloading and caching pre-trained model weights.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import torch

# Model registry with URLs and checksums
MODEL_REGISTRY = {
    "img2pose_v1": {
        "url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/img2pose_v1.pth",
        "sha256": None,  # Will be populated when model is uploaded
        "filename": "img2pose_v1.pth",
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


def _download_file(url: str, destination: Path, show_progress: bool = True) -> None:
    """Download a file from URL to destination with optional progress bar."""

    def _progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            sys.stdout.write(f"\rDownloading: [{bar}] {percent}%")
            sys.stdout.flush()

    destination.parent.mkdir(parents=True, exist_ok=True)

    hook = _progress_hook if show_progress else None
    try:
        urllib.request.urlretrieve(url, str(destination), reporthook=hook)
        if show_progress:
            print()  # newline after progress bar
    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")


def _verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verify SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_sha256


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
    model_path = cache_dir / model_info["filename"]

    # Check if already cached
    if model_path.exists() and not force_download:
        # Optionally verify checksum
        expected_sha256 = model_info.get("sha256")
        if expected_sha256 and not _verify_checksum(model_path, expected_sha256):
            print(f"Checksum mismatch for {model_path}, re-downloading...")
        else:
            return model_path

    # Download the model
    print(f"Downloading {model_name} to {model_path}...")
    _download_file(model_info["url"], model_path)

    # Verify checksum if available
    expected_sha256 = model_info.get("sha256")
    if expected_sha256 and not _verify_checksum(model_path, expected_sha256):
        model_path.unlink()
        raise RuntimeError(f"Downloaded file checksum mismatch for {model_name}")

    return model_path


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
    checkpoint = torch.load(model_path, map_location=device)

    if "fpn_model" not in checkpoint:
        raise RuntimeError(
            f"Invalid checkpoint format: missing 'fpn_model' key. "
            f"Found keys: {list(checkpoint.keys())}"
        )

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

    checkpoint = torch.load(path, map_location=device)

    if "fpn_model" not in checkpoint:
        raise RuntimeError(
            f"Invalid checkpoint format: missing 'fpn_model' key. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    return checkpoint
