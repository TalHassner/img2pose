"""Pytest configuration and fixtures for img2pose tests."""

import sys
from pathlib import Path


def pytest_configure(config):
    """Fix Python path before tests run.

    This removes the repo root from sys.path to avoid the img2pose.py file
    shadowing the installed img2pose package.
    """
    repo_root = str(Path(__file__).parent.parent)
    src_path = str(Path(__file__).parent.parent / "src")

    # Remove repo root if present (to avoid img2pose.py conflict)
    while repo_root in sys.path:
        sys.path.remove(repo_root)

    # Ensure src directory is at the front
    if src_path in sys.path:
        sys.path.remove(src_path)
    sys.path.insert(0, src_path)


import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    return Image.new("RGB", (640, 480), color="white")


@pytest.fixture
def sample_rgb_array():
    """Create a sample RGB numpy array for testing."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


@pytest.fixture
def sample_face_dict():
    """Create a sample face detection result."""
    return {
        "box": [100, 100, 200, 200],
        "confidence": 0.95,
        "pose": [0.1, -0.2, 0.05, 0.0, 0.0, 1.0],
        "keypoints": {
            "left_eye": [130, 130],
            "right_eye": [170, 130],
            "nose": [150, 150],
            "mouth_left": [135, 175],
            "mouth_right": [165, 175],
        },
    }


@pytest.fixture
def pose_reference_68():
    """Load or create 68-point 3D reference."""
    try:
        from pathlib import Path
        ref_path = Path(__file__).parent.parent / "src" / "img2pose" / "data" / "reference_3d_68_points_trans.npy"
        return np.load(ref_path)
    except FileNotFoundError:
        # Return random points for testing if file not found
        return np.random.randn(68, 3)


@pytest.fixture
def pose_reference_5():
    """Load or create 5-point 3D reference."""
    try:
        from pathlib import Path
        ref_path = Path(__file__).parent.parent / "src" / "img2pose" / "data" / "reference_3d_5_points_trans.npy"
        return np.load(ref_path)
    except FileNotFoundError:
        # Return random points for testing if file not found
        return np.random.randn(5, 3)
