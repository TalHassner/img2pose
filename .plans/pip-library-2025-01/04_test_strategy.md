# Test Strategy: img2pose pip-installable Library

**Version:** 1.0
**Date:** 2025-01-27
**Reference:** `02_design_spec.md`, `03_implementation_task_breakdown.md`

---

## API Reference

The canonical API is defined in the main plan. Tests should use:
- **Method:** `detect_faces()` (not `detect()`)
- **Callable:** `detector(image)` as alias for `detect_faces()`
- **Output keys:** `"confidence"` (not `"score"`), `"keypoints"` dict (not `"landmarks_5pt"`)

---

## 1. Executive Summary

This document defines the testing strategy for the img2pose pip-installable library conversion. The existing repository has no pytest tests - only evaluation scripts and Jupyter notebooks. We will establish a comprehensive test suite from scratch, with a focus on:

1. **Numerical equivalence** with the existing codebase
2. **API contract verification** for the new public interface
3. **Offline execution** capability after initial model download
4. **CPU-only CI compatibility**

---

## 2. Test Taxonomy

### 2.1 Unit Tests

Unit tests verify individual components in isolation using mocks and fixtures.

| Test Module | Target | Key Tests |
|-------------|--------|-----------|
| `test_detector_init.py` | `Img2Pose.__init__()` | Device selection, backbone validation, threshold validation |
| `test_detector_input.py` | `Img2Pose._to_tensor()` | Path input, PIL input, numpy input, invalid input handling |
| `test_detector_output.py` | `Img2Pose._format_prediction()` | Score filtering, max_faces limiting, output format |
| `test_weights.py` | `_weights.py` | Cache dir resolution, download logic, checksum validation |
| `test_visualization.py` | `_visualization.py` | Box drawing, color handling, empty results |
| `test_pose_operations.py` | `utils/pose_operations.py` | Pose transformations, quaternion conversion |

### 2.2 Integration Tests

Integration tests verify component interactions with real model inference.

| Test Module | Target | Key Tests |
|-------------|--------|-----------|
| `test_integration_single.py` | Single image API | End-to-end detection on test image |
| `test_integration_batch.py` | Batch API | Multiple images, consistent results |
| `test_integration_device.py` | Device handling | CPU inference, GPU inference (if available) |
| `test_integration_visualization.py` | Full visualization pipeline | Detect + visualize flow |

### 2.3 Equivalence Tests

Equivalence tests ensure the new library produces identical results to the original code.

| Test Module | Target | Tolerance |
|-------------|--------|-----------|
| `test_equivalence_boxes.py` | Bounding box output | `atol=1e-4` (float32 precision) |
| `test_equivalence_poses.py` | 6DoF pose output | `atol=1e-4` (float32 precision) |
| `test_equivalence_scores.py` | Confidence scores | `atol=1e-6` (softmax precision) |
| `test_equivalence_aflw2000.py` | AFLW2000-3D benchmark | Full dataset comparison |

### 2.4 Negative Tests

Negative tests verify proper error handling for invalid inputs.

| Test Module | Target | Key Tests |
|-------------|--------|-----------|
| `test_errors_input.py` | Input validation | Invalid image path, wrong array shape, unsupported type |
| `test_errors_config.py` | Configuration validation | Invalid backbone depth, out-of-range threshold |
| `test_errors_model.py` | Model loading | Corrupted weights, missing files, wrong format |
| `test_errors_download.py` | Download failures | Network errors, invalid URL, checksum mismatch |

### 2.5 Schema/Contract Tests

Contract tests verify output format stability.

| Test Module | Target | Key Tests |
|-------------|--------|-----------|
| `test_output_schema.py` | Detection result format | Required keys, value types, array shapes |
| `test_config_schema.py` | Configuration options | Environment variable handling, path resolution |

---

## 3. Fixtures Strategy

### 3.1 Test Images

Create small, synthetic test images to avoid large file dependencies.

```python
# tests/conftest.py

import numpy as np
from PIL import Image
import pytest
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def small_test_image():
    """Create a small 64x64 RGB test image."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    return img

@pytest.fixture(scope="session")
def test_image_path(tmp_path_factory):
    """Create a temporary test image file."""
    path = tmp_path_factory.mktemp("images") / "test.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    img.save(path)
    return path

@pytest.fixture(scope="session")
def test_image_with_face():
    """Load the bundled test image with a known face."""
    return Image.open(TEST_DATA_DIR / "face_test.jpg")

@pytest.fixture(scope="session")
def test_image_numpy():
    """Create a numpy RGB array."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture(scope="session")
def batch_test_images(small_test_image):
    """Create a batch of test images."""
    return [small_test_image.copy() for _ in range(3)]
```

### 3.2 Mock Weights for Unit Tests

Unit tests should not depend on actual model weights.

```python
# tests/conftest.py

import torch
import pytest

@pytest.fixture
def mock_model_weights(tmp_path):
    """Create mock model weights for unit testing."""
    weights_path = tmp_path / "mock_model.pth"

    # Create minimal valid state dict structure
    mock_state = {
        "fpn_model": {}  # Empty state dict for testing load logic
    }
    torch.save(mock_state, weights_path)
    return weights_path

@pytest.fixture
def mock_pose_files(tmp_path):
    """Create mock pose mean/stddev files."""
    pose_mean = np.zeros(6, dtype=np.float32)
    pose_stddev = np.ones(6, dtype=np.float32)

    mean_path = tmp_path / "pose_mean.npy"
    std_path = tmp_path / "pose_stddev.npy"

    np.save(mean_path, pose_mean)
    np.save(std_path, pose_stddev)

    return {"mean": mean_path, "std": std_path}
```

### 3.3 Detector Fixtures

```python
# tests/conftest.py

@pytest.fixture(scope="module")
def detector_cpu():
    """Initialize detector on CPU for testing.

    Note: This downloads real weights on first run (~85MB).
    Subsequent runs use cached weights.
    """
    from img2pose import Img2Pose
    return Img2Pose(device="cpu", score_threshold=0.5)

@pytest.fixture(scope="module")
def detector_gpu():
    """Initialize detector on GPU if available."""
    import torch
    from img2pose import Img2Pose

    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    return Img2Pose(device="cuda", score_threshold=0.5)
```

### 3.4 Reference Data for Equivalence Tests

```python
# tests/conftest.py

@pytest.fixture(scope="session")
def reference_results():
    """Load pre-computed reference results for equivalence testing.

    Generated once from original code and stored in tests/data/reference/.
    """
    ref_dir = TEST_DATA_DIR / "reference"
    return {
        "boxes": np.load(ref_dir / "reference_boxes.npy"),
        "poses": np.load(ref_dir / "reference_poses.npy"),
        "scores": np.load(ref_dir / "reference_scores.npy"),
    }
```

---

## 4. Nondeterminism Handling

### 4.1 GPU vs CPU Float Differences

GPU and CPU may produce slightly different floating-point results due to:
- Different CUDA/CPU implementations of mathematical operations
- Parallel reduction order differences
- Fused multiply-add operations

**Strategy:**

```python
# tests/test_equivalence_tolerance.py

import numpy as np

# Tolerance levels for different precision requirements
TOLERANCES = {
    "boxes": {"atol": 1e-3, "rtol": 1e-4},      # Pixel coordinates (loose)
    "poses": {"atol": 1e-4, "rtol": 1e-5},      # Radians/pixels (medium)
    "scores": {"atol": 1e-5, "rtol": 1e-6},     # Probabilities (tight)
}

def assert_results_equivalent(actual, expected, category="boxes"):
    """Assert results are equivalent within tolerance."""
    tol = TOLERANCES[category]
    np.testing.assert_allclose(
        actual, expected,
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"Results differ beyond {category} tolerance"
    )
```

### 4.2 NMS Order Sensitivity

Non-Maximum Suppression can produce different orderings for faces with similar scores.

**Strategy:**

```python
def sort_detections_canonical(detections):
    """Sort detections in a canonical order for comparison.

    Sort by: (score descending, box_x, box_y)
    """
    return sorted(
        detections,
        key=lambda d: (-d["score"], d["box"][0], d["box"][1])
    )

def test_detection_equivalence(detector, test_image, reference_results):
    """Compare detection results with canonical ordering."""
    results = detector.detect(test_image)

    actual_sorted = sort_detections_canonical(results)
    expected_sorted = sort_detections_canonical(reference_results)

    assert len(actual_sorted) == len(expected_sorted)

    for actual, expected in zip(actual_sorted, expected_sorted):
        assert_results_equivalent(actual["box"], expected["box"], "boxes")
        assert_results_equivalent(actual["pose"], expected["pose"], "poses")
```

### 4.3 Random Seed Control

Set deterministic mode for reproducible tests.

```python
# tests/conftest.py

@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic mode for reproducible results."""
    import torch
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 4.4 PyTorch Version Differences

Different PyTorch versions may have implementation differences.

**Strategy:**
- Pin minimum PyTorch version in `pyproject.toml`
- Test on multiple PyTorch versions in CI
- Document known version-specific behaviors

---

## 5. Test Data Management

### 5.1 Test Image Requirements

| Image | Size | Purpose | Source |
|-------|------|---------|--------|
| `face_test.jpg` | 640x480 | Single face detection | Synthetic or CC0 license |
| `multi_face.jpg` | 800x600 | Multiple face detection | Synthetic or CC0 license |
| `no_face.jpg` | 320x240 | Negative test case | Synthetic (solid color) |
| `large_image.jpg` | 4000x3000 | Memory/performance test | Synthetic |

### 5.2 Reference Data Generation

One-time script to generate reference results using original code.

```python
# scripts/generate_reference_data.py
"""Generate reference data for equivalence tests.

Run once from repo root after initial development:
    python scripts/generate_reference_data.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from PIL import Image
from torchvision import transforms

from img2pose import img2poseModel
from model_loader import load_model

def generate_reference():
    # Load original model
    pose_mean = np.load("models/WIDER_train_pose_mean_v1.npy")
    pose_stddev = np.load("models/WIDER_train_pose_stddev_v1.npy")
    threed_68_points = np.load("pose_references/reference_3d_68_points_trans.npy")

    model = img2poseModel(
        depth=18,
        min_size=640,
        max_size=1280,
        pose_mean=pose_mean,
        pose_stddev=pose_stddev,
        threed_68_points=threed_68_points,
    )
    load_model(model.fpn_model, "models/img2pose_v1.pth", model_only=True)
    model.evaluate()

    # Run on test image
    img = Image.open("tests/data/face_test.jpg")
    transform = transforms.ToTensor()

    predictions = model.predict([transform(img)])
    pred = predictions[0]

    # Save reference data
    ref_dir = Path("tests/data/reference")
    ref_dir.mkdir(exist_ok=True)

    np.save(ref_dir / "reference_boxes.npy", pred["boxes"].cpu().numpy())
    np.save(ref_dir / "reference_poses.npy", pred["dofs"].cpu().numpy())
    np.save(ref_dir / "reference_scores.npy", pred["scores"].cpu().numpy())

if __name__ == "__main__":
    generate_reference()
```

---

## 6. Local Execution Strategy

### 6.1 Test Environment Setup

```bash
# Create isolated test environment
python -m venv .venv-test
source .venv-test/bin/activate

# Install package with dev dependencies
pip install -e ".[dev]"

# Download model weights (one-time, cached in ~/.cache/img2pose/)
python -c "from img2pose import Img2Pose; Img2Pose()"
```

### 6.2 Running Tests Locally

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast, no model needed)
pytest tests/unit/ -v

# Run integration tests (requires model)
pytest tests/integration/ -v

# Run equivalence tests (requires model + reference data)
pytest tests/equivalence/ -v

# Run with coverage
pytest tests/ --cov=src/img2pose --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "test_detector" -v

# Run tests in parallel
pytest tests/ -n auto
```

### 6.3 Offline Testing After Model Download

Once models are cached, tests run fully offline.

```python
# tests/test_offline_mode.py

import os
import pytest

def test_offline_execution(detector_cpu, test_image_path, monkeypatch):
    """Verify tests work without network after model download."""
    # Simulate offline by blocking network
    def block_network(*args, **kwargs):
        raise RuntimeError("Network access blocked for offline test")

    monkeypatch.setattr("urllib.request.urlretrieve", block_network)

    # Should work with cached model
    results = detector_cpu.detect(test_image_path)
    assert isinstance(results, list)
```

### 6.4 CPU-Only Testing

```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" pytest tests/ -v

# Or in pytest.ini
# [pytest]
# env = CUDA_VISIBLE_DEVICES=
```

---

## 7. Test Directory Structure

```
tests/
|-- __init__.py
|-- conftest.py                    # Shared fixtures
|-- pytest.ini                     # pytest configuration
|
|-- unit/                          # Unit tests (no model needed)
|   |-- __init__.py
|   |-- test_detector_init.py
|   |-- test_detector_input.py
|   |-- test_detector_output.py
|   |-- test_weights.py
|   |-- test_visualization.py
|   |-- test_pose_operations.py
|   `-- test_exceptions.py
|
|-- integration/                   # Integration tests (model needed)
|   |-- __init__.py
|   |-- test_single_image.py
|   |-- test_batch_images.py
|   |-- test_device_selection.py
|   `-- test_visualization_flow.py
|
|-- equivalence/                   # Equivalence tests (model + reference)
|   |-- __init__.py
|   |-- test_equivalence_boxes.py
|   |-- test_equivalence_poses.py
|   |-- test_equivalence_scores.py
|   `-- test_equivalence_aflw2000.py
|
|-- negative/                      # Negative/error tests
|   |-- __init__.py
|   |-- test_invalid_input.py
|   |-- test_invalid_config.py
|   `-- test_download_errors.py
|
|-- schema/                        # Schema/contract tests
|   |-- __init__.py
|   |-- test_output_format.py
|   `-- test_config_format.py
|
`-- data/                          # Test data files
    |-- face_test.jpg              # Single face test image
    |-- multi_face.jpg             # Multiple faces test image
    |-- no_face.jpg                # No face test image
    `-- reference/                 # Pre-computed reference results
        |-- reference_boxes.npy
        |-- reference_poses.npy
        `-- reference_scores.npy
```

---

## 8. pytest Configuration

```ini
# tests/pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
markers =
    unit: Unit tests (no model required)
    integration: Integration tests (model required)
    equivalence: Numerical equivalence tests
    slow: Tests that take >10 seconds
    gpu: Tests requiring GPU
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
```

---

## 9. Test Examples

### 9.1 Unit Test Example

```python
# tests/unit/test_detector_input.py
"""Unit tests for input handling."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path

class TestInputConversion:
    """Test _to_tensor input conversion."""

    def test_string_path_input(self, test_image_path, mocker):
        """Test conversion from string path."""
        from img2pose.detector import Img2Pose

        # Mock model initialization
        mocker.patch.object(Img2Pose, '__init__', lambda self, **kw: None)
        detector = Img2Pose.__new__(Img2Pose)
        detector._device = "cpu"

        # Test conversion
        from torchvision import transforms
        detector._transform = transforms.ToTensor()

        tensor = detector._to_tensor(str(test_image_path))
        assert tensor.dim() == 3
        assert tensor.shape[0] == 3  # RGB channels

    def test_pil_image_input(self, small_test_image, mocker):
        """Test conversion from PIL Image."""
        # ... similar structure

    def test_numpy_array_input(self, test_image_numpy, mocker):
        """Test conversion from numpy array."""
        # ... similar structure

    def test_invalid_input_raises(self, mocker):
        """Test that invalid input raises InvalidInputError."""
        from img2pose.detector import Img2Pose
        from img2pose._exceptions import InvalidInputError

        mocker.patch.object(Img2Pose, '__init__', lambda self, **kw: None)
        detector = Img2Pose.__new__(Img2Pose)

        with pytest.raises(InvalidInputError):
            detector._to_tensor(12345)  # Invalid type
```

### 9.2 Integration Test Example

```python
# tests/integration/test_single_image.py
"""Integration tests for single image detection."""

import pytest
import numpy as np

@pytest.mark.integration
class TestSingleImageDetection:
    """Test single image detection pipeline."""

    def test_detect_returns_list(self, detector_cpu, test_image_with_face):
        """Test that detect() returns a list of face dictionaries."""
        results = detector_cpu.detect(test_image_with_face)

        assert isinstance(results, list)
        assert len(results) > 0  # Should find at least one face

    def test_result_contains_required_keys(self, detector_cpu, test_image_with_face):
        """Test that results contain all required keys."""
        results = detector_cpu.detect(test_image_with_face)

        required_keys = {"box", "score", "pose"}
        for face in results:
            assert required_keys.issubset(face.keys())

    def test_box_format(self, detector_cpu, test_image_with_face):
        """Test bounding box format: (left, top, right, bottom)."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            box = face["box"]
            assert isinstance(box, np.ndarray)
            assert box.shape == (4,)
            assert box.dtype == np.float32
            assert box[2] > box[0]  # right > left
            assert box[3] > box[1]  # bottom > top

    def test_pose_format(self, detector_cpu, test_image_with_face):
        """Test pose format: (rx, ry, rz, tx, ty, tz)."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            pose = face["pose"]
            assert isinstance(pose, np.ndarray)
            assert pose.shape == (6,)
            assert pose.dtype == np.float32

    def test_score_range(self, detector_cpu, test_image_with_face):
        """Test score is in valid range [0, 1]."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            assert 0.0 <= face["score"] <= 1.0
```

### 9.3 Equivalence Test Example

```python
# tests/equivalence/test_equivalence_aflw2000.py
"""Numerical equivalence test on AFLW2000-3D benchmark."""

import pytest
import numpy as np
from pathlib import Path

@pytest.mark.equivalence
@pytest.mark.slow
class TestAFLW2000Equivalence:
    """Test equivalence on AFLW2000-3D benchmark (2000 images)."""

    AFLW2000_PATH = Path("datasets/AFLW2000")  # Configure per environment
    TOLERANCE_BOXES = 1e-3
    TOLERANCE_POSES = 1e-4

    @pytest.fixture(scope="class")
    def aflw2000_images(self):
        """Load AFLW2000 image paths."""
        if not self.AFLW2000_PATH.exists():
            pytest.skip("AFLW2000 dataset not available")

        return sorted(self.AFLW2000_PATH.glob("*.jpg"))

    def test_equivalence_sample(self, detector_cpu, aflw2000_images, reference_results):
        """Test equivalence on first 10 images (smoke test)."""
        sample_images = aflw2000_images[:10]

        for img_path in sample_images:
            results = detector_cpu.detect(img_path)
            ref_key = img_path.stem

            if ref_key in reference_results:
                self._assert_results_match(results, reference_results[ref_key])

    @pytest.mark.slow
    def test_equivalence_full(self, detector_cpu, aflw2000_images, reference_results):
        """Test equivalence on full AFLW2000 dataset."""
        mismatches = []

        for img_path in aflw2000_images:
            results = detector_cpu.detect(img_path)
            ref_key = img_path.stem

            if ref_key in reference_results:
                try:
                    self._assert_results_match(results, reference_results[ref_key])
                except AssertionError as e:
                    mismatches.append((img_path.name, str(e)))

        if mismatches:
            pytest.fail(f"Equivalence failures on {len(mismatches)} images: {mismatches[:5]}")

    def _assert_results_match(self, actual, expected):
        """Assert detection results match within tolerance."""
        assert len(actual) == len(expected), "Different number of detections"

        for a, e in zip(actual, expected):
            np.testing.assert_allclose(
                a["box"], e["box"],
                atol=self.TOLERANCE_BOXES,
                err_msg="Box mismatch"
            )
            np.testing.assert_allclose(
                a["pose"], e["pose"],
                atol=self.TOLERANCE_POSES,
                err_msg="Pose mismatch"
            )
```

---

## 10. Continuous Integration Strategy

### 10.1 CI Pipeline Stages

```yaml
# .github/workflows/test.yml (conceptual)
stages:
  - lint      # Fast, no dependencies
  - unit      # Fast, no model
  - integration  # Slow, requires model download
  - equivalence  # Slow, requires model + reference data
```

### 10.2 Model Caching in CI

```yaml
# Cache model weights between CI runs
- uses: actions/cache@v3
  with:
    path: ~/.cache/img2pose
    key: img2pose-weights-v1
```

### 10.3 CPU-Only CI Matrix

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python: ["3.8", "3.9", "3.10", "3.11"]
```

---

## 11. Summary

| Test Category | Count | Model Required | Estimated Duration |
|---------------|-------|----------------|-------------------|
| Unit Tests | ~30 | No | 10-20 seconds |
| Integration Tests | ~15 | Yes | 30-60 seconds |
| Equivalence Tests | ~10 | Yes + Reference | 2-5 minutes |
| Negative Tests | ~15 | No | 10-20 seconds |
| Schema Tests | ~10 | No | 5-10 seconds |
| **Total** | **~80** | | **3-7 minutes** |

The test strategy ensures:
1. Fast feedback via unit tests (no model)
2. Confidence via integration tests
3. Correctness via equivalence tests
4. Robustness via negative tests
5. Stability via schema tests
