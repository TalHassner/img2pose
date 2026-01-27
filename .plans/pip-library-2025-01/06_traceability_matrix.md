# Traceability Matrix: img2pose pip-installable Library

**Version:** 1.0
**Date:** 2025-01-27
**Reference:** `00_scope_and_success_criteria.md`, `03_implementation_task_breakdown.md`, `04_test_strategy.md`

---

## API Reference

The canonical API is defined in the main plan. Tests should use:
- **Method:** `detect_faces()` (not `detect()`)
- **Callable:** `detector(image)` as alias for `detect_faces()`
- **Output keys:** `"confidence"` (not `"score"`), `"keypoints"` dict (not `"landmarks_5pt"`)

---

## 1. Requirements to Milestone Mapping

### 1.1 MUST Requirements

| Req ID | Requirement | Milestone | Test File | Test Function | Priority |
|--------|-------------|-----------|-----------|---------------|----------|
| M1 | Pip-installable package | M1: Scaffolding | `tests/integration/test_pip_install.py` | `test_pip_install_editable`, `test_pip_install_wheel` | P0 |
| M2 | Single image API | M4: Public API | `tests/integration/test_single_image.py` | `test_detect_single_image`, `test_detect_path_input`, `test_detect_pil_input`, `test_detect_numpy_input` | P0 |
| M3 | Batch image API | M4: Public API | `tests/integration/test_batch_images.py` | `test_detect_batch_images`, `test_batch_consistent_results`, `test_batch_empty_list` | P0 |
| M4 | GPU/CPU auto-selection | M4: Public API | `tests/integration/test_device_selection.py` | `test_auto_device_selection`, `test_explicit_cpu`, `test_explicit_gpu`, `test_cpu_fallback` | P0 |
| M5 | Numerical equivalence | M6: Testing | `tests/equivalence/test_equivalence_aflw2000.py` | `test_equivalence_boxes`, `test_equivalence_poses`, `test_equivalence_scores`, `test_equivalence_full_benchmark` | P0 |
| M6 | Bounding box output | M4: Public API | `tests/schema/test_output_format.py` | `test_box_format`, `test_box_dtype`, `test_box_values_valid` | P0 |
| M7 | 6DoF pose output | M4: Public API | `tests/schema/test_output_format.py` | `test_pose_format`, `test_pose_dtype`, `test_pose_contains_6dof` | P0 |
| M8 | Pre-trained model support | M3: Weights | `tests/integration/test_weight_download.py` | `test_auto_download`, `test_cached_weights`, `test_checksum_validation` | P0 |
| M9 | Clean production code | M2: Core Modules | `tests/unit/test_no_training_imports.py` | `test_no_training_imports_in_src`, `test_no_tensorboard_import`, `test_no_lmdb_import` | P1 |

### 1.2 SHOULD Requirements

| Req ID | Requirement | Milestone | Test File | Test Function | Priority |
|--------|-------------|-----------|-----------|---------------|----------|
| S1 | Visualization methods | M5: Visualization | `tests/integration/test_visualization.py` | `test_visualize_returns_image`, `test_visualize_draws_boxes`, `test_visualize_custom_colors` | P1 |
| S2 | Confidence threshold | M4: Public API | `tests/unit/test_detector_output.py` | `test_score_threshold_filtering`, `test_threshold_property`, `test_threshold_setter_validation` | P1 |
| S3 | Max faces limit | M4: Public API | `tests/unit/test_detector_output.py` | `test_max_faces_limiting`, `test_max_faces_none`, `test_max_faces_sorting` | P1 |

---

## 2. Detailed Test Specifications

### 2.1 M1: Pip-installable Package

**Requirement:** Users can install via `pip install .` or from PyPI

**Milestone:** M1: Package Scaffolding

**Test File:** `tests/integration/test_pip_install.py`

```python
# tests/integration/test_pip_install.py
"""Tests for pip installability."""

import subprocess
import sys
import pytest

@pytest.mark.integration
class TestPipInstall:
    """Test pip installation scenarios."""

    def test_pip_install_editable(self, tmp_path):
        """Test editable install: pip install -e ."""
        # Create fresh virtualenv
        venv_path = tmp_path / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        pip_path = venv_path / "bin" / "pip"

        # Install in editable mode
        result = subprocess.run(
            [str(pip_path), "install", "-e", "."],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Install failed: {result.stderr}"

        # Verify import works
        python_path = venv_path / "bin" / "python"
        result = subprocess.run(
            [str(python_path), "-c", "from img2pose import Img2Pose; print('OK')"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_pip_install_wheel(self, tmp_path):
        """Test wheel install: pip install ."""
        # Similar to above but non-editable
        pass

    def test_version_accessible(self):
        """Test that __version__ is accessible."""
        from img2pose import __version__
        assert isinstance(__version__, str)
        assert len(__version__.split(".")) >= 2  # At least major.minor
```

**Acceptance Criteria:**
- `pip install -e .` succeeds from repo root
- `python -c "from img2pose import Img2Pose"` succeeds
- `python -c "import img2pose; print(img2pose.__version__)"` prints version

---

### 2.2 M2: Single Image API

**Requirement:** Provide a function to process one image and return detections

**Milestone:** M4: Public API Implementation

**Test File:** `tests/integration/test_single_image.py`

```python
# tests/integration/test_single_image.py
"""Tests for single image detection API."""

import numpy as np
from PIL import Image
from pathlib import Path
import pytest

@pytest.mark.integration
class TestSingleImageAPI:
    """Test single image detection scenarios."""

    def test_detect_single_image(self, detector_cpu, test_image_with_face):
        """Test basic single image detection."""
        results = detector_cpu.detect(test_image_with_face)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_detect_path_input(self, detector_cpu, test_image_path):
        """Test detection with file path input."""
        # String path
        results = detector_cpu.detect(str(test_image_path))
        assert isinstance(results, list)

        # pathlib.Path
        results = detector_cpu.detect(test_image_path)
        assert isinstance(results, list)

    def test_detect_pil_input(self, detector_cpu, test_image_with_face):
        """Test detection with PIL Image input."""
        results = detector_cpu.detect(test_image_with_face)
        assert isinstance(results, list)

    def test_detect_numpy_input(self, detector_cpu, test_image_numpy):
        """Test detection with numpy array input (RGB uint8)."""
        results = detector_cpu.detect(test_image_numpy)
        assert isinstance(results, list)

    def test_single_vs_batch_equivalence(self, detector_cpu, test_image_with_face):
        """Test that single and batch[1] return equivalent results."""
        single_result = detector_cpu.detect(test_image_with_face)
        batch_result = detector_cpu.detect([test_image_with_face])

        assert len(single_result) == len(batch_result[0])
```

**Acceptance Criteria:**
- `detect(image)` returns `List[Dict]` for single image
- Accepts string path, Path, PIL Image, numpy array
- Returns empty list for images with no faces

---

### 2.3 M3: Batch Image API

**Requirement:** Provide a function to process multiple images efficiently

**Milestone:** M4: Public API Implementation

**Test File:** `tests/integration/test_batch_images.py`

```python
# tests/integration/test_batch_images.py
"""Tests for batch image detection API."""

import pytest

@pytest.mark.integration
class TestBatchImageAPI:
    """Test batch image detection scenarios."""

    def test_detect_batch_images(self, detector_cpu, batch_test_images):
        """Test basic batch detection."""
        results = detector_cpu.detect(batch_test_images)

        assert isinstance(results, list)
        assert len(results) == len(batch_test_images)
        for img_results in results:
            assert isinstance(img_results, list)

    def test_batch_consistent_results(self, detector_cpu, test_image_with_face):
        """Test that batch processing gives consistent results."""
        batch = [test_image_with_face, test_image_with_face]
        results = detector_cpu.detect(batch)

        # Same image should produce same results
        assert len(results[0]) == len(results[1])
        for r0, r1 in zip(results[0], results[1]):
            np.testing.assert_allclose(r0["box"], r1["box"])
            np.testing.assert_allclose(r0["pose"], r1["pose"])

    def test_batch_empty_list(self, detector_cpu):
        """Test batch detection with empty list."""
        results = detector_cpu.detect([])
        assert results == []

    def test_batch_mixed_input_types(self, detector_cpu, test_image_path, test_image_with_face):
        """Test batch with mixed input types."""
        batch = [str(test_image_path), test_image_with_face]
        results = detector_cpu.detect(batch)
        assert len(results) == 2
```

**Acceptance Criteria:**
- `detect([img1, img2, ...])` returns `List[List[Dict]]`
- Outer list length equals input list length
- Empty list input returns empty list

---

### 2.4 M4: GPU/CPU Auto-selection

**Requirement:** Automatically use GPU if available, fall back to CPU

**Milestone:** M4: Public API Implementation

**Test File:** `tests/integration/test_device_selection.py`

```python
# tests/integration/test_device_selection.py
"""Tests for device selection."""

import torch
import pytest

@pytest.mark.integration
class TestDeviceSelection:
    """Test device selection scenarios."""

    def test_auto_device_selection(self):
        """Test that device is auto-selected based on CUDA availability."""
        from img2pose import Img2Pose

        detector = Img2Pose()
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert detector.device.type == expected_device

    def test_explicit_cpu(self):
        """Test explicit CPU device selection."""
        from img2pose import Img2Pose

        detector = Img2Pose(device="cpu")
        assert detector.device.type == "cpu"

    @pytest.mark.gpu
    def test_explicit_gpu(self):
        """Test explicit GPU device selection."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from img2pose import Img2Pose

        detector = Img2Pose(device="cuda")
        assert detector.device.type == "cuda"

    def test_cpu_fallback(self, monkeypatch):
        """Test CPU fallback when CUDA unavailable."""
        from img2pose import Img2Pose

        # Mock CUDA as unavailable
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        detector = Img2Pose()
        assert detector.device.type == "cpu"

    def test_invalid_device_raises(self):
        """Test that invalid device string raises error."""
        from img2pose import Img2Pose

        with pytest.raises((ValueError, RuntimeError)):
            Img2Pose(device="invalid_device")
```

**Acceptance Criteria:**
- `Img2Pose()` auto-selects CUDA if available, else CPU
- `Img2Pose(device="cpu")` forces CPU
- `Img2Pose(device="cuda")` forces GPU (fails if unavailable)

---

### 2.5 M5: Numerical Equivalence

**Requirement:** The library must produce identical results to the current code on AFLW2000-3D benchmark

**Milestone:** M6: Testing and Validation

**Test File:** `tests/equivalence/test_equivalence_aflw2000.py`

```python
# tests/equivalence/test_equivalence_aflw2000.py
"""Numerical equivalence tests on AFLW2000-3D benchmark."""

import numpy as np
import pytest
from pathlib import Path

# Tolerance levels
TOLERANCE_BOXES = 1e-3   # Pixel coordinates
TOLERANCE_POSES = 1e-4   # Radians and translation
TOLERANCE_SCORES = 1e-5  # Probabilities

@pytest.mark.equivalence
class TestEquivalenceAFLW2000:
    """Test numerical equivalence on AFLW2000-3D benchmark."""

    @pytest.fixture(scope="class")
    def reference_results(self):
        """Load pre-computed reference results."""
        ref_dir = Path("tests/data/reference/aflw2000")
        if not ref_dir.exists():
            pytest.skip("Reference data not available")
        return np.load(ref_dir / "reference_results.npz", allow_pickle=True)

    def test_equivalence_boxes(self, detector_cpu, test_image_with_face, reference_results):
        """Test bounding box equivalence."""
        results = detector_cpu.detect(test_image_with_face)
        ref = reference_results["test_image"]

        assert len(results) == len(ref["boxes"])
        for actual, expected in zip(sorted(results, key=lambda x: x["score"], reverse=True),
                                     ref["boxes"]):
            np.testing.assert_allclose(
                actual["box"], expected,
                atol=TOLERANCE_BOXES,
                err_msg="Bounding box mismatch"
            )

    def test_equivalence_poses(self, detector_cpu, test_image_with_face, reference_results):
        """Test 6DoF pose equivalence."""
        results = detector_cpu.detect(test_image_with_face)
        ref = reference_results["test_image"]

        for actual, expected in zip(sorted(results, key=lambda x: x["score"], reverse=True),
                                     ref["poses"]):
            np.testing.assert_allclose(
                actual["pose"], expected,
                atol=TOLERANCE_POSES,
                err_msg="Pose mismatch"
            )

    def test_equivalence_scores(self, detector_cpu, test_image_with_face, reference_results):
        """Test detection score equivalence."""
        results = detector_cpu.detect(test_image_with_face)
        ref = reference_results["test_image"]

        actual_scores = sorted([r["score"] for r in results], reverse=True)
        expected_scores = sorted(ref["scores"], reverse=True)

        np.testing.assert_allclose(
            actual_scores, expected_scores,
            atol=TOLERANCE_SCORES,
            err_msg="Score mismatch"
        )

    @pytest.mark.slow
    def test_equivalence_full_benchmark(self, detector_cpu, reference_results):
        """Test equivalence on full AFLW2000-3D dataset (2000 images)."""
        aflw_path = Path("datasets/AFLW2000")
        if not aflw_path.exists():
            pytest.skip("AFLW2000 dataset not available")

        images = sorted(aflw_path.glob("*.jpg"))
        assert len(images) == 2000, f"Expected 2000 images, got {len(images)}"

        mismatches = []
        for img_path in images:
            results = detector_cpu.detect(img_path)
            ref_key = img_path.stem

            if ref_key in reference_results:
                try:
                    self._compare_results(results, reference_results[ref_key])
                except AssertionError as e:
                    mismatches.append((img_path.name, str(e)))

        assert len(mismatches) == 0, f"Mismatches on {len(mismatches)} images"

    def _compare_results(self, actual, expected):
        """Helper to compare detection results."""
        assert len(actual) == len(expected["boxes"])
        for a, e_box, e_pose, e_score in zip(
            sorted(actual, key=lambda x: x["score"], reverse=True),
            expected["boxes"], expected["poses"], expected["scores"]
        ):
            np.testing.assert_allclose(a["box"], e_box, atol=TOLERANCE_BOXES)
            np.testing.assert_allclose(a["pose"], e_pose, atol=TOLERANCE_POSES)
```

**Acceptance Criteria:**
- Bounding boxes match within 1e-3 (sub-pixel)
- 6DoF poses match within 1e-4
- Scores match within 1e-5
- 100% of AFLW2000-3D images produce equivalent results

---

### 2.6 M6: Bounding Box Output

**Requirement:** Return face bounding boxes in standard format

**Milestone:** M4: Public API Implementation

**Test File:** `tests/schema/test_output_format.py`

```python
# tests/schema/test_output_format.py
"""Tests for output format contracts."""

import numpy as np
import pytest

@pytest.mark.unit
class TestOutputFormat:
    """Test output format contracts."""

    def test_box_format(self, detector_cpu, test_image_with_face):
        """Test bounding box format: (left, top, right, bottom)."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            assert "box" in face
            box = face["box"]
            assert box.shape == (4,), f"Expected shape (4,), got {box.shape}"

    def test_box_dtype(self, detector_cpu, test_image_with_face):
        """Test bounding box dtype is float32."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            assert face["box"].dtype == np.float32

    def test_box_values_valid(self, detector_cpu, test_image_with_face):
        """Test bounding box values are valid coordinates."""
        results = detector_cpu.detect(test_image_with_face)
        img_width, img_height = test_image_with_face.size

        for face in results:
            box = face["box"]
            left, top, right, bottom = box

            # Basic sanity checks
            assert right > left, "right must be > left"
            assert bottom > top, "bottom must be > top"
            assert left >= 0, "left must be >= 0"
            assert top >= 0, "top must be >= 0"
            # Boxes may extend slightly beyond image due to projection
            assert right <= img_width * 1.5, "right too large"
            assert bottom <= img_height * 1.5, "bottom too large"
```

**Acceptance Criteria:**
- Box is `np.ndarray` with shape `(4,)`
- Box dtype is `float32`
- Box format is `(left, top, right, bottom)`
- `right > left` and `bottom > top`

---

### 2.7 M7: 6DoF Pose Output

**Requirement:** Return 6DoF pose estimates (rx, ry, rz, tx, ty, tz)

**Milestone:** M4: Public API Implementation

**Test File:** `tests/schema/test_output_format.py`

```python
# tests/schema/test_output_format.py (continued)

    def test_pose_format(self, detector_cpu, test_image_with_face):
        """Test pose format: (rx, ry, rz, tx, ty, tz)."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            assert "pose" in face
            pose = face["pose"]
            assert pose.shape == (6,), f"Expected shape (6,), got {pose.shape}"

    def test_pose_dtype(self, detector_cpu, test_image_with_face):
        """Test pose dtype is float32."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            assert face["pose"].dtype == np.float32

    def test_pose_contains_6dof(self, detector_cpu, test_image_with_face):
        """Test that pose contains valid 6DoF values."""
        results = detector_cpu.detect(test_image_with_face)

        for face in results:
            pose = face["pose"]
            rx, ry, rz, tx, ty, tz = pose

            # Rotation values should be in reasonable range (radians)
            assert -np.pi <= rx <= np.pi, f"rx out of range: {rx}"
            assert -np.pi <= ry <= np.pi, f"ry out of range: {ry}"
            assert -np.pi <= rz <= np.pi, f"rz out of range: {rz}"

            # Translation values should be finite
            assert np.isfinite(tx)
            assert np.isfinite(ty)
            assert np.isfinite(tz)
```

**Acceptance Criteria:**
- Pose is `np.ndarray` with shape `(6,)`
- Pose dtype is `float32`
- Pose format is `(rx, ry, rz, tx, ty, tz)` (radians, pixels)
- All values are finite

---

### 2.8 M8: Pre-trained Model Support

**Requirement:** Use the authors' pre-trained models without requiring users to download separately

**Milestone:** M3: Model Weight Management

**Test File:** `tests/integration/test_weight_download.py`

```python
# tests/integration/test_weight_download.py
"""Tests for weight download functionality."""

import pytest
from pathlib import Path
import shutil

@pytest.mark.integration
class TestWeightDownload:
    """Test weight download functionality."""

    def test_auto_download(self, tmp_path, monkeypatch):
        """Test that weights auto-download on first use."""
        from img2pose._weights import get_model_paths, CACHE_DIR

        # Use temporary cache directory
        test_cache = tmp_path / "cache"
        monkeypatch.setenv("IMG2POSE_WEIGHTS_DIR", str(test_cache))

        # First call should download
        paths = get_model_paths()

        assert paths["model_path"].exists()
        assert paths["pose_mean_path"].exists()
        assert paths["pose_stddev_path"].exists()

    def test_cached_weights(self, tmp_path, monkeypatch):
        """Test that cached weights are reused."""
        from img2pose._weights import get_model_paths

        test_cache = tmp_path / "cache"
        monkeypatch.setenv("IMG2POSE_WEIGHTS_DIR", str(test_cache))

        # First call downloads
        paths1 = get_model_paths()
        mtime1 = paths1["model_path"].stat().st_mtime

        # Second call should reuse cache
        paths2 = get_model_paths()
        mtime2 = paths2["model_path"].stat().st_mtime

        assert mtime1 == mtime2, "File was re-downloaded"

    def test_checksum_validation(self, tmp_path, monkeypatch):
        """Test that corrupted downloads are detected."""
        # This test requires implementing checksum validation
        pytest.skip("Checksum validation not yet implemented")

    def test_custom_model_path(self, mock_model_weights, mock_pose_files):
        """Test loading from custom model path."""
        from img2pose import Img2Pose

        # Should work with custom path
        detector = Img2Pose(model_path=mock_model_weights, device="cpu")
        assert detector is not None
```

**Acceptance Criteria:**
- `Img2Pose()` auto-downloads weights on first use
- Weights are cached in `~/.cache/img2pose/`
- Subsequent calls reuse cached weights
- Custom `model_path` works

---

### 2.9 M9: Clean Production Code

**Requirement:** Remove training-only code from the public API

**Milestone:** M2: Copy and Adapt Core Modules

**Test File:** `tests/unit/test_no_training_imports.py`

```python
# tests/unit/test_no_training_imports.py
"""Tests to ensure no training imports in src/."""

import ast
import pytest
from pathlib import Path

# Training-only modules that should NOT be imported in src/img2pose/
TRAINING_MODULES = [
    "tensorboard",
    "lmdb",
    "train_logger",
    "data_loader_lmdb",
    "data_loader_lmdb_augmenter",
    "early_stop",
]

@pytest.mark.unit
class TestNoTrainingImports:
    """Ensure src/img2pose/ has no training dependencies."""

    def test_no_training_imports_in_src(self):
        """Check that no training modules are imported in src/."""
        src_path = Path("src/img2pose")
        violations = []

        for py_file in src_path.rglob("*.py"):
            imports = self._get_imports(py_file)
            for imp in imports:
                for training_mod in TRAINING_MODULES:
                    if training_mod in imp:
                        violations.append(f"{py_file}: imports {imp}")

        assert len(violations) == 0, f"Training imports found:\n" + "\n".join(violations)

    def test_no_tensorboard_import(self):
        """Specifically check for tensorboard."""
        self._check_no_import("tensorboard")

    def test_no_lmdb_import(self):
        """Specifically check for lmdb."""
        self._check_no_import("lmdb")

    def _get_imports(self, py_file: Path) -> list:
        """Extract all imports from a Python file."""
        with open(py_file) as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def _check_no_import(self, module_name: str):
        """Check that a specific module is not imported."""
        src_path = Path("src/img2pose")
        for py_file in src_path.rglob("*.py"):
            imports = self._get_imports(py_file)
            assert module_name not in imports, f"{py_file} imports {module_name}"
```

**Acceptance Criteria:**
- No imports of `tensorboard`, `lmdb`, or other training modules in `src/img2pose/`
- Training code remains in repo root for research use
- `losses.py` import is conditional/guarded

---

### 2.10 S1: Visualization Methods

**Requirement:** Optional 2D visualization of bounding boxes and landmarks

**Milestone:** M5: Visualization Support

**Test File:** `tests/integration/test_visualization.py`

```python
# tests/integration/test_visualization.py
"""Tests for visualization functionality."""

import pytest
from PIL import Image

@pytest.mark.integration
class TestVisualization:
    """Test visualization methods."""

    def test_visualize_returns_image(self, detector_cpu, test_image_with_face):
        """Test that visualize() returns a PIL Image."""
        result = detector_cpu.visualize(test_image_with_face)
        assert isinstance(result, Image.Image)

    def test_visualize_draws_boxes(self, detector_cpu, test_image_with_face):
        """Test that boxes are drawn on the image."""
        original = test_image_with_face.copy()
        result = detector_cpu.visualize(test_image_with_face, draw_boxes=True)

        # Image should be different from original (boxes drawn)
        import numpy as np
        orig_arr = np.array(original)
        result_arr = np.array(result)
        assert not np.array_equal(orig_arr, result_arr), "No changes made to image"

    def test_visualize_custom_colors(self, detector_cpu, test_image_with_face):
        """Test custom box colors."""
        result = detector_cpu.visualize(
            test_image_with_face,
            box_color=(255, 0, 0)  # Red boxes
        )
        assert isinstance(result, Image.Image)

    def test_visualize_no_detections(self, detector_cpu, small_test_image):
        """Test visualization with no detections."""
        result = detector_cpu.visualize(small_test_image)
        assert isinstance(result, Image.Image)
```

**Acceptance Criteria:**
- `visualize()` returns `PIL.Image`
- Bounding boxes are drawn when `draw_boxes=True`
- Custom colors are respected
- Works with zero detections

---

### 2.11 S2/S3: Threshold and Max Faces

**Requirement:** Allow filtering detections by confidence score and limiting number of faces

**Milestone:** M4: Public API Implementation

**Test File:** `tests/unit/test_detector_output.py`

```python
# tests/unit/test_detector_output.py
"""Tests for detector output filtering."""

import pytest

@pytest.mark.unit
class TestFiltering:
    """Test detection filtering functionality."""

    def test_score_threshold_filtering(self, detector_cpu, test_image_with_face):
        """Test that low-score detections are filtered."""
        # Get all detections
        detector_cpu.score_threshold = 0.0
        all_results = detector_cpu.detect(test_image_with_face)

        # Filter with high threshold
        detector_cpu.score_threshold = 0.9
        filtered_results = detector_cpu.detect(test_image_with_face)

        assert len(filtered_results) <= len(all_results)
        for face in filtered_results:
            assert face["score"] >= 0.9

    def test_threshold_property(self, detector_cpu):
        """Test score_threshold property getter."""
        detector_cpu._score_threshold = 0.7
        assert detector_cpu.score_threshold == 0.7

    def test_threshold_setter_validation(self, detector_cpu):
        """Test score_threshold setter validates range."""
        with pytest.raises(ValueError):
            detector_cpu.score_threshold = 1.5

        with pytest.raises(ValueError):
            detector_cpu.score_threshold = -0.1

    def test_max_faces_limiting(self, detector_cpu, test_image_with_face):
        """Test that max_faces limits output."""
        from img2pose import Img2Pose

        detector = Img2Pose(device="cpu", max_faces=1)
        results = detector.detect(test_image_with_face)
        assert len(results) <= 1

    def test_max_faces_none(self, detector_cpu, test_image_with_face):
        """Test that max_faces=None returns all detections."""
        results = detector_cpu.detect(test_image_with_face)
        # Should return all detections above threshold
        assert isinstance(results, list)

    def test_max_faces_sorting(self, detector_cpu, test_image_with_face):
        """Test that results are sorted by score (highest first)."""
        from img2pose import Img2Pose

        detector = Img2Pose(device="cpu", max_faces=2, score_threshold=0.1)
        results = detector.detect(test_image_with_face)

        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]
```

**Acceptance Criteria:**
- `score_threshold` filters detections with score below threshold
- `max_faces` limits number of returned faces
- Results are sorted by score (highest first)
- Setter validates `0.0 <= threshold <= 1.0`

---

## 3. Summary Matrix

| Req ID | Test File | Test Function | Status |
|--------|-----------|---------------|--------|
| M1 | `test_pip_install.py` | `test_pip_install_editable` | Pending |
| M1 | `test_pip_install.py` | `test_pip_install_wheel` | Pending |
| M2 | `test_single_image.py` | `test_detect_single_image` | Pending |
| M2 | `test_single_image.py` | `test_detect_path_input` | Pending |
| M2 | `test_single_image.py` | `test_detect_pil_input` | Pending |
| M2 | `test_single_image.py` | `test_detect_numpy_input` | Pending |
| M3 | `test_batch_images.py` | `test_detect_batch_images` | Pending |
| M3 | `test_batch_images.py` | `test_batch_consistent_results` | Pending |
| M3 | `test_batch_images.py` | `test_batch_empty_list` | Pending |
| M4 | `test_device_selection.py` | `test_auto_device_selection` | Pending |
| M4 | `test_device_selection.py` | `test_explicit_cpu` | Pending |
| M4 | `test_device_selection.py` | `test_explicit_gpu` | Pending |
| M4 | `test_device_selection.py` | `test_cpu_fallback` | Pending |
| M5 | `test_equivalence_aflw2000.py` | `test_equivalence_boxes` | Pending |
| M5 | `test_equivalence_aflw2000.py` | `test_equivalence_poses` | Pending |
| M5 | `test_equivalence_aflw2000.py` | `test_equivalence_scores` | Pending |
| M5 | `test_equivalence_aflw2000.py` | `test_equivalence_full_benchmark` | Pending |
| M6 | `test_output_format.py` | `test_box_format` | Pending |
| M6 | `test_output_format.py` | `test_box_dtype` | Pending |
| M6 | `test_output_format.py` | `test_box_values_valid` | Pending |
| M7 | `test_output_format.py` | `test_pose_format` | Pending |
| M7 | `test_output_format.py` | `test_pose_dtype` | Pending |
| M7 | `test_output_format.py` | `test_pose_contains_6dof` | Pending |
| M8 | `test_weight_download.py` | `test_auto_download` | Pending |
| M8 | `test_weight_download.py` | `test_cached_weights` | Pending |
| M8 | `test_weight_download.py` | `test_checksum_validation` | Pending |
| M9 | `test_no_training_imports.py` | `test_no_training_imports_in_src` | Pending |
| M9 | `test_no_training_imports.py` | `test_no_tensorboard_import` | Pending |
| M9 | `test_no_training_imports.py` | `test_no_lmdb_import` | Pending |
| S1 | `test_visualization.py` | `test_visualize_returns_image` | Pending |
| S1 | `test_visualization.py` | `test_visualize_draws_boxes` | Pending |
| S1 | `test_visualization.py` | `test_visualize_custom_colors` | Pending |
| S2 | `test_detector_output.py` | `test_score_threshold_filtering` | Pending |
| S2 | `test_detector_output.py` | `test_threshold_property` | Pending |
| S2 | `test_detector_output.py` | `test_threshold_setter_validation` | Pending |
| S3 | `test_detector_output.py` | `test_max_faces_limiting` | Pending |
| S3 | `test_detector_output.py` | `test_max_faces_none` | Pending |
| S3 | `test_detector_output.py` | `test_max_faces_sorting` | Pending |

---

## 4. Test Execution Order

### 4.1 Milestone-Based Execution

| Milestone | Tests to Run | Gate |
|-----------|--------------|------|
| M1: Scaffolding | `test_pip_install.py` | Must pass before M2 |
| M2: Core Modules | `test_no_training_imports.py` | Must pass before M4 |
| M3: Weights | `test_weight_download.py` | Must pass before M4 |
| M4: Public API | `test_single_image.py`, `test_batch_images.py`, `test_device_selection.py`, `test_output_format.py`, `test_detector_output.py` | Must pass before M5 |
| M5: Visualization | `test_visualization.py` | Must pass before M6 |
| M6: Testing | `test_equivalence_aflw2000.py` | Must pass before M7 |
| M7: Release | All tests | Final gate |

### 4.2 CI Pipeline Order

```yaml
jobs:
  unit-tests:  # Fast, no model
    - test_no_training_imports.py
    - test_detector_output.py (mocked)

  integration-tests:  # Model required
    - test_pip_install.py
    - test_weight_download.py
    - test_single_image.py
    - test_batch_images.py
    - test_device_selection.py
    - test_output_format.py
    - test_visualization.py

  equivalence-tests:  # Slow, optional
    - test_equivalence_aflw2000.py
```

---

## 5. Verification Checklist

Before release, verify all requirements are covered:

- [ ] M1: `pip install` works (2 tests)
- [ ] M2: Single image API works with all input types (4 tests)
- [ ] M3: Batch API works correctly (3 tests)
- [ ] M4: Device selection works (4 tests)
- [ ] M5: Numerical equivalence verified (4 tests)
- [ ] M6: Box output format correct (3 tests)
- [ ] M7: Pose output format correct (3 tests)
- [ ] M8: Weight download works (3 tests)
- [ ] M9: No training imports (3 tests)
- [ ] S1: Visualization works (3 tests)
- [ ] S2: Score threshold filtering works (3 tests)
- [ ] S3: Max faces limiting works (3 tests)

**Total: 38 tests covering all requirements**
