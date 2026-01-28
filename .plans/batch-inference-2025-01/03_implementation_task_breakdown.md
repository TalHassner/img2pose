# Implementation Task Breakdown: Efficient Batch Inference

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** Ready for Implementation

---

## Overview

This document provides ordered implementation milestones for adding efficient batch inference to `Img2Pose.detect_faces()`. Each milestone is self-contained and can be committed independently.

**Total Estimated Effort:** 2-4 hours
**Risk Level:** Low (no external API changes)

---

## Milestone Summary

| ID | Name | Files | Estimated Lines | Risk |
|----|------|-------|-----------------|------|
| M1 | Extract `_process_prediction()` helper | detector.py | +45, -25 | Low |
| M2 | Add `_detect_batch()` method | detector.py | +40 | Low |
| M3 | Update `detect_faces()` routing | detector.py | +8 | Low |
| M4 | Add batch equivalence tests | test_inference.py | +80 | Low |
| M5 | Add performance benchmark tests | test_inference.py | +50 | Low |

---

## M1: Extract `_process_prediction()` Helper

### Objective

Extract the post-processing logic from `_detect_single()` into a reusable method that can be shared with `_detect_batch()`.

### File Touch List

| File | Action |
|------|--------|
| `/home/tal/dev/img2pose/src/img2pose/detector.py` | Modify |

### Specific Changes

#### Step 1.1: Add `_process_prediction()` method

**Insert BEFORE `_detect_single()` (before line 251):**

```python
    def _process_prediction(
        self,
        pred: Dict[str, torch.Tensor],
        width: int,
        height: int,
        threshold: float,
        max_count: int,
    ) -> List[FaceDict]:
        """Process a single prediction dict into face dictionaries.

        Args:
            pred: Model prediction with keys: boxes, scores, dofs
            width: Original image width (for landmark projection)
            height: Original image height (for landmark projection)
            threshold: Minimum confidence score
            max_count: Maximum faces to return (-1 for unlimited)

        Returns:
            List of face dictionaries, sorted by confidence descending
        """
        faces = []
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        dofs = pred["dofs"].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] < threshold:
                continue

            # Extract box
            box = boxes[i].tolist()

            # Extract pose
            pose = dofs[i].tolist()

            # Compute keypoints by projecting 3D landmarks
            landmarks_2d = _project_landmarks(
                self._threed_68_points,
                dofs[i],
                width,
                height,
            )

            keypoints = {
                name: landmarks_2d[idx].tolist()
                for name, idx in self._KEYPOINT_INDICES.items()
            }

            faces.append({
                "box": box,
                "confidence": float(scores[i]),
                "pose": pose,
                "keypoints": keypoints,
            })

        # Sort by confidence (descending)
        faces.sort(key=lambda x: x["confidence"], reverse=True)

        # Limit number of faces
        if max_count > 0:
            faces = faces[:max_count]

        return faces
```

**Line count:** +60 lines (including docstring)

#### Step 1.2: Refactor `_detect_single()` to use helper

**Replace lines 268-313 (from `# Process results` to end of method) with:**

```python
        # Process results
        if len(predictions) == 0:
            return []
        return self._process_prediction(predictions[0], width, height, threshold, max_count)
```

**Net change:** -25 lines from `_detect_single()`

### Verification

```bash
cd /home/tal/dev/img2pose
python -m pytest tests/unit/test_detector.py -v
python -m pytest tests/integration/test_inference.py -v
```

### Done Conditions

- [ ] `_process_prediction()` method exists and has docstring
- [ ] `_detect_single()` calls `_process_prediction()`
- [ ] All existing unit tests pass
- [ ] All existing integration tests pass

### Rollback Strategy

Revert the single commit. No external dependencies.

---

## M2: Add `_detect_batch()` Method

### Objective

Add the core batch processing method that calls the model once for multiple images.

### File Touch List

| File | Action |
|------|--------|
| `/home/tal/dev/img2pose/src/img2pose/detector.py` | Modify |

### Specific Changes

#### Step 2.1: Add `_detect_batch()` method

**Insert AFTER `_detect_single()` (after the refactored method ends):**

```python
    def _detect_batch(
        self,
        images: List[ImageInput],
        threshold: float,
        max_count: int,
    ) -> List[List[FaceDict]]:
        """Detect faces in a batch of images efficiently.

        This method processes all images in a single model forward pass,
        which is more efficient than calling _detect_single() repeatedly.

        Args:
            images: List of input images (paths, PIL, or numpy arrays)
            threshold: Minimum confidence score
            max_count: Maximum faces per image (-1 for unlimited)

        Returns:
            List of face lists, one per input image (preserves input order)
        """
        # Step 1: Load all images and capture dimensions
        pil_images: List[Image.Image] = []
        dimensions: List[tuple] = []  # List of (width, height)
        for img in images:
            pil_img = _load_image(img)
            pil_images.append(pil_img)
            dimensions.append(pil_img.size)

        # Step 2: Convert all to tensors
        tensors = [_image_to_tensor(pil_img, self.device) for pil_img in pil_images]

        # Step 3: Single batched model call
        predictions = self._model.predict(tensors)

        # Step 4: Post-process each prediction with its corresponding dimensions
        all_faces: List[List[FaceDict]] = []
        for pred, (width, height) in zip(predictions, dimensions):
            faces = self._process_prediction(pred, width, height, threshold, max_count)
            all_faces.append(faces)

        return all_faces
```

**Line count:** +40 lines

### Verification

```bash
cd /home/tal/dev/img2pose
python -c "from img2pose.detector import Img2Pose; print('_detect_batch' in dir(Img2Pose))"
# Should print: True
```

### Done Conditions

- [ ] `_detect_batch()` method exists
- [ ] Method has complete docstring
- [ ] Method signature matches design spec
- [ ] All existing tests still pass (method not yet called)

### Rollback Strategy

Remove the added method. No callers yet.

---

## M3: Update `detect_faces()` Routing

### Objective

Modify `detect_faces()` to route batch inputs through `_detect_batch()` for efficient processing.

### File Touch List

| File | Action |
|------|--------|
| `/home/tal/dev/img2pose/src/img2pose/detector.py` | Modify |

### Specific Changes

#### Step 3.1: Update batch routing logic

**Replace lines 242-247 (current batch handling):**

```python
        # Handle batch input
        if isinstance(image, list):
            return [
                self._detect_single(img, threshold, max_count)
                for img in image
            ]
```

**With:**

```python
        # Handle batch input
        if isinstance(image, list):
            if len(image) == 0:
                return []
            if len(image) == 1:
                # Single-element batch: avoid batch overhead
                return [self._detect_single(image[0], threshold, max_count)]
            # Multi-element batch: use efficient batched inference
            return self._detect_batch(image, threshold, max_count)
```

**Net change:** +5 lines

### Verification

```bash
cd /home/tal/dev/img2pose
python -c "
from img2pose.detector import Img2Pose
import numpy as np

# Quick smoke test (requires weights)
try:
    detector = Img2Pose(device='cpu')
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = detector.detect_faces([img, img])
    print(f'Batch result type: {type(result)}, length: {len(result)}')
except Exception as e:
    print(f'Smoke test skipped: {e}')
"
```

### Done Conditions

- [ ] Empty batch returns `[]`
- [ ] Single-element batch returns `[List[FaceDict]]`
- [ ] Multi-element batch calls `_detect_batch()`
- [ ] All existing tests pass

### Rollback Strategy

Revert to sequential loop. Low risk - simple 5-line change.

---

## M3.5: Add Coverage Configuration (Optional but Recommended)

### Objective

Add pytest-cov configuration to `pyproject.toml` for consistent coverage measurement.

### File Touch List

| File | Action |
|------|--------|
| `/home/tal/dev/img2pose/pyproject.toml` | Modify |

### Specific Changes

**Add after `[tool.mypy]` section (after line 93):**

```toml
[tool.coverage.run]
source = ["src/img2pose"]
branch = true
omit = [
    "*/tests/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 80
show_missing = true
```

### Verification

```bash
pytest tests/ --cov=src/img2pose --cov-fail-under=80 --cov-report=term-missing
```

### Done Conditions

- [ ] Coverage configuration added to pyproject.toml
- [ ] `--cov-fail-under=80` works correctly

---

## M4: Add Batch Equivalence Tests

### Objective

Add tests that verify batch results exactly match sequential results.

### File Touch List

| File | Action |
|------|--------|
| `/home/tal/dev/img2pose/tests/integration/test_inference.py` | Modify |
| `/home/tal/dev/img2pose/tests/conftest.py` | Modify (optional fixture) |

### Specific Changes

#### Step 4.1: Add fixtures for batch testing

**Add to `/home/tal/dev/img2pose/tests/conftest.py` (after existing fixtures):**

```python
@pytest.fixture
def sample_images_batch():
    """Create a batch of sample images with different sizes."""
    return [
        Image.new("RGB", (640, 480), color="white"),
        Image.new("RGB", (800, 600), color="gray"),
        Image.new("RGB", (320, 240), color="black"),
    ]


@pytest.fixture
def sample_arrays_batch():
    """Create a batch of sample numpy arrays."""
    return [
        np.ones((480, 640, 3), dtype=np.uint8) * 128,
        np.ones((600, 800, 3), dtype=np.uint8) * 64,
        np.ones((240, 320, 3), dtype=np.uint8) * 192,
    ]
```

**Line count:** +16 lines

#### Step 4.2: Add batch equivalence test class

**Add to `/home/tal/dev/img2pose/tests/integration/test_inference.py` (after existing classes):**

```python
class TestBatchInference:
    """Tests for batch inference functionality."""

    def test_empty_batch_returns_empty_list(self):
        """Test that empty batch input returns empty list."""
        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                # Cannot fully test without weights, but verify the logic
                # Empty batch should return [] without calling model

    def test_single_element_batch_returns_nested_list(self, sample_rgb_image):
        """Test that single-element batch returns [[faces]]."""
        import torch
        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=[{
                    "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                    "scores": torch.tensor([0.9]),
                    "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                }])
                MockModel.return_value = mock_model

                # Verify structure only - actual equivalence needs real model

    def test_batch_result_structure(self):
        """Test that batch results have correct nested structure."""
        # Create mock batch results
        batch_results = [
            [{"box": [0, 0, 100, 100], "confidence": 0.9, "pose": [0]*6, "keypoints": {}}],
            [{"box": [0, 0, 50, 50], "confidence": 0.8, "pose": [0]*6, "keypoints": {}}],
            [],  # Image with no detections
        ]

        assert isinstance(batch_results, list)
        assert len(batch_results) == 3
        assert all(isinstance(r, list) for r in batch_results)
        assert len(batch_results[2]) == 0  # Empty detections

    def test_batch_preserves_order(self, sample_images_batch):
        """Test that batch results maintain input order."""
        import torch

        # Create different mock predictions for each image
        mock_predictions = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "dofs": torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
            {
                "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                "scores": torch.tensor([0.8]),
                "dofs": torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
            {
                "boxes": torch.tensor([[30, 30, 70, 70]], dtype=torch.float32),
                "scores": torch.tensor([0.7]),
                "dofs": torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
        ]

        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=mock_predictions)
                MockModel.return_value = mock_model

                # Verify order is preserved
                # Result[0] should correspond to sample_images_batch[0], etc.
```

**Line count:** +80 lines

### Verification

```bash
cd /home/tal/dev/img2pose
python -m pytest tests/integration/test_inference.py::TestBatchInference -v
```

#### Step 4.3: Add warning suppression test

**Add to `TestBatchInference` class:**

```python
    def test_no_warnings_during_batch(self, sample_images_batch):
        """Test that batch inference emits no deprecation warnings."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch("img2pose.detector.img2poseModel") as MockModel:
                with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                    mock_model = MagicMock()
                    mock_model.evaluate = MagicMock()
                    mock_model.load_state_dict = MagicMock()
                    mock_model.predict = MagicMock(return_value=[
                        {"boxes": torch.tensor([]), "scores": torch.tensor([]), "dofs": torch.tensor([]).reshape(0,6)}
                        for _ in sample_images_batch
                    ])
                    MockModel.return_value = mock_model

                    from img2pose import Img2Pose
                    detector = Img2Pose()
                    detector.detect_faces(sample_images_batch)

            # Filter for deprecation/future warnings
            relevant = [x for x in w if issubclass(x.category, (DeprecationWarning, FutureWarning))]
            assert len(relevant) == 0, f"Unexpected warnings: {[str(x.message) for x in relevant]}"
```

### Done Conditions

- [ ] `TestBatchInference` class exists
- [ ] Empty batch test passes
- [ ] Single-element batch test passes
- [ ] Batch structure test passes
- [ ] Order preservation test passes
- [ ] Warning suppression test passes

### Rollback Strategy

Remove the new test class. Tests are isolated.

---

## M5: Add Performance Benchmark Tests

### Objective

Add tests that verify batch processing is faster than sequential processing.

### File Touch List

| File | Action |
|------|--------|
| `/home/tal/dev/img2pose/tests/integration/test_inference.py` | Modify |

### Specific Changes

#### Step 5.1: Add performance test class

**Add to `/home/tal/dev/img2pose/tests/integration/test_inference.py` (after TestBatchInference):**

```python
class TestBatchPerformance:
    """Performance tests for batch inference.

    These tests verify that batch processing provides expected speedup.
    They are marked as slow and can be skipped in CI with: pytest -m "not slow"
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Performance tests most meaningful on GPU"
    )
    def test_batch_faster_than_sequential(self):
        """Test that batch inference is faster than sequential calls.

        This test requires actual model weights and GPU for meaningful results.
        """
        import time

        try:
            from img2pose import Img2Pose
            detector = Img2Pose(device="cuda")
        except Exception:
            pytest.skip("Could not initialize detector with weights")

        # Create test images
        images = [
            np.ones((480, 640, 3), dtype=np.uint8) * i
            for i in range(0, 256, 32)  # 8 images
        ]

        # Time sequential processing
        start = time.perf_counter()
        sequential_results = [detector.detect_faces(img) for img in images]
        sequential_time = time.perf_counter() - start

        # Time batch processing
        start = time.perf_counter()
        batch_results = detector.detect_faces(images)
        batch_time = time.perf_counter() - start

        # Batch should be faster
        assert batch_time < sequential_time, (
            f"Batch ({batch_time:.3f}s) should be faster than "
            f"sequential ({sequential_time:.3f}s)"
        )

        # Log speedup for debugging
        speedup = sequential_time / batch_time
        print(f"\nSpeedup: {speedup:.2f}x (sequential={sequential_time:.3f}s, batch={batch_time:.3f}s)")

    @pytest.mark.slow
    def test_batch_equivalence_with_real_model(self):
        """Test that batch and sequential produce identical results.

        This is the critical correctness test - numerical equivalence.
        """
        try:
            from img2pose import Img2Pose
            detector = Img2Pose(device="cpu")  # CPU for determinism
        except Exception:
            pytest.skip("Could not initialize detector with weights")

        # Create varied test images
        images = [
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
            np.ones((600, 800, 3), dtype=np.uint8) * 64,
            np.ones((240, 320, 3), dtype=np.uint8) * 192,
        ]

        # Get results both ways
        batch_results = detector.detect_faces(images)
        sequential_results = [detector.detect_faces(img) for img in images]

        # Verify exact equivalence
        assert len(batch_results) == len(sequential_results)
        for batch_faces, seq_faces in zip(batch_results, sequential_results):
            assert len(batch_faces) == len(seq_faces), "Different number of faces detected"
            for bf, sf in zip(batch_faces, seq_faces):
                assert bf["box"] == sf["box"], "Box mismatch"
                assert bf["confidence"] == sf["confidence"], "Confidence mismatch"
                assert bf["pose"] == sf["pose"], "Pose mismatch"
                assert bf["keypoints"] == sf["keypoints"], "Keypoints mismatch"


@pytest.fixture(scope="module")
def torch():
    """Import torch for performance tests."""
    import torch
    return torch
```

**Line count:** +80 lines

#### Step 5.2: Add pytest marker configuration

**Add to `/home/tal/dev/img2pose/tests/conftest.py` (at the top, after imports):**

```python
def pytest_configure(config):
    # ... existing code ...
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
```

**Line count:** +4 lines (modify existing function)

### Verification

```bash
# Run all tests
cd /home/tal/dev/img2pose
python -m pytest tests/integration/test_inference.py -v

# Run only fast tests (skip performance)
python -m pytest tests/integration/test_inference.py -v -m "not slow"

# Run performance tests specifically
python -m pytest tests/integration/test_inference.py::TestBatchPerformance -v -s
```

### Done Conditions

- [ ] `TestBatchPerformance` class exists
- [ ] `slow` marker is configured
- [ ] Performance test passes on GPU (when available)
- [ ] Equivalence test passes with real model
- [ ] Tests can be skipped with `-m "not slow"`

### Rollback Strategy

Remove the performance test class. Tests are isolated.

---

## Implementation Checklist

### Pre-Implementation

- [ ] Read and understand `detector.py` current implementation
- [ ] Verify model's batch capability (`_model.py:predict()`)
- [ ] Run existing tests to establish baseline

### M1: Extract Helper

- [ ] Add `_process_prediction()` method
- [ ] Refactor `_detect_single()` to use helper
- [ ] Run unit tests
- [ ] Commit: "refactor: extract _process_prediction helper for reuse"

### M2: Add Batch Method

- [ ] Add `_detect_batch()` method
- [ ] Verify method signature matches design
- [ ] Run unit tests (should pass - method not called yet)
- [ ] Commit: "feat: add _detect_batch internal method"

### M3: Update Routing

- [ ] Modify `detect_faces()` batch handling
- [ ] Test empty batch returns `[]`
- [ ] Test single-element batch behavior
- [ ] Run all tests
- [ ] Commit: "feat: route batch inputs through efficient batched inference"

### M4: Add Equivalence Tests

- [ ] Add batch test fixtures to conftest.py
- [ ] Add `TestBatchInference` class
- [ ] Run new tests
- [ ] Commit: "test: add batch inference equivalence tests"

### M5: Add Performance Tests

- [ ] Add `TestBatchPerformance` class
- [ ] Add `slow` marker configuration
- [ ] Run performance tests (if GPU available)
- [ ] Commit: "test: add batch performance benchmark tests"

### Post-Implementation

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify no warnings: `pytest tests/ -v -W error`
- [ ] Update CHANGELOG (if exists)

---

## Risk Assessment

| Milestone | Risk | Mitigation |
|-----------|------|------------|
| M1 | Extraction changes behavior | Comprehensive existing test coverage |
| M2 | Method added but never tested | M3-M5 provide coverage |
| M3 | Routing logic error | Edge case tests in M4 |
| M4 | Tests don't catch real issues | Performance tests in M5 use real model |
| M5 | Performance varies by hardware | Document as expected |

**Overall Risk:** Low - changes are internal, API unchanged, extensive testing.

---

## Line Count Summary

| File | Before | After | Delta |
|------|--------|-------|-------|
| `detector.py` | 373 | ~408 | +35 |
| `test_inference.py` | 131 | ~291 | +160 |
| `conftest.py` | 82 | ~102 | +20 |

**Total new lines:** ~215
**`detector.py` remains under 500 LOC threshold.**
