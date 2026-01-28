# Test Strategy: Efficient Batch Inference for img2pose

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** Ready for Review

---

## 1. Test Taxonomy

This test strategy covers five categories of tests to ensure correctness, equivalence, and performance of the batch inference implementation.

### 1.1 Unit Tests (Function-Level, Mocked Model)

**Purpose:** Verify individual functions and edge cases without requiring model weights.

| Test ID | Test Name | Description | Location |
|---------|-----------|-------------|----------|
| U1 | `test_empty_batch_returns_empty_list` | Empty list input returns empty list | `tests/unit/test_detector.py` |
| U2 | `test_single_element_batch_uses_single_path` | Single-element batch avoids batch overhead | `tests/unit/test_detector.py` |
| U3 | `test_process_prediction_shape` | `_process_prediction()` returns correct structure | `tests/unit/test_detector.py` |
| U4 | `test_process_prediction_threshold_filtering` | Faces below threshold are filtered | `tests/unit/test_detector.py` |
| U5 | `test_process_prediction_max_count_limiting` | `max_count` limits returned faces | `tests/unit/test_detector.py` |
| U6 | `test_process_prediction_confidence_sorting` | Faces sorted by confidence descending | `tests/unit/test_detector.py` |
| U7 | `test_detect_batch_calls_model_once` | Batch method calls model.predict() once | `tests/unit/test_detector.py` |
| U8 | `test_detect_batch_preserves_order` | Results maintain input image order | `tests/unit/test_detector.py` |

### 1.2 Integration Tests (Mocked Model, Verify API Contracts)

**Purpose:** Verify the public API contracts and interaction between components.

| Test ID | Test Name | Description | Location |
|---------|-----------|-------------|----------|
| I1 | `test_batch_returns_list_of_lists` | Batch input returns `List[List[FaceDict]]` | `tests/integration/test_inference.py` |
| I2 | `test_single_returns_list` | Single image returns `List[FaceDict]` | `tests/integration/test_inference.py` |
| I3 | `test_batch_result_structure` | Each nested list contains valid FaceDict | `tests/integration/test_inference.py` |
| I4 | `test_batch_handles_mixed_input_types` | Batch accepts paths, PIL, numpy mixed | `tests/integration/test_inference.py` |
| I5 | `test_batch_applies_threshold_per_image` | Score threshold applied to each image | `tests/integration/test_inference.py` |
| I6 | `test_batch_applies_max_faces_per_image` | max_faces limit applied per image | `tests/integration/test_inference.py` |
| I7 | `test_callable_interface_batch` | `__call__` method supports batch | `tests/integration/test_inference.py` |
| I8 | `test_no_warnings_during_batch` | No deprecation warnings emitted | `tests/integration/test_inference.py` |

### 1.3 Equivalence Tests (Batch vs Sequential)

**Purpose:** Verify batch results exactly match sequential results.

| Test ID | Test Name | Description | Location |
|---------|-----------|-------------|----------|
| E1 | `test_batch_equals_sequential_mock` | Mocked model produces identical results | `tests/equivalence/test_batch_sequential.py` |
| E2 | `test_batch_equals_sequential_real_model` | Real model produces identical results | `tests/equivalence/test_batch_sequential.py` |
| E3 | `test_batch_equals_sequential_varying_sizes` | Images of different sizes produce same results | `tests/equivalence/test_batch_sequential.py` |
| E4 | `test_box_coordinates_exact_match` | Bounding box values exactly match | `tests/equivalence/test_batch_sequential.py` |
| E5 | `test_confidence_exact_match` | Confidence scores exactly match | `tests/equivalence/test_batch_sequential.py` |
| E6 | `test_pose_exact_match` | 6DoF pose values exactly match | `tests/equivalence/test_batch_sequential.py` |
| E7 | `test_keypoints_exact_match` | Keypoint coordinates exactly match | `tests/equivalence/test_batch_sequential.py` |

### 1.4 Performance Tests (Benchmark Batch vs Sequential)

**Purpose:** Verify batch processing provides expected speedup.

| Test ID | Test Name | Description | Location |
|---------|-----------|-------------|----------|
| P1 | `test_batch_faster_than_sequential_cpu` | Batch is faster on CPU | `tests/integration/test_performance.py` |
| P2 | `test_batch_faster_than_sequential_gpu` | Batch is faster on GPU | `tests/integration/test_performance.py` |
| P3 | `test_batch_speedup_increases_with_size` | Speedup increases with batch size | `tests/integration/test_performance.py` |
| P4 | `test_single_image_no_overhead` | Single image has no batch overhead | `tests/integration/test_performance.py` |

### 1.5 Negative Tests (Error Handling, Invalid Inputs)

**Purpose:** Verify graceful error handling for invalid inputs.

| Test ID | Test Name | Description | Location |
|---------|-----------|-------------|----------|
| N1 | `test_batch_with_invalid_image_raises` | Invalid image in batch raises error | `tests/unit/test_detector.py` |
| N2 | `test_batch_with_nonexistent_path_raises` | Nonexistent file path raises FileNotFoundError | `tests/unit/test_detector.py` |
| N3 | `test_batch_with_wrong_dtype_raises` | Non-uint8 array raises ValueError | `tests/unit/test_detector.py` |
| N4 | `test_batch_with_wrong_shape_raises` | Non-RGB array raises ValueError | `tests/unit/test_detector.py` |
| N5 | `test_batch_with_unsupported_type_raises` | Unsupported type raises TypeError | `tests/unit/test_detector.py` |
| N6 | `test_batch_partial_failure_no_partial_results` | One invalid image fails entire batch | `tests/unit/test_detector.py` |

---

## 2. Fixtures Strategy

### 2.1 Existing Fixtures (from `/home/tal/dev/img2pose/tests/conftest.py`)

| Fixture | Description | Used By |
|---------|-------------|---------|
| `sample_rgb_image` | PIL Image 640x480 white | Unit, Integration |
| `sample_rgb_array` | numpy array 480x640x3 gray | Unit, Integration |
| `sample_face_dict` | Example FaceDict with all keys | Unit, Integration |
| `pose_reference_68` | 68-point 3D reference landmarks | Equivalence |
| `pose_reference_5` | 5-point 3D reference landmarks | Equivalence |

### 2.2 New Fixtures for Batch Testing

Add to `/home/tal/dev/img2pose/tests/conftest.py`:

```python
@pytest.fixture
def sample_images_batch():
    """Create a batch of sample PIL images with different sizes."""
    return [
        Image.new("RGB", (640, 480), color="white"),
        Image.new("RGB", (800, 600), color="gray"),
        Image.new("RGB", (320, 240), color="black"),
    ]


@pytest.fixture
def sample_arrays_batch():
    """Create a batch of sample numpy arrays with different sizes."""
    return [
        np.ones((480, 640, 3), dtype=np.uint8) * 128,
        np.ones((600, 800, 3), dtype=np.uint8) * 64,
        np.ones((240, 320, 3), dtype=np.uint8) * 192,
    ]


@pytest.fixture
def mock_model_predictions():
    """Create mock model predictions for batch testing."""
    import torch
    return [
        {
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "scores": torch.tensor([0.95]),
            "dofs": torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.0, 1.0]]),
        },
        {
            "boxes": torch.tensor([[50, 50, 150, 150], [200, 200, 300, 300]], dtype=torch.float32),
            "scores": torch.tensor([0.90, 0.85]),
            "dofs": torch.tensor([[0.2, 0.1, 0.0, 0.1, 0.1, 1.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        },
        {
            "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
            "scores": torch.tensor([]),
            "dofs": torch.tensor([], dtype=torch.float32).reshape(0, 6),
        },
    ]


@pytest.fixture
def mock_batch_detector(mock_model_predictions):
    """Create a mock Img2Pose detector with mocked model."""
    from unittest.mock import MagicMock, patch

    with patch("img2pose.detector.img2poseModel") as MockModel:
        with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
            mock_model = MagicMock()
            mock_model.evaluate = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.predict = MagicMock(return_value=mock_model_predictions)
            MockModel.return_value = mock_model

            # Return the mock model for test customization
            yield mock_model
```

### 2.3 Marker Configuration

Add to `/home/tal/dev/img2pose/tests/conftest.py` in `pytest_configure()`:

```python
def pytest_configure(config):
    # ... existing code ...
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_weights: marks tests that need model weights"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that need CUDA GPU"
    )
```

---

## 3. Local Execution Strategy

All tests can be run locally without external services or network access.

### 3.1 Running Tests Offline

```bash
# Run all tests (fast, mocked)
cd /home/tal/dev/img2pose
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run integration tests (excluding slow)
python -m pytest tests/integration/ -v -m "not slow"

# Run equivalence tests (mocked)
python -m pytest tests/equivalence/ -v -m "not requires_weights"
```

### 3.2 Running Tests with Model Weights

```bash
# Run all tests including those requiring weights
python -m pytest tests/ -v -m "not slow"

# Run performance tests (requires weights and GPU)
python -m pytest tests/integration/test_performance.py -v -s

# Run full equivalence tests (requires weights)
python -m pytest tests/equivalence/ -v
```

### 3.3 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `IMG2POSE_CACHE` | Cache directory for weights | `~/.cache/img2pose` |
| `CUDA_VISIBLE_DEVICES` | GPU selection for tests | (all available) |

### 3.4 Skipping Tests

Tests that require unavailable resources are automatically skipped:

```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
def test_gpu_specific():
    ...

@pytest.mark.skipif(
    not Path("~/.cache/img2pose/img2pose_v1.pth").expanduser().exists(),
    reason="Model weights not available"
)
def test_with_real_model():
    ...
```

---

## 4. Nondeterminism Controls

### 4.1 Model Inference Determinism

The img2pose model is **deterministic** in evaluation mode:

1. **No dropout**: Model uses `model.eval()` which disables dropout
2. **No random operations**: Inference path has no stochastic operations
3. **Deterministic NMS**: Non-Maximum Suppression is deterministic

**Verification:**

```python
def test_inference_determinism():
    """Verify repeated inference produces identical results."""
    detector = Img2Pose(device="cpu")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128

    results1 = detector.detect_faces(image)
    results2 = detector.detect_faces(image)

    assert results1 == results2  # Exact equality
```

### 4.2 Float Comparison Tolerance

**Batch vs Sequential comparison requires NO tolerance** - values should be exactly equal because:

1. Same tensor creation path (`_image_to_tensor()`)
2. Same model forward pass (batched vs single)
3. Same post-processing (`_process_prediction()`)

```python
def test_batch_sequential_exact_equality():
    """Verify exact float equality between batch and sequential."""
    detector = Img2Pose(device="cpu")
    images = [np.ones((480, 640, 3), dtype=np.uint8) * 128] * 3

    batch_results = detector.detect_faces(images)
    sequential_results = [detector.detect_faces(img) for img in images]

    # Exact equality - no tolerance needed
    assert batch_results == sequential_results
```

### 4.3 Device-Specific Behavior

| Aspect | CPU | CUDA |
|--------|-----|------|
| Numerical precision | FP32 | FP32 |
| Determinism | Yes | Yes (eval mode) |
| Float equality | Exact | Exact |

**Note:** If mixed precision (FP16) is ever added, tolerance may be needed. Current implementation uses FP32 only.

---

## 5. Test Matrix

### 5.1 Batch Routing Tests

| Test Case | Input | Expected Output | Test Type |
|-----------|-------|-----------------|-----------|
| Empty list | `[]` | `[]` | Unit |
| Single element | `[img]` | `[[faces]]` | Unit |
| Two elements | `[img1, img2]` | `[[faces1], [faces2]]` | Unit |
| Many elements | `[img1, ..., img8]` | `[[faces1], ..., [faces8]]` | Integration |
| Single image (not list) | `img` | `[faces]` | Unit |

### 5.2 Input Type Tests

| Test Case | Input | Expected Output | Test Type |
|-----------|-------|-----------------|-----------|
| PIL Image | `Image.new(...)` | `[faces]` | Unit |
| Numpy array | `np.array(...)` | `[faces]` | Unit |
| File path | `"image.jpg"` | `[faces]` | Integration |
| Mixed batch | `[pil, numpy, path]` | `[[f1], [f2], [f3]]` | Integration |
| Grayscale PIL | `Image.new("L", ...)` | `[faces]` (converted) | Unit |
| RGBA PIL | `Image.new("RGBA", ...)` | `[faces]` (converted) | Unit |

### 5.3 Error Handling Tests

| Test Case | Input | Expected Error | Test Type |
|-----------|-------|----------------|-----------|
| Non-uint8 array | `np.zeros(..., dtype=float32)` | `ValueError` | Negative |
| Wrong shape | `np.zeros((100, 100))` | `ValueError` | Negative |
| Unsupported type | `{"not": "image"}` | `TypeError` | Negative |
| Missing file | `"nonexistent.jpg"` | `FileNotFoundError` | Negative |
| Invalid in batch | `[valid, invalid, valid]` | Propagated error | Negative |

### 5.4 Equivalence Tests

| Test Case | Batch Input | Sequential Input | Expected | Test Type |
|-----------|-------------|------------------|----------|-----------|
| Same images | `[img1, img2, img3]` | `img1`, `img2`, `img3` | Exact match | Equivalence |
| Different sizes | `[small, medium, large]` | `small`, `medium`, `large` | Exact match | Equivalence |
| No faces | `[blank, blank]` | `blank`, `blank` | `[[], []]` | Equivalence |
| Many faces | `[crowded]` | `crowded` | Exact match | Equivalence |

### 5.5 Performance Tests

| Test Case | Batch Size | Expected Speedup | Test Type |
|-----------|------------|------------------|-----------|
| Batch of 2 | 2 | > 1.0x | Performance |
| Batch of 4 | 4 | > 1.5x | Performance |
| Batch of 8 | 8 | > 2.0x | Performance |
| Batch of 16 | 16 | > 3.0x | Performance |
| Single image | 1 | 1.0x (no overhead) | Performance |

---

## 6. Test File Organization

```
tests/
  conftest.py                      # Fixtures and configuration
  unit/
    __init__.py
    test_detector.py               # Unit tests for detector module
      - TestImageLoading          (existing)
      - TestImageToTensor         (existing)
      - TestLandmarkProjection    (existing)
      - TestFaceOutputFormat      (existing)
      - TestProcessPrediction     (NEW - for extracted helper)
      - TestDetectBatch           (NEW - batch method unit tests)
      - TestBatchRouting          (NEW - routing logic tests)
      - TestBatchNegativeCases    (NEW - error handling)
  integration/
    __init__.py
    test_inference.py              # Integration tests
      - TestImg2PoseInference     (existing)
      - TestVisualization         (existing)
      - TestWeightManagement      (existing)
      - TestBatchInference        (NEW - batch API tests)
    test_performance.py            (NEW - performance benchmarks)
      - TestBatchPerformance
  equivalence/
    __init__.py
    test_numerical.py              # Existing equivalence tests
    test_batch_sequential.py       (NEW - batch vs sequential)
      - TestBatchSequentialEquivalence
```

---

## 7. Mock Strategies

### 7.1 Mocking the Model

For unit tests that don't require real inference:

```python
from unittest.mock import MagicMock, patch

def test_with_mock_model():
    with patch("img2pose.detector.img2poseModel") as MockModel:
        with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
            mock_model = MagicMock()
            mock_model.evaluate = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.predict = MagicMock(return_value=[
                {"boxes": torch.tensor([[0, 0, 100, 100]]),
                 "scores": torch.tensor([0.9]),
                 "dofs": torch.tensor([[0, 0, 0, 0, 0, 1]])}
            ])
            MockModel.return_value = mock_model

            from img2pose import Img2Pose
            detector = Img2Pose()
            # ... test code
```

### 7.2 Mocking Individual Functions

For testing helper functions in isolation:

```python
def test_detect_batch_calls_model_once():
    """Verify _detect_batch makes single model call."""
    with patch.object(detector._model, 'predict') as mock_predict:
        mock_predict.return_value = [mock_pred1, mock_pred2, mock_pred3]

        detector._detect_batch([img1, img2, img3], 0.5, -1)

        # Should be called exactly once with all tensors
        assert mock_predict.call_count == 1
        call_args = mock_predict.call_args[0][0]
        assert len(call_args) == 3  # Three tensors passed
```

---

## 8. Continuous Integration Considerations

### 8.1 Test Categories for CI

| Category | Markers | Run in CI | Notes |
|----------|---------|-----------|-------|
| Unit | (none) | Always | Fast, no dependencies |
| Integration | `not slow` | Always | Mocked, fast |
| Equivalence (mocked) | `not requires_weights` | Always | No weights needed |
| Performance | `slow` | Optional | Requires weights + GPU |
| Full equivalence | `requires_weights` | Optional | Requires weights |

### 8.2 CI Commands

```yaml
# GitHub Actions example
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run fast tests
        run: |
          pip install -e ".[dev]"
          pytest tests/ -v -m "not slow and not requires_weights and not requires_gpu"
```

---

## 9. Appendix: Test Implementation Templates

### 9.1 Unit Test Template

```python
class TestProcessPrediction:
    """Unit tests for _process_prediction helper."""

    def test_returns_list_of_face_dicts(self):
        """Test return type is List[FaceDict]."""
        from img2pose.detector import Img2Pose, _process_prediction
        # ... implementation

    def test_filters_by_threshold(self):
        """Test faces below threshold are excluded."""
        # ... implementation
```

### 9.2 Integration Test Template

```python
class TestBatchInference:
    """Integration tests for batch inference."""

    def test_batch_returns_correct_structure(self, mock_batch_detector, sample_images_batch):
        """Test batch input returns List[List[FaceDict]]."""
        # ... implementation
```

### 9.3 Equivalence Test Template

```python
class TestBatchSequentialEquivalence:
    """Tests verifying batch == sequential results."""

    @pytest.mark.requires_weights
    def test_exact_equality(self):
        """Verify batch and sequential produce identical results."""
        detector = Img2Pose(device="cpu")
        images = [...]

        batch = detector.detect_faces(images)
        sequential = [detector.detect_faces(img) for img in images]

        assert batch == sequential
```

### 9.4 Performance Test Template

```python
class TestBatchPerformance:
    """Performance benchmarks for batch inference."""

    @pytest.mark.slow
    @pytest.mark.requires_weights
    @pytest.mark.requires_gpu
    def test_batch_speedup(self):
        """Verify batch provides expected speedup."""
        import time

        detector = Img2Pose(device="cuda")
        images = [...]

        # Warm up
        detector.detect_faces(images[0])

        # Time sequential
        start = time.perf_counter()
        for img in images:
            detector.detect_faces(img)
        seq_time = time.perf_counter() - start

        # Time batch
        start = time.perf_counter()
        detector.detect_faces(images)
        batch_time = time.perf_counter() - start

        assert batch_time < seq_time
```
