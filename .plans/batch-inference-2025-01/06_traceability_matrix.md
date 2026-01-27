# Traceability Matrix: Efficient Batch Inference for img2pose

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** Ready for Review

---

## 1. MUST Requirements Traceability

This matrix maps each MUST requirement from `00_scope_and_success_criteria.md` to implementation steps, tests, and documentation.

| Req ID | Requirement | Implementation Step | Test(s) | Docs |
|--------|-------------|---------------------|---------|------|
| M1 | `detect_faces(single_image)` returns identical results before/after | M1 (extract `_process_prediction`), M3 (routing unchanged for single) | `test_detector.py::TestImageLoading`, `test_detector.py::TestFaceOutputFormat`, `test_inference.py::TestImg2PoseInference` | None |
| M2 | `detect_faces([img1, img2, ...])` returns identical results to sequential | M2 (`_detect_batch`), M4 (equivalence tests) | `test_batch_sequential.py::test_batch_equals_sequential_real_model`, `test_batch_sequential.py::test_exact_equality` | None |
| M3 | API signature unchanged: `detect_faces(image, score_threshold=None, max_faces=None)` | No signature changes in M1-M3 | `test_detector.py::TestFaceOutputFormat::test_output_has_required_keys` (implicit), mypy type check | None |
| M4 | Return type unchanged: `List[FaceDict]` for single, `List[List[FaceDict]]` for batch | M3 (routing logic) | `test_inference.py::TestBatchInference::test_batch_returns_list_of_lists`, `test_inference.py::TestBatchInference::test_single_returns_list` | None |
| M5 | All existing tests pass without modification | All milestones maintain backward compatibility | All tests in `tests/unit/`, `tests/integration/`, `tests/equivalence/` | None |
| M6 | No user-visible warnings from torch.meshgrid or backbone_name | Handled in existing `_model.py` (out of scope for this plan) | `test_inference.py::TestBatchInference::test_no_warnings_during_batch` | None |
| M7 | Batch processing of N images completes in less time than N x single-image time | M2 (`_detect_batch` uses single model call) | `test_performance.py::TestBatchPerformance::test_batch_faster_than_sequential_cpu`, `test_performance.py::TestBatchPerformance::test_batch_faster_than_sequential_gpu` | None |

---

## 2. SHOULD Requirements Traceability

| Req ID | Requirement | Implementation Step | Test(s) | Docs |
|--------|-------------|---------------------|---------|------|
| S1 | Performance gain of at least 30% for batch size 4+ on GPU | M2 (`_detect_batch` batches tensors) | `test_performance.py::TestBatchPerformance::test_batch_speedup_increases_with_size` | None |
| S2 | No memory increase beyond expected batch overhead | M2 (no additional allocations) | Manual verification (not automated) | None |
| S3 | Works correctly on both CPU and CUDA devices | M2, M3 (device-agnostic implementation) | `test_performance.py::test_batch_faster_than_sequential_cpu`, `test_performance.py::test_batch_faster_than_sequential_gpu` | None |

---

## 3. Implementation Step to Test Mapping

### M1: Extract `_process_prediction()` Helper

| Step | Description | Unit Test | Integration Test | Equivalence Test |
|------|-------------|-----------|------------------|------------------|
| 1.1 | Add `_process_prediction()` method | `test_detector.py::TestProcessPrediction::test_returns_list_of_face_dicts` | - | - |
| 1.1 | Handle empty predictions | `test_detector.py::TestProcessPrediction::test_empty_prediction_returns_empty_list` | - | - |
| 1.1 | Filter by threshold | `test_detector.py::TestProcessPrediction::test_filters_by_threshold` | - | - |
| 1.1 | Sort by confidence | `test_detector.py::TestProcessPrediction::test_sorts_by_confidence_descending` | - | - |
| 1.1 | Limit by max_count | `test_detector.py::TestProcessPrediction::test_limits_by_max_count` | - | - |
| 1.2 | Refactor `_detect_single()` | All existing `test_detector.py` tests (regression) | `test_inference.py::TestImg2PoseInference` | `test_numerical.py` |

### M2: Add `_detect_batch()` Method

| Step | Description | Unit Test | Integration Test | Equivalence Test |
|------|-------------|-----------|------------------|------------------|
| 2.1 | Add `_detect_batch()` method | `test_detector.py::TestDetectBatch::test_calls_model_once` | - | - |
| 2.1 | Load all images | `test_detector.py::TestDetectBatch::test_loads_all_images` | - | - |
| 2.1 | Capture dimensions | `test_detector.py::TestDetectBatch::test_captures_dimensions` | - | - |
| 2.1 | Convert to tensors | `test_detector.py::TestDetectBatch::test_converts_to_tensors` | - | - |
| 2.1 | Process predictions | `test_detector.py::TestDetectBatch::test_processes_all_predictions` | - | - |
| 2.1 | Preserve order | `test_detector.py::TestDetectBatch::test_preserves_order` | `test_inference.py::TestBatchInference::test_batch_preserves_order` | `test_batch_sequential.py::test_order_preserved` |

### M3: Update `detect_faces()` Routing

| Step | Description | Unit Test | Integration Test | Equivalence Test |
|------|-------------|-----------|------------------|------------------|
| 3.1 | Handle empty list | `test_detector.py::TestBatchRouting::test_empty_list_returns_empty` | `test_inference.py::TestBatchInference::test_empty_batch` | - |
| 3.1 | Handle single-element list | `test_detector.py::TestBatchRouting::test_single_element_list` | `test_inference.py::TestBatchInference::test_single_element_batch` | - |
| 3.1 | Route multi-element to batch | `test_detector.py::TestBatchRouting::test_multi_element_uses_batch` | `test_inference.py::TestBatchInference::test_multi_element_batch` | - |
| 3.1 | Single image (not list) unchanged | `test_detector.py::TestBatchRouting::test_single_image_unchanged` | All existing single-image tests | - |

### M4: Add Batch Equivalence Tests

| Step | Description | Unit Test | Integration Test | Equivalence Test |
|------|-------------|-----------|------------------|------------------|
| 4.1 | Add batch fixtures | - | - | - |
| 4.2 | Add TestBatchInference | - | All `TestBatchInference` tests | - |
| 4.2 | Test empty batch | - | `test_empty_batch_returns_empty_list` | - |
| 4.2 | Test single-element batch | - | `test_single_element_batch_returns_nested_list` | - |
| 4.2 | Test result structure | - | `test_batch_result_structure` | - |
| 4.2 | Test order preservation | - | `test_batch_preserves_order` | - |

### M5: Add Performance Benchmark Tests

| Step | Description | Unit Test | Integration Test | Equivalence Test |
|------|-------------|-----------|------------------|------------------|
| 5.1 | Add TestBatchPerformance | - | `test_batch_faster_than_sequential` | - |
| 5.1 | Add equivalence with real model | - | - | `test_batch_equivalence_with_real_model` |
| 5.2 | Add slow marker | - | - | - |

---

## 4. Test Coverage by Feature

### 4.1 Batch Routing Feature

| Feature Aspect | Test Name | Test File | Status |
|----------------|-----------|-----------|--------|
| Empty list input | `test_empty_list_returns_empty` | `test_detector.py` | Planned |
| Single-element list | `test_single_element_list` | `test_detector.py` | Planned |
| Multi-element list routing | `test_multi_element_uses_batch` | `test_detector.py` | Planned |
| Single image (backward compat) | `test_single_image_unchanged` | `test_detector.py` | Planned |
| Batch returns nested list | `test_batch_returns_list_of_lists` | `test_inference.py` | Planned |

### 4.2 Batch Processing Feature

| Feature Aspect | Test Name | Test File | Status |
|----------------|-----------|-----------|--------|
| Single model call | `test_calls_model_once` | `test_detector.py` | Planned |
| Order preservation | `test_preserves_order` | `test_detector.py` | Planned |
| Mixed input types | `test_batch_handles_mixed_input_types` | `test_inference.py` | Planned |
| Per-image threshold | `test_batch_applies_threshold_per_image` | `test_inference.py` | Planned |
| Per-image max_faces | `test_batch_applies_max_faces_per_image` | `test_inference.py` | Planned |

### 4.3 Numerical Equivalence Feature

| Feature Aspect | Test Name | Test File | Status |
|----------------|-----------|-----------|--------|
| Box coordinates match | `test_box_coordinates_exact_match` | `test_batch_sequential.py` | Planned |
| Confidence scores match | `test_confidence_exact_match` | `test_batch_sequential.py` | Planned |
| Pose values match | `test_pose_exact_match` | `test_batch_sequential.py` | Planned |
| Keypoints match | `test_keypoints_exact_match` | `test_batch_sequential.py` | Planned |
| Full result match | `test_batch_equals_sequential_real_model` | `test_batch_sequential.py` | Planned |

### 4.4 Performance Feature

| Feature Aspect | Test Name | Test File | Status |
|----------------|-----------|-----------|--------|
| Faster on CPU | `test_batch_faster_than_sequential_cpu` | `test_performance.py` | Planned |
| Faster on GPU | `test_batch_faster_than_sequential_gpu` | `test_performance.py` | Planned |
| Speedup scales with size | `test_batch_speedup_increases_with_size` | `test_performance.py` | Planned |
| No single-image overhead | `test_single_image_no_overhead` | `test_performance.py` | Planned |

---

## 5. Error Handling Coverage

| Error Type | Input | Expected Exception | Test Name | Test File |
|------------|-------|-------------------|-----------|-----------|
| Invalid image in batch | `[valid, invalid]` | Propagated error | `test_batch_partial_failure_no_partial_results` | `test_detector.py` |
| Nonexistent path | `["missing.jpg"]` | `FileNotFoundError` | `test_batch_with_nonexistent_path_raises` | `test_detector.py` |
| Wrong dtype | `[float32_array]` | `ValueError` | `test_batch_with_wrong_dtype_raises` | `test_detector.py` |
| Wrong shape | `[grayscale_array]` | `ValueError` | `test_batch_with_wrong_shape_raises` | `test_detector.py` |
| Unsupported type | `[{"dict": "value"}]` | `TypeError` | `test_batch_with_unsupported_type_raises` | `test_detector.py` |

---

## 6. Backward Compatibility Matrix

| API Element | Pre-Change Behavior | Post-Change Behavior | Verified By |
|-------------|---------------------|----------------------|-------------|
| `detect_faces(img)` | Returns `List[FaceDict]` | Returns `List[FaceDict]` | Existing tests |
| `detect_faces([img])` | Returns `[List[FaceDict]]` | Returns `[List[FaceDict]]` | `test_single_element_batch` |
| `detect_faces([img1, img2])` | Returns `[[f1], [f2]]` (sequential) | Returns `[[f1], [f2]]` (batched) | `test_batch_equals_sequential` |
| `detect_faces([])` | Returns `[]` | Returns `[]` | `test_empty_batch` |
| `__call__(img)` | Same as `detect_faces(img)` | Same as `detect_faces(img)` | Existing tests |
| `score_threshold` param | Applied per face | Applied per face | Existing tests |
| `max_faces` param | Applied per image | Applied per image | Existing tests |
| FaceDict structure | Has all keys | Has all keys | `test_output_has_required_keys` |

---

## 7. Test Dependencies

### 7.1 Fixture Dependencies

| Test | Required Fixtures |
|------|-------------------|
| `test_batch_returns_list_of_lists` | `sample_images_batch`, `mock_batch_detector` |
| `test_batch_equals_sequential_mock` | `mock_model_predictions`, `sample_images_batch` |
| `test_batch_faster_than_sequential_gpu` | None (creates own images) |
| `test_process_prediction_shape` | `pose_reference_68` |

### 7.2 External Dependencies

| Test Category | Requires Model Weights | Requires GPU | Network Required |
|---------------|------------------------|--------------|------------------|
| Unit tests | No | No | No |
| Integration (mocked) | No | No | No |
| Equivalence (mocked) | No | No | No |
| Equivalence (real) | Yes | Optional | First run only |
| Performance | Yes | Recommended | First run only |

---

## 8. Documentation Updates

| Requirement | Documentation File | Update Type | Status |
|-------------|-------------------|-------------|--------|
| M1-M7 | `src/img2pose/README.md` | None required | N/A |
| Batch usage | `src/img2pose/README.md` | Optional: Add batch example | Optional |
| API reference | Docstrings | Already documented | Complete |

### Docstring Verification

The following docstrings must be accurate after implementation:

1. **`detect_faces()`** - Already documents batch behavior:
   ```python
   """
   Args:
       image: Input image(s). Can be:
           - File path (str or Path)
           - PIL Image
           - Numpy array (RGB uint8 HWC)
           - List of the above for batch processing

   Returns:
       List of face dictionaries for single image, or
       list of lists for batch input.
   """
   ```

2. **`_detect_batch()`** - New method, requires docstring:
   ```python
   """Detect faces in a batch of images efficiently.

   This method processes all images in a single model forward pass,
   which is more efficient than calling _detect_single() repeatedly.
   """
   ```

3. **`_process_prediction()`** - New method, requires docstring:
   ```python
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
   ```

---

## 9. Verification Summary

### 9.1 Requirement Verification Status

| Req ID | Requirement Summary | Planned Tests | Status |
|--------|---------------------|---------------|--------|
| M1 | Single image unchanged | 5+ existing, 1 new | Ready |
| M2 | Batch equals sequential | 7 new | Ready |
| M3 | API signature unchanged | Type check + implicit | Ready |
| M4 | Return types correct | 2 new | Ready |
| M5 | Existing tests pass | All existing | Ready |
| M6 | No warnings | 1 new | Ready |
| M7 | Batch faster | 4 new | Ready |
| S1 | 30% speedup on GPU | 1 new | Ready |
| S2 | No memory increase | Manual | N/A |
| S3 | CPU and CUDA work | 2 new | Ready |

### 9.2 Test Count Summary

| Category | Existing Tests | New Tests | Total |
|----------|----------------|-----------|-------|
| Unit | 18 | 16 | 34 |
| Integration | 11 | 10 | 21 |
| Equivalence | 8 | 7 | 15 |
| Performance | 0 | 4 | 4 |
| **Total** | **37** | **37** | **74** |

---

## 10. Sign-off Checklist

Before marking implementation complete, verify:

- [ ] All MUST requirements have passing tests
- [ ] All SHOULD requirements have passing tests (or documented exceptions)
- [ ] All new code has >= 80% coverage
- [ ] All existing tests pass without modification
- [ ] Type checking passes (`mypy`)
- [ ] Format checking passes (`black`, `isort`)
- [ ] No deprecation warnings emitted
- [ ] Performance benchmarks show expected improvement
- [ ] Docstrings are complete and accurate
