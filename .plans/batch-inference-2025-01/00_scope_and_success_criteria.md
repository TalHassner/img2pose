# Scope and Success Criteria: Efficient Batch Processing for img2pose

## Plan ID: batch-inference-2025-01

## Problem Restatement

The `Img2Pose.detect_faces()` method in `src/img2pose/detector.py` currently processes batch inputs using a sequential Python loop (lines 243-247):

```python
if isinstance(image, list):
    return [
        self._detect_single(img, threshold, max_count)
        for img in image
    ]
```

This approach:
1. Creates tensors one at a time
2. Calls the model's `predict()` method once per image
3. Does not leverage the underlying model's native batch processing capability

The underlying `FasterDoFRCNN` (via `GeneralizedRCNN`) **already supports batch inference** by accepting `List[Tensor]` and processing them in a single forward pass through the backbone, RPN, and ROI heads. The sequential loop in `detector.py` negates this benefit.

## Goals (From User Request)

1. **100% backward compatibility**: No changes to existing behavior, API signatures, or output formats
2. **Suppress library warnings**: All torch/torchvision deprecation warnings should be suppressed
3. **Performance improvement**: Batch processing N images must be faster than N sequential calls
4. **Numerical equivalence**: Batch processing must produce **exactly** the same results as sequential processing

## Definition of Success

### MUST Requirements

| ID | Requirement | Verification Method |
|----|-------------|---------------------|
| M1 | `detect_faces(single_image)` returns identical results before/after change | Unit test comparing outputs |
| M2 | `detect_faces([img1, img2, ...])` returns identical results to sequential calls | Equivalence test |
| M3 | API signature unchanged: `detect_faces(image, score_threshold=None, max_faces=None)` | Static analysis / type check |
| M4 | Return type unchanged: `List[FaceDict]` for single, `List[List[FaceDict]]` for batch | Type test |
| M5 | All existing tests pass without modification | pytest run |
| M6 | No user-visible warnings from torch.meshgrid or backbone_name | Integration test capturing stderr |
| M7 | Batch processing of N images completes in less time than N × single-image time | Performance benchmark |

### SHOULD Requirements

| ID | Requirement | Notes |
|----|-------------|-------|
| S1 | Performance gain of at least 30% for batch size 4+ on GPU | Benchmark target |
| S2 | No memory increase beyond expected batch overhead | Memory profiling |
| S3 | Works correctly on both CPU and CUDA devices | Device-specific tests |

### Explicit Backward Compatibility Definition

The following must remain **unchanged**:

1. **Public API**:
   - `Img2Pose.__init__(device, score_threshold, max_faces, model_path, min_size, max_size)`
   - `Img2Pose.detect_faces(image, score_threshold=None, max_faces=None)`
   - `Img2Pose.__call__(image, **kwargs)` as alias for `detect_faces`
   - `Img2Pose.visualize(...)` method

2. **Output Format**:
   - Single image: `List[FaceDict]`
   - Batch: `List[List[FaceDict]]`
   - FaceDict keys: `"box"`, `"confidence"`, `"pose"`, `"keypoints"`
   - Keypoints keys: `"left_eye"`, `"right_eye"`, `"nose"`, `"mouth_left"`, `"mouth_right"`

3. **Behavior**:
   - Order of faces sorted by confidence (descending)
   - `max_faces` limiting behavior
   - Score threshold filtering

4. **Type Hints**:
   - `ImageInput = Union[str, Path, Image.Image, np.ndarray]`
   - `FaceDict = Dict[str, Any]`

## Equivalence Definition

For any image set `[img1, img2, ..., imgN]`:
```
batch_results = detector.detect_faces([img1, img2, ..., imgN])
sequential_results = [detector.detect_faces(img) for img in [img1, img2, ..., imgN]]

assert batch_results == sequential_results  # Exact equality
```

Exact equality means:
- Same number of faces detected per image
- Identical bounding box coordinates (float equality)
- Identical confidence scores (float equality)
- Identical pose values (float equality)
- Identical keypoint coordinates (float equality)

### Numerical Equivalence Clarification

Exact equality is expected because:
1. Same normalization path (`_image_to_tensor()`)
2. Same model forward pass operations (just batched)
3. Same post-processing (`_process_prediction()`)
4. PyTorch in eval mode with deterministic operations

**Tolerance Fallback:** If exact equality fails on GPU due to floating-point operation ordering differences, use tolerance-based comparison:
- Box coordinates: `atol=1e-5`
- Confidence scores: `atol=1e-6`
- Pose values: `atol=1e-5`
- Keypoint coordinates: `atol=1e-4`

Tests should first attempt exact equality, then fall back to tolerance if needed on GPU.

## Constraints

1. **Offline operation**: No network calls during inference (weights pre-loaded)
2. **Hardware**: Must work on CPU-only systems and CUDA systems
3. **No external services**: All computation local
4. **Python 3.8+ compatibility**: Must work with all supported Python versions
5. **PyTorch 1.7+ compatibility**: Must work with all supported PyTorch versions
6. **Inference only**: Library is inference-only (no training support needed)

## Out of Scope

1. Training functionality
2. Distributed inference (multi-node)
3. ONNX export
4. Mixed precision inference (FP16)
5. Dynamic batching / batch accumulation
6. Changes to model architecture or weights

## Key Files Affected

| File | Change Type | Risk Level |
|------|-------------|------------|
| `src/img2pose/detector.py` | Major modification | Medium |
| `src/img2pose/_model.py` | Possible minor adjustment | Low |
| `tests/unit/test_detector.py` | Add batch tests | Low |
| `tests/integration/test_inference.py` | Add performance tests | Low |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Different tensor normalization per-image vs batch | M2 failure | Use identical normalization path |
| Image size variation in batch | Performance loss | May require padding strategy |
| Memory exhaustion on large batches | Runtime failure | Document memory requirements |
| Device mismatch in batch | Runtime error | Validate all tensors on same device |
