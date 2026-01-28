# Design Specification: Efficient Batch Inference for img2pose

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** Draft

---

## 1. Proposed Architecture

### 1.1 Current Flow (Sequential)

```
detect_faces([img1, img2, img3])
    |
    +---> _detect_single(img1)  ---> model.predict([tensor1]) ---> post-process ---> faces1
    +---> _detect_single(img2)  ---> model.predict([tensor2]) ---> post-process ---> faces2
    +---> _detect_single(img3)  ---> model.predict([tensor3]) ---> post-process ---> faces3
    |
    v
[faces1, faces2, faces3]
```

**Problem:** 3 images = 3 separate model forward passes = 3x GPU kernel launches.

### 1.2 Proposed Flow (Batched)

```
detect_faces([img1, img2, img3])
    |
    +---> _detect_batch([img1, img2, img3])
    |       |
    |       +---> Load images, capture dimensions: [(w1,h1), (w2,h2), (w3,h3)]
    |       +---> Convert all to tensors: [tensor1, tensor2, tensor3]
    |       +---> model.predict([tensor1, tensor2, tensor3])  # SINGLE forward pass
    |       +---> Post-process each with its dimensions
    |       |
    |       v
    |     [faces1, faces2, faces3]
    |
    v
[faces1, faces2, faces3]
```

**Benefit:** 3 images = 1 model forward pass = efficient GPU utilization.

### 1.3 Routing Logic

```python
def detect_faces(self, image, ...):
    if isinstance(image, list):
        if len(image) == 0:
            return []
        if len(image) == 1:
            return [self._detect_single(image[0], ...)]  # No batch overhead
        return self._detect_batch(image, ...)  # Batch processing
    return self._detect_single(image, ...)  # Single image
```

---

## 2. Module Boundaries

All changes are contained within a single file:

| File | Role | Changes |
|------|------|---------|
| `/home/tal/dev/img2pose/src/img2pose/detector.py` | Public API | Add `_detect_batch()`, modify `detect_faces()` routing |

**No new files required.** The existing module structure is preserved.

### 2.1 Dependency Direction

```
detect_faces() --> _detect_batch() --> _model.predict()
                                   --> _load_image()
                                   --> _image_to_tensor()
                                   --> _project_landmarks()

detect_faces() --> _detect_single() --> (unchanged, preserved for single images)
```

---

## 3. Public Interfaces (Unchanged)

### 3.1 `Img2Pose.__init__()`

```python
def __init__(
    self,
    device: str = "auto",
    score_threshold: float = 0.5,
    max_faces: int = -1,
    model_path: Optional[str] = None,
    min_size: int = 640,
    max_size: int = 1400,
):
```

**No changes.** Signature, defaults, and behavior remain identical.

### 3.2 `Img2Pose.detect_faces()`

```python
def detect_faces(
    self,
    image: Union[ImageInput, List[ImageInput]],
    score_threshold: Optional[float] = None,
    max_faces: Optional[int] = None,
) -> Union[List[FaceDict], List[List[FaceDict]]]:
```

**No changes to signature or return type.**

| Input Type | Return Type | Behavior |
|------------|-------------|----------|
| Single `ImageInput` | `List[FaceDict]` | Single image processing |
| `List[ImageInput]` | `List[List[FaceDict]]` | Batch processing (now efficient) |

### 3.3 `Img2Pose.__call__()`

```python
def __call__(
    self,
    image: Union[ImageInput, List[ImageInput]],
    **kwargs,
) -> Union[List[FaceDict], List[List[FaceDict]]]:
```

**No changes.** Remains an alias for `detect_faces()`.

### 3.4 Type Aliases (Unchanged)

```python
# Line 20-21 of detector.py
ImageInput = Union[str, Path, Image.Image, np.ndarray]
FaceDict = Dict[str, Any]
```

### 3.5 FaceDict Structure (Unchanged)

```python
{
    "box": [x1, y1, x2, y2],           # List[float], xyxy format
    "confidence": float,                # 0.0 to 1.0
    "pose": [rx, ry, rz, tx, ty, tz],  # List[float], 6DoF
    "keypoints": {
        "left_eye": [x, y],
        "right_eye": [x, y],
        "nose": [x, y],
        "mouth_left": [x, y],
        "mouth_right": [x, y],
    }
}
```

---

## 4. Internal Changes

### 4.1 New Method: `_detect_batch()`

**Location:** `detector.py`, insert after `_detect_single()` (after line 313)

**Signature:**
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
        List of face lists, one per input image (same order as input)
    """
```

**Implementation Approach:**

```python
def _detect_batch(
    self,
    images: List[ImageInput],
    threshold: float,
    max_count: int,
) -> List[List[FaceDict]]:
    # Step 1: Load all images and capture dimensions
    pil_images = []
    dimensions = []  # List of (width, height) tuples
    for img in images:
        pil_img = _load_image(img)
        pil_images.append(pil_img)
        dimensions.append(pil_img.size)  # (width, height)

    # Step 2: Convert all to tensors
    tensors = [_image_to_tensor(pil_img, self.device) for pil_img in pil_images]

    # Step 3: Single batched model call
    predictions = self._model.predict(tensors)

    # Step 4: Post-process each prediction with its dimensions
    all_faces = []
    for pred, (width, height) in zip(predictions, dimensions):
        faces = self._process_prediction(pred, width, height, threshold, max_count)
        all_faces.append(faces)

    return all_faces
```

### 4.2 New Method: `_process_prediction()` (Optional Refactor)

To avoid code duplication between `_detect_single()` and `_detect_batch()`, extract the post-processing logic into a shared helper.

**Location:** `detector.py`, insert before `_detect_single()` (around line 251)

**Signature:**
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
        List of face dictionaries, sorted by confidence
    """
```

**Implementation:** Extract lines 269-313 from `_detect_single()`.

### 4.3 Modified Method: `detect_faces()`

**Current Code (lines 242-249):**
```python
# Handle batch input
if isinstance(image, list):
    return [
        self._detect_single(img, threshold, max_count)
        for img in image
    ]

return self._detect_single(image, threshold, max_count)
```

**New Code:**
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

return self._detect_single(image, threshold, max_count)
```

### 4.4 Modified Method: `_detect_single()` (Minimal)

**Option A (Recommended):** Refactor to use `_process_prediction()`:

```python
def _detect_single(
    self,
    image: ImageInput,
    threshold: float,
    max_count: int,
) -> List[FaceDict]:
    """Detect faces in a single image."""
    # Load and convert image
    pil_image = _load_image(image)
    width, height = pil_image.size

    # Convert to tensor
    img_tensor = _image_to_tensor(pil_image, self.device)

    # Run inference
    predictions = self._model.predict([img_tensor])

    # Process results
    if len(predictions) == 0:
        return []
    return self._process_prediction(predictions[0], width, height, threshold, max_count)
```

**Option B (Minimal Change):** Keep `_detect_single()` unchanged, accept minor code duplication.

**Recommendation:** Option A reduces maintenance burden and ensures consistent behavior.

---

## 5. Data Flow Diagram

### 5.1 Batch Processing Path

```
Input: [img1, img2, img3] (mixed types: path, PIL, numpy)
           |
           v
    +------+------+
    | _load_image | (per image)
    +------+------+
           |
           v
pil_images: [PIL1, PIL2, PIL3]
dimensions: [(640,480), (800,600), (1024,768)]
           |
           v
    +------+------+
    |_image_to_tensor| (per image)
    +------+------+
           |
           v
tensors: [Tensor(3,480,640), Tensor(3,600,800), Tensor(3,768,1024)]
           |
           v
    +------+------+
    | model.predict | (SINGLE batched call)
    +------+------+
           |
           v
predictions: [pred1, pred2, pred3]  # List of dicts
           |
           +---> pred1 + (640,480) ---> _process_prediction ---> faces1
           +---> pred2 + (800,600) ---> _process_prediction ---> faces2
           +---> pred3 + (1024,768) ---> _process_prediction ---> faces3
           |
           v
Output: [faces1, faces2, faces3]
```

### 5.2 Dimension Tracking

The key insight is that `predictions` maintains the same order as the input `tensors`:

```python
# This relationship is guaranteed by GeneralizedRCNN.forward()
predictions[i] corresponds to tensors[i] corresponds to dimensions[i]
```

Evidence from `/home/tal/dev/img2pose/src/img2pose/_rcnn.py` lines 69-87:

```python
original_image_sizes: List[Tuple[int, int]] = []
for img in images:
    ...
    original_image_sizes.append((img.shape[-2], img.shape[-1]))
...
# Post-processing uses original_image_sizes in same order
detections = self.postprocess(detections, images.image_sizes, original_image_sizes)
```

---

## 6. Backward Compatibility Rules

### 6.1 Behavioral Guarantees

| Invariant | Verification |
|-----------|--------------|
| Single image returns `List[FaceDict]` | Type check in tests |
| Batch returns `List[List[FaceDict]]` | Type check in tests |
| Empty batch returns `[]` | Unit test |
| Single-element batch returns `[List[FaceDict]]` | Unit test |
| Results sorted by confidence (descending) | Equivalence test |
| `max_faces` limit applied per-image | Unit test |
| `score_threshold` filtering applied per-image | Unit test |

### 6.2 Numerical Equivalence

For any input list `[img1, ..., imgN]`:

```python
batch_result = detector.detect_faces([img1, ..., imgN])
sequential_result = [detector.detect_faces(img) for img in [img1, ..., imgN]]

assert batch_result == sequential_result  # Exact float equality
```

This is achievable because:
1. Same normalization path (`_image_to_tensor()`)
2. Same model forward pass (just batched)
3. Same post-processing (`_process_prediction()`)
4. Deterministic operations (no random components in inference)

### 6.3 API Stability

| API Element | Change Allowed |
|-------------|----------------|
| `detect_faces()` signature | No |
| `detect_faces()` return type | No |
| `__call__()` behavior | No |
| `FaceDict` keys | No |
| `ImageInput` type alias | No |
| `_detect_single()` existence | Yes (internal) |
| `_detect_batch()` addition | Yes (internal) |

---

## 7. Bloat Control

### 7.1 Current File Size

`detector.py` is currently **373 lines** (including docstrings and blank lines).

### 7.2 Estimated Additions

| Component | Lines |
|-----------|-------|
| `_process_prediction()` method | ~45 lines (extracted, not new) |
| `_detect_batch()` method | ~35 lines |
| Modified `detect_faces()` routing | +6 lines |
| Refactored `_detect_single()` | -25 lines (extraction) |

**Net Change:** +16 lines

**Estimated Final Size:** ~389 lines (well under 500 LOC threshold)

### 7.3 Refactor Triggers

If `detector.py` exceeds 500 LOC in the future:

1. Extract `_process_prediction()` to `_postprocess.py`
2. Extract image loading utilities to `_image_utils.py`
3. Keep public `Img2Pose` class as thin facade

### 7.4 No New Files Policy

This implementation adds **zero new files**. All changes are contained within the existing module structure.

---

## 8. Error Handling

### 8.1 Empty Batch

```python
detect_faces([])  # Returns: []
```

### 8.2 Mixed Input Types

```python
detect_faces(["path.jpg", pil_image, np_array])  # Works: all converted via _load_image()
```

### 8.3 Invalid Images in Batch

If any image in the batch fails to load, the error propagates immediately (fail-fast):

```python
detect_faces(["valid.jpg", "nonexistent.jpg", "valid2.jpg"])
# Raises: FileNotFoundError from _load_image()
```

This matches current behavior - no partial results.

### 8.4 Device Consistency

All tensors are placed on `self.device` by `_image_to_tensor()`. No additional validation needed.

### 8.5 Warning Suppression Verification

Before M3 completion, verify no new warnings during batch inference:

```bash
python -c "
import warnings
warnings.filterwarnings('error')  # Turn warnings into errors
from img2pose import Img2Pose
import numpy as np
d = Img2Pose(device='cpu')
imgs = [np.ones((480, 640, 3), dtype=np.uint8)] * 4
d.detect_faces(imgs)
print('No warnings detected')
" 2>&1
```

If new warnings appear, add filters to `src/img2pose/__init__.py` following the existing pattern at lines 20-32.

---

## 9. Performance Characteristics

### 9.1 Expected Improvements

| Batch Size | Sequential | Batched | Speedup |
|------------|------------|---------|---------|
| 1 | 1.0x | 1.0x | No overhead |
| 4 | 4.0x | ~1.2x | ~3x faster |
| 8 | 8.0x | ~1.5x | ~5x faster |
| 16 | 16.0x | ~2.0x | ~8x faster |

(Estimated based on typical R-CNN batch efficiency)

### 9.2 Memory Considerations

- Batch processing uses more peak GPU memory
- Memory scales approximately linearly with batch size
- No documentation changes needed (user decides batch size)

### 9.3 CPU vs GPU

- Batch benefits are more pronounced on GPU
- CPU inference may see modest improvement due to reduced Python loop overhead

---

## 10. Test Strategy Summary

| Test Type | Purpose | Location |
|-----------|---------|----------|
| Unit: empty batch | Edge case | `tests/unit/test_detector.py` |
| Unit: single-element batch | Edge case | `tests/unit/test_detector.py` |
| Integration: batch equivalence | Correctness | `tests/integration/test_inference.py` |
| Integration: batch types | Type safety | `tests/integration/test_inference.py` |
| Performance: batch vs sequential | Performance | `tests/integration/test_inference.py` |

---

## 11. Versioning and Migration

### 11.1 Versioning

This change is **backward compatible** and does not require a version bump for the public API.

If the package uses semver:
- Patch version bump appropriate (bug fix / performance improvement)
- No breaking changes

### 11.2 Migration

**No migration needed.** Existing code calling `detect_faces()` with lists will automatically benefit from batch processing without any changes.

---

## 12. Summary of Changes

| File | Change Type | Lines Affected |
|------|-------------|----------------|
| `detector.py` | Add method | +35 (`_detect_batch`) |
| `detector.py` | Add method | +45 (`_process_prediction`) |
| `detector.py` | Modify method | 242-249 (`detect_faces`) |
| `detector.py` | Refactor method | 257-313 (`_detect_single`) |
| `test_detector.py` | Add tests | +30 (batch edge cases) |
| `test_inference.py` | Add tests | +60 (equivalence, performance) |

**Total:** ~170 lines of changes across 3 files.
