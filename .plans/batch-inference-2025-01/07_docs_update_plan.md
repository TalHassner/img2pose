# Documentation Update Plan: Efficient Batch Inference

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** Ready for Review

---

## 1. Summary

This change is **100% backward compatible** and **internal only**. No public API changes are made, so documentation updates are **minimal**.

### Documentation Status

| Document | Update Required | Reason |
|----------|-----------------|--------|
| `src/img2pose/README.md` | No | Batch processing already documented (lines 64, 111-113, 238-252) |
| `CHANGELOG.md` | Yes | Add performance improvement entry |
| Docstrings | Yes | New internal methods need docstrings |
| Type hints | Yes | New methods need complete type hints |

---

## 2. CHANGELOG.md Update

### File: `/home/tal/dev/img2pose/CHANGELOG.md`

**Action:** Add new entry under a `[1.0.1]` or `[Unreleased]` section.

**Location:** Insert after line 7 (after the format description, before `[1.0.0]`)

**Content to Add:**

```markdown
## [Unreleased]

### Changed

- Batch processing now uses efficient single-pass inference instead of sequential processing
- Performance improvement: batch processing N images is now significantly faster than N individual calls

### Technical Notes

- Internal refactoring: extracted `_process_prediction()` helper for code reuse
- Internal addition: `_detect_batch()` method for batched model calls
- No API changes: all public interfaces remain unchanged
```

---

## 3. README.md Assessment

### File: `/home/tal/dev/img2pose/src/img2pose/README.md`

**Action:** No changes required.

**Reason:** Batch processing is already documented:

1. **Input types section (line 64):**
   > `- List of any of the above (batch processing)`

2. **Input formats example (lines 111-113):**
   ```python
   # Batch processing
   faces = detector.detect_faces(["img1.jpg", "img2.jpg", "img3.jpg"])
   ```

3. **Batch processing example (lines 238-252):**
   ```python
   # Process batch
   all_faces = detector.detect_faces(images)
   ```

The documentation makes no performance claims that would need updating. Users will automatically benefit from the performance improvement without any code changes.

---

## 4. Internal Docstrings

### 4.1 New Method: `_process_prediction()`

**File:** `/home/tal/dev/img2pose/src/img2pose/detector.py`

**Required Docstring:**

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

    This method converts raw model output into the standardized FaceDict
    format, applying threshold filtering, confidence sorting, and max_count
    limiting.

    Args:
        pred: Model prediction dict with keys:
            - "boxes": Tensor of shape [N, 4] with xyxy bounding boxes
            - "scores": Tensor of shape [N] with confidence scores
            - "dofs": Tensor of shape [N, 6] with 6DoF poses
        width: Original image width (for landmark projection)
        height: Original image height (for landmark projection)
        threshold: Minimum confidence score (faces below are excluded)
        max_count: Maximum faces to return (-1 for unlimited)

    Returns:
        List of face dictionaries sorted by confidence (descending).
        Each dict contains: box, confidence, pose, keypoints.
    """
```

### 4.2 New Method: `_detect_batch()`

**File:** `/home/tal/dev/img2pose/src/img2pose/detector.py`

**Required Docstring:**

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
    The underlying FasterDoFRCNN model natively supports batch inference.

    Args:
        images: List of input images. Each can be:
            - File path (str or Path)
            - PIL Image
            - Numpy array (RGB uint8 HWC)
        threshold: Minimum confidence score for detections
        max_count: Maximum faces per image (-1 for unlimited)

    Returns:
        List of face lists, one per input image. Order matches input order.
        Each face list contains FaceDict objects with: box, confidence,
        pose, keypoints.

    Note:
        For a single image, use _detect_single() to avoid batch overhead.
        The detect_faces() method handles this routing automatically.
    """
```

---

## 5. Type Hints Verification

### 5.1 Required Type Imports

Ensure these imports exist at the top of `detector.py`:

```python
from typing import Any, Dict, List, Optional, Union
```

Already present at line 9.

### 5.2 New Method Signatures

Both new methods must have complete type annotations:

```python
def _process_prediction(
    self,
    pred: Dict[str, torch.Tensor],
    width: int,
    height: int,
    threshold: float,
    max_count: int,
) -> List[FaceDict]:
    ...

def _detect_batch(
    self,
    images: List[ImageInput],
    threshold: float,
    max_count: int,
) -> List[List[FaceDict]]:
    ...
```

### 5.3 Type Verification Command

```bash
mypy src/img2pose/detector.py --show-error-codes
```

Expected output: `Success: no issues found`

---

## 6. Existing Documentation Conventions

### 6.1 Docstring Style

The codebase uses Google-style docstrings. Example from `detect_faces()`:

```python
"""Detect faces in an image or batch of images.

Args:
    image: Input image(s). Can be:
        - File path (str or Path)
        - PIL Image
        - Numpy array (RGB uint8 HWC)
        - List of the above for batch processing
    score_threshold: Override instance threshold (optional)
    max_faces: Override instance max_faces (optional)

Returns:
    List of face dictionaries for single image, or
    list of lists for batch input.
"""
```

**Convention:** Follow this style for new docstrings.

### 6.2 README Style

The README uses:
- Headers with `##` for major sections
- Code blocks with language specifiers (```python)
- Tables with pipes and dashes
- No emojis

**Convention:** If adding to README (optional), follow this style.

---

## 7. Documentation Verification Checklist

After implementation, verify:

- [ ] `_process_prediction()` has complete docstring
- [ ] `_detect_batch()` has complete docstring
- [ ] Both methods have complete type annotations
- [ ] `mypy src/img2pose/detector.py` passes
- [ ] `CHANGELOG.md` has new entry
- [ ] No changes to `README.md` public API docs (none needed)

---

## 8. Optional Enhancement (Not Required)

If desired, the README could be enhanced with a performance note:

**Potential Addition (NOT REQUIRED):**

```markdown
### Performance Tips

For processing multiple images, pass them as a list for optimal performance:

```python
# Efficient: single batched call
all_faces = detector.detect_faces(["img1.jpg", "img2.jpg", "img3.jpg"])

# Less efficient: separate calls
faces1 = detector.detect_faces("img1.jpg")
faces2 = detector.detect_faces("img2.jpg")
faces3 = detector.detect_faces("img3.jpg")
```
```

**Decision:** Not adding - the current documentation already shows batch usage, and adding performance claims creates maintenance burden.

---

## 9. Summary of Changes

| File | Section | Action | Lines Changed |
|------|---------|--------|---------------|
| `CHANGELOG.md` | New section | Add | +12 lines |
| `detector.py` | `_process_prediction` | Add docstring | +18 lines |
| `detector.py` | `_detect_batch` | Add docstring | +22 lines |
| `README.md` | - | None | 0 lines |

**Total Documentation Changes:** ~52 lines
