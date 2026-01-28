# Plan Digest: Efficient Batch Inference for img2pose

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** READY FOR IMPLEMENTATION

---

## Quick Summary

Add efficient batch inference to `Img2Pose.detect_faces()` by leveraging the underlying model's native batch processing capability. Currently, batch inputs are processed sequentially in a loop; this change will process all images in a single model forward pass.

**Key Benefit:** Processing N images in a batch will be significantly faster than N individual calls.

**Backward Compatibility:** 100% - no API changes, identical output format, numerically equivalent results.

---

## Milestones

| ID | Description | Files | Est. Lines |
|----|-------------|-------|------------|
| M1 | Extract `_process_prediction()` helper | detector.py | +60, -25 |
| M2 | Add `_detect_batch()` method | detector.py | +40 |
| M3 | Update `detect_faces()` routing | detector.py | +8 |
| M3.5 | Add coverage config (optional) | pyproject.toml | +15 |
| M4 | Add batch equivalence tests | test_inference.py, conftest.py | +100 |
| M5 | Add performance benchmark tests | test_inference.py | +80 |

---

## File Ownership

| File | Owner | Changes |
|------|-------|---------|
| `src/img2pose/detector.py` | Implementation | M1, M2, M3 |
| `pyproject.toml` | Configuration | M3.5 |
| `tests/integration/test_inference.py` | Testing | M4, M5 |
| `tests/conftest.py` | Testing | M4 |
| `CHANGELOG.md` | Documentation | Post-implementation |

---

## Required Test Gates

### Pre-Commit (per milestone)
```bash
black --check src/img2pose/detector.py
isort --check src/img2pose/detector.py
mypy src/img2pose/detector.py
pytest tests/unit/test_detector.py -v
```

### Pre-Merge (after all milestones)
```bash
pytest tests/ -m "not slow" --cov=src/img2pose --cov-fail-under=80
```

### Performance Validation (optional, requires weights)
```bash
pytest tests/integration/test_inference.py::TestBatchPerformance -v -s
```

---

## Definition of Done

### Implementation Complete When:
- [ ] M1: `_process_prediction()` method exists and is used by `_detect_single()`
- [ ] M2: `_detect_batch()` method exists with complete docstring
- [ ] M3: `detect_faces()` routes batch inputs through `_detect_batch()`
- [ ] M3: Empty batch `[]` returns `[]`
- [ ] M3: Single-element batch `[img]` uses `_detect_single()` (no overhead)
- [ ] All existing tests pass without modification

### Testing Complete When:
- [ ] M4: `TestBatchInference` class with 6+ tests
- [ ] M4: `test_no_warnings_during_batch` passes
- [ ] M5: `TestBatchPerformance` class with speedup verification
- [ ] Coverage >= 80% for `src/img2pose/detector.py`

### Documentation Complete When:
- [ ] New methods have complete docstrings
- [ ] CHANGELOG.md updated with performance improvement entry

---

## Critical Implementation Details

### Batch Detection Flow
```python
def _detect_batch(self, images, threshold, max_count):
    # 1. Load all images, capture dimensions
    pil_images = [_load_image(img) for img in images]
    dimensions = [pil.size for pil in pil_images]  # (width, height)

    # 2. Convert to tensors
    tensors = [_image_to_tensor(pil, self.device) for pil in pil_images]

    # 3. Single model call (key optimization)
    predictions = self._model.predict(tensors)

    # 4. Post-process with per-image dimensions
    return [
        self._process_prediction(pred, w, h, threshold, max_count)
        for pred, (w, h) in zip(predictions, dimensions)
    ]
```

### Routing Logic Change
```python
# In detect_faces():
if isinstance(image, list):
    if len(image) == 0:
        return []
    if len(image) == 1:
        return [self._detect_single(image[0], threshold, max_count)]
    return self._detect_batch(image, threshold, max_count)
```

---

## Key Constraints

1. **No API changes** - Signatures must remain identical
2. **No new dependencies** - Only use existing packages
3. **No new files** - All code in `detector.py`
4. **File size < 500 LOC** - `detector.py` currently 373 lines, expected ~410 after
5. **Numerical equivalence** - Batch results must equal sequential results

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU float differences | Documented tolerance fallback (atol=1e-5) |
| New warnings during batch | Verification step added (design spec 8.5) |
| Performance regression | Benchmark tests verify speedup |
| Breaking existing behavior | All existing tests must pass unchanged |

---

## Quick Start Commands

```bash
# 1. Verify current state
cd /home/tal/dev/img2pose
pytest tests/ -v --tb=short

# 2. After each milestone
pytest tests/unit/test_detector.py -v

# 3. After M3 (implementation complete)
python -c "
from img2pose import Img2Pose
import numpy as np
d = Img2Pose(device='cpu')
imgs = [np.ones((100, 100, 3), dtype=np.uint8)] * 3
result = d.detect_faces(imgs)
print(f'Batch result: {len(result)} items, type: {type(result)}')
"

# 4. Final verification
pytest tests/ -m "not slow" --cov=src/img2pose --cov-fail-under=80
```

---

## Plan Documents Reference

| Document | Purpose |
|----------|---------|
| `00_scope_and_success_criteria.md` | Requirements and success definition |
| `01_repo_inventory.md` | Evidence-backed code analysis |
| `02_design_spec.md` | Architecture and interface design |
| `03_implementation_task_breakdown.md` | Step-by-step implementation guide |
| `04_test_strategy.md` | Test taxonomy and fixtures |
| `05_coverage_and_gates.md` | Quality gates and coverage targets |
| `06_traceability_matrix.md` | Requirement to test mapping |
| `07_docs_update_plan.md` | Documentation changes |
| `90_gap_report.md` | Audit results (PASS) |

---

**PLAN STATUS: COMPLETE - READY FOR /build-implementation**
