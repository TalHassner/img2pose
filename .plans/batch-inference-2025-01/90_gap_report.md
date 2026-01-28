# Gap Report: Batch Inference Implementation Plan

**Plan ID:** batch-inference-2025-01
**Audit Date:** 2025-01-27
**Auditor:** Plan Auditor Agent

---

## Audit Summary

The batch inference implementation plan is **well-structured and thorough**. The documents demonstrate clear understanding of the codebase, provide evidence-backed claims, and include comprehensive test coverage planning.

**Overall Assessment:** PASS - All warnings addressed

| Category | BLOCKER | WARNING | INFO |
|----------|---------|---------|------|
| Unstated Assumptions | 0 | 0 (fixed) | 1 |
| Missing Evidence | 0 | 0 | 1 |
| Missing Tests | 0 | 0 (fixed) | 0 |
| TBD/TODO/Vague Language | 0 | 0 | 1 |
| Goal Drift | 0 | 0 | 0 |
| Backward Compatibility Gaps | 0 | 0 | 1 |
| Warning Suppression Gaps | 0 | 0 (fixed) | 0 |
| Coverage Gaps | 0 | 0 (fixed) | 0 |
| **Total** | **0** | **0** | **4** |

### Warnings Resolved

- **WARNING-1**: Float equality assumption documented with tolerance fallback in `00_scope_and_success_criteria.md`
- **WARNING-2**: `test_no_warnings_during_batch` implementation added to `03_implementation_task_breakdown.md`
- **WARNING-3**: Warning suppression verification step added to `02_design_spec.md` section 8.5
- **WARNING-4**: Coverage configuration added as M3.5 step in `03_implementation_task_breakdown.md`

---

## Detailed Findings

### 1. Unstated Assumptions

#### WARNING-1: PyTorch determinism assumption for exact float equality

**Location:** `00_scope_and_success_criteria.md` lines 88-94, `04_test_strategy.md` lines 268-297

**Issue:** The plan claims exact float equality between batch and sequential results without acknowledging potential non-determinism from:
- GPU floating-point operation ordering
- cuDNN non-deterministic algorithms
- Different tensor memory layouts in batched vs single inference

**Evidence:** The plan states (02_design_spec.md line 401):
```python
assert batch_result == sequential_result  # Exact float equality
```

While `_rcnn.py` does maintain order preservation (lines 69-87), the underlying PyTorch operations may produce slightly different results due to floating-point associativity when processing batched tensors.

**Risk:** Tests may fail intermittently on GPU due to floating-point differences.

**Recommended Fix:** Add to `00_scope_and_success_criteria.md` section 78-94:

```markdown
### Numerical Equivalence Clarification

Exact equality is expected because:
1. Same normalization path (`_image_to_tensor()`)
2. Same model forward pass operations
3. Same post-processing (`_process_prediction()`)
4. PyTorch in eval mode with deterministic operations

If exact equality fails on GPU, use tolerance-based comparison:
- Box coordinates: atol=1e-5
- Confidence scores: atol=1e-6
- Pose values: atol=1e-5
- Keypoint coordinates: atol=1e-4

Tests should first attempt exact equality, then fall back to tolerance if needed.
```

---

#### INFO-1: Model wrapper behavior with DataParallel

**Location:** `_model.py` lines 126-133

**Issue:** The plan does not explicitly discuss how `DataParallel` wrapper affects batch processing on GPU.

**Evidence from source:**
```python
# _model.py lines 130-133
self.fpn_model = DataParallel(self.fpn_model)
self.fpn_model = self.fpn_model.to(self.device)
```

**Risk:** Low - DataParallel handles batching transparently, but this is an implicit assumption.

**No action required** - behavior is correct, but could be documented in 01_repo_inventory.md for completeness.

---

### 2. Missing Evidence

#### INFO-2: Line number references may drift

**Location:** Throughout all documents

**Issue:** Many line number references (e.g., "detector.py:243-247") are based on current source state. After implementation, these will be outdated.

**Evidence:** Source file `/home/tal/dev/img2pose/src/img2pose/detector.py` is 373 lines. The sequential loop is at lines 243-247 as stated.

**Risk:** Low - documentation becomes stale but doesn't affect implementation.

**No action required** - line numbers are currently accurate. Post-implementation, update references if documents are kept.

---

### 3. Missing Tests

#### WARNING-2: No test for M6 warning suppression during batch inference

**Location:** `04_test_strategy.md` line 41, `06_traceability_matrix.md` line 20

**Issue:** The traceability matrix references `test_no_warnings_during_batch` but this test is not fully specified in any test implementation template.

**Evidence from 04_test_strategy.md line 41:**
```
| I8 | `test_no_warnings_during_batch` | No deprecation warnings emitted | `tests/integration/test_inference.py` |
```

But no implementation template is provided for this test.

**Risk:** M6 requirement may not be verified during batch processing specifically.

**Recommended Fix:** Add test implementation to `03_implementation_task_breakdown.md` M4 section:

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
                mock_model.predict = MagicMock(return_value=[])
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose()
                detector.detect_faces(sample_images_batch)

        # Filter for deprecation warnings from img2pose
        relevant = [x for x in w if "img2pose" in str(x.filename)]
        assert len(relevant) == 0, f"Unexpected warnings: {relevant}"
```

---

### 4. TBD/TODO/Vague Language

#### INFO-3: "Optional Refactor" in design spec

**Location:** `02_design_spec.md` lines 221-251, 310-312

**Issue:** The design spec presents `_process_prediction()` as "Optional Refactor" with two options (A and B).

**Evidence:**
```markdown
### 4.2 New Method: `_process_prediction()` (Optional Refactor)
...
**Option A (Recommended):** Refactor to use `_process_prediction()`
**Option B (Minimal Change):** Keep `_detect_single()` unchanged
```

**Risk:** Low - the task breakdown (03_implementation_task_breakdown.md) clearly chooses Option A in M1.

**No action required** - the ambiguity is resolved in the implementation plan.

---

### 5. Goal Drift

No goal drift identified. All plan elements trace back to the original requirements in `00_scope_and_success_criteria.md`.

---

### 6. Backward Compatibility Gaps

#### INFO-4: Empty batch edge case not in existing tests

**Location:** `04_test_strategy.md` section 5.1

**Issue:** The plan adds tests for empty batch `[]` returning `[]`, but this behavior is not explicitly tested in existing tests.

**Evidence from current source (`detector.py` lines 243-247):**
```python
if isinstance(image, list):
    return [
        self._detect_single(img, threshold, max_count)
        for img in image
    ]
```

Current code would return `[]` for empty list (list comprehension over empty list), so behavior is preserved.

**Risk:** None - behavior is already correct and will be explicitly tested.

**No action required** - plan correctly identifies and tests this edge case.

---

### 7. Warning Suppression Gaps

#### WARNING-3: Potential new warnings from batch processing not addressed

**Location:** `01_repo_inventory.md` lines 296-302, `__init__.py` lines 20-32

**Issue:** The inventory mentions potential additional warnings for batch processing but doesn't specify suppression strategy:

**Evidence from 01_repo_inventory.md:**
```markdown
### Potential Additional Warnings

With batch processing, may need to suppress:
- DataParallel warnings for single-GPU multi-image batches
- Memory warnings for large batches
- Tensor size mismatch warnings from GeneralizedRCNNTransform
```

Current suppression in `__init__.py` only covers:
- `torch.meshgrid` deprecation
- `backbone_name` positional argument

**Risk:** New warnings may appear during batch inference.

**Recommended Fix:** Add to `02_design_spec.md` section 8 (Error Handling) or create new section:

```markdown
## 8.5 Warning Suppression Verification

Before M3 completion, run batch inference and capture stderr:
```bash
python -c "
import warnings
warnings.filterwarnings('error')  # Turn warnings into errors
from img2pose import Img2Pose
import numpy as np
d = Img2Pose(device='cpu')
imgs = [np.ones((480, 640, 3), dtype=np.uint8)] * 4
d.detect_faces(imgs)
" 2>&1
```

If new warnings appear, add filters to `src/img2pose/__init__.py` following existing pattern.
```

---

### 8. Coverage Gaps

#### WARNING-4: Coverage configuration not in pyproject.toml

**Location:** `05_coverage_and_gates.md` lines 107-128

**Issue:** The plan specifies coverage configuration to add to `pyproject.toml`, but this is listed as a recommendation rather than a required implementation step.

**Evidence from `pyproject.toml` (current):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short --import-mode=importlib"
pythonpath = ["src"]
```

No `[tool.coverage]` section exists.

**Risk:** Coverage gate (`--cov-fail-under=80`) may not work consistently without configuration.

**Recommended Fix:** Add to `03_implementation_task_breakdown.md` as prerequisite step before M4:

```markdown
### M3.5 (Optional): Add Coverage Configuration

**File:** `/home/tal/dev/img2pose/pyproject.toml`

Add after `[tool.mypy]` section:

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
```

---

## Test File Verification

The following test files referenced in plans exist:
- `/home/tal/dev/img2pose/tests/unit/test_detector.py` - EXISTS (194 lines)
- `/home/tal/dev/img2pose/tests/integration/test_inference.py` - EXISTS (131 lines)
- `/home/tal/dev/img2pose/tests/equivalence/test_numerical.py` - EXISTS (143 lines)
- `/home/tal/dev/img2pose/tests/conftest.py` - EXISTS (82 lines)

The following planned test files do not exist yet (as expected):
- `/home/tal/dev/img2pose/tests/integration/test_performance.py` - PLANNED
- `/home/tal/dev/img2pose/tests/equivalence/test_batch_sequential.py` - PLANNED

---

## Source Code Verification

### Verified Claims

| Claim | Location | Status |
|-------|----------|--------|
| Sequential loop at detector.py:243-247 | 01_repo_inventory.md | VERIFIED |
| `_load_image()` at detector.py:29-59 | 01_repo_inventory.md | VERIFIED (lines 29-59) |
| `_image_to_tensor()` at detector.py:62-67 | 01_repo_inventory.md | VERIFIED (lines 62-67) |
| `_project_landmarks()` at detector.py:75-107 | 01_repo_inventory.md | VERIFIED (lines 75-107) |
| `img2poseModel.predict()` accepts `List[Tensor]` | 01_repo_inventory.md | VERIFIED (_model.py:152) |
| `GeneralizedRCNN.forward()` accepts `List[Tensor]` | 01_repo_inventory.md | VERIFIED (_rcnn.py:49-52) |
| Warning filters in `__init__.py:20-32` | 01_repo_inventory.md | VERIFIED (lines 20-32) |
| `detector.py` is 373 lines | 02_design_spec.md | VERIFIED |
| CHANGELOG.md exists | 07_docs_update_plan.md | VERIFIED |
| Existing test class names | 01_repo_inventory.md | VERIFIED |

---

## Traceability Matrix Verification

### Consistency Check

The traceability matrix (`06_traceability_matrix.md`) is consistent with:
- Requirements in `00_scope_and_success_criteria.md`
- Implementation tasks in `03_implementation_task_breakdown.md`
- Tests specified in `04_test_strategy.md`

No inconsistencies found.

---

## Recommended Priority Order for Fixes

1. **WARNING-3** (Warning Suppression): Add verification step to ensure no new warnings during batch
2. **WARNING-4** (Coverage Config): Add coverage configuration to pyproject.toml
3. **WARNING-2** (Missing Test): Add `test_no_warnings_during_batch` implementation
4. **WARNING-1** (Float Equality): Document tolerance fallback for GPU tests

---

## Conclusion

The plan is **ready for implementation** with the noted warnings. All MUST requirements have implementation steps, tests, and no documentation updates are needed (beyond CHANGELOG).

The warnings identified are:
- Minor documentation clarifications
- One missing test implementation template
- Configuration recommendations

None of these block implementation. They can be addressed during or immediately after implementation.

**Audit Result: PASS**

