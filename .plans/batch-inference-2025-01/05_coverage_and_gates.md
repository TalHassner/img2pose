# Coverage and Gates: Efficient Batch Inference for img2pose

**Plan ID:** batch-inference-2025-01
**Date:** 2025-01-27
**Status:** Ready for Review

---

## 1. Coverage Targets

### 1.1 Overall Coverage Goal

| Metric | Target | Rationale |
|--------|--------|-----------|
| Line coverage (new code) | >= 80% | Standard for production code |
| Branch coverage (new code) | >= 75% | Ensure all paths tested |
| Function coverage | 100% | All new functions must be tested |

### 1.2 Coverage by Module

| Module | File | Current Coverage | Target Coverage | Notes |
|--------|------|------------------|-----------------|-------|
| detector | `src/img2pose/detector.py` | ~70% | >= 80% | Focus on new batch code |
| New: `_process_prediction()` | `detector.py` | 0% | >= 90% | Critical new function |
| New: `_detect_batch()` | `detector.py` | 0% | >= 90% | Critical new function |
| Modified: `detect_faces()` | `detector.py` | ~80% | >= 85% | Routing logic additions |

### 1.3 Critical Paths Requiring Full Coverage

The following code paths MUST have 100% line coverage:

1. **Empty batch handling** (line coverage)
   ```python
   if len(image) == 0:
       return []
   ```

2. **Single-element batch optimization** (line coverage)
   ```python
   if len(image) == 1:
       return [self._detect_single(image[0], threshold, max_count)]
   ```

3. **Multi-element batch routing** (line coverage)
   ```python
   return self._detect_batch(image, threshold, max_count)
   ```

4. **Threshold filtering in `_process_prediction()`** (branch coverage)
   ```python
   if scores[i] < threshold:
       continue
   ```

5. **Max count limiting** (branch coverage)
   ```python
   if max_count > 0:
       faces = faces[:max_count]
   ```

---

## 2. Coverage Measurement

### 2.1 Tools Configuration

The project uses `pytest-cov` (already in `pyproject.toml` dev dependencies):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    # ...
]
```

### 2.2 Coverage Commands

```bash
# Basic coverage run
cd /home/tal/dev/img2pose
python -m pytest tests/ --cov=src/img2pose --cov-report=term-missing

# Coverage with HTML report
python -m pytest tests/ --cov=src/img2pose --cov-report=html --cov-report=term-missing

# Coverage for specific module (detector.py)
python -m pytest tests/ --cov=src/img2pose/detector --cov-report=term-missing

# Coverage with branch tracking
python -m pytest tests/ --cov=src/img2pose --cov-branch --cov-report=term-missing

# Coverage excluding slow tests
python -m pytest tests/ -m "not slow" --cov=src/img2pose --cov-report=term-missing
```

### 2.3 Coverage Report Locations

| Report Type | Location | Use Case |
|-------------|----------|----------|
| Terminal | stdout | Quick check |
| HTML | `htmlcov/index.html` | Detailed analysis |
| XML | `coverage.xml` | CI integration |

### 2.4 Coverage Configuration

Add to `pyproject.toml` for consistent coverage settings:

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

---

## 3. Required Gates

### 3.1 Pre-Commit Gates

These must pass before committing any changes:

| Gate | Command | Threshold | Status |
|------|---------|-----------|--------|
| Unit tests | `pytest tests/unit/` | 100% pass | Required |
| Type check | `mypy src/img2pose/detector.py` | No errors | Required |
| Format check | `black --check src/img2pose/` | No changes | Required |
| Import sort | `isort --check src/img2pose/` | No changes | Required |

### 3.2 Pre-Merge Gates

These must pass before merging to main branch:

| Gate | Command | Threshold | Status |
|------|---------|-----------|--------|
| All tests | `pytest tests/ -m "not slow"` | 100% pass | Required |
| Coverage | `pytest --cov --cov-fail-under=80` | >= 80% | Required |
| Integration tests | `pytest tests/integration/` | 100% pass | Required |
| No warnings | `pytest -W error::DeprecationWarning` | No warnings | Required |

### 3.3 Gate Commands

```bash
# Run all gates locally before commit
cd /home/tal/dev/img2pose

# 1. Format check
black --check src/img2pose/detector.py

# 2. Import sort check
isort --check src/img2pose/detector.py

# 3. Type check
mypy src/img2pose/detector.py

# 4. Unit tests
pytest tests/unit/ -v

# 5. All tests with coverage
pytest tests/ -m "not slow" --cov=src/img2pose --cov-fail-under=80

# One-liner for all gates:
black --check src/img2pose/ && isort --check src/img2pose/ && mypy src/img2pose/detector.py && pytest tests/ -m "not slow" --cov=src/img2pose --cov-fail-under=80
```

---

## 4. Existing Tool Configuration

The project already has these tools configured in `pyproject.toml`:

### 4.1 pytest

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short --import-mode=importlib"
pythonpath = ["src"]
```

### 4.2 black

```toml
[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311", "py312"]
```

### 4.3 isort

```toml
[tool.isort]
profile = "black"
line_length = 100
```

### 4.4 mypy

```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

---

## 5. Type Checking Details

### 5.1 New Type Annotations Required

The following new code must have complete type annotations:

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

### 5.2 mypy Check Command

```bash
# Check detector module
mypy src/img2pose/detector.py --strict

# Check with detailed output
mypy src/img2pose/detector.py --show-error-codes --show-column-numbers
```

### 5.3 Expected mypy Results

After implementation, mypy should report:

```
Success: no issues found in 1 source file
```

---

## 6. Regression Testing

### 6.1 API Regression Tests

Existing tests that must continue to pass (no modification allowed):

| Test File | Test Class/Function | Purpose |
|-----------|---------------------|---------|
| `test_detector.py` | `TestImageLoading` | Input handling unchanged |
| `test_detector.py` | `TestImageToTensor` | Tensor conversion unchanged |
| `test_detector.py` | `TestFaceOutputFormat` | Output format unchanged |
| `test_inference.py` | `TestImg2PoseInference` | API contracts unchanged |
| `test_inference.py` | `TestVisualization` | Visualization unchanged |

### 6.2 Regression Command

```bash
# Run existing tests to verify no regressions
pytest tests/unit/test_detector.py tests/integration/test_inference.py -v

# Compare with baseline (if available)
pytest tests/ --tb=short 2>&1 | diff - tests/baseline_output.txt
```

---

## 7. Warning Suppression Verification

### 7.1 Required: No User-Visible Warnings

The implementation must not emit any deprecation warnings during normal use.

### 7.2 Warning Check Command

```bash
# Run tests and fail on any warnings
pytest tests/ -W error::DeprecationWarning -W error::FutureWarning

# Check for torch.meshgrid warning specifically
python -c "
import warnings
warnings.filterwarnings('error')
from img2pose import Img2Pose
# If no exception, warnings are suppressed
print('No warnings emitted')
"
```

### 7.3 Warnings to Suppress

| Warning | Source | Suppression Location |
|---------|--------|----------------------|
| `torch.meshgrid` indexing | torchvision RPN | `_model.py` or `_rcnn.py` |
| `backbone_name` deprecation | torchvision | `_model.py` |

---

## 8. Performance Gate (Optional)

### 8.1 Performance Regression Test

While not a hard gate, performance should not regress:

```python
@pytest.mark.slow
def test_no_performance_regression():
    """Ensure batch inference is not slower than sequential."""
    detector = Img2Pose(device="cpu")
    images = [np.ones((480, 640, 3), dtype=np.uint8)] * 4

    # Baseline: sequential
    import time
    start = time.perf_counter()
    for img in images:
        detector.detect_faces(img)
    seq_time = time.perf_counter() - start

    # New: batch
    start = time.perf_counter()
    detector.detect_faces(images)
    batch_time = time.perf_counter() - start

    # Batch must not be slower
    assert batch_time <= seq_time * 1.1, f"Batch {batch_time:.3f}s > Sequential {seq_time:.3f}s"
```

### 8.2 Performance Benchmark Command

```bash
# Run performance tests with output
pytest tests/integration/test_performance.py -v -s --tb=short
```

---

## 9. Coverage Checklist for Implementation

### 9.1 M1: Extract `_process_prediction()` Helper

Coverage requirements:

- [ ] Empty predictions handled (0 faces)
- [ ] Single face processed correctly
- [ ] Multiple faces processed correctly
- [ ] Threshold filtering works
- [ ] Confidence sorting works
- [ ] max_count=0 returns all faces
- [ ] max_count>0 limits faces

### 9.2 M2: Add `_detect_batch()` Method

Coverage requirements:

- [ ] Multiple images processed
- [ ] Dimensions captured correctly
- [ ] Model called once with all tensors
- [ ] Results mapped to correct images
- [ ] Empty predictions handled per-image

### 9.3 M3: Update `detect_faces()` Routing

Coverage requirements:

- [ ] Empty list returns `[]`
- [ ] Single-element list uses `_detect_single`
- [ ] Multi-element list uses `_detect_batch`
- [ ] Single image (not list) uses `_detect_single`

---

## 10. CI Configuration Example

### 10.1 GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Lint with black
        run: black --check src/img2pose/

      - name: Check imports with isort
        run: isort --check src/img2pose/

      - name: Type check with mypy
        run: mypy src/img2pose/detector.py

      - name: Run tests with coverage
        run: |
          pytest tests/ -m "not slow and not requires_weights" \
            --cov=src/img2pose \
            --cov-report=xml \
            --cov-fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml
```

---

## 11. Summary

### Gates Summary Table

| Gate | Type | Command | Threshold | Blocking |
|------|------|---------|-----------|----------|
| black | Format | `black --check` | No changes | Yes |
| isort | Format | `isort --check` | No changes | Yes |
| mypy | Type | `mypy` | No errors | Yes |
| pytest | Test | `pytest -m "not slow"` | 100% pass | Yes |
| coverage | Coverage | `--cov-fail-under=80` | >= 80% | Yes |
| warnings | Quality | `-W error` | No warnings | Yes |
| performance | Perf | `test_performance.py` | No regression | No |

### Quick Reference Commands

```bash
# Full gate check
make check  # if Makefile exists, or:

# Manual full check
cd /home/tal/dev/img2pose && \
  black --check src/img2pose/ && \
  isort --check src/img2pose/ && \
  mypy src/img2pose/detector.py && \
  pytest tests/ -m "not slow" --cov=src/img2pose --cov-fail-under=80 -W error::DeprecationWarning
```
