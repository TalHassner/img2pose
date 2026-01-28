# Coverage and Quality Gates: img2pose pip-installable Library

**Version:** 1.0
**Date:** 2025-01-27
**Reference:** `02_design_spec.md`, `04_test_strategy.md`

---

## 1. Coverage Target

### 1.1 Overall Coverage Goal

**Target: >= 80% line coverage on new modules in `src/img2pose/`**

| Module | Target Coverage | Rationale |
|--------|-----------------|-----------|
| `detector.py` | >= 90% | Public API, critical for users |
| `_weights.py` | >= 85% | Download logic, error handling |
| `_visualization.py` | >= 80% | Optional feature, less critical |
| `_model.py` | >= 70% | Wrapper around complex model code |
| `_models.py` | >= 60% | Inference paths only, training excluded |
| `_rpn.py` | >= 60% | Inference paths only |
| `_generalized_rcnn.py` | >= 60% | Inference paths only |
| `utils/pose_operations.py` | >= 75% | Core math operations |
| `utils/image_operations.py` | >= 70% | Utility functions |

### 1.2 Coverage Exclusions

The following are intentionally excluded from coverage requirements:

```python
# pyproject.toml
[tool.coverage.run]
source = ["src/img2pose"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/_model_loader.py",  # Thin wrapper
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if torch.cuda.is_available():",  # GPU-specific code
    "except ImportError:",  # Optional dependency handling
]
```

### 1.3 Coverage Measurement

```bash
# Run tests with coverage
pytest tests/ --cov=src/img2pose --cov-report=term-missing --cov-report=html

# Generate coverage report
coverage report --show-missing

# Check coverage threshold
coverage report --fail-under=80
```

---

## 2. Quality Gates

### 2.1 Gate Summary

| Gate | Tool | Threshold | Blocking |
|------|------|-----------|----------|
| Lint | ruff | 0 errors | Yes |
| Format | black | Compliant | Yes |
| Type Check | mypy | 0 errors (new code) | Yes |
| Unit Tests | pytest | 100% pass | Yes |
| Integration Tests | pytest | 100% pass | Yes |
| Coverage | pytest-cov | >= 80% | Yes |

### 2.2 Gate Rationale

The repository does NOT currently use lint/format/typecheck tools. We add them because:

1. **Lint (ruff)**: Catches common errors and enforces Python best practices
2. **Format (black)**: Ensures consistent code style across new modules
3. **Type Check (mypy)**: Validates type hints in public API for user safety

We scope these gates to **new code only** (`src/img2pose/`) to avoid breaking existing training code.

---

## 3. Gate Commands

### 3.1 Lint with ruff

```bash
# Install
pip install ruff

# Run lint on new code only
ruff check src/img2pose/

# Auto-fix safe issues
ruff check src/img2pose/ --fix

# Configuration in pyproject.toml
```

**Configuration:**

```toml
# pyproject.toml
[tool.ruff]
target-version = "py38"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
]
ignore = [
    "E501",   # line too long (handled by black)
    "B008",   # do not perform function calls in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__
```

### 3.2 Format with black

```bash
# Install
pip install black

# Check formatting (CI mode)
black --check src/img2pose/

# Auto-format
black src/img2pose/

# Check with diff
black --check --diff src/img2pose/
```

**Configuration:**

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311"]
include = '\.pyi?$'
extend-exclude = '''
/(
    \.git
    | \.venv
    | build
    | dist
)/
'''
```

### 3.3 Type Check with mypy

```bash
# Install
pip install mypy

# Run type check on new code
mypy src/img2pose/

# Strict mode for public API
mypy src/img2pose/detector.py --strict

# Generate type stubs report
mypy src/img2pose/ --txt-report mypy_report.txt
```

**Configuration:**

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
exclude = [
    "tests/",
    "build/",
]

[[tool.mypy.overrides]]
module = "img2pose.detector"
strict = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "img2pose._*"
disallow_untyped_defs = false
check_untyped_defs = true
```

### 3.4 Run Tests with pytest

```bash
# Install
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/img2pose --cov-fail-under=80

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run with parallel execution
pip install pytest-xdist
pytest tests/ -n auto

# Run with timeout (prevent hanging tests)
pip install pytest-timeout
pytest tests/ --timeout=60
```

**Configuration:**

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
markers = [
    "unit: Unit tests (no model required)",
    "integration: Integration tests (model required)",
    "equivalence: Numerical equivalence tests",
    "slow: Tests that take >10 seconds",
    "gpu: Tests requiring GPU",
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
]
```

### 3.5 Coverage with pytest-cov

```bash
# Run with coverage
pytest tests/ --cov=src/img2pose --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src/img2pose --cov-report=html
# Open htmlcov/index.html

# Fail if coverage below threshold
pytest tests/ --cov=src/img2pose --cov-fail-under=80

# Generate XML for CI integration
pytest tests/ --cov=src/img2pose --cov-report=xml
```

**Configuration:**

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/img2pose"]
branch = true
omit = [
    "*/__pycache__/*",
    "*/tests/*",
]

[tool.coverage.report]
show_missing = true
skip_covered = false
fail_under = 80
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]

[tool.coverage.html]
directory = "htmlcov"
```

---

## 4. CI Integration

### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install ruff
        run: pip install ruff
      - name: Run ruff
        run: ruff check src/img2pose/

  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install black
        run: pip install black
      - name: Check formatting
        run: black --check src/img2pose/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install mypy
      - name: Run mypy
        run: mypy src/img2pose/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache model weights
        uses: actions/cache@v4
        with:
          path: ~/.cache/img2pose
          key: img2pose-weights-v1

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run unit tests
        run: pytest tests/unit/ -v

      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          CUDA_VISIBLE_DEVICES: ""  # Force CPU

      - name: Run tests with coverage
        run: pytest tests/ --cov=src/img2pose --cov-report=xml --cov-fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: false
```

### 4.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
        files: ^src/img2pose/

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        files: ^src/img2pose/

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        files: ^src/img2pose/detector\.py$
        additional_dependencies: [torch, numpy, pillow]
```

Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

---

## 5. Local Development Workflow

### 5.1 Quick Check Commands

```bash
# One-liner to run all gates locally
ruff check src/img2pose/ && black --check src/img2pose/ && mypy src/img2pose/ && pytest tests/ --cov=src/img2pose --cov-fail-under=80
```

### 5.2 Makefile for Convenience

```makefile
# Makefile
.PHONY: lint format typecheck test coverage all

lint:
	ruff check src/img2pose/

format:
	black src/img2pose/

format-check:
	black --check src/img2pose/

typecheck:
	mypy src/img2pose/

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

coverage:
	pytest tests/ --cov=src/img2pose --cov-report=html --cov-fail-under=80
	@echo "Coverage report: htmlcov/index.html"

all: lint format-check typecheck test coverage
	@echo "All gates passed!"

fix:
	ruff check src/img2pose/ --fix
	black src/img2pose/
```

Usage:
```bash
make lint        # Run linter
make format      # Auto-format code
make test        # Run all tests
make coverage    # Run tests with coverage report
make all         # Run all gates
make fix         # Auto-fix lint and format issues
```

---

## 6. Coverage Reporting

### 6.1 Coverage Report Interpretation

```
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
src/img2pose/__init__.py                 5      0   100%
src/img2pose/detector.py               120     12    90%   45-48, 112-115
src/img2pose/_weights.py                65      8    88%   78-85
src/img2pose/_visualization.py          45      9    80%   62-70
src/img2pose/_model.py                  89     27    70%   45-60, 88-99
src/img2pose/_models.py                340    136    60%   ...
------------------------------------------------------------------
TOTAL                                  664    192    71%
```

### 6.2 Improving Coverage

Priority areas for coverage improvement:

1. **Error paths**: Ensure exception handlers are tested
2. **Edge cases**: Empty inputs, boundary values
3. **Configuration variants**: Different device, threshold combinations

```python
# Example: Testing error paths
def test_invalid_backbone_raises():
    """Test that invalid backbone depth raises ValueError."""
    with pytest.raises(ValueError, match="backbone_depth must be"):
        Img2Pose(backbone_depth=99)

def test_corrupted_weights_raises(mock_corrupted_weights):
    """Test that corrupted weights raise ModelLoadError."""
    from img2pose._exceptions import ModelLoadError
    with pytest.raises(ModelLoadError):
        Img2Pose(model_path=mock_corrupted_weights)
```

---

## 7. Gate Failure Remediation

### 7.1 Lint Failure

```bash
# View specific issues
ruff check src/img2pose/ --output-format=text

# Auto-fix safe issues
ruff check src/img2pose/ --fix

# Ignore specific rule (use sparingly)
ruff check src/img2pose/ --ignore E501
```

### 7.2 Format Failure

```bash
# View diff of required changes
black --check --diff src/img2pose/

# Auto-format
black src/img2pose/
```

### 7.3 Type Check Failure

```bash
# View detailed errors
mypy src/img2pose/ --show-error-codes

# Generate HTML report
mypy src/img2pose/ --html-report mypy_report

# Add type ignore for third-party issues
def some_func() -> int:
    return external_lib.call()  # type: ignore[no-any-return]
```

### 7.4 Test Failure

```bash
# Run failed test with more detail
pytest tests/test_failing.py -v --tb=long

# Run with pdb on failure
pytest tests/test_failing.py --pdb

# Run single test
pytest tests/test_detector.py::TestDetector::test_specific -v
```

### 7.5 Coverage Failure

```bash
# View uncovered lines
pytest tests/ --cov=src/img2pose --cov-report=term-missing

# Generate HTML for visual inspection
pytest tests/ --cov=src/img2pose --cov-report=html
# Open htmlcov/index.html

# Identify lowest coverage modules
coverage report --sort=cover
```

---

## 8. Summary

### 8.1 Gate Checklist for PRs

Before merging any PR:

- [ ] `ruff check src/img2pose/` passes with 0 errors
- [ ] `black --check src/img2pose/` passes (no formatting changes needed)
- [ ] `mypy src/img2pose/` passes with 0 errors
- [ ] `pytest tests/` passes with 100% test success
- [ ] `pytest tests/ --cov=src/img2pose --cov-fail-under=80` passes

### 8.2 Commands Summary

| Gate | Check Command | Fix Command |
|------|--------------|-------------|
| Lint | `ruff check src/img2pose/` | `ruff check src/img2pose/ --fix` |
| Format | `black --check src/img2pose/` | `black src/img2pose/` |
| Type | `mypy src/img2pose/` | Manual fixes |
| Test | `pytest tests/ -v` | Manual fixes |
| Coverage | `pytest --cov=src/img2pose --cov-fail-under=80` | Add more tests |

### 8.3 Dependencies Summary

```toml
[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "pytest-xdist>=2.0",  # Parallel test execution
    "pytest-timeout>=2.0",  # Test timeouts
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
```
