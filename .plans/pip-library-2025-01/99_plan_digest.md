# Plan Digest: img2pose Pip Library

**Plan ID:** pip-library-2025-01
**Status:** Ready for implementation

## Quick Summary

Convert img2pose research repository into a pip-installable Python library for face detection and 6DoF pose estimation. API compatible with MTCNN, RetinaFace, and Ultralytics YOLO.

## Milestones

| # | Milestone | Key Deliverables | Done When |
|---|-----------|------------------|-----------|
| 0 | Model Conversion | `scripts/convert_to_lean_model.py` | Lean model created, equivalence verified |
| 1 | Package Scaffolding | `pyproject.toml`, `src/img2pose/__init__.py` | `pip install -e .` works |
| 2 | Copy Core Modules | `_model.py`, `_models.py`, `_rpn.py`, `_rcnn.py`, `_pose_ops.py` | Imports succeed without training deps |
| 3 | Weight Management | `_weights.py` with auto-download | Lean model downloads to `~/.cache/img2pose/` |
| 4 | Public API | `detector.py` with `Img2Pose` class | `detector.detect_faces()` and `detector()` work |
| 5 | Visualization | `_visualization.py` | `detector.visualize()` draws boxes/landmarks |
| 6 | Testing | pytest suite, equivalence tests | 80%+ coverage, equivalence passes |
| 7 | Documentation | `src/img2pose/README.md`, `CHANGELOG.md` | `pip install .` from clean clone works |

## File Ownership

### New Files (CREATE)
```
scripts/convert_to_lean_model.py  # One-time model conversion
pyproject.toml
src/img2pose/__init__.py
src/img2pose/detector.py          # Public API: Img2Pose class
src/img2pose/_model.py            # img2poseModel wrapper
src/img2pose/_models.py           # FasterDoFRCNN, DOFRoIHeads
src/img2pose/_rpn.py              # RPN components
src/img2pose/_rcnn.py             # GeneralizedRCNN
src/img2pose/_pose_ops.py         # Pose operations
src/img2pose/_weights.py          # Model download/cache
src/img2pose/_visualization.py    # 2D visualization
src/img2pose/data/                # Bundled .npy files
src/img2pose/README.md            # Library documentation
tests/unit/*.py
tests/integration/*.py
tests/equivalence/*.py
CHANGELOG.md
```

### Preserved Files (NO CHANGE)
```
README.md                         # Repo README unchanged
img2pose.py                       # Original, still works
models.py, rpn.py, etc.           # Training scripts use these
train.py, evaluation/*.py         # Unchanged
```

## Public API

```python
from img2pose import Img2Pose

detector = Img2Pose(device="auto", score_threshold=0.5)

# MTCNN/RetinaFace style
faces = detector.detect_faces("image.jpg")

# YOLO style callable
faces = detector("image.jpg")

# Output format (MTCNN-compatible)
# [{"box": [x1,y1,x2,y2], "confidence": 0.95, "pose": [...], "keypoints": {...}}]
```

## Test Gates

```bash
# All must pass
make lint          # ruff check
make format-check  # black --check
make typecheck     # mypy src/img2pose
make test          # pytest with 80%+ coverage
make test-equiv    # Equivalence tests
```

## Definition of Done

1. `pip install img2pose` works (or `pip install .` from source)
2. `Img2Pose().detect_faces("image.jpg")` returns faces
3. `Img2Pose()("image.jpg")` callable interface works
4. Output format matches MTCNN: `{"box", "confidence", "keypoints", "pose"}`
5. GPU auto-selected when available, CPU fallback works
6. Results match original code on benchmark (within tolerance)
7. Original repo files unchanged
8. 80%+ test coverage on new code

## Commands to Verify

```bash
# After implementation
pip install -e .
python -c "
from img2pose import Img2Pose
d = Img2Pose()
# faces = d.detect_faces('test.jpg')
# faces = d('test.jpg')
print('API OK')
"
pytest --cov=src/img2pose --cov-fail-under=80
```
