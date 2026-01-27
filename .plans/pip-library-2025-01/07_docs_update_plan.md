# Documentation Update Plan

## Overview

This plan specifies exactly which documentation files to create for the img2pose pip-installable library.

**Important:** No existing files are modified. All documentation is new and lives within the package.

## Files to Create

### 1. src/img2pose/README.md (CREATE)

**Location:** `/home/tal/dev/img2pose/src/img2pose/README.md`

**Purpose:** Self-contained library documentation within the package.

**Content:**

```markdown
# img2pose - Face Detection and 6DoF Pose Estimation

A pip-installable library for face detection and 6DoF head pose estimation.

API compatible with [MTCNN](https://github.com/ipazc/mtcnn), [RetinaFace](https://github.com/serengil/retinaface), and [Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/).

## Installation

```bash
pip install img2pose
# Or from source:
pip install -e .
```

## Quick Start

```python
from img2pose import Img2Pose

# Initialize detector (auto-downloads model on first use)
detector = Img2Pose()

# Detect faces - MTCNN/RetinaFace style method name
faces = detector.detect_faces("path/to/image.jpg")

# Or use YOLO-style callable
faces = detector("path/to/image.jpg")

# Batch processing
faces_batch = detector.detect_faces(["img1.jpg", "img2.jpg"])

for face in faces:
    print(f"Box: {face['box']}")              # [x1, y1, x2, y2]
    print(f"Confidence: {face['confidence']}")  # 0.0-1.0
    print(f"Pose: {face['pose']}")            # [rx, ry, rz, tx, ty, tz]
    print(f"Keypoints: {face['keypoints']}")  # dict with left_eye, right_eye, nose, mouth_left, mouth_right

# Visualization
vis_image = detector.visualize("image.jpg")
vis_image.save("output.jpg")
```

## Output Format (MTCNN-compatible)

Each detected face is a dictionary:

```python
{
    "box": [x1, y1, x2, y2],           # Bounding box (xyxy format)
    "confidence": 0.95,                 # Detection confidence
    "pose": [rx, ry, rz, tx, ty, tz],  # 6DoF head pose (unique to img2pose)
    "keypoints": {                      # 5-point landmarks (MTCNN format)
        "left_eye": [x, y],
        "right_eye": [x, y],
        "nose": [x, y],
        "mouth_left": [x, y],
        "mouth_right": [x, y]
    }
}
```

## Constructor Options

```python
Img2Pose(
    device="auto",           # "auto", "cuda", "cpu", "cuda:0"
    score_threshold=0.5,     # Minimum detection confidence
    max_faces=-1             # Max faces per image (-1 = unlimited)
)
```

## Device Selection

- `"auto"` (default): Uses CUDA if available, else CPU
- `"cuda"`: Forces GPU (raises error if unavailable)
- `"cuda:0"`, `"cuda:1"`: Specific GPU
- `"cpu"`: Forces CPU execution

## Model Weights

On first use, model weights are automatically downloaded to `~/.cache/img2pose/`.
```

### 2. CHANGELOG.md (CREATE)

**Location:** `/home/tal/dev/img2pose/CHANGELOG.md`

**Content:**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Added
- Pip-installable package structure (`pip install img2pose`)
- `Img2Pose` class with `detect_faces()` method (MTCNN-compatible API)
- Callable interface `detector(image)` (YOLO-compatible)
- Automatic lean model weight downloading
- `visualize()` method for 2D visualization of detections
- Batch image processing support
- Type hints throughout public API
- pytest test suite with 80%+ coverage

### API Compatibility
- Method name `detect_faces()` matches MTCNN and RetinaFace
- Callable `detector(image)` matches Ultralytics YOLO
- Output format with `confidence` and `keypoints` dict matches MTCNN
- `pose` key provides 6DoF head pose (unique to img2pose)

### Notes
- Original repository structure preserved (training, evaluation scripts unchanged)
- Library code is self-contained in `src/img2pose/`
```

## Files NOT Modified

The following existing files remain **unchanged**:
- `README.md` (repo root) - Research documentation preserved as-is
- `train.py`, `evaluation/`, etc. - Training/evaluation code untouched
- All original Python modules at repo root

## Documentation Conventions

1. **Self-contained**: Library docs live within `src/img2pose/`
2. **API compatibility**: Document MTCNN/RetinaFace/YOLO alignment
3. **Examples**: Include runnable code snippets
4. **No duplication**: Don't repeat existing repo documentation
