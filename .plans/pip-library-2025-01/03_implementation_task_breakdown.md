# Implementation Task Breakdown: img2pose pip-installable Library

**Version:** 1.0
**Date:** 2025-01-27
**Reference:** `02_design_spec.md`

---

## API Reference

The canonical API is defined in the main plan:
- **Method:** `detect_faces()` (not `detect()`)
- **Callable:** `detector(image)` as alias for `detect_faces()`
- **Output keys:** `"confidence"` (not `"score"`), `"keypoints"` dict (not `"landmarks_5pt"`)

---

## Overview

This document breaks down the implementation into ordered milestones with explicit file touch lists, done conditions, and risk mitigations.

**Total Estimated Effort:** 4-6 development days
**Risk Level:** Low-Medium (no changes to existing functionality)

---

## Milestone 0: Model Analysis and Conversion (One-time)

**Goal:** Analyze original model checkpoint and create lean production weights.

**Duration:** 0.5 day

### 0.1 File Touch List

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `scripts/convert_to_lean_model.py` | Strip optimizer state from checkpoint |

### 0.2 Steps

1. Download original `img2pose_v1.pth` from Model Zoo
2. Analyze checkpoint contents (check for optimizer state)
3. Create lean model with only `fpn_model` weights
4. Verify equivalence between original and lean model
5. Upload lean model for distribution

### 0.3 Done Conditions

- [ ] Lean model created (`models/img2pose_v1_lean.pth`)
- [ ] No optimizer state in lean model
- [ ] Inference results identical with original model
- [ ] Size reduction verified (if optimizer was present)

### 0.4 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Original model already lean | Medium | None | Script detects and reports "no conversion needed" |
| Equivalence test fails | Low | High | Compare layer-by-layer, ensure correct loading |

---

## Milestone 1: Package Scaffolding

**Goal:** Create the basic package structure without any functionality.

**Duration:** 0.5 day

### 1.1 File Touch List

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/img2pose/__init__.py` | Empty placeholder, exports `__version__` only |
| CREATE | `src/img2pose/utils/__init__.py` | Empty placeholder |
| CREATE | `pyproject.toml` | Package metadata and dependencies |
| CREATE | `setup.py` | Backward compatibility shim (calls pyproject.toml) |
| CREATE | `MANIFEST.in` | Include data files in sdist |

### 1.2 pyproject.toml Content

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "img2pose"
version = "1.0.0"
description = "Face detection and 6DoF head pose estimation"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Vitor Albiero", email = "vitor.albiero@gmail.com"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Image Recognition",
]
requires-python = ">=3.8"
dependencies = [
    "torch>=1.7.0",
    "torchvision>=0.8.0",
    "numpy>=1.19.0",
    "scipy>=1.5.0",
    "Pillow>=8.0.0",
    "opencv-python>=4.4.0",
]

[project.optional-dependencies]
visualization = ["matplotlib>=3.3.0"]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=22.0",
    "mypy>=0.900",
]

[project.urls]
Homepage = "https://github.com/vitoralbiero/img2pose"
Documentation = "https://github.com/vitoralbiero/img2pose#readme"
Repository = "https://github.com/vitoralbiero/img2pose"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"img2pose.data" = ["*.npy"]
```

### 1.3 Done Conditions

- [ ] `pip install -e .` succeeds from repo root
- [ ] `python -c "import img2pose; print(img2pose.__version__)"` prints `1.0.0`
- [ ] `pip show img2pose` displays correct metadata
- [ ] Original scripts still run: `python img2pose.py` imports successfully

### 1.4 Risks

| Risk | Mitigation |
|------|------------|
| Name collision with repo-root `img2pose.py` | src-layout ensures pip install takes precedence |
| Missing build dependencies | Test on clean virtualenv |

---

## Milestone 2: Copy and Adapt Core Modules

**Goal:** Copy inference code into package with internal imports.

**Duration:** 1 day

### 2.1 File Touch List

| Action | File | Source | Description |
|--------|------|--------|-------------|
| CREATE | `src/img2pose/_model.py` | `img2pose.py` | Adapted with relative imports |
| CREATE | `src/img2pose/_models.py` | `models.py` | Adapted with relative imports |
| CREATE | `src/img2pose/_generalized_rcnn.py` | `generalized_rcnn.py` | Minimal changes |
| CREATE | `src/img2pose/_rpn.py` | `rpn.py` | Minimal changes |
| CREATE | `src/img2pose/_model_loader.py` | `model_loader.py` | Remove training-only imports |
| CREATE | `src/img2pose/utils/pose_operations.py` | `utils/pose_operations.py` | Adapted imports |
| CREATE | `src/img2pose/utils/image_operations.py` | `utils/image_operations.py` | Minimal changes |
| CREATE | `src/img2pose/utils/face_align.py` | `utils/face_align.py` | Minimal changes |
| CREATE | `src/img2pose/data/` | `pose_references/` | Copy .npy files |

### 2.2 Import Adaptations

Original (`img2pose.py`):
```python
from model_loader import load_model
from models import FasterDoFRCNN
```

Adapted (`src/img2pose/_model.py`):
```python
from img2pose._model_loader import load_model
from img2pose._models import FasterDoFRCNN
```

Original (`models.py`):
```python
from generalized_rcnn import GeneralizedRCNN
from losses import fastrcnn_loss
from rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from utils.pose_operations import transform_pose_global_project_bbox
```

Adapted (`src/img2pose/_models.py`):
```python
from img2pose._generalized_rcnn import GeneralizedRCNN
# REMOVE: from losses import fastrcnn_loss (training only - guard with try/except or remove)
from img2pose._rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from img2pose.utils.pose_operations import transform_pose_global_project_bbox
```

### 2.3 Handling `losses.py` Import

The `losses.py` import in `models.py` is only used during training (in `DOFRoIHeads.forward()` when `targets is not None`). Options:

**Option A (Recommended): Conditional import**
```python
# In _models.py
def _get_fastrcnn_loss():
    try:
        from losses import fastrcnn_loss
        return fastrcnn_loss
    except ImportError:
        def _no_loss(*args, **kwargs):
            raise RuntimeError("Training requires losses.py - install with pip install .[training]")
        return _no_loss
```

**Option B: Remove training path entirely**
- Simplify `DOFRoIHeads.forward()` to only handle inference
- Cleaner but diverges more from original

**Decision:** Use Option A to maintain maximum compatibility.

### 2.4 Data File Copy

```bash
# Create data directory and copy reference files
mkdir -p src/img2pose/data/
cp pose_references/reference_3d_68_points_trans.npy src/img2pose/data/
cp pose_references/reference_3d_5_points_trans.npy src/img2pose/data/
```

### 2.5 Done Conditions

- [ ] All internal imports resolve: `python -c "from img2pose._model import img2poseModel"`
- [ ] Data files are bundled: `python -c "import importlib.resources; print(importlib.resources.files('img2pose.data'))"`
- [ ] Original repo files unchanged: `git diff img2pose.py models.py` shows no changes
- [ ] Import cycle test: No circular import errors

### 2.6 Risks

| Risk | Mitigation |
|------|------------|
| Circular imports | Test each module import independently |
| Missing files in package | Verify with `pip install . && pip show -f img2pose` |
| Data files not bundled | Test with `MANIFEST.in` and `package_data` |

---

## Milestone 3: Model Weight Management

**Goal:** Implement auto-download of model weights.

**Duration:** 1 day

### 3.1 File Touch List

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/img2pose/_weights.py` | Download manager with checksum validation |
| MODIFY | `src/img2pose/__init__.py` | Export `download_weights()` utility |

### 3.2 Weight Download Implementation

```python
# src/img2pose/_weights.py
"""Model weight management with automatic download."""

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional
import shutil

WEIGHTS_REGISTRY = {
    "img2pose_v1": {
        "model_url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/img2pose_v1.pth",
        "model_sha256": None,  # TODO: Calculate after first upload
        "pose_mean_url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/WIDER_train_pose_mean_v1.npy",
        "pose_stddev_url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/WIDER_train_pose_stddev_v1.npy",
    },
}

DEFAULT_MODEL = "img2pose_v1"

def get_cache_dir() -> Path:
    """Get cache directory, respecting environment variable."""
    env_dir = os.environ.get("IMG2POSE_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cache" / "img2pose"

def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download file with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    try:
        # Simple progress indicator
        print(f"{desc}: {url}")
        urllib.request.urlretrieve(url, tmp_dest)
        shutil.move(str(tmp_dest), str(dest))
        print(f"Saved to: {dest}")
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")

def get_model_paths(model_name: str = DEFAULT_MODEL) -> dict:
    """
    Get paths to model files, downloading if necessary.

    Returns dict with keys: model_path, pose_mean_path, pose_stddev_path
    """
    if model_name not in WEIGHTS_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(WEIGHTS_REGISTRY.keys())}")

    registry = WEIGHTS_REGISTRY[model_name]
    cache_dir = get_cache_dir() / model_name

    paths = {
        "model_path": cache_dir / "model.pth",
        "pose_mean_path": cache_dir / "pose_mean.npy",
        "pose_stddev_path": cache_dir / "pose_stddev.npy",
    }

    # Download missing files
    if not paths["model_path"].exists():
        download_file(registry["model_url"], paths["model_path"], "Downloading model weights")

    if not paths["pose_mean_path"].exists():
        download_file(registry["pose_mean_url"], paths["pose_mean_path"], "Downloading pose mean")

    if not paths["pose_stddev_path"].exists():
        download_file(registry["pose_stddev_url"], paths["pose_stddev_path"], "Downloading pose stddev")

    return paths
```

### 3.3 Done Conditions

- [ ] `from img2pose._weights import get_model_paths` works
- [ ] First call downloads files to `~/.cache/img2pose/`
- [ ] Second call reuses cached files (no download)
- [ ] Custom cache dir via `IMG2POSE_WEIGHTS_DIR` works
- [ ] Clear error message if download fails

### 3.4 Risks

| Risk | Mitigation |
|------|------------|
| GitHub rate limiting | Use releases API which has higher limits |
| Incomplete downloads | Use .tmp file and atomic move |
| Checksum mismatch | Verify SHA256 after download (deferred to v1.1) |
| Firewall blocking | Document manual download fallback |

---

## Milestone 4: Public API Implementation

**Goal:** Create the `Img2Pose` class with `detect()` method.

**Duration:** 1.5 days

### 4.1 File Touch List

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/img2pose/detector.py` | Main public API class |
| CREATE | `src/img2pose/_exceptions.py` | Custom exception classes |
| MODIFY | `src/img2pose/__init__.py` | Export `Img2Pose` class |

### 4.2 Detector Implementation Outline

```python
# src/img2pose/detector.py
"""Public API for face detection and pose estimation."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from img2pose._model import img2poseModel
from img2pose._model_loader import load_model
from img2pose._weights import get_model_paths
from img2pose._exceptions import Img2PoseError, InvalidInputError, ModelLoadError


class Img2Pose:
    """Face detection and 6DoF head pose estimation."""

    def __init__(
        self,
        *,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        max_faces: Optional[int] = None,
        backbone_depth: int = 18,
    ) -> None:
        # Validate backbone_depth
        if backbone_depth not in (18, 50, 101):
            raise ValueError(f"backbone_depth must be 18, 50, or 101, got {backbone_depth}")

        self._score_threshold = score_threshold
        self._max_faces = max_faces

        # Device selection
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Get model paths (auto-download if needed)
        if model_path is None:
            paths = get_model_paths()
            model_path = paths["model_path"]
            pose_mean_path = paths["pose_mean_path"]
            pose_stddev_path = paths["pose_stddev_path"]
        else:
            # User provided model_path, expect pose files alongside
            model_path = Path(model_path)
            pose_mean_path = model_path.parent / "pose_mean.npy"
            pose_stddev_path = model_path.parent / "pose_stddev.npy"

        # Load pose statistics
        pose_mean = np.load(pose_mean_path)
        pose_stddev = np.load(pose_stddev_path)

        # Load 3D reference points (bundled with package)
        threed_68_points = self._load_bundled_data("reference_3d_68_points_trans.npy")

        # Create model
        self._model = img2poseModel(
            depth=backbone_depth,
            min_size=640,
            max_size=1280,
            device=self._device,
            pose_mean=pose_mean,
            pose_stddev=pose_stddev,
            threed_68_points=threed_68_points,
        )

        # Load weights
        try:
            load_model(self._model.fpn_model_without_ddp, str(model_path),
                      cpu_mode=(self._device.type == "cpu"))
        except Exception as e:
            raise ModelLoadError(f"Failed to load model weights: {e}")

        self._model.evaluate()

        # Image transform
        self._transform = transforms.ToTensor()

    def _load_bundled_data(self, filename: str) -> np.ndarray:
        """Load .npy file bundled with package."""
        import importlib.resources
        with importlib.resources.files("img2pose.data").joinpath(filename).open("rb") as f:
            return np.load(f)

    def detect(
        self,
        images: Union[str, Path, Image.Image, np.ndarray, List],
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Detect faces and estimate head poses."""
        # Handle single vs batch input
        single_input = not isinstance(images, list)
        if single_input:
            images = [images]

        # Convert all inputs to tensors
        tensors = [self._to_tensor(img) for img in images]

        # Run inference
        with torch.no_grad():
            predictions = self._model.predict(tensors)

        # Convert to output format
        results = []
        for pred in predictions:
            faces = self._format_prediction(pred)
            results.append(faces)

        return results[0] if single_input else results

    def _to_tensor(self, image) -> torch.Tensor:
        """Convert various input formats to tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise InvalidInputError(f"Unsupported image type: {type(image)}")

        return self._transform(image).to(self._device)

    def _format_prediction(self, pred: Dict[str, torch.Tensor]) -> List[Dict]:
        """Format model output to public API format."""
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        poses = pred["dofs"].cpu().numpy()

        faces = []
        for i in range(len(boxes)):
            if scores[i] < self._score_threshold:
                continue

            faces.append({
                "box": boxes[i].astype(np.float32),
                "score": float(scores[i]),
                "pose": poses[i].astype(np.float32),
            })

        # Sort by score descending
        faces.sort(key=lambda x: x["score"], reverse=True)

        # Apply max_faces limit
        if self._max_faces is not None:
            faces = faces[:self._max_faces]

        return faces

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        self._score_threshold = value
```

### 4.3 Done Conditions

- [ ] `from img2pose import Img2Pose` works
- [ ] `detector = Img2Pose()` auto-downloads model and loads successfully
- [ ] `detector.detect("test.jpg")` returns list of face dicts
- [ ] `detector.detect([img1, img2])` returns list of lists
- [ ] Score threshold filtering works
- [ ] max_faces limiting works
- [ ] CPU fallback works when CUDA unavailable

### 4.4 Risks

| Risk | Mitigation |
|------|------------|
| Model takes too long to load | Document expected load time (5-10s first time) |
| Memory errors on large images | Document max recommended size |
| Unexpected input formats | Extensive input validation |

---

## Milestone 5: Visualization Support

**Goal:** Add optional 2D visualization of results.

**Duration:** 0.5 day

### 5.1 File Touch List

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/img2pose/_visualization.py` | Drawing utilities |
| MODIFY | `src/img2pose/detector.py` | Add `visualize()` method |

### 5.2 Visualization Implementation

```python
# src/img2pose/_visualization.py
"""Visualization utilities for detection results."""

from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


def draw_detections(
    image: Image.Image,
    faces: List[Dict],
    draw_boxes: bool = True,
    draw_landmarks: bool = False,  # Landmarks need additional computation
    box_color: Tuple[int, int, int] = (0, 255, 0),
    landmark_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """Draw detection results on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for face in faces:
        if draw_boxes:
            box = face["box"]
            draw.rectangle(
                [box[0], box[1], box[2], box[3]],
                outline=box_color,
                width=line_width
            )
            # Draw score
            score_text = f"{face['score']:.2f}"
            draw.text((box[0], box[1] - font_size - 2), score_text, fill=box_color)

    return img
```

### 5.3 Done Conditions

- [ ] `detector.visualize(image)` returns PIL Image with boxes
- [ ] Boxes drawn with correct coordinates
- [ ] Scores displayed above boxes
- [ ] Custom colors work

### 5.4 Risks

| Risk | Mitigation |
|------|------------|
| Font rendering issues | Use default PIL font, no external fonts |
| Large images slow to draw | Draw is O(n) in faces, acceptable |

---

## Milestone 6: Testing and Validation

**Goal:** Ensure numerical equivalence with original code.

**Duration:** 1 day

### 6.1 File Touch List

| Action | File | Description |
|--------|------|-------------|
| CREATE | `tests/__init__.py` | Test package |
| CREATE | `tests/test_detector.py` | Unit tests for Img2Pose |
| CREATE | `tests/test_equivalence.py` | Numerical equivalence tests |
| CREATE | `tests/conftest.py` | pytest fixtures |
| CREATE | `tests/data/` | Test images |

### 6.2 Equivalence Test Strategy

```python
# tests/test_equivalence.py
"""Test numerical equivalence with original img2pose code."""

import numpy as np
import torch
from PIL import Image

# Test that new API produces identical results to original
def test_equivalence_with_original():
    """Compare Img2Pose output with original img2poseModel."""
    # Load same image
    img = Image.open("tests/data/test_face.jpg")

    # Original code path (from repo root)
    import sys
    sys.path.insert(0, ".")  # Add repo root
    from img2pose import img2poseModel as OriginalModel
    from model_loader import load_model as original_load

    # ... run original model ...

    # New library path
    from img2pose import Img2Pose
    detector = Img2Pose()
    new_results = detector.detect(img)

    # Compare boxes within tolerance
    assert np.allclose(original_boxes, new_boxes, atol=1e-4)
    assert np.allclose(original_poses, new_poses, atol=1e-4)
```

### 6.3 Done Conditions

- [ ] `pytest tests/` passes
- [ ] Numerical equivalence within 1e-4 tolerance
- [ ] Tests run on CPU and GPU
- [ ] Edge cases tested: empty image, no faces, many faces

### 6.4 Risks

| Risk | Mitigation |
|------|------------|
| Numerical drift between implementations | Use identical code paths, only wrap |
| Test images too large to commit | Use small synthetic test images |
| CI GPU unavailable | Test CPU fallback in CI |

---

## Milestone 7: Documentation and Release

**Goal:** Update README and prepare for release.

**Duration:** 0.5 day

### 7.1 File Touch List

| Action | File | Description |
|--------|------|-------------|
| MODIFY | `README.md` | Add installation and quick start |
| CREATE | `CHANGELOG.md` | Release notes |
| MODIFY | `pyproject.toml` | Final version bump |

### 7.2 README Additions

```markdown
## Installation

```bash
pip install img2pose
```

For development:
```bash
git clone https://github.com/vitoralbiero/img2pose.git
cd img2pose
pip install -e ".[dev]"
```

## Quick Start

```python
from img2pose import Img2Pose

# Initialize detector (auto-downloads model on first use)
detector = Img2Pose()

# Detect faces in an image
results = detector.detect("photo.jpg")

for face in results:
    print(f"Box: {face['box']}")
    print(f"Score: {face['score']:.2f}")
    print(f"Pose (rx, ry, rz, tx, ty, tz): {face['pose']}")

# Visualize results
vis_image = detector.visualize("photo.jpg")
vis_image.save("output.jpg")
```
```

### 7.3 Done Conditions

- [ ] README has working installation instructions
- [ ] Quick start example runs successfully
- [ ] CHANGELOG.md documents v1.0.0 features
- [ ] `pip install .` from clean clone works

---

## Risk Summary and Rollback Strategy

### Overall Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing training code | Low | High | src-layout isolation, no original file changes |
| Model download failures | Medium | Medium | Clear error messages, manual download docs |
| Numerical differences | Low | High | Use identical code, extensive testing |
| Dependency conflicts | Medium | Low | Pin minimum versions, test across Python versions |

### Rollback Strategy

1. **If pip package breaks repo**: Delete `src/` directory, remove `pyproject.toml`
2. **If training code breaks**: Verify original files unchanged, revert any modifications
3. **If model download fails**: Document manual download URLs in error messages

### Pre-release Checklist

- [ ] All milestones complete
- [ ] Tests pass on Python 3.8, 3.9, 3.10, 3.11
- [ ] Tests pass on Linux, macOS, Windows
- [ ] Clean install works on fresh virtualenv
- [ ] Original training scripts still work
- [ ] README examples run without errors

---

## Milestone Summary

| # | Milestone | Duration | Dependencies |
|---|-----------|----------|--------------|
| 1 | Package Scaffolding | 0.5 day | None |
| 2 | Copy and Adapt Core Modules | 1 day | M1 |
| 3 | Model Weight Management | 1 day | M1 |
| 4 | Public API Implementation | 1.5 days | M2, M3 |
| 5 | Visualization Support | 0.5 day | M4 |
| 6 | Testing and Validation | 1 day | M4 |
| 7 | Documentation and Release | 0.5 day | M6 |
| **Total** | | **6 days** | |

### Critical Path

M1 -> M2 -> M4 -> M6 -> M7

M3 can be developed in parallel with M2.
M5 can be developed in parallel with M6.
