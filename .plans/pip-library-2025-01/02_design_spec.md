# Design Specification: img2pose pip-installable Library

**Version:** 1.0
**Date:** 2025-01-27
**Status:** Draft

---

## 1. Package Structure

### 1.1 Recommended Layout: src-layout

The src-layout is recommended for pip packages to avoid import confusion during development.

```
img2pose/                           # Repository root (unchanged)
|-- src/
|   `-- img2pose/                   # NEW: Package source
|       |-- __init__.py             # Public API exports
|       |-- detector.py             # NEW: Img2Pose class (public API)
|       |-- _model.py               # Renamed from img2pose.py (internal)
|       |-- _models.py              # Renamed from models.py (internal)
|       |-- _generalized_rcnn.py    # Renamed from generalized_rcnn.py (internal)
|       |-- _rpn.py                 # Renamed from rpn.py (internal)
|       |-- _model_loader.py        # Renamed from model_loader.py (internal)
|       |-- _weights.py             # NEW: Model weight management
|       |-- _visualization.py       # NEW: Optional 2D visualization
|       |-- utils/
|       |   |-- __init__.py
|       |   |-- pose_operations.py  # Copied from utils/pose_operations.py
|       |   |-- image_operations.py # Copied from utils/image_operations.py
|       |   `-- face_align.py       # Copied from utils/face_align.py
|       `-- data/                   # Bundled data files
|           |-- reference_3d_68_points_trans.npy
|           `-- reference_3d_5_points_trans.npy
|-- pyproject.toml                  # NEW: Package metadata
|-- setup.py                        # NEW: Backward compatibility shim
|-- MANIFEST.in                     # NEW: Include data files
|-- img2pose.py                     # PRESERVED: Original file (for training scripts)
|-- models.py                       # PRESERVED: Original file
|-- generalized_rcnn.py             # PRESERVED: Original file
|-- rpn.py                          # PRESERVED: Original file
|-- model_loader.py                 # PRESERVED: Original file
|-- train.py                        # PRESERVED: Unchanged
|-- ... (all other original files)  # PRESERVED: Unchanged
```

### 1.2 Rationale for src-layout

1. **Clear separation**: `src/img2pose/` is the installable package; root files remain for research use
2. **Avoid import shadowing**: Running `python` from repo root imports the package, not local files
3. **Backward compatibility**: All existing scripts continue to work with relative imports
4. **Clean pip install**: Only `src/img2pose/` gets installed, training code stays out

### 1.3 File Naming Convention

- Public modules: Normal names (e.g., `detector.py`, `utils/`)
- Internal modules: Underscore prefix (e.g., `_model.py`, `_rpn.py`)
- This signals to users which APIs are stable vs implementation details

---

## 2. Public API Design

### 2.1 Main Class: `Img2Pose`

**File:** `src/img2pose/detector.py`

```python
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import torch
from PIL import Image

class Img2Pose:
    """
    Face detection and 6DoF head pose estimation.

    API designed to match MTCNN, RetinaFace, and Ultralytics YOLO conventions.

    Example:
        >>> detector = Img2Pose()  # Auto-downloads weights
        >>> results = detector.detect_faces("photo.jpg")  # MTCNN-style method name
        >>> results = detector("photo.jpg")  # YOLO-style callable
        >>> for face in results:
        ...     print(f"Box: {face['box']}, Confidence: {face['confidence']}")
    """

    def __init__(
        self,
        *,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        max_faces: Optional[int] = None,
        backbone_depth: int = 18,
    ) -> None:
        """
        Initialize the Img2Pose detector.

        Args:
            model_path: Path to .pth weights file. If None, downloads default model.
            device: Device string ("cuda", "cuda:0", "cpu"). If None, auto-selects.
            score_threshold: Minimum confidence score for detections (0.0-1.0).
            max_faces: Maximum number of faces to return per image. None = no limit.
            backbone_depth: ResNet backbone depth (18, 50, or 101). Default 18.

        Raises:
            ValueError: If backbone_depth not in (18, 50, 101).
            RuntimeError: If model weights cannot be loaded.
        """
        ...

    def detect_faces(
        self,
        images: Union[
            str, Path, Image.Image, np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]]
        ],
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Detect faces and estimate head poses.

        Method name matches MTCNN and RetinaFace conventions.

        Args:
            images: Single image or list of images. Accepts:
                - File path (str or Path)
                - PIL Image
                - NumPy array (H, W, 3) in RGB, uint8 [0-255]
                - List of any of the above for batch processing

        Returns:
            For single image: List of face dictionaries.
            For batch: List of lists (one list per image).

            Each face dictionary contains (matching MTCNN format):
                - "box": np.ndarray[4] - (x1, y1, x2, y2) in xyxy format
                - "confidence": float - Detection confidence (0.0-1.0)
                - "pose": np.ndarray[6] - (rx, ry, rz, tx, ty, tz) in radians/pixels
                - "keypoints": dict - 5 facial landmarks with named keys:
                    {"left_eye": [x, y], "right_eye": [x, y], "nose": [x, y],
                     "mouth_left": [x, y], "mouth_right": [x, y]}

        Example:
            >>> results = detector.detect_faces("group_photo.jpg")
            >>> print(f"Found {len(results)} faces")
            >>> for face in results:
            ...     print(f"Confidence: {face['confidence']:.2f}, Pose: {face['pose']}")
        """
        ...

    def __call__(
        self,
        images: Union[
            str, Path, Image.Image, np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]]
        ],
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Callable interface for detection (matches Ultralytics YOLO pattern).

        Equivalent to detect_faces().
        """
        return self.detect_faces(images)

    def visualize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        results: Optional[List[Dict]] = None,
        *,
        draw_boxes: bool = True,
        draw_landmarks: bool = True,
        draw_pose_axes: bool = False,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        landmark_color: Tuple[int, int, int] = (255, 0, 0),
        line_width: int = 2,
    ) -> Image.Image:
        """
        Visualize detection results on an image.

        Args:
            image: Input image (same formats as detect()).
            results: Detection results from detect(). If None, runs detection.
            draw_boxes: Draw bounding boxes.
            draw_landmarks: Draw 5-point facial landmarks.
            draw_pose_axes: Draw 3D pose axes (requires matplotlib).
            box_color: RGB color for bounding boxes.
            landmark_color: RGB color for landmarks.
            line_width: Line width for drawing.

        Returns:
            PIL Image with visualizations overlaid.
        """
        ...

    @property
    def device(self) -> torch.device:
        """The device the model is running on."""
        ...

    @property
    def score_threshold(self) -> float:
        """Current score threshold for filtering detections."""
        ...

    @score_threshold.setter
    def score_threshold(self, value: float) -> None:
        """Set score threshold (0.0-1.0)."""
        ...
```

### 2.2 Return Type Specification

Output format matches MTCNN/RetinaFace conventions for interoperability.

```python
# Type alias for face detection result
FaceResult = Dict[str, Union[np.ndarray, float, Dict]]

# Concrete structure (MTCNN-compatible):
{
    "box": np.ndarray,       # shape (4,), dtype float32, format (x1, y1, x2, y2) xyxy
    "confidence": float,     # range [0.0, 1.0], matches MTCNN key name
    "pose": np.ndarray,      # shape (6,), dtype float32, format (rx, ry, rz, tx, ty, tz)
    "keypoints": {           # dict format matches MTCNN exactly
        "left_eye": [x, y],
        "right_eye": [x, y],
        "nose": [x, y],
        "mouth_left": [x, y],
        "mouth_right": [x, y],
    }
}
```

### 2.3 Package Exports (`__init__.py`)

```python
"""
img2pose - Face Detection and 6DoF Head Pose Estimation

API compatible with MTCNN, RetinaFace, and Ultralytics YOLO.

Example:
    >>> from img2pose import Img2Pose
    >>> detector = Img2Pose()
    >>> results = detector.detect_faces("photo.jpg")  # MTCNN-style
    >>> results = detector("photo.jpg")  # YOLO-style callable
"""

from img2pose.detector import Img2Pose

__version__ = "1.0.0"
__all__ = ["Img2Pose", "__version__"]
```

---

## 3. Model Weight Management

### 3.1 Strategy: Auto-download from GitHub Releases

**File:** `src/img2pose/_weights.py`

```python
"""
Model weight management with automatic download.

Default weights location: ~/.cache/img2pose/
"""

WEIGHTS_REGISTRY = {
    "img2pose_v1": {
        "url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/img2pose_v1.pth",
        "sha256": "<checksum>",
        "size_mb": 85,
        "pose_mean_url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/WIDER_train_pose_mean_v1.npy",
        "pose_stddev_url": "https://github.com/vitoralbiero/img2pose/releases/download/v1.0.0/WIDER_train_pose_stddev_v1.npy",
    },
}

DEFAULT_MODEL = "img2pose_v1"
CACHE_DIR = Path.home() / ".cache" / "img2pose"

def get_weights_path(model_name: str = DEFAULT_MODEL, force_download: bool = False) -> Path:
    """
    Get path to model weights, downloading if necessary.

    Downloads to ~/.cache/img2pose/ on first use.
    Validates checksum after download.
    """
    ...

def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None) -> None:
    """
    Download file with progress bar and checksum validation.

    Uses requests library with streaming for large files.
    Shows tqdm progress bar if available.
    """
    ...
```

### 3.2 Weight Location Priority

1. Explicit `model_path` argument (user-provided)
2. Environment variable `IMG2POSE_WEIGHTS_DIR`
3. Default cache: `~/.cache/img2pose/`

### 3.3 Bundled vs Downloaded Files

| File Type | Strategy | Rationale |
|-----------|----------|-----------|
| `reference_3d_*.npy` | **Bundle** with package | Small (< 100KB), required for all inference |
| `pose_mean.npy`, `pose_stddev.npy` | **Download** with model | Tied to specific model version |
| Model weights `.pth` | **Download** on first use | Large (~85MB), versioned |

---

## 4. Backward Compatibility Approach

### 4.1 Guarantee

**All existing code in the repository continues to work unchanged.**

### 4.2 Implementation Strategy

1. **Preserve original files**: Do NOT modify `img2pose.py`, `models.py`, etc. at repo root
2. **Copy to src/**: The pip package gets copies of these files with minor adaptations
3. **Shim imports**: Original files can be imported directly for training scripts

### 4.3 Import Path Resolution

| Use Case | Import Path | Resolved To |
|----------|-------------|-------------|
| Training scripts (repo root) | `from img2pose import img2poseModel` | `./img2pose.py` |
| Pip-installed library | `from img2pose import Img2Pose` | `site-packages/img2pose/detector.py` |
| Internal package use | `from img2pose._model import img2poseModel` | `site-packages/img2pose/_model.py` |

### 4.4 Dual-mode Operation

When developing in the repo:
- `python train.py` uses local `img2pose.py`, `models.py`, etc.
- `from img2pose import Img2Pose` uses the installed package (if pip installed in editable mode)

The separation is achieved by:
1. NOT having `__init__.py` at repo root for `img2pose.py`
2. Having proper package structure in `src/img2pose/`

---

## 5. Bloat Control Analysis

### 5.1 Files Exceeding 500 LOC Threshold

| File | Lines | Analysis | Decision |
|------|-------|----------|----------|
| `models.py` | 568 | Contains `FasterDoFRCNN` (149 LOC), `DOFRoIHeads` (347 LOC), predictors (68 LOC) | **Keep as-is** |
| `rpn.py` | 539 | Contains `AnchorGenerator` (156 LOC), `RPNHead` (32 LOC), `RegionProposalNetwork` (284 LOC), utils (67 LOC) | **Keep as-is** |

### 5.2 Rationale for Not Splitting

1. **Tight coupling**: Classes share internal state and are always used together
2. **Torchvision alignment**: Structure mirrors torchvision detection modules
3. **Single responsibility**: Each file handles one layer of the detection pipeline
4. **No clear split point**: Splitting would create artificial boundaries

### 5.3 Refactor Triggers (Future)

Consider splitting if:
- File exceeds 800 LOC
- New features require adding >100 LOC to a single file
- Unit test coverage requires isolating specific components
- Performance profiling identifies hot paths to optimize separately

### 5.4 LOC Budget for New Code

| New File | Estimated LOC | Purpose |
|----------|---------------|---------|
| `detector.py` | 150-200 | Public API wrapper |
| `_weights.py` | 80-100 | Download management |
| `_visualization.py` | 100-150 | Optional drawing utilities |
| **Total new code** | **330-450** | Under 500 LOC budget |

---

## 6. Dependency Specification

### 6.1 Runtime Dependencies (Minimal)

```toml
[project]
dependencies = [
    "torch>=1.7.0",
    "torchvision>=0.8.0",
    "numpy>=1.19.0",
    "scipy>=1.5.0",
    "Pillow>=8.0.0",
    "opencv-python>=4.4.0",
]
```

### 6.2 Optional Dependencies

```toml
[project.optional-dependencies]
visualization = [
    "matplotlib>=3.3.0",
]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=22.0",
    "mypy>=0.900",
]
training = [
    "tensorboard>=2.4.0",
    "lmdb>=1.0.0",
    "scikit-image>=0.18.0",
    "tqdm>=4.50.0",
]
```

---

## 7. Error Handling Strategy

### 7.1 Exception Hierarchy

```python
class Img2PoseError(Exception):
    """Base exception for img2pose errors."""
    pass

class ModelLoadError(Img2PoseError):
    """Raised when model weights cannot be loaded."""
    pass

class DownloadError(Img2PoseError):
    """Raised when model download fails."""
    pass

class InvalidInputError(Img2PoseError):
    """Raised when input image format is invalid."""
    pass
```

### 7.2 Graceful Degradation

- If GPU unavailable: Fall back to CPU with warning
- If download fails: Provide clear error with manual download instructions
- If image format unknown: Try PIL.Image.open() before raising error

---

## 8. Thread Safety and Performance

### 8.1 Thread Safety

- `Img2Pose` instances are **NOT thread-safe** (model state is shared)
- For multi-threaded use: Create one instance per thread OR use process-based parallelism
- Batch processing via `detect([img1, img2, ...])` is preferred over threading

### 8.2 Memory Management

- Model is loaded once and kept in memory
- Input images are converted to tensors, processed, then released
- Results are returned as numpy arrays (CPU memory)
- GPU tensors are freed after each `detect()` call

---

## 9. Versioning Strategy

### 9.1 Semantic Versioning

- **Major (X.0.0)**: Breaking API changes
- **Minor (1.X.0)**: New features, backward compatible
- **Patch (1.0.X)**: Bug fixes only

### 9.2 Initial Release

- Version `1.0.0`: First stable release with core API
- Pre-release: `0.9.x` for testing

### 9.3 Model Versioning

Model weights are versioned separately:
- `img2pose_v1.pth` - Original WIDER-trained model
- Future models: `img2pose_v2.pth`, etc.
- Library supports multiple model versions via `model_path` argument

---

## 10. Configuration Schema

### 10.1 No Config Files Required

The library is designed to work with sensible defaults and explicit arguments:
- No YAML/JSON config files
- No environment variables required (optional override only)
- All settings via constructor arguments

### 10.2 Environment Variable Overrides (Optional)

| Variable | Purpose | Default |
|----------|---------|---------|
| `IMG2POSE_WEIGHTS_DIR` | Custom weights directory | `~/.cache/img2pose/` |
| `IMG2POSE_DEVICE` | Force device selection | Auto-detect |

---

## 11. Integration Points

### 11.1 Input Formats Supported

```python
# All valid input formats for detect_faces() and __call__:
detector.detect_faces("path/to/image.jpg")     # String path
detector.detect_faces(Path("path/to/image.jpg"))  # pathlib.Path
detector.detect_faces(Image.open("image.jpg")) # PIL Image
detector.detect_faces(cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB))  # NumPy RGB
detector.detect_faces([img1, img2, img3])      # Batch of any above

# YOLO-style callable (equivalent to detect_faces):
detector("path/to/image.jpg")
detector([img1, img2, img3])
```

### 11.2 Output Integration

Results are standard Python/NumPy types for easy integration:
- Bounding boxes: `np.ndarray` compatible with OpenCV, matplotlib
- Poses: `np.ndarray` compatible with scipy.spatial.transform.Rotation
- Images from visualize(): PIL Image, saveable or displayable

---

## 12. Documentation Plan

### 12.1 Docstrings

- All public methods: Google-style docstrings with Args, Returns, Raises, Example
- Type hints: Full type annotations for static analysis

### 12.2 Library Documentation

- Create `src/img2pose/README.md` with installation and usage instructions
- Existing repo README remains unchanged
- Library documentation is self-contained within the package

### 12.3 API Reference (Future)

- Generate with Sphinx + autodoc
- Host on Read the Docs or GitHub Pages
