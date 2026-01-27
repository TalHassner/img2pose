# Changelog

All notable changes to the img2pose library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-27

### Added

- Initial release of img2pose as a pip-installable library
- `Img2Pose` class for face detection and 6DoF pose estimation
- `detect_faces()` method with MTCNN/RetinaFace-compatible output format
- Callable interface (`detector(image)`) matching Ultralytics YOLO pattern
- Support for multiple input types: file paths, PIL Images, numpy arrays
- Batch processing support via list inputs
- `visualize()` method for drawing detections on images
- Automatic model weight downloading and caching
- GPU/CPU auto-detection with configurable device selection
- Score threshold filtering
- Maximum faces limit
- Custom model path support

### Output Format

The library uses a standardized output format compatible with other face detection libraries:

```python
{
    "box": [x1, y1, x2, y2],        # xyxy format (like RetinaFace)
    "confidence": 0.95,             # matches MTCNN key name
    "pose": [rx, ry, rz, tx, ty, tz], # 6DoF (unique to img2pose)
    "keypoints": {                   # matches MTCNN format
        "left_eye": [x, y],
        "right_eye": [x, y],
        "nose": [x, y],
        "mouth_left": [x, y],
        "mouth_right": [x, y]
    }
}
```

### Technical Notes

- Based on the CVPR 2021 paper "img2pose: Face Alignment and Detection via 6DoF"
- Uses ResNet-18 FPN backbone for efficient inference
- Model weights are cached in `~/.cache/img2pose/` by default
- Supports PyTorch 1.7+ and Python 3.8+

### Migration from Research Repository

If migrating from the original research repository:

```python
# Old (research repo)
from img2pose import img2poseModel
model = img2poseModel(depth=18, ...)
load_model(model.fpn_model, "model.pth")
predictions = model.predict([tensor])

# New (library)
from img2pose import Img2Pose
detector = Img2Pose()
faces = detector.detect_faces("image.jpg")
```

Key differences:
- Simplified API with automatic model loading
- Accepts file paths and PIL Images directly (no manual tensor conversion)
- MTCNN-compatible output format with named keypoints
- Inference-only (training remains in research repository)
