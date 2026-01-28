# img2pose

Real-time face detection with 6DoF pose estimation.

img2pose performs face detection and 6 degrees of freedom (6DoF) pose estimation without requiring prior face detection or facial landmark localization. Originally published at CVPR 2021.

## Installation

```bash
pip install img2pose
```

### Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision 0.8+

## Quick Start

```python
from img2pose import Img2Pose

# Initialize detector (downloads model automatically on first use)
detector = Img2Pose()

# Detect faces
faces = detector.detect_faces("image.jpg")

for face in faces:
    print(f"Box: {face['box']}")
    print(f"Confidence: {face['confidence']:.2f}")
    print(f"Pose (rx, ry, rz, tx, ty, tz): {face['pose']}")
    print(f"Keypoints: {face['keypoints']}")
```

## API Reference

### Img2Pose

```python
Img2Pose(
    device="auto",           # "auto", "cuda", "cpu", "cuda:0"
    score_threshold=0.5,     # Min confidence (0-1)
    max_faces=-1,            # -1 = unlimited
    model_path=None,         # Custom model weights path
    min_size=640,            # Min image dimension
    max_size=1400,           # Max image dimension
)
```

### detect_faces()

```python
faces = detector.detect_faces(image)
# or
faces = detector(image)  # Callable interface
```

**Input types:**
- File path (str or Path)
- PIL Image
- NumPy array (RGB uint8 HWC)
- List of any of the above (batch processing)

**Output format:**
```python
[
    {
        "box": [x1, y1, x2, y2],        # xyxy format
        "confidence": 0.95,              # 0-1 score
        "pose": [rx, ry, rz, tx, ty, tz], # 6DoF pose
        "keypoints": {
            "left_eye": [x, y],
            "right_eye": [x, y],
            "nose": [x, y],
            "mouth_left": [x, y],
            "mouth_right": [x, y]
        }
    },
    ...
]
```

### visualize()

```python
vis_image = detector.visualize("image.jpg")
# or with pre-computed results:
vis_image = detector.visualize(image, faces)
```

Returns a NumPy array (RGB uint8 HWC) with detections drawn.

## Input Formats

```python
# File path
faces = detector.detect_faces("path/to/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("image.jpg")
faces = detector.detect_faces(img)

# NumPy array (RGB uint8)
import numpy as np
img = np.array(Image.open("image.jpg"))
faces = detector.detect_faces(img)

# Batch processing
faces = detector.detect_faces(["img1.jpg", "img2.jpg", "img3.jpg"])
```

## Output Format (MTCNN Compatible)

The output follows [MTCNN](https://github.com/ipazc/mtcnn) and [RetinaFace](https://github.com/serengil/retinaface) conventions:

| Key | Format | Description |
|-----|--------|-------------|
| `box` | `[x1, y1, x2, y2]` | Bounding box (xyxy) |
| `confidence` | `float` | Detection confidence 0-1 |
| `pose` | `[rx, ry, rz, tx, ty, tz]` | 6DoF pose (rotation + translation) |
| `keypoints` | `dict` | 5 facial landmarks |

### Pose Format

The 6DoF pose consists of:
- **Rotation** (rx, ry, rz): Axis-angle rotation vector in radians
- **Translation** (tx, ty, tz): Translation in normalized coordinates

To convert to Euler angles:
```python
from scipy.spatial.transform import Rotation
pose = face["pose"]
euler = Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)
print(f"Pitch: {euler[0]:.1f}, Yaw: {euler[1]:.1f}, Roll: {euler[2]:.1f}")
```

## Device Selection

```python
# Auto-select (GPU if available, else CPU)
detector = Img2Pose(device="auto")

# Force CPU
detector = Img2Pose(device="cpu")

# Specific GPU
detector = Img2Pose(device="cuda:0")
```

## Configuration

### Score Threshold

```python
# High threshold for fewer, more confident detections
detector = Img2Pose(score_threshold=0.8)

# Low threshold for more detections
detector = Img2Pose(score_threshold=0.3)

# Override per call
faces = detector.detect_faces(image, score_threshold=0.9)
```

### Max Faces

```python
# Return at most 5 faces
detector = Img2Pose(max_faces=5)

# Override per call
faces = detector.detect_faces(image, max_faces=1)  # Top face only
```

### Custom Model Weights

```python
detector = Img2Pose(model_path="/path/to/custom_weights.pth")
```

## Model Caching

Models are automatically downloaded to `~/.cache/img2pose/` on first use.

Set custom cache directory:
```bash
export IMG2POSE_CACHE=/path/to/cache
```

## Visualization

```python
# Basic visualization
vis = detector.visualize("image.jpg")

# With options
vis = detector.visualize(
    image,
    show_box=True,
    show_keypoints=True,
    show_confidence=True,
    box_color=(0, 255, 0),      # Green boxes
    keypoint_color=(255, 0, 0),  # Red keypoints
    thickness=2,
)

# Save result
from PIL import Image
Image.fromarray(vis).save("output.jpg")
```

## Examples

### Face Detection with Pose

```python
from img2pose import Img2Pose
import numpy as np
from scipy.spatial.transform import Rotation

detector = Img2Pose()
faces = detector.detect_faces("selfie.jpg")

for i, face in enumerate(faces):
    # Get Euler angles
    euler = Rotation.from_rotvec(face["pose"][:3]).as_euler("xyz", degrees=True)

    print(f"Face {i+1}:")
    print(f"  Position: {face['box']}")
    print(f"  Pitch: {euler[0]:.1f}°")
    print(f"  Yaw: {euler[1]:.1f}°")
    print(f"  Roll: {euler[2]:.1f}°")
```

### Batch Processing

```python
import glob
from img2pose import Img2Pose

detector = Img2Pose()
images = glob.glob("photos/*.jpg")

# Process batch
all_faces = detector.detect_faces(images)

for img_path, faces in zip(images, all_faces):
    print(f"{img_path}: {len(faces)} faces")
```

### Filter by Pose

```python
from scipy.spatial.transform import Rotation

# Get frontal faces only (yaw < 15 degrees)
frontal_faces = []
for face in faces:
    euler = Rotation.from_rotvec(face["pose"][:3]).as_euler("xyz", degrees=True)
    if abs(euler[1]) < 15:  # Yaw angle
        frontal_faces.append(face)
```

## Citation

If you use img2pose in your research, please cite:

```bibtex
@inproceedings{albiero2021img2pose,
    title={img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation},
    author={Albiero, Vítor and Chen, Xingyu and Yin, Xi and Pang, Guan and Hassner, Tal},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2021}
}
```

## License

BSD-3-Clause License. See [LICENSE](../../license.md) for details.
