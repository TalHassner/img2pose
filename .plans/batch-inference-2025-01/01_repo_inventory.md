# Repository Inventory: Batch Inference Enhancement

**Date:** 2025-01-27  
**Plan ID:** batch-inference-2025-01  
**Goal:** Add efficient batch inference to `Img2Pose.detect_faces()`

---

## 1. Call Graph for Batch Processing Path

The current batch processing path flows through these functions:

```
Img2Pose.detect_faces()    # /home/tal/dev/img2pose/src/img2pose/detector.py:212-249
    └── [SEQUENTIAL LOOP]  # Lines 243-247
         └── _detect_single()           # /home/tal/dev/img2pose/src/img2pose/detector.py:251-313
              ├── _load_image()         # /home/tal/dev/img2pose/src/img2pose/detector.py:29-59
              ├── _image_to_tensor()    # /home/tal/dev/img2pose/src/img2pose/detector.py:62-67
              ├── img2poseModel.predict()  # /home/tal/dev/img2pose/src/img2pose/_model.py:152-166
              │    └── FasterDoFRCNN.forward()  # via self.fpn_model(imgs)
              │         └── GeneralizedRCNN.forward()  # /home/tal/dev/img2pose/src/img2pose/_rcnn.py:49-95
              └── _project_landmarks()  # /home/tal/dev/img2pose/src/img2pose/detector.py:75-107
```

### Function Signatures

| Function | Location | Signature |
|----------|----------|-----------|
| `detect_faces` | detector.py:212 | `(self, image: Union[ImageInput, List[ImageInput]], score_threshold: Optional[float], max_faces: Optional[int]) -> Union[List[FaceDict], List[List[FaceDict]]]` |
| `_detect_single` | detector.py:251 | `(self, image: ImageInput, threshold: float, max_count: int) -> List[FaceDict]` |
| `_load_image` | detector.py:29 | `(image: ImageInput) -> Image.Image` |
| `_image_to_tensor` | detector.py:62 | `(image: Image.Image, device: torch.device) -> torch.Tensor` |
| `img2poseModel.predict` | _model.py:152 | `(self, imgs: List[Tensor]) -> List[Dict[str, Tensor]]` |
| `GeneralizedRCNN.forward` | _rcnn.py:49 | `(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]) -> List[Dict[str, Tensor]]` |

---

## 2. Native Batch Support Analysis

**Key Finding:** The underlying model already supports batch inference natively.

### Evidence from `_rcnn.py:GeneralizedRCNN.forward()` (lines 49-95)

```python
def forward(
    self,
    images: List[Tensor],  # <-- Already accepts List[Tensor]
    targets: Optional[List[Dict[str, Tensor]]] = None,
) -> List[Dict[str, Tensor]]:
```

The forward pass processes all images together:
- Line 69-73: Collects `original_image_sizes` for all images in the list
- Line 75: `self.transform(images, targets)` - transforms batch together
- Line 77: `self.backbone(images.tensors)` - backbone processes batched tensor
- Line 81-84: RPN and ROI heads process batch
- Line 85-87: Post-processes detections for all images

### Evidence from `_model.py:img2poseModel.predict()` (lines 152-166)

```python
def predict(self, imgs: List[Tensor]) -> List[Dict[str, Tensor]]:
    """Run inference on a list of image tensors.

    Args:
        imgs: List of image tensors [C, H, W] on the model's device

    Returns:
        List of detection dicts with keys: boxes, labels, scores, dofs
    """
```

The model wrapper simply passes the list through:
```python
with torch.no_grad():
    predictions = self.fpn_model(imgs)  # <-- Passes entire list
return predictions
```

### Evidence from `_models.py:DOFRoIHeads.postprocess_detections()` (lines 140-232)

The post-processing iterates over `image_shapes` (line 166-168), confirming batch handling:
```python
for boxes, dofs, scores, image_shape in zip(
    pred_boxes_list, pred_dofs_list, pred_scores_list, image_shapes
):
```

---

## 3. Current Sequential Implementation

### Location of the Sequential Loop

**File:** `/home/tal/dev/img2pose/src/img2pose/detector.py`  
**Lines 243-247:**

```python
# Handle batch input
if isinstance(image, list):
    return [
        self._detect_single(img, threshold, max_count)  # <-- Sequential loop
        for img in image
    ]
```

### Per-Image Operations in `_detect_single()` (lines 251-313)

| Operation | Lines | Description |
|-----------|-------|-------------|
| Image loading | 259 | `pil_image = _load_image(image)` |
| Dimension capture | 260 | `width, height = pil_image.size` |
| Tensor conversion | 263 | `img_tensor = _image_to_tensor(pil_image, self.device)` |
| Model inference | 266 | `predictions = self._model.predict([img_tensor])` |
| Result extraction | 270-274 | Extract boxes, scores, dofs from predictions |
| Landmark projection | 287-292 | `_project_landmarks(..., width, height)` |
| Face dict creation | 294-304 | Build output dictionaries |
| Sort and limit | 307-311 | Sort by confidence, apply max_count |

### Image Dimension Usage

The `width` and `height` values (line 260) are used in `_project_landmarks()` (lines 287-292):

```python
landmarks_2d = _project_landmarks(
    self._threed_68_points,
    dofs[i],
    width,   # <-- Per-image width needed
    height,  # <-- Per-image height needed
)
```

The `_project_landmarks()` function (lines 75-107) uses these for camera intrinsics:
```python
focal = image_width + image_height
intrinsics = np.array([
    [focal, 0, image_width / 2],
    [0, focal, image_height / 2],
    [0, 0, 1]
])
```

**Implication:** Post-processing must track which image dimensions correspond to which detection results.

---

## 4. Interfaces and Invariants That Must Be Preserved

### Type Aliases (`detector.py:20-21`)

```python
ImageInput = Union[str, Path, Image.Image, np.ndarray]
FaceDict = Dict[str, Any]
```

### `detect_faces()` Public API (lines 212-249)

**Signature must remain:**
```python
def detect_faces(
    self,
    image: Union[ImageInput, List[ImageInput]],
    score_threshold: Optional[float] = None,
    max_faces: Optional[int] = None,
) -> Union[List[FaceDict], List[List[FaceDict]]]:
```

**Return type invariants:**
- Single image input: Returns `List[FaceDict]`
- List input: Returns `List[List[FaceDict]]` (one list per image)

### `FaceDict` Structure (lines 299-304)

Each face dictionary must contain:
```python
{
    "box": [x1, y1, x2, y2],     # List[float], xyxy format
    "confidence": float,          # 0.0 to 1.0
    "pose": [rx, ry, rz, tx, ty, tz],  # List[float], 6DoF
    "keypoints": {
        "left_eye": [x, y],
        "right_eye": [x, y],
        "nose": [x, y],
        "mouth_left": [x, y],
        "mouth_right": [x, y],
    }
}
```

### `_detect_single()` Internal Contract (lines 251-256)

```python
def _detect_single(
    self,
    image: ImageInput,
    threshold: float,
    max_count: int,
) -> List[FaceDict]:
```

This function:
1. Handles all input types (str, Path, PIL.Image, np.ndarray)
2. Applies threshold filtering
3. Applies max_count limiting
4. Returns sorted list (by confidence, descending)

### `_project_landmarks()` Requirements (lines 75-107)

Requires per-image dimensions:
- `image_width: int` - needed for camera intrinsics
- `image_height: int` - needed for camera intrinsics

### Keypoint Indices (lines 133-140)

```python
_KEYPOINT_INDICES = {
    "left_eye": 36,
    "right_eye": 45,
    "nose": 30,
    "mouth_left": 48,
    "mouth_right": 54,
}
```

---

## 5. Test Structure

### Test File Locations

| File | Purpose |
|------|---------|
| `/home/tal/dev/img2pose/tests/unit/test_detector.py` | Unit tests for detector utilities |
| `/home/tal/dev/img2pose/tests/integration/test_inference.py` | Integration tests for inference |
| `/home/tal/dev/img2pose/tests/equivalence/test_numerical.py` | Numerical equivalence tests |
| `/home/tal/dev/img2pose/tests/conftest.py` | Shared pytest fixtures |

### Existing Test Coverage

**Unit Tests (`test_detector.py`):**
- `TestImg2PoseInit` - Device selection (lines 9-38)
- `TestImageLoading` - Image loading utilities (lines 41-92)
- `TestImageToTensor` - Tensor conversion (lines 95-118)
- `TestLandmarkProjection` - Landmark projection (lines 121-147)
- `TestFaceOutputFormat` - Output format validation (lines 150-193)

**Integration Tests (`test_inference.py`):**
- `TestImg2PoseInference` - End-to-end inference (lines 9-49)
- `TestVisualization` - Visualization module (lines 52-91)
- `TestWeightManagement` - Weight loading (lines 94-130)

**Equivalence Tests (`test_numerical.py`):**
- `TestPoseOperationsEquivalence` - Pose operations (lines 11-76)
- `TestProjectionEquivalence` - Projection functions (lines 79-118)
- `TestModelEquivalence` - Model output comparison (skipped, lines 121-143)

### Available Fixtures (`conftest.py`)

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_rgb_image` | PIL.Image | 640x480 white image |
| `sample_rgb_array` | np.ndarray | 480x640x3 gray image |
| `sample_face_dict` | dict | Sample face detection result |
| `pose_reference_68` | np.ndarray | 68-point 3D reference |
| `pose_reference_5` | np.ndarray | 5-point 3D reference |

### Gaps in Batch Testing Coverage

1. **No actual batch inference tests** - `test_batch_processing_returns_list_of_lists` (line 40-49) only tests mock data
2. **No variable-size batch tests** - Images of different dimensions in same batch
3. **No batch performance tests** - No verification that batch is faster than sequential
4. **No mixed-input-type batch tests** - e.g., mixing PIL and numpy in same batch

---

## 6. Warning Suppression

### Current Filters (`__init__.py:20-32`)

```python
# Suppress torch.meshgrid deprecation warning (internal to PyTorch RPN)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
    category=UserWarning,
)

# Suppress torchvision backbone_name positional argument warning
warnings.filterwarnings(
    "ignore",
    message="Using 'backbone_name' as positional parameter",
    category=UserWarning,
)
```

### Potential Additional Warnings

With batch processing, may need to suppress:
- DataParallel warnings for single-GPU multi-image batches
- Memory warnings for large batches
- Tensor size mismatch warnings from GeneralizedRCNNTransform (images of different sizes are padded)

---

## 7. Summary of Key Findings

### What Enables Batch Inference

1. **Model natively supports batches**: `GeneralizedRCNN.forward()` accepts `List[Tensor]`
2. **Wrapper passes through**: `img2poseModel.predict()` passes list directly
3. **Transform handles padding**: `GeneralizedRCNNTransform` handles variable-size images

### What Requires Modification

1. **Sequential loop** at detector.py:243-247 must be replaced
2. **Per-image dimensions** must be tracked for landmark projection
3. **Post-processing** must be adapted to handle batch results

### Minimal Change Strategy

1. Load all images and capture dimensions upfront
2. Convert all images to tensors at once
3. Single call to `self._model.predict(tensor_list)`
4. Post-process each result with corresponding image dimensions
5. Maintain `_detect_single()` for backward compatibility (optional)

### Performance Characteristics

- **Current**: N images = N model forward passes
- **After**: N images = 1 model forward pass (batched)
- **Memory**: Batch processing uses more GPU memory per call
- **Latency**: Single batch is faster than N sequential calls

---

## 8. Module Dependency Map

```
img2pose (public package)
├── __init__.py          # Exports Img2Pose, warning filters
└── detector.py          # Img2Pose class (TARGET FOR CHANGES)
    ├── _model.py        # img2poseModel wrapper
    │   └── _models.py   # FasterDoFRCNN, DOFRoIHeads
    │       ├── _rcnn.py # GeneralizedRCNN base
    │       └── _rpn.py  # Region Proposal Network
    ├── _weights.py      # Weight download/loading
    ├── _pose_ops.py     # Pose transformations
    └── _visualization.py # Optional visualization
```

**Files to Modify:** Only `/home/tal/dev/img2pose/src/img2pose/detector.py`

**Files That Must Not Change Their Interfaces:**
- `_model.py:img2poseModel.predict()` - stable
- `_rcnn.py:GeneralizedRCNN.forward()` - stable
- All type aliases and public API signatures in `detector.py`
