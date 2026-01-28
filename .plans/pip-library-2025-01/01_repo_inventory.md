# Repository Inventory: img2pose pip-installable Library Conversion

**Generated:** 2025-01-27  
**Scope:** Inference path only (not training)

---

## 1. Relevant Entrypoints and Call Graph (Inference Path)

### Primary Entry Point: `img2poseModel` class

**File:** `/home/tal/dev/img2pose/img2pose.py`

#### Constructor Signature (`img2pose.py:19-38`)
```python
class img2poseModel:
    def __init__(
        self,
        depth,                          # ResNet depth (18, 50, 101)
        min_size,                       # Minimum image size
        max_size,                       # Maximum image size  
        model_path=None,                # Path to pretrained weights
        device=None,                    # torch device
        pose_mean=None,                 # Pose normalization mean
        pose_stddev=None,               # Pose normalization stddev
        distributed=False,              # Distributed training flag
        gpu=0,                          # GPU index
        threed_68_points=None,          # 68 facial landmarks (3D reference)
        threed_5_points=None,           # 5-point alignment landmarks
        rpn_pre_nms_top_n_test=6000,    # RPN parameter
        rpn_post_nms_top_n_test=1000,   # RPN parameter
        bbox_x_factor=1.1,              # Bounding box X expansion
        bbox_y_factor=1.1,              # Bounding box Y expansion
        expand_forehead=0.3,            # Forehead expansion factor
    ):
```

#### Inference Method: `predict()` (`img2pose.py:131-137`)
```python
def predict(self, imgs):
    assert self.fpn_model.training is False  # Line 132
    with torch.no_grad():                     # Line 134
        predictions = self.run_model(imgs)    # Line 135 -> calls fpn_model
    return predictions                        # Line 137
```

### Call Graph for Inference

```
img2poseModel.predict() [img2pose.py:131]
    |
    +-> run_model(imgs) [img2pose.py:121-124]
         |
         +-> fpn_model(imgs, targets=None) 
              |
              +-> FasterDoFRCNN.forward() [models.py:70, inherits from GeneralizedRCNN]
                   |
                   +-> GeneralizedRCNN.forward() [generalized_rcnn.py:40-121]
                        |
                        +-> self.transform(images, targets) [Line 79]
                        |    - GeneralizedRCNNTransform (torchvision)
                        |
                        +-> self.backbone(images.tensors) [Line 98]
                        |    - resnet_fpn_backbone (torchvision)
                        |
                        +-> self.rpn(images, features, targets) [Line 101]
                        |    - RegionProposalNetwork [rpn.py:255-539]
                        |    - Returns: proposals, proposal_losses
                        |
                        +-> self.roi_heads(features, proposals, image_sizes, targets) [Line 102-103]
                        |    - DOFRoIHeads [models.py:221-568]
                        |    - In inference mode (targets=None): returns detections
                        |
                        +-> self.transform.postprocess() [Line 105-107]
                             - Rescales detections to original image size
```

### Detailed Flow in `DOFRoIHeads.forward()` (Inference Branch)

**File:** `/home/tal/dev/img2pose/models.py:461-568`

When `targets is None` (inference mode), the flow is:
1. **Feature extraction** (`models.py:524-529`):
   - `box_roi_pool(features, proposals, image_shapes)` 
   - `box_head(box_features)` - TwoMLPHead
   - `box_predictor(box_features)` - FastRCNNDoFPredictor returns 6DoF poses
   - `class_predictor(class_features)` - FastRCNNClassPredictor returns class scores

2. **Post-processing** (`models.py:554-566`):
   - `postprocess_detections()` called at line 554
   - Returns dict with: `boxes`, `labels`, `scores`, `dofs`

### Key Postprocessing: `postprocess_detections()` (`models.py:367-459`)

This function:
1. Applies softmax to class logits (`models.py:382`)
2. Filters low-scoring boxes (`models.py:411`)
3. Calls `transform_pose_global_project_bbox()` (`models.py:429-439`)
4. Applies batched NMS (`models.py:442`)
5. Returns final boxes, dofs, scores, labels

---

## 2. Existing Interfaces and Invariants to Preserve

### 2.1 `img2poseModel` Constructor (`img2pose.py:19-38`)

**Must preserve:**
- All parameter names and their defaults
- Device auto-detection logic (`img2pose.py:46-49`)
- Model wrapping for CPU/GPU modes (`img2pose.py:84-104`)
- Auto-load and evaluate when `model_path` provided (`img2pose.py:106-108`)

**Critical parameters for library users:**
- `depth`: 18, 50, or 101 (ResNet variants)
- `min_size`, `max_size`: Image size constraints
- `pose_mean`, `pose_stddev`: Required for inference
- `threed_68_points`: Required for bounding box projection
- `bbox_x_factor`, `bbox_y_factor`, `expand_forehead`: Output customization

### 2.2 `predict()` Method (`img2pose.py:131-137`)

**Input:** `List[Tensor]` - List of image tensors (C, H, W) in RGB, normalized 0-1  
**Output:** `List[Dict[str, Tensor]]` - One dict per image containing:
- `boxes`: Tensor[N, 4] - (left, top, right, bottom) bounding boxes
- `labels`: Tensor[N] - Class labels (always 1 for face)
- `scores`: Tensor[N] - Detection confidence scores (0-1)
- `dofs`: Tensor[N, 6] - 6DoF poses (rx, ry, rz, tx, ty, tz)

**Invariant:** Must be called after `evaluate()` (`img2pose.py:132` assertion)

### 2.3 `transform_pose_global_project_bbox()` (`utils/pose_operations.py:238-301`)

**Signature:**
```python
def transform_pose_global_project_bbox(
    boxes,              # Tensor[N, 4]
    dofs,               # Tensor[N, 6] 
    pose_mean,          # Tensor[6]
    pose_stddev,        # Tensor[6]
    image_shape,        # Tuple[int, int] - (H, W)
    threed_68_points=None,  # Tensor[68, 3]
    bbox_x_factor=1.1,
    bbox_y_factor=1.1,
    expand_forehead=0.3,
):
```

**Behavior:**
1. Denormalizes poses using mean/stddev (`utils/pose_operations.py:270`)
2. Converts pose from bbox-local to global image coordinates (`utils/pose_operations.py:276`)
3. Projects 3D landmarks to compute new bounding boxes (`utils/pose_operations.py:281-292`)
4. Returns `(projected_boxes, global_dofs)` as tensors

### 2.4 `load_model()` (`model_loader.py:30-39`)

**Signature:**
```python
def load_model(fpn_model, model_path, model_only=True, optimizer=None, cpu_mode=False):
```

**Behavior:**
- Loads checkpoint with `torch.load()` 
- Maps to CPU if `cpu_mode=True` (`model_loader.py:31-34`)
- Loads only `fpn_model` state dict from checkpoint (`model_loader.py:36`)

---

## 3. Current Test Structure and Patterns

### 3.1 No pytest Tests Found

Searching for `test_*.py` and `*test*.py` patterns returned no results. The repository does not have a formal test suite.

### 3.2 Evaluation Scripts (Serve as Integration Tests)

#### WIDER FACE Evaluation (`/home/tal/dev/img2pose/evaluation/evaluate_wider.py`)

**Purpose:** Benchmark face detection on WIDER FACE validation set

**Usage pattern (from README):**
```bash
python3 evaluation/evaluate_wider.py \
--dataset_path datasets/WIDER_Face/WIDER_val/images/ \
--dataset_list datasets/WIDER_Face/wider_face_split/wider_face_val_bbx_gt.txt \
--pose_mean models/WIDER_train_pose_mean_v1.npy \
--pose_stddev models/WIDER_train_pose_stddev_v1.npy \
--pretrained_path models/img2pose_v1.pth \
--output_path results/WIDER_FACE/Val/
```

**Key code patterns:**
- Model creation (`evaluate_wider.py:34-48`):
```python
img2pose_model = img2poseModel(
    args.depth,
    args.min_size[-1],
    args.max_size,
    pose_mean=self.pose_mean,
    pose_stddev=self.pose_stddev,
    threed_68_points=self.threed_68_points,
)
load_model(img2pose_model.fpn_model, args.pretrained_path, ...)
img2pose_model.evaluate()
```

- Inference call (`evaluate_wider.py:150`):
```python
res = self.img2pose_model.predict([self.transform(run_img)])
```

- Result extraction (`evaluate_wider.py:157-160`):
```python
bbox = res["boxes"].cpu().numpy()[i].astype("int")
score = res["scores"].cpu().numpy()[i]
pose = res["dofs"].cpu().numpy()[i]
```

### 3.3 Jupyter Notebooks (Interactive Integration Tests)

**Location:** `/home/tal/dev/img2pose/evaluation/jupyter_notebooks/`

| Notebook | Purpose |
|----------|---------|
| `test_own_images.ipynb` | End-to-end inference on custom images |
| `visualize_trained_model_predictions.ipynb` | Visual validation of model outputs |
| `aflw_2000_3d_evaluation.ipynb` | Pose estimation benchmark on AFLW2000-3D |
| `biwi_evaluation.ipynb` | Pose estimation benchmark on BIWI dataset |

**Key pattern from `test_own_images.ipynb`:**
```python
# Model setup
img2pose_model = img2poseModel(
    DEPTH, MIN_SIZE, MAX_SIZE, 
    pose_mean=pose_mean, pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)
load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
img2pose_model.evaluate()

# Inference
transform = transforms.Compose([transforms.ToTensor()])
res = img2pose_model.predict([transform(img)])[0]

# Result access
all_bboxes = res["boxes"].cpu().numpy().astype('float')
pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
score = res["scores"][i]
```

### 3.4 Validation Methodology

Results are validated by comparison with benchmark datasets:
- **WIDER FACE:** mAP comparison using official eval tools
- **AFLW2000-3D:** Mean Absolute Error on Euler angles
- **BIWI:** Mean Absolute Error on Euler angles

---

## 4. External File Dependencies

### 4.1 Pose Reference Files (Required for Inference)

| File | Description | Usage |
|------|-------------|-------|
| `/home/tal/dev/img2pose/pose_references/reference_3d_68_points_trans.npy` | 68 facial landmarks in 3D | Bounding box projection in `transform_pose_global_project_bbox()` |
| `/home/tal/dev/img2pose/pose_references/reference_3d_5_points_trans.npy` | 5-point alignment landmarks | Face alignment in `align_faces()` |

**Loading pattern:**
```python
threed_68_points = np.load('./pose_references/reference_3d_68_points_trans.npy')
```

### 4.2 Pose Statistics Files (Required for Inference)

| File | Description | Default Path |
|------|-------------|--------------|
| `models/WIDER_train_pose_mean_v1.npy` | Pose normalization mean | `./models/WIDER_train_pose_mean_v1.npy` |
| `models/WIDER_train_pose_stddev_v1.npy` | Pose normalization stddev | `./models/WIDER_train_pose_stddev_v1.npy` |

**Note:** These paths are referenced in README and Jupyter notebooks but the actual .npy files are NOT present in the repository (must be downloaded from Model Zoo).

### 4.3 Model Weights (Required for Inference)

| File | Description | Source |
|------|-------------|--------|
| `models/img2pose_v1.pth` | Pre-trained WIDER FACE model | Download from Model Zoo |

**Checkpoint structure** (from `model_loader.py:36`):
```python
checkpoint = {
    "fpn_model": state_dict,
    "optimizer": optimizer_state_dict  # Optional
}
```

### 4.4 Renderer Dependencies (Visualization Only)

| File | Description |
|------|-------------|
| `/home/tal/dev/img2pose/pose_references/vertices_trans.npy` | 3D face mesh vertices |
| `/home/tal/dev/img2pose/pose_references/triangles.npy` | Face mesh triangles |

**Note:** These are only needed for the `Renderer` class in visualization notebooks, not for core inference.

---

## 5. Key Files with Line Counts (Bloat Awareness)

### Files Over 500 LOC Threshold

| File | Lines | Notes |
|------|-------|-------|
| `/home/tal/dev/img2pose/models.py` | **568** | Core model classes - NEEDED for library |
| `/home/tal/dev/img2pose/rpn.py` | **539** | Region Proposal Network - NEEDED for library |

### Files Under Threshold (Inference Core)

| File | Lines | Notes |
|------|-------|-------|
| `/home/tal/dev/img2pose/img2pose.py` | 137 | Main model wrapper - NEEDED |
| `/home/tal/dev/img2pose/generalized_rcnn.py` | 121 | Base RCNN class - NEEDED |
| `/home/tal/dev/img2pose/model_loader.py` | 39 | Model loading - NEEDED |
| `/home/tal/dev/img2pose/utils/pose_operations.py` | 322 | Pose utilities - NEEDED |
| `/home/tal/dev/img2pose/utils/image_operations.py` | 321 | Image utilities - NEEDED (used by pose_operations) |

### Training-Only Files (NOT Needed for Library)

| File | Lines | Notes |
|------|-------|-------|
| `/home/tal/dev/img2pose/utils/augmentation.py` | **439** | Training augmentation - EXCLUDE |
| `/home/tal/dev/img2pose/losses.py` | 130 | Training losses - EXCLUDE |
| `/home/tal/dev/img2pose/train.py` | varies | Training script - EXCLUDE |
| `/home/tal/dev/img2pose/data_loader_*.py` | varies | Data loading - EXCLUDE |

---

## 6. Summary: Minimal Module Set for pip Library

### Required Files (Inference Core)
1. `img2pose.py` (137 lines)
2. `models.py` (568 lines) 
3. `generalized_rcnn.py` (121 lines)
4. `rpn.py` (539 lines)
5. `model_loader.py` (39 lines)
6. `utils/pose_operations.py` (322 lines)
7. `utils/image_operations.py` (321 lines)
8. `utils/face_align.py` (referenced by pose_operations)

### Required Data Files (Bundle with Package)
1. `pose_references/reference_3d_68_points_trans.npy`
2. `pose_references/reference_3d_5_points_trans.npy`

### Files to Download Separately (Model Zoo)
1. `models/WIDER_train_pose_mean_v1.npy`
2. `models/WIDER_train_pose_stddev_v1.npy`
3. `models/img2pose_v1.pth`

### External Dependencies (from PyPI)
- torch
- torchvision
- numpy
- scipy
- Pillow
- opencv-python (cv2)
