# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

img2pose is a real-time 6DoF (six degrees of freedom) face pose estimation system that performs face detection and alignment without requiring prior face detection or facial landmark localization. It was published at CVPR 2021.

The model outputs for each detected face:
- **6DoF pose**: rotation vector (rx, ry, rz) + translation vector (tx, ty, tz)
- **Projected bounding boxes**: left, top, right, bottom
- **Face confidence scores**: 0 to 1

## Build and Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build the Sim3DR renderer (required for visualization)
cd Sim3DR && sh build_sim3dr.sh
```

## Training Commands

### Prepare dataset (convert JSON annotations to LMDB)
```bash
# Training set (generates pose mean/stddev files)
python3 convert_json_list_to_lmdb.py \
  --json_list ./annotations/WIDER_train_annotations.txt \
  --dataset_path ./datasets/WIDER_Face/WIDER_train/images/ \
  --dest ./datasets/lmdb/ --train

# Validation set
python3 convert_json_list_to_lmdb.py \
  --json_list ./annotations/WIDER_val_annotations.txt \
  --dataset_path ./datasets/WIDER_Face/WIDER_val/images/ \
  --dest ./datasets/lmdb
```

### Single GPU training
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --pose_mean ./datasets/lmdb/WIDER_train_annotations_pose_mean.npy \
  --pose_stddev ./datasets/lmdb/WIDER_train_annotations_pose_stddev.npy \
  --workspace ./workspace/ \
  --train_source ./datasets/lmdb/WIDER_train_annotations.lmdb \
  --val_source ./datasets/lmdb/WIDER_val_annotations.lmdb \
  --prefix trial_1 --batch_size 2 --lr_plateau --early_stop \
  --random_flip --random_crop --max_size 1400
```

### Multi-GPU training (4 GPUs)
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
  [same args as above] --distributed
```

## Evaluation Commands

### WIDER FACE evaluation
```bash
python3 evaluation/evaluate_wider.py \
  --dataset_path datasets/WIDER_Face/WIDER_val/images/ \
  --dataset_list datasets/WIDER_Face/wider_face_split/wider_face_val_bbx_gt.txt \
  --pose_mean models/WIDER_train_pose_mean_v1.npy \
  --pose_stddev models/WIDER_train_pose_stddev_v1.npy \
  --pretrained_path models/img2pose_v1.pth \
  --output_path results/WIDER_FACE/Val/
```

### Face alignment inference
```bash
python3 run_face_alignment.py \
  --pose_mean models/WIDER_train_pose_mean_v1.npy \
  --pose_stddev models/WIDER_train_pose_stddev_v1.npy \
  --pretrained_path models/img2pose_v1.pth \
  --images_path image_path_or_list \
  --output_path path_to_save_aligned_faces
```

### Jupyter notebooks for evaluation
- `evaluation/jupyter_notebooks/test_own_images.ipynb` - Test on custom images
- `evaluation/jupyter_notebooks/aflw_2000_3d_evaluation.ipynb` - AFLW2000-3D benchmark
- `evaluation/jupyter_notebooks/biwi_evaluation.ipynb` - BIWI benchmark
- `evaluation/jupyter_notebooks/visualize_trained_model_predictions.ipynb` - Visualization

## Architecture

The system is based on Faster R-CNN with a 6DoF regression head instead of bounding box regression:

```
Image → ResNet-FPN Backbone → RPN (proposals) → ROI Heads with 6DoF Predictor → outputs
```

### Key modules

| File | Purpose |
|------|---------|
| `img2pose.py` | High-level model wrapper (`img2poseModel` class) |
| `models.py` | Core model: `FasterDoFRCNN`, `FastRCNNDoFPredictor`, `FastRCNNClassPredictor` |
| `generalized_rcnn.py` | Base R-CNN framework |
| `rpn.py` | Region Proposal Network, anchor generation |
| `losses.py` | Training losses (classification + 6DoF + point projection) |
| `train.py` | Training pipeline with distributed support |
| `config.py` | Configuration management |

### Data pipeline

| File | Purpose |
|------|---------|
| `data_loader_lmdb.py` | LMDB dataset class |
| `data_loader_lmdb_augmenter.py` | Training loader with augmentation |
| `convert_json_list_to_lmdb.py` | Convert JSON annotations → LMDB |

### Utilities

| File | Purpose |
|------|---------|
| `utils/pose_operations.py` | 6DoF transformations, point projection, alignment |
| `utils/image_operations.py` | Bounding box manipulation |
| `utils/augmentation.py` | Online data augmentation |
| `utils/dist.py` | Distributed training (NCCL) |
| `Sim3DR/` | C++/Cython 3D renderer for visualization |

## Programmatic Usage

```python
from img2pose import img2poseModel
from model_loader import load_model
import numpy as np

# Load pose statistics and 3D reference points
pose_mean = np.load("models/WIDER_train_pose_mean_v1.npy")
pose_stddev = np.load("models/WIDER_train_pose_stddev_v1.npy")
threed_points = np.load("pose_references/reference_3d_68_points_trans.npy")

# Create and load model
model = img2poseModel(
    depth=18,  # ResNet depth: 18, 50, or 101
    min_size=640,
    max_size=1400,
    pose_mean=pose_mean,
    pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)
load_model(model.fpn_model, "models/img2pose_v1.pth", model_only=True)
model.evaluate()

# Predict (input: list of transformed image tensors)
predictions = model.predict([image_tensor])
# Returns: {"boxes": [N,4], "scores": [N], "dofs": [N,6], "landmarks": [N,68,2]}
```

### Face Alignment

The library also supports face alignment - producing cropped, normalized face images from detected faces:

```python
from img2pose import Img2Pose
from PIL import Image

# Initialize detector
detector = Img2Pose()

# Detect and align in one step (convenient for simple pipelines)
results = detector.detect_and_align("group_photo.jpg")
for i, face_data in enumerate(results):
    face_data["image"].save(f"aligned_face_{i}.jpg")
    print(f"Face {i}: confidence={face_data['confidence']:.2f}")

# Or detect and align separately (for more control)
image = Image.open("photo.jpg")
faces = detector.detect_faces(image)
aligned_crops = detector.align_faces(image, faces, output_size=224)
for i, crop in enumerate(aligned_crops):
    crop.save(f"face_{i}.jpg")

# Batch processing for training pipelines
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
all_faces = detector.detect_faces(images)
all_aligned = detector.align_faces_batch(images, all_faces, return_array=True)
# all_aligned[i] contains numpy arrays for faces in images[i]
```

**Note:** Face alignment requires opencv-python. Install with:
```bash
pip install img2pose[alignment]
```

## Key Concepts

- **6DoF pose**: Rotation vector (axis-angle) + translation vector, normalized using pose mean/stddev
- **Pose transformation**: Conversions between global image coordinates and bounding box-relative coordinates are handled in `utils/pose_operations.py`
- **LMDB storage**: Efficient dataset format for large-scale training
- **3D reference points**: Located in `pose_references/` - 68 facial landmarks and 5 keypoints for alignment
