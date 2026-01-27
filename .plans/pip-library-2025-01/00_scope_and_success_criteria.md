# Scope and Success Criteria

## Problem Restatement

The img2pose repository contains a CVPR 2021 research implementation for face detection and 6DoF pose estimation. The current code is designed for research use - running evaluation scripts, Jupyter notebooks, and training pipelines. The goal is to transform this into a clean, pip-installable Python library suitable for production use as a face detection and pose estimation tool.

Key challenges:
1. The code uses relative imports and assumes the working directory is the repository root
2. Model weights and pose reference files are expected at fixed relative paths
3. The API is designed for research (argparse scripts) rather than library usage
4. Dependencies include training-only packages (tensorboard, jupyter, etc.)
5. No existing packaging infrastructure (setup.py, pyproject.toml)

## Definition of Success

### MUST Requirements (from goals)

1. **M1: Pip-installable package** - Users can install via `pip install .` or from PyPI
2. **M2: Single image API** - Provide a function to process one image and return detections
3. **M3: Batch image API** - Provide a function to process multiple images efficiently
4. **M4: GPU/CPU auto-selection** - Automatically use GPU if available, fall back to CPU
5. **M5: Numerical equivalence** - The library must produce identical results to the current code on at least one benchmark (AFLW2000-3D is smallest at 2000 images)
6. **M6: Bounding box output** - Return face bounding boxes in standard format
7. **M7: 6DoF pose output** - Return 6DoF pose estimates (rx, ry, rz, tx, ty, tz)
8. **M8: Pre-trained model support** - Use the authors' pre-trained models without requiring users to download separately
9. **M9: Clean production code** - Remove training-only code from the public API

### SHOULD Requirements

1. **S1: Visualization methods** - Optional 2D visualization of bounding boxes and 5-point landmarks
2. **S2: Confidence threshold** - Allow filtering detections by confidence score
3. **S3: Max faces limit** - Allow limiting the number of returned faces

### Non-Goals (Explicit Exclusions)

1. **NG1: Training functionality** - Training code remains in repo but not exposed via library API
2. **NG2: 3D rendering/visualization** - The Sim3DR-based 3D rendering is excluded from the library
3. **NG3: Downstream face processing** - Face alignment, recognition, etc. are out of scope

## Backward Compatibility Definition

### What is preserved (existing repo functionality)
- All existing Python scripts (train.py, run_face_alignment.py, evaluate_wider.py, etc.) continue to work
- All existing imports and module structure remain functional
- The Jupyter notebooks continue to work
- Training pipeline is unaffected

### What is new (library functionality)
- New `img2pose` package exposable via pip install
- New public API classes and functions
- New dependency specification (minimal runtime deps vs full dev deps)

### Guarantee
All existing scripts in the repository root directory continue to produce identical results. The new library code is additive and does not modify any existing files.

## Constraints

1. **Offline operation** - Model weights must be bundled or auto-downloadable; no online API calls for inference
2. **Hardware** - Must work on CPU-only machines; GPU is optional
3. **Python version** - Support Python 3.8+ (matching PyTorch support)
4. **Dependencies** - Runtime dependencies limited to: torch, torchvision, numpy, opencv-python, scipy, Pillow
5. **Model weights** - Use the pre-trained img2pose_v1.pth from the Model Zoo (download on first use or bundle)
