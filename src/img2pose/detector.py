"""Public API for img2pose face detection and pose estimation.

This module provides the Img2Pose class, the main entry point for using
img2pose for face detection and 6DoF pose estimation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from ._model import img2poseModel
from ._weights import get_pose_stats_paths, load_weights, load_weights_from_path

# Type aliases
ImageInput = Union[str, Path, Image.Image, np.ndarray]
FaceDict = Dict[str, Any]


def _get_package_data_path(filename: str) -> Path:
    """Get path to a file in the package data directory."""
    return Path(__file__).parent / "data" / filename


def _load_image(image: ImageInput) -> Image.Image:
    """Load an image from various input types.

    Args:
        image: File path, PIL Image, or numpy array (RGB uint8 HWC)

    Returns:
        PIL Image in RGB mode
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    elif isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")
        return Image.fromarray(image)
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Expected file path, PIL Image, or numpy array."
        )


def _image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL Image to tensor for model input."""
    img_array = np.array(image, dtype=np.float32)
    # HWC -> CHW and normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0
    return img_tensor.to(device)


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform 3D points using pose."""
    return points.dot(Rotation.from_rotvec(pose[:3]).as_matrix().T) + pose[3:]


def _project_landmarks(
    threed_points: np.ndarray,
    pose: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project 3D landmarks to 2D image coordinates.

    Args:
        threed_points: 3D landmark points [N, 3]
        pose: 6DoF pose [rx, ry, rz, tx, ty, tz]
        image_width: Image width
        image_height: Image height

    Returns:
        2D landmarks [N, 2]
    """
    # Create camera intrinsics
    focal = image_width + image_height
    intrinsics = np.array([
        [focal, 0, image_width / 2],
        [0, focal, image_height / 2],
        [0, 0, 1]
    ])

    # Transform points
    points_3d = _transform_points(threed_points, pose)

    # Project to 2D
    points_proj = intrinsics.dot(points_3d.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]

    return points_2d


class Img2Pose:
    """Face detection with 6DoF pose estimation.

    This class provides a simple interface for detecting faces and estimating
    their 6 degrees of freedom pose (rotation + translation) in images.

    Example:
        >>> from img2pose import Img2Pose
        >>> detector = Img2Pose()
        >>> faces = detector.detect_faces("image.jpg")
        >>> for face in faces:
        ...     print(f"Box: {face['box']}")
        ...     print(f"Confidence: {face['confidence']:.2f}")
        ...     print(f"Pose: {face['pose']}")

    The output format follows MTCNN/RetinaFace conventions:
        - "box": [x1, y1, x2, y2] in xyxy format
        - "confidence": Detection confidence score
        - "pose": [rx, ry, rz, tx, ty, tz] 6DoF pose
        - "keypoints": Dict with "left_eye", "right_eye", "nose",
                      "mouth_left", "mouth_right" keys
    """

    # Indices for 5 keypoints in 68-point format
    _KEYPOINT_INDICES = {
        "left_eye": 36,      # Left eye outer corner
        "right_eye": 45,     # Right eye outer corner
        "nose": 30,          # Nose tip
        "mouth_left": 48,    # Left mouth corner
        "mouth_right": 54,   # Right mouth corner
    }

    def __init__(
        self,
        device: str = "auto",
        score_threshold: float = 0.5,
        max_faces: int = -1,
        model_path: Optional[str] = None,
        min_size: int = 640,
        max_size: int = 1400,
    ):
        """Initialize the face detector.

        Args:
            device: Device for inference. Options:
                - "auto": Automatically select GPU if available
                - "cuda": Use GPU
                - "cpu": Use CPU
                - "cuda:0", "cuda:1", etc.: Use specific GPU
            score_threshold: Minimum confidence score for detections (0-1)
            max_faces: Maximum number of faces to return (-1 for unlimited)
            model_path: Path to custom model weights. If None, downloads
                       pre-trained weights automatically.
            min_size: Minimum image dimension for processing
            max_size: Maximum image dimension for processing
        """
        self.score_threshold = score_threshold
        self.max_faces = max_faces

        # Handle device selection
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load reference 3D points
        self._threed_68_points = np.load(
            _get_package_data_path("reference_3d_68_points_trans.npy")
        )
        self._threed_5_points = np.load(
            _get_package_data_path("reference_3d_5_points_trans.npy")
        )

        # Load model weights and pose statistics
        if model_path:
            checkpoint = load_weights_from_path(model_path, device=str(self.device))
            # Use default pose stats for custom models
            self._pose_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            self._pose_stddev = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            checkpoint = load_weights(device=str(self.device))
            # Load pose statistics from downloaded files
            pose_mean_path, pose_stddev_path = get_pose_stats_paths()
            self._pose_mean = np.load(pose_mean_path)
            self._pose_stddev = np.load(pose_stddev_path)

        # Initialize model
        self._model = img2poseModel(
            depth=18,
            min_size=min_size,
            max_size=max_size,
            device=self.device,
            pose_mean=self._pose_mean,
            pose_stddev=self._pose_stddev,
            threed_68_points=self._threed_68_points,
            threed_5_points=self._threed_5_points,
        )

        # Load weights
        self._model.load_state_dict(checkpoint["fpn_model"])
        self._model.evaluate()

    def detect_faces(
        self,
        image: Union[ImageInput, List[ImageInput]],
        score_threshold: Optional[float] = None,
        max_faces: Optional[int] = None,
    ) -> Union[List[FaceDict], List[List[FaceDict]]]:
        """Detect faces in an image or batch of images.

        Args:
            image: Input image(s). Can be:
                - File path (str or Path)
                - PIL Image
                - Numpy array (RGB uint8 HWC)
                - List of the above for batch processing
            score_threshold: Override instance threshold (optional)
            max_faces: Override instance max_faces (optional)

        Returns:
            List of face dictionaries for single image, or
            list of lists for batch input.

            Each face dict contains:
                - "box": [x1, y1, x2, y2] bounding box
                - "confidence": Detection confidence (0-1)
                - "pose": [rx, ry, rz, tx, ty, tz] 6DoF pose
                - "keypoints": Dict with 5 facial landmarks
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        max_count = max_faces if max_faces is not None else self.max_faces

        # Handle batch input
        if isinstance(image, list):
            return [
                self._detect_single(img, threshold, max_count)
                for img in image
            ]

        return self._detect_single(image, threshold, max_count)

    def _detect_single(
        self,
        image: ImageInput,
        threshold: float,
        max_count: int,
    ) -> List[FaceDict]:
        """Detect faces in a single image."""
        # Load and convert image
        pil_image = _load_image(image)
        width, height = pil_image.size

        # Convert to tensor
        img_tensor = _image_to_tensor(pil_image, self.device)

        # Run inference
        predictions = self._model.predict([img_tensor])

        # Process results
        faces = []
        if len(predictions) > 0:
            pred = predictions[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            dofs = pred["dofs"].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] < threshold:
                    continue

                # Extract box
                box = boxes[i].tolist()

                # Extract pose
                pose = dofs[i].tolist()

                # Compute keypoints by projecting 3D landmarks
                landmarks_2d = _project_landmarks(
                    self._threed_68_points,
                    dofs[i],
                    width,
                    height,
                )

                keypoints = {
                    name: landmarks_2d[idx].tolist()
                    for name, idx in self._KEYPOINT_INDICES.items()
                }

                faces.append({
                    "box": box,
                    "confidence": float(scores[i]),
                    "pose": pose,
                    "keypoints": keypoints,
                })

        # Sort by confidence (descending)
        faces.sort(key=lambda x: x["confidence"], reverse=True)

        # Limit number of faces
        if max_count > 0:
            faces = faces[:max_count]

        return faces

    def __call__(
        self,
        image: Union[ImageInput, List[ImageInput]],
        **kwargs,
    ) -> Union[List[FaceDict], List[List[FaceDict]]]:
        """Callable interface (like Ultralytics YOLO).

        This is an alias for detect_faces().
        """
        return self.detect_faces(image, **kwargs)

    def visualize(
        self,
        image: ImageInput,
        faces: Optional[List[FaceDict]] = None,
        show_box: bool = True,
        show_keypoints: bool = True,
        show_confidence: bool = True,
        box_color: tuple = (0, 255, 0),
        keypoint_color: tuple = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Visualize detections on an image.

        Args:
            image: Input image
            faces: Pre-computed face detections. If None, runs detection.
            show_box: Whether to draw bounding boxes
            show_keypoints: Whether to draw keypoints
            show_confidence: Whether to show confidence scores
            box_color: RGB color for bounding boxes
            keypoint_color: RGB color for keypoints
            thickness: Line thickness

        Returns:
            Numpy array (RGB uint8 HWC) with visualizations drawn
        """
        # Import visualization module (optional dependency)
        from ._visualization import draw_detections

        # Load image
        pil_image = _load_image(image)

        # Run detection if not provided
        if faces is None:
            faces = self.detect_faces(image)

        # Draw and return
        return draw_detections(
            pil_image,
            faces,
            show_box=show_box,
            show_keypoints=show_keypoints,
            show_confidence=show_confidence,
            box_color=box_color,
            keypoint_color=keypoint_color,
            thickness=thickness,
        )
