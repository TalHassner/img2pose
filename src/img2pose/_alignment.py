"""Internal alignment functions for face cropping and normalization.

This module provides functions for aligning detected faces based on their
6DoF poses. It implements similarity transform estimation without scikit-image,
using only numpy and scipy for core math operations.

The alignment process:
1. Project 3D facial landmarks to 2D using pose
2. Select best-fit template from 5 orientation-specific templates
3. Estimate similarity transform from landmarks to template
4. Warp face region to normalized output

All functions in this module are internal and should not be used directly.
Use the public API methods on Img2Pose instead:
- align_faces()
- align_faces_batch()
- detect_and_align()
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def _get_cv2():
    """Lazy import opencv with helpful error message.

    Returns:
        The cv2 module.

    Raises:
        ImportError: If opencv-python is not installed.
    """
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "Face alignment requires opencv-python. "
            "Install with: pip install img2pose[alignment]"
        )


# Alignment templates for 112x112 output size.
# Shape: [5, 5, 2] - 5 templates, each with 5 landmarks (x, y).
# Templates are ordered by head orientation:
#   0: left profile
#   1: left 3/4 view
#   2: frontal
#   3: right 3/4 view
#   4: right profile
# Landmarks are in order: left_eye, right_eye, nose, mouth_left, mouth_right
ALIGNMENT_TEMPLATES_112 = np.array(
    [
        # Template 0: left profile
        [
            [51.642, 50.115],
            [57.617, 49.990],
            [35.740, 69.007],
            [51.157, 89.050],
            [57.025, 89.702],
        ],
        # Template 1: left 3/4
        [
            [45.031, 50.118],
            [65.568, 50.872],
            [39.677, 68.111],
            [45.177, 86.190],
            [64.246, 86.758],
        ],
        # Template 2: frontal
        [
            [39.730, 51.138],
            [72.270, 51.138],
            [56.000, 68.493],
            [42.463, 87.010],
            [69.537, 87.010],
        ],
        # Template 3: right 3/4
        [
            [46.845, 50.872],
            [67.382, 50.118],
            [72.737, 68.111],
            [48.167, 86.758],
            [67.236, 86.190],
        ],
        # Template 4: right profile
        [
            [54.796, 49.990],
            [60.771, 50.115],
            [76.673, 69.007],
            [55.388, 89.702],
            [61.257, 89.050],
        ],
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Estimate similarity transform from source to destination points.

    Computes the optimal similarity transform (rotation, uniform scale,
    translation) that maps src_points to dst_points using least squares.

    This is a pure numpy implementation that does not require scikit-image.
    The algorithm uses Procrustes analysis with SVD for rotation estimation.

    Args:
        src_points: Source points, shape [N, 2].
        dst_points: Destination points, shape [N, 2].

    Returns:
        Affine transformation matrix of shape [2, 3] suitable for cv2.warpAffine.
        The matrix maps source coordinates to destination coordinates.
    """
    # Ensure float64 for numerical stability
    src = src_points.astype(np.float64)
    dst = dst_points.astype(np.float64)

    # Step 1: Center both point sets
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)

    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    # Step 2: Compute scale using RMS distances
    src_rms = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    dst_rms = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))

    # Handle degenerate case
    if src_rms < 1e-10:
        # Return identity-like transform centered on dst
        return np.array(
            [[1.0, 0.0, dst_centroid[0]], [0.0, 1.0, dst_centroid[1]]],
            dtype=np.float32,
        )

    scale = dst_rms / src_rms

    # Normalize points
    src_normalized = src_centered / src_rms
    dst_normalized = dst_centered / dst_rms

    # Step 3: Compute optimal rotation using SVD
    # H = src^T @ dst
    H = src_normalized.T @ dst_normalized

    U, _, Vt = np.linalg.svd(H)

    # R = V @ U^T
    R = Vt.T @ U.T

    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 4: Compute final transformation
    # Transform: dst = scale * R @ src + translation
    # translation = dst_centroid - scale * R @ src_centroid
    translation = dst_centroid - scale * (R @ src_centroid)

    # Build [2, 3] affine matrix
    transform = np.zeros((2, 3), dtype=np.float32)
    transform[:2, :2] = scale * R
    transform[:2, 2] = translation

    return transform


def estimate_alignment_transform(
    landmarks_2d: np.ndarray, output_size: int
) -> Tuple[np.ndarray, int]:
    """Estimate the best alignment transform for given landmarks.

    Selects the best-fit template from 5 orientation-specific templates
    (left profile, left 3/4, frontal, right 3/4, right profile) and
    computes the similarity transform to map landmarks to that template.

    Args:
        landmarks_2d: 2D facial landmarks, shape [5, 2].
            Order: left_eye, right_eye, nose, mouth_left, mouth_right.
        output_size: Output face size (112 or 224).

    Returns:
        Tuple of (transform_matrix, template_index):
            - transform_matrix: [2, 3] affine matrix for cv2.warpAffine
            - template_index: Index of selected template (0-4)

    Raises:
        ValueError: If landmarks shape is not [5, 2] or output_size is invalid.
    """
    if landmarks_2d.shape != (5, 2):
        raise ValueError(f"Expected landmarks shape (5, 2), got {landmarks_2d.shape}")

    if output_size not in (112, 224):
        raise ValueError(f"output_size must be 112 or 224, got {output_size}")

    # Scale templates for output size
    scale_factor = output_size / 112.0
    templates = ALIGNMENT_TEMPLATES_112 * scale_factor

    # Convert landmarks to homogeneous coordinates for transform application
    landmarks = landmarks_2d.astype(np.float64)
    landmarks_homo = np.hstack([landmarks, np.ones((5, 1))])

    # Find best template by minimizing alignment error
    min_error = float("inf")
    best_transform = None
    best_index = 0

    for i in range(templates.shape[0]):
        template = templates[i]

        # Estimate transform from landmarks to template
        transform = estimate_similarity_transform(landmarks, template)

        # Apply transform to landmarks
        transformed = (transform @ landmarks_homo.T).T

        # Compute alignment error (sum of Euclidean distances)
        error = np.sum(np.sqrt(np.sum((transformed - template) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            best_transform = transform
            best_index = i

    return best_transform, best_index


def project_5_landmarks(
    threed_5_points: np.ndarray,
    poses: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project 3D landmarks to 2D image coordinates using poses.

    Projects the 5 canonical facial landmarks (eyes, nose, mouth corners)
    from 3D to 2D using the provided 6DoF poses and camera intrinsics.

    Args:
        threed_5_points: 3D reference landmarks, shape [5, 3].
        poses: 6DoF poses. Either single pose [6] or batch [N, 6].
            Format: [rx, ry, rz, tx, ty, tz] (rotation vector + translation).
        image_width: Image width for computing intrinsics.
        image_height: Image height for computing intrinsics.

    Returns:
        2D landmarks. Shape [5, 2] for single pose or [N, 5, 2] for batch.
    """
    # Compute camera intrinsics (standard img2pose convention)
    focal = image_width + image_height
    cx = image_width / 2.0
    cy = image_height / 2.0
    intrinsics = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

    # Handle single pose vs batch
    single_pose = poses.ndim == 1
    if single_pose:
        poses = poses[np.newaxis, :]

    n_poses = poses.shape[0]
    landmarks_2d = np.zeros((n_poses, 5, 2), dtype=np.float64)

    for i in range(n_poses):
        pose = poses[i]
        rvec = pose[:3]
        tvec = pose[3:]

        # Convert rotation vector to rotation matrix
        rotation = Rotation.from_rotvec(rvec)
        R = rotation.as_matrix()

        # Transform 3D points: P_cam = R @ P_world + t
        points_3d_transformed = (R @ threed_5_points.T).T + tvec

        # Project to 2D: p = K @ P_cam, then normalize
        points_projected = (intrinsics @ points_3d_transformed.T).T
        points_2d = points_projected[:, :2] / points_projected[:, 2:3]

        landmarks_2d[i] = points_2d

    if single_pose:
        return landmarks_2d[0]

    return landmarks_2d


def warp_face(
    image: np.ndarray, transform: np.ndarray, output_size: int
) -> np.ndarray:
    """Warp face region using affine transform.

    Args:
        image: Input image, shape [H, W, 3] or [H, W].
        transform: Affine transformation matrix, shape [2, 3].
        output_size: Output face size (typically 112 or 224).

    Returns:
        Warped face image, shape [output_size, output_size, C].
    """
    cv2 = _get_cv2()

    warped = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        borderValue=0,
    )

    return warped


def align_faces_from_image(
    image: np.ndarray,
    poses: np.ndarray,
    threed_5_points: np.ndarray,
    output_size: int = 112,
) -> List[np.ndarray]:
    """Align multiple faces from an image using their poses.

    This is the main alignment function that combines landmark projection,
    template matching, and face warping.

    Args:
        image: Input image as numpy array, shape [H, W, 3].
        poses: 6DoF poses, shape [N, 6] or [6] for single face.
            Format: [rx, ry, rz, tx, ty, tz].
        threed_5_points: 3D reference landmarks, shape [5, 3].
        output_size: Output face size (112 or 224). Default 112.

    Returns:
        List of aligned face images. Each is a numpy array of shape
        [output_size, output_size, 3]. Returns empty list if poses is empty.
    """
    # Handle empty input
    if poses is None or (isinstance(poses, np.ndarray) and poses.size == 0):
        return []

    # Ensure poses is 2D
    poses = np.asarray(poses)
    if poses.ndim == 1:
        poses = poses[np.newaxis, :]

    if poses.shape[0] == 0:
        return []

    # Get image dimensions
    height, width = image.shape[:2]

    # Project landmarks for all poses
    landmarks_2d = project_5_landmarks(threed_5_points, poses, width, height)

    # Align each face
    aligned_faces = []
    for i in range(poses.shape[0]):
        # landmarks_2d has shape [N, 5, 2] since we passed batch poses
        landmarks = landmarks_2d[i]

        # Estimate alignment transform
        transform, _ = estimate_alignment_transform(landmarks, output_size)

        # Warp face
        face = warp_face(image, transform, output_size)
        aligned_faces.append(face)

    return aligned_faces
