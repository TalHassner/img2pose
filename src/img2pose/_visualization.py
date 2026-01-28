"""Internal visualization utilities for img2pose.

This module provides functions for drawing face detections on images.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image: Image.Image,
    faces: List[Dict],
    show_box: bool = True,
    show_keypoints: bool = True,
    show_confidence: bool = True,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    keypoint_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw face detections on an image.

    Args:
        image: PIL Image to draw on
        faces: List of face detection dicts from Img2Pose
        show_box: Whether to draw bounding boxes
        show_keypoints: Whether to draw keypoints
        show_confidence: Whether to show confidence scores
        box_color: RGB color for bounding boxes
        keypoint_color: RGB color for keypoints
        thickness: Line thickness for boxes

    Returns:
        Numpy array (RGB uint8 HWC) with drawings
    """
    # Make a copy to avoid modifying the original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Try to get a font for text (fall back to default if not available)
    font = None
    if show_confidence:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except (IOError, OSError):
                font = ImageFont.load_default()

    for face in faces:
        # Draw bounding box
        if show_box:
            box = face["box"]
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=thickness)

            # Draw confidence score
            if show_confidence:
                conf = face["confidence"]
                text = f"{conf:.2f}"
                # Position text above the box
                text_y = max(0, y1 - 20)
                draw.text((x1, text_y), text, fill=box_color, font=font)

        # Draw keypoints
        if show_keypoints and "keypoints" in face:
            keypoints = face["keypoints"]
            radius = max(2, thickness)
            for name, (kx, ky) in keypoints.items():
                # Draw filled circle
                draw.ellipse(
                    [kx - radius, ky - radius, kx + radius, ky + radius],
                    fill=keypoint_color,
                    outline=keypoint_color,
                )

    return np.array(img_copy)


def draw_pose_axes(
    image: Image.Image,
    faces: List[Dict],
    axis_length: float = 50.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw 3D pose axes on faces (optional advanced visualization).

    Args:
        image: PIL Image to draw on
        faces: List of face detection dicts from Img2Pose
        axis_length: Length of axis lines in pixels
        thickness: Line thickness

    Returns:
        Numpy array (RGB uint8 HWC) with pose axes drawn
    """
    from scipy.spatial.transform import Rotation

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    # Colors for X, Y, Z axes
    axis_colors = {
        "x": (255, 0, 0),    # Red
        "y": (0, 255, 0),    # Green
        "z": (0, 0, 255),    # Blue
    }

    for face in faces:
        pose = face["pose"]
        box = face["box"]

        # Get face center
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        # Get rotation from pose
        rvec = np.array(pose[:3])
        rot_mat = Rotation.from_rotvec(rvec).as_matrix()

        # Draw axes
        axes = {
            "x": np.array([axis_length, 0, 0]),
            "y": np.array([0, axis_length, 0]),
            "z": np.array([0, 0, axis_length]),
        }

        for axis_name, axis_vec in axes.items():
            # Rotate axis
            rotated = rot_mat.dot(axis_vec)

            # Project to 2D (simple orthographic projection)
            end_x = center_x + rotated[0]
            end_y = center_y - rotated[1]  # Y is inverted in image coords

            draw.line(
                [center_x, center_y, end_x, end_y],
                fill=axis_colors[axis_name],
                width=thickness,
            )

    return np.array(img_copy)
