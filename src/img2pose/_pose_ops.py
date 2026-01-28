"""Internal pose operations for img2pose.

This module contains pose transformation utilities for converting between
coordinate systems and projecting 3D landmarks to 2D.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def bbox_is_dict(bbox):
    """Convert bbox array to dict format if needed."""
    if not isinstance(bbox, dict):
        return {
            "left": bbox[0],
            "top": bbox[1],
            "right": bbox[2],
            "bottom": bbox[3],
        }
    return bbox


def expand_bbox_rectangle(
    w: int,
    h: int,
    bbox_x_factor: float = 2.0,
    bbox_y_factor: float = 2.0,
    lms: np.ndarray = None,
    expand_forehead: float = 0.3,
    roll: float = 0,
) -> np.ndarray:
    """Expand bounding box based on facial landmarks."""
    min_pt_x = np.min(lms[:, 0], axis=0)
    max_pt_x = np.max(lms[:, 0], axis=0)
    min_pt_y = np.min(lms[:, 1], axis=0)
    max_pt_y = np.max(lms[:, 1], axis=0)

    bbox_size_x = int(np.max(max_pt_x - min_pt_x) * bbox_x_factor)
    center_pt_x = 0.5 * min_pt_x + 0.5 * max_pt_x

    bbox_size_y = int(np.max(max_pt_y - min_pt_y) * bbox_y_factor)
    center_pt_y = 0.5 * min_pt_y + 0.5 * max_pt_y

    bbox_min_x = center_pt_x - bbox_size_x * 0.5
    bbox_max_x = center_pt_x + bbox_size_x * 0.5
    bbox_min_y = center_pt_y - bbox_size_y * 0.5
    bbox_max_y = center_pt_y + bbox_size_y * 0.5

    if abs(roll) > 2.5:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_max_y += expand_forehead_size
    elif roll > 1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_max_x += expand_forehead_size
    elif roll < -1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_min_x -= expand_forehead_size
    else:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_min_y -= expand_forehead_size

    bbox_min_x = bbox_min_x.astype(np.int32)
    bbox_max_x = bbox_max_x.astype(np.int32)
    bbox_min_y = bbox_min_y.astype(np.int32)
    bbox_max_y = bbox_max_y.astype(np.int32)

    padding_left = abs(min(bbox_min_x, 0))
    padding_top = abs(min(bbox_min_y, 0))
    padding_right = max(bbox_max_x - w, 0)
    padding_bottom = max(bbox_max_y - h, 0)

    crop_left = 0 if padding_left > 0 else bbox_min_x
    crop_top = 0 if padding_top > 0 else bbox_min_y
    crop_right = w if padding_right > 0 else bbox_max_x
    crop_bottom = h if padding_bottom > 0 else bbox_max_y

    return np.array([crop_left, crop_top, crop_right, crop_bottom])


def get_bbox_intrinsics(image_intrinsics: np.ndarray, bbox: dict) -> np.ndarray:
    """Get camera intrinsics centered on bbox."""
    bbox_center_x = bbox["left"] + ((bbox["right"] - bbox["left"]) // 2)
    bbox_center_y = bbox["top"] + ((bbox["bottom"] - bbox["top"]) // 2)

    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics


def pose_bbox_to_full_image(
    pose: np.ndarray, image_intrinsics: np.ndarray, bbox
) -> np.ndarray:
    """Convert pose from bbox coordinates to full image coordinates."""
    bbox = bbox_is_dict(bbox)

    rvec = pose[:3].copy()
    tvec = pose[3:].copy()

    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    focal_length = image_intrinsics[0, 0]

    bbox_width = bbox["right"] - bbox["left"]
    bbox_height = bbox["bottom"] - bbox["top"]
    bbox_size = bbox_width + bbox_height

    tvec[2] *= focal_length / bbox_size

    projected_point = bbox_intrinsics.dot(tvec.T)
    tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))

    rmat = Rotation.from_rotvec(rvec).as_matrix()
    projected_point = bbox_intrinsics.dot(rmat)
    rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    return np.concatenate([rvec, tvec])


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform 3D points using pose."""
    return points.dot(Rotation.from_rotvec(pose[:3]).as_matrix().T) + pose[3:]


def plot_3d_landmark(
    verts: np.ndarray, campose: np.ndarray, intrinsics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D landmarks to 2D."""
    lm_3d_trans = transform_points(verts, campose)

    lms_3d_trans_proj = intrinsics.dot(lm_3d_trans.T).T
    lms_projected = lms_3d_trans_proj[:, :2] / np.tile(lms_3d_trans_proj[:, 2], (2, 1)).T

    return lms_projected, lms_3d_trans_proj


def transform_pose_global_project_bbox(
    boxes: torch.Tensor,
    dofs: torch.Tensor,
    pose_mean: torch.Tensor,
    pose_stddev: torch.Tensor,
    image_shape: Tuple[int, int],
    threed_68_points: torch.Tensor = None,
    bbox_x_factor: float = 1.1,
    bbox_y_factor: float = 1.1,
    expand_forehead: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform pose from bbox to global coordinates and project to get bounding box."""
    if len(dofs) == 0:
        return boxes, dofs

    device = dofs.device

    boxes = boxes.cpu().numpy()
    dofs = dofs.cpu().numpy()

    threed_68_points = threed_68_points.numpy()

    (h, w) = image_shape
    global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    pose_mean = pose_mean.numpy()
    pose_stddev = pose_stddev.numpy()

    dof_mean = pose_mean
    dof_std = pose_stddev
    dofs = dofs * dof_std + dof_mean

    projected_boxes = []
    global_dofs = []

    for i in range(dofs.shape[0]):
        global_dof = pose_bbox_to_full_image(dofs[i], global_intrinsics, boxes[i])
        global_dofs.append(global_dof)

        if threed_68_points is not None:
            projected_lms, _ = plot_3d_landmark(threed_68_points, global_dof, global_intrinsics)
            projected_bbox = expand_bbox_rectangle(
                w,
                h,
                bbox_x_factor=bbox_x_factor,
                bbox_y_factor=bbox_y_factor,
                lms=projected_lms,
                roll=global_dof[2],
                expand_forehead=expand_forehead,
            )
        else:
            projected_bbox = boxes[i]

        projected_boxes.append(projected_bbox)

    global_dofs = torch.from_numpy(np.asarray(global_dofs)).float()
    projected_boxes = torch.from_numpy(np.asarray(projected_boxes)).float()

    return projected_boxes.to(device), global_dofs.to(device)
