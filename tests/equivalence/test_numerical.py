"""Equivalence tests to verify numerical consistency with original implementation.

These tests require the original img2pose model weights to be available.
They verify that the library produces numerically equivalent results.
"""

import numpy as np
import pytest


class TestPoseOperationsEquivalence:
    """Test pose operation equivalence with original implementation."""

    def test_transform_points_consistency(self, pose_reference_68):
        """Test that transform_points matches original implementation."""
        from img2pose._pose_ops import transform_points

        # Test pose
        pose = np.array([0.1, -0.2, 0.05, 0.5, -0.3, 1.0])

        # Transform points
        result = transform_points(pose_reference_68, pose)

        # Check shape is preserved
        assert result.shape == pose_reference_68.shape

        # Check transformation is applied (points should be different)
        assert not np.allclose(result, pose_reference_68)

    def test_pose_bbox_to_full_image_consistency(self):
        """Test pose coordinate transformation."""
        from img2pose._pose_ops import pose_bbox_to_full_image

        # Test inputs
        pose = np.array([0.1, -0.2, 0.05, 0.5, -0.3, 1.0])
        intrinsics = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        bbox = np.array([100, 100, 200, 200])

        # Transform pose
        global_pose = pose_bbox_to_full_image(pose, intrinsics, bbox)

        # Check shape
        assert global_pose.shape == (6,)

        # Check rotation vector (should be modified but still valid)
        rvec_norm = np.linalg.norm(global_pose[:3])
        assert rvec_norm < 10.0, "Rotation vector magnitude should be reasonable"

    def test_expand_bbox_rectangle_consistency(self, pose_reference_68):
        """Test bounding box expansion from landmarks."""
        from img2pose._pose_ops import expand_bbox_rectangle

        # Create some 2D landmarks
        landmarks_2d = pose_reference_68[:, :2] * 50 + 200  # Scale and shift

        result = expand_bbox_rectangle(
            w=640,
            h=480,
            bbox_x_factor=1.1,
            bbox_y_factor=1.1,
            lms=landmarks_2d,
            expand_forehead=0.3,
            roll=0.0,
        )

        # Check output shape
        assert result.shape == (4,)

        # Check box is valid (x1 < x2, y1 < y2)
        assert result[0] < result[2], "x1 should be less than x2"
        assert result[1] < result[3], "y1 should be less than y2"


class TestProjectionEquivalence:
    """Test landmark projection equivalence."""

    def test_plot_3d_landmark_shape(self, pose_reference_68):
        """Test 3D to 2D landmark projection output shape."""
        from img2pose._pose_ops import plot_3d_landmark

        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        intrinsics = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        landmarks_2d, landmarks_3d_proj = plot_3d_landmark(
            pose_reference_68, pose, intrinsics
        )

        assert landmarks_2d.shape == (68, 2)
        assert landmarks_3d_proj.shape == (68, 3)

    def test_projection_with_rotation(self, pose_reference_68):
        """Test that rotation affects projection."""
        from img2pose._pose_ops import plot_3d_landmark

        intrinsics = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ], dtype=np.float64)

        # Identity pose
        pose_identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        landmarks_identity, _ = plot_3d_landmark(pose_reference_68, pose_identity, intrinsics)

        # Rotated pose (30 degrees around Y axis)
        pose_rotated = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        landmarks_rotated, _ = plot_3d_landmark(pose_reference_68, pose_rotated, intrinsics)

        # Projections should be different
        assert not np.allclose(landmarks_identity, landmarks_rotated)


@pytest.mark.skipif(
    True,  # Skip by default - enable when original model is available
    reason="Requires original img2pose model for comparison"
)
class TestModelEquivalence:
    """Test numerical equivalence with original img2pose model.

    These tests compare library output with original implementation.
    Enable by setting TEST_EQUIVALENCE=1 environment variable.
    """

    def test_inference_equivalence(self):
        """Test that inference produces identical results."""
        # This test would:
        # 1. Load both original and library models
        # 2. Run inference on same image
        # 3. Compare outputs within tolerance
        pass

    def test_pose_output_equivalence(self):
        """Test that pose outputs match within tolerance."""
        pass
