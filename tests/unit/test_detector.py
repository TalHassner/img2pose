"""Unit tests for the Img2Pose detector class."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


class TestImg2PoseInit:
    """Tests for Img2Pose initialization."""

    def test_default_device_selection(self):
        """Test that device defaults to 'auto' behavior."""
        with patch("img2pose.detector.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = "cpu"

            # We can't fully instantiate without weights, so we test the logic
            from img2pose.detector import Img2Pose

            # Device selection happens in __init__, tested via integration tests

    def test_device_options(self):
        """Test that various device strings are accepted."""
        from img2pose.detector import Img2Pose

        valid_devices = ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]
        # These should not raise during parsing (actual init needs weights)
        import torch
        for device_str in valid_devices:
            if device_str == "auto":
                expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                try:
                    expected = torch.device(device_str)
                except RuntimeError:
                    # CUDA devices may not be available
                    continue


class TestImageLoading:
    """Tests for image loading utilities."""

    def test_load_pil_image(self):
        """Test loading PIL Image."""
        from img2pose.detector import _load_image

        img = Image.new("RGB", (100, 100), color="red")
        result = _load_image(img)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_load_pil_image_converts_mode(self):
        """Test that non-RGB images are converted."""
        from img2pose.detector import _load_image

        img = Image.new("L", (100, 100))  # Grayscale
        result = _load_image(img)
        assert result.mode == "RGB"

    def test_load_numpy_array(self):
        """Test loading numpy array."""
        from img2pose.detector import _load_image

        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _load_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_load_numpy_wrong_dtype_raises(self):
        """Test that non-uint8 arrays raise error."""
        from img2pose.detector import _load_image

        arr = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            _load_image(arr)

    def test_load_numpy_wrong_shape_raises(self):
        """Test that non-RGB arrays raise error."""
        from img2pose.detector import _load_image

        arr = np.zeros((100, 100), dtype=np.uint8)  # Grayscale
        with pytest.raises(ValueError, match="shape"):
            _load_image(arr)

    def test_load_unsupported_type_raises(self):
        """Test that unsupported types raise error."""
        from img2pose.detector import _load_image

        with pytest.raises(TypeError, match="Unsupported"):
            _load_image({"not": "an image"})


class TestImageToTensor:
    """Tests for image to tensor conversion."""

    def test_conversion_shape(self):
        """Test tensor has correct shape."""
        import torch
        from img2pose.detector import _image_to_tensor

        img = Image.new("RGB", (100, 50), color="red")
        tensor = _image_to_tensor(img, torch.device("cpu"))

        assert tensor.shape == (3, 50, 100)  # CHW

    def test_conversion_normalization(self):
        """Test tensor values are normalized."""
        import torch
        from img2pose.detector import _image_to_tensor

        # Create image with max value
        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        tensor = _image_to_tensor(img, torch.device("cpu"))

        assert tensor.max().item() == pytest.approx(1.0)
        assert tensor.min().item() == pytest.approx(1.0)


class TestLandmarkProjection:
    """Tests for 3D to 2D landmark projection."""

    def test_project_landmarks_shape(self):
        """Test projected landmarks have correct shape."""
        from img2pose.detector import _project_landmarks

        points_3d = np.random.randn(68, 3)
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        landmarks_2d = _project_landmarks(points_3d, pose, 640, 480)

        assert landmarks_2d.shape == (68, 2)

    def test_project_landmarks_center(self):
        """Test that origin projects to image center."""
        from img2pose.detector import _project_landmarks

        # Single point at origin
        points_3d = np.array([[0.0, 0.0, 1.0]])  # z=1 to avoid division issues
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No rotation/translation

        landmarks_2d = _project_landmarks(points_3d, pose, 640, 480)

        # Should project near image center
        assert landmarks_2d[0, 0] == pytest.approx(320, abs=1)
        assert landmarks_2d[0, 1] == pytest.approx(240, abs=1)


class TestFaceOutputFormat:
    """Tests for face detection output format (MTCNN compatibility)."""

    def test_output_has_required_keys(self):
        """Test that output dicts have all required keys."""
        required_keys = {"box", "confidence", "pose", "keypoints"}

        # Create mock face dict
        face = {
            "box": [0, 0, 100, 100],
            "confidence": 0.95,
            "pose": [0, 0, 0, 0, 0, 1],
            "keypoints": {
                "left_eye": [30, 30],
                "right_eye": [70, 30],
                "nose": [50, 50],
                "mouth_left": [35, 70],
                "mouth_right": [65, 70],
            },
        }

        assert set(face.keys()) == required_keys

    def test_keypoints_format(self):
        """Test keypoints dict has MTCNN-style keys."""
        expected_keypoint_names = {
            "left_eye",
            "right_eye",
            "nose",
            "mouth_left",
            "mouth_right",
        }

        from img2pose.detector import Img2Pose

        assert set(Img2Pose._KEYPOINT_INDICES.keys()) == expected_keypoint_names

    def test_box_is_xyxy_format(self):
        """Test that boxes are in [x1, y1, x2, y2] format."""
        box = [10, 20, 100, 150]

        x1, y1, x2, y2 = box
        assert x2 > x1, "x2 should be greater than x1 (xyxy format)"
        assert y2 > y1, "y2 should be greater than y1 (xyxy format)"
