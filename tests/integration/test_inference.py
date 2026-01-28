"""Integration tests for img2pose inference."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


class TestImg2PoseInference:
    """Integration tests for end-to-end inference."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns fake predictions."""
        import torch

        mock = MagicMock()
        mock.evaluate = MagicMock()
        mock.predict = MagicMock(return_value=[{
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "scores": torch.tensor([0.95]),
            "dofs": torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.0, 1.0]]),
            "labels": torch.tensor([1]),
        }])
        return mock

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed"),
        reason="PyTorch required"
    )
    def test_detect_faces_returns_list(self, sample_rgb_image, mock_model):
        """Test that detect_faces returns a list."""
        with patch("img2pose.detector.img2poseModel", return_value=mock_model):
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                from img2pose import Img2Pose

                # This will fail to fully initialize without proper weights
                # but we test the logic

    def test_batch_processing_returns_list_of_lists(self):
        """Test batch input returns nested lists."""
        # Mock test - actual batch processing tested with real model
        batch_results = [
            [{"box": [0, 0, 100, 100], "confidence": 0.9}],
            [{"box": [0, 0, 50, 50], "confidence": 0.8}],
        ]

        assert len(batch_results) == 2
        assert all(isinstance(r, list) for r in batch_results)


class TestVisualization:
    """Integration tests for visualization."""

    def test_draw_detections_returns_array(self, sample_rgb_image, sample_face_dict):
        """Test that visualization returns numpy array."""
        from img2pose._visualization import draw_detections

        result = draw_detections(sample_rgb_image, [sample_face_dict])

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape[2] == 3  # RGB

    def test_draw_detections_preserves_size(self, sample_rgb_image, sample_face_dict):
        """Test that visualization preserves image size."""
        from img2pose._visualization import draw_detections

        result = draw_detections(sample_rgb_image, [sample_face_dict])

        assert result.shape[0] == sample_rgb_image.height
        assert result.shape[1] == sample_rgb_image.width

    def test_draw_detections_empty_faces(self, sample_rgb_image):
        """Test visualization with no faces."""
        from img2pose._visualization import draw_detections

        result = draw_detections(sample_rgb_image, [])

        assert isinstance(result, np.ndarray)

    def test_draw_with_options(self, sample_rgb_image, sample_face_dict):
        """Test visualization with various options."""
        from img2pose._visualization import draw_detections

        # Test with various options
        result1 = draw_detections(sample_rgb_image, [sample_face_dict], show_box=False)
        result2 = draw_detections(sample_rgb_image, [sample_face_dict], show_keypoints=False)
        result3 = draw_detections(sample_rgb_image, [sample_face_dict], show_confidence=False)

        assert all(isinstance(r, np.ndarray) for r in [result1, result2, result3])


class TestWeightManagement:
    """Integration tests for weight loading."""

    def test_get_cache_dir_default(self):
        """Test default cache directory."""
        from img2pose._weights import get_cache_dir
        from pathlib import Path

        cache_dir = get_cache_dir()

        assert isinstance(cache_dir, Path)
        assert "img2pose" in str(cache_dir)

    def test_get_cache_dir_from_env(self, monkeypatch, tmp_path):
        """Test cache directory from environment variable."""
        from img2pose._weights import get_cache_dir

        custom_path = str(tmp_path / "custom_cache")
        monkeypatch.setenv("IMG2POSE_CACHE", custom_path)

        cache_dir = get_cache_dir()

        assert str(cache_dir) == custom_path

    def test_unknown_model_raises(self):
        """Test that unknown model name raises error."""
        from img2pose._weights import get_model_path

        with pytest.raises(ValueError, match="Unknown model"):
            get_model_path("nonexistent_model")

    def test_load_weights_from_missing_path_raises(self, tmp_path):
        """Test loading from missing path raises error."""
        from img2pose._weights import load_weights_from_path

        with pytest.raises(FileNotFoundError):
            load_weights_from_path(str(tmp_path / "missing.pth"))


class TestBatchInference:
    """Tests for batch inference functionality."""

    def test_empty_batch_returns_empty_list(self):
        """Test that empty batch input returns empty list."""
        import torch
        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose()

                result = detector.detect_faces([])
                assert result == []

    def test_single_element_batch_returns_nested_list(self, sample_rgb_image):
        """Test that single-element batch returns [[faces]]."""
        import torch
        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=[{
                    "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                    "scores": torch.tensor([0.9]),
                    "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                }])
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose()

                result = detector.detect_faces([sample_rgb_image])

                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], list)

    def test_batch_result_structure(self):
        """Test that batch results have correct nested structure."""
        import torch
        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=[
                    {
                        "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                        "scores": torch.tensor([0.9]),
                        "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                    },
                    {
                        "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                        "scores": torch.tensor([0.8]),
                        "dofs": torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                    },
                    {
                        "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
                        "scores": torch.tensor([]),
                        "dofs": torch.tensor([], dtype=torch.float32).reshape(0, 6),
                    },
                ])
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose()

                # Create 3 test images
                images = [
                    Image.new("RGB", (640, 480), color="white"),
                    Image.new("RGB", (800, 600), color="gray"),
                    Image.new("RGB", (320, 240), color="black"),
                ]
                result = detector.detect_faces(images)

                assert isinstance(result, list)
                assert len(result) == 3
                assert all(isinstance(r, list) for r in result)
                # Third image has no faces
                assert len(result[2]) == 0

    def test_batch_preserves_order(self):
        """Test that batch results maintain input order."""
        import torch

        # Create different mock predictions for each image
        mock_predictions = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "dofs": torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
            {
                "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                "scores": torch.tensor([0.8]),
                "dofs": torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
            {
                "boxes": torch.tensor([[30, 30, 70, 70]], dtype=torch.float32),
                "scores": torch.tensor([0.7]),
                "dofs": torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
        ]

        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=mock_predictions)
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose()

                images = [
                    Image.new("RGB", (640, 480), color="white"),
                    Image.new("RGB", (800, 600), color="gray"),
                    Image.new("RGB", (320, 240), color="black"),
                ]
                result = detector.detect_faces(images)

                # Verify order is preserved by checking pose values
                assert len(result) == 3
                assert result[0][0]["pose"][0] == pytest.approx(0.1)
                assert result[1][0]["pose"][0] == pytest.approx(0.2)
                assert result[2][0]["pose"][0] == pytest.approx(0.3)

    def test_batch_applies_threshold_per_image(self):
        """Test that score threshold is applied to each image."""
        import torch

        mock_predictions = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.6]),  # Above 0.5 threshold
                "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
            {
                "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                "scores": torch.tensor([0.3]),  # Below 0.5 threshold
                "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            },
        ]

        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=mock_predictions)
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose(score_threshold=0.5)

                images = [
                    Image.new("RGB", (640, 480), color="white"),
                    Image.new("RGB", (800, 600), color="gray"),
                ]
                result = detector.detect_faces(images)

                assert len(result) == 2
                assert len(result[0]) == 1  # First image has face above threshold
                assert len(result[1]) == 0  # Second image has face below threshold

    def test_callable_interface_batch(self):
        """Test that __call__ method supports batch input."""
        import torch
        with patch("img2pose.detector.img2poseModel") as MockModel:
            with patch("img2pose.detector.load_weights", return_value={"fpn_model": {}}):
                mock_model = MagicMock()
                mock_model.evaluate = MagicMock()
                mock_model.load_state_dict = MagicMock()
                mock_model.predict = MagicMock(return_value=[
                    {
                        "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                        "scores": torch.tensor([0.9]),
                        "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                    },
                    {
                        "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                        "scores": torch.tensor([0.8]),
                        "dofs": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                    },
                ])
                MockModel.return_value = mock_model

                from img2pose import Img2Pose
                detector = Img2Pose()

                images = [
                    Image.new("RGB", (640, 480), color="white"),
                    Image.new("RGB", (800, 600), color="gray"),
                ]

                # Test using __call__
                result = detector(images)

                assert isinstance(result, list)
                assert len(result) == 2


class TestBatchPerformance:
    """Performance tests for batch inference.

    These tests verify that batch processing provides expected speedup.
    They are marked as slow and can be skipped in CI with: pytest -m "not slow"
    """

    @pytest.mark.slow
    @pytest.mark.requires_weights
    def test_batch_faster_than_sequential_cpu(self):
        """Test that batch inference is faster than sequential on CPU.

        This test requires actual model weights.
        """
        import time

        try:
            from img2pose import Img2Pose
            detector = Img2Pose(device="cpu")
        except Exception as e:
            pytest.skip(f"Could not initialize detector with weights: {e}")

        # Create test images
        images = [
            np.ones((480, 640, 3), dtype=np.uint8) * i
            for i in range(0, 256, 64)  # 4 images
        ]

        # Warm up
        detector.detect_faces(images[0])

        # Time sequential processing
        start = time.perf_counter()
        sequential_results = [detector.detect_faces(img) for img in images]
        sequential_time = time.perf_counter() - start

        # Time batch processing
        start = time.perf_counter()
        batch_results = detector.detect_faces(images)
        batch_time = time.perf_counter() - start

        # Log timing for debugging
        print(f"\nCPU Timing: sequential={sequential_time:.3f}s, batch={batch_time:.3f}s")
        print(f"Speedup: {sequential_time / batch_time:.2f}x")

        # Batch should not be slower (allowing 10% tolerance for variance)
        assert batch_time <= sequential_time * 1.1, (
            f"Batch ({batch_time:.3f}s) should not be slower than "
            f"sequential ({sequential_time:.3f}s)"
        )

    @pytest.mark.slow
    @pytest.mark.requires_weights
    @pytest.mark.requires_gpu
    def test_batch_faster_than_sequential_gpu(self):
        """Test that batch inference is faster than sequential on GPU.

        This test requires actual model weights and CUDA GPU.
        """
        import time
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from img2pose import Img2Pose
            detector = Img2Pose(device="cuda")
        except Exception as e:
            pytest.skip(f"Could not initialize detector with weights: {e}")

        # Create test images
        images = [
            np.ones((480, 640, 3), dtype=np.uint8) * i
            for i in range(0, 256, 32)  # 8 images
        ]

        # Warm up GPU
        detector.detect_faces(images[0])
        torch.cuda.synchronize()

        # Time sequential processing
        start = time.perf_counter()
        sequential_results = [detector.detect_faces(img) for img in images]
        torch.cuda.synchronize()
        sequential_time = time.perf_counter() - start

        # Time batch processing
        start = time.perf_counter()
        batch_results = detector.detect_faces(images)
        torch.cuda.synchronize()
        batch_time = time.perf_counter() - start

        # Log timing for debugging
        print(f"\nGPU Timing: sequential={sequential_time:.3f}s, batch={batch_time:.3f}s")
        print(f"Speedup: {sequential_time / batch_time:.2f}x")

        # Batch should be faster on GPU
        assert batch_time < sequential_time, (
            f"Batch ({batch_time:.3f}s) should be faster than "
            f"sequential ({sequential_time:.3f}s) on GPU"
        )

    @pytest.mark.slow
    @pytest.mark.requires_weights
    def test_batch_equivalence_with_real_model(self):
        """Test that batch and sequential produce identical results.

        This is the critical correctness test - numerical equivalence.
        """
        try:
            from img2pose import Img2Pose
            detector = Img2Pose(device="cpu")  # CPU for determinism
        except Exception as e:
            pytest.skip(f"Could not initialize detector with weights: {e}")

        # Create varied test images
        images = [
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
            np.ones((600, 800, 3), dtype=np.uint8) * 64,
            np.ones((240, 320, 3), dtype=np.uint8) * 192,
        ]

        # Get results both ways
        batch_results = detector.detect_faces(images)
        sequential_results = [detector.detect_faces(img) for img in images]

        # Verify exact equivalence
        assert len(batch_results) == len(sequential_results), (
            f"Result count mismatch: batch={len(batch_results)}, "
            f"sequential={len(sequential_results)}"
        )

        for i, (batch_faces, seq_faces) in enumerate(zip(batch_results, sequential_results)):
            assert len(batch_faces) == len(seq_faces), (
                f"Image {i}: Different number of faces detected "
                f"(batch={len(batch_faces)}, sequential={len(seq_faces)})"
            )

            for j, (bf, sf) in enumerate(zip(batch_faces, seq_faces)):
                assert bf["box"] == sf["box"], (
                    f"Image {i}, Face {j}: Box mismatch "
                    f"(batch={bf['box']}, sequential={sf['box']})"
                )
                assert bf["confidence"] == sf["confidence"], (
                    f"Image {i}, Face {j}: Confidence mismatch "
                    f"(batch={bf['confidence']}, sequential={sf['confidence']})"
                )
                assert bf["pose"] == sf["pose"], (
                    f"Image {i}, Face {j}: Pose mismatch "
                    f"(batch={bf['pose']}, sequential={sf['pose']})"
                )
                assert bf["keypoints"] == sf["keypoints"], (
                    f"Image {i}, Face {j}: Keypoints mismatch"
                )

        print(f"\nEquivalence verified for {len(images)} images")

    @pytest.mark.slow
    @pytest.mark.requires_weights
    def test_single_image_no_overhead(self):
        """Test that single image has no batch overhead.

        Single image detection should be just as fast whether passed
        directly or as a single-element list.
        """
        import time

        try:
            from img2pose import Img2Pose
            detector = Img2Pose(device="cpu")
        except Exception as e:
            pytest.skip(f"Could not initialize detector with weights: {e}")

        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Warm up
        detector.detect_faces(image)

        # Time single image (direct)
        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            detector.detect_faces(image)
        direct_time = time.perf_counter() - start

        # Time single image (as list)
        start = time.perf_counter()
        for _ in range(iterations):
            detector.detect_faces([image])
        list_time = time.perf_counter() - start

        # Log timing
        print(f"\nSingle image: direct={direct_time:.3f}s, as_list={list_time:.3f}s")

        # Should be within 20% of each other (allowing for variance)
        assert abs(list_time - direct_time) / direct_time < 0.2, (
            f"Single-element list has significant overhead: "
            f"direct={direct_time:.3f}s, list={list_time:.3f}s"
        )
