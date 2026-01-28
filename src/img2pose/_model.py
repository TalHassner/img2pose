"""Internal model wrapper for img2pose.

This module provides the img2poseModel class that wraps FasterDoFRCNN
for inference-only face detection and 6DoF pose estimation.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from ._models import FasterDoFRCNN


class WrappedModel(Module):
    """Wrapper for CPU inference compatibility."""

    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, images: List[Tensor], targets=None):
        return self.module(images, targets)


class img2poseModel:
    """Internal model class for face detection and 6DoF pose estimation.

    This class wraps FasterDoFRCNN and handles device placement and inference.
    For the public API, use the Img2Pose class instead.
    """

    def __init__(
        self,
        depth: int,
        min_size: int,
        max_size: int,
        device: Optional[Union[str, torch.device]] = None,
        pose_mean: Optional[np.ndarray] = None,
        pose_stddev: Optional[np.ndarray] = None,
        threed_68_points: Optional[np.ndarray] = None,
        threed_5_points: Optional[np.ndarray] = None,
        rpn_pre_nms_top_n_test: int = 6000,
        rpn_post_nms_top_n_test: int = 1000,
        bbox_x_factor: float = 1.1,
        bbox_y_factor: float = 1.1,
        expand_forehead: float = 0.3,
    ):
        """Initialize the model.

        Args:
            depth: ResNet backbone depth (18, 50, or 101)
            min_size: Minimum image size for transform
            max_size: Maximum image size for transform
            device: Device for inference ("cpu", "cuda", "cuda:0", etc.)
            pose_mean: Mean pose for normalization
            pose_stddev: Std pose for normalization
            threed_68_points: 3D reference points for 68 landmarks
            threed_5_points: 3D reference points for 5 landmarks
            rpn_pre_nms_top_n_test: Pre-NMS proposals to keep
            rpn_post_nms_top_n_test: Post-NMS proposals to keep
            bbox_x_factor: Bounding box expansion factor (x)
            bbox_y_factor: Bounding box expansion factor (y)
            expand_forehead: Forehead expansion factor
        """
        self.depth = depth
        self.min_size = min_size
        self.max_size = max_size

        # Handle device selection
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Create backbone with pretrained=False (we'll load our own weights)
        # Use weights=None for newer torchvision versions
        try:
            backbone = resnet_fpn_backbone(f"resnet{self.depth}", weights=None)
        except TypeError:
            # Fallback for older torchvision versions
            backbone = resnet_fpn_backbone(f"resnet{self.depth}", pretrained=False)

        # Convert numpy arrays to tensors
        pose_mean_tensor = None
        pose_stddev_tensor = None
        threed_68_points_tensor = None
        threed_5_points_tensor = None

        if pose_mean is not None:
            pose_mean_tensor = torch.tensor(pose_mean, dtype=torch.float32)
            pose_stddev_tensor = torch.tensor(pose_stddev, dtype=torch.float32)

        if threed_68_points is not None:
            threed_68_points_tensor = torch.tensor(threed_68_points, dtype=torch.float32)

        if threed_5_points is not None:
            threed_5_points_tensor = torch.tensor(threed_5_points, dtype=torch.float32)

        # Create the FPN model
        self.fpn_model = FasterDoFRCNN(
            backbone,
            num_classes=2,
            min_size=self.min_size,
            max_size=self.max_size,
            pose_mean=pose_mean_tensor,
            pose_stddev=pose_stddev_tensor,
            threed_68_points=threed_68_points_tensor,
            threed_5_points=threed_5_points_tensor,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            bbox_x_factor=bbox_x_factor,
            bbox_y_factor=bbox_y_factor,
            expand_forehead=expand_forehead,
        )

        # Store reference to model without wrappers
        self.fpn_model_without_ddp = self.fpn_model

        # Wrap model based on device
        if str(self.device) == "cpu":
            self.fpn_model = WrappedModel(self.fpn_model)
            self.fpn_model_without_ddp = self.fpn_model
        else:
            # Use DataParallel for GPU (even single GPU for consistency)
            self.fpn_model = DataParallel(self.fpn_model)
            self.fpn_model = self.fpn_model.to(self.device)
            self.fpn_model_without_ddp = self.fpn_model

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Load model weights from state dict.

        Args:
            state_dict: Model state dict (typically from checkpoint["fpn_model"])
        """
        if str(self.device) == "cpu":
            # For CPU, the model is wrapped in WrappedModel
            self.fpn_model.module.load_state_dict(state_dict)
        else:
            # For GPU with DataParallel
            self.fpn_model.module.load_state_dict(state_dict)

    def evaluate(self) -> None:
        """Set model to evaluation mode."""
        self.fpn_model.eval()

    def predict(self, imgs: List[Tensor]) -> List[Dict[str, Tensor]]:
        """Run inference on a list of image tensors.

        Args:
            imgs: List of image tensors [C, H, W] on the model's device

        Returns:
            List of detection dicts with keys: boxes, labels, scores, dofs
        """
        assert not self.fpn_model.training, "Model must be in eval mode"

        with torch.no_grad():
            predictions = self.fpn_model(imgs)

        return predictions
