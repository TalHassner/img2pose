"""Internal Generalized R-CNN framework for img2pose.

This module contains the base R-CNN architecture adapted for 6DoF pose estimation.
"""

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


class GeneralizedRCNN(nn.Module):
    """Base class for Generalized R-CNN with 6DoF head.

    Arguments:
        backbone: Feature extraction backbone
        rpn: Region Proposal Network
        roi_heads: ROI heads for classification and pose regression
        transform: Input transformation module
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        roi_heads: nn.Module,
        transform: nn.Module,
    ):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self,
        losses: Dict[str, Tensor],
        detections: List[Dict[str, Tensor]],
        evaluating: bool,
    ):
        if evaluating:
            return losses
        return detections

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> List[Dict[str, Tensor]]:
        """Forward pass for inference.

        Arguments:
            images: List of image tensors [C, H, W]
            targets: Not used in inference (kept for compatibility)

        Returns:
            List of detection dicts with keys: boxes, labels, scores, dofs
        """
        if self.training:
            raise RuntimeError(
                "img2pose library is inference-only. "
                "For training, use the original repository."
            )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return ({}, detections)
        else:
            return self.eager_outputs({}, detections, targets is not None)
