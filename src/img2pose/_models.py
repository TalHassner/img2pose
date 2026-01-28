"""Internal model definitions for img2pose.

This module contains the FasterDoFRCNN model and DOFRoIHeads for 6DoF pose estimation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.models.detection._utils as det_utils
from torch import Tensor, nn
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from ._pose_ops import transform_pose_global_project_bbox
from ._rcnn import GeneralizedRCNN
from ._rpn import AnchorGenerator, RegionProposalNetwork, RPNHead


class FastRCNNDoFPredictor(nn.Module):
    """6DoF pose predictor head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        hidden_layer = 256
        self.dof_pred = nn.Sequential(
            nn.Linear(in_channels, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, num_classes * 6),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        dof = self.dof_pred(x)
        return dof


class FastRCNNClassPredictor(nn.Module):
    """Face/background classification head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        return scores


class DOFRoIHeads(RoIHeads):
    """ROI Heads with 6DoF pose regression instead of bounding box regression."""

    def __init__(
        self,
        box_roi_pool: MultiScaleRoIAlign,
        box_head: nn.Module,
        box_predictor: nn.Module,
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        bbox_reg_weights: Optional[Tuple[float, ...]],
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        out_channels: int,
        mask_roi_pool: Optional[nn.Module] = None,
        mask_head: Optional[nn.Module] = None,
        mask_predictor: Optional[nn.Module] = None,
        keypoint_roi_pool: Optional[nn.Module] = None,
        keypoint_head: Optional[nn.Module] = None,
        keypoint_predictor: Optional[nn.Module] = None,
        pose_mean: Optional[Tensor] = None,
        pose_stddev: Optional[Tensor] = None,
        threed_68_points: Optional[Tensor] = None,
        threed_5_points: Optional[Tensor] = None,
        bbox_x_factor: float = 1.1,
        bbox_y_factor: float = 1.1,
        expand_forehead: float = 0.3,
    ):
        # Initialize parent class without calling its __init__
        nn.Module.__init__(self)

        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        num_classes = 2
        self.class_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        self.class_head = TwoMLPHead(out_channels * resolution**2, representation_size)
        self.class_predictor = FastRCNNClassPredictor(representation_size, num_classes)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.pose_mean = pose_mean
        self.pose_stddev = pose_stddev
        self.threed_68_points = threed_68_points
        self.threed_5_points = threed_5_points

        self.bbox_x_factor = bbox_x_factor
        self.bbox_y_factor = bbox_y_factor
        self.expand_forehead = expand_forehead

    def postprocess_detections(
        self,
        class_logits: Tensor,
        dof_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Post-process detections to get final boxes, poses, and scores."""
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = torch.cat(proposals, dim=0)
        N = dof_regression.shape[0]
        pred_boxes = pred_boxes.reshape(N, -1, 4)
        pred_dofs = dof_regression.reshape(N, -1, 6)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_dofs_list = pred_dofs.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_dofs = []

        for boxes, dofs, scores, image_shape in zip(
            pred_boxes_list, pred_dofs_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # Remove background predictions
            dofs = dofs[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            dofs = dofs.reshape(-1, 6)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # Remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, dofs, scores, labels = (
                boxes[inds],
                dofs[inds],
                scores[inds],
                labels[inds],
            )

            # Remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, dofs, scores, labels = (
                boxes[keep],
                dofs[keep],
                scores[keep],
                labels[keep],
            )

            # Create boxes from predicted poses
            boxes, dofs = transform_pose_global_project_bbox(
                boxes,
                dofs,
                self.pose_mean,
                self.pose_stddev,
                image_shape,
                self.threed_68_points,
                bbox_x_factor=self.bbox_x_factor,
                bbox_y_factor=self.bbox_y_factor,
                expand_forehead=self.expand_forehead,
            )

            # Non-maximum suppression
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            boxes, dofs, scores, labels = (
                boxes[keep],
                dofs[keep],
                scores[keep],
                labels[keep],
            )

            # Keep only top-k
            keep = keep[: self.detections_per_img]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_dofs.append(dofs)

        return all_boxes, all_dofs, all_scores, all_labels

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Forward pass for inference."""
        if self.training or targets is not None:
            raise RuntimeError(
                "DOFRoIHeads in img2pose library is inference-only. "
                "For training, use the original repository."
            )

        # Inference path
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        dof_regression = self.box_predictor(box_features)

        class_features = self.class_roi_pool(features, proposals, image_shapes)
        class_features = self.class_head(class_features)
        class_logits = self.class_predictor(class_features)

        boxes, dofs, scores, labels = self.postprocess_detections(
            class_logits, dof_regression, proposals, image_shapes
        )

        result: List[Dict[str, Tensor]] = []
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "dofs": dofs[i],
                }
            )

        return result, {}


class FasterDoFRCNN(GeneralizedRCNN):
    """Faster R-CNN with 6DoF pose regression head.

    This model performs face detection and 6DoF pose estimation in a single forward pass.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: Optional[int] = None,
        # Transform parameters
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # RPN parameters
        rpn_anchor_generator: Optional[AnchorGenerator] = None,
        rpn_head: Optional[RPNHead] = None,
        rpn_pre_nms_top_n_train: int = 6000,
        rpn_pre_nms_top_n_test: int = 6000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.4,
        rpn_fg_iou_thresh: float = 0.5,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        # Box parameters
        box_roi_pool: Optional[MultiScaleRoIAlign] = None,
        box_head: Optional[nn.Module] = None,
        box_predictor: Optional[nn.Module] = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 1000,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        bbox_reg_weights: Optional[Tuple[float, ...]] = None,
        # Pose parameters
        pose_mean: Optional[Tensor] = None,
        pose_stddev: Optional[Tensor] = None,
        threed_68_points: Optional[Tensor] = None,
        threed_5_points: Optional[Tensor] = None,
        bbox_x_factor: float = 1.1,
        bbox_y_factor: float = 1.1,
        expand_forehead: float = 0.3,
    ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = {
            "training": rpn_pre_nms_top_n_train,
            "testing": rpn_pre_nms_top_n_test,
        }
        rpn_post_nms_top_n = {
            "training": rpn_post_nms_top_n_train,
            "testing": rpn_post_nms_top_n_test,
        }

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNDoFPredictor(representation_size, num_classes)

        roi_heads = DOFRoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            out_channels,
            pose_mean=pose_mean,
            pose_stddev=pose_stddev,
            threed_68_points=threed_68_points,
            threed_5_points=threed_5_points,
            bbox_x_factor=bbox_x_factor,
            bbox_y_factor=bbox_y_factor,
            expand_forehead=expand_forehead,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_heads, transform)

    def set_max_min_size(self, max_size: int, min_size: int) -> None:
        """Update the input size constraints."""
        self.min_size = (min_size,)
        self.max_size = max_size
        self.transform.min_size = self.min_size
        self.transform.max_size = self.max_size
