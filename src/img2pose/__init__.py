"""img2pose: Real-time 6DoF face pose estimation.

This library provides face detection with 6 degrees of freedom (6DoF) pose estimation,
performing face detection and alignment without requiring prior face detection or
facial landmark localization.

Example usage:
    >>> from img2pose import Img2Pose
    >>> detector = Img2Pose()
    >>> faces = detector.detect_faces("image.jpg")
    >>> for face in faces:
    ...     print(f"Box: {face['box']}, Confidence: {face['confidence']:.2f}")
    ...     print(f"Pose: {face['pose']}")

The output format follows MTCNN/RetinaFace conventions for compatibility.
"""

import warnings

# Suppress torch.meshgrid deprecation warning (internal to PyTorch RPN)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
    category=UserWarning,
)

# Suppress torchvision backbone_name positional argument warning
warnings.filterwarnings(
    "ignore",
    message="Using 'backbone_name' as positional parameter",
    category=UserWarning,
)

__version__ = "1.0.0"
__author__ = "Vitor Albiero"

from .detector import Img2Pose

__all__ = ["Img2Pose", "__version__"]
