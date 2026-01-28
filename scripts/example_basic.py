#!/usr/bin/env python3
"""Basic face detection with img2pose.

Detects faces in a single image and prints pose information for each face.
Model weights are downloaded automatically on first run.

Usage:
    python scripts/example_basic.py --input photo.jpg
    python scripts/example_basic.py --input photo.jpg --threshold 0.8
    python scripts/example_basic.py --input photo.jpg --device cpu
"""

import argparse
import sys

from img2pose import Img2Pose


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces and estimate 6DoF pose in a single image"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)"
    )
    args = parser.parse_args()

    # Initialize detector (downloads weights on first run)
    detector = Img2Pose(device=args.device, score_threshold=args.threshold)

    # Detect faces
    faces = detector.detect_faces(args.input)

    if not faces:
        print("No faces detected.")
        sys.exit(0)

    # Print results
    print(f"Detected {len(faces)} face(s):\n")
    for i, face in enumerate(faces):
        print(f"Face {i + 1}:")
        print(f"  Confidence: {face['confidence']:.3f}")
        print(f"  Box (xyxy): [{face['box'][0]:.1f}, {face['box'][1]:.1f}, "
              f"{face['box'][2]:.1f}, {face['box'][3]:.1f}]")

        # Pose: [rx, ry, rz, tx, ty, tz]
        pose = face["pose"]
        print(f"  Rotation (rad): rx={pose[0]:.3f}, ry={pose[1]:.3f}, rz={pose[2]:.3f}")
        print(f"  Translation: tx={pose[3]:.3f}, ty={pose[4]:.3f}, tz={pose[5]:.3f}")

        # 5-point keypoints
        kp = face["keypoints"]
        print(f"  Keypoints:")
        print(f"    left_eye:    ({kp['left_eye'][0]:.1f}, {kp['left_eye'][1]:.1f})")
        print(f"    right_eye:   ({kp['right_eye'][0]:.1f}, {kp['right_eye'][1]:.1f})")
        print(f"    nose:        ({kp['nose'][0]:.1f}, {kp['nose'][1]:.1f})")
        print(f"    mouth_left:  ({kp['mouth_left'][0]:.1f}, {kp['mouth_left'][1]:.1f})")
        print(f"    mouth_right: ({kp['mouth_right'][0]:.1f}, {kp['mouth_right'][1]:.1f})")
        print()


if __name__ == "__main__":
    main()
