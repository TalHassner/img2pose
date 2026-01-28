#!/usr/bin/env python3
"""Batch face detection with img2pose.

Processes multiple images in a single batched inference call for efficiency.
This is faster than processing images sequentially when you have multiple images.

Usage:
    python scripts/example_batch.py --input images/*.jpg
    python scripts/example_batch.py --input img1.jpg img2.jpg img3.jpg --output results.json
    python scripts/example_batch.py --input images/ --threshold 0.6
"""

import argparse
import json
import os
import sys
from pathlib import Path

from img2pose import Img2Pose


def collect_images(input_paths):
    """Collect image paths from arguments (files or directories)."""
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for path in input_paths:
        p = Path(path)
        if p.is_dir():
            # Collect all images in directory
            for ext in valid_extensions:
                images.extend(p.glob(f"*{ext}"))
                images.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in valid_extensions:
            images.append(p)

    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(
        description="Batch face detection across multiple images"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input image files or directories"
    )
    parser.add_argument(
        "--output", "-o",
        help="Optional JSON file to save results"
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

    # Collect image paths
    image_paths = collect_images(args.input)
    if not image_paths:
        print("No valid images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)...")

    # Initialize detector
    detector = Img2Pose(device=args.device, score_threshold=args.threshold)

    # Batch inference: all images processed in one call
    all_faces = detector.detect_faces([str(p) for p in image_paths])

    # Print summary and collect results
    results = {}
    total_faces = 0

    for path, faces in zip(image_paths, all_faces):
        total_faces += len(faces)
        print(f"{path.name}: {len(faces)} face(s)")

        # Store results for JSON output
        results[str(path)] = [
            {
                "box": face["box"],
                "confidence": face["confidence"],
                "pose": face["pose"],
            }
            for face in faces
        ]

    print(f"\nTotal: {total_faces} face(s) in {len(image_paths)} image(s)")

    # Save to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
