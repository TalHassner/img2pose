#!/usr/bin/env python3
"""Face alignment with img2pose.

Detects faces and produces aligned, normalized face crops suitable for
face recognition or other downstream tasks. Uses 2D similarity transform
based on predicted facial landmarks.

Usage:
    python scripts/example_alignment.py --input photo.jpg --output aligned/
    python scripts/example_alignment.py --input photo.jpg --output aligned/ --size 112
    python scripts/example_alignment.py --input photos/*.jpg --output aligned/ --threshold 0.8

Output files are named: {original_name}_{face_index}.jpg
"""

import argparse
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
            for ext in valid_extensions:
                images.extend(p.glob(f"*{ext}"))
                images.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in valid_extensions:
            images.append(p)

    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(
        description="Detect and align faces from images"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input image files or directories"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for aligned faces"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        choices=[112, 224],
        default=224,
        help="Output face size in pixels (default: 224)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)"
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)"
    )
    args = parser.parse_args()

    # Collect images
    image_paths = collect_images(args.input)
    if not image_paths:
        print("No valid images found.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_paths)} image(s)...")

    # Initialize detector
    detector = Img2Pose(device=args.device, score_threshold=args.threshold)

    # Process each image
    total_saved = 0
    for image_path in image_paths:
        # Detect and align in one call
        results = detector.detect_and_align(
            str(image_path),
            output_size=args.size,
        )

        if not results:
            print(f"{image_path.name}: no faces")
            continue

        # Save aligned faces
        stem = image_path.stem
        for idx, face_data in enumerate(results):
            output_name = f"{stem}_{idx}.{args.format}"
            output_path = output_dir / output_name
            face_data["image"].save(output_path)
            total_saved += 1

        print(f"{image_path.name}: {len(results)} face(s) saved")

    print(f"\nTotal: {total_saved} aligned face(s) saved to {output_dir}")


if __name__ == "__main__":
    main()
