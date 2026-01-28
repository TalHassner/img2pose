#!/usr/bin/env python3
"""Test script to compare sequential vs batch inference performance.

Usage:
    python test_batch_inference.py /path/to/images
    python test_batch_inference.py /path/to/images --output /path/to/output
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from img2pose import Img2Pose


def load_images(folder: Path) -> list:
    """Load all images from a folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_paths:
        raise ValueError(f"No images found in {folder}")

    print(f"Found {len(image_paths)} images")
    return image_paths


def save_results(results: list, image_paths: list, output_dir: Path, prefix: str, detector: Img2Pose):
    """Save detection results as text files and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, faces in zip(image_paths, results):
        # Save text results
        output_file = output_dir / f"{prefix}_{path.stem}.txt"
        with open(output_file, "w") as f:
            f.write(f"# Detections for {path.name}\n")
            f.write(f"# Faces detected: {len(faces)}\n\n")
            for i, face in enumerate(faces):
                f.write(f"Face {i + 1}:\n")
                f.write(f"  box: {face['box']}\n")
                f.write(f"  confidence: {face['confidence']:.4f}\n")
                f.write(f"  pose: {[f'{v:.4f}' for v in face['pose']]}\n")
                f.write(f"  keypoints:\n")
                for name, coords in face["keypoints"].items():
                    f.write(f"    {name}: {[f'{v:.2f}' for v in coords]}\n")
                f.write("\n")

        # Save visualization
        vis_file = output_dir / f"{prefix}_{path.stem}.jpg"
        vis_image = detector.visualize(path, faces)
        Image.fromarray(vis_image).save(vis_file)


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential vs batch inference performance"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing images to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(".output"),
        help="Output folder for results (default: .output/)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda (default: auto)"
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")

    # Load images
    image_paths = load_images(args.folder)
    images = [Image.open(p).convert("RGB") for p in image_paths]

    # Initialize detector
    print(f"\nInitializing detector (device={args.device})...")
    detector = Img2Pose(device=args.device)

    # Warm up
    print("Warming up...")
    _ = detector.detect_faces(images[0])

    # Sequential inference
    print(f"\nRunning sequential inference on {len(images)} images...")
    start = time.perf_counter()
    sequential_results = [detector.detect_faces(img) for img in images]
    sequential_time = time.perf_counter() - start

    # Batch inference
    print(f"Running batch inference on {len(images)} images...")
    start = time.perf_counter()
    batch_results = detector.detect_faces(images)
    batch_time = time.perf_counter() - start

    # Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Images processed: {len(images)}")
    print(f"Sequential time:  {sequential_time:.3f}s ({sequential_time/len(images):.3f}s per image)")
    print(f"Batch time:       {batch_time:.3f}s ({batch_time/len(images):.3f}s per image)")
    print(f"Speedup:          {sequential_time / batch_time:.2f}x")
    print("=" * 50)

    # Verify results match
    total_seq = sum(len(faces) for faces in sequential_results)
    total_batch = sum(len(faces) for faces in batch_results)
    print(f"\nFaces detected (sequential): {total_seq}")
    print(f"Faces detected (batch):      {total_batch}")

    # Save results
    print(f"\nSaving results to {args.output}/")
    save_results(sequential_results, image_paths, args.output, "sequential", detector)
    save_results(batch_results, image_paths, args.output, "batch", detector)
    print("Done!")


if __name__ == "__main__":
    main()
