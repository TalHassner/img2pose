#!/usr/bin/env python3
"""Convert img2pose checkpoint to lean inference-only weights.

This script strips optimizer state from training checkpoints to create
smaller model files suitable for production inference.

Usage:
    python scripts/convert_to_lean_model.py \
        --input models/img2pose_v1.pth \
        --output models/img2pose_v1_lean.pth

The lean model contains only the 'fpn_model' state dict, removing:
- Optimizer state (~85MB if Adam was used)
- Learning rate scheduler state
- Training metadata
"""

import argparse
import os
import sys

import torch


def convert_to_lean_model(input_path: str, output_path: str, verify: bool = True) -> dict:
    """Convert a training checkpoint to lean inference weights.

    Args:
        input_path: Path to the original checkpoint
        output_path: Path to save the lean model
        verify: Whether to verify the conversion

    Returns:
        Dict with conversion statistics
    """
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Analyze checkpoint contents
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    if "fpn_model" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'fpn_model' key. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    fpn_model = checkpoint["fpn_model"]

    # Calculate sizes
    def get_state_dict_size(state_dict):
        total_bytes = 0
        total_params = 0
        for key, tensor in state_dict.items():
            total_bytes += tensor.numel() * tensor.element_size()
            total_params += tensor.numel()
        return total_bytes, total_params

    model_bytes, model_params = get_state_dict_size(fpn_model)
    print(f"\nModel weights: {model_bytes / 1e6:.2f} MB ({model_params:,} parameters)")

    # Check for optimizer state
    has_optimizer = "optimizer" in checkpoint
    if has_optimizer:
        opt_state = checkpoint["optimizer"]
        if isinstance(opt_state, dict) and "state" in opt_state:
            opt_bytes = sum(
                v.numel() * v.element_size()
                for s in opt_state["state"].values()
                for v in s.values()
                if isinstance(v, torch.Tensor)
            )
            print(f"Optimizer state: {opt_bytes / 1e6:.2f} MB (will be removed)")

    # Create lean model with only inference weights
    lean_weights = {"fpn_model": fpn_model}

    # Save with optimized serialization
    print(f"\nSaving lean model to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(lean_weights, output_path, _use_new_zipfile_serialization=True)

    # Report size reduction
    orig_size = os.path.getsize(input_path)
    lean_size = os.path.getsize(output_path)
    reduction = 100 * (1 - lean_size / orig_size)

    print(f"\nSize comparison:")
    print(f"  Original: {orig_size / 1e6:.2f} MB")
    print(f"  Lean:     {lean_size / 1e6:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")

    # Verify the lean model can be loaded
    if verify:
        print("\nVerifying lean model...")
        loaded = torch.load(output_path, map_location="cpu")
        assert "fpn_model" in loaded, "Lean model missing 'fpn_model' key"
        assert "optimizer" not in loaded, "Lean model should not contain optimizer"

        # Verify all keys match
        orig_keys = set(fpn_model.keys())
        loaded_keys = set(loaded["fpn_model"].keys())
        assert orig_keys == loaded_keys, f"Key mismatch: {orig_keys ^ loaded_keys}"

        # Verify values match
        for key in fpn_model:
            assert torch.equal(fpn_model[key], loaded["fpn_model"][key]), f"Value mismatch for {key}"

        print("Verification passed!")

    return {
        "original_size_mb": orig_size / 1e6,
        "lean_size_mb": lean_size / 1e6,
        "reduction_percent": reduction,
        "num_parameters": model_params,
        "num_tensors": len(fpn_model),
        "had_optimizer": has_optimizer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert img2pose checkpoint to lean inference weights"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the original checkpoint (e.g., models/img2pose_v1.pth)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save the lean model (e.g., models/img2pose_v1_lean.pth)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of the converted model"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(args.output):
        response = input(f"Output file exists: {args.output}. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        stats = convert_to_lean_model(
            args.input,
            args.output,
            verify=not args.no_verify
        )
        print(f"\nConversion successful!")
        print(f"Lean model saved to: {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
