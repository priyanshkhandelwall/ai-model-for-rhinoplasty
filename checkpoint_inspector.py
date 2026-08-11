"""
checkpoint_inspector.py
========================

Phase 0 diagnostic tool: inspect the raw structure of a PyTorch checkpoint
*without* instantiating any model class.

Background
----------
This project started with an undocumented, pre-trained Pix2Pix generator
checkpoint (a ``*_net_G.pth`` file) and no reliable record of which
generator architecture, channel counts, or normalization scheme it was
trained with. Before writing any model-loading code, the checkpoint itself
had to be treated as a source of ground truth: a raw ``state_dict`` is just
an ``OrderedDict`` mapping layer names to tensors, and it can be inspected
directly (via ``torch.load``) with no dependency on the original training
code.

This script:
  1. Loads the checkpoint with ``torch.load(..., map_location="cpu")``.
  2. Unwraps common save-wrapper patterns (e.g. ``{"state_dict": ...}``,
     ``{"netG": ...}``) so the underlying tensor dict is exposed.
  3. Strips a ``module.`` prefix if present (a tell-tale sign the checkpoint
     was saved from a ``torch.nn.DataParallel``-wrapped model).
  4. Prints every layer name with its tensor shape, dtype, and parameter
     count, plus summary statistics.

This is intentionally "dumb" and model-agnostic: it never imports a
generator class. That separation is what let the Phase 0 investigation
determine the required input/output channel counts (from the shapes of the
first and last conv layers) and the exact architecture depth (from how many
downsampling/upsampling blocks were present in the layer names) before ever
attempting to load the weights into a candidate model.

Usage
-----
    python checkpoint_inspector.py path/to/checkpoint.pth
    python checkpoint_inspector.py path/to/checkpoint.pth --grep down_conv
"""

import argparse
import sys
from collections import OrderedDict
from typing import Optional

import torch


def load_raw_state_dict(checkpoint_path: str) -> "OrderedDict[str, torch.Tensor]":
    """
    Load a checkpoint file and return the raw tensor dict, unwrapping the
    common wrapper patterns seen in Pix2Pix / CycleGAN-style checkpoints.

    This performs NO reshaping, renaming beyond prefix-stripping, or
    validation against any architecture -- it only normalizes the checkpoint
    file's outer container so the leaf tensors can be inspected.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Some checkpoints save {"state_dict": <dict>} or {"netG": <dict>} instead
    # of a bare state_dict. Unwrap one level if we find a known key.
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "netG", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a dict-like state_dict after unwrapping, got {type(checkpoint)}. "
            "This file may not be a plain generator checkpoint."
        )

    # torch.nn.DataParallel prefixes every key with "module." -- strip it so
    # layer names match a plain (non-parallel) model definition.
    state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, tensor in checkpoint.items():
        clean_key = key.replace("module.", "")
        state_dict[clean_key] = tensor

    return state_dict


def summarize(state_dict: "OrderedDict[str, torch.Tensor]", grep: Optional[str] = None) -> None:
    """Print every (layer name, shape, dtype) triple plus summary stats."""
    total_params = 0
    matched = 0

    print(f"{'Layer name':<70} {'Shape':<25} {'Dtype':<12} {'#Params'}")
    print("-" * 130)

    for name, tensor in state_dict.items():
        if grep is not None and grep not in name:
            continue
        matched += 1
        n_params = tensor.numel()
        total_params += n_params
        print(f"{name:<70} {str(tuple(tensor.shape)):<25} {str(tensor.dtype):<12} {n_params:,}")

    print("-" * 130)
    print(f"Entries shown: {matched} / {len(state_dict)}")
    if grep is None:
        print(f"Total parameters in checkpoint: {total_params:,}")

    # These two lines are what actually let us reverse-engineer the
    # generator's I/O channel counts in Phase 0: the very first conv layer's
    # input channels and the very last conv layer's output channels.
    first_key = next(iter(state_dict))
    last_key = next(reversed(state_dict))
    print()
    print("First tensor in checkpoint (often the outermost input conv weight):")
    print(f"  {first_key}: {tuple(state_dict[first_key].shape)}")
    print("Last tensor in checkpoint (often the outermost output conv weight):")
    print(f"  {last_key}: {tuple(state_dict[last_key].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the raw layer names / tensor shapes of a PyTorch checkpoint, "
        "with no dependency on any model class."
    )
    parser.add_argument("checkpoint", help="Path to the .pth checkpoint file")
    parser.add_argument(
        "--grep",
        default=None,
        help="Only show layer names containing this substring",
    )
    args = parser.parse_args()

    try:
        state_dict = load_raw_state_dict(args.checkpoint)
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    summarize(state_dict, grep=args.grep)


if __name__ == "__main__":
    main()
