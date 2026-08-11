"""
preprocessing.py
================

Builds the exact 4-channel model input format verified during the Phase 0
investigation for this Pix2Pix generator (unet_512, input_nc=4, output_nc=3,
norm=instance).

Verified preprocessing pipeline
--------------------------------
The checkpoint was trained on images resized/aligned to exactly
512 (width) x 1536 (height). Given an arbitrary input image, the verified
pipeline is:

  1. Convert to RGB.
  2. Resize by HEIGHT to 1536px, preserving aspect ratio (width scales
     proportionally, using BICUBIC interpolation).
  3. Align width to 512px:
       - if the resized width is >= 512, take the LEFT-most 512px crop
         (crop((0, 0, 512, 1536))).
       - if the resized width is < 512, LEFT-pad (i.e. paste the image at
         the top-left corner of a black 512x1536 canvas, padding the
         remainder on the right) so the final size is exactly 512x1536.
     Either way the image is anchored at (0, 0) -- there is no centering.
  4. Convert to a tensor, scale from [0, 255] -> [0, 1] via
     ``torchvision.transforms.ToTensor()``, then remap to [-1, 1] via
     ``x * 2 - 1`` (equivalent to ``Normalize(mean=0.5, std=0.5)`` per
     channel), matching the reference Pix2Pix ``test.py`` data pipeline.
  5. Append a 4th channel: an all-zero mask channel of shape
     (1, 1536, 512), i.e. no mask supplied. (A real, non-zero mask channel
     is supported by the trained model, but this module documents the
     baseline "no mask" input format that was verified end-to-end.)

The result is a single tensor of shape (4, 1536, 512): RGB in [-1, 1]
followed by an all-zero mask channel.

Usage
-----
    from preprocessing import preprocess_image
    x = preprocess_image("input.jpg")   # torch.Tensor, shape (1, 4, 1536, 512)

Or standalone:
    python preprocessing.py input.jpg
"""

import argparse
import sys

import torch
import torchvision.transforms as T
from PIL import Image

# Verified target dimensions for this checkpoint.
TARGET_WIDTH = 512
TARGET_HEIGHT = 1536


def align_to_target_size(img: Image.Image, target_w: int = TARGET_WIDTH, target_h: int = TARGET_HEIGHT) -> Image.Image:
    """
    Resize `img` by height to `target_h` (preserving aspect ratio), then
    left-crop or left-pad the width to exactly `target_w`.

    This mirrors the exact alignment logic verified against the training
    data format: scale proportionally by height first, then anchor the
    result at the top-left corner of the target canvas.
    """
    img = img.convert("RGB")
    w, h = img.size

    if (w, h) != (target_w, target_h):
        new_w = round(w * target_h / h)
        img = img.resize((new_w, target_h), Image.BICUBIC)

        if new_w >= target_w:
            # Wide enough (or exactly wide enough): take the left `target_w` px.
            img = img.crop((0, 0, target_w, target_h))
        else:
            # Too narrow: pad on the right with black, anchored at (0, 0).
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            canvas.paste(img, (0, 0))
            img = canvas

    return img


def to_model_tensor(img_aligned: Image.Image) -> torch.Tensor:
    """
    Convert an already-aligned (512x1536) RGB PIL image into the verified
    4-channel model input tensor: RGB normalized to [-1, 1] concatenated
    with an all-zero mask channel.

    Returns a tensor of shape (4, 1536, 512).
    """
    assert img_aligned.size == (TARGET_WIDTH, TARGET_HEIGHT), (
        f"Expected aligned image of size {(TARGET_WIDTH, TARGET_HEIGHT)}, got {img_aligned.size}"
    )

    rgb = T.ToTensor()(img_aligned)      # (3, H, W), values in [0, 1]
    rgb = rgb * 2.0 - 1.0                # -> [-1, 1], equivalent to Normalize(0.5, 0.5)

    mask = torch.zeros(1, TARGET_HEIGHT, TARGET_WIDTH)  # no mask supplied

    return torch.cat([rgb, mask], dim=0)  # (4, H, W)


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    End-to-end: load an image from disk and produce a batched model input
    tensor of shape (1, 4, 1536, 512), ready to feed to the generator.
    """
    img = Image.open(image_path)
    aligned = align_to_target_size(img)
    tensor = to_model_tensor(aligned)
    return tensor.unsqueeze(0)  # add batch dimension -> (1, 4, 1536, 512)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess an image into the verified 4-channel Pix2Pix "
        "generator input tensor (512x1536, RGB in [-1,1] + zero mask channel)."
    )
    parser.add_argument("image", help="Path to an input image")
    args = parser.parse_args()

    try:
        tensor = preprocess_image(args.image)
    except FileNotFoundError:
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Preprocessed tensor shape: {tuple(tensor.shape)}")
    print(f"Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"Channel 3 (mask) is all zero: {bool(torch.all(tensor[0, 3] == 0))}")


if __name__ == "__main__":
    main()
