"""
postprocessing.py
==================

Converts the generator's raw output tensor back into a viewable image.

Verified postprocessing pipeline
---------------------------------
The generator's output (and its training targets) live in [-1, 1], the
mirror image of the [-1, 1] normalization applied to the RGB input channels
during preprocessing (see preprocessing.py). To get back to a displayable
image:

  1. Drop the batch dimension if present (take the first, and only, sample).
  2. Clamp to [-1, 1] (the raw network output can slightly overshoot this
     range; clamping avoids wraparound artifacts on the next step).
  3. Rescale [-1, 1] -> [0, 1] via ``(x + 1) / 2``.
  4. Rescale [0, 1] -> [0, 255] and cast to ``uint8``.
  5. Rearrange from (C, H, W) to (H, W, C) and wrap as a PIL Image.

Usage
-----
    from postprocessing import tensor_to_image
    image = tensor_to_image(output_tensor)   # PIL.Image.Image
    image.save("result.png")

Or standalone (loads a saved .pt tensor file and converts it):
    python postprocessing.py output_tensor.pt result.png
"""

import argparse

import torch
from PIL import Image


def tensor_to_image(output_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a generator output tensor into a viewable RGB PIL Image.

    Accepts shape (3, H, W) or (1, 3, H, W); values are expected to be
    nominally in [-1, 1] (the network's un-normalized output range).
    """
    tensor = output_tensor
    if tensor.dim() == 4:
        tensor = tensor[0]  # drop batch dimension

    tensor = tensor.detach().cpu()
    tensor = tensor.clamp(-1.0, 1.0)          # guard against slight overshoot
    tensor = (tensor + 1.0) / 2.0             # [-1, 1] -> [0, 1]
    tensor = (tensor * 255.0).byte()          # [0, 1]  -> [0, 255], uint8

    array = tensor.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a saved generator output tensor (.pt) into a viewable image."
    )
    parser.add_argument("tensor_path", help="Path to a .pt file containing the output tensor")
    parser.add_argument("output_path", help="Where to save the resulting image (e.g. result.png)")
    args = parser.parse_args()

    output_tensor = torch.load(args.tensor_path, map_location="cpu")
    image = tensor_to_image(output_tensor)
    image.save(args.output_path)
    print(f"Saved image to: {args.output_path}")


if __name__ == "__main__":
    main()
