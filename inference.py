"""
inference.py
============

Loads the Phase 0-verified Pix2Pix generator, runs a single forward pass on
a preprocessed input tensor, and returns the raw output tensor.

This module wires together the three previously-verified pieces:
  - the confirmed architecture (see architecture_verification.py):
      define_G(input_nc=4, output_nc=3, ngf=64, netG="unet_512",
               norm="instance", use_dropout=False)
  - the confirmed checkpoint-loading procedure (strict state_dict load,
    with `module.` prefix stripped and common wrapper keys unwrapped)
  - the confirmed preprocessing pipeline (see preprocessing.py):
      resize-by-height to 1536 -> left-crop/pad width to 512 ->
      4-channel tensor (RGB in [-1,1] + zero mask channel)

`define_G` itself is third-party code (the original Pix2Pix/CycleGAN
reference implementation's `models/networks.py`) and is not vendored in
this repo -- point `--pix2pix-root` at a checkout that provides it.

Usage
-----
    from inference import load_generator, run_inference
    from preprocessing import preprocess_image

    net = load_generator("245_net_G.pth", pix2pix_root=r"D:\\path\\to\\pix2pix_vgg")
    x = preprocess_image("input.jpg")
    y = run_inference(net, x)   # torch.Tensor, shape (1, 3, 1536, 512), values in [-1, 1]

Or standalone:
    python inference.py checkpoint.pth input.jpg --pix2pix-root D:\\path\\to\\pix2pix_vgg
"""

import argparse
import sys

import torch
import torch.nn as nn


def load_generator(checkpoint_path: str, pix2pix_root: str, device: str = "cpu") -> nn.Module:
    """
    Build the verified generator architecture, load the checkpoint into it
    with a strict state_dict match, and return it in eval mode on `device`.
    """
    sys.path.insert(0, pix2pix_root)
    try:
        from models.networks import define_G  # third-party, not included here
    except ImportError as exc:
        raise ImportError(
            f"Could not import define_G from '{pix2pix_root}/models/networks.py'. "
            "Point --pix2pix-root at a checkout of the original Pix2Pix/CycleGAN "
            "reference implementation that provides models/networks.py."
        ) from exc

    net = define_G(
        input_nc=4,
        output_nc=3,
        ngf=64,
        netG="unet_512",
        norm="instance",
        use_dropout=False,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("state_dict", "netG", "model", "module"):
            value = state_dict.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.to(device)
    return net


@torch.no_grad()
def run_inference(net: nn.Module, input_tensor: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Run a single forward pass through the generator.

    `input_tensor` must already be preprocessed (see preprocessing.py):
    shape (1, 4, 1536, 512), RGB channels in [-1, 1], 4th channel the mask.

    Returns the raw model output tensor, shape (1, 3, 1536, 512), values
    nominally in [-1, 1] (see postprocessing.py to convert to a viewable
    image).
    """
    input_tensor = input_tensor.to(device)
    output_tensor = net(input_tensor)
    return output_tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the verified Pix2Pix generator on a single input image."
    )
    parser.add_argument("checkpoint", help="Path to the .pth checkpoint file")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument(
        "--pix2pix-root",
        required=True,
        help="Path to a checkout of the original Pix2Pix/CycleGAN repo "
        "containing models/networks.py. Not included in this repo.",
    )
    parser.add_argument("--device", default="cpu", help="Device to run inference on (default: cpu)")
    parser.add_argument("--output", default="output.png", help="Where to save the result image")
    args = parser.parse_args()

    from preprocessing import preprocess_image
    from postprocessing import tensor_to_image

    print("Loading generator...")
    net = load_generator(args.checkpoint, args.pix2pix_root, device=args.device)

    print("Preprocessing input image...")
    x = preprocess_image(args.image)

    print("Running inference...")
    y = run_inference(net, x, device=args.device)
    print(f"Output tensor shape: {tuple(y.shape)}")

    image = tensor_to_image(y)
    image.save(args.output)
    print(f"Saved output image to: {args.output}")


if __name__ == "__main__":
    main()
