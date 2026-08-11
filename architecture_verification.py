"""
architecture_verification.py
=============================

Phase 0 diagnostic tool: prove that a given checkpoint matches a specific
candidate generator architecture, by loading it with ``strict=True`` and
checking for zero missing/unexpected keys.

Background
----------
``checkpoint_inspector.py`` let us read the raw layer names and shapes out
of the checkpoint without any model class. From that inspection (first conv
input channels = 4, last conv output channels = 3, and the depth/pattern of
the down-sampling and up-sampling blocks matching a U-Net with 512-sized
innermost feature maps) we formed a candidate architecture:

    define_G(
        input_nc=4,        # RGB (3) + one mask channel (1)
        output_nc=3,       # RGB
        ngf=64,            # default generator filter count
        netG="unet_512",   # U-Net variant sized for 512px inputs
        norm="instance",   # instance normalization (no learned running stats
                            # in the checkpoint, which ruled out batch norm)
        use_dropout=False, # matches training with --no_dropout
    )

``define_G`` / ``UnetGenerator`` come from the original Pix2Pix / CycleGAN
reference implementation (``pix2pix_vgg/models/networks.py`` in this
project) and are NOT reproduced here -- only referenced. Point
``PIX2PIX_ROOT`` below at a checked-out copy of that repository (or your own
vendored copy of ``models/networks.py``) to run this script.

The actual verification step is deliberately strict: ``load_state_dict``
is called with ``strict=True`` so PyTorch raises/reports on ANY mismatch --
missing keys, unexpected keys, or shape mismatches. A clean, silent load is
the proof that this is the correct architecture for this checkpoint (as
opposed to a "close enough" architecture that only loads under
``strict=False`` while silently dropping or ignoring layers).

Usage
-----
    python architecture_verification.py path/to/checkpoint.pth --pix2pix-root D:\\path\\to\\pix2pix_vgg
"""

import argparse
import sys


def load_define_G(pix2pix_root: str):
    """
    Import ``define_G`` from the original (third-party) Pix2Pix repository.

    This project does not vendor or redistribute that file -- it is expected
    to already exist on disk (e.g. as a git submodule, a sibling checkout,
    or copied in separately with its original license/attribution intact).
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
    return define_G


def build_verified_generator(define_G):
    """
    Construct the generator with the exact parameters established in
    Phase 0 as matching this checkpoint's structure.
    """
    return define_G(
        input_nc=4,          # RGB (3 channels) + 1 mask channel
        output_nc=3,         # RGB output
        ngf=64,               # default number of generator filters
        netG="unet_512",     # U-Net generator variant for 512px-scale inputs
        norm="instance",     # instance normalization
        use_dropout=False,   # trained with --no_dropout
    )


def load_raw_state_dict(checkpoint_path: str):
    """Load a checkpoint and unwrap common wrapper keys / DataParallel prefix."""
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("state_dict", "netG", "model", "module"):
            value = state_dict.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict


def verify(checkpoint_path: str, pix2pix_root: str) -> bool:
    """
    Attempt a strict load of `checkpoint_path` into the verified architecture.
    Returns True on a clean strict match, False otherwise (details printed).
    """
    define_G = load_define_G(pix2pix_root)
    net = build_verified_generator(define_G)
    state_dict = load_raw_state_dict(checkpoint_path)

    try:
        # strict=True: raises RuntimeError on ANY missing/unexpected/shape
        # mismatch. This is the actual pass/fail verification step.
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print("FAILED strict load. Architecture does NOT match this checkpoint.")
        print()
        print(str(exc))
        return False

    print("SUCCESS: checkpoint loaded with strict=True, zero missing/unexpected keys.")
    print("Confirmed architecture:")
    print("  define_G(input_nc=4, output_nc=3, ngf=64, netG='unet_512', "
          "norm='instance', use_dropout=False)")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {n_params:,}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a checkpoint against the confirmed Pix2Pix UnetGenerator "
        "(unet_512, input_nc=4, output_nc=3, norm=instance) via strict load_state_dict."
    )
    parser.add_argument("checkpoint", help="Path to the .pth checkpoint file")
    parser.add_argument(
        "--pix2pix-root",
        required=True,
        help="Path to a checkout of the original Pix2Pix/CycleGAN repo "
        "containing models/networks.py (define_G, UnetGenerator). Not included in this repo.",
    )
    args = parser.parse_args()

    ok = verify(args.checkpoint, args.pix2pix_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
