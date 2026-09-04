#!/usr/bin/env python3
"""Convert an RGB image and white-damage mask to NumPy tensors."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert(image_path: Path, mask_path: Path, output_dir: Path) -> None:
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(mask_path) as opened:
        mask_image = opened.convert("L")
        if mask_image.size != (image.shape[1], image.shape[0]):
            mask_image = mask_image.resize(
                (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
            )
        repair_mask = (np.asarray(mask_image) > 127).astype(np.float32)

    image_nchw = np.transpose(image, (2, 0, 1))[None, ...]
    mask_nchw = repair_mask[None, None, ...]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "original_image_numpy.npy", image_nchw)
    np.save(output_dir / "mask_numpy.npy", mask_nchw)
    print("image:", image_nchw.shape, "mask:", mask_nchw.shape)
    print("Mask convention: 1 = reconstruct, 0 = preserve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.image, args.mask, args.output_dir)


if __name__ == "__main__":
    main()
