#!/usr/bin/env python3
"""Reconstruct one damaged ion image using an external binary mask."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foresight.checkpoint import choose_device
from foresight.image_io import load_repair_mask, load_rgb_square, save_mask, save_rgb
from foresight.inference import reconstruct_array


def load_image(path: Path, image_size: int) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        return load_rgb_square(path, image_size)[0]
    array = np.load(path)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Unsupported NumPy image shape: {}".format(array.shape))
    pixels = np.clip(array, 0.0, 1.0)
    pil = Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB")
    pil = pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def load_mask(path: Path, image_size: int) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        return load_repair_mask(path, image_size)
    array = np.squeeze(np.load(path))
    pil = Image.fromarray((array > 0.5).astype(np.uint8) * 255, mode="L")
    pil = pil.resize((image_size, image_size), Image.Resampling.NEAREST)
    return (np.asarray(pil) > 127).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    original = load_image(args.image, args.image_size)
    repair_mask = load_mask(args.mask, args.image_size)
    device = choose_device(args.device)
    masked, reconstructed, metadata = reconstruct_array(
        original, repair_mask, args.model, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output_dir / "original.png", original)
    save_mask(args.output_dir / "repair_mask.png", repair_mask)
    save_rgb(args.output_dir / "masked_input.png", masked)
    save_rgb(args.output_dir / "reconstructed.png", reconstructed)
    np.save(args.output_dir / "reconstructed.npy", reconstructed)
    summary = {
        "device": str(device),
        "mask_convention": "1 = reconstruct, 0 = preserve",
        "checkpoint_epoch": metadata.get("epoch"),
    }
    (args.output_dir / "reconstruction.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
