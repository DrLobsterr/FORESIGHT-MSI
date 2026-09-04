"""Image and mask input/output helpers."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb_square(path: Path, image_size: int = 256) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load RGB image, pad the short side with black, and resize to a square."""
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        source_size = rgb.size
        if rgb.width != rgb.height:
            side = max(rgb.width, rgb.height)
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
            rgb = canvas
        rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array, source_size


def load_repair_mask(path: Path, image_size: int = 256) -> np.ndarray:
    """Load a binary mask using 1 = reconstruct and 0 = preserve."""
    with Image.open(path) as opened:
        mask = opened.convert("L").resize(
            (image_size, image_size), Image.Resampling.NEAREST
        )
        return (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)


def normalize_hwc(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)


def denormalize_chw(image: np.ndarray) -> np.ndarray:
    hwc = np.transpose(image, (1, 2, 0))
    hwc = hwc * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
    return np.clip(hwc, 0.0, 1.0)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L").save(path)
