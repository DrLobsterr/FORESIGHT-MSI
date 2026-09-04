"""Shared inference functions."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .model import FORESIGHT
from .image_io import denormalize_chw, normalize_hwc


def reconstruct_with_model(
    original_rgb: np.ndarray,
    repair_mask: np.ndarray,
    model: FORESIGHT,
    image_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct an RGB [0,1] image using a binary 1=reconstruct mask."""
    if original_rgb.shape != (image_size, image_size, 3):
        raise ValueError(
            "Expected image shape ({0}, {0}, 3), got {1}".format(
                image_size, original_rgb.shape
            )
        )
    if repair_mask.shape != (image_size, image_size):
        raise ValueError(
            "Expected mask shape ({0}, {0}), got {1}".format(
                image_size, repair_mask.shape
            )
        )
    normalized = normalize_hwc(original_rgb)
    image_tensor = torch.from_numpy(
        np.transpose(normalized, (2, 0, 1)).copy()
    ).float().unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(repair_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    masked_tensor = image_tensor * (1.0 - mask_tensor)
    with torch.inference_mode():
        prediction = model(masked_tensor, mask_tensor)
    masked_rgb = denormalize_chw(masked_tensor[0].cpu().numpy())
    reconstructed_rgb = denormalize_chw(prediction[0].cpu().numpy())
    return masked_rgb, reconstructed_rgb


def reconstruct_array(
    original_rgb: np.ndarray,
    repair_mask: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model, model_config, metadata = load_checkpoint(checkpoint_path, device)
    masked, reconstructed = reconstruct_with_model(
        original_rgb, repair_mask, model, int(model_config["img_size"]), device
    )
    return masked, reconstructed, metadata
