"""Image-reconstruction metrics with explicit evaluation regions."""

import math
from typing import Dict, Optional

import numpy as np
from skimage.metrics import structural_similarity


def mse(a: np.ndarray, b: np.ndarray, region: Optional[np.ndarray] = None) -> float:
    error = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if region is None:
        return float(np.mean(error))
    selected = error[region.astype(bool)]
    if selected.size == 0:
        raise ValueError("The metric region contains no pixels.")
    return float(np.mean(selected))


def psnr(mse_value: float, data_range: float = 1.0) -> float:
    if mse_value == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range * data_range) / mse_value))


def standard_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        structural_similarity(a, b, channel_axis=2, data_range=1.0)
    )


def region_ssim_map_mean(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """Mean of the standard local SSIM map over a declared spatial region."""
    _, ssim_map = structural_similarity(
        a, b, channel_axis=2, data_range=1.0, full=True
    )
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)
    selected = ssim_map[region.astype(bool)]
    if selected.size == 0:
        raise ValueError("The metric region contains no pixels.")
    return float(np.mean(selected))


def calculate_metrics(
    original: np.ndarray,
    candidate: np.ndarray,
    repair_mask: np.ndarray,
    foreground: np.ndarray,
) -> Dict[str, float]:
    full_mse = mse(original, candidate)
    repair_mse = mse(original, candidate, repair_mask)
    foreground_mse = mse(original, candidate, foreground)
    return {
        "mse_full": full_mse,
        "psnr_full_db": psnr(full_mse),
        "ssim_full": standard_ssim(original, candidate),
        "mse_repair_region": repair_mse,
        "psnr_repair_region_db": psnr(repair_mse),
        "ssim_map_mean_repair_region": region_ssim_map_mean(
            original, candidate, repair_mask
        ),
        "mse_tissue_foreground": foreground_mse,
        "psnr_tissue_foreground_db": psnr(foreground_mse),
        "ssim_map_mean_tissue_foreground": region_ssim_map_mean(
            original, candidate, foreground
        ),
    }
