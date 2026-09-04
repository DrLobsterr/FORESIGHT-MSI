"""Mask generation used by FORESIGHT.

Public convention: 1 = reconstruct, 0 = preserve.
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np


def detect_foreground(normalized_hwc: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Reproduce the intensity-range foreground rule used during training."""
    gray = normalized_hwc.mean(axis=2)
    threshold = float(gray.min() + fraction * (gray.max() - gray.min()))
    foreground = gray > threshold
    if not np.any(foreground):
        raise ValueError("No foreground was detected.")
    return foreground


def exact_foreground_block_mask(
    foreground: np.ndarray,
    ratio: float,
    seed: int,
    block_sizes: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Mask an exact fraction of detected foreground with random small blocks.

    This deterministic implementation was used for the five-image independent
    DON validation in Supplementary Figure S10.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")
    rng = random.Random(seed)
    height, width = foreground.shape
    points = np.argwhere(foreground)
    target = max(1, int(round(len(points) * ratio)))
    mask = np.zeros((height, width), dtype=bool)
    attempts = 0
    max_attempts = target * 30 + 1000

    while int(np.count_nonzero(mask & foreground)) < target and attempts < max_attempts:
        attempts += 1
        center_y, center_x = points[rng.randrange(len(points))]
        block_size = rng.choice(block_sizes)
        top = max(0, min(height - block_size, int(center_y) - rng.randrange(block_size)))
        left = max(0, min(width - block_size, int(center_x) - rng.randrange(block_size)))
        local = foreground[top : top + block_size, left : left + block_size]
        available = np.argwhere(local & ~mask[top : top + block_size, left : left + block_size])
        if not len(available):
            continue
        remaining = target - int(np.count_nonzero(mask & foreground))
        candidates = available.tolist()
        rng.shuffle(candidates)
        for local_y, local_x in candidates[:remaining]:
            mask[top + local_y, left + local_x] = True

    remaining = target - int(np.count_nonzero(mask & foreground))
    if remaining:
        available = np.argwhere(foreground & ~mask).tolist()
        for y, x in rng.sample(available, remaining):
            mask[y, x] = True

    repair = mask.astype(np.float32)
    return repair, mask_statistics(repair, foreground, ratio)


def legacy_v4_block_mask(
    foreground: np.ndarray,
    ratio: float,
    rng: Optional[random.Random] = None,
    block_sizes: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Reproduce the online random-block algorithm used for model training.

    Historical behavior is intentionally retained: candidate blocks may overlap,
    and a block is eligible when at least 30% of its pixels are foreground.
    Consequently, the realized foreground ratio can differ from the nominal ratio.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")
    rng = rng or random
    height, width = foreground.shape
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count == 0:
        foreground = np.ones_like(foreground, dtype=bool)
        foreground_count = height * width

    target_area = max(1, int(foreground_count * ratio))
    possible = []
    for size in block_sizes:
        for row in range(0, height - size + 1):
            for col in range(0, width - size + 1):
                region = foreground[row : row + size, col : col + size]
                if int(np.count_nonzero(region)) >= size * size * 0.30:
                    possible.append((row, col, size))

    if len(possible) < target_area / 9.0:
        possible = []
        for size in block_sizes:
            for row in range(0, height - size + 1):
                for col in range(0, width - size + 1):
                    if np.any(foreground[row : row + size, col : col + size]):
                        possible.append((row, col, size))

    rng.shuffle(possible)
    selected = []
    cumulative_area = 0
    for block in possible:
        if cumulative_area >= target_area:
            break
        selected.append(block)
        cumulative_area += block[2] * block[2]

    repair = np.zeros((height, width), dtype=np.float32)
    for row, col, size in selected:
        repair[row : row + size, col : col + size] = 1.0
    stats = mask_statistics(repair, foreground, ratio)
    stats["number_of_selected_blocks"] = len(selected)
    stats["cumulative_selected_block_area"] = cumulative_area
    return repair, stats


def mask_statistics(
    repair_mask: np.ndarray, foreground: np.ndarray, target_ratio: float
) -> Dict[str, float]:
    foreground_count = int(np.count_nonzero(foreground))
    masked_foreground = int(np.count_nonzero((repair_mask > 0.5) & foreground))
    return {
        "target_ratio": float(target_ratio),
        "foreground_pixels": foreground_count,
        "masked_foreground_pixels": masked_foreground,
        "realized_foreground_ratio": (
            masked_foreground / foreground_count if foreground_count else 0.0
        ),
        "realized_whole_image_ratio": float(np.mean(repair_mask > 0.5)),
    }
