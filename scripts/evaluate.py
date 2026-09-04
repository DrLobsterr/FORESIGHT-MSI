#!/usr/bin/env python3
"""Evaluate intact ion images after deterministic random masking."""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foresight.checkpoint import choose_device, load_checkpoint
from foresight.data import discover_images
from foresight.image_io import load_rgb_square, normalize_hwc, save_mask, save_rgb
from foresight.inference import reconstruct_with_model
from foresight.masks import (
    detect_foreground,
    exact_foreground_block_mask,
    legacy_v4_block_mask,
)
from foresight.metrics import calculate_metrics


def fingerprint(image: np.ndarray) -> str:
    pixels = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def reference_fingerprints(path: Path, image_size: int) -> Dict[str, Path]:
    if path is None:
        return {}
    result = {}
    for image_path in discover_images(path):
        image, _ = load_rgb_square(image_path, image_size)
        result.setdefault(fingerprint(image), image_path)
    return result


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--mask-algorithm", choices=["exact", "legacy-v4"], default="exact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--reference-data-dir", type=Path)
    args = parser.parse_args()

    device = choose_device(args.device)
    model, model_config, _ = load_checkpoint(args.model, device)
    checkpoint_size = int(model_config["img_size"])
    if checkpoint_size != args.image_size:
        raise ValueError(
            "--image-size {} does not match checkpoint size {}".format(
                args.image_size, checkpoint_size
            )
        )
    images = discover_images(args.input_dir)
    references = reference_fingerprints(args.reference_data_dir, args.image_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, path in enumerate(images):
        original, source_size = load_rgb_square(path, args.image_size)
        if fingerprint(original) in references:
            raise ValueError(
                "Data leakage: {} matches {}".format(
                    path, references[fingerprint(original)]
                )
            )
        foreground = detect_foreground(normalize_hwc(original))
        if args.mask_algorithm == "exact":
            repair_mask, mask_info = exact_foreground_block_mask(
                foreground, args.mask_ratio, args.seed + index
            )
        else:
            repair_mask, mask_info = legacy_v4_block_mask(
                foreground, args.mask_ratio, random.Random(args.seed + index)
            )
        masked, reconstructed = reconstruct_with_model(
            original, repair_mask, model, checkpoint_size, device
        )
        before = calculate_metrics(original, masked, repair_mask, foreground)
        after = calculate_metrics(original, reconstructed, repair_mask, foreground)
        sample_dir = args.output_dir / "{:02d}_{}".format(index + 1, path.stem)
        save_rgb(sample_dir / "original.png", original)
        save_mask(sample_dir / "repair_mask.png", repair_mask)
        save_rgb(sample_dir / "masked_input.png", masked)
        save_rgb(sample_dir / "reconstructed.png", reconstructed)
        row = {
            "image": path.name,
            "source_size": "{}x{}".format(*source_size),
            "mask_algorithm": args.mask_algorithm,
            "target_mask_ratio": args.mask_ratio,
            "realized_mask_ratio_foreground": mask_info["realized_foreground_ratio"],
            "realized_mask_ratio_whole_image": mask_info["realized_whole_image_ratio"],
        }
        row.update({"before_{}".format(k): v for k, v in before.items()})
        row.update({"after_{}".format(k): v for k, v in after.items()})
        rows.append(row)
        print(
            "[{}/{}] {} PSNR {:.2f} -> {:.2f} dB; SSIM {:.4f} -> {:.4f}".format(
                index + 1,
                len(images),
                path.name,
                before["psnr_full_db"],
                after["psnr_full_db"],
                before["ssim_full"],
                after["ssim_full"],
            )
        )

    write_csv(args.output_dir / "metrics_per_image.csv", rows)
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {
        "number_of_images": len(rows),
        "device": str(device),
        "mask_convention": "1 = reconstruct, 0 = preserve",
        "metric_scope": {
            "full": "all 256 x 256 RGB pixels, including black canvas",
            "repair_region": "only pixels selected for reconstruction",
            "tissue_foreground": "only pixels detected by the published intensity-range rule",
        },
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "sample_standard_deviation": {
            key: float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0
            for key in numeric
        },
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(args.output_dir / "metrics_per_image.csv")


if __name__ == "__main__":
    main()
