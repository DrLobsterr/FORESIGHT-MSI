#!/usr/bin/env python3
"""Generate a non-study ion-like image and mask for a smoke test."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("examples/synthetic"))
    args = parser.parse_args()
    size = 256
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    tissue = ((x / 0.72) ** 2 + (y / 0.86) ** 2) <= 1.0
    signal = np.exp(-5.0 * ((x + 0.18) ** 2 + (y - 0.08) ** 2))
    signal += 0.55 * np.exp(-12.0 * ((x - 0.27) ** 2 + (y + 0.25) ** 2))
    signal = np.clip(signal * tissue, 0.0, 1.0)
    rgb = np.stack(
        [signal, np.sqrt(signal) * 0.72, np.clip(1.2 - signal, 0, 1) * tissue * 0.25],
        axis=2,
    )
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[92:120, 78:116] = 255
    mask[136:162, 141:178] = 255
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB").save(
        args.output_dir / "synthetic_ion.png"
    )
    Image.fromarray(mask, mode="L").save(args.output_dir / "synthetic_mask.png")
    labelme = {
        "version": "5.x",
        "flags": {},
        "shapes": [
            {
                "label": "damage",
                "points": [[78, 92], [115, 92], [115, 119], [78, 119]],
                "group_id": None,
                "description": "synthetic test region",
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "damage",
                "points": [[141, 136], [177, 136], [177, 161], [141, 161]],
                "group_id": None,
                "description": "synthetic test region",
                "shape_type": "polygon",
                "flags": {},
            },
        ],
        "imagePath": "synthetic_ion.png",
        "imageData": None,
        "imageHeight": size,
        "imageWidth": size,
    }
    (args.output_dir / "synthetic_labelme.json").write_text(
        json.dumps(labelme, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
