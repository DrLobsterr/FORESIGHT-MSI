#!/usr/bin/env python3
"""Center an ion image on a square black canvas."""

import argparse
from pathlib import Path

from PIL import Image


def add_black_margin(input_path: Path, output_path: Path, margin_ratio: float) -> None:
    with Image.open(input_path) as opened:
        image = opened.convert("RGBA")
        width, height = image.size
        margin = int(max(width, height) * margin_ratio)
        side = max(width + 2 * margin, height + 2 * margin)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 255))
        position = ((side - width) // 2, (side - height) // 2)
        canvas.paste(image, position, image)
        canvas.convert("RGB").save(output_path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--margin-ratio", type=float, default=0.1)
    args = parser.parse_args()
    add_black_margin(args.input, args.output, args.margin_ratio)
    print(args.output)


if __name__ == "__main__":
    main()
