#!/usr/bin/env python3
"""Convert LabelMe polygon annotations to a binary repair mask."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def convert(json_path: Path, output_path: Path) -> None:
    annotation = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation.get("shapes", []):
        points = [tuple(point) for point in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    mask.save(output_path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.annotation, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
