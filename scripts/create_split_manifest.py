#!/usr/bin/env python3
"""Record the 70/15/15 filename split for later reproducibility."""

import argparse
import csv
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy-file-order",
        action="store_true",
        help="Use filesystem order, matching the historical training loader.",
    )
    parser.add_argument(
        "--replace-label",
        help="Rename historical filename text for release, formatted OLD:NEW.",
    )
    args = parser.parse_args()
    names = os.listdir(str(args.data_dir))
    if not args.legacy_file_order:
        names.sort()
    paths = [
        args.data_dir / name
        for name in names
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]
    from sklearn.model_selection import train_test_split

    train, temporary = train_test_split(paths, train_size=0.70, random_state=args.seed)
    validation, test = train_test_split(
        temporary, train_size=0.50, random_state=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["filename", "split"])
        old_label, new_label = (None, None)
        if args.replace_label:
            old_label, new_label = args.replace_label.split(":", 1)
        for split, items in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            for path in items:
                name = path.name.replace(old_label, new_label) if old_label else path.name
                writer.writerow([name, split])
    print("train={}, validation={}, test={}".format(len(train), len(validation), len(test)))
    print(args.output)


if __name__ == "__main__":
    main()
