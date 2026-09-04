#!/usr/bin/env python3
"""Train a FORESIGHT model from a JSON experiment configuration."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foresight.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    best = train_from_config(
        args.config, args.data_dir, args.output_dir, args.manifest
    )
    print("Best checkpoint: {}".format(best))


if __name__ == "__main__":
    main()
