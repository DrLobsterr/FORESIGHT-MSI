#!/usr/bin/env python3
"""Generate the twelve reported training configurations."""

import json
from pathlib import Path


MODEL_COUNTS = {"14d1": 1000, "17d1": 1200, "17d2": 900}
MASK_RATIOS = (0.4, 0.5, 0.6, 0.7)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "configs"
    for model_name, image_count in MODEL_COUNTS.items():
        for ratio in MASK_RATIOS:
            config = {
                "experiment": {
                    "model_name": model_name,
                    "manuscript_group": "14 dpi" if model_name == "14d1" else "17 dpi",
                    "mask_ratio": ratio,
                    "mask_ratio_basis": "detected tissue foreground",
                    "mask_algorithm": "legacy_v4_random_blocks",
                    "seed": 42,
                    "legacy_file_order": True,
                    "expected_image_count": image_count,
                },
                "model": {
                    "img_size": 256,
                    "in_channels": 3,
                    "out_channels": 3,
                    "hidden_dim": 128,
                    "num_blocks": 4,
                    "modes": 8,
                    "dropout": 0.1,
                },
                "training": {
                    "batch_size": 16,
                    "num_epochs": 100,
                    "learning_rate": 0.0001,
                    "weight_decay": 0.0001,
                    "min_learning_rate": 0.000001,
                    "warmup_epochs": 5,
                    "betas": [0.9, 0.999],
                    "early_stopping_patience": 7,
                    "perceptual_weight": 0.5,
                    "legacy_detached_perceptual": True,
                    "use_amp": False,
                    "num_workers": 2,
                    "training_augmentation": "horizontal flip with probability 0.3",
                },
            }
            path = output_dir / "{}_mask{:02d}.json".format(model_name, int(ratio * 100))
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(path.name)


if __name__ == "__main__":
    main()
