"""Training dataset and deterministic split utilities."""

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .image_io import IMAGENET_MEAN, IMAGENET_STD
from .masks import detect_foreground, legacy_v4_block_mask

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_images(data_dir: Path, legacy_order: bool = False) -> List[Path]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError("Data directory not found: {}".format(data_dir))
    names = os.listdir(str(data_dir))
    if not legacy_order:
        names.sort()
    paths = [data_dir / name for name in names if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError("No supported images found in {}".format(data_dir))
    return paths


def split_paths(
    paths: Sequence[Path], seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Match the manuscript's 70/15/15 split using scikit-learn."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise ImportError("Training split reproduction requires scikit-learn.") from error
    train, temporary = train_test_split(list(paths), train_size=0.70, random_state=seed)
    validation, test = train_test_split(temporary, train_size=0.50, random_state=seed)
    return list(train), list(validation), list(test)


def write_split_manifest(
    output_path: Path,
    train: Sequence[Path],
    validation: Sequence[Path],
    test: Sequence[Path],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["filename", "split"])
        for split, items in (("train", train), ("validation", validation), ("test", test)):
            for path in items:
                writer.writerow([Path(path).name, split])


def read_split_manifest(data_dir: Path, manifest: Path) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {"train": [], "validation": [], "test": []}
    with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in result:
                raise ValueError("Unknown split in manifest: {}".format(row["split"]))
            path = Path(data_dir) / row["filename"]
            if not path.is_file():
                raise FileNotFoundError("Manifest image not found: {}".format(path))
            result[row["split"]].append(path)
    return result


class IonImageDataset(Dataset):
    """PNG/JPEG ion images with deterministic online legacy masks."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int = 256,
        mask_ratio: float = 0.40,
        training: bool = False,
        seed: int = 42,
    ):
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.training = training
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.image_paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            rng = random.Random(self.seed + self.epoch * len(self) + index)
            if self.training and rng.random() < 0.30:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            rgb = np.asarray(image, dtype=np.float32) / 255.0

        normalized = (rgb - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
        foreground = detect_foreground(normalized)
        repair_mask, stats = legacy_v4_block_mask(foreground, self.mask_ratio, rng)
        image_tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1)).copy()).float()
        mask_tensor = torch.from_numpy(repair_mask).unsqueeze(0).float()
        return {
            "masked_image": image_tensor * (1.0 - mask_tensor),
            "original_image": image_tensor,
            "repair_mask": mask_tensor,
            "foreground": torch.from_numpy(foreground.astype(np.float32)).unsqueeze(0),
            "path": str(path),
            "realized_foreground_ratio": torch.tensor(
                stats["realized_foreground_ratio"], dtype=torch.float32
            ),
        }


def make_loaders(
    data_dir: Path,
    batch_size: int,
    mask_ratio: float,
    image_size: int = 256,
    seed: int = 42,
    workers: int = 2,
    manifest: Optional[Path] = None,
    legacy_file_order: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if manifest:
        split = read_split_manifest(data_dir, manifest)
        train_paths = split["train"]
        val_paths = split["validation"]
        test_paths = split["test"]
    else:
        paths = discover_images(data_dir, legacy_order=legacy_file_order)
        train_paths, val_paths, test_paths = split_paths(paths, seed)

    datasets = (
        IonImageDataset(train_paths, image_size, mask_ratio, True, seed),
        IonImageDataset(val_paths, image_size, mask_ratio, False, seed + 1000000),
        IonImageDataset(test_paths, image_size, mask_ratio, False, seed + 2000000),
    )
    loaders = []
    for index, dataset in enumerate(datasets):
        generator = torch.Generator().manual_seed(seed + index)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(index == 0),
                num_workers=workers,
                pin_memory=False,
                drop_last=(index == 0),
                generator=generator,
                persistent_workers=False,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]
