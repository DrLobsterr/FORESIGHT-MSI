"""Training loop for FORESIGHT."""

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm

from .data import make_loaders
from .losses import FORESIGHTLoss
from .model import FORESIGHT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Dict[str, Any]) -> FORESIGHT:
    return FORESIGHT(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        modes=config["modes"],
        dropout=config["dropout"],
    )


def run_epoch(
    model: FORESIGHT,
    loader: torch.utils.data.DataLoader,
    criterion: FORESIGHTLoss,
    device: torch.device,
    optimizer: Any = None,
    scaler: Any = None,
    use_amp: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc="train" if training else "evaluate")
        for batch in progress:
            masked = batch["masked_image"].to(device)
            target = batch["original_image"].to(device)
            repair_mask = batch["repair_mask"].to(device)
            if training:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model(masked, repair_mask)
                loss, parts = criterion(prediction, target, repair_mask)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            batch_size = masked.size(0)
            total += float(loss.detach().item()) * batch_size
            samples += batch_size
            progress.set_postfix(parts)
    return total / samples


def train_from_config(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    manifest: Path = None,
) -> Path:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_config = config["model"]
    training_config = config["training"]
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)
    loaders = make_loaders(
        data_dir=data_dir,
        batch_size=int(training_config["batch_size"]),
        mask_ratio=float(experiment["mask_ratio"]),
        image_size=int(model_config["img_size"]),
        seed=seed,
        workers=int(training_config["num_workers"]),
        manifest=manifest,
        legacy_file_order=bool(experiment.get("legacy_file_order", True)),
    )
    train_loader, validation_loader, test_loader = loaders

    criterion = FORESIGHTLoss(
        perceptual_weight=float(training_config["perceptual_weight"]),
        legacy_detached_perceptual=bool(
            training_config["legacy_detached_perceptual"]
        ),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        betas=tuple(training_config["betas"]),
    )
    epochs = int(training_config["num_epochs"])
    warmup_epochs = int(training_config["warmup_epochs"])
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=float(training_config["min_learning_rate"]),
    )
    use_amp = bool(training_config["use_amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    best_loss = float("inf")
    patience = 0
    history = {"train": [], "validation": []}
    best_path = output_dir / "best_model.pth"

    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch)
        validation_loader.dataset.set_epoch(epoch)
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp
        )
        validation_loss = run_epoch(
            model, validation_loader, criterion, device, use_amp=use_amp
        )
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        (warmup if epoch < warmup_epochs else cosine).step()

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": cosine.state_dict(),
            "warmup_scheduler_state_dict": warmup.state_dict(),
            "val_loss": validation_loss,
            "train_losses": history["train"],
            "val_losses": history["validation"],
            "config": training_config,
            "model_config": model_config,
        }
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience = 0
            torch.save(payload, best_path)
        else:
            patience += 1
            if patience >= int(training_config["early_stopping_patience"]):
                break

    test_loader.dataset.set_epoch(epoch)
    test_loss = run_epoch(model, test_loader, criterion, device, use_amp=use_amp)
    summary = {
        "best_validation_loss": best_loss,
        "final_test_loss": test_loss,
        "epochs_completed": epoch + 1,
        "device": str(device),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_path
