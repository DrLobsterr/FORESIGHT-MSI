"""Checkpoint loading utilities."""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .model import FORESIGHT


DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "img_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "hidden_dim": 128,
    "num_blocks": 4,
    "modes": 8,
    "dropout": 0.1,
}


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[FORESIGHT, Dict[str, Any], Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update(checkpoint_config)
    model = FORESIGHT(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        modes=config["modes"],
        dropout=config["dropout"],
    )
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    metadata = payload if isinstance(payload, dict) else {}
    return model, config, metadata


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Complex FFT support varies across PyTorch/MPS versions; CPU is reproducible.
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)
