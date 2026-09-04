"""Loss used by the historical FORESIGHT training runs."""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FORESIGHTLoss(nn.Module):
    """Masked L1 plus the reported VGG feature value.

    The published checkpoints were trained with ``legacy_detached_perceptual=True``.
    In that mode both VGG feature tensors are detached, exactly matching the original
    code. The perceptual value affects logged loss/early stopping but contributes no
    gradient. This behavior is retained for provenance, not presented as a correction.
    """

    def __init__(
        self,
        perceptual_weight: float = 0.5,
        legacy_detached_perceptual: bool = True,
    ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.legacy_detached_perceptual = legacy_detached_perceptual
        self.l1 = nn.L1Loss()
        self.vgg = None
        if perceptual_weight > 0:
            try:
                from torchvision import models
            except ImportError as error:
                raise ImportError(
                    "torchvision is required when perceptual_weight is non-zero"
                ) from error
            try:
                self.vgg = models.vgg16(pretrained=True).features[:16]
            except TypeError:
                self.vgg = models.vgg16(weights="DEFAULT").features[:16]
            for parameter in self.vgg.parameters():
                parameter.requires_grad = False
            self.vgg.eval()

    def perceptual(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return prediction.new_tensor(0.0)
        self.vgg.to(prediction.device)
        mean = prediction.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = prediction.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        prediction = (prediction - mean) / std
        target = (target - mean) / std
        if self.legacy_detached_perceptual:
            with torch.no_grad():
                prediction_features = self.vgg(prediction)
                target_features = self.vgg(target)
        else:
            prediction_features = self.vgg(prediction)
            with torch.no_grad():
                target_features = self.vgg(target)
        return F.l1_loss(prediction_features, target_features)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        repair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if repair_mask.dim() == 3:
            repair_mask = repair_mask.unsqueeze(1)
        if repair_mask.size(1) != prediction.size(1):
            repair_mask = repair_mask.repeat(1, prediction.size(1), 1, 1)
        l1_value = self.l1(prediction * repair_mask, target * repair_mask)
        perceptual_value = self.perceptual(
            prediction * repair_mask, target * repair_mask
        ) * self.perceptual_weight
        total = l1_value + perceptual_value
        return total, {
            "l1": float(l1_value.detach().item()),
            "perceptual": float(perceptual_value.detach().item()),
            "total": float(total.detach().item()),
        }
