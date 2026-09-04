"""FORESIGHT network used for the experiments reported in the manuscript.

The module/parameter names intentionally match the historical checkpoints.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FourierConv(nn.Module):
    """Low-frequency spectral convolution with learned complex weights."""

    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        real = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft.real[:, :, : self.modes, : self.modes],
            self.weights_real,
        ) - torch.einsum(
            "bixy,ioxy->boxy",
            x_ft.imag[:, :, : self.modes, : self.modes],
            self.weights_imag,
        )
        imag = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft.real[:, :, : self.modes, : self.modes],
            self.weights_imag,
        ) + torch.einsum(
            "bixy,ioxy->boxy",
            x_ft.imag[:, :, : self.modes, : self.modes],
            self.weights_real,
        )
        out_ft[:, :, : self.modes, : self.modes] = torch.complex(real, imag)
        return torch.fft.irfft2(out_ft, s=(height, width))


class ResidualFourierBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 16, dropout: float = 0.1):
        super().__init__()
        self.fourier_conv = FourierConv(channels, channels, modes)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.linear = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fourier_conv(self.norm1(x))
        x = self.dropout(x) + residual
        residual = x
        x = F.gelu(self.linear(self.norm2(x)))
        return self.dropout(x) + residual


class FORESIGHT(nn.Module):
    """Full-resolution Fourier residual image-reconstruction network.

    ``repair_mask`` uses the public convention: 1 = reconstruct, 0 = preserve.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        modes: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualFourierBlock(hidden_dim, modes, dropout) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, 3, padding=1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, repair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        original_x = x
        if repair_mask is not None:
            if repair_mask.dim() == 3:
                repair_mask = repair_mask.unsqueeze(1)
            if repair_mask.size(1) != x.size(1):
                repair_mask = repair_mask.repeat(1, x.size(1), 1, 1)
            x = x * (1.0 - repair_mask)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)

        if repair_mask is not None:
            x = original_x * (1.0 - repair_mask) + x * repair_mask
        return x


# Historical class name retained so original checkpoints and scripts remain usable.
LaMa = FORESIGHT
