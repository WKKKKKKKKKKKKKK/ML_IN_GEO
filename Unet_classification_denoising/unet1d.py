from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


NormType = Literal["batch", "group", "none"]


def _make_norm(norm: NormType, num_channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm == "group":
        # A common stable choice for 1D convs across batch sizes.
        num_groups = 8
        if num_channels < num_groups:
            num_groups = 1
        elif num_channels % num_groups != 0:
            # Fall back to a divisor.
            for g in (4, 2, 1):
                if num_channels % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        norm: NormType = "group",
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.residual = residual and (in_channels == out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = _make_norm(norm, out_channels)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = _make_norm(norm, out_channels)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if self.residual:
            x = x + identity
        x = self.act2(x)
        x = self.drop2(x)
        return x


@dataclass(frozen=True)
class UNet1DConfig:
    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4
    base_channels: int = 32
    channel_mult: int = 2
    kernel_size: int = 5
    norm: NormType = "group"
    dropout: float = 0.0
    residual: bool = True


class UNet1D(nn.Module):
    """
    1D U-Net for reconstruction (autoencoder-style).

    Handles arbitrary input lengths by padding to a multiple of 2**depth and cropping back.
    """

    def __init__(self, cfg: UNet1DConfig) -> None:
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("depth must be >= 1")
        self.cfg = cfg

        chs: list[int] = [cfg.base_channels * (cfg.channel_mult**i) for i in range(cfg.depth)]

        self.in_proj = nn.Conv1d(cfg.in_channels, chs[0], kernel_size=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(cfg.depth):
            in_ch = chs[i] if i == 0 else chs[i]
            self.enc_blocks.append(
                ConvBlock1D(
                    in_ch,
                    chs[i],
                    kernel_size=cfg.kernel_size,
                    norm=cfg.norm,
                    dropout=cfg.dropout,
                    residual=cfg.residual,
                )
            )
            if i < cfg.depth - 1:
                self.downs.append(nn.Conv1d(chs[i], chs[i + 1], kernel_size=4, stride=2, padding=1))

        bottleneck_ch = chs[-1]
        self.bottleneck = ConvBlock1D(
            bottleneck_ch,
            bottleneck_ch,
            kernel_size=cfg.kernel_size,
            norm=cfg.norm,
            dropout=cfg.dropout,
            residual=cfg.residual,
        )

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(cfg.depth - 1)):
            in_ch = chs[i + 1]
            out_ch = chs[i]
            self.ups.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            self.dec_blocks.append(
                ConvBlock1D(
                    out_ch + out_ch,
                    out_ch,
                    kernel_size=cfg.kernel_size,
                    norm=cfg.norm,
                    dropout=cfg.dropout,
                    residual=cfg.residual,
                )
            )

        self.out_proj = nn.Conv1d(chs[0], cfg.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        orig_len = x.shape[-1]
        mult = 2 ** (self.cfg.depth - 1)
        if mult > 1:
            target_len = ((orig_len + mult - 1) // mult) * mult
            pad_right = target_len - orig_len
            if pad_right:
                x = nn.functional.pad(x, (0, pad_right), mode="reflect")

        x = self.in_proj(x)
        skips: list[torch.Tensor] = []
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        x = self.bottleneck(x)

        for up, dec_blk, skip in zip(self.ups, self.dec_blocks, reversed(skips[:-1])):
            x = up(x)
            # Handle any off-by-one from padding/stride rounding.
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = nn.functional.pad(x, (0, diff))
                else:
                    x = x[..., : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = dec_blk(x)

        x = self.out_proj(x)
        x = x[..., :orig_len]
        return x

