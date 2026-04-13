from __future__ import annotations

import torch
import torch.nn as nn
import torchbnn as bnn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ResidualBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualSegNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ResidualBlock(base_channels, base_channels, dropout=dropout)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, stride=2, dropout=dropout)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2, dropout=dropout)
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8, stride=2, dropout=dropout)

        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        xb = self.bottleneck(x3)

        x = self.dec3(xb, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.head(x)


class BayesianResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.conv1 = bnn.BayesConv2d(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = bnn.BayesConv2d(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = bnn.BayesConv2d(
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


class BayesianDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = BayesianResidualBlock(
            in_channels + skip_channels,
            out_channels,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class BayesianResidualSegNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.stem_conv = bnn.BayesConv2d(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.enc1 = BayesianResidualBlock(base_channels, base_channels, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.enc2 = BayesianResidualBlock(
            base_channels,
            base_channels * 2,
            stride=2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )
        self.enc3 = BayesianResidualBlock(
            base_channels * 2,
            base_channels * 4,
            stride=2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )
        self.bottleneck = BayesianResidualBlock(
            base_channels * 4,
            base_channels * 8,
            stride=2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

        self.dec3 = BayesianDecoderBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )
        self.dec2 = BayesianDecoderBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )
        self.dec1 = BayesianDecoderBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )
        self.head = bnn.BayesConv2d(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_channels=base_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.relu(self.stem_conv(x))
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        xb = self.bottleneck(x3)

        x = self.dec3(xb, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.head(x)


__all__ = [
    "BayesianResidualSegNet",
    "DecoderBlock",
    "ResidualBlock",
    "ResidualSegNet",
]
