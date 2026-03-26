from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """
    Simple 2D U-Net for denoising + interpolation.

    Treats seismic gather as an image:
      input/output: (B, 1, T, R)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        xb = self.bottleneck(self.pool3(x3))

        u3 = self.up3(xb)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        return self.out(d1)


def get_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def snr_db(clean: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    SNR(clean, estimate) where noise = estimate-clean. Returns per-sample SNR in dB.

    clean/estimate: (B, 1, T, R)
    """
    signal_power = (clean**2).mean(dim=(1, 2, 3))
    noise_power = ((estimate - clean) ** 2).mean(dim=(1, 2, 3)) + eps
    return 10.0 * torch.log10(signal_power / noise_power)


@dataclass
class TrainHistory:
    train_mse: list[float]
    test_mse: list[float]


def train_unet2d(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: Optional[str | torch.device] = None,
    epochs: int = 5,
    lr: float = 1e-3,
) -> TrainHistory:
    device_t = get_device(device)
    model = model.to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    hist = TrainHistory(train_mse=[], test_mse=[])

    for _epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_train += int(xb.size(0))

        train_mse = total_loss / max(n_train, 1)

        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                pred = model(xb)
                test_loss += float(loss_fn(pred, yb).item()) * xb.size(0)
                n_test += int(xb.size(0))
        test_mse = test_loss / max(n_test, 1)

        hist.train_mse.append(train_mse)
        hist.test_mse.append(test_mse)

    return hist


@torch.no_grad()
def predict(model: nn.Module, xb: torch.Tensor, *, device: Optional[str | torch.device] = None) -> torch.Tensor:
    device_t = get_device(device)
    model = model.to(device_t)
    model.eval()
    return model(xb.to(device_t))


@torch.no_grad()
def snr_improvement_for_batch(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    device: Optional[str | torch.device] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (snr_in, snr_out, delta) as numpy arrays, one value per sample.
    """
    device_t = get_device(device)
    model = model.to(device_t)
    model.eval()
    xb = xb.to(device_t)
    yb = yb.to(device_t)
    pred = model(xb)

    snr_in = snr_db(yb, xb).detach().cpu().numpy()
    snr_out = snr_db(yb, pred).detach().cpu().numpy()
    delta = snr_out - snr_in
    return snr_in, snr_out, delta

