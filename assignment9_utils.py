"""Reusable utilities for Assignment 9.

This module contains data loading, training, evaluation, and plotting helpers.
The notebook stays as the main program and calls these functions.
"""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else path.stem


class OpenFWIVelocityDataset(Dataset):
    """Memory-mapped OpenFWI velocity-model dataset.

    Each .npy file is expected to have shape (N, 1, 70, 70). Samples are
    normalized to [0, 1] on the fly, which keeps memory use low.
    """

    def __init__(self, data_dir, max_files=None, max_samples=None):
        self.data_dir = Path(data_dir)
        paths = sorted(self.data_dir.glob("*.npy"), key=natural_key)
        if not paths:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir.resolve()}")

        self.paths = paths if max_files is None else paths[:max_files]
        self.arrays = [np.load(path, mmap_mode="r") for path in self.paths]
        sample_shapes = {tuple(arr.shape[1:]) for arr in self.arrays}
        if len(sample_shapes) != 1:
            raise ValueError(f"Inconsistent sample shapes: {sample_shapes}")

        self.sample_shape = next(iter(sample_shapes))
        self.file_lengths = np.array([arr.shape[0] for arr in self.arrays], dtype=np.int64)
        self.cumulative = np.cumsum(self.file_lengths)
        total = int(self.cumulative[-1])
        self.length = min(total, int(max_samples)) if max_samples is not None else total

        self.vmin = min(float(arr.min()) for arr in self.arrays)
        self.vmax = max(float(arr.max()) for arr in self.arrays)
        if self.vmax <= self.vmin:
            raise ValueError("Invalid velocity range for normalization.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx = int(np.searchsorted(self.cumulative, idx, side="right"))
        file_start = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        local_idx = idx - file_start
        x = np.array(self.arrays[file_idx][local_idx], dtype=np.float32, copy=True)
        x = ((x - self.vmin) / (self.vmax - self.vmin + 1e-8)).astype(np.float32, copy=False)
        return torch.from_numpy(x)

    def summary(self):
        return {
            "data_dir": str(self.data_dir),
            "num_files": len(self.paths),
            "first_file": self.paths[0].name,
            "last_file": self.paths[-1].name,
            "num_samples": len(self),
            "sample_shape": self.sample_shape,
            "velocity_min": self.vmin,
            "velocity_max": self.vmax,
        }


def create_loaders(dataset, batch_size=64, train_fraction=0.9, seed=42, num_workers=0):
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def reference_batch(dataset, n=512):
    count = min(n, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=int)
    return torch.stack([dataset[int(idx)] for idx in indices], dim=0)


def show_grid(images, title, n=8, vmin=0.0, vmax=1.0, cmap="viridis"):
    imgs = images.detach().cpu() if isinstance(images, torch.Tensor) else torch.tensor(images)
    imgs = imgs[:n].float().clamp(0, 1)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.2))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, imgs):
        ax.imshow(img.squeeze().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_sample_sets(sample_sets, n=8, cmap="viridis"):
    fig, axes = plt.subplots(len(sample_sets), n, figsize=(2.0 * n, 2.2 * len(sample_sets)))
    if len(sample_sets) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, (name, images) in enumerate(sample_sets):
        imgs = images.detach().cpu().float().clamp(0, 1)
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx].squeeze().numpy(), cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(name, rotation=0, labelpad=36, va="center")
    plt.tight_layout()
    plt.show()


def vae_loss(recon_x, x, mean, log_var, beta=1.0):
    recon = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return recon + beta * kld, recon, kld


def evaluate_vae(model, loader, device, flatten=False):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_pixels = 0
    total_items = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_model = x.view(x.size(0), -1) if flatten else x
            recon, mean, log_var = model(x_model)
            loss, _, _ = vae_loss(recon, x_model, mean, log_var)
            recon_img = recon.view_as(x) if flatten else recon
            total_loss += loss.item() * x.size(0)
            total_mse += F.mse_loss(recon_img, x, reduction="sum").item()
            total_pixels += x.numel()
            total_items += x.size(0)
    return {"val_loss": total_loss / total_items, "val_mse": total_mse / total_pixels}


def train_vae(model, train_loader, val_loader, device, epochs, lr, flatten, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_items = 0
        for x in train_loader:
            x = x.to(device)
            x_model = x.view(x.size(0), -1) if flatten else x
            optimizer.zero_grad()
            recon, mean, log_var = model(x_model)
            loss, _, _ = vae_loss(recon, x_model, mean, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            total_items += x.size(0)
        val_metrics = evaluate_vae(model, val_loader, device=device, flatten=flatten)
        row = {"epoch": epoch, "train_loss": train_loss / total_items, **val_metrics}
        history.append(row)
        print(f"{name} epoch {epoch:02d}/{epochs} | train loss {row['train_loss']:.3f} | val mse {row['val_mse']:.6f}")
    print(f"{name} finished in {(time.time() - start) / 60:.1f} min")
    return history


def sample_vae(model, n_samples, latent_dim, img_shape, device, flatten=False):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        samples = model.decoder(z)
        if flatten:
            samples = samples.view(n_samples, *img_shape)
        return samples.clamp(0, 1).cpu()


def train_gan(generator, discriminator, train_loader, device, epochs, lr, latent_dim, name):
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        batches = 0

        for x in train_loader:
            real_imgs = x.to(device) * 2.0 - 1.0
            batch_size = real_imgs.size(0)
            valid = torch.full((batch_size, 1), 0.9, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            optimizer_g.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_d.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            batches += 1

        row = {"epoch": epoch, "d_loss": d_loss_sum / batches, "g_loss": g_loss_sum / batches}
        history.append(row)
        print(f"{name} epoch {epoch:02d}/{epochs} | D loss {row['d_loss']:.4f} | G loss {row['g_loss']:.4f}")

    print(f"{name} finished in {(time.time() - start) / 60:.1f} min")
    return history


def sample_gan(generator, n_samples, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        samples = generator(z)
        return ((samples + 1.0) / 2.0).clamp(0, 1).cpu()


def to_numpy_images(images):
    arr = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    return arr.astype(np.float32)


def generation_metrics(real_images, fake_images, bins=40):
    real_arr = to_numpy_images(real_images)
    fake_arr = to_numpy_images(fake_images)
    n = min(len(real_arr), len(fake_arr))
    real_arr = real_arr[:n]
    fake_arr = np.clip(fake_arr[:n], 0, 1)
    real_flat = real_arr.reshape(n, -1)
    fake_flat = fake_arr.reshape(n, -1)

    real_stats = np.array([real_flat.mean(), real_flat.std(), real_flat.min(), real_flat.max()])
    fake_stats = np.array([fake_flat.mean(), fake_flat.std(), fake_flat.min(), fake_flat.max()])
    stats_error = float(np.mean(np.abs(real_stats - fake_stats)))

    hist_real, _ = np.histogram(real_flat, bins=bins, range=(0, 1))
    hist_fake, _ = np.histogram(fake_flat, bins=bins, range=(0, 1))
    hist_real = hist_real / (hist_real.sum() + 1e-8)
    hist_fake = hist_fake / (hist_fake.sum() + 1e-8)
    hist_error = float(np.mean(np.abs(hist_real - hist_fake)))

    def grad_stats(arr):
        gx = np.diff(arr, axis=-1)
        gy = np.diff(arr, axis=-2)
        return np.array([np.mean(np.abs(gx)), np.mean(np.abs(gy)), np.std(gx), np.std(gy)])

    grad_error = float(np.mean(np.abs(grad_stats(real_arr) - grad_stats(fake_arr))))
    score = stats_error + hist_error + grad_error
    return {
        "generation_score": score,
        "stats_error": stats_error,
        "hist_error": hist_error,
        "grad_error": grad_error,
    }


def result_row(model_name, family, real_images, fake_images, extra=None):
    row = {"model": model_name, "family": family}
    row.update(generation_metrics(real_images, fake_images))
    if extra:
        row.update(extra)
    return row
