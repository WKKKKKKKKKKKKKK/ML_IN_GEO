from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SaltDataset(Dataset):
    def __init__(self, image_files: list[Path], mask_dir: Path, image_size: int = 128) -> None:
        self.image_files = list(image_files)
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]
        mask_path = self.mask_dir / image_path.name

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask for {image_path.name}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(require_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA is required for this notebook, but torch.cuda.is_available() is False.")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_image_paths(image_dir: Path, seed: int = 42) -> tuple[list[Path], list[Path], list[Path]]:
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")

    train_val_paths, test_paths = train_test_split(
        image_paths,
        test_size=0.10,
        random_state=seed,
        shuffle=True,
    )
    train_paths, val_paths = train_test_split(
        train_val_paths,
        test_size=0.111111,
        random_state=seed,
        shuffle=True,
    )
    return list(train_paths), list(val_paths), list(test_paths)


def make_loaders(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    mask_dir: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_dataset = SaltDataset(train_paths, mask_dir, image_size)
    val_dataset = SaltDataset(val_paths, mask_dir, image_size)
    test_dataset = SaltDataset(test_paths, mask_dir, image_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def segmentation_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float]:
    preds = (probabilities > threshold).float()
    pixel_accuracy = (preds == targets).float().mean().item()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = ((intersection + eps) / (union + eps)).mean().item()
    return pixel_accuracy, iou


def _make_scaler(device: torch.device, amp_enabled: bool) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    extra_loss_fn: Callable[[nn.Module], torch.Tensor] | None = None,
    amp_enabled: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits.detach())
        acc, iou = segmentation_metrics(probs, masks)

        total_loss += loss.item()
        total_acc += acc
        total_iou += iou
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "iou": total_iou / total_batches,
    }


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    extra_loss_fn: Callable[[nn.Module], torch.Tensor] | None = None,
    amp_enabled: bool = False,
    label: str = "Epoch",
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float]:
    scaler = _make_scaler(device, amp_enabled)
    best_state = None
    best_val_iou = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            extra_loss_fn=extra_loss_fn,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            extra_loss_fn=extra_loss_fn,
            amp_enabled=amp_enabled,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"{label} {epoch:02d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f}, train IoU={train_metrics['iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, val IoU={val_metrics['iou']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training completed without producing a best model state.")

    return best_state, history, best_val_iou


def mc_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 20,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_images = []
    all_masks = []
    all_mean_probs = []
    all_std_probs = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            samples = []
            for _ in range(n_samples):
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                samples.append(torch.sigmoid(logits).unsqueeze(0))

            stacked = torch.cat(samples, dim=0)
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_mean_probs.append(stacked.mean(dim=0).cpu())
            all_std_probs.append(stacked.std(dim=0).cpu())

    return (
        torch.cat(all_images, dim=0),
        torch.cat(all_masks, dim=0),
        torch.cat(all_mean_probs, dim=0),
        torch.cat(all_std_probs, dim=0),
    )


def save_history_csv(history: list[dict[str, float]], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return path


def save_json(data: object, path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def plot_sample_batch(dataset: SaltDataset, save_path: Path, num_examples: int = 4) -> Path:
    ensure_dir(save_path.parent)
    fig, axes = plt.subplots(num_examples, 2, figsize=(6, 3 * num_examples))
    if num_examples == 1:
        axes = np.array([axes])

    for i in range(num_examples):
        image, mask = dataset[i]
        axes[i, 0].imshow(image.squeeze(0), cmap="gray")
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask.squeeze(0), cmap="gray")
        axes[i, 1].set_title(f"Mask {i + 1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_history(history: list[dict[str, float]], save_path: Path, title_prefix: str) -> Path:
    ensure_dir(save_path.parent)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_iou = [item["train_iou"] for item in history]
    val_iou = [item["val_iou"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_iou, label="Train")
    axes[1].plot(epochs, val_iou, label="Validation")
    axes[1].set_title(f"{title_prefix} IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_deterministic_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    probs: torch.Tensor,
    save_path: Path,
    num_examples: int = 4,
) -> Path:
    ensure_dir(save_path.parent)
    num_examples = min(num_examples, images.shape[0])
    preds = (probs > 0.5).float()

    fig, axes = plt.subplots(num_examples, 4, figsize=(14, 3 * num_examples))
    if num_examples == 1:
        axes = np.array([axes])

    for i in range(num_examples):
        axes[i, 0].imshow(images[i, 0], cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        prob_plot = axes[i, 2].imshow(probs[i, 0], cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Predicted Probability")
        axes[i, 2].axis("off")
        fig.colorbar(prob_plot, ax=axes[i, 2], fraction=0.046, pad=0.04)

        axes[i, 3].imshow(preds[i, 0], cmap="gray")
        axes[i, 3].set_title("Predicted Mask")
        axes[i, 3].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_bayesian_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    mean_probs: torch.Tensor,
    std_probs: torch.Tensor,
    save_path: Path,
    num_examples: int = 4,
) -> Path:
    ensure_dir(save_path.parent)
    num_examples = min(num_examples, images.shape[0])
    preds = (mean_probs > 0.5).float()

    fig, axes = plt.subplots(num_examples, 5, figsize=(18, 3 * num_examples))
    if num_examples == 1:
        axes = np.array([axes])

    for i in range(num_examples):
        axes[i, 0].imshow(images[i, 0], cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        mean_plot = axes[i, 2].imshow(mean_probs[i, 0], cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Mean Probability")
        axes[i, 2].axis("off")
        fig.colorbar(mean_plot, ax=axes[i, 2], fraction=0.046, pad=0.04)

        axes[i, 3].imshow(preds[i, 0], cmap="gray")
        axes[i, 3].set_title("Mean Prediction")
        axes[i, 3].axis("off")

        std_plot = axes[i, 4].imshow(std_probs[i, 0], cmap="magma")
        axes[i, 4].set_title("Predictive Uncertainty")
        axes[i, 4].axis("off")
        fig.colorbar(std_plot, ax=axes[i, 4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
