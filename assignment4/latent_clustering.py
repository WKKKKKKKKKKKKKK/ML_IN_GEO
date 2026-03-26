from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from unet1d import UNet1D, UNet1DConfig


@dataclass
class LatentClusteringResult:
    """Container for PCA + K-means outputs."""

    latent: np.ndarray
    latent_pca: np.ndarray
    pca: PCA
    kmeans: KMeans
    cluster_labels: np.ndarray


def _get_device(device: Optional[torch.device | str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_best_unet(
    ckpt_path: str = "best_unet1d.pt",
    device: Optional[torch.device | str] = None,
) -> Tuple[UNet1D, UNet1DConfig, dict[str, Any]]:
    """
    Load the best UNet1D model checkpoint saved by unet_search.py.

    Returns (model, config, raw_checkpoint_dict).
    """
    device_t = _get_device(device)
    ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device_t)
    cfg = UNet1DConfig(**ckpt["cfg"])
    model = UNet1D(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device_t)
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def extract_bottleneck_features(
    model: UNet1D,
    X: np.ndarray,
    batch_size: int = 128,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """
    Extract bottleneck latent representations for 1D traces.

    Parameters
    ----------
    model:
        Trained UNet1D model.
    X:
        Numpy array of shape (N, L) containing 1D traces.
    batch_size:
        Mini-batch size for feature extraction.
    device:
        Torch device or string, e.g. "cpu" / "cuda". If None, auto-selects.

    Returns
    -------
    features : np.ndarray
        Array of shape (N, D) with flattened bottleneck features.
    """
    device_t = _get_device(device)
    model = model.to(device_t)

    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]
    feats: list[np.ndarray] = []

    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        xb = torch.from_numpy(X[start:end]).unsqueeze(1).to(device_t)  # (B, 1, L)

        # Manually follow UNet1D.forward until after the bottleneck.
        orig_len = xb.shape[-1]
        mult = 2 ** (model.cfg.depth - 1)
        if mult > 1:
            target_len = ((orig_len + mult - 1) // mult) * mult
            pad_right = target_len - orig_len
            if pad_right:
                xb = nn.functional.pad(xb, (0, pad_right), mode="reflect")

        x = model.in_proj(xb)
        skips = []
        for i, blk in enumerate(model.enc_blocks):
            x = blk(x)
            skips.append(x)
            if i < len(model.downs):
                x = model.downs[i](x)

        x = model.bottleneck(x)  # (B, C, L_bottleneck)

        feat = x.flatten(start_dim=1).cpu().numpy()
        feats.append(feat)

    features = np.concatenate(feats, axis=0)
    return features


def run_pca_kmeans(
    latent: np.ndarray,
    n_components_pca: int = 10,
    n_clusters: int = 2,
    random_state: int = 42,
) -> LatentClusteringResult:
    """
    Apply PCA (for dimensionality reduction) and K-means clustering.

    Parameters
    ----------
    latent:
        Latent features of shape (N, D).
    n_components_pca:
        Number of PCA components to keep before clustering.
    n_clusters:
        Number of K-means clusters.
    random_state:
        Random seed for PCA and K-means.
    """
    latent = np.asarray(latent, dtype=np.float32)

    pca = PCA(n_components=n_components_pca, random_state=random_state)
    latent_pca = pca.fit_transform(latent)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(latent_pca)

    return LatentClusteringResult(
        latent=latent,
        latent_pca=latent_pca,
        pca=pca,
        kmeans=kmeans,
        cluster_labels=cluster_labels,
    )


def pca_2d_for_visualization(
    latent: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Compute a 2D PCA projection of latent features for visualization only.
    """
    latent = np.asarray(latent, dtype=np.float32)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(latent)
    return coords, pca

