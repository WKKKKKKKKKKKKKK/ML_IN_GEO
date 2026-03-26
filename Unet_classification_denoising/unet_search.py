from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from unet1d import UNet1D, UNet1DConfig  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def eval_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, xb, reduction="sum")
        total += float(loss.item())
        count += int(xb.numel())
    return total / max(count, 1)


def train_one(
    cfg: UNet1DConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    amp: bool,
) -> dict[str, Any]:
    model = UNet1D(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(xb)
                loss = nn.functional.mse_loss(pred, xb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

        val_mse = eval_mse(model, val_loader, device)
        if val_mse + 1e-12 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "val_mse": best_val,
        "state_dict": model.state_dict(),
        "best_epoch": best_epoch,
    }


def train_fixed_epochs(
    cfg: UNet1DConfig,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
) -> dict[str, Any]:
    """Train for a fixed number of epochs (no early stopping)."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    model = UNet1D(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for _epoch in range(1, epochs + 1):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(xb)
                loss = nn.functional.mse_loss(pred, xb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

    return {"state_dict": model.state_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="SiesmicEventsClassification_Normalized.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--results", default="unet_search_results.jsonl")
    ap.add_argument("--save-best", default="best_unet1d.pt")
    ap.add_argument(
        "--full-retrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After selecting best config/epoch via val, retrain on train+val for best_epoch and save that model.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    npz = np.load(args.data)
    X = npz["data"].astype(np.float32)
    y = npz["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train, X_val, _, _ = train_test_split(
        X_train, y_train, test_size=0.125, random_state=args.seed, stratify=y_train
    )  # 0.8 * 0.125 = 0.1 => 70/10/20 split

    def to_loader(arr: np.ndarray, shuffle: bool) -> DataLoader:
        t = torch.from_numpy(arr).unsqueeze(1)  # (N, 1, L)
        ds = TensorDataset(t)
        return DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=shuffle,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )

    train_loader = to_loader(X_train, shuffle=True)
    val_loader = to_loader(X_val, shuffle=False)
    test_loader = to_loader(X_test, shuffle=False)

    grid = {
        "depth": [3, 4, 5],
        "base_channels": [16, 32],
        "kernel_size": [3, 5],
        "dropout": [0.0, 0.1],
        "norm": ["none"],
        "residual": [True],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    best = {"val_mse": float("inf")}
    best_cfg: UNet1DConfig | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None

    with open(args.results, "w", encoding="utf-8") as f:
        for idx, vals in enumerate(combos, start=1):
            cfg_kwargs = dict(zip(keys, vals))
            cfg = UNet1DConfig(**cfg_kwargs)

            out = train_one(
                cfg,
                train_loader,
                val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                amp=amp,
            )
            val_mse = float(out["val_mse"])
            trial_best_epoch = int(out["best_epoch"] or args.epochs)

            trial_model = UNet1D(cfg).to(device)
            trial_model.load_state_dict(out["state_dict"])
            test_mse = eval_mse(trial_model, test_loader, device)

            record = {
                "trial": idx,
                "cfg": asdict(cfg),
                "val_mse": val_mse,
                "test_mse": float(test_mse),
                "best_epoch": trial_best_epoch,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            if val_mse < best["val_mse"]:
                best = record
                best_cfg = cfg
                best_state = {k: v.detach().cpu().clone() for k, v in out["state_dict"].items()}
                best_epoch = trial_best_epoch

    if best_cfg is None or best_state is None:
        raise RuntimeError("No trials ran.")

    final_state = best_state
    final_test_mse: float

    if args.full_retrain:
        # Full retrain on train+val using the epoch chosen by validation early stopping.
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        trainval_loader = to_loader(X_trainval, shuffle=True)
        retrain_epochs = int(best_epoch or args.epochs)

        retr_out = train_fixed_epochs(
            best_cfg,
            trainval_loader,
            device=device,
            epochs=retrain_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amp=amp,
        )
        final_state = {k: v.detach().cpu().clone() for k, v in retr_out["state_dict"].items()}

        retr_model = UNet1D(best_cfg).to(device)
        retr_model.load_state_dict(final_state)
        final_test_mse = eval_mse(retr_model, test_loader, device)
    else:
        # Final test eval using the best checkpoint selected on validation.
        best_model = UNet1D(best_cfg).to(device)
        best_model.load_state_dict(best_state)
        final_test_mse = eval_mse(best_model, test_loader, device)

    torch.save(
        {
            "cfg": asdict(best_cfg),
            "state_dict": final_state,
            "seed": args.seed,
            "best_val_mse": best["val_mse"],
            "best_test_mse": final_test_mse,
            "best_epoch": int(best_epoch or args.epochs),
            "full_retrain": bool(args.full_retrain),
        },
        args.save_best,
    )

    print("Best UNet1DConfig:", best_cfg)
    print(f"Best val MSE:  {best['val_mse']:.8f}")
    print(f"Best test MSE: {final_test_mse:.8f}")
    if args.full_retrain:
        print(f"Full retrain: enabled (train+val) for {int(best_epoch or args.epochs)} epochs")
    print(f"Saved: {args.save_best}")
    print(f"Trials log: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
