import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator


DEFAULT_NOISE_LEVELS = torch.linspace(0.5, 1.0, 10)
NOTEBOOK_NOISE_INDEX = 5
NOTEBOOK_NOISE_STD = float(DEFAULT_NOISE_LEVELS[NOTEBOOK_NOISE_INDEX])
EPSILON = 1e-10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path: Path) -> torch.Tensor:
    data = np.load(data_path)
    data = np.transpose(data, (1, 0, 2))[14:15, :, :]
    return torch.tensor(data, dtype=torch.float32)


def build_noisy_dataset(clean_data: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    set_seed(seed)
    dat = clean_data.clone()
    noisy_shots = []
    clean_shots = []
    snr_values = []

    for i in range(dat.shape[0]):
        for noise_level in DEFAULT_NOISE_LEVELS:
            noise = torch.normal(mean=0.0, std=float(noise_level), size=dat[i].shape)
            clean_shot = dat[i]
            noisy_shot = dat[i] + noise
            signal_power = torch.mean(dat[i] ** 2)
            noise_power = torch.mean(noise ** 2) + EPSILON
            snr = 10.0 * torch.log10(signal_power / noise_power)
            snr_values.append(float(snr))
            noisy_shots.append(noisy_shot)
            clean_shots.append(clean_shot)

    data_noisy = torch.stack(noisy_shots).reshape(-1, 1, dat.shape[1], dat.shape[2])
    data_clean = torch.stack(clean_shots).reshape(-1, 1, dat.shape[1], dat.shape[2])
    return data_noisy, data_clean, snr_values


def notebook_snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    difference_power = torch.mean((estimate - reference) ** 2)
    noisy_power = torch.mean(estimate ** 2) + EPSILON
    snr = 10.0 * torch.log10(noisy_power / (difference_power + EPSILON))
    return float(snr)


def patchify(inp: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int], inv: bool = False, orig_size: tuple[int, int] | None = None) -> torch.Tensor:
    if not inv:
        out = inp.unfold(0, kernel_size[0], stride[0]).unfold(1, kernel_size[1], stride[1])
        return out.reshape(-1, kernel_size[0], kernel_size[1])

    if orig_size is None:
        raise ValueError("orig_size is required when inv=True")

    out = inp.reshape(-1, kernel_size[0] * kernel_size[1])
    out = F.fold(out.transpose(0, 1), output_size=orig_size, kernel_size=kernel_size, stride=stride)[0]
    divisor = torch.ones_like(inp)
    divisor = divisor.reshape(-1, kernel_size[0] * kernel_size[1])
    divisor = F.fold(divisor.transpose(0, 1), output_size=orig_size, kernel_size=kernel_size, stride=stride)[0]
    return out / divisor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNPatchUNET(nn.Module):
    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        w1 = base_width
        w2 = base_width * 2
        w3 = base_width * 4

        self.enc1 = ConvBlock(in_channels, w1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(w1, w2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(w2, w3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(w3, w3)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(w3 + w3, w3)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(w3 + w2, w2)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(w2 + w1, w1)
        self.out = nn.Conv2d(w1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat((d1, e3), dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat((d3, e1), dim=1))
        return self.out(d3)


def create_plot(noisy_input: torch.Tensor, denoised: torch.Tensor, clean_target: torch.Tensor, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    font = {"family": "DejaVu Sans", "weight": "bold", "size": 10}
    plt.rc("font", **font)
    fig, ax = plt.subplots(1, 4, figsize=(14, 10), sharey=False)

    dx = 20
    dt = 0.006
    panels = [
        (noisy_input, "Noisy Input"),
        (denoised, "Denoised Data"),
        (clean_target, "Clean Reference"),
        (denoised - clean_target, "Error"),
    ]

    for axis, (panel, panel_title) in zip(ax, panels):
        vmin, vmax = torch.quantile(panel, torch.tensor([0.05, 0.95], device=panel.device))
        axis.imshow(
            panel.detach().cpu(),
            aspect="auto",
            cmap="gray",
            vmin=float(vmin),
            vmax=float(vmax),
            extent=[0, panel.shape[1] * dx / 1000.0, panel.shape[0] * dt, 0],
        )
        axis.set_xlabel("Position (km)", fontsize="large", fontweight="bold")
        axis.set_ylabel("Time (s)", fontsize="large", fontweight="bold")
        axis.set_title(panel_title, fontsize="large", fontweight="bold")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def reconstruct_shot(model: nn.Module, patches: torch.Tensor, patch_size: int, stride: int, batch_size: int, device: torch.device, shot_shape: tuple[int, int]) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            batch = patches[start:start + batch_size].to(device)
            pred = model(batch)
            outputs.append(pred.detach().cpu())
    pred_patches = torch.cat(outputs, dim=0).squeeze(1)
    out = patchify(pred_patches, (patch_size, patch_size), (stride, stride), inv=True, orig_size=shot_shape)
    return out


def train_config(
    *,
    clean_data: torch.Tensor,
    noisy_data: torch.Tensor,
    device: torch.device,
    patch_size: int,
    stride: int,
    base_width: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    plot_interval: int,
    output_dir: Path,
    seed: int,
) -> dict:
    set_seed(seed)

    data_noisy1 = noisy_data[NOTEBOOK_NOISE_INDEX, 0]
    data_clean1 = clean_data[NOTEBOOK_NOISE_INDEX, 0]
    noisy_patches = patchify(data_noisy1.cpu(), (patch_size, patch_size), (stride, stride))
    train_inputs = noisy_patches.unsqueeze(1).float()

    dataset = torch.utils.data.TensorDataset(train_inputs, train_inputs)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CNNPatchUNET(in_channels=1, base_width=base_width).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    before_snr = notebook_snr(data_clean1, data_noisy1)
    best_snr = -float("inf")
    best_epoch = -1
    history = []
    plot_paths = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * features.shape[0]

        reconstructed = reconstruct_shot(
            model=model,
            patches=train_inputs,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            shot_shape=(data_noisy1.shape[0], data_noisy1.shape[1]),
        )
        after_snr = notebook_snr(data_clean1, reconstructed)
        epoch_loss /= len(dataset)
        history.append({"epoch": epoch, "loss": epoch_loss, "snr_after": after_snr})

        if after_snr > best_snr:
            best_snr = after_snr
            best_epoch = epoch

        if epoch == 1 or epoch % plot_interval == 0 or epoch == num_epochs:
            plot_path = output_dir / f"cnn_p{patch_size}_s{stride}_w{base_width}_lr{learning_rate:g}_b{batch_size}_epoch{epoch:04d}.png"
            create_plot(
                noisy_input=data_noisy1,
                denoised=reconstructed,
                clean_target=data_clean1,
                save_path=plot_path,
                title=f"CNN-PatchUNET p={patch_size}, s={stride}, w={base_width}, lr={learning_rate}, b={batch_size}, epoch={epoch}",
            )
            plot_paths.append(str(plot_path))

    return {
        "config": {
            "patch_size": patch_size,
            "stride": stride,
            "base_width": base_width,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "plot_interval": plot_interval,
            "seed": seed,
        },
        "before_snr": before_snr,
        "final_snr": history[-1]["snr_after"],
        "best_snr": best_snr,
        "best_epoch": best_epoch,
        "history": history,
        "plot_paths": plot_paths,
    }


def load_reference(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_experiment(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    data = load_data(Path(args.data_path))
    noisy_data, clean_data, noise_snr_values = build_noisy_dataset(data, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparameter_candidates = [
        {"patch_size": 32, "stride": 4, "base_width": 16, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 32, "stride": 4, "base_width": 32, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 32, "stride": 8, "base_width": 32, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 4, "base_width": 16, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 4, "base_width": 32, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 8, "base_width": 32, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 8, "base_width": 32, "learning_rate": 5e-4, "batch_size": 256},
        {"patch_size": 48, "stride": 4, "base_width": 64, "learning_rate": 5e-4, "batch_size": 128},
        {"patch_size": 64, "stride": 8, "base_width": 16, "learning_rate": 1e-3, "batch_size": 128},
        {"patch_size": 64, "stride": 8, "base_width": 32, "learning_rate": 5e-4, "batch_size": 128},
    ]

    search_results = []
    for candidate in hyperparameter_candidates:
        result = train_config(
            clean_data=clean_data,
            noisy_data=noisy_data,
            device=device,
            patch_size=candidate["patch_size"],
            stride=candidate["stride"],
            base_width=candidate["base_width"],
            learning_rate=candidate["learning_rate"],
            batch_size=candidate["batch_size"],
            num_epochs=args.search_epochs,
            plot_interval=args.plot_interval,
            output_dir=output_dir / "search",
            seed=args.seed,
        )
        search_results.append(result)

    best_search = max(search_results, key=lambda item: item["best_snr"])
    best_cfg = best_search["config"]

    final_result = train_config(
        clean_data=clean_data,
        noisy_data=noisy_data,
        device=device,
        patch_size=best_cfg["patch_size"],
        stride=best_cfg["stride"],
        base_width=best_cfg["base_width"],
        learning_rate=best_cfg["learning_rate"],
        batch_size=best_cfg["batch_size"],
        num_epochs=args.final_epochs,
        plot_interval=args.plot_interval,
        output_dir=output_dir / "final_run",
        seed=args.seed,
    )

    q1_reference = load_reference(Path(args.q1_results_path))
    mlp_reference = load_reference(Path(args.patchunet_results_path))
    comparison = {}
    if q1_reference is not None:
        comparison["q1_unet"] = {
            "path": str(Path(args.q1_results_path).resolve()),
            "best_snr": float(q1_reference["final_run"]["best_snr"]),
            "final_snr": float(q1_reference["final_run"]["final_snr"]),
            "best_snr_gain": final_result["best_snr"] - float(q1_reference["final_run"]["best_snr"]),
            "final_snr_gain": final_result["final_snr"] - float(q1_reference["final_run"]["final_snr"]),
        }
    if mlp_reference is not None:
        comparison["mlp_patchunet"] = {
            "path": str(Path(args.patchunet_results_path).resolve()),
            "best_snr": float(mlp_reference["final_run"]["best_snr"]),
            "final_snr": float(mlp_reference["final_run"]["final_snr"]),
            "best_snr_gain": final_result["best_snr"] - float(mlp_reference["final_run"]["best_snr"]),
            "final_snr_gain": final_result["final_snr"] - float(mlp_reference["final_run"]["final_snr"]),
        }

    return {
        "selection_mode": "cnn_patchunet_notebook_reproduction_with_clean_snr_tracking",
        "data_path": str(Path(args.data_path).resolve()),
        "device": str(device),
        "noise_std": NOTEBOOK_NOISE_STD,
        "noise_snr_values": noise_snr_values,
        "search_epochs": args.search_epochs,
        "final_epochs": args.final_epochs,
        "hyperparameter_candidates": hyperparameter_candidates,
        "search_results": [
            {
                "config": item["config"],
                "before_snr": item["before_snr"],
                "final_snr": item["final_snr"],
                "best_snr": item["best_snr"],
                "best_epoch": item["best_epoch"],
            }
            for item in search_results
        ],
        "best_search_config": best_search["config"],
        "final_run": final_result,
        "comparison": comparison,
    }


def write_summary(results: dict, output_path: Path) -> None:
    final_run = results["final_run"]
    cfg = final_run["config"]
    comparison_lines = []
    for label, values in results["comparison"].items():
        comparison_lines.extend(
            [
                f"- {label} best SNR: {values['best_snr']:.4f} dB",
                f"- CNN-PatchUNET best-SNR gain vs {label}: {values['best_snr_gain']:.4f} dB",
                f"- {label} final SNR: {values['final_snr']:.4f} dB",
                f"- CNN-PatchUNET final-SNR gain vs {label}: {values['final_snr_gain']:.4f} dB",
            ]
        )
    comparison_text = "\n".join(comparison_lines) if comparison_lines else "Reference comparison files were not found."

    summary = f"""CNN-PatchUNET experiment summary
============================

Setup
-----
- Model: CNN-based PatchUNET replacing the fully connected layers with convolutional encoder-decoder blocks
- Training target: noisy patches
- Device: {results['device']}
- Selected noise standard deviation: {results['noise_std']:.6f}

Best hyperparameters after the search
-------------------------------------
- patch_size = {cfg['patch_size']}
- stride = {cfg['stride']}
- base_width = {cfg['base_width']}
- learning_rate = {cfg['learning_rate']}
- batch_size = {cfg['batch_size']}

SNR
---
- Original noisy-shot SNR: {final_run['before_snr']:.4f} dB
- Best denoised-shot SNR during the final run: {final_run['best_snr']:.4f} dB
- Best epoch: {final_run['best_epoch']}
- Final epoch SNR: {final_run['final_snr']:.4f} dB
- Improvement at the best epoch: {final_run['best_snr'] - final_run['before_snr']:.4f} dB

Comparison
----------
{comparison_text}

Saved notebook-style plots
--------------------------
""" + "\n".join(f"- {plot_path}" for plot_path in final_run["plot_paths"]) + "\n"
    output_path.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CNN PatchUNET experiment and compare it with MLP PatchUNET and Q1 UNET.")
    parser.add_argument("--data-path", default="assignment5/Marmousi2_5hz_Data.npy")
    parser.add_argument("--q1-results-path", default="assignment5/no_skip/results.json")
    parser.add_argument("--patchunet-results-path", default="assignment5/patchunet/results.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--search-epochs", type=int, default=120)
    parser.add_argument("--final-epochs", type=int, default=300)
    parser.add_argument("--plot-interval", type=int, default=50)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--output-dir", default="assignment5/patchunet_cnn/plots")
    parser.add_argument("--results-json", default="assignment5/patchunet_cnn/results.json")
    parser.add_argument("--summary-path", default="assignment5/patchunet_cnn/summary.txt")
    args = parser.parse_args()

    results = run_experiment(args)
    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, Path(args.summary_path))
    print(
        json.dumps(
            {
                "best_search_config": results["best_search_config"],
                "before_snr": results["final_run"]["before_snr"],
                "best_snr": results["final_run"]["best_snr"],
                "best_epoch": results["final_run"]["best_epoch"],
                "final_snr": results["final_run"]["final_snr"],
                "comparison": results["comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
