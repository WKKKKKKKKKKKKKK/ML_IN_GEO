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


class PatchUNET(nn.Module):
    def __init__(self, patch_dim: int, base_width: int, latent_dim: int):
        super().__init__()
        width2 = base_width // 2
        width3 = base_width // 4
        width4 = base_width // 8
        width5 = base_width // 16
        if min(width2, width3, width4, width5, latent_dim) < 1:
            raise ValueError("Invalid PatchUNET width configuration")

        self.e1 = nn.Linear(patch_dim, base_width)
        self.ae1 = nn.ELU(1.0)
        self.e2 = nn.Linear(base_width, width2)
        self.ae2 = nn.ELU(1.0)
        self.e3 = nn.Linear(width2, width3)
        self.ae3 = nn.ELU(1.0)
        self.e4 = nn.Linear(width3, width4)
        self.ae4 = nn.ELU(1.0)
        self.e5 = nn.Linear(width4, width5)
        self.ae5 = nn.ELU(1.0)
        self.e6 = nn.Linear(width5, latent_dim)
        self.ae6 = nn.ELU(1.0)

        self.d1 = nn.Linear(latent_dim, latent_dim)
        self.de1 = nn.ELU(1.0)
        self.d2 = nn.Linear(2 * latent_dim, width5)
        self.de2 = nn.ELU(1.0)
        self.d3 = nn.Linear(2 * width5, width4)
        self.de3 = nn.ELU(1.0)
        self.d4 = nn.Linear(2 * width4, width3)
        self.de4 = nn.ELU(1.0)
        self.d5 = nn.Linear(2 * width3, width2)
        self.de5 = nn.ELU(1.0)
        self.d6 = nn.Linear(2 * width2, base_width)
        self.de6 = nn.ELU(1.0)
        self.out = nn.Linear(2 * base_width, patch_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.ae1(self.e1(inputs))
        e2 = self.ae2(self.e2(e1))
        e3 = self.ae3(self.e3(e2))
        e4 = self.ae4(self.e4(e3))
        e5 = self.ae5(self.e5(e4))
        e6 = self.ae6(self.e6(e5))

        d1 = self.de1(self.d1(e6))
        d1 = torch.cat((d1, e6), dim=-1)
        d2 = self.de2(self.d2(d1))
        d2 = torch.cat((d2, e5), dim=-1)
        d3 = self.de3(self.d3(d2))
        d3 = torch.cat((d3, e4), dim=-1)
        d4 = self.de4(self.d4(d3))
        d4 = torch.cat((d4, e3), dim=-1)
        d5 = self.de5(self.d5(d4))
        d5 = torch.cat((d5, e2), dim=-1)
        d6 = self.de6(self.d6(d5))
        d6 = torch.cat((d6, e1), dim=-1)
        return self.out(d6)


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
    pred_patches = torch.cat(outputs, dim=0)
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
    latent_dim: int,
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
    patch_dim = patch_size * patch_size
    train_inputs = noisy_patches.reshape(-1, patch_dim).float()

    dataset = torch.utils.data.TensorDataset(train_inputs, train_inputs)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = PatchUNET(patch_dim=patch_dim, base_width=base_width, latent_dim=latent_dim).to(device)
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
            plot_path = output_dir / f"p{patch_size}_s{stride}_w{base_width}_z{latent_dim}_lr{learning_rate:g}_b{batch_size}_epoch{epoch:04d}.png"
            create_plot(
                noisy_input=data_noisy1,
                denoised=reconstructed,
                clean_target=data_clean1,
                save_path=plot_path,
                title=f"PatchUNET p={patch_size}, s={stride}, w={base_width}, z={latent_dim}, lr={learning_rate}, b={batch_size}, epoch={epoch}",
            )
            plot_paths.append(str(plot_path))

    return {
        "config": {
            "patch_size": patch_size,
            "stride": stride,
            "base_width": base_width,
            "latent_dim": latent_dim,
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


def load_q1_reference(path: Path) -> dict | None:
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
        {"patch_size": 48, "stride": 8, "base_width": 128, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 8, "base_width": 128, "latent_dim": 4, "learning_rate": 5e-4, "batch_size": 256},
        {"patch_size": 48, "stride": 8, "base_width": 128, "latent_dim": 8, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 8, "base_width": 256, "latent_dim": 8, "learning_rate": 1e-3, "batch_size": 128},
        {"patch_size": 48, "stride": 8, "base_width": 64, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 4, "base_width": 128, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 48, "stride": 16, "base_width": 128, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 32, "stride": 8, "base_width": 128, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 32, "stride": 4, "base_width": 128, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 256},
        {"patch_size": 64, "stride": 8, "base_width": 128, "latent_dim": 4, "learning_rate": 1e-3, "batch_size": 128},
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
            latent_dim=candidate["latent_dim"],
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
        latent_dim=best_cfg["latent_dim"],
        learning_rate=best_cfg["learning_rate"],
        batch_size=best_cfg["batch_size"],
        num_epochs=args.final_epochs,
        plot_interval=args.plot_interval,
        output_dir=output_dir / "final_run",
        seed=args.seed,
    )

    q1_reference = load_q1_reference(Path(args.q1_results_path))
    comparison = None
    if q1_reference is not None:
        q1_best = float(q1_reference["final_run"]["best_snr"])
        q1_final = float(q1_reference["final_run"]["final_snr"])
        comparison = {
            "q1_path": str(Path(args.q1_results_path).resolve()),
            "q1_best_snr": q1_best,
            "q1_final_snr": q1_final,
            "patchunet_best_snr": final_result["best_snr"],
            "patchunet_final_snr": final_result["final_snr"],
            "best_snr_gain_vs_q1": final_result["best_snr"] - q1_best,
            "final_snr_gain_vs_q1": final_result["final_snr"] - q1_final,
        }

    return {
        "selection_mode": "patchunet_notebook_reproduction_with_clean_snr_tracking",
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
        "comparison_to_q1": comparison,
    }


def write_summary(results: dict, output_path: Path) -> None:
    final_run = results["final_run"]
    cfg = final_run["config"]
    comparison = results["comparison_to_q1"]
    comparison_text = "Q1 comparison file was not found."
    if comparison is not None:
        comparison_text = (
            f"- Q1 best SNR: {comparison['q1_best_snr']:.4f} dB\n"
            f"- PatchUNET best SNR: {comparison['patchunet_best_snr']:.4f} dB\n"
            f"- Best-SNR gain vs Q1: {comparison['best_snr_gain_vs_q1']:.4f} dB\n"
            f"- Q1 final SNR: {comparison['q1_final_snr']:.4f} dB\n"
            f"- PatchUNET final SNR: {comparison['patchunet_final_snr']:.4f} dB\n"
            f"- Final-SNR gain vs Q1: {comparison['final_snr_gain_vs_q1']:.4f} dB"
        )

    summary = f"""PatchUNET notebook reproduction summary
=================================

Setup
-----
- Model: notebook-style PatchUNET with patch-wise MLP encoder-decoder
- Training target: noisy patches
- Device: {results['device']}
- Selected noise standard deviation: {results['noise_std']:.6f}

Best hyperparameters after the search
-------------------------------------
- patch_size = {cfg['patch_size']}
- stride = {cfg['stride']}
- base_width = {cfg['base_width']}
- latent_dim = {cfg['latent_dim']}
- learning_rate = {cfg['learning_rate']}
- batch_size = {cfg['batch_size']}

SNR
---
- Original noisy-shot SNR: {final_run['before_snr']:.4f} dB
- Best denoised-shot SNR during the final run: {final_run['best_snr']:.4f} dB
- Best epoch: {final_run['best_epoch']}
- Final epoch SNR: {final_run['final_snr']:.4f} dB
- Improvement at the best epoch: {final_run['best_snr'] - final_run['before_snr']:.4f} dB

Comparison to Q1 UNET
---------------------
{comparison_text}

Saved notebook-style plots
--------------------------
""" + "\n".join(f"- {plot_path}" for plot_path in final_run["plot_paths"]) + "\n"
    output_path.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the Week7DIP PatchUNET experiment with saved plots.")
    parser.add_argument("--data-path", default="assignment5/Marmousi2_5hz_Data.npy")
    parser.add_argument("--q1-results-path", default="assignment5/no_skip/results.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--search-epochs", type=int, default=120)
    parser.add_argument("--final-epochs", type=int, default=300)
    parser.add_argument("--plot-interval", type=int, default=50)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--output-dir", default="assignment5/patchunet/plots")
    parser.add_argument("--results-json", default="assignment5/patchunet/results.json")
    parser.add_argument("--summary-path", default="assignment5/patchunet/summary.txt")
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
                "comparison_to_q1": results["comparison_to_q1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
