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


class UNET(nn.Module):
    def __init__(
        self,
        input_channels: int,
        fm1: int,
        fm2: int,
        fm3: int,
        bottleneck_dim: int,
        output_channels: int,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.encoder_conv1 = nn.Conv2d(input_channels, fm1, kernel_size=3, stride=1, padding=1)
        self.encoder_ac1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv2 = nn.Conv2d(fm1, fm2, kernel_size=3, stride=1, padding=1)
        self.encoder_ac2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv3 = nn.Conv2d(fm2, fm3, kernel_size=3, stride=1, padding=1)
        self.encoder_ac3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_bottleneck = nn.Conv2d(fm3, bottleneck_dim, kernel_size=3, stride=1, padding=1)
        self.encoder_bottleneck_ac = nn.LeakyReLU(0.1)

        self.decoder_upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder_conv1 = nn.Conv2d(bottleneck_dim, fm3, kernel_size=3, padding=1)
        self.decoder_ac1 = nn.LeakyReLU(0.1)

        self.decoder_upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder_conv2 = nn.Conv2d(fm3, fm2, kernel_size=3, padding=1)
        self.decoder_ac2 = nn.LeakyReLU(0.1)

        self.decoder_upsample3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder_conv3 = nn.Conv2d(fm2, fm1, kernel_size=3, padding=1)
        self.decoder_ac3 = nn.LeakyReLU(0.1)

        self.decoder_convf = nn.Conv2d(fm1, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.encoder_ac1(self.encoder_conv1(x))
        p1 = self.pool1(e1)

        e2 = self.encoder_ac2(self.encoder_conv2(p1))
        p2 = self.pool2(e2)

        e3 = self.encoder_ac3(self.encoder_conv3(p2))
        p3 = self.pool3(e3)

        bottleneck = self.encoder_bottleneck_ac(self.encoder_bottleneck(p3))

        d1 = self.decoder_ac1(self.decoder_conv1(self.decoder_upsample1(bottleneck)))
        if self.use_skip_connections:
            d1 = d1 + e3
        d2 = self.decoder_ac2(self.decoder_conv2(self.decoder_upsample2(d1)))
        if self.use_skip_connections:
            d2 = d2 + e2
        d3 = self.decoder_ac3(self.decoder_conv3(self.decoder_upsample3(d2)))
        if self.use_skip_connections:
            d3 = d3 + e1
        out = self.decoder_convf(d3)
        return bottleneck, out


def create_plot(net_input: torch.Tensor, outputs: torch.Tensor, target: torch.Tensor, device: torch.device, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    font = {"family": "DejaVu Sans", "weight": "bold", "size": 10}
    plt.rc("font", **font)
    fig, ax = plt.subplots(1, 4, figsize=(14, 10), sharey=False)

    dx = 20
    dt = 0.006
    shotnum = 0

    panels = [
        (net_input[shotnum, 0], "Input Noise"),
        (outputs[shotnum, 0], "Denoised Data"),
        (target[shotnum, 0], "Target Data"),
        (outputs[shotnum, 0] - target[shotnum, 0], "Error"),
    ]

    for axis, (panel, panel_title) in zip(ax, panels):
        selected_shot = panel.to(device)
        vmin, vmax = torch.quantile(selected_shot, torch.tensor([0.05, 0.95], device=device))
        axis.imshow(
            selected_shot.detach().cpu(),
            aspect="auto",
            cmap="gray",
            vmin=float(vmin),
            vmax=float(vmax),
            extent=[0, selected_shot.shape[1] * dx / 1000.0, selected_shot.shape[0] * dt, 0],
        )
        axis.set_xlabel("Position (km)", fontsize="large", fontweight="bold")
        axis.set_ylabel("Time (s)", fontsize="large", fontweight="bold")
        axis.set_title(panel_title, fontsize="large", fontweight="bold")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train_config(
    *,
    clean_data: torch.Tensor,
    noisy_data: torch.Tensor,
    device: torch.device,
    fm1: int,
    learning_rate: float,
    reg_noise_std: float,
    init_mode: str,
    init_scale: float,
    num_epochs: int,
    plot_interval: int,
    output_dir: Path,
    seed: int,
    use_skip_connections: bool,
) -> dict:
    set_seed(seed)

    data_noisy1 = noisy_data[NOTEBOOK_NOISE_INDEX : NOTEBOOK_NOISE_INDEX + 1].to(device)
    data_clean1 = clean_data[NOTEBOOK_NOISE_INDEX : NOTEBOOK_NOISE_INDEX + 1].to(device)
    shape = list(data_noisy1.shape)

    if init_mode == "gaussian":
        net_input = torch.zeros(shape, device=device).normal_()
    elif init_mode == "uniform":
        net_input = torch.rand(shape, device=device) * 2.0 - 1.0
    elif init_mode == "zeros":
        net_input = torch.zeros(shape, device=device)
    else:
        raise ValueError(f"Unsupported init mode: {init_mode}")

    net_input = net_input * init_scale
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    fm2 = 2 * fm1
    fm3 = 2 * fm2
    bottleneck_dim = 2 * fm3
    model = UNET(1, fm1, fm2, fm3, bottleneck_dim, 1, use_skip_connections=use_skip_connections).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    before_snr = notebook_snr(data_clean1[0, 0].detach().cpu(), data_noisy1[0, 0].detach().cpu())
    best_snr = -float("inf")
    best_epoch = -1
    best_output = None
    history = []
    plot_paths = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        optimizer.zero_grad()
        _, outputs = model(net_input)
        loss = criterion(outputs[0], data_noisy1)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            after_snr = notebook_snr(data_clean1[0, 0].detach().cpu(), outputs[0, 0].detach().cpu())
            history.append({"epoch": epoch, "loss": float(loss.item()), "snr_after": after_snr})
            if after_snr > best_snr:
                best_snr = after_snr
                best_epoch = epoch
                best_output = outputs.detach().cpu()

        if epoch == 1 or epoch % plot_interval == 0 or epoch == num_epochs:
            plot_path = output_dir / f"{init_mode}_fm{fm1}_lr{learning_rate:g}_rn{reg_noise_std:g}_epoch{epoch:04d}.png"
            create_plot(net_input.detach().cpu(), outputs.detach().cpu(), data_noisy1.detach().cpu(), device, plot_path, f"{init_mode}, fm1={fm1}, lr={learning_rate}, reg={reg_noise_std}, epoch={epoch}")
            plot_paths.append(str(plot_path))

    return {
        "config": {
            "fm1": fm1,
            "fm2": fm2,
            "fm3": fm3,
            "bottleneck_dim": bottleneck_dim,
            "learning_rate": learning_rate,
            "reg_noise_std": reg_noise_std,
            "init_mode": init_mode,
            "init_scale": init_scale,
            "num_epochs": num_epochs,
            "plot_interval": plot_interval,
            "seed": seed,
            "use_skip_connections": use_skip_connections,
        },
        "before_snr": before_snr,
        "final_snr": history[-1]["snr_after"],
        "best_snr": best_snr,
        "best_epoch": best_epoch,
        "history": history,
        "plot_paths": plot_paths,
    }


def run_experiment(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    data = load_data(Path(args.data_path))
    noisy_data, clean_data, noise_snr_values = build_noisy_dataset(data, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_modes = ["gaussian", "uniform", "zeros"]
    baseline_fm1 = 64
    baseline_lr = 1e-4
    baseline_reg_noise_std = 0.3
    hyperparameter_candidates = [
        {"fm1": 64, "learning_rate": 1e-4, "reg_noise_std": 0.2},
        {"fm1": 64, "learning_rate": 1e-4, "reg_noise_std": 0.3},
        {"fm1": 64, "learning_rate": 1e-4, "reg_noise_std": 0.4},
        {"fm1": 64, "learning_rate": 1e-4, "reg_noise_std": 0.5},
        {"fm1": 64, "learning_rate": 2e-4, "reg_noise_std": 0.2},
        {"fm1": 64, "learning_rate": 2e-4, "reg_noise_std": 0.3},
        {"fm1": 64, "learning_rate": 2e-4, "reg_noise_std": 0.4},
        {"fm1": 64, "learning_rate": 2e-4, "reg_noise_std": 0.5},
        {"fm1": 64, "learning_rate": 3e-4, "reg_noise_std": 0.2},
        {"fm1": 64, "learning_rate": 3e-4, "reg_noise_std": 0.3},
        {"fm1": 64, "learning_rate": 3e-4, "reg_noise_std": 0.4},
        {"fm1": 64, "learning_rate": 3e-4, "reg_noise_std": 0.5},
    ]

    init_results = []
    for init_mode in init_modes:
        init_scale = baseline_reg_noise_std if args.init_scale is None else args.init_scale
        result = train_config(
            clean_data=clean_data,
            noisy_data=noisy_data,
            device=device,
            fm1=baseline_fm1,
            learning_rate=baseline_lr,
            reg_noise_std=baseline_reg_noise_std,
            init_mode=init_mode,
            init_scale=init_scale,
            num_epochs=args.search_epochs,
            plot_interval=args.plot_interval,
            output_dir=output_dir / "init_search",
            seed=args.seed,
            use_skip_connections=args.use_skip_connections,
        )
        init_results.append(result)

    best_init = max(init_results, key=lambda item: item["best_snr"])
    best_init_mode = best_init["config"]["init_mode"]

    hyperparameter_results = []
    for candidate in hyperparameter_candidates:
        reg_noise_std = candidate["reg_noise_std"]
        init_scale = reg_noise_std if args.init_scale is None else args.init_scale
        result = train_config(
            clean_data=clean_data,
            noisy_data=noisy_data,
            device=device,
            fm1=candidate["fm1"],
            learning_rate=candidate["learning_rate"],
            reg_noise_std=reg_noise_std,
            init_mode=best_init_mode,
            init_scale=init_scale,
            num_epochs=args.search_epochs,
            plot_interval=args.plot_interval,
            output_dir=output_dir / "hyperparameter_search",
            seed=args.seed,
            use_skip_connections=args.use_skip_connections,
        )
        hyperparameter_results.append(result)

    best_search = max(hyperparameter_results, key=lambda item: item["best_snr"])
    best_cfg = best_search["config"]

    final_result = train_config(
        clean_data=clean_data,
        noisy_data=noisy_data,
        device=device,
        fm1=best_cfg["fm1"],
        learning_rate=best_cfg["learning_rate"],
        reg_noise_std=best_cfg["reg_noise_std"],
        init_mode=best_cfg["init_mode"],
        init_scale=best_cfg["init_scale"],
        num_epochs=args.final_epochs,
        plot_interval=args.plot_interval,
        output_dir=output_dir / "final_run",
        seed=args.seed,
        use_skip_connections=args.use_skip_connections,
    )

    return {
        "selection_mode": "notebook_reproduction_with_clean_snr_tracking",
        "data_path": str(Path(args.data_path).resolve()),
        "device": str(device),
        "use_skip_connections": args.use_skip_connections,
        "noise_std": NOTEBOOK_NOISE_STD,
        "noise_snr_values": noise_snr_values,
        "search_epochs": args.search_epochs,
        "final_epochs": args.final_epochs,
        "init_search_baseline": {
            "fm1": baseline_fm1,
            "learning_rate": baseline_lr,
            "reg_noise_std": baseline_reg_noise_std,
        },
        "hyperparameter_candidates": hyperparameter_candidates,
        "init_search_results": [
            {
                "config": item["config"],
                "before_snr": item["before_snr"],
                "final_snr": item["final_snr"],
                "best_snr": item["best_snr"],
                "best_epoch": item["best_epoch"],
            }
            for item in init_results
        ],
        "selected_init_mode": best_init_mode,
        "hyperparameter_search_results": [
            {
                "config": item["config"],
                "before_snr": item["before_snr"],
                "final_snr": item["final_snr"],
                "best_snr": item["best_snr"],
                "best_epoch": item["best_epoch"],
            }
            for item in hyperparameter_results
        ],
        "best_search_config": best_search["config"],
        "final_run": final_result,
    }


def write_summary(results: dict, output_path: Path) -> None:
    final_run = results["final_run"]
    cfg = final_run["config"]
    init_search_lines = "\n".join(
        f"- {item['config']['init_mode']}: best SNR = {item['best_snr']:.4f} dB at epoch {item['best_epoch']}"
        for item in results["init_search_results"]
    )
    model_name = "notebook-style UNET with additive skip connections" if results["use_skip_connections"] else "notebook-style UNET without active skip connections"
    summary = f"""Week7DIP UNET reproduction summary
================================

Setup
-----
- Model: {model_name}
- Training target: noisy shot
- Input: random noise tensor z with additive input perturbation during optimization
- Device: {results['device']}
- Selected noise standard deviation: {results['noise_std']:.6f}

Noise initialization comparison
-------------------------------
{init_search_lines}

Selected best initialization
----------------------------
- init_mode = {results['selected_init_mode']}

Best hyperparameters after the search
-------------------------------------
- fm1 = {cfg['fm1']}
- fm2 = {cfg['fm2']}
- fm3 = {cfg['fm3']}
- bottleneck_dim = {cfg['bottleneck_dim']}
- learning_rate = {cfg['learning_rate']}
- reg_noise_std = {cfg['reg_noise_std']}
- init_scale = {cfg['init_scale']}

SNR
---
- Original noisy-shot SNR: {final_run['before_snr']:.4f} dB
- Best denoised-shot SNR during the final run: {final_run['best_snr']:.4f} dB
- Best epoch: {final_run['best_epoch']}
- Final epoch SNR: {final_run['final_snr']:.4f} dB
- Improvement at the best epoch: {final_run['best_snr'] - final_run['before_snr']:.4f} dB

Saved notebook-style plots
--------------------------
""" + "\n".join(f"- {plot_path}" for plot_path in final_run["plot_paths"]) + "\n"
    output_path.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the Week7DIP UNET experiment with saved plots.")
    parser.add_argument("--data-path", default="assignment5/Marmousi2_5hz_Data.npy")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--search-epochs", type=int, default=1600)
    parser.add_argument("--final-epochs", type=int, default=2200)
    parser.add_argument("--plot-interval", type=int, default=500)
    parser.add_argument("--init-scale", type=float, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--use-skip-connections", action="store_true")
    parser.add_argument("--output-dir", default="assignment5/no_skip/plots")
    parser.add_argument("--results-json", default="assignment5/no_skip/results.json")
    parser.add_argument("--summary-path", default="assignment5/no_skip/summary.txt")
    args = parser.parse_args()

    results = run_experiment(args)
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
