from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


RNN_TYPES = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}


@dataclass
class NormalizationStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        model_type: str,
        hidden_dims: Sequence[int],
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        model_key = model_type.upper()
        if model_key not in RNN_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")

        recurrent_cls = RNN_TYPES[model_key]
        self.layers = nn.ModuleList()
        directions = 2 if bidirectional else 1

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = recurrent_cls(
                input_size=current_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.layers.append(layer)
            current_dim = hidden_dim * directions

        self.regressor = nn.Linear(sequence_length * current_dim, sequence_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)[0]
        flattened = torch.flatten(outputs, start_dim=1)
        return self.regressor(flattened)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def validate_device(requested_device: Optional[str] = None) -> str:
    if requested_device is None:
        return get_device()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-enabled GPU is available in this environment.")
    return requested_device


def get_runtime_summary(device: Optional[str] = None) -> Dict[str, object]:
    resolved_device = validate_device(device)
    summary: Dict[str, object] = {
        "device": resolved_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if resolved_device == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        summary["gpu_name"] = None
    return summary


def load_well_log_data(
    workbook_path: str | Path,
    feature_columns: Sequence[str],
    target_column: str,
    sheet_number: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    sheet_name = sheet_number - 1
    required_columns = list(feature_columns) + [target_column]
    dataframe = pd.read_excel(
        workbook_path,
        sheet_name=sheet_name,
        engine="openpyxl",
        usecols=required_columns,
    )
    dataframe = dataframe.dropna(subset=required_columns).reset_index(drop=True)

    features = dataframe.loc[:, feature_columns].to_numpy(dtype=np.float32)
    targets = dataframe.loc[:, target_column].to_numpy(dtype=np.float32)
    return features, targets


def build_sequence_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    usable_rows = (len(targets) // sequence_length) * sequence_length
    if usable_rows == 0:
        raise ValueError("Not enough rows to build one sequence.")

    trimmed_features = features[:usable_rows]
    trimmed_targets = targets[:usable_rows]

    x_sequences = trimmed_features.reshape(-1, sequence_length, trimmed_features.shape[1])
    y_sequences = trimmed_targets.reshape(-1, sequence_length)
    return x_sequences, y_sequences


def split_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        shuffle=True,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def compute_normalization_stats(x_train: np.ndarray, y_train: np.ndarray) -> NormalizationStats:
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_std = x_train.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0

    return NormalizationStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def apply_normalization(
    splits: Mapping[str, np.ndarray],
    stats: NormalizationStats,
) -> Dict[str, np.ndarray]:
    normalized = dict(splits)
    for key in ("x_train", "x_val", "x_test"):
        normalized[key] = ((splits[key] - stats.x_mean) / stats.x_std).astype(np.float32)
    for key in ("y_train", "y_val", "y_test"):
        normalized[key] = ((splits[key] - stats.y_mean) / stats.y_std).astype(np.float32)
    return normalized


def prepare_data_splits(
    workbook_path: str | Path,
    feature_columns: Sequence[str] = ("DTCO", "ECGR", "RHOB", "PHIT"),
    target_column: str = "DTSM",
    sheet_number: int = 1,
    sequence_length: int = 8,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dict[str, np.ndarray], NormalizationStats]:
    features, targets = load_well_log_data(
        workbook_path=workbook_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sheet_number=sheet_number,
    )
    x_sequences, y_sequences = build_sequence_dataset(
        features=features,
        targets=targets,
        sequence_length=sequence_length,
    )
    splits = split_dataset(
        features=x_sequences,
        targets=y_sequences,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    stats = compute_normalization_stats(splits["x_train"], splits["y_train"])
    normalized_splits = apply_normalization(splits, stats)
    return normalized_splits, stats


def subset_split_arrays(
    splits: Mapping[str, np.ndarray],
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    subset = dict(splits)
    rng = np.random.default_rng(seed)

    if train_limit is not None and train_limit < len(subset["x_train"]):
        indices = rng.permutation(len(subset["x_train"]))[:train_limit]
        subset["x_train"] = subset["x_train"][indices]
        subset["y_train"] = subset["y_train"][indices]

    if val_limit is not None and val_limit < len(subset["x_val"]):
        indices = rng.permutation(len(subset["x_val"]))[:val_limit]
        subset["x_val"] = subset["x_val"][indices]
        subset["y_val"] = subset["y_val"][indices]

    return subset


def make_dataloaders(
    splits: Mapping[str, np.ndarray],
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Dict[str, DataLoader]:
    resolved_device = validate_device(device)
    pin_memory = resolved_device == "cuda"
    train_dataset = SequenceDataset(splits["x_train"], splits["y_train"])
    val_dataset = SequenceDataset(splits["x_val"], splits["y_val"])
    test_dataset = SequenceDataset(splits["x_test"], splits["y_test"])

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory),
    }


def build_hidden_dims(base_hidden_size: int, num_layers: int) -> List[int]:
    return [base_hidden_size * (2 ** index) for index in range(num_layers)]


def create_model(
    model_type: str,
    input_dim: int,
    sequence_length: int,
    base_hidden_size: int,
    num_layers: int,
    bidirectional: bool = False,
    device: Optional[str] = None,
) -> RecurrentRegressor:
    resolved_device = validate_device(device)
    model = RecurrentRegressor(
        input_dim=input_dim,
        sequence_length=sequence_length,
        model_type=model_type,
        hidden_dims=build_hidden_dims(base_hidden_size, num_layers),
        bidirectional=bidirectional,
    )
    return model.to(resolved_device)


def train_one_model(
    model: nn.Module,
    dataloaders: Mapping[str, DataLoader],
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 20,
    patience: int = 4,
    device: Optional[str] = None,
) -> Dict[str, object]:
    device = validate_device(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    transfer_kwargs = {"non_blocking": device == "cuda"}

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(max_epochs):
        model.train()
        train_losses: List[float] = []
        for features, targets in dataloaders["train"]:
            features = features.to(device, **transfer_kwargs)
            targets = targets.to(device, **transfer_kwargs)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for features, targets in dataloaders["val"]:
                features = features.to(device, **transfer_kwargs)
                targets = targets.to(device, **transfer_kwargs)
                predictions = model(features)
                loss = criterion(predictions, targets)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_loss": best_val_loss}


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device = validate_device(device)
    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    transfer_kwargs = {"non_blocking": device == "cuda"}

    with torch.no_grad():
        for features, batch_targets in dataloader:
            features = features.to(device, **transfer_kwargs)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def inverse_transform_targets(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return values * stats.y_std + stats.y_mean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    return {"rmse": rmse, "r2": r2}


def train_and_evaluate(
    model_type: str,
    splits: Mapping[str, np.ndarray],
    stats: NormalizationStats,
    base_hidden_size: int,
    num_layers: int,
    learning_rate: float,
    batch_size: int,
    bidirectional: bool = False,
    weight_decay: float = 0.0,
    max_epochs: int = 20,
    patience: int = 4,
    device: Optional[str] = None,
) -> Dict[str, object]:
    device = validate_device(device)
    dataloaders = make_dataloaders(splits, batch_size=batch_size, device=device)
    model = create_model(
        model_type=model_type,
        input_dim=splits["x_train"].shape[-1],
        sequence_length=splits["y_train"].shape[-1],
        base_hidden_size=base_hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        device=device,
    )
    train_result = train_one_model(
        model=model,
        dataloaders=dataloaders,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )

    val_predictions_norm, val_targets_norm = predict(train_result["model"], dataloaders["val"], device=device)
    test_predictions_norm, test_targets_norm = predict(train_result["model"], dataloaders["test"], device=device)

    val_predictions = inverse_transform_targets(val_predictions_norm, stats)
    val_targets = inverse_transform_targets(val_targets_norm, stats)
    test_predictions = inverse_transform_targets(test_predictions_norm, stats)
    test_targets = inverse_transform_targets(test_targets_norm, stats)

    val_metrics = compute_metrics(val_targets, val_predictions)
    test_metrics = compute_metrics(test_targets, test_predictions)

    return {
        "model": train_result["model"],
        "history": train_result["history"],
        "config": {
            "model_type": model_type,
            "base_hidden_size": base_hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "bidirectional": bidirectional,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "predictions": {
            "val": val_predictions,
            "test": test_predictions,
        },
        "targets": {
            "val": val_targets,
            "test": test_targets,
        },
    }


def iter_search_space(search_space: Mapping[str, Sequence[object]]) -> Iterable[Dict[str, object]]:
    keys = list(search_space.keys())
    value_lists = [list(search_space[key]) for key in keys]
    for values in product(*value_lists):
        yield dict(zip(keys, values))


def tune_hyperparameters(
    model_type: str,
    splits: Mapping[str, np.ndarray],
    stats: NormalizationStats,
    search_space: Mapping[str, Sequence[object]],
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    tune_epochs: int = 12,
    tune_patience: int = 3,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, object]:
    subset_splits = subset_split_arrays(
        splits=splits,
        train_limit=train_limit,
        val_limit=val_limit,
        seed=seed,
    )

    records: List[Dict[str, object]] = []
    best_run: Optional[Dict[str, object]] = None
    best_rmse = float("inf")

    for config in iter_search_space(search_space):
        run = train_and_evaluate(
            model_type=model_type,
            splits=subset_splits,
            stats=stats,
            base_hidden_size=int(config["base_hidden_size"]),
            num_layers=int(config["num_layers"]),
            learning_rate=float(config["learning_rate"]),
            batch_size=int(config["batch_size"]),
            bidirectional=False,
            weight_decay=float(config.get("weight_decay", 0.0)),
            max_epochs=tune_epochs,
            patience=tune_patience,
            device=device,
        )
        record = {
            "model": model_type,
            "sequence_length": int(subset_splits["y_train"].shape[-1]),
            **run["config"],
            "val_rmse": run["val_metrics"]["rmse"],
            "val_r2": run["val_metrics"]["r2"],
        }
        records.append(record)

        if run["val_metrics"]["rmse"] < best_rmse:
            best_rmse = run["val_metrics"]["rmse"]
            best_run = run

    if best_run is None:
        raise RuntimeError(f"No successful hyperparameter run for {model_type}.")

    return {
        "best_config": best_run["config"],
        "search_records": records,
        "best_val_metrics": best_run["val_metrics"],
    }


def run_assignment_workflow(
    workbook_path: str | Path,
    search_space: Optional[Mapping[str, Sequence[object]]] = None,
    model_types: Sequence[str] = ("RNN", "GRU", "LSTM"),
    feature_columns: Sequence[str] = ("DTCO", "ECGR", "RHOB", "PHIT"),
    target_column: str = "DTSM",
    sheet_number: int = 1,
    sequence_length: int = 8,
    test_size: float = 0.1,
    val_size: float = 0.1,
    tune_train_limit: Optional[int] = 12000,
    tune_val_limit: Optional[int] = 3000,
    tune_epochs: int = 10,
    tune_patience: int = 3,
    final_epochs: int = 18,
    final_patience: int = 4,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, object]:
    set_seed(seed)
    device = validate_device(device)

    if search_space is None:
        search_space = {
            "base_hidden_size": [8, 16],
            "num_layers": [2, 4],
            "learning_rate": [1e-3, 5e-4],
            "batch_size": [128],
            "weight_decay": [0.0],
        }

    splits, stats = prepare_data_splits(
        workbook_path=workbook_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sheet_number=sheet_number,
        sequence_length=sequence_length,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
    )

    search_results: Dict[str, Dict[str, object]] = {}
    final_rows: List[Dict[str, object]] = []
    runs: Dict[str, Dict[str, object]] = {}

    for model_type in model_types:
        tune_result = tune_hyperparameters(
            model_type=model_type,
            splits=splits,
            stats=stats,
            search_space=search_space,
            train_limit=tune_train_limit,
            val_limit=tune_val_limit,
            tune_epochs=tune_epochs,
            tune_patience=tune_patience,
            device=device,
            seed=seed,
        )
        search_results[model_type] = tune_result
        best_config = tune_result["best_config"]

        base_run = train_and_evaluate(
            model_type=model_type,
            splits=splits,
            stats=stats,
            base_hidden_size=int(best_config["base_hidden_size"]),
            num_layers=int(best_config["num_layers"]),
            learning_rate=float(best_config["learning_rate"]),
            batch_size=int(best_config["batch_size"]),
            bidirectional=False,
            weight_decay=float(best_config.get("weight_decay", 0.0)),
            max_epochs=final_epochs,
            patience=final_patience,
            device=device,
        )
        bidirectional_run = train_and_evaluate(
            model_type=model_type,
            splits=splits,
            stats=stats,
            base_hidden_size=int(best_config["base_hidden_size"]),
            num_layers=int(best_config["num_layers"]),
            learning_rate=float(best_config["learning_rate"]),
            batch_size=int(best_config["batch_size"]),
            bidirectional=True,
            weight_decay=float(best_config.get("weight_decay", 0.0)),
            max_epochs=final_epochs,
            patience=final_patience,
            device=device,
        )

        base_name = model_type.upper()
        bidi_name = f"Bi{model_type.upper()}"
        runs[base_name] = base_run
        runs[bidi_name] = bidirectional_run

        final_rows.append(
            {
                "model": base_name,
                "sequence_length": sequence_length,
                "rmse": base_run["test_metrics"]["rmse"],
                "r2": base_run["test_metrics"]["r2"],
                "base_hidden_size": best_config["base_hidden_size"],
                "num_layers": best_config["num_layers"],
                "learning_rate": best_config["learning_rate"],
                "batch_size": best_config["batch_size"],
            }
        )
        final_rows.append(
            {
                "model": bidi_name,
                "sequence_length": sequence_length,
                "rmse": bidirectional_run["test_metrics"]["rmse"],
                "r2": bidirectional_run["test_metrics"]["r2"],
                "base_hidden_size": best_config["base_hidden_size"],
                "num_layers": best_config["num_layers"],
                "learning_rate": best_config["learning_rate"],
                "batch_size": best_config["batch_size"],
            }
        )

    final_rows.sort(key=lambda row: row["rmse"])

    dataset_summary = {
        "num_sequences": int(splits["x_train"].shape[0] + splits["x_val"].shape[0] + splits["x_test"].shape[0]),
        "sequence_length": int(splits["x_train"].shape[1]),
        "num_features": int(splits["x_train"].shape[2]),
        "train_sequences": int(splits["x_train"].shape[0]),
        "val_sequences": int(splits["x_val"].shape[0]),
        "test_sequences": int(splits["x_test"].shape[0]),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }

    return {
        "dataset_summary": dataset_summary,
        "search_results": search_results,
        "comparison_rows": final_rows,
        "runs": runs,
        "normalization_stats": stats,
    }


def rows_to_markdown(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body_lines])


def summarize_best_model(comparison_rows: Sequence[Mapping[str, object]]) -> str:
    if not comparison_rows:
        return "No experiment results available."

    best = comparison_rows[0]
    return (
        f"The best model is {best['model']} with RMSE={best['rmse']:.4f} "
        f"and R2={best['r2']:.4f} at sequence_length={best['sequence_length']}. "
        f"It used base_hidden_size={best['base_hidden_size']}, "
        f"num_layers={best['num_layers']}, learning_rate={best['learning_rate']}, "
        f"and batch_size={best['batch_size']}."
    )


def summarize_assignment_rationale(
    comparison_rows: Sequence[Mapping[str, object]],
    best_rows_by_sequence: Sequence[Mapping[str, object]],
) -> str:
    if not comparison_rows:
        return "No experiment results available."

    best_overall = comparison_rows[0]
    top_two = comparison_rows[:2]
    rmse_margin = None
    if len(top_two) > 1:
        rmse_margin = float(top_two[1]["rmse"]) - float(top_two[0]["rmse"])

    bidirectional_rows = [
        row for row in comparison_rows
        if str(row.get("model", "")).startswith("Bi")
    ]
    bidirectional_best = bidirectional_rows[0] if bidirectional_rows else None

    lines = ["### Final Discussion and Rationale"]
    lines.append(
        f"- The selected best model is `{best_overall['model']}` at sequence length `{best_overall['sequence_length']}` because it achieved the lowest test RMSE ({best_overall['rmse']:.4f}) while also keeping a strong R2 score ({best_overall['r2']:.4f})."
    )

    if rmse_margin is not None:
        lines.append(
            f"- The gap to the second-best model is `{rmse_margin:.4f}` RMSE, so the final choice is based on a direct performance advantage rather than only model complexity."
        )

    if bidirectional_best is not None:
        lines.append(
            f"- The strongest bidirectional variant is `{bidirectional_best['model']}` with RMSE={bidirectional_best['rmse']:.4f} and R2={bidirectional_best['r2']:.4f}, which helps show whether using bidirectional recurrent layers improves prediction quality on this dataset."
        )

    if best_rows_by_sequence:
        sequence_summaries = []
        for row in sorted(best_rows_by_sequence, key=lambda item: item["sequence_length"]):
            sequence_summaries.append(
                f"`seq={row['sequence_length']}`: `{row['model']}` (RMSE={row['rmse']:.4f}, R2={row['r2']:.4f})"
            )
        lines.append(
            "- Comparing sequence lengths shows the best result at each setting: "
            + "; ".join(sequence_summaries)
            + "."
        )

        if len(best_rows_by_sequence) >= 2:
            best_sequence_row = min(best_rows_by_sequence, key=lambda row: row["rmse"])
            lines.append(
                f"- Based on the sequence-length comparison, sequence length `{best_sequence_row['sequence_length']}` is preferred because its best model produced the lowest RMSE among the tested sequence settings."
            )

    lines.append(
        "- This gives a practical rationale for model selection: prioritize the configuration with the lowest RMSE, verify that its R2 remains competitive, and then use the sequence-length comparison to justify whether a longer temporal context helps."
    )
    return "\n".join(lines)
