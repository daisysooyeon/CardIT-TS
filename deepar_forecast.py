"""Leakage-safe DeepAR forecasts for the Berka master tables.

This first DeepAR implementation intentionally supports only the native
recursive forecast.  A single global LSTM model is trained for each flow
(``inflow`` and ``outflow``) across all selected accounts.  The model predicts
the parameters of a Normal distribution in ``log1p(flow / account_scale)``
space.  At test time only predictions are appended to the history; test
actuals are used only for evaluation.

Examples::

    .\\ts\\Scripts\\python.exe deepar_forecast.py \\
        --frequency monthly --output-dir outputs/deepar --device cuda

    .\\ts\\Scripts\\python.exe deepar_forecast.py \\
        --frequency both --output-dir outputs/deepar --device cuda

Use ``--max-accounts`` for a small smoke test.  Omitting it loads every
account in the parquet table.  The direct version is deliberately excluded
because horizon-specific DeepAR models would multiply the training cost by
6 or 27 for each flow.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    add_derived_flow_columns,
    calculate_metrics,
    derive_test_start,
    get_available_account_ids,
    read_master,
    save_forecast_plot,
    train_normalization_scale,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepar"
LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class DeepARData:
    frequency: str
    horizon: int
    df: pd.DataFrame
    account_ids: list[int]
    dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]
    test_start: pd.Timestamp
    test_start_index: int
    values_by_flow: dict[str, np.ndarray]
    scales_by_flow: dict[str, np.ndarray]
    dynamic_features: np.ndarray
    dynamic_feature_names: list[str]
    static_features: np.ndarray
    static_feature_names: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_regular_dates(dates: pd.DatetimeIndex, frequency: str) -> None:
    if frequency == "monthly":
        expected = pd.date_range(dates[0], dates[-1], freq="MS")
    else:
        expected = pd.date_range(dates[0], dates[-1], freq="W-SUN")
    if not dates.equals(expected):
        raise ValueError(
            f"{frequency} dates are not regular: expected {len(expected)} dates, "
            f"found {len(dates)}"
        )


def build_dynamic_features(
    dates: list[pd.Timestamp],
    frequency: str,
) -> tuple[np.ndarray, list[str]]:
    """Build covariates known for both historical and future periods."""
    index = pd.DatetimeIndex(dates)
    n_periods = len(index)
    time_position = np.arange(n_periods, dtype=np.float32)
    time_position /= max(n_periods - 1, 1)

    month = np.asarray(index.month, dtype=np.float32)
    features = [
        time_position,
        np.sin(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32),
        np.cos(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32),
    ]
    names = ["time_position", "month_sin", "month_cos"]

    if frequency == "weekly":
        week = np.asarray(index.isocalendar().week, dtype=np.float32)
        features.extend(
            [
                np.sin(2.0 * np.pi * (week - 1.0) / 52.0).astype(np.float32),
                np.cos(2.0 * np.pi * (week - 1.0) / 52.0).astype(np.float32),
            ]
        )
        names.extend(["week_sin", "week_cos"])

    return np.column_stack(features).astype(np.float32), names


def build_static_features(
    df: pd.DataFrame,
    account_ids: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Create static numeric and one-hot categorical account features."""
    static = (
        df.sort_values(["account_id", "period_start"])
        .groupby("account_id", sort=True)[
            ["birth_year", "gender", "account_frequency"]
        ]
        .first()
        .reindex(account_ids)
    )

    birth_year = pd.to_numeric(static["birth_year"], errors="coerce").astype(float)
    birth_year = birth_year.fillna(birth_year.median())
    birth_mean = float(birth_year.mean())
    birth_std = float(birth_year.std(ddof=0))
    if not np.isfinite(birth_std) or birth_std == 0:
        birth_std = 1.0
    parts = [((birth_year.to_numpy() - birth_mean) / birth_std).astype(np.float32)]
    names = ["birth_year_standardized"]

    for column in ("gender", "account_frequency"):
        values = static[column].astype("string").fillna("missing")
        categories = sorted(str(value) for value in values.unique())
        for category in categories:
            parts.append((values == category).to_numpy(dtype=np.float32))
            names.append(f"{column}={category}")

    return np.column_stack(parts).astype(np.float32), names


def value_matrix(
    df: pd.DataFrame,
    account_ids: list[int],
    dates: list[pd.Timestamp],
    value_column: str,
) -> np.ndarray:
    pivot = (
        df.pivot(index="account_id", columns="period_start", values=value_column)
        .reindex(index=account_ids, columns=dates)
    )
    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())
        raise ValueError(
            f"{value_column} has {missing:,} missing account-period values; "
            "DeepAR requires a regular panel."
        )
    return pivot.to_numpy(dtype=np.float32)


def create_deepar_data(
    frequency: str,
    account_ids: list[int] | None,
) -> DeepARData:
    df = read_master(frequency, account_ids)
    account_ids = sorted(int(value) for value in df["account_id"].unique())
    dates_index = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in df["period_start"].unique()))
    validate_regular_dates(dates_index, frequency)
    dates = list(dates_index)
    test_start = derive_test_start(frequency, df)
    test_start_index = int(np.searchsorted(dates_index.values, test_start.to_datetime64()))
    horizon = FREQUENCY_CONFIG[frequency]["horizon"]
    test_dates = dates[test_start_index : test_start_index + horizon]
    if len(test_dates) != horizon:
        raise ValueError(
            f"Expected {horizon} test dates for {frequency}, found {len(test_dates)}"
        )

    values_by_flow = {
        flow_name: value_matrix(df, account_ids, dates, value_column)
        for flow_name, value_column in FLOW_COLUMNS.items()
    }
    scales_by_flow = {}
    for flow_name, values in values_by_flow.items():
        scale = np.mean(np.abs(values[:, :test_start_index]), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        scales_by_flow[flow_name] = np.maximum(scale, 1.0).astype(np.float32)

    dynamic_features, dynamic_feature_names = build_dynamic_features(dates, frequency)
    static_features, static_feature_names = build_static_features(df, account_ids)
    return DeepARData(
        frequency=frequency,
        horizon=horizon,
        df=df,
        account_ids=account_ids,
        dates=dates,
        test_dates=test_dates,
        test_start=test_start,
        test_start_index=test_start_index,
        values_by_flow=values_by_flow,
        scales_by_flow=scales_by_flow,
        dynamic_features=dynamic_features,
        dynamic_feature_names=dynamic_feature_names,
        static_features=static_features,
        static_feature_names=static_feature_names,
    )


def make_window_indices(
    n_accounts: int,
    context_length: int,
    target_start: int,
    target_end: int,
) -> np.ndarray:
    """Return (account_index, target_index) pairs; target_end is exclusive."""
    target_start = max(int(target_start), context_length)
    target_end = int(target_end)
    if target_end <= target_start:
        return np.empty((0, 2), dtype=np.int64)
    target_indices = np.arange(target_start, target_end, dtype=np.int64)
    account_indices = np.repeat(np.arange(n_accounts, dtype=np.int64), len(target_indices))
    repeated_targets = np.tile(target_indices, n_accounts)
    return np.column_stack([account_indices, repeated_targets])


class DeepARWindowDataset(Dataset):
    """On-demand one-step training windows to avoid duplicating sequences."""

    def __init__(
        self,
        values: np.ndarray,
        scales: np.ndarray,
        dynamic_features: np.ndarray,
        static_features: np.ndarray,
        indices: np.ndarray,
        context_length: int,
    ) -> None:
        self.values = torch.from_numpy(values.astype(np.float32, copy=False))
        self.scales = torch.from_numpy(scales.astype(np.float32, copy=False))
        self.dynamic_features = torch.from_numpy(dynamic_features.astype(np.float32, copy=False))
        self.static_features = torch.from_numpy(static_features.astype(np.float32, copy=False))
        self.indices = indices
        self.context_length = int(context_length)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        account_index, target_index = self.indices[index]
        start = int(target_index) - self.context_length
        scale = self.scales[account_index]
        context_raw = torch.clamp(self.values[account_index, start:target_index], min=0.0)
        context_target = torch.log1p(context_raw / scale)
        target_raw = torch.clamp(self.values[account_index, target_index], min=0.0)
        target = torch.log1p(target_raw / scale)
        context_dynamic = self.dynamic_features[start:target_index]
        future_dynamic = self.dynamic_features[target_index]
        static = self.static_features[account_index]
        return context_target, context_dynamic, future_dynamic, static, target


class DeepARNetwork(nn.Module):
    """One-step probabilistic autoregressive RNN."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_size: int,
        rnn_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.config = {
            "dynamic_dim": dynamic_dim,
            "static_dim": static_dim,
            "hidden_size": hidden_size,
            "rnn_layers": rnn_layers,
            "dropout": dropout,
        }
        input_dim = 1 + dynamic_dim + static_dim
        recurrent_dropout = dropout if rnn_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        head_dim = hidden_size + dynamic_dim + static_dim
        self.head = nn.Sequential(
            nn.Linear(head_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.loc_head = nn.Linear(hidden_size, 1)
        self.log_scale_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        context_target: torch.Tensor,
        context_dynamic: torch.Tensor,
        future_dynamic: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        static_sequence = static.unsqueeze(1).expand(-1, context_target.size(1), -1)
        recurrent_input = torch.cat(
            [context_target.unsqueeze(-1), context_dynamic, static_sequence], dim=-1
        )
        encoded, _ = self.rnn(recurrent_input)
        last_hidden = encoded[:, -1, :]
        head_input = torch.cat([last_hidden, future_dynamic, static], dim=-1)
        hidden = self.head(head_input)
        loc = self.loc_head(hidden).squeeze(-1)
        log_scale = torch.clamp(self.log_scale_head(hidden).squeeze(-1), -5.0, 2.0)
        scale = F.softplus(log_scale) + 1e-3
        return loc, scale


def gaussian_nll(
    target: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    standardized = (target - loc) / scale
    return 0.5 * (
        standardized.square() + 2.0 * torch.log(scale) + LOG_2PI
    ).mean()


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate_nll(
    model: DeepARNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        context_target, context_dynamic, future_dynamic, static, target = move_batch(batch, device)
        loc, scale = model(context_target, context_dynamic, future_dynamic, static)
        loss = gaussian_nll(target, loc, scale)
        count = len(target)
        total_loss += float(loss.item()) * count
        total_count += count
    return total_loss / total_count if total_count else float("nan")


def fit_network(
    data: DeepARData,
    flow_name: str,
    train_dataset: DeepARWindowDataset,
    validation_dataset: DeepARWindowDataset | None,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    seed: int,
    early_stopping: bool,
) -> tuple[DeepARNetwork, dict[str, object]]:
    set_seed(seed)
    model = DeepARNetwork(
        dynamic_dim=data.dynamic_features.shape[1],
        static_dim=data.static_features.shape[1],
        hidden_size=args.hidden_size,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    validation_loader = (
        make_loader(
            validation_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )
        if validation_dataset is not None and len(validation_dataset) > 0
        else None
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            context_target, context_dynamic, future_dynamic, static, target = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loc, scale = model(context_target, context_dynamic, future_dynamic, static)
            loss = gaussian_nll(target, loc, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            count = len(target)
            total_loss += float(loss.item()) * count
            total_count += count

        train_loss = total_loss / total_count if total_count else float("nan")
        val_loss = evaluate_nll(model, validation_loader, device) if validation_loader else float("nan")
        history.append({"epoch": float(epoch), "train_nll": train_loss, "validation_nll": val_loss})

        if validation_loader:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping and epochs_without_improvement >= args.patience:
                    break

        if epoch == 1 or epoch == epochs or epoch % max(args.log_every, 1) == 0:
            val_text = f", val_nll={val_loss:.5f}" if validation_loader else ""
            print(
                f"[{flow_name}] epoch={epoch}/{epochs}, train_nll={train_loss:.5f}{val_text}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch == 0:
        best_epoch = epochs

    info = {
        "flow": flow_name,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_validation_nll": best_val_loss if validation_loader else np.nan,
        "last_train_nll": history[-1]["train_nll"] if history else np.nan,
        "history": history,
    }
    return model, info


def train_flow_model(
    data: DeepARData,
    flow_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[DeepARNetwork, dict[str, object]]:
    context_length = args.context_length or (
        12 if data.frequency == "monthly" else 52
    )
    if context_length >= data.test_start_index:
        raise ValueError(
            f"context_length={context_length} is too long for {data.frequency} "
            f"pre-test history of {data.test_start_index} periods"
        )

    validation_length = args.validation_length or data.horizon
    validation_length = min(validation_length, data.test_start_index - context_length)
    validation_start = data.test_start_index - validation_length
    values = data.values_by_flow[flow_name]
    scales = data.scales_by_flow[flow_name]

    train_indices = make_window_indices(
        len(data.account_ids), context_length, context_length, validation_start
    )
    validation_indices = make_window_indices(
        len(data.account_ids), context_length, validation_start, data.test_start_index
    )
    train_dataset = DeepARWindowDataset(
        values, scales, data.dynamic_features, data.static_features,
        train_indices, context_length,
    )
    validation_dataset = DeepARWindowDataset(
        values, scales, data.dynamic_features, data.static_features,
        validation_indices, context_length,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No training windows for {flow_name}/{data.frequency}")

    validation_model, validation_info = fit_network(
        data,
        flow_name,
        train_dataset,
        validation_dataset,
        device,
        args,
        epochs=args.epochs,
        seed=args.seed + (0 if flow_name == "inflow" else 1),
        early_stopping=True,
    )

    # Retrain on all pre-test windows after selecting the epoch count.  The
    # final test periods are never included in this dataset.
    final_indices = make_window_indices(
        len(data.account_ids), context_length, context_length, data.test_start_index
    )
    final_dataset = DeepARWindowDataset(
        values, scales, data.dynamic_features, data.static_features,
        final_indices, context_length,
    )
    final_model, final_info = fit_network(
        data,
        flow_name,
        final_dataset,
        None,
        device,
        args,
        epochs=int(validation_info["best_epoch"]),
        seed=args.seed + 100 + (0 if flow_name == "inflow" else 1),
        early_stopping=False,
    )
    info = {
        **{key: value for key, value in validation_info.items() if key != "history"},
        "final_train_samples": len(final_dataset),
        "final_epochs": final_info["epochs_ran"],
        "context_length": context_length,
        "validation_start": str(data.dates[validation_start].date()),
    }
    return final_model, info


def transformed_to_raw(
    loc: torch.Tensor,
    scale: torch.Tensor,
    quantile_z: float = 0.0,
) -> torch.Tensor:
    transformed = loc + quantile_z * scale
    transformed = torch.clamp(transformed, min=-20.0, max=20.0)
    return torch.expm1(transformed)


@torch.no_grad()
def predict_one_step(
    model: DeepARNetwork,
    histories: np.ndarray,
    scales: np.ndarray,
    data: DeepARData,
    target_index: int,
    context_length: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    n_accounts = len(data.account_ids)
    result: dict[str, list[np.ndarray]] = {
        "mean": [], "p10": [], "p50": [], "p90": [], "raw": []
    }
    start = target_index - context_length
    context_dynamic = np.broadcast_to(
        data.dynamic_features[start:target_index],
        (n_accounts, context_length, data.dynamic_features.shape[1]),
    ).copy()
    future_dynamic = np.broadcast_to(
        data.dynamic_features[target_index],
        (n_accounts, data.dynamic_features.shape[1]),
    ).copy()
    context_values = np.maximum(histories[:, -context_length:], 0.0)
    context_target = np.log1p(context_values / scales[:, None]).astype(np.float32)

    for batch_start in range(0, n_accounts, batch_size):
        batch_end = min(batch_start + batch_size, n_accounts)
        context_tensor = torch.from_numpy(context_target[batch_start:batch_end]).to(device)
        context_dynamic_tensor = torch.from_numpy(context_dynamic[batch_start:batch_end]).to(device)
        future_dynamic_tensor = torch.from_numpy(future_dynamic[batch_start:batch_end]).to(device)
        static_tensor = torch.from_numpy(data.static_features[batch_start:batch_end]).to(device)
        loc, distribution_scale = model(
            context_tensor,
            context_dynamic_tensor,
            future_dynamic_tensor,
            static_tensor,
        )
        account_scale = torch.from_numpy(scales[batch_start:batch_end]).to(device)
        mean_transformed = torch.clamp(
            loc + 0.5 * distribution_scale.square(), min=-20.0, max=20.0
        )
        mean = account_scale * torch.expm1(mean_transformed)
        p10 = account_scale * transformed_to_raw(
            loc, distribution_scale, quantile_z=-1.2815515655
        )
        p50 = account_scale * transformed_to_raw(loc, distribution_scale)
        p90 = account_scale * transformed_to_raw(
            loc, distribution_scale, quantile_z=1.2815515655
        )
        raw = mean
        result["mean"].append(mean.cpu().numpy())
        result["p10"].append(p10.cpu().numpy())
        result["p50"].append(p50.cpu().numpy())
        result["p90"].append(p90.cpu().numpy())
        result["raw"].append(raw.cpu().numpy())

    return {key: np.concatenate(value).astype(float) for key, value in result.items()}


def recursive_forecast(
    data: DeepARData,
    models: dict[str, DeepARNetwork],
    training_info: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Forecast test periods sequentially using predictions as future inputs."""
    context_lengths = {
        row["flow"]: int(row["context_length"])
        for _, row in training_info.iterrows()
    }
    histories = {
        flow_name: data.values_by_flow[flow_name][:, : data.test_start_index].copy()
        for flow_name in FLOW_TARGETS
    }
    records: list[dict[str, object]] = []

    for step, target_date in enumerate(data.test_dates, start=1):
        target_index = data.test_start_index + step - 1
        forecasts = {
            flow_name: predict_one_step(
                models[flow_name],
                histories[flow_name],
                data.scales_by_flow[flow_name],
                data,
                target_index,
                context_lengths[flow_name],
                device,
                args.batch_size,
            )
            for flow_name in FLOW_TARGETS
        }

        for account_index, account_id in enumerate(data.account_ids):
            inflow = forecasts["inflow"]
            outflow = forecasts["outflow"]
            record = {
                "account_id": int(account_id),
                "period_start": target_date,
                "origin_period": (
                    data.test_dates[0]
                    - FREQUENCY_CONFIG[data.frequency]["step"]
                    + (step - 1) * FREQUENCY_CONFIG[data.frequency]["step"]
                ),
                "horizon_step": step,
                "actual_inflow": float(data.values_by_flow["inflow"][account_index, target_index]),
                "actual_outflow": float(data.values_by_flow["outflow"][account_index, target_index]),
                "prediction_inflow_raw": float(inflow["raw"][account_index]),
                "prediction_inflow": float(max(inflow["mean"][account_index], 0.0)),
                "prediction_inflow_p10": float(max(inflow["p10"][account_index], 0.0)),
                "prediction_inflow_p50": float(max(inflow["p50"][account_index], 0.0)),
                "prediction_inflow_p90": float(max(inflow["p90"][account_index], 0.0)),
                "prediction_outflow_raw": float(outflow["raw"][account_index]),
                "prediction_outflow": float(max(outflow["mean"][account_index], 0.0)),
                "prediction_outflow_p10": float(max(outflow["p10"][account_index], 0.0)),
                "prediction_outflow_p50": float(max(outflow["p50"][account_index], 0.0)),
                "prediction_outflow_p90": float(max(outflow["p90"][account_index], 0.0)),
            }
            records.append(record)

        # Critical leakage rule: only the operational point prediction is
        # appended.  Actual test flows are never used as future inputs.
        histories["inflow"] = np.column_stack(
            [histories["inflow"], forecasts["inflow"]["mean"]]
        )
        histories["outflow"] = np.column_stack(
            [histories["outflow"], forecasts["outflow"]["mean"]]
        )

    return add_derived_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )


def save_run_outputs(
    data: DeepARData,
    predictions: pd.DataFrame,
    training_info: pd.DataFrame,
    models: dict[str, DeepARNetwork],
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    run_dir = output_dir / f"{data.frequency}_recursive"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    predictions.insert(0, "frequency", data.frequency)
    predictions.insert(1, "method", "recursive")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    scale_df = data.df.copy()
    scale_df["net_flow_for_scale"] = scale_df["inflow_amount"] - scale_df["outflow_amount"]
    metric_specs = [
        (
            "inflow", "actual_inflow", "prediction_inflow", "prediction_inflow_raw",
            "inflow_amount", args.over_weight, args.under_weight,
        ),
        (
            "outflow", "actual_outflow", "prediction_outflow", "prediction_outflow_raw",
            "outflow_amount", args.under_weight, args.over_weight,
        ),
        (
            "net_flow", "actual_net_flow", "prediction_net_flow", "prediction_net_flow_raw",
            "net_flow_for_scale", args.over_weight, args.under_weight,
        ),
    ]
    period_rows: list[pd.DataFrame] = []
    for target, actual_col, prediction_col, raw_col, scale_col, under_weight, over_weight in metric_specs:
        scale = train_normalization_scale(
            scale_df, data.test_start, value_column=scale_col
        )
        period_rows.append(
            calculate_metrics(
                predictions,
                scale,
                actual_column=actual_col,
                prediction_column=prediction_col,
                raw_prediction_column=raw_col,
                under_weight=under_weight,
                over_weight=over_weight,
            ).assign(
                target=target,
                frequency=data.frequency,
                method="recursive",
            )
        )
    metrics = pd.concat(period_rows, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    horizon_predictions = (
        predictions.groupby("account_id", as_index=False)[
            [
                "actual_inflow", "actual_outflow", "prediction_inflow",
                "prediction_inflow_raw", "prediction_outflow", "prediction_outflow_raw",
            ]
        ].sum()
    )
    horizon_predictions = add_derived_flow_columns(horizon_predictions)
    horizon_predictions["horizon_step"] = "horizon_total"
    horizon_rows: list[pd.DataFrame] = []
    for target, actual_col, prediction_col, raw_col, scale_col, under_weight, over_weight in metric_specs:
        scale = train_normalization_scale(
            scale_df, data.test_start, value_column=scale_col
        )
        horizon_rows.append(
            calculate_metrics(
                horizon_predictions,
                scale,
                actual_column=actual_col,
                prediction_column=prediction_col,
                raw_prediction_column=raw_col,
                under_weight=under_weight,
                over_weight=over_weight,
                scale_multiplier=data.horizon,
            ).assign(
                target=target,
                frequency=data.frequency,
                method="recursive",
                horizon_periods=data.horizon,
                under_weight=under_weight,
                over_weight=over_weight,
            )
        )
    horizon_metrics = pd.concat(horizon_rows, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[
        horizon_metrics["horizon_step"].eq("horizon_total")
    ]
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)
    training_info.assign(frequency=data.frequency, method="recursive").to_csv(
        run_dir / "training_info.csv", index=False
    )
    save_forecast_plot(
        predictions, data.frequency, "recursive", run_dir / "aggregate_test_forecast.png"
    )

    for flow_name, model in models.items():
        torch.save(
            {
                "flow": flow_name,
                "frequency": data.frequency,
                "model_config": model.config,
                "state_dict": model.state_dict(),
                "dynamic_feature_names": data.dynamic_feature_names,
                "static_feature_names": data.static_feature_names,
            },
            run_dir / f"{flow_name}_model.pt",
        )

    metadata = {
        "frequency": data.frequency,
        "method": "recursive",
        "horizon": data.horizon,
        "test_start": str(data.test_start.date()),
        "test_dates": [str(date.date()) for date in data.test_dates],
        "n_accounts": len(data.account_ids),
        "target_models": list(FLOW_TARGETS),
        "account_id_embedding": False,
        "target_transform": "log1p(flow / account_scale)",
        "account_scale": "pre-test mean absolute flow per account, minimum 1",
        "dynamic_feature_names": data.dynamic_feature_names,
        "static_feature_names": data.static_feature_names,
        "device": str(device),
        "torch_version": torch.__version__,
        "config": {
            "context_length": args.context_length,
            "hidden_size": args.hidden_size,
            "rnn_layers": args.rnn_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "validation_length": args.validation_length,
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"completed {data.frequency}/recursive: "
        f"{len(data.account_ids):,} accounts, horizon={data.horizon}, device={device}"
    )
    return horizon_metrics


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument(
        "--method", choices=["recursive"], default="recursive",
        help="Only native DeepAR recursive forecasting is implemented.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--validation-length", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_accounts is not None and args.max_accounts < 1:
        raise ValueError("--max-accounts must be at least 1")
    if args.context_length is not None and args.context_length < 1:
        raise ValueError("--context-length must be at least 1")
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("--batch-size and --epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.under_weight <= 0 or args.over_weight <= 0:
        raise ValueError("under/over weights must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]

    horizon_rows: list[pd.DataFrame] = []
    for frequency in frequencies:
        account_ids = args.account_ids
        if account_ids is None and args.max_accounts is not None:
            account_ids = get_available_account_ids(frequency, args.max_accounts)
        data = create_deepar_data(frequency, account_ids)
        models: dict[str, DeepARNetwork] = {}
        training_rows: list[dict[str, object]] = []
        for flow_name in FLOW_TARGETS:
            model, info = train_flow_model(data, flow_name, device, args)
            models[flow_name] = model
            training_rows.append(info)
        training_info = pd.DataFrame(training_rows)
        predictions = recursive_forecast(data, models, training_info, device, args)
        horizon_rows.append(
            save_run_outputs(
                data, predictions, training_info, models,
                args.output_dir, device, args,
            )
        )

    if horizon_rows:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(horizon_rows, ignore_index=True).to_csv(
            args.output_dir / "horizon_comparison_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
