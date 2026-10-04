"""Shared weekly multi-horizon runners for TFT, iTransformer, and TimesNet.

The three public scripts in this project intentionally share the same data
split, target construction, recursive-history policy, and output schema.  The
architectures differ, but all of them predict the full weekly test horizon in
one forward pass from the last pre-test origin.

Two target modes are supported:

``absolute``
    Predict absolute inflow and outflow.

``residual``
    Predict inflow and outflow residuals after removing the deterministic loan
    repayment schedule.  The schedule is added back before metrics are saved.

Only ``loan.payments`` is used as a fixed-cost baseline.  The schedule and the
calendar/holiday variables are available for the whole future horizon, while
test actual flows are never supplied to the models.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from deepar_forecast import set_seed
from deepar_forecast_enriched import EnrichedDeepARData, create_enriched_deepar_data
from enriched_forecast_common import enriched_master_path, get_available_enriched_account_ids
from fixed_cost_schedule import fixed_cost_policy
from linear_regression_forecast import (
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    add_derived_flow_columns,
    calculate_metrics,
    save_forecast_plot,
    train_normalization_scale,
)


TargetMode = Literal["absolute", "residual"]
ModelKind = Literal["tft", "itransformer", "timesnet"]


def resolve_device(requested: str, strict_cuda: bool = False) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        if strict_cuda:
            raise RuntimeError("CUDA was requested, but no GPU is available")
        print("[warning] CUDA unavailable; using CPU")
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass
class SequenceData:
    base: EnrichedDeepARData
    target_mode: TargetMode
    schedule: np.ndarray
    target_values: dict[str, np.ndarray]
    target_scales: dict[str, np.ndarray]

    @property
    def frequency(self) -> str:
        return self.base.frequency

    @property
    def horizon(self) -> int:
        return self.base.horizon

    @property
    def df(self) -> pd.DataFrame:
        return self.base.df

    @property
    def account_ids(self) -> list[int]:
        return self.base.account_ids

    @property
    def dates(self) -> list[pd.Timestamp]:
        return self.base.dates

    @property
    def test_dates(self) -> list[pd.Timestamp]:
        return self.base.test_dates

    @property
    def test_start(self) -> pd.Timestamp:
        return self.base.test_start

    @property
    def test_start_index(self) -> int:
        return self.base.test_start_index

    @property
    def dynamic_features(self) -> np.ndarray:
        return self.base.dynamic_features

    @property
    def dynamic_feature_names(self) -> list[str]:
        return self.base.dynamic_feature_names

    @property
    def static_features(self) -> np.ndarray:
        return self.base.static_features

    @property
    def static_feature_names(self) -> list[str]:
        return self.base.static_feature_names


def create_sequence_data(
    frequency: str,
    target_mode: TargetMode,
    account_ids: list[int] | None,
) -> SequenceData:
    if frequency != "weekly":
        raise ValueError("TFT/iTransformer/TimesNet runners currently support weekly only")
    base = create_enriched_deepar_data(frequency, account_ids)
    schedule_index = base.dynamic_feature_names.index(
        "scheduled_loan_repayment_scaled"
    )
    schedule = (
        base.dynamic_features[:, :, schedule_index]
        * float(base.schedule_dynamic_scale)
    ).astype(np.float32)

    absolute = {
        name: base.values_by_flow[name].astype(np.float32)
        for name in FLOW_TARGETS
    }
    if target_mode == "absolute":
        target_values = absolute
    else:
        target_values = {
            "inflow": absolute["inflow"].copy(),
            "outflow": (absolute["outflow"] - schedule).astype(np.float32),
        }

    target_scales: dict[str, np.ndarray] = {}
    for flow_name, values in target_values.items():
        scale = np.mean(np.abs(values[:, : base.test_start_index]), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        target_scales[flow_name] = np.maximum(scale, 1.0).astype(np.float32)

    return SequenceData(
        base=base,
        target_mode=target_mode,
        schedule=schedule,
        target_values=target_values,
        target_scales=target_scales,
    )


def signed_log1p(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(torch.abs(values) / scale)


def signed_expm1(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.expm1(torch.clamp(torch.abs(values), 0.0, 20.0))


class MultiHorizonDataset(Dataset):
    """One sample is one account and one forecast origin."""

    def __init__(
        self,
        values: np.ndarray,
        scales: np.ndarray,
        dynamic_features: np.ndarray,
        static_features: np.ndarray,
        indices: np.ndarray,
        context_length: int,
        horizon: int,
    ) -> None:
        self.values = torch.from_numpy(values.astype(np.float32, copy=False))
        self.scales = torch.from_numpy(scales.astype(np.float32, copy=False))
        self.dynamic_features = torch.from_numpy(
            dynamic_features.astype(np.float32, copy=False)
        )
        self.static_features = torch.from_numpy(
            static_features.astype(np.float32, copy=False)
        )
        self.indices = indices
        self.context_length = int(context_length)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        account_index, origin = self.indices[index]
        origin = int(origin)
        start = origin - self.context_length
        end = origin + self.horizon
        scale = self.scales[account_index]
        context_raw = self.values[account_index, start:origin]
        target_raw = self.values[account_index, origin:end]
        context_target = signed_log1p(context_raw, scale)
        target = signed_log1p(target_raw, scale)
        return (
            context_target,
            self.dynamic_features[account_index, start:origin],
            self.dynamic_features[account_index, origin:end],
            self.static_features[account_index],
            target,
            scale,
        )


def window_indices(
    n_accounts: int,
    context_length: int,
    horizon: int,
    origin_start: int,
    origin_end: int,
) -> np.ndarray:
    origins = np.arange(
        max(context_length, origin_start),
        max(context_length, origin_end),
        dtype=np.int64,
    )
    if len(origins) == 0:
        return np.empty((0, 2), dtype=np.int64)
    accounts = np.arange(n_accounts, dtype=np.int64)
    return np.stack(
        np.meshgrid(accounts, origins, indexing="ij"), axis=-1
    ).reshape(-1, 2)


class VariableSelection(nn.Module):
    """Small variable-selection block used by the TFT implementation."""

    def __init__(self, n_variables: int, hidden_size: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(n_variables)]
        )
        self.selector = nn.Linear(n_variables, n_variables)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [batch, time, variables]
        weights = torch.softmax(self.selector(values), dim=-1)
        embedded = torch.stack(
            [layer(values[..., index : index + 1]) for index, layer in enumerate(self.embeddings)],
            dim=-2,
        )
        return self.norm((weights.unsqueeze(-1) * embedded).sum(dim=-2))


class TFTForecaster(nn.Module):
    """Compact Temporal Fusion Transformer-style multi-horizon model."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        horizon: int,
        hidden_size: int,
        rnn_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.config = {
            "architecture": "tft",
            "dynamic_dim": dynamic_dim,
            "static_dim": static_dim,
            "horizon": horizon,
            "hidden_size": hidden_size,
            "rnn_layers": rnn_layers,
            "dropout": dropout,
        }
        self.past_selection = VariableSelection(1 + dynamic_dim, hidden_size)
        self.future_selection = VariableSelection(dynamic_dim, hidden_size)
        self.static_projection = nn.Linear(static_dim, hidden_size)
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        heads = max(1, min(4, hidden_size))
        while hidden_size % heads:
            heads -= 1
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.future_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, context, context_dynamic, future_dynamic, static):
        static_context = self.static_projection(static)
        past_values = torch.cat([context.unsqueeze(-1), context_dynamic], dim=-1)
        past = self.past_selection(past_values) + static_context.unsqueeze(1)
        encoded, _ = self.encoder(past)
        future = self.future_selection(future_dynamic) + static_context.unsqueeze(1)
        attended, _ = self.attention(future, encoded, encoded, need_weights=False)
        gate = torch.sigmoid(self.future_gate(torch.cat([future, attended], dim=-1)))
        fused = gate * attended + (1.0 - gate) * future
        return self.output(fused).squeeze(-1)


class ITransformerForecaster(nn.Module):
    """Inverted-transformer-style model with variables as transformer tokens."""

    def __init__(
        self,
        context_length: int,
        dynamic_dim: int,
        static_dim: int,
        horizon: int,
        hidden_size: int,
        rnn_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del rnn_layers
        self.config = {
            "architecture": "itransformer",
            "context_length": context_length,
            "dynamic_dim": dynamic_dim,
            "static_dim": static_dim,
            "horizon": horizon,
            "hidden_size": hidden_size,
            "dropout": dropout,
        }
        channels = 1 + dynamic_dim
        self.channel_embedding = nn.Linear(context_length, hidden_size)
        self.channel_id = nn.Parameter(torch.randn(channels, hidden_size) * 0.02)
        heads = max(1, min(4, hidden_size))
        while hidden_size % heads:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.static_projection = nn.Linear(static_dim, hidden_size)
        self.future_projection = nn.Linear(dynamic_dim, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, context, context_dynamic, future_dynamic, static):
        channels = torch.cat(
            [context.unsqueeze(-1), context_dynamic], dim=-1
        ).transpose(1, 2)
        tokens = self.channel_embedding(channels) + self.channel_id.unsqueeze(0)
        tokens = self.encoder(tokens)
        summary = tokens.mean(dim=1) + self.static_projection(static)
        future = self.future_projection(future_dynamic)
        summary = summary.unsqueeze(1).expand(-1, future.size(1), -1)
        return self.head(torch.cat([summary, future], dim=-1)).squeeze(-1)


class TimesBlock(nn.Module):
    """A compact FFT-period convolution block inspired by TimesNet."""

    def __init__(self, hidden_size: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, hidden]
        length = x.size(1)
        spectrum = torch.fft.rfft(x.mean(dim=-1), dim=1)
        power = spectrum.abs()
        if power.size(1) <= 2:
            return x
        power[:, :2] = 0.0
        k = min(self.top_k, max(1, power.size(1) - 1))
        weights, indices = torch.topk(power, k=k, dim=1)
        outputs = []
        for period_index in range(k):
            period = int(max(2, round(length / int(indices[0, period_index].item()))))
            padded_length = int(math.ceil(length / period) * period)
            pad = padded_length - length
            padded = F.pad(x, (0, 0, 0, pad))
            image = padded.reshape(x.size(0), padded_length // period, period, -1)
            image = image.permute(0, 3, 1, 2)
            mixed = self.conv(image)
            mixed = mixed.permute(0, 2, 3, 1).reshape(x.size(0), padded_length, -1)
            outputs.append(mixed[:, :length])
        stacked = torch.stack(outputs, dim=-1)
        normalized_weights = torch.softmax(weights, dim=1).unsqueeze(1).unsqueeze(2)
        return x + (stacked * normalized_weights).sum(dim=-1)


class TimesNetForecaster(nn.Module):
    """TimesNet-style periodic mixing plus known-future covariates."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        horizon: int,
        hidden_size: int,
        rnn_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del rnn_layers
        self.config = {
            "architecture": "timesnet",
            "dynamic_dim": dynamic_dim,
            "static_dim": static_dim,
            "horizon": horizon,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "top_k_periods": 3,
        }
        self.input_projection = nn.Linear(1 + dynamic_dim, hidden_size)
        self.period_block = TimesBlock(hidden_size, top_k=3, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.static_projection = nn.Linear(static_dim, hidden_size)
        self.future_projection = nn.Linear(dynamic_dim, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, context, context_dynamic, future_dynamic, static):
        inputs = torch.cat([context.unsqueeze(-1), context_dynamic], dim=-1)
        encoded = self.norm(self.period_block(self.input_projection(inputs)))
        pooled = encoded.mean(dim=1) + self.static_projection(static)
        last = encoded[:, -1]
        future = self.future_projection(future_dynamic)
        pooled = pooled.unsqueeze(1).expand(-1, future.size(1), -1)
        last = last.unsqueeze(1).expand(-1, future.size(1), -1)
        return self.head(torch.cat([pooled, last, future], dim=-1)).squeeze(-1)


def build_model(
    kind: ModelKind,
    data: SequenceData,
    context_length: int,
    hidden_size: int,
    rnn_layers: int,
    dropout: float,
) -> nn.Module:
    common = dict(
        dynamic_dim=data.dynamic_features.shape[2],
        static_dim=data.static_features.shape[1],
        horizon=data.horizon,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
    )
    if kind == "tft":
        return TFTForecaster(**common)
    if kind == "itransformer":
        return ITransformerForecaster(context_length=context_length, **common)
    if kind == "timesnet":
        return TimesNetForecaster(**common)
    raise ValueError(f"Unknown model kind: {kind}")


def move_batch(batch, device: torch.device):
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


def raw_prediction(transformed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return signed_expm1(transformed) * scale.unsqueeze(-1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
) -> dict[str, float]:
    if loader is None:
        return {"loss": float("nan"), "rmse": float("nan")}
    model.eval()
    total_loss = 0.0
    total_squared_error = 0.0
    total_count = 0
    for batch in loader:
        context, context_dynamic, future_dynamic, static, target, scale = move_batch(
            batch, device
        )
        prediction = model(context, context_dynamic, future_dynamic, static)
        loss = F.mse_loss(prediction, target)
        predicted_raw = raw_prediction(prediction, scale)
        target_raw = raw_prediction(target, scale)
        total_loss += float(loss.item()) * target.numel()
        total_squared_error += float((predicted_raw - target_raw).square().sum().item())
        total_count += target.numel()
    return {
        "loss": total_loss / total_count if total_count else float("nan"),
        "rmse": math.sqrt(total_squared_error / total_count)
        if total_count
        else float("nan"),
    }


def fit_model(
    kind: ModelKind,
    data: SequenceData,
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    args,
    device: torch.device,
    epochs: int,
    seed: int,
    flow_name: str,
    early_stopping: bool,
) -> tuple[nn.Module, dict[str, object]]:
    set_seed(seed)
    model = build_model(
        kind,
        data,
        args.context_length_effective,
        args.hidden_size,
        args.rnn_layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    validation_loader = (
        make_loader(validation_dataset, args.batch_size, False, args.num_workers, device)
        if validation_dataset is not None and len(validation_dataset)
        else None
    )
    best_state = None
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            context, context_dynamic, future_dynamic, static, target, scale = move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(context, context_dynamic, future_dynamic, static)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            total_count += target.numel()
        train_loss = total_loss / total_count if total_count else float("nan")
        validation = evaluate_model(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_rmse": validation["rmse"],
            }
        )
        if validation_loader is not None:
            if validation["rmse"] < best_rmse:
                best_rmse = validation["rmse"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if early_stopping and stale >= args.patience:
                    break
        if epoch == 1 or epoch == epochs or epoch % max(args.log_every, 1) == 0:
            message = (
                f"[{kind}/{flow_name}] epoch={epoch}/{epochs}, "
                f"train_loss={train_loss:.6f}"
            )
            if validation_loader is not None:
                message += f", val_rmse={validation['rmse']:.4f}"
            print(message)
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "flow": flow_name,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset)
        if validation_dataset is not None
        else 0,
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch or epochs,
        "best_validation_rmse": best_rmse
        if validation_loader is not None
        else np.nan,
        "last_train_loss": history[-1]["train_loss"],
        "history": history,
    }
    return model, info


def make_datasets(
    data: SequenceData,
    flow_name: str,
    context_length: int,
    validation_length: int,
) -> tuple[Dataset, Dataset, Dataset, int]:
    validation_length = min(
        validation_length,
        data.test_start_index - context_length - data.horizon + 1,
    )
    if validation_length < 1:
        raise ValueError("No room for a validation window; reduce context length")
    validation_start = data.test_start_index - validation_length
    values = data.target_values[flow_name]
    scales = data.target_scales[flow_name]
    n_accounts = len(data.account_ids)
    train_indices = window_indices(
        n_accounts,
        context_length,
        data.horizon,
        context_length,
        validation_start - data.horizon + 1,
    )
    validation_indices = window_indices(
        n_accounts,
        context_length,
        data.horizon,
        validation_start,
        data.test_start_index - data.horizon + 1,
    )
    final_indices = window_indices(
        n_accounts,
        context_length,
        data.horizon,
        context_length,
        data.test_start_index - data.horizon + 1,
    )
    datasets = tuple(
        MultiHorizonDataset(
            values,
            scales,
            data.dynamic_features,
            data.static_features,
            indices,
            context_length,
            data.horizon,
        )
        for indices in (train_indices, validation_indices, final_indices)
    )
    if not len(datasets[0]):
        raise ValueError(f"No training windows for {flow_name}/{data.frequency}")
    return datasets[0], datasets[1], datasets[2], validation_start


def predict_horizon(
    model: nn.Module,
    data: SequenceData,
    history_values: np.ndarray,
    scale: np.ndarray,
    context_length: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    start = data.test_start_index - context_length
    context = signed_log1p(
        torch.from_numpy(history_values[:, start : data.test_start_index]),
        torch.from_numpy(scale[:, None]),
    ).numpy().astype(np.float32)
    context_dynamic = data.dynamic_features[:, start : data.test_start_index]
    future_dynamic = data.dynamic_features[
        :, data.test_start_index : data.test_start_index + data.horizon
    ]
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, len(data.account_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(data.account_ids))
            output = model(
                torch.from_numpy(context[batch_start:batch_end]).to(device),
                torch.from_numpy(context_dynamic[batch_start:batch_end]).to(device),
                torch.from_numpy(future_dynamic[batch_start:batch_end]).to(device),
                torch.from_numpy(data.static_features[batch_start:batch_end]).to(device),
            )
            predictions.append(
                raw_prediction(
                    output,
                    torch.from_numpy(scale[batch_start:batch_end]).to(device),
                ).cpu().numpy()
            )
    return np.concatenate(predictions, axis=0).astype(float)


def add_residual_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    result = add_derived_flow_columns(predictions)
    result["actual_net_flow_residual"] = (
        result["actual_inflow_residual"] - result["actual_outflow_residual"]
    )
    result["prediction_net_flow_residual_raw"] = (
        result["prediction_inflow_residual_raw"]
        - result["prediction_outflow_residual_raw"]
    )
    result["prediction_net_flow_residual"] = (
        result["prediction_inflow_residual"]
        - result["prediction_outflow_residual"]
    )
    return result


def forecast(
    data: SequenceData,
    models: dict[str, nn.Module],
    device: torch.device,
    args,
) -> pd.DataFrame:
    absolute = data.base.values_by_flow
    predictions = {
        flow_name: predict_horizon(
            models[flow_name],
            data,
            data.target_values[flow_name][:, : data.test_start_index],
            data.target_scales[flow_name],
            args.context_length_effective,
            device,
            args.batch_size,
        )
        for flow_name in FLOW_TARGETS
    }
    fixed_outflow = data.schedule[:, data.test_start_index : data.test_start_index + data.horizon]
    origin_period = data.test_dates[0] - FREQUENCY_CONFIG[data.frequency]["step"]
    records: list[dict[str, object]] = []
    for account_index, account_id in enumerate(data.account_ids):
        for step, target_date in enumerate(data.test_dates, start=1):
            horizon_index = step - 1
            if data.target_mode == "residual":
                raw_inflow_residual = float(predictions["inflow"][account_index, horizon_index])
                raw_outflow_residual = float(predictions["outflow"][account_index, horizon_index])
                raw_inflow = raw_inflow_residual
                raw_outflow = float(fixed_outflow[account_index, horizon_index]) + raw_outflow_residual
            else:
                raw_inflow = float(predictions["inflow"][account_index, horizon_index])
                raw_outflow = float(predictions["outflow"][account_index, horizon_index])
                raw_inflow_residual = raw_inflow
                raw_outflow_residual = raw_outflow - float(fixed_outflow[account_index, horizon_index])
            prediction_inflow = max(raw_inflow, 0.0)
            prediction_outflow = max(raw_outflow, 0.0)
            fixed_value = float(fixed_outflow[account_index, horizon_index])
            actual_inflow = float(absolute["inflow"][account_index, data.test_start_index + horizon_index])
            actual_outflow = float(absolute["outflow"][account_index, data.test_start_index + horizon_index])
            records.append(
                {
                    "account_id": int(account_id),
                    "period_start": target_date,
                    "origin_period": origin_period,
                    "horizon_step": step,
                    "target_mode": data.target_mode,
                    "fixed_cost_inflow": 0.0,
                    "fixed_cost_outflow": fixed_value,
                    "fixed_cost_net_flow": -fixed_value,
                    "actual_inflow": actual_inflow,
                    "actual_outflow": actual_outflow,
                    "actual_inflow_residual": actual_inflow,
                    "actual_outflow_residual": actual_outflow - fixed_value,
                    "prediction_inflow_raw": raw_inflow,
                    "prediction_outflow_raw": raw_outflow,
                    "prediction_inflow": prediction_inflow,
                    "prediction_outflow": prediction_outflow,
                    "prediction_inflow_residual_raw": raw_inflow_residual,
                    "prediction_outflow_residual_raw": raw_outflow_residual,
                    "prediction_inflow_residual": prediction_inflow,
                    "prediction_outflow_residual": prediction_outflow - fixed_value,
                }
            )
    return add_residual_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )


def metric_scales(data: SequenceData) -> dict[str, dict[str, pd.Series]]:
    scale_df = data.df.copy()
    scale_df["net_flow_for_scale"] = scale_df["inflow_amount"] - scale_df["outflow_amount"]
    absolute = {
        "inflow": train_normalization_scale(scale_df, data.test_start, "inflow_amount"),
        "outflow": train_normalization_scale(scale_df, data.test_start, "outflow_amount"),
        "net_flow": train_normalization_scale(scale_df, data.test_start, "net_flow_for_scale"),
    }
    ids = pd.Index(data.account_ids, name="account_id")
    inflow = data.target_values["inflow"][:, : data.test_start_index]
    outflow = data.target_values["outflow"][:, : data.test_start_index]
    if data.target_mode == "absolute":
        residual_inflow = inflow
        residual_outflow = outflow - data.schedule[:, : data.test_start_index]
    else:
        residual_inflow = inflow
        residual_outflow = outflow
    residual = {}
    for name, values in {
        "inflow": residual_inflow,
        "outflow": residual_outflow,
        "net_flow": residual_inflow - residual_outflow,
    }.items():
        values_scale = np.mean(np.abs(values), axis=1)
        values_scale = np.where(np.isfinite(values_scale) & (values_scale > 0), values_scale, 1.0)
        residual[name] = pd.Series(np.maximum(values_scale, 1.0), index=ids)
    return {"absolute": absolute, "residual": residual}


def _metric_specs(residual: bool):
    if residual:
        return [
            ("inflow", "actual_inflow_residual", "prediction_inflow_residual", "prediction_inflow_residual_raw", "inflow"),
            ("outflow", "actual_outflow_residual", "prediction_outflow_residual", "prediction_outflow_residual_raw", "outflow"),
            ("net_flow", "actual_net_flow_residual", "prediction_net_flow_residual", "prediction_net_flow_residual_raw", "net_flow"),
        ]
    return [
        ("inflow", "actual_inflow", "prediction_inflow", "prediction_inflow_raw", "inflow"),
        ("outflow", "actual_outflow", "prediction_outflow", "prediction_outflow_raw", "outflow"),
        ("net_flow", "actual_net_flow", "prediction_net_flow", "prediction_net_flow_raw", "net_flow"),
    ]


def _weights(target: str, under_weight: float, over_weight: float):
    if target in {"inflow", "net_flow"}:
        return over_weight, under_weight
    return under_weight, over_weight


def save_outputs(
    data: SequenceData,
    predictions: pd.DataFrame,
    training_info: pd.DataFrame,
    output_dir: Path,
    model_kind: ModelKind,
    under_weight: float,
    over_weight: float,
) -> tuple[Path, pd.DataFrame]:
    run_dir = output_dir / f"{data.frequency}_direct_{data.target_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = predictions.copy()
    output.insert(0, "model", model_kind)
    output.insert(1, "frequency", data.frequency)
    output.insert(2, "method", "direct")
    output.to_csv(run_dir / "predictions.csv", index=False)
    output.to_parquet(run_dir / "predictions.parquet", index=False)

    scales = metric_scales(data)
    horizon_columns = [
        "actual_inflow", "actual_outflow", "prediction_inflow", "prediction_inflow_raw",
        "prediction_outflow", "prediction_outflow_raw", "fixed_cost_inflow",
        "fixed_cost_outflow", "fixed_cost_net_flow", "actual_inflow_residual",
        "actual_outflow_residual", "prediction_inflow_residual",
        "prediction_inflow_residual_raw", "prediction_outflow_residual",
        "prediction_outflow_residual_raw",
    ]
    horizon_predictions = output.groupby("account_id", as_index=False)[horizon_columns].sum()
    horizon_predictions = add_residual_columns(horizon_predictions)
    horizon_predictions["horizon_step"] = "horizon_total"

    period_frames = []
    horizon_frames = []
    for evaluation_target in ("absolute", "residual"):
        metric_rows = []
        for target, actual_col, prediction_col, raw_col, scale_name in _metric_specs(evaluation_target == "residual"):
            target_under, target_over = _weights(target, under_weight, over_weight)
            metric_rows.append(
                calculate_metrics(
                    output,
                    scales[evaluation_target][scale_name],
                    actual_column=actual_col,
                    prediction_column=prediction_col,
                    raw_prediction_column=raw_col,
                    under_weight=target_under,
                    over_weight=target_over,
                ).assign(
                    model=model_kind,
                    frequency=data.frequency,
                    method="direct",
                    target=target,
                    evaluation_target=evaluation_target,
                    target_mode=data.target_mode,
                )
            )
            horizon_frames.append(
                calculate_metrics(
                    horizon_predictions,
                    scales[evaluation_target][scale_name],
                    actual_column=actual_col,
                    prediction_column=prediction_col,
                    raw_prediction_column=raw_col,
                    under_weight=target_under,
                    over_weight=target_over,
                    scale_multiplier=data.horizon,
                ).assign(
                    model=model_kind,
                    frequency=data.frequency,
                    method="direct",
                    target=target,
                    evaluation_target=evaluation_target,
                    target_mode=data.target_mode,
                    horizon_periods=data.horizon,
                )
            )
        period_frames.append(pd.concat(metric_rows, ignore_index=True))
    period_metrics = pd.concat(period_frames, ignore_index=True)
    horizon_metrics = pd.concat(horizon_frames, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[horizon_metrics["horizon_step"].eq("horizon_total")]
    period_metrics.to_csv(run_dir / "metrics.csv", index=False)
    period_metrics.loc[period_metrics["evaluation_target"].eq("residual")].to_csv(
        run_dir / "residual_metrics.csv", index=False
    )
    horizon_metrics.loc[horizon_metrics["evaluation_target"].eq("absolute")].to_csv(
        run_dir / "horizon_metrics.csv", index=False
    )
    horizon_metrics.loc[horizon_metrics["evaluation_target"].eq("residual")].to_csv(
        run_dir / "residual_horizon_metrics.csv", index=False
    )
    training_info.assign(
        model=model_kind,
        frequency=data.frequency,
        method="direct",
        target_mode=data.target_mode,
    ).to_csv(run_dir / "training_info.csv", index=False)
    save_forecast_plot(
        output,
        data.frequency,
        f"{model_kind}_direct_{data.target_mode}",
        run_dir / "aggregate_test_forecast.png",
    )
    metadata = {
        "model": model_kind,
        "frequency": data.frequency,
        "method": "direct_multi_horizon",
        "target_mode": data.target_mode,
        "master_file": str(enriched_master_path(data.frequency)),
        "fixed_cost_policy": fixed_cost_policy(),
        "future_known_features": data.dynamic_feature_names,
        "static_features": data.static_feature_names,
        "account_id_embedding": False,
        "test_actuals_used_as_inputs": False,
        "prediction_definition": (
            "one model forward pass for the full weekly horizon; "
            "net_flow is derived from inflow and outflow"
        ),
        "residual_transform": "sign(x) * log1p(abs(x) / account_scale)",
        "n_accounts": len(data.account_ids),
        "test_start": str(data.test_start.date()),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return run_dir, horizon_metrics.loc[
        horizon_metrics["evaluation_target"].eq("absolute")
    ].copy()


def run_experiment(
    kind: ModelKind,
    frequency: str,
    target_modes: list[TargetMode],
    args,
) -> None:
    if frequency != "weekly":
        raise ValueError("These sequence-model scripts currently support --frequency weekly")
    device = resolve_device(args.device, args.strict_cuda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    account_ids = args.account_ids
    if account_ids is None and args.max_accounts is not None:
        account_ids = get_available_enriched_account_ids("weekly", args.max_accounts)
    combined_horizon_rows = []

    for target_mode in target_modes:
        data = create_sequence_data("weekly", target_mode, account_ids)
        args.context_length_effective = args.context_length or 52
        if args.context_length_effective >= data.test_start_index - data.horizon:
            raise ValueError("context length leaves no training windows")
        train_models: dict[str, nn.Module] = {}
        training_rows = []
        for flow_name in FLOW_TARGETS:
            train_dataset, validation_dataset, final_dataset, validation_start = make_datasets(
                data,
                flow_name,
                args.context_length_effective,
                args.validation_length or data.horizon,
            )
            validation_model, validation_info = fit_model(
                kind,
                data,
                train_dataset,
                validation_dataset,
                args,
                device,
                args.epochs,
                args.seed + (0 if flow_name == "inflow" else 1),
                flow_name,
                True,
            )
            final_model, final_info = fit_model(
                kind,
                data,
                final_dataset,
                None,
                args,
                device,
                int(validation_info["best_epoch"]),
                args.seed + 100 + (0 if flow_name == "inflow" else 1),
                flow_name,
                False,
            )
            train_models[flow_name] = final_model
            training_rows.append(
                {
                    "flow": flow_name,
                    "target_mode": target_mode,
                    "context_length": args.context_length_effective,
                    "validation_start": str(data.dates[validation_start].date()),
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "final_train_samples": len(final_dataset),
                    "best_epoch": validation_info["best_epoch"],
                    "validation_rmse": validation_info["best_validation_rmse"],
                    "final_epochs": final_info["epochs_ran"],
                    "hidden_size": args.hidden_size,
                    "rnn_layers": args.rnn_layers,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "device": str(device),
                }
            )
        training_info = pd.DataFrame(training_rows)
        predictions = forecast(data, train_models, device, args)
        run_dir, horizon_metrics = save_outputs(
            data,
            predictions,
            training_info,
            args.output_dir,
            kind,
            args.under_weight,
            args.over_weight,
        )
        for flow_name, model in train_models.items():
            torch.save(
                {
                    "model_config": model.config,
                    "state_dict": model.state_dict(),
                    "model": kind,
                    "frequency": "weekly",
                    "target_mode": target_mode,
                    "dynamic_feature_names": data.dynamic_feature_names,
                    "static_feature_names": data.static_feature_names,
                },
                run_dir / f"{flow_name}_model.pt",
            )
        combined_horizon_rows.append(horizon_metrics)
        print(
            f"completed {kind} weekly/direct/{target_mode}: "
            f"{len(data.account_ids):,} accounts, device={device}"
        )

    if combined_horizon_rows:
        pd.concat(combined_horizon_rows, ignore_index=True).to_csv(
            args.output_dir / "horizon_comparison_metrics.csv", index=False
        )


def add_common_arguments(parser) -> None:
    parser.add_argument("--frequency", choices=["weekly"], default="weekly")
    parser.add_argument("--target-mode", choices=["absolute", "residual", "both"], default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--context-length", type=int, default=52)
    parser.add_argument("--validation-length", type=int, default=27)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
