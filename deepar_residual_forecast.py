"""DeepAR recursive forecasts with a loan-schedule baseline and residual targets.

This entry point is the DeepAR counterpart of the residual tree/LR runners.
Only ``loan.payments`` is treated as a deterministic future fixed cost.  The
model learns residual flows:

    inflow residual  = inflow_amount - 0
    outflow residual = outflow_amount - scheduled_loan_repayment

The outflow residual can be negative, so this file uses a signed log transform
instead of the non-negative log1p transform used by the original DeepAR
implementation.  During forecasting, the predicted residual is added back to
the known loan schedule and the resulting absolute flow is clipped at zero.

DeepAR remains recursive only.  At each test period the same shared LSTM for
each flow predicts one step, and only the final absolute prediction is added
to the next period's autoregressive history.  Test actuals are never used as
future inputs.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from deepar_forecast import (
    DeepARNetwork,
    gaussian_nll,
    make_loader,
    make_window_indices,
    move_batch,
    resolve_device as resolve_base_device,
    set_seed,
)
from deepar_forecast_enriched import (
    EnrichedDeepARData,
    create_enriched_deepar_data,
)
from enriched_forecast_common import enriched_master_path, get_available_enriched_account_ids
from fixed_cost_schedule import fixed_cost_policy
from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    add_derived_flow_columns,
    calculate_metrics,
    save_forecast_plot,
    train_normalization_scale,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepar_residual"

TUNING_KEYS = (
    "hidden_size",
    "rnn_layers",
    "dropout",
    "learning_rate",
    "batch_size",
)
DEFAULT_TUNING_GRID: dict[str, list[float | int]] = {
    "hidden_size": [16, 32, 64],
    "rnn_layers": [1, 2],
    "dropout": [0.1, 0.2],
    "learning_rate": [0.001, 0.0003],
    "batch_size": [64, 128],
}


def resolve_device(requested: str, strict_cuda: bool = False) -> torch.device:
    """Resolve CUDA when available and otherwise fall back to CPU."""
    if requested == "cuda" and not torch.cuda.is_available():
        if strict_cuda:
            raise RuntimeError("CUDA was requested, but no GPU is available")
        print("[warning] CUDA unavailable; using CPU")
        return torch.device("cpu")
    return resolve_base_device(requested)


@dataclass
class ResidualDeepARData:
    """Enriched DeepAR data with residual-valued flow histories."""

    base: EnrichedDeepARData
    schedule: np.ndarray
    values_by_flow: dict[str, np.ndarray]
    scales_by_flow: dict[str, np.ndarray]

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

    @property
    def test_known_features(self) -> dict[pd.Timestamp, pd.DataFrame]:
        return self.base.test_known_features

    @property
    def schedule_dynamic_scale(self) -> float:
        return self.base.schedule_dynamic_scale


def create_residual_deepar_data(
    frequency: str,
    account_ids: list[int] | None,
) -> ResidualDeepARData:
    """Reuse enriched covariates and replace absolute targets with residuals."""
    base = create_enriched_deepar_data(frequency, account_ids)

    # The final dynamic feature is the leakage-safe loan schedule divided by
    # the same global scale used by enriched DeepAR.  Reconstructing it keeps
    # the baseline exactly aligned with the covariate seen by the model.
    schedule_index = base.dynamic_feature_names.index(
        "scheduled_loan_repayment_scaled"
    )
    schedule = (
        base.dynamic_features[:, :, schedule_index]
        * float(base.schedule_dynamic_scale)
    ).astype(np.float32)

    absolute_inflow = base.values_by_flow["inflow"].astype(np.float32)
    absolute_outflow = base.values_by_flow["outflow"].astype(np.float32)
    values_by_flow = {
        "inflow": absolute_inflow.copy(),
        "outflow": (absolute_outflow - schedule).astype(np.float32),
    }
    scales_by_flow: dict[str, np.ndarray] = {}
    for flow_name, values in values_by_flow.items():
        scale = np.mean(np.abs(values[:, : base.test_start_index]), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        scales_by_flow[flow_name] = np.maximum(scale, 1.0).astype(np.float32)

    return ResidualDeepARData(
        base=base,
        schedule=schedule,
        values_by_flow=values_by_flow,
        scales_by_flow=scales_by_flow,
    )


def signed_log1p(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map signed residuals to a stable real-valued training space."""
    return torch.sign(values) * torch.log1p(torch.abs(values) / scale)


def signed_expm1(values: torch.Tensor) -> torch.Tensor:
    """Invert ``signed_log1p`` without assuming non-negative targets."""
    return torch.sign(values) * torch.expm1(torch.clamp(torch.abs(values), 0.0, 20.0))


class ResidualDeepARWindowDataset(Dataset):
    """Window dataset using signed-log residual targets."""

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
        self.dynamic_features = torch.from_numpy(
            dynamic_features.astype(np.float32, copy=False)
        )
        self.static_features = torch.from_numpy(
            static_features.astype(np.float32, copy=False)
        )
        self.indices = indices
        self.context_length = int(context_length)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        account_index, target_index = self.indices[index]
        start = int(target_index) - self.context_length
        scale = self.scales[account_index]
        context_raw = self.values[account_index, start:target_index]
        context_target = signed_log1p(context_raw, scale)
        target_raw = self.values[account_index, target_index]
        target = signed_log1p(target_raw, scale)
        return (
            context_target,
            self.dynamic_features[account_index, start:target_index],
            self.dynamic_features[account_index, target_index],
            self.static_features[account_index],
            target,
        )


def evaluate_nll(model, loader: DataLoader | None, device: torch.device) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            context_target, context_dynamic, future_dynamic, static, target = move_batch(
                batch, device
            )
            loc, scale = model(
                context_target, context_dynamic, future_dynamic, static
            )
            loss = gaussian_nll(target, loc, scale)
            total_loss += float(loss.item()) * len(target)
            total_count += len(target)
    return total_loss / total_count if total_count else float("nan")


def fit_network(
    data: ResidualDeepARData,
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    seed: int,
    flow_name: str,
    early_stopping: bool,
) -> tuple[DeepARNetwork, dict[str, object]]:
    set_seed(seed)
    model = DeepARNetwork(
        dynamic_dim=data.dynamic_features.shape[2],
        static_dim=data.static_features.shape[1],
        hidden_size=args.hidden_size,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    validation_loader = (
        make_loader(
            validation_dataset, args.batch_size, False, args.num_workers, device
        )
        if validation_dataset is not None and len(validation_dataset)
        else None
    )
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            context_target, context_dynamic, future_dynamic, static, target = move_batch(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            loc, scale = model(
                context_target, context_dynamic, future_dynamic, static
            )
            loss = gaussian_nll(target, loc, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            total_count += len(target)
        train_loss = total_loss / total_count if total_count else float("nan")
        val_loss = evaluate_nll(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_loss,
                "validation_nll": val_loss,
            }
        )
        if validation_loader is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if early_stopping and stale >= args.patience:
                    break
        if epoch == 1 or epoch == epochs or epoch % max(args.log_every, 1) == 0:
            message = (
                f"[{flow_name} residual] epoch={epoch}/{epochs}, "
                f"train_nll={train_loss:.5f}"
            )
            if validation_loader is not None:
                message += f", val_nll={val_loss:.5f}"
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
        "best_validation_nll": best_val if validation_loader is not None else np.nan,
        "last_train_nll": history[-1]["train_nll"],
        "history": history,
    }
    return model, info


def _trial_args(
    args: argparse.Namespace, params: dict[str, float | int]
) -> argparse.Namespace:
    trial = copy.copy(args)
    for key, value in params.items():
        setattr(trial, key, value)
    return trial


def make_tuning_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    values = {
        "hidden_size": args.tune_hidden_sizes,
        "rnn_layers": args.tune_rnn_layers,
        "dropout": args.tune_dropouts,
        "learning_rate": args.tune_learning_rates,
        "batch_size": args.tune_batch_sizes,
    }
    return [
        dict(zip(TUNING_KEYS, combination))
        for combination in itertools.product(*(values[key] for key in TUNING_KEYS))
    ]


def tuning_result_key(
    flow_name: str,
    params: dict[str, float | int],
    max_epochs: int,
    patience: int,
    context_length: int,
) -> tuple[object, ...]:
    return (
        flow_name,
        int(params["hidden_size"]),
        int(params["rnn_layers"]),
        float(params["dropout"]),
        float(params["learning_rate"]),
        int(params["batch_size"]),
        int(max_epochs),
        int(patience),
        int(context_length),
    )


def _make_flow_datasets(
    data: ResidualDeepARData,
    flow_name: str,
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, Dataset, int, int]:
    context_length = args.context_length or (12 if data.frequency == "monthly" else 52)
    if context_length >= data.test_start_index:
        raise ValueError(f"context_length={context_length} is too long")
    validation_length = min(
        args.validation_length or data.horizon,
        data.test_start_index - context_length,
    )
    validation_start = data.test_start_index - validation_length
    values = data.values_by_flow[flow_name]
    scales = data.scales_by_flow[flow_name]
    train_indices = make_window_indices(
        len(data.account_ids), context_length, context_length, validation_start
    )
    validation_indices = make_window_indices(
        len(data.account_ids), context_length, validation_start, data.test_start_index
    )
    final_indices = make_window_indices(
        len(data.account_ids), context_length, context_length, data.test_start_index
    )
    datasets = tuple(
        ResidualDeepARWindowDataset(
            values,
            scales,
            data.dynamic_features,
            data.static_features,
            indices,
            context_length,
        )
        for indices in (train_indices, validation_indices, final_indices)
    )
    if not len(datasets[0]):
        raise ValueError(f"No training windows for {flow_name}/{data.frequency}")
    return datasets[0], datasets[1], datasets[2], context_length, validation_start


def validate_flow_hyperparameters(
    data: ResidualDeepARData,
    flow_name: str,
    device: torch.device,
    args: argparse.Namespace,
    params: dict[str, float | int],
) -> dict[str, object]:
    trial = _trial_args(args, params)
    train_dataset, validation_dataset, _, context_length, validation_start = _make_flow_datasets(
        data, flow_name, trial
    )
    _, info = fit_network(
        data,
        train_dataset,
        validation_dataset,
        device,
        trial,
        trial.epochs,
        trial.seed + (0 if flow_name == "inflow" else 1),
        flow_name,
        True,
    )
    return {
        "frequency": data.frequency,
        "flow": flow_name,
        **params,
        "max_epochs": trial.epochs,
        "patience": trial.patience,
        "context_length": context_length,
        "validation_start": str(data.dates[validation_start].date()),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": info["best_epoch"],
        "validation_nll": info["best_validation_nll"],
        "epochs_ran": info["epochs_ran"],
    }


def train_flow_model(
    data: ResidualDeepARData,
    flow_name: str,
    device: torch.device,
    args: argparse.Namespace,
    params: dict[str, float | int],
    validation_result: dict[str, object] | None = None,
) -> tuple[DeepARNetwork, dict[str, object]]:
    trial = _trial_args(args, params)
    train_dataset, validation_dataset, final_dataset, context_length, validation_start = _make_flow_datasets(
        data, flow_name, trial
    )
    if validation_result is None:
        _, validation_info = fit_network(
            data,
            train_dataset,
            validation_dataset,
            device,
            trial,
            trial.epochs,
            trial.seed + (0 if flow_name == "inflow" else 1),
            flow_name,
            True,
        )
    else:
        validation_info = {
            "flow": flow_name,
            "best_epoch": int(validation_result["best_epoch"]),
            "best_validation_nll": float(validation_result["validation_nll"]),
            "epochs_ran": int(validation_result["epochs_ran"]),
        }
    final_model, final_info = fit_network(
        data,
        final_dataset,
        None,
        device,
        trial,
        int(validation_info["best_epoch"]),
        trial.seed + 100 + (0 if flow_name == "inflow" else 1),
        flow_name,
        False,
    )
    info = {key: value for key, value in validation_info.items() if key != "history"}
    info.update(
        {
            **params,
            "max_epochs": trial.epochs,
            "patience": trial.patience,
            "final_train_samples": len(final_dataset),
            "final_epochs": final_info["epochs_ran"],
            "context_length": context_length,
            "validation_start": str(data.dates[validation_start].date()),
        }
    )
    return final_model, info


@torch.no_grad()
def predict_one_step(
    model: DeepARNetwork,
    histories: np.ndarray,
    scales: np.ndarray,
    data: ResidualDeepARData,
    target_index: int,
    context_length: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Predict a residual step; the operational point is inverse(loc)."""
    model.eval()
    n_accounts = len(data.account_ids)
    start = target_index - context_length
    context_dynamic = data.dynamic_features[:, start:target_index, :]
    future_dynamic = data.dynamic_features[:, target_index, :]
    context_target = signed_log1p(
        torch.from_numpy(histories[:, -context_length:]),
        torch.from_numpy(scales[:, None]),
    ).numpy().astype(np.float32)
    outputs: list[np.ndarray] = []
    for batch_start in range(0, n_accounts, batch_size):
        batch_end = min(batch_start + batch_size, n_accounts)
        tensors = [
            torch.from_numpy(context_target[batch_start:batch_end]).to(device),
            torch.from_numpy(context_dynamic[batch_start:batch_end]).to(device),
            torch.from_numpy(future_dynamic[batch_start:batch_end]).to(device),
            torch.from_numpy(data.static_features[batch_start:batch_end]).to(device),
        ]
        loc, distribution_scale = model(*tensors)
        account_scale = torch.from_numpy(scales[batch_start:batch_end]).to(device)
        # In signed-log space loc maps to the median residual.  The Gaussian
        # mean has no simple closed form after the signed inverse, so loc is
        # used as the deterministic point forecast.
        raw = account_scale * signed_expm1(loc)
        outputs.append(raw.cpu().numpy())
    raw_residual = np.concatenate(outputs).astype(float)
    return {
        "raw_residual": raw_residual,
        "residual": raw_residual,
    }


def recursive_forecast(
    data: ResidualDeepARData,
    models: dict[str, DeepARNetwork],
    training_info: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Forecast one residual step at a time and append absolute predictions."""
    context_lengths = {
        row["flow"]: int(row["context_length"])
        for _, row in training_info.iterrows()
    }
    histories = {
        name: data.values_by_flow[name][:, : data.test_start_index].copy()
        for name in FLOW_TARGETS
    }
    absolute_inflow = data.base.values_by_flow["inflow"]
    absolute_outflow = data.base.values_by_flow["outflow"]
    records: list[dict[str, object]] = []
    step_delta = FREQUENCY_CONFIG[data.frequency]["step"]
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
        known = data.test_known_features[target_date]
        fixed_outflow = data.schedule[:, target_index].astype(float)
        for account_index, account_id in enumerate(data.account_ids):
            raw_inflow_residual = float(forecasts["inflow"]["raw_residual"][account_index])
            raw_outflow_residual = float(forecasts["outflow"]["raw_residual"][account_index])
            fixed_inflow = 0.0
            fixed_outflow_value = float(fixed_outflow[account_index])
            raw_inflow = fixed_inflow + raw_inflow_residual
            raw_outflow = fixed_outflow_value + raw_outflow_residual
            prediction_inflow = max(raw_inflow, 0.0)
            prediction_outflow = max(raw_outflow, 0.0)
            actual_inflow_value = float(absolute_inflow[account_index, target_index])
            actual_outflow_value = float(absolute_outflow[account_index, target_index])
            record: dict[str, object] = {
                "account_id": int(account_id),
                "period_start": target_date,
                "origin_period": target_date - step_delta,
                "horizon_step": step,
                "fixed_cost_inflow": fixed_inflow,
                "fixed_cost_outflow": fixed_outflow_value,
                "fixed_cost_net_flow": fixed_inflow - fixed_outflow_value,
                "actual_inflow": actual_inflow_value,
                "actual_outflow": actual_outflow_value,
                "actual_inflow_residual": actual_inflow_value - fixed_inflow,
                "actual_outflow_residual": actual_outflow_value - fixed_outflow_value,
                "prediction_inflow_raw": raw_inflow,
                "prediction_outflow_raw": raw_outflow,
                "prediction_inflow": prediction_inflow,
                "prediction_outflow": prediction_outflow,
                "prediction_inflow_residual_raw": raw_inflow_residual,
                "prediction_outflow_residual_raw": raw_outflow_residual,
                "prediction_inflow_residual": prediction_inflow - fixed_inflow,
                "prediction_outflow_residual": prediction_outflow - fixed_outflow_value,
            }
            for name in known.columns:
                record[name] = float(known.loc[int(account_id), name])
            records.append(record)
        histories["inflow"] = np.column_stack(
            [
                histories["inflow"],
                np.maximum(forecasts["inflow"]["residual"], 0.0),
            ]
        )
        histories["outflow"] = np.column_stack(
            [
                histories["outflow"],
                np.maximum(forecasts["outflow"]["residual"] + fixed_outflow, 0.0),
            ]
        )
    result = add_residual_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )
    # Both histories contain final absolute operational predictions.  This is
    # important because future lag/rolling inputs must see the same clipped
    # values that were reported to the caller.
    return result


def _absolute_scales(data: ResidualDeepARData) -> dict[str, pd.Series]:
    scale_df = data.df.copy()
    scale_df["net_flow_for_scale"] = (
        scale_df["inflow_amount"] - scale_df["outflow_amount"]
    )
    return {
        "inflow": train_normalization_scale(
            scale_df, data.test_start, "inflow_amount"
        ),
        "outflow": train_normalization_scale(
            scale_df, data.test_start, "outflow_amount"
        ),
        "net_flow": train_normalization_scale(
            scale_df, data.test_start, "net_flow_for_scale"
        ),
    }


def _residual_scales(data: ResidualDeepARData) -> dict[str, pd.Series]:
    ids = pd.Index(data.account_ids, name="account_id")
    result = {}
    inflow = data.values_by_flow["inflow"][:, : data.test_start_index]
    outflow = data.values_by_flow["outflow"][:, : data.test_start_index]
    for name, values in {
        "inflow": inflow,
        "outflow": outflow,
        "net_flow": inflow - outflow,
    }.items():
        scale = np.mean(np.abs(values), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        result[name] = pd.Series(np.maximum(scale, 1.0), index=ids, dtype=float)
    return result


def _metric_specs(residual: bool):
    if residual:
        return [
            (
                "inflow",
                "actual_inflow_residual",
                "prediction_inflow_residual",
                "prediction_inflow_residual_raw",
                "inflow",
            ),
            (
                "outflow",
                "actual_outflow_residual",
                "prediction_outflow_residual",
                "prediction_outflow_residual_raw",
                "outflow",
            ),
            (
                "net_flow",
                "actual_net_flow_residual",
                "prediction_net_flow_residual",
                "prediction_net_flow_residual_raw",
                "net_flow",
            ),
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


def add_residual_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and residual net-flow columns used by metric writers."""
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


def _write_metric_files(
    predictions: pd.DataFrame,
    data: ResidualDeepARData,
    run_dir: Path,
    under_weight: float,
    over_weight: float,
) -> pd.DataFrame:
    scales_by_kind = {
        "absolute": _absolute_scales(data),
        "residual": _residual_scales(data),
    }
    horizon_columns = [
        "actual_inflow", "actual_outflow", "prediction_inflow", "prediction_inflow_raw",
        "prediction_outflow", "prediction_outflow_raw", "fixed_cost_inflow",
        "fixed_cost_outflow", "fixed_cost_net_flow", "actual_inflow_residual",
        "actual_outflow_residual", "prediction_inflow_residual",
        "prediction_inflow_residual_raw", "prediction_outflow_residual",
        "prediction_outflow_residual_raw",
    ]
    horizon_predictions = predictions.groupby("account_id", as_index=False)[horizon_columns].sum()
    horizon_predictions = add_residual_columns(horizon_predictions)
    horizon_predictions["horizon_step"] = "horizon_total"
    all_horizon_rows: list[pd.DataFrame] = []
    for kind, scales in scales_by_kind.items():
        rows: list[pd.DataFrame] = []
        for target, actual_col, prediction_col, raw_col, scale_name in _metric_specs(kind == "residual"):
            target_under, target_over = _weights(target, under_weight, over_weight)
            rows.append(
                calculate_metrics(
                    predictions,
                    scales[scale_name],
                    actual_column=actual_col,
                    prediction_column=prediction_col,
                    raw_prediction_column=raw_col,
                    under_weight=target_under,
                    over_weight=target_over,
                ).assign(target=target, evaluation_target=kind)
            )
            all_horizon_rows.append(
                calculate_metrics(
                    horizon_predictions,
                    scales[scale_name],
                    actual_column=actual_col,
                    prediction_column=prediction_col,
                    raw_prediction_column=raw_col,
                    under_weight=target_under,
                    over_weight=target_over,
                    scale_multiplier=data.horizon,
                ).assign(
                    target=target,
                    evaluation_target=kind,
                    horizon_periods=data.horizon,
                )
            )
        pd.concat(rows, ignore_index=True).to_csv(
            run_dir / ("residual_metrics.csv" if kind == "residual" else "metrics.csv"),
            index=False,
        )
    horizon_metrics = pd.concat(all_horizon_rows, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[
        horizon_metrics["horizon_step"].astype(str).eq("horizon_total")
    ]
    horizon_metrics.loc[horizon_metrics["evaluation_target"].eq("absolute")].to_csv(
        run_dir / "horizon_metrics.csv", index=False
    )
    horizon_metrics.loc[horizon_metrics["evaluation_target"].eq("residual")].to_csv(
        run_dir / "residual_horizon_metrics.csv", index=False
    )
    return horizon_metrics.loc[horizon_metrics["evaluation_target"].eq("absolute")].copy()


def save_residual_deepar_outputs(
    data: ResidualDeepARData,
    predictions: pd.DataFrame,
    training_info: pd.DataFrame,
    output_dir: Path,
    under_weight: float,
    over_weight: float,
) -> tuple[Path, pd.DataFrame]:
    run_dir = output_dir / f"{data.frequency}_recursive"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = predictions.copy()
    output.insert(0, "frequency", data.frequency)
    output.insert(1, "method", "recursive")
    output.to_csv(run_dir / "predictions.csv", index=False)
    output.to_parquet(run_dir / "predictions.parquet", index=False)
    horizon_metrics = _write_metric_files(
        output, data, run_dir, under_weight, over_weight
    )
    training_info.assign(
        frequency=data.frequency,
        method="recursive",
        model="DeepAR_residual",
    ).to_csv(run_dir / "training_info.csv", index=False)
    save_forecast_plot(
        output,
        data.frequency,
        "recursive_residual",
        run_dir / "aggregate_test_forecast.png",
    )
    metadata = {
        "model": "DeepAR_residual",
        "frequency": data.frequency,
        "method": "recursive",
        "master_file": str(enriched_master_path(data.frequency)),
        "fixed_cost_policy": fixed_cost_policy(),
        "residual_targets": {
            "inflow": "inflow_amount - 0",
            "outflow": "outflow_amount - scheduled_loan_repayment",
            "net_flow": "net_flow - fixed_cost_net_flow",
        },
        "residual_transform": "sign(x) * log1p(abs(x) / account_scale)",
        "point_forecast": "inverse signed-log of Gaussian loc (median in transformed space)",
        "account_id_embedding": False,
        "dynamic_feature_names": data.dynamic_feature_names,
        "static_feature_names": data.static_feature_names,
        "scheduled_loan_repayment": (
            "known account-specific covariate; contracts masked as-of forecast origin"
        ),
        "device": str(training_info.attrs.get("device", "unknown")),
        "n_accounts": len(data.account_ids),
        "test_start": str(data.test_start.date()),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return run_dir, horizon_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive"], default="recursive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--strict-cuda", action="store_true")
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
    parser.add_argument("--skip-hyperparameter-search", action="store_true")
    parser.add_argument("--limit-tuning-combinations", type=int, default=None)
    parser.add_argument("--tune-hidden-sizes", nargs="+", type=int, default=DEFAULT_TUNING_GRID["hidden_size"])
    parser.add_argument("--tune-rnn-layers", nargs="+", type=int, default=DEFAULT_TUNING_GRID["rnn_layers"])
    parser.add_argument("--tune-dropouts", nargs="+", type=float, default=DEFAULT_TUNING_GRID["dropout"])
    parser.add_argument("--tune-learning-rates", nargs="+", type=float, default=DEFAULT_TUNING_GRID["learning_rate"])
    parser.add_argument("--tune-batch-sizes", nargs="+", type=int, default=DEFAULT_TUNING_GRID["batch_size"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_tuning_combinations is not None and args.limit_tuning_combinations < 1:
        raise ValueError("--limit-tuning-combinations must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device, args.strict_cuda)
    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "hyperparameter_sweep_progress.csv"
    all_sweep_rows: list[dict[str, object]] = []
    if progress_path.exists() and not args.skip_hyperparameter_search:
        try:
            all_sweep_rows = pd.read_csv(progress_path).to_dict("records")
            print(f"loaded {len(all_sweep_rows):,} completed residual DeepAR tuning rows")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            all_sweep_rows = []
    all_best_rows: list[dict[str, object]] = []
    horizon_rows: list[pd.DataFrame] = []

    for frequency in frequencies:
        account_ids = args.account_ids
        if account_ids is None and args.max_accounts is not None:
            account_ids = get_available_enriched_account_ids(frequency, args.max_accounts)
        data = create_residual_deepar_data(frequency, account_ids)

        if args.skip_hyperparameter_search:
            selected_params = {key: getattr(args, key) for key in TUNING_KEYS}
            selected_results = {flow_name: None for flow_name in FLOW_TARGETS}
            print(f"skipping residual DeepAR search for {frequency}: {selected_params}")
        else:
            tuning_grid = make_tuning_grid(args)
            if args.limit_tuning_combinations is not None:
                tuning_grid = tuning_grid[: args.limit_tuning_combinations]
            print(
                f"running residual DeepAR validation search for {frequency}: "
                f"{len(tuning_grid)} combinations x {len(FLOW_TARGETS)} flows"
            )
            frequency_rows = [
                row for row in all_sweep_rows if str(row.get("frequency")) == frequency
            ]
            effective_context = args.context_length or (12 if frequency == "monthly" else 52)
            completed = {
                tuning_result_key(
                    str(row["flow"]),
                    {key: row[key] for key in TUNING_KEYS},
                    int(row.get("max_epochs", -1)),
                    int(row.get("patience", -1)),
                    int(row.get("context_length", -1)),
                )
                for row in frequency_rows
                if all(key in row for key in TUNING_KEYS + ("max_epochs", "patience", "context_length"))
            }
            for flow_name in FLOW_TARGETS:
                for combination_index, params in enumerate(tuning_grid, start=1):
                    candidate_key = tuning_result_key(
                        flow_name, params, args.epochs, args.patience, effective_context
                    )
                    if candidate_key in completed:
                        print(f"skipping completed residual tuning {frequency}/{flow_name}: {params}")
                        continue
                    print(
                        f"tuning residual {frequency}/{flow_name}: "
                        f"{combination_index}/{len(tuning_grid)} {params}"
                    )
                    row = validate_flow_hyperparameters(
                        data, flow_name, device, args, params
                    )
                    row["combination_id"] = combination_index
                    frequency_rows.append(row)
                    all_sweep_rows.append(row)
                    completed.add(candidate_key)
                    pd.DataFrame(all_sweep_rows).to_csv(progress_path, index=False)
            frequency_sweep = pd.DataFrame(frequency_rows)
            frequency_sweep.to_csv(
                args.output_dir / f"{frequency}_hyperparameter_sweep.csv", index=False
            )
            selected_results = {}
            selected_params = {}
            for flow_name in FLOW_TARGETS:
                flow_rows = frequency_sweep.loc[frequency_sweep["flow"].eq(flow_name)]
                winner = flow_rows.loc[flow_rows["validation_nll"].idxmin()]
                winner_dict = winner.to_dict()
                selected_results[flow_name] = winner_dict
                selected_params[flow_name] = {key: winner_dict[key] for key in TUNING_KEYS}
                all_best_rows.append(winner_dict)
            pd.DataFrame(all_best_rows).to_csv(
                args.output_dir / "best_hyperparameters_progress.csv", index=False
            )

        models: dict[str, DeepARNetwork] = {}
        training_rows: list[dict[str, object]] = []
        for flow_name in FLOW_TARGETS:
            model, info = train_flow_model(
                data,
                flow_name,
                device,
                args,
                selected_params if args.skip_hyperparameter_search else selected_params[flow_name],
                None if args.skip_hyperparameter_search else selected_results[flow_name],
            )
            info["selected_by"] = "cli_values" if args.skip_hyperparameter_search else "validation_nll"
            models[flow_name] = model
            training_rows.append(info)
        training_info = pd.DataFrame(training_rows)
        training_info.attrs["device"] = str(device)
        predictions = recursive_forecast(data, models, training_info, device, args)
        run_dir, horizon_metrics = save_residual_deepar_outputs(
            data,
            predictions,
            training_info,
            args.output_dir,
            args.under_weight,
            args.over_weight,
        )
        for flow_name, model in models.items():
            torch.save(
                {
                    "flow": flow_name,
                    "frequency": frequency,
                    "model_config": model.config,
                    "state_dict": model.state_dict(),
                    "dynamic_feature_names": data.dynamic_feature_names,
                    "static_feature_names": data.static_feature_names,
                    "residual_transform": "signed_log1p",
                },
                run_dir / f"{flow_name}_residual_model.pt",
            )
        horizon_rows.append(horizon_metrics.assign(frequency=frequency, method="recursive"))
        print(
            f"completed residual DeepAR {frequency}/recursive: "
            f"{len(data.account_ids):,} accounts, device={device}"
        )

    if not args.skip_hyperparameter_search:
        pd.DataFrame(all_sweep_rows).to_csv(
            args.output_dir / "hyperparameter_sweep.csv", index=False
        )
        pd.DataFrame(all_best_rows).to_csv(
            args.output_dir / "best_hyperparameters.csv", index=False
        )
    if horizon_rows:
        pd.concat(horizon_rows, ignore_index=True).to_csv(
            args.output_dir / "horizon_comparison_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
