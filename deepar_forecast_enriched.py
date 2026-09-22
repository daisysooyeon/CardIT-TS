"""DeepAR recursive forecasts with enriched known-future covariates.

This is a separate DeepAR entry point for the enriched master tables.  The
calendar/holiday features are global known covariates; scheduled loan
repayment is an account-specific known covariate.  Therefore dynamic features
have shape ``(account, period, feature)`` rather than being broadcast across
accounts.  The model still trains one shared LSTM per flow and does not use an
account-id embedding.

DeepAR remains recursive only, as in the original implementation.  At test
time the future covariates are looked up from the pre-test origin and only
predicted flow values are appended to the autoregressive history.
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
    transformed_to_raw,
)
from enriched_forecast_common import (
    MONTHLY_KNOWN_FEATURES,
    WEEKLY_KNOWN_FEATURES,
    enriched_master_path,
    get_available_enriched_account_ids,
    load_loan_events,
    read_enriched_master,
    save_enriched_run_outputs,
)
from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    derive_test_start,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepar_enriched"

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
    """Resolve the runtime device, falling back to CPU when CUDA is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        if strict_cuda:
            raise RuntimeError(
                "CUDA was requested, but this runtime has no available GPU."
            )
        print(
            "[warning] CUDA is unavailable in this runtime; "
            "falling back to CPU."
        )
        return torch.device("cpu")
    return resolve_base_device(requested)


@dataclass
class EnrichedDeepARData:
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
    test_known_features: dict[pd.Timestamp, pd.DataFrame]
    schedule_dynamic_scale: float


def validate_regular_dates(dates: pd.DatetimeIndex, frequency: str) -> None:
    expected = (
        pd.date_range(dates[0], dates[-1], freq="MS")
        if frequency == "monthly"
        else pd.date_range(dates[0], dates[-1], freq="W-SUN")
    )
    if not dates.equals(expected):
        raise ValueError(f"{frequency} dates are not regular")


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
        raise ValueError(f"{value_column} has missing account-period values")
    return pivot.to_numpy(dtype=np.float32)


def build_static_features(
    df: pd.DataFrame,
    account_ids: list[int],
) -> tuple[np.ndarray, list[str]]:
    static = (
        df.sort_values(["account_id", "period_start"])
        .groupby("account_id", sort=True)[["birth_year", "gender", "account_frequency"]]
        .first()
        .reindex(account_ids)
    )
    birth = pd.to_numeric(static["birth_year"], errors="coerce").astype(float)
    birth = birth.fillna(birth.median())
    std = float(birth.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        std = 1.0
    parts = [((birth.to_numpy() - float(birth.mean())) / std).astype(np.float32)]
    names = ["birth_year_standardized"]
    for column in ("gender", "account_frequency"):
        values = static[column].astype("string").fillna("missing")
        for category in sorted(str(value) for value in values.unique()):
            parts.append((values == category).to_numpy(dtype=np.float32))
            names.append(f"{column}={category}")
    return np.column_stack(parts).astype(np.float32), names


def _calendar_table(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    names = MONTHLY_KNOWN_FEATURES if frequency == "monthly" else WEEKLY_KNOWN_FEATURES
    names = [name for name in names if name != "scheduled_loan_repayment"]
    return (
        df[["period_start", *names]]
        .drop_duplicates("period_start")
        .set_index("period_start")
        .sort_index()
    )


def build_schedule_matrix(
    account_ids: list[int],
    dates: list[pd.Timestamp],
    test_start: pd.Timestamp,
    frequency: str,
) -> tuple[np.ndarray, dict[pd.Timestamp, pd.Series]]:
    """Build a schedule known at each period's beginning.

    For historical target period ``p``, a loan is usable only when its
    contract date is before ``p``.  For test periods, the information set is
    frozen at the last pre-test date, so loans opened during the test window
    never leak into the covariate.
    """
    events = load_loan_events(frequency)
    dates_index = pd.DatetimeIndex(dates)
    account_lookup = {account_id: index for index, account_id in enumerate(account_ids)}
    date_lookup = {pd.Timestamp(date): index for index, date in enumerate(dates_index)}
    test_start_index = int(np.searchsorted(dates_index.values, test_start.to_datetime64()))
    raw = np.zeros((len(account_ids), len(dates)), dtype=np.float32)
    for event in events.itertuples(index=False):
        due_period = pd.Timestamp(event.period_start)
        if int(event.account_id) not in account_lookup or due_period not in date_lookup:
            continue
        due_index = date_lookup[due_period]
        as_of = test_start if due_index >= test_start_index else due_period
        if pd.Timestamp(event.loan_date) < as_of:
            raw[account_lookup[int(event.account_id)], due_index] += float(event.scheduled_amount)

    test_lookup: dict[pd.Timestamp, pd.Series] = {}
    for date in dates[test_start_index : test_start_index + FREQUENCY_CONFIG[frequency]["horizon"]]:
        test_lookup[pd.Timestamp(date)] = pd.Series(
            raw[:, date_lookup[pd.Timestamp(date)]], index=account_ids, dtype=float
        )
    return raw, test_lookup


def create_enriched_deepar_data(
    frequency: str,
    account_ids: list[int] | None,
) -> EnrichedDeepARData:
    df = read_enriched_master(frequency, account_ids)
    account_ids = sorted(int(value) for value in df["account_id"].unique())
    dates_index = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in df["period_start"].unique()))
    validate_regular_dates(dates_index, frequency)
    dates = list(dates_index)
    test_start = derive_test_start(frequency, df)
    test_start_index = int(np.searchsorted(dates_index.values, test_start.to_datetime64()))
    horizon = FREQUENCY_CONFIG[frequency]["horizon"]
    test_dates = dates[test_start_index : test_start_index + horizon]
    if len(test_dates) != horizon:
        raise ValueError(f"Expected {horizon} test dates for {frequency}, found {len(test_dates)}")

    values_by_flow = {
        name: value_matrix(df, account_ids, dates, column)
        for name, column in FLOW_COLUMNS.items()
    }
    scales_by_flow: dict[str, np.ndarray] = {}
    for name, values in values_by_flow.items():
        scale = np.mean(np.abs(values[:, :test_start_index]), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        scales_by_flow[name] = np.maximum(scale, 1.0).astype(np.float32)

    calendar = _calendar_table(df, frequency)
    schedule_raw, _ = build_schedule_matrix(account_ids, dates, test_start, frequency)
    schedule_scale = float(np.nanpercentile(schedule_raw[:, :test_start_index], 99)) if schedule_raw[:, :test_start_index].size else 1.0
    if not np.isfinite(schedule_scale) or schedule_scale <= 0:
        schedule_scale = 1.0

    time_position = np.arange(len(dates), dtype=np.float32) / max(len(dates) - 1, 1)
    dynamic_names = ["time_position"]
    global_values = [time_position]
    calendar_names = list(calendar.columns)
    for name in calendar_names:
        values = calendar.reindex(dates)[name].to_numpy(dtype=np.float32)
        if name == "holiday_count":
            values = values / max(float(values.max()), 1.0)
        global_values.append(values)
        dynamic_names.append(name)
    global_dynamic = np.column_stack(global_values).astype(np.float32)
    dynamic_features = np.repeat(global_dynamic[None, :, :], len(account_ids), axis=0)
    dynamic_features = np.concatenate(
        [dynamic_features, (schedule_raw / schedule_scale).astype(np.float32)[:, :, None]],
        axis=2,
    )
    dynamic_names.append("scheduled_loan_repayment_scaled")

    static_features, static_names = build_static_features(df, account_ids)
    test_known: dict[pd.Timestamp, pd.DataFrame] = {}
    known_names = calendar_names + ["scheduled_loan_repayment"]
    for date in test_dates:
        frame = calendar.loc[[date]].copy()
        frame = frame.loc[frame.index.repeat(len(account_ids))].copy()
        frame.index = account_ids
        frame["scheduled_loan_repayment"] = schedule_raw[:, date_lookup_index(dates, date)]
        test_known[pd.Timestamp(date)] = frame[known_names]

    return EnrichedDeepARData(
        frequency=frequency, horizon=horizon, df=df, account_ids=account_ids,
        dates=dates, test_dates=test_dates, test_start=test_start,
        test_start_index=test_start_index, values_by_flow=values_by_flow,
        scales_by_flow=scales_by_flow, dynamic_features=dynamic_features,
        dynamic_feature_names=dynamic_names, static_features=static_features,
        static_feature_names=static_names, test_known_features=test_known,
        schedule_dynamic_scale=schedule_scale,
    )


def date_lookup_index(dates: list[pd.Timestamp], date: pd.Timestamp) -> int:
    return int(pd.DatetimeIndex(dates).get_loc(pd.Timestamp(date)))


class EnrichedDeepARWindowDataset(Dataset):
    def __init__(self, values, scales, dynamic_features, static_features, indices, context_length):
        self.values = torch.from_numpy(values.astype(np.float32, copy=False))
        self.scales = torch.from_numpy(scales.astype(np.float32, copy=False))
        self.dynamic_features = torch.from_numpy(dynamic_features.astype(np.float32, copy=False))
        self.static_features = torch.from_numpy(static_features.astype(np.float32, copy=False))
        self.indices = indices
        self.context_length = int(context_length)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        account_index, target_index = self.indices[index]
        start = int(target_index) - self.context_length
        scale = self.scales[account_index]
        context_raw = torch.clamp(self.values[account_index, start:target_index], min=0.0)
        context_target = torch.log1p(context_raw / scale)
        target_raw = torch.clamp(self.values[account_index, target_index], min=0.0)
        target = torch.log1p(target_raw / scale)
        return (
            context_target,
            self.dynamic_features[account_index, start:target_index],
            self.dynamic_features[account_index, target_index],
            self.static_features[account_index],
            target,
        )


def evaluate_nll(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            context_target, context_dynamic, future_dynamic, static, target = move_batch(batch, device)
            loc, scale = model(context_target, context_dynamic, future_dynamic, static)
            loss = gaussian_nll(target, loc, scale)
            total_loss += float(loss.item()) * len(target)
            total_count += len(target)
    return total_loss / total_count if total_count else float("nan")


def fit_network(data, train_dataset, validation_dataset, device, args, epochs, seed, flow_name, early_stopping):
    set_seed(seed)
    model = DeepARNetwork(
        dynamic_dim=data.dynamic_features.shape[2],
        static_dim=data.static_features.shape[1],
        hidden_size=args.hidden_size, rnn_layers=args.rnn_layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    validation_loader = make_loader(validation_dataset, args.batch_size, False, args.num_workers, device) if validation_dataset is not None and len(validation_dataset) else None
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
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
            total_loss += float(loss.item()) * len(target)
            total_count += len(target)
        train_loss = total_loss / total_count if total_count else float("nan")
        val_loss = evaluate_nll(model, validation_loader, device) if validation_loader else float("nan")
        history.append({"epoch": epoch, "train_nll": train_loss, "validation_nll": val_loss})
        if validation_loader:
            if val_loss < best_val:
                best_val, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if early_stopping and stale >= args.patience:
                    break
        if epoch == 1 or epoch == epochs or epoch % max(args.log_every, 1) == 0:
            print(f"[{flow_name}] epoch={epoch}/{epochs}, train_nll={train_loss:.5f}" + (f", val_nll={val_loss:.5f}" if validation_loader else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {
        "flow": flow_name, "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
        "epochs_requested": epochs, "epochs_ran": len(history),
        "best_epoch": best_epoch or epochs,
        "best_validation_nll": best_val if validation_loader else np.nan,
        "last_train_nll": history[-1]["train_nll"], "history": history,
    }
    return model, info


def _trial_args(args: argparse.Namespace, params: dict[str, float | int]) -> argparse.Namespace:
    """Copy CLI arguments and replace only the tuned hyperparameters."""
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
    """Create a stable key so an interrupted search can resume safely."""
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
    data: EnrichedDeepARData,
    flow_name: str,
    args: argparse.Namespace,
) -> tuple[
    EnrichedDeepARWindowDataset,
    EnrichedDeepARWindowDataset,
    EnrichedDeepARWindowDataset,
    int,
    int,
]:
    context_length = args.context_length or (12 if data.frequency == "monthly" else 52)
    if context_length >= data.test_start_index:
        raise ValueError(f"context_length={context_length} is too long")
    validation_length = min(
        args.validation_length or data.horizon,
        data.test_start_index - context_length,
    )
    validation_start = data.test_start_index - validation_length
    values, scales = data.values_by_flow[flow_name], data.scales_by_flow[flow_name]
    train_indices = make_window_indices(
        len(data.account_ids), context_length, context_length, validation_start
    )
    validation_indices = make_window_indices(
        len(data.account_ids), context_length, validation_start, data.test_start_index
    )
    final_indices = make_window_indices(
        len(data.account_ids), context_length, context_length, data.test_start_index
    )
    train_dataset = EnrichedDeepARWindowDataset(
        values, scales, data.dynamic_features, data.static_features,
        train_indices, context_length,
    )
    validation_dataset = EnrichedDeepARWindowDataset(
        values, scales, data.dynamic_features, data.static_features,
        validation_indices, context_length,
    )
    final_dataset = EnrichedDeepARWindowDataset(
        values, scales, data.dynamic_features, data.static_features,
        final_indices, context_length,
    )
    if not len(train_dataset):
        raise ValueError(f"No training windows for {flow_name}/{data.frequency}")
    return train_dataset, validation_dataset, final_dataset, context_length, validation_start


def validate_flow_hyperparameters(
    data: EnrichedDeepARData,
    flow_name: str,
    device: torch.device,
    args: argparse.Namespace,
    params: dict[str, float | int],
) -> dict[str, object]:
    """Train one candidate only on the train/validation split."""
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
    data,
    flow_name,
    device,
    args,
    params=None,
    validation_result: dict[str, object] | None = None,
):
    """Select an epoch count on validation, then retrain on all pre-test data."""
    params = params or {
        key: getattr(args, key)
        for key in TUNING_KEYS
    }
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
    info = {
        key: value for key, value in validation_info.items() if key != "history"
    }
    info.update({
        **params,
        "max_epochs": trial.epochs,
        "patience": trial.patience,
        "final_train_samples": len(final_dataset),
        "final_epochs": final_info["epochs_ran"],
        "context_length": context_length,
        "validation_start": str(data.dates[validation_start].date()),
    })
    return final_model, info


@torch.no_grad()
def predict_one_step(model, histories, scales, data, target_index, context_length, device, batch_size):
    model.eval()
    n_accounts = len(data.account_ids)
    start = target_index - context_length
    context_dynamic = data.dynamic_features[:, start:target_index, :]
    future_dynamic = data.dynamic_features[:, target_index, :]
    context_values = np.maximum(histories[:, -context_length:], 0.0)
    context_target = np.log1p(context_values / scales[:, None]).astype(np.float32)
    output = {"mean": [], "p10": [], "p50": [], "p90": [], "raw": []}
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
        mean = account_scale * torch.expm1(torch.clamp(loc + 0.5 * distribution_scale.square(), -20.0, 20.0))
        values = {
            "mean": mean,
            "p10": account_scale * transformed_to_raw(loc, distribution_scale, -1.2815515655),
            "p50": account_scale * transformed_to_raw(loc, distribution_scale),
            "p90": account_scale * transformed_to_raw(loc, distribution_scale, 1.2815515655),
            "raw": mean,
        }
        for name, value in values.items():
            output[name].append(value.cpu().numpy())
    return {name: np.concatenate(values).astype(float) for name, values in output.items()}


def recursive_forecast(data, models, training_info, device, args):
    context_lengths = {row["flow"]: int(row["context_length"]) for _, row in training_info.iterrows()}
    histories = {name: data.values_by_flow[name][:, :data.test_start_index].copy() for name in FLOW_TARGETS}
    records = []
    for step, target_date in enumerate(data.test_dates, start=1):
        target_index = data.test_start_index + step - 1
        forecasts = {name: predict_one_step(models[name], histories[name], data.scales_by_flow[name], data, target_index, context_lengths[name], device, args.batch_size) for name in FLOW_TARGETS}
        known = data.test_known_features[target_date]
        for account_index, account_id in enumerate(data.account_ids):
            inflow, outflow = forecasts["inflow"], forecasts["outflow"]
            record = {
                "account_id": int(account_id), "period_start": target_date,
                "origin_period": target_date - FREQUENCY_CONFIG[data.frequency]["step"],
                "horizon_step": step,
                "actual_inflow": float(data.values_by_flow["inflow"][account_index, target_index]),
                "actual_outflow": float(data.values_by_flow["outflow"][account_index, target_index]),
                "prediction_inflow_raw": float(inflow["raw"][account_index]),
                "prediction_inflow": float(max(inflow["mean"][account_index], 0.0)),
                "prediction_outflow_raw": float(outflow["raw"][account_index]),
                "prediction_outflow": float(max(outflow["mean"][account_index], 0.0)),
            }
            for name in data.test_known_features[target_date].columns:
                record[name] = float(known.loc[account_id, name])
            records.append(record)
        histories["inflow"] = np.column_stack([histories["inflow"], forecasts["inflow"]["mean"]])
        histories["outflow"] = np.column_stack([histories["outflow"], forecasts["outflow"]["mean"]])
    from linear_regression_forecast import add_derived_flow_columns
    return add_derived_flow_columns(pd.DataFrame(records).sort_values(["account_id", "period_start"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive"], default="recursive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--strict-cuda",
        action="store_true",
        help="Fail instead of falling back to CPU when CUDA is unavailable.",
    )
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
    parser.add_argument(
        "--skip-hyperparameter-search",
        action="store_true",
        help="Skip the 48-combination validation search and use CLI values.",
    )
    parser.add_argument(
        "--limit-tuning-combinations",
        type=int,
        default=None,
        help="Run only the first N tuning combinations for a smoke test.",
    )
    parser.add_argument(
        "--tune-hidden-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TUNING_GRID["hidden_size"],
    )
    parser.add_argument(
        "--tune-rnn-layers",
        nargs="+",
        type=int,
        default=DEFAULT_TUNING_GRID["rnn_layers"],
    )
    parser.add_argument(
        "--tune-dropouts",
        nargs="+",
        type=float,
        default=DEFAULT_TUNING_GRID["dropout"],
    )
    parser.add_argument(
        "--tune-learning-rates",
        nargs="+",
        type=float,
        default=DEFAULT_TUNING_GRID["learning_rate"],
    )
    parser.add_argument(
        "--tune-batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TUNING_GRID["batch_size"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit_tuning_combinations is not None and args.limit_tuning_combinations < 1:
        raise ValueError("--limit-tuning-combinations must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device, strict_cuda=args.strict_cuda)
    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon_rows = []
    progress_path = args.output_dir / "hyperparameter_sweep_progress.csv"
    if progress_path.exists() and not args.skip_hyperparameter_search:
        try:
            all_sweep_rows: list[dict[str, object]] = pd.read_csv(
                progress_path
            ).to_dict("records")
            print(
                f"loaded {len(all_sweep_rows):,} completed tuning rows from "
                f"{progress_path}"
            )
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            all_sweep_rows = []
    else:
        all_sweep_rows = []
    all_best_rows: list[dict[str, object]] = []
    for frequency in frequencies:
        account_ids = args.account_ids
        if account_ids is None and args.max_accounts is not None:
            account_ids = get_available_enriched_account_ids(frequency, args.max_accounts)
        data = create_enriched_deepar_data(frequency, account_ids)

        if args.skip_hyperparameter_search:
            selected_params = {
                key: getattr(args, key)
                for key in TUNING_KEYS
            }
            selected_results = {
                flow_name: None
                for flow_name in FLOW_TARGETS
            }
            print(
                f"skipping DeepAR hyperparameter search for {frequency}; "
                f"using CLI/default values {selected_params}"
            )
        else:
            tuning_grid = make_tuning_grid(args)
            if args.limit_tuning_combinations is not None:
                tuning_grid = tuning_grid[: args.limit_tuning_combinations]
            print(
                f"running DeepAR validation search for {frequency}: "
                f"{len(tuning_grid)} combinations x {len(FLOW_TARGETS)} flows"
            )
            frequency_sweep_rows: list[dict[str, object]] = [
                row for row in all_sweep_rows
                if str(row.get("frequency")) == frequency
            ]
            effective_context_length = args.context_length or (
                12 if frequency == "monthly" else 52
            )
            completed_keys = {
                tuning_result_key(
                    str(row["flow"]),
                    {key: row[key] for key in TUNING_KEYS},
                    int(row.get("max_epochs", -1)),
                    int(row.get("patience", -1)),
                    int(row.get("context_length", -1)),
                )
                for row in frequency_sweep_rows
                if all(key in row for key in TUNING_KEYS)
                and "max_epochs" in row
                and "patience" in row
                and "context_length" in row
            }
            for flow_name in FLOW_TARGETS:
                for combination_index, params in enumerate(tuning_grid, start=1):
                    candidate_key = tuning_result_key(
                        flow_name,
                        params,
                        args.epochs,
                        args.patience,
                        effective_context_length,
                    )
                    if candidate_key in completed_keys:
                        print(
                            f"skipping completed tuning {frequency}/{flow_name}: "
                            f"{combination_index}/{len(tuning_grid)} {params}"
                        )
                        continue
                    print(
                        f"tuning {frequency}/{flow_name}: "
                        f"{combination_index}/{len(tuning_grid)} {params}"
                    )
                    row = validate_flow_hyperparameters(
                        data, flow_name, device, args, params
                    )
                    row["combination_id"] = combination_index
                    frequency_sweep_rows.append(row)
                    all_sweep_rows.append(row)
                    completed_keys.add(candidate_key)
                    pd.DataFrame(all_sweep_rows).to_csv(progress_path, index=False)

            frequency_sweep = pd.DataFrame(frequency_sweep_rows)
            frequency_sweep.to_csv(
                args.output_dir / f"{frequency}_hyperparameter_sweep.csv",
                index=False,
            )
            selected_results = {}
            selected_params = {}
            all_best_rows = [
                row for row in all_best_rows
                if str(row.get("frequency")) != frequency
            ]
            for flow_name in FLOW_TARGETS:
                flow_rows = frequency_sweep.loc[
                    frequency_sweep["flow"].eq(flow_name)
                ]
                winner = flow_rows.loc[flow_rows["validation_nll"].idxmin()]
                winner_dict = winner.to_dict()
                selected_results[flow_name] = winner_dict
                selected_params[flow_name] = {
                    key: winner_dict[key]
                    for key in TUNING_KEYS
                }
                all_best_rows.append(winner_dict)
            pd.DataFrame(all_best_rows).to_csv(
                args.output_dir / "best_hyperparameters_progress.csv",
                index=False,
            )

        models = {}
        training_rows = []
        for flow_name in FLOW_TARGETS:
            if args.skip_hyperparameter_search:
                model, info = train_flow_model(
                    data, flow_name, device, args,
                    params=selected_params,
                )
            else:
                model, info = train_flow_model(
                    data,
                    flow_name,
                    device,
                    args,
                    params=selected_params[flow_name],
                    validation_result=selected_results[flow_name],
                )
            info["selected_by"] = (
                "cli_values" if args.skip_hyperparameter_search else "validation_nll"
            )
            models[flow_name] = model
            training_rows.append(info)
        training_info = pd.DataFrame(training_rows)
        training_info.to_csv(
            args.output_dir / f"{frequency}_selected_training_info.csv",
            index=False,
        )
        predictions = recursive_forecast(data, models, training_info, device, args)
        run_dir, horizon_metrics = save_enriched_run_outputs(data, predictions, training_info, args.output_dir, "recursive", args.under_weight, args.over_weight)
        for flow_name, model in models.items():
            torch.save({"flow": flow_name, "frequency": frequency, "model_config": model.config, "state_dict": model.state_dict(), "dynamic_feature_names": data.dynamic_feature_names, "static_feature_names": data.static_feature_names}, run_dir / f"{flow_name}_model.pt")
        metadata = {
            "model": "DeepAR", "frequency": frequency, "method": "recursive",
            "master_file": str(enriched_master_path(frequency)),
            "account_id_embedding": False,
            "dynamic_feature_names": data.dynamic_feature_names,
            "static_feature_names": data.static_feature_names,
            "scheduled_loan_repayment": "account-specific known covariate; loan contracts masked as-of forecast origin",
            "scheduled_loan_repayment_dynamic_scale": data.schedule_dynamic_scale,
            "device": str(device), "n_accounts": len(data.account_ids),
            "hyperparameter_selection": (
                "CLI values" if args.skip_hyperparameter_search else "validation NLL"
            ),
            "selected_hyperparameters": selected_params,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        horizon_rows.append(horizon_metrics)
        print(f"completed enriched DeepAR {frequency}/recursive: {len(data.account_ids):,} accounts, device={device}")
    if not args.skip_hyperparameter_search:
        pd.DataFrame(all_sweep_rows).to_csv(
            args.output_dir / "hyperparameter_sweep.csv",
            index=False,
        )
        pd.DataFrame(all_best_rows).to_csv(
            args.output_dir / "best_hyperparameters.csv",
            index=False,
        )
    if horizon_rows:
        pd.concat(horizon_rows, ignore_index=True).to_csv(args.output_dir / "horizon_comparison_metrics.csv", index=False)


if __name__ == "__main__":
    main()
