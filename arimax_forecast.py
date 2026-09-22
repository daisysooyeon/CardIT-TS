"""Leakage-safe, non-seasonal ARIMAX forecasts for the Berka master tables.

This script extends the existing account-level ARIMA experiment with dynamic
exogenous variables.  It intentionally uses ``SARIMAX`` with
``seasonal_order=(0, 0, 0, 0)``: the seasonal component is disabled, so the
model is a non-seasonal ARIMAX.

Supported configurations:

* monthly + recursive one-step forecast
* monthly + direct fixed-parameter multi-step forecast
* weekly + recursive one-step forecast
* weekly + direct fixed-parameter multi-step forecast

Inflow and outflow are fitted separately and net flow is derived afterwards.
The exogenous table is rebuilt from each account's observed or predicted
history.  Existing precomputed feature columns are not read, because their
test-period values could contain future actual information.

The direct implementation fits one ARIMAX per account and flow.  Its dynamic
future exogenous rows are generated sequentially from the model predictions,
but the fitted ARIMAX parameters are not refit.  This preserves the intended
direct-vs-recursive distinction while avoiding leakage.

Example smoke run::

    .\\ts\\Scripts\\python.exe arimax_forecast.py \\
        --frequency monthly --method both --max-accounts 2 \\
        --orders 0,1,0 1,1,1 --output-dir outputs/arimax_smoke

Example full run for all accounts and the default p,d,q grid::

    .\\ts\\Scripts\\python.exe arimax_forecast.py \\
        --frequency both --method both --output-dir outputs/arimax
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arima_forecast import (
    build_prediction_record,
    fit_and_forecast as fit_plain_arima_and_forecast,
    make_pretest_histories,
    run_parallel,
    update_diagnostic,
)
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
    test_actual_lookup,
    train_normalization_scale,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "arimax"
DEFAULT_ORDERS = tuple(product(range(3), repeat=3))
MIN_HISTORY_LENGTH = 8
EXOG_EPSILON = 1e-12


@dataclass
class ForecastOutcome:
    """One fit/forecast outcome with the same diagnostic fields as ARIMA."""

    raw_forecast: np.ndarray
    result: Any | None
    fallback_used: bool
    converged: bool
    aic: float
    status: str
    message: str


@dataclass
class FittedModel:
    """A fitted ARIMAX model plus the transform needed for future exog rows."""

    result: Any | None
    state_result: Any | None
    scaler: StandardScaler | None
    exog_columns: list[str]
    model_type: str
    fallback_forecast: np.ndarray | None
    fallback_value: float
    fallback_used: bool
    converged: bool
    aic: float
    status: str
    message: str


def parse_order(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("ARIMAX order must contain exactly three integers: p d q")
    order = tuple(int(value) for value in values)
    if any(value < 0 for value in order):
        raise ValueError(f"ARIMAX order values must be non-negative: {order}")
    return order


def parse_order_spec(value: str) -> tuple[int, int, int]:
    """Parse one CLI order written as ``p,d,q``."""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid ARIMAX order {value!r}; use p,d,q, for example 1,1,1"
        )
    try:
        return parse_order([int(part.strip()) for part in parts])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def order_label(order: tuple[int, int, int]) -> str:
    p, d, q = order
    return f"order_p{p}_d{d}_q{q}"


def fallback_value(history: list[float]) -> float:
    if not history:
        return 0.0
    value = float(history[-1])
    return value if np.isfinite(value) else 0.0


def bagged_smooth(values: np.ndarray, windows: list[int]) -> float:
    """Return a causal ensemble of rolling means.

    This matches the meaning of the master-table ``*_bagged_smooth`` columns:
    it averages the rolling means over the configured windows.  The feature is
    calculated only from values available at the current forecast origin.
    """
    rolling_means = [
        float(np.mean(values[-window:]))
        for window in windows
        if len(values) >= window
    ]
    if len(rolling_means) != len(windows):
        return np.nan
    return float(np.mean(rolling_means))


def make_dynamic_feature_row(
    histories: dict[str, list[float]],
    frequency: str,
) -> pd.DataFrame:
    """Build one leakage-safe exog row from histories through the origin."""
    config = FREQUENCY_CONFIG[frequency]
    row: dict[str, float] = {}
    for flow_name in FLOW_COLUMNS:
        values = np.asarray(histories[flow_name], dtype=float)
        for lag in config["lags"]:
            name = f"{flow_name}_lag_{lag}"
            row[name] = (
                float(values[-1 - lag]) if len(values) > lag else np.nan
            )
        for window in config["windows"]:
            recent = values[-window:]
            if len(recent) < window:
                for operation in ("mean", "median", "std"):
                    row[f"{flow_name}_rolling_{operation}_{window}"] = np.nan
            else:
                row[f"{flow_name}_rolling_mean_{window}"] = float(
                    np.mean(recent)
                )
                row[f"{flow_name}_rolling_median_{window}"] = float(
                    np.median(recent)
                )
                row[f"{flow_name}_rolling_std_{window}"] = float(
                    np.std(recent, ddof=1)
                )
        row[f"{flow_name}_bagged_smooth"] = bagged_smooth(
            values,
            list(config["windows"]),
        )
    return pd.DataFrame([row])


def build_dynamic_feature_table(
    histories: dict[str, list[float]],
    frequency: str,
) -> pd.DataFrame:
    """Build origin features for every historical period of one account."""
    lengths = {len(values) for values in histories.values()}
    if len(lengths) != 1:
        raise ValueError(f"Flow histories have different lengths: {lengths}")
    n_periods = next(iter(lengths), 0)
    rows: list[pd.DataFrame] = []
    for end_index in range(n_periods):
        prefix = {
            flow_name: values[: end_index + 1]
            for flow_name, values in histories.items()
        }
        rows.append(make_dynamic_feature_row(prefix, frequency))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_training_arrays(
    history_by_flow: dict[str, list[float]],
    target_flow: str,
    frequency: str,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Align ``feature_origin_t`` with ``target_{t+1}`` without leakage."""
    feature_table = build_dynamic_feature_table(history_by_flow, frequency)
    target_values = np.asarray(history_by_flow[target_flow], dtype=float)
    if len(target_values) < 2:
        return np.array([], dtype=float), pd.DataFrame(), []

    # Row t contains information through t; it predicts the next observation.
    exog = feature_table.iloc[:-1].reset_index(drop=True)
    target = target_values[1:]
    valid = np.array(exog.notna().all(axis=1).to_numpy(), dtype=bool, copy=True)
    valid &= np.isfinite(target)
    exog = exog.loc[valid].reset_index(drop=True)
    target = target[valid]
    if exog.empty:
        return target, exog, []

    # A constant exog column cannot be identified separately from an
    # intercept in an account-level model.  Drop it per account/flow.
    variation = exog.std(axis=0, ddof=0).to_numpy(dtype=float)
    varying_columns = [
        column
        for column, standard_deviation in zip(exog.columns, variation)
        if np.isfinite(standard_deviation) and standard_deviation > EXOG_EPSILON
    ]
    return target, exog[varying_columns], varying_columns


def plain_fallback_model(
    history: list[float],
    order: tuple[int, int, int],
    steps: int,
    reason: str,
) -> tuple[ForecastOutcome, FittedModel]:
    """Retry plain ARIMA when ARIMAX cannot be fitted safely."""
    plain = fit_plain_arima_and_forecast(history, order=order, steps=steps)
    if plain.result is not None:
        status = "fallback_arima_only"
    else:
        status = plain.status
    outcome = ForecastOutcome(
        raw_forecast=np.asarray(plain.raw_forecast, dtype=float),
        result=plain.result,
        fallback_used=True,
        converged=plain.converged,
        aic=plain.aic,
        status=status,
        message=f"ARIMAX: {reason}; plain ARIMA: {plain.message}",
    )
    model = FittedModel(
        result=plain.result,
        state_result=plain.result,
        scaler=None,
        exog_columns=[],
        model_type="arima_fallback" if plain.result is not None else "last_value",
        fallback_forecast=np.asarray(plain.raw_forecast, dtype=float),
        fallback_value=fallback_value(history),
        fallback_used=True,
        converged=plain.converged,
        aic=plain.aic,
        status=status,
        message=outcome.message,
    )
    return outcome, model


def fit_arimax_model(
    history_by_flow: dict[str, list[float]],
    target_flow: str,
    frequency: str,
    order: tuple[int, int, int],
    fallback_steps: int,
) -> tuple[ForecastOutcome, FittedModel]:
    """Fit one account/flow ARIMAX model, with a plain-ARIMA fallback."""
    history = history_by_flow[target_flow]
    target, exog, exog_columns = build_training_arrays(
        history_by_flow,
        target_flow,
        frequency,
    )
    minimum_rows = max(MIN_HISTORY_LENGTH, sum(order) + 2)
    if len(target) < minimum_rows:
        return plain_fallback_model(
            history,
            order,
            fallback_steps,
            f"usable training rows {len(target)} < {minimum_rows}",
        )
    if not exog_columns:
        return plain_fallback_model(
            history,
            order,
            fallback_steps,
            "no varying dynamic exogenous columns",
        )
    if not np.isfinite(target).all() or not np.isfinite(exog.to_numpy()).all():
        return plain_fallback_model(
            history,
            order,
            fallback_steps,
            "non-finite training endog/exog",
        )

    scaler = StandardScaler()
    exog_scaled = scaler.fit_transform(exog[exog_columns])
    trend = "c" if order[1] == 0 else "n"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SARIMAX(
                endog=target,
                exog=exog_scaled,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(method_kwargs={"maxiter": 100}, disp=False)
        mle_retvals = getattr(result, "mle_retvals", {}) or {}
        converged = bool(mle_retvals.get("converged", True))
        warning_names = sorted(
            {
                type(item.message).__name__
                for item in caught
                if isinstance(item.message, Warning)
            }
        )
        status = "fit_converged" if converged else "fit_not_converged"
        message = "; ".join(warning_names)
        outcome = ForecastOutcome(
            raw_forecast=np.array([], dtype=float),
            result=result,
            fallback_used=False,
            converged=converged,
            aic=float(getattr(result, "aic", np.nan)),
            status=status,
            message=message,
        )
        fitted = FittedModel(
            result=result,
            state_result=result,
            scaler=scaler,
            exog_columns=exog_columns,
            model_type="arimax",
            fallback_forecast=None,
            fallback_value=fallback_value(history),
            fallback_used=False,
            converged=converged,
            aic=outcome.aic,
            status=status,
            message=message,
        )
        return outcome, fitted
    except Exception as exc:  # noqa: BLE001 - one account must not stop a run
        return plain_fallback_model(
            history,
            order,
            fallback_steps,
            f"SARIMAX fit error {type(exc).__name__}: {exc}",
        )


def forecast_fitted_model(
    fitted: FittedModel,
    future_exog: pd.DataFrame,
    step_index: int,
    update_state: bool,
) -> float:
    """Forecast one point, optionally advancing a fixed-parameter state."""
    if fitted.model_type != "arimax" or fitted.state_result is None:
        if fitted.fallback_forecast is not None and step_index < len(fitted.fallback_forecast):
            value = float(fitted.fallback_forecast[step_index])
        else:
            value = fitted.fallback_value
        return value if np.isfinite(value) else fitted.fallback_value

    try:
        exog_row = future_exog[fitted.exog_columns]
        if not np.isfinite(exog_row.to_numpy(dtype=float)).all():
            raise ValueError("future exog contains non-finite values")
        exog_scaled = fitted.scaler.transform(exog_row)
        result = fitted.state_result
        forecast = np.asarray(
            result.get_forecast(steps=1, exog=exog_scaled).predicted_mean,
            dtype=float,
        )
        if len(forecast) != 1 or not np.isfinite(forecast[0]):
            raise ValueError("SARIMAX returned a non-finite forecast")
        value = float(forecast[0])

        if update_state:
            # Append the operational (non-negative) prediction without
            # refitting. This updates the ARIMA state for the next horizon
            # step while keeping all fitted parameters fixed.
            operational_value = max(value, 0.0)
            fitted.state_result = result.append(
                [operational_value],
                exog=exog_scaled,
                refit=False,
            )
        return value
    except Exception as exc:  # noqa: BLE001 - preserve a usable forecast
        fitted.fallback_used = True
        fitted.status = "fallback_forecast_error"
        fitted.message = f"{type(exc).__name__}: {exc}"
        return fitted.fallback_value


def make_outcome_from_fitted(
    fitted: FittedModel,
    raw_forecast: list[float],
) -> ForecastOutcome:
    return ForecastOutcome(
        raw_forecast=np.asarray(raw_forecast, dtype=float),
        result=fitted.result,
        fallback_used=fitted.fallback_used,
        converged=fitted.converged,
        aic=fitted.aic,
        status=fitted.status,
        message=fitted.message,
    )


def compact_result_artifact(
    fitted: FittedModel,
    order: tuple[int, int, int],
) -> dict[str, object]:
    """Store compact, serializable diagnostics rather than full result objects."""
    artifact: dict[str, object] = {
        "model_type": fitted.model_type,
        "order": order,
        "seasonal_order": (0, 0, 0, 0),
        "exog_columns": list(fitted.exog_columns),
        "status": fitted.status,
        "message": fitted.message,
        "aic": fitted.aic,
    }
    if fitted.result is not None:
        artifact["params"] = np.asarray(fitted.result.params, dtype=float)
    if fitted.scaler is not None:
        artifact["exog_scaler_mean"] = np.asarray(fitted.scaler.mean_, dtype=float)
        artifact["exog_scaler_scale"] = np.asarray(fitted.scaler.scale_, dtype=float)
    return artifact


def recursive_forecast_one_account(
    account_id: int,
    history_by_flow: dict[str, list[float]],
    test_dates: list[pd.Timestamp],
    frequency: str,
    order: tuple[int, int, int],
    actual_for_account: dict[tuple[int, pd.Timestamp], dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    """Refit ARIMAX after each test prediction for one account."""
    working_history = {
        flow_name: list(values)
        for flow_name, values in history_by_flow.items()
    }
    local_diagnostics: dict[tuple[int, str], dict[str, object]] = {}
    final_artifacts: dict[tuple[int, str], dict[str, object]] = {}
    records: list[dict[str, object]] = []
    step_delta = FREQUENCY_CONFIG[frequency]["step"]

    for step, target_date in enumerate(test_dates, start=1):
        origin_period = test_dates[0] - step_delta + (step - 1) * step_delta
        origin_exog = make_dynamic_feature_row(working_history, frequency)
        raw_predictions: dict[str, float] = {}
        fitted_models: dict[str, FittedModel] = {}
        for flow_name in FLOW_TARGETS:
            _, fitted = fit_arimax_model(
                working_history,
                flow_name,
                frequency,
                order,
                fallback_steps=1,
            )
            raw_predictions[flow_name] = forecast_fitted_model(
                fitted,
                origin_exog,
                step_index=0,
                update_state=False,
            )
            fitted_models[flow_name] = fitted
            outcome = make_outcome_from_fitted(
                fitted,
                [raw_predictions[flow_name]],
            )
            update_diagnostic(local_diagnostics, account_id, flow_name, outcome)
            if fitted.result is not None:
                final_artifacts[(account_id, flow_name)] = compact_result_artifact(
                    fitted,
                    order,
                )

        record = build_prediction_record(
            account_id=account_id,
            target_date=target_date,
            origin_period=origin_period,
            step=step,
            actual=actual_for_account,
            raw_inflow=raw_predictions["inflow"],
            raw_outflow=raw_predictions["outflow"],
        )
        records.append(record)
        working_history["inflow"].append(float(record["prediction_inflow"]))
        working_history["outflow"].append(float(record["prediction_outflow"]))

    return records, list(local_diagnostics.values()), final_artifacts


def recursive_forecast(
    histories: dict[int, dict[str, list[float]]],
    test_dates: list[pd.Timestamp],
    frequency: str,
    order: tuple[int, int, int],
    actual: dict[tuple[int, pd.Timestamp], dict[str, float]],
    n_jobs: int,
    parallel_backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], dict[str, object]]]:
    """Run account-independent recursive ARIMAX forecasts in parallel."""
    account_ids = sorted(histories)
    arguments = [
        (
            account_id,
            histories[account_id],
            test_dates,
            frequency,
            order,
            {
                (account_id, target_date): actual[(account_id, target_date)]
                for target_date in test_dates
            },
        )
        for account_id in account_ids
    ]
    outputs = run_parallel(
        recursive_forecast_one_account,
        arguments,
        n_jobs,
        parallel_backend,
    )
    records = [record for output in outputs for record in output[0]]
    diagnostic_rows = [row for output in outputs for row in output[1]]
    final_artifacts = {
        key: artifact
        for output in outputs
        for key, artifact in output[2].items()
    }
    result = add_derived_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )
    return result, pd.DataFrame(diagnostic_rows), final_artifacts


def direct_forecast_one_account(
    account_id: int,
    history_by_flow: dict[str, list[float]],
    test_dates: list[pd.Timestamp],
    frequency: str,
    order: tuple[int, int, int],
    actual_for_account: dict[tuple[int, pd.Timestamp], dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    """Fit once per flow and forecast all horizons with fixed parameters."""
    fitted_models: dict[str, FittedModel] = {}
    local_diagnostics: dict[tuple[int, str], dict[str, object]] = {}
    final_artifacts: dict[tuple[int, str], dict[str, object]] = {}
    for flow_name in FLOW_TARGETS:
        _, fitted = fit_arimax_model(
            history_by_flow,
            flow_name,
            frequency,
            order,
            fallback_steps=len(test_dates),
        )
        fitted_models[flow_name] = fitted
        if fitted.result is not None:
            final_artifacts[(account_id, flow_name)] = compact_result_artifact(
                fitted,
                order,
            )

    working_history = {
        flow_name: list(values)
        for flow_name, values in history_by_flow.items()
    }
    raw_by_flow = {flow_name: [] for flow_name in FLOW_TARGETS}
    records: list[dict[str, object]] = []
    origin_period = test_dates[0] - FREQUENCY_CONFIG[frequency]["step"]
    for step, target_date in enumerate(test_dates, start=1):
        origin_exog = make_dynamic_feature_row(working_history, frequency)
        raw_predictions: dict[str, float] = {}
        for flow_name in FLOW_TARGETS:
            raw_value = forecast_fitted_model(
                fitted_models[flow_name],
                origin_exog,
                step_index=step - 1,
                update_state=True,
            )
            raw_predictions[flow_name] = raw_value
            raw_by_flow[flow_name].append(raw_value)

        record = build_prediction_record(
            account_id=account_id,
            target_date=target_date,
            origin_period=origin_period,
            step=step,
            actual=actual_for_account,
            raw_inflow=raw_predictions["inflow"],
            raw_outflow=raw_predictions["outflow"],
        )
        records.append(record)
        # Future feature rows use only operational predictions, never test
        # actuals. This is required even though the ARIMAX parameters are fixed.
        working_history["inflow"].append(float(record["prediction_inflow"]))
        working_history["outflow"].append(float(record["prediction_outflow"]))

    for flow_name in FLOW_TARGETS:
        fitted = fitted_models[flow_name]
        outcome = make_outcome_from_fitted(fitted, raw_by_flow[flow_name])
        update_diagnostic(local_diagnostics, account_id, flow_name, outcome)
        if fitted.result is not None:
            final_artifacts[(account_id, flow_name)] = compact_result_artifact(
                fitted,
                order,
            )
    return records, list(local_diagnostics.values()), final_artifacts


def direct_forecast(
    histories: dict[int, dict[str, list[float]]],
    test_dates: list[pd.Timestamp],
    frequency: str,
    order: tuple[int, int, int],
    actual: dict[tuple[int, pd.Timestamp], dict[str, float]],
    n_jobs: int,
    parallel_backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], dict[str, object]]]:
    """Run fixed-parameter direct forecasts in parallel by account."""
    account_ids = sorted(histories)
    arguments = [
        (
            account_id,
            histories[account_id],
            test_dates,
            frequency,
            order,
            {
                (account_id, target_date): actual[(account_id, target_date)]
                for target_date in test_dates
            },
        )
        for account_id in account_ids
    ]
    outputs = run_parallel(
        direct_forecast_one_account,
        arguments,
        n_jobs,
        parallel_backend,
    )
    records = [record for output in outputs for record in output[0]]
    diagnostic_rows = [row for output in outputs for row in output[1]]
    fitted_artifacts = {
        key: artifact
        for output in outputs
        for key, artifact in output[2].items()
    }
    result = add_derived_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )
    return result, pd.DataFrame(diagnostic_rows), fitted_artifacts


def metric_specs(under_weight: float, over_weight: float) -> list[tuple[str, ...]]:
    return [
        (
            "inflow",
            "actual_inflow",
            "prediction_inflow",
            "prediction_inflow_raw",
            "inflow_amount",
            over_weight,
            under_weight,
        ),
        (
            "outflow",
            "actual_outflow",
            "prediction_outflow",
            "prediction_outflow_raw",
            "outflow_amount",
            under_weight,
            over_weight,
        ),
        (
            "net_flow",
            "actual_net_flow",
            "prediction_net_flow",
            "prediction_net_flow_raw",
            "net_flow_for_scale",
            over_weight,
            under_weight,
        ),
    ]


def evaluate_and_save(
    predictions: pd.DataFrame,
    data_df: pd.DataFrame,
    test_start: pd.Timestamp,
    horizon: int,
    frequency: str,
    method: str,
    order: tuple[int, int, int],
    run_dir: Path,
    under_weight: float,
    over_weight: float,
) -> pd.DataFrame:
    """Save predictions and the same metric files used by ARIMA/LR/XGBoost."""
    predictions = predictions.copy()
    predictions.insert(0, "frequency", frequency)
    predictions.insert(1, "method", method)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    scale_df = data_df.copy()
    scale_df["net_flow_for_scale"] = (
        scale_df["inflow_amount"] - scale_df["outflow_amount"]
    )
    specs = metric_specs(under_weight, over_weight)
    period_metric_rows: list[pd.DataFrame] = []
    for (
        target,
        actual_column,
        prediction_column,
        raw_column,
        scale_column,
        target_under_weight,
        target_over_weight,
    ) in specs:
        scale = train_normalization_scale(
            df=scale_df,
            test_start=test_start,
            value_column=scale_column,
        )
        period_metric_rows.append(
            calculate_metrics(
                predictions,
                scale,
                actual_column=actual_column,
                prediction_column=prediction_column,
                raw_prediction_column=raw_column,
                under_weight=target_under_weight,
                over_weight=target_over_weight,
            ).assign(
                target=target,
                frequency=frequency,
                method=method,
                order=str(order),
                order_label=order_label(order),
            )
        )
    pd.concat(period_metric_rows, ignore_index=True).to_csv(
        run_dir / "metrics.csv",
        index=False,
    )

    horizon_predictions = (
        predictions.groupby("account_id", as_index=False)[
            [
                "actual_inflow",
                "actual_outflow",
                "prediction_inflow",
                "prediction_inflow_raw",
                "prediction_outflow",
                "prediction_outflow_raw",
            ]
        ]
        .sum()
    )
    horizon_predictions = add_derived_flow_columns(horizon_predictions)
    horizon_predictions["horizon_step"] = "horizon_total"
    horizon_metric_rows: list[pd.DataFrame] = []
    for (
        target,
        actual_column,
        prediction_column,
        raw_column,
        scale_column,
        target_under_weight,
        target_over_weight,
    ) in specs:
        scale = train_normalization_scale(
            df=scale_df,
            test_start=test_start,
            value_column=scale_column,
        )
        horizon_metric_rows.append(
            calculate_metrics(
                horizon_predictions,
                scale,
                actual_column=actual_column,
                prediction_column=prediction_column,
                raw_prediction_column=raw_column,
                under_weight=target_under_weight,
                over_weight=target_over_weight,
                scale_multiplier=horizon,
            ).assign(
                target=target,
                frequency=frequency,
                method=method,
                order=str(order),
                order_label=order_label(order),
                horizon_periods=horizon,
                under_weight=target_under_weight,
                over_weight=target_over_weight,
            )
        )
    horizon_metrics = pd.concat(horizon_metric_rows, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[
        horizon_metrics["horizon_step"].eq("horizon_total")
    ]
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)
    save_forecast_plot(
        predictions,
        frequency,
        method,
        run_dir / "aggregate_test_forecast.png",
    )
    return horizon_metrics


def run_one(
    frequency: str,
    method: str,
    output_dir: Path,
    account_ids: list[int] | None,
    order: tuple[int, int, int],
    n_jobs: int,
    parallel_backend: str,
    under_weight: float,
    over_weight: float,
) -> pd.DataFrame:
    config = FREQUENCY_CONFIG[frequency]
    df = read_master(frequency, account_ids)
    test_start = derive_test_start(frequency, df)
    all_dates = sorted(pd.Timestamp(value) for value in df["period_start"].unique())
    test_dates = [date for date in all_dates if date >= test_start][: config["horizon"]]
    if len(test_dates) != config["horizon"]:
        raise ValueError(
            f"Expected {config['horizon']} test dates, found {len(test_dates)}"
        )

    histories = make_pretest_histories(df, test_start, frequency)
    actual = test_actual_lookup(df, test_dates)
    if method == "recursive":
        predictions, diagnostics, fitted_results = recursive_forecast(
            histories,
            test_dates,
            frequency,
            order,
            actual,
            n_jobs,
            parallel_backend,
        )
    else:
        predictions, diagnostics, fitted_results = direct_forecast(
            histories,
            test_dates,
            frequency,
            order,
            actual,
            n_jobs,
            parallel_backend,
        )

    run_dir = output_dir / order_label(order) / f"{frequency}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    horizon_metrics = evaluate_and_save(
        predictions=predictions,
        data_df=df,
        test_start=test_start,
        horizon=config["horizon"],
        frequency=frequency,
        method=method,
        order=order,
        run_dir=run_dir,
        under_weight=under_weight,
        over_weight=over_weight,
    )

    diagnostics = diagnostics.copy()
    if not diagnostics.empty:
        diagnostics.insert(0, "frequency", frequency)
        diagnostics.insert(1, "method", method)
        diagnostics["order"] = str(order)
    diagnostics.to_csv(run_dir / "fit_diagnostics.csv", index=False)

    training_info = pd.DataFrame(
        [
            {
                "target": flow_name,
                "frequency": frequency,
                "method": method,
                "order": str(order),
                "n_accounts": len(histories),
                "n_fits": int(
                    diagnostics.loc[diagnostics["target"].eq(flow_name), "n_fits"].sum()
                )
                if not diagnostics.empty
                else 0,
                "n_fallbacks": int(
                    diagnostics.loc[
                        diagnostics["target"].eq(flow_name), "n_fallbacks"
                    ].sum()
                )
                if not diagnostics.empty
                else 0,
                "horizon": config["horizon"],
                "n_dynamic_exog_candidates": len(
                    FLOW_TARGETS
                    * (
                        len(config["lags"])
                        + 3 * len(config["windows"])
                        + 1
                    )
                ),
            }
            for flow_name in FLOW_TARGETS
        ]
    )
    training_info.to_csv(run_dir / "training_info.csv", index=False)
    joblib.dump(fitted_results, run_dir / "model.joblib", compress=3)

    metadata = {
        "model": "ARIMAX",
        "implementation": "statsmodels.SARIMAX",
        "frequency": frequency,
        "method": method,
        "order": list(order),
        "seasonal_order": [0, 0, 0, 0],
        "n_jobs": n_jobs,
        "parallel_backend": parallel_backend,
        "horizon": config["horizon"],
        "test_start": str(test_start.date()),
        "test_dates": [str(date.date()) for date in test_dates],
        "n_accounts": len(histories),
        "targets": list(FLOW_TARGETS),
        "dynamic_exog_lags": list(config["lags"]),
        "dynamic_exog_windows": list(config["windows"]),
        "dynamic_exog_operations": ["mean", "median", "std", "bagged_smooth"],
        "static_features_used": [],
        "training_alignment": "origin_features_at_t_to_target_at_t_plus_1",
        "recursive_refit_each_test_step": method == "recursive",
        "recursive_uses_predictions_in_history": True,
        "direct_fits_once_with_fixed_parameters": method == "direct",
        "direct_future_exog_generated_from_predictions": method == "direct",
        "exog_standardization": "StandardScaler fitted on each account-flow training exog",
        "negative_forecasts_clipped_to_zero": True,
        "under_weight": under_weight,
        "over_weight": over_weight,
        "fallback_policy": "ARIMAX -> plain ARIMA -> last history value",
        "model_joblib_contents": "compact final fit artifacts, not full statsmodels results",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    fallback_fits = int(diagnostics["n_fallbacks"].sum()) if not diagnostics.empty else 0
    print(
        f"completed {frequency}/{method}: {len(histories):,} accounts, "
        f"horizon={config['horizon']}, order={order}, fallback_fits={fallback_fits:,}"
    )
    return horizon_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for independent account series; -1 uses all available CPUs.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["processes", "threads"],
        default="processes",
        help="Processes are usually faster; threads can avoid Windows process restrictions.",
    )
    parser.add_argument(
        "--order",
        nargs=3,
        type=int,
        metavar=("P", "D", "Q"),
        default=None,
        help="Run one ARIMAX order, e.g. --order 1 1 1.",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=parse_order_spec,
        default=None,
        metavar="P,D,Q",
        help="Run orders such as --orders 0,1,0 1,1,1.",
    )
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.order is not None and args.orders is not None:
        parser.error("use either --order or --orders, not both")
    if args.orders is not None:
        orders = list(dict.fromkeys(args.orders))
    elif args.order is not None:
        orders = [parse_order(args.order)]
    else:
        orders = list(DEFAULT_ORDERS)

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tested_orders = pd.DataFrame(
        {
            "order": [str(order) for order in orders],
            "order_label": [order_label(order) for order in orders],
        }
    )
    orders_path = args.output_dir / "orders_tested.csv"
    if orders_path.exists():
        previous_orders = pd.read_csv(orders_path)
        tested_orders = pd.concat([previous_orders, tested_orders], ignore_index=True)
        tested_orders = tested_orders.drop_duplicates("order_label", keep="last")
    tested_orders.to_csv(orders_path, index=False)

    comparison_path = args.output_dir / "horizon_comparison_metrics.csv"
    previous_comparison = (
        pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    )
    selected_order_labels = {order_label(order) for order in orders}
    if not previous_comparison.empty:
        previous_comparison = previous_comparison.loc[
            ~(
                previous_comparison["order_label"].isin(selected_order_labels)
                & previous_comparison["frequency"].isin(frequencies)
                & previous_comparison["method"].isin(methods)
            )
        ]

    horizon_rows: list[pd.DataFrame] = []
    for order in orders:
        for frequency in frequencies:
            account_ids = args.account_ids
            if account_ids is None and args.max_accounts is not None:
                account_ids = get_available_account_ids(frequency, args.max_accounts)
            for method in methods:
                horizon_rows.append(
                    run_one(
                        frequency=frequency,
                        method=method,
                        output_dir=args.output_dir,
                        account_ids=account_ids,
                        order=order,
                        n_jobs=args.n_jobs,
                        parallel_backend=args.parallel_backend,
                        under_weight=args.under_weight,
                        over_weight=args.over_weight,
                    )
                )
                pd.concat(
                    [previous_comparison, *horizon_rows],
                    ignore_index=True,
                ).to_csv(comparison_path, index=False)

    if horizon_rows:
        pd.concat([previous_comparison, *horizon_rows], ignore_index=True).to_csv(
            comparison_path,
            index=False,
        )


if __name__ == "__main__":
    main()
