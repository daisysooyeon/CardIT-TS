"""Leakage-safe account-level ARIMA forecasts for the Berka master tables.

Four configurations are supported:

* monthly + recursive one-step forecast
* monthly + direct multi-step forecast
* weekly + recursive one-step forecast
* weekly + direct multi-step forecast

Unlike the Linear Regression baseline, plain ARIMA is fitted separately for
each account and flow.  Inflow and outflow are modeled independently and net
flow is derived afterwards.  The recursive implementation refits ARIMA at
each test step after appending only the previous prediction; it never appends
the test-period actual value.

Run from the project root, for example:

    .\\ts\\Scripts\\python.exe arima_forecast.py --frequency monthly --method both --max-accounts 10
    .\\ts\\Scripts\\python.exe arima_forecast.py --frequency both --method both
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "arima"
DEFAULT_ORDER = (1, 1, 1)
# Exhaustive p,d,q grid requested for this experiment. Users can replace it
# with --orders when they want to run a smaller or custom candidate set.
DEFAULT_ORDERS = tuple(product(range(3), repeat=3))
MIN_HISTORY_LENGTH = 8


@dataclass
class FitOutcome:
    raw_forecast: np.ndarray
    result: Any | None
    fallback_used: bool
    converged: bool
    aic: float
    status: str
    message: str


def parse_order(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("ARIMA order must contain exactly three integers: p d q")
    order = tuple(int(value) for value in values)
    if any(value < 0 for value in order):
        raise ValueError(f"ARIMA order values must be non-negative: {order}")
    return order


def parse_order_spec(value: str) -> tuple[int, int, int]:
    """Parse one CLI order written as ``p,d,q``."""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid ARIMA order {value!r}; use p,d,q, for example 1,1,1"
        )
    try:
        return parse_order([int(part.strip()) for part in parts])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def order_label(order: tuple[int, int, int]) -> str:
    """Return a filesystem-safe, explicit label for one ARIMA order."""
    p, d, q = order
    return f"order_p{p}_d{d}_q{q}"


def validate_regular_series(
    group: pd.DataFrame,
    frequency: str,
    account_id: int,
) -> pd.Series:
    """Return one complete, chronological flow series for an account."""
    dates = pd.DatetimeIndex(pd.to_datetime(group["period_start"]).sort_values().unique())
    if len(dates) == 0:
        raise ValueError(f"Account {account_id} has no observations")

    if frequency == "monthly":
        expected = pd.date_range(dates[0], dates[-1], freq="MS")
    else:
        expected = pd.date_range(dates[0], dates[-1], freq="7D")
    if not dates.equals(expected):
        missing = expected.difference(dates)
        extra = dates.difference(expected)
        raise ValueError(
            f"Account {account_id} does not have a regular {frequency} series; "
            f"missing={list(missing[:5])}, extra={list(extra[:5])}"
        )

    series = (
        group.sort_values("period_start")
        .set_index("period_start")
        .iloc[:, 0]
        .astype(float)
    )
    if series.isna().any():
        raise ValueError(f"Account {account_id} contains missing flow values")
    series.index = pd.DatetimeIndex(series.index)
    return series


def make_pretest_histories(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    frequency: str,
) -> dict[int, dict[str, list[float]]]:
    """Build chronological pre-test histories for both flows."""
    pretest = df.loc[df["period_start"] < test_start].copy()
    histories: dict[int, dict[str, list[float]]] = {}
    for account_id, group in pretest.groupby("account_id", sort=True):
        account_id = int(account_id)
        histories[account_id] = {}
        for flow_name, value_column in FLOW_COLUMNS.items():
            flow_group = group[["period_start", value_column]].rename(
                columns={value_column: flow_name}
            )
            series = validate_regular_series(flow_group, frequency, account_id)
            histories[account_id][flow_name] = series.tolist()
    if not histories:
        raise ValueError("No pre-test account histories were found")
    return histories


def fallback_value(history: list[float]) -> float:
    if not history:
        return 0.0
    value = float(history[-1])
    return value if np.isfinite(value) else 0.0


def fit_and_forecast(
    history: list[float],
    order: tuple[int, int, int],
    steps: int,
) -> FitOutcome:
    """Fit one ARIMA model and return forecasts with a safe fallback."""
    values = np.asarray(history, dtype=float)
    fallback = fallback_value(history)
    if len(values) < max(MIN_HISTORY_LENGTH, sum(order) + 2):
        return FitOutcome(
            raw_forecast=np.full(steps, fallback, dtype=float),
            result=None,
            fallback_used=True,
            converged=False,
            aic=np.nan,
            status="fallback_short_history",
            message=f"history length {len(values)} is too short for order {order}",
        )

    if not np.isfinite(values).all():
        return FitOutcome(
            raw_forecast=np.full(steps, fallback, dtype=float),
            result=None,
            fallback_used=True,
            converged=False,
            aic=np.nan,
            status="fallback_nonfinite_history",
            message="history contains non-finite values",
        )

    # A constant series is a valid financial series but often produces a
    # singular ARIMA likelihood.  Its last value is the natural fallback.
    if np.nanstd(values) <= 1e-12:
        return FitOutcome(
            raw_forecast=np.full(steps, fallback, dtype=float),
            result=None,
            fallback_used=True,
            converged=True,
            aic=np.nan,
            status="fallback_constant_history",
            message="history is constant",
        )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ARIMA(
                values,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(method_kwargs={"maxiter": 100})
        raw_forecast = np.asarray(result.forecast(steps=steps), dtype=float)
        if not np.isfinite(raw_forecast).all():
            raise ValueError("ARIMA returned a non-finite forecast")

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
        return FitOutcome(
            raw_forecast=raw_forecast,
            result=result,
            fallback_used=False,
            converged=converged,
            aic=float(getattr(result, "aic", np.nan)),
            status=status,
            message=message,
        )
    except Exception as exc:  # noqa: BLE001 - one bad account must not stop all runs
        return FitOutcome(
            raw_forecast=np.full(steps, fallback, dtype=float),
            result=None,
            fallback_used=True,
            converged=False,
            aic=np.nan,
            status="fallback_fit_error",
            message=f"{type(exc).__name__}: {exc}",
        )


def build_prediction_record(
    account_id: int,
    target_date: pd.Timestamp,
    origin_period: pd.Timestamp,
    step: int,
    actual: dict[tuple[int, pd.Timestamp], dict[str, float]],
    raw_inflow: float,
    raw_outflow: float,
) -> dict[str, object]:
    predicted_inflow = float(max(raw_inflow, 0.0))
    predicted_outflow = float(max(raw_outflow, 0.0))
    return {
        "account_id": int(account_id),
        "period_start": target_date,
        "origin_period": origin_period,
        "horizon_step": step,
        "actual_inflow": actual[(int(account_id), target_date)]["inflow"],
        "actual_outflow": actual[(int(account_id), target_date)]["outflow"],
        "prediction_inflow_raw": float(raw_inflow),
        "prediction_inflow": predicted_inflow,
        "prediction_outflow_raw": float(raw_outflow),
        "prediction_outflow": predicted_outflow,
    }


def update_diagnostic(
    diagnostics: dict[tuple[int, str], dict[str, object]],
    account_id: int,
    flow_name: str,
    outcome: FitOutcome,
) -> None:
    key = (int(account_id), flow_name)
    row = diagnostics.setdefault(
        key,
        {
            "account_id": int(account_id),
            "target": flow_name,
            "n_fits": 0,
            "n_fallbacks": 0,
            "n_not_converged": 0,
            "last_aic": np.nan,
            "last_status": "",
            "last_message": "",
        },
    )
    row["n_fits"] = int(row["n_fits"]) + 1
    row["n_fallbacks"] = int(row["n_fallbacks"]) + int(outcome.fallback_used)
    row["n_not_converged"] = int(row["n_not_converged"]) + int(
        not outcome.converged
    )
    row["last_aic"] = outcome.aic
    row["last_status"] = outcome.status
    row["last_message"] = outcome.message


def run_threaded(
    function: Any,
    argument_tuples: list[tuple[Any, ...]],
    n_jobs: int,
) -> list[Any]:
    """Run independent account jobs without spawning Windows processes."""
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    workers = os.cpu_count() or 1 if n_jobs == -1 else n_jobs
    if workers == 1:
        return [function(*arguments) for arguments in argument_tuples]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(function, *arguments)
            for arguments in argument_tuples
        ]
        return [future.result() for future in futures]


def run_parallel(
    function: Any,
    argument_tuples: list[tuple[Any, ...]],
    n_jobs: int,
    backend: str,
) -> list[Any]:
    if backend == "threads":
        return run_threaded(function, argument_tuples, n_jobs)
    if backend != "processes":
        raise ValueError("parallel backend must be 'processes' or 'threads'")
    return Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(function)(*arguments)
        for arguments in argument_tuples
    )


def compact_result_artifact(
    result: Any,
    order: tuple[int, int, int],
) -> dict[str, object]:
    return {
        "order": order,
        "params": np.asarray(result.params, dtype=float),
        "aic": float(getattr(result, "aic", np.nan)),
    }


def recursive_forecast_one_account(
    account_id: int,
    history_by_flow: dict[str, list[float]],
    test_dates: list[pd.Timestamp],
    frequency: str,
    order: tuple[int, int, int],
    actual_for_account: dict[tuple[int, pd.Timestamp], dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    """Run the full recursive loop for one independent account."""
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
        outcomes: dict[str, FitOutcome] = {}
        for flow_name in FLOW_TARGETS:
            outcome = fit_and_forecast(
                working_history[flow_name],
                order=order,
                steps=1,
            )
            outcomes[flow_name] = outcome
            update_diagnostic(local_diagnostics, account_id, flow_name, outcome)
            if outcome.result is not None:
                final_artifacts[(account_id, flow_name)] = compact_result_artifact(
                    outcome.result,
                    order,
                )

        record = build_prediction_record(
            account_id=account_id,
            target_date=target_date,
            origin_period=origin_period,
            step=step,
            actual=actual_for_account,
            raw_inflow=float(outcomes["inflow"].raw_forecast[0]),
            raw_outflow=float(outcomes["outflow"].raw_forecast[0]),
        )
        records.append(record)

        # Only operational predictions are appended.  Test actuals are never
        # used as future inputs.
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
    """Refit one-step ARIMA models after each predicted test period."""
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
    raw_forecasts: dict[str, np.ndarray] = {}
    local_diagnostics: dict[tuple[int, str], dict[str, object]] = {}
    fitted_artifacts: dict[tuple[int, str], dict[str, object]] = {}
    for flow_name in FLOW_TARGETS:
        outcome = fit_and_forecast(
            history_by_flow[flow_name],
            order=order,
            steps=len(test_dates),
        )
        raw_forecasts[flow_name] = outcome.raw_forecast
        update_diagnostic(local_diagnostics, account_id, flow_name, outcome)
        if outcome.result is not None:
            fitted_artifacts[(account_id, flow_name)] = compact_result_artifact(
                outcome.result,
                order,
            )

    origin_period = test_dates[0] - FREQUENCY_CONFIG[frequency]["step"]
    records = []
    for step, target_date in enumerate(test_dates, start=1):
        records.append(
            build_prediction_record(
                account_id=account_id,
                target_date=target_date,
                origin_period=origin_period,
                step=step,
                actual=actual_for_account,
                raw_inflow=float(raw_forecasts["inflow"][step - 1]),
                raw_outflow=float(raw_forecasts["outflow"][step - 1]),
            )
        )
    return records, list(local_diagnostics.values()), fitted_artifacts


def direct_forecast(
    histories: dict[int, dict[str, list[float]]],
    test_dates: list[pd.Timestamp],
    frequency: str,
    order: tuple[int, int, int],
    actual: dict[tuple[int, pd.Timestamp], dict[str, float]],
    n_jobs: int,
    parallel_backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], dict[str, object]]]:
    """Fit once per account/flow and forecast the full horizon."""
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

    # Keep every ARIMA order isolated so predictions and fitted artifacts
    # cannot overwrite one another during a grid experiment.
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
            }
            for flow_name in FLOW_TARGETS
        ]
    )
    training_info.to_csv(run_dir / "training_info.csv", index=False)

    # Saving every full statsmodels result can be very large.  The forecast
    # functions therefore retain compact final fit artifacts with order,
    # parameters, AIC, and account/flow mapping.
    joblib.dump(fitted_results, run_dir / "model.joblib", compress=3)

    metadata = {
        "model": "ARIMA",
        "frequency": frequency,
        "method": method,
        "order": list(order),
        "n_jobs": n_jobs,
        "parallel_backend": parallel_backend,
        "horizon": config["horizon"],
        "test_start": str(test_start.date()),
        "test_dates": [str(date.date()) for date in test_dates],
        "n_accounts": len(histories),
        "targets": list(FLOW_TARGETS),
        "recursive_refit_each_test_step": method == "recursive",
        "recursive_uses_predictions_in_history": method == "recursive",
        "direct_fits_once_and_forecasts_full_horizon": method == "direct",
        "negative_forecasts_clipped_to_zero": True,
        "under_weight": under_weight,
        "over_weight": over_weight,
        "fallback_policy": "last history value for short/constant/failed series",
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
        help="Parallel backend. Processes are faster for ARIMA fitting; threads avoid process restrictions.",
    )
    parser.add_argument(
        "--order",
        nargs=3,
        type=int,
        metavar=("P", "D", "Q"),
        default=None,
        help="Run one ARIMA order as three separate values, e.g. --order 1 1 1.",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=parse_order_spec,
        default=None,
        metavar="P,D,Q",
        help=(
            "Run several orders written with commas, e.g. "
            "--orders 0,1,0 1,1,0 1,1,1. "
            "If omitted, the built-in DEFAULT_ORDERS grid is used."
        ),
    )
    parser.add_argument(
        "--under-weight",
        type=float,
        default=2.0,
        help="Penalty multiplier for underprediction in asymmetric metrics.",
    )
    parser.add_argument(
        "--over-weight",
        type=float,
        default=1.0,
        help="Penalty multiplier for overprediction in asymmetric metrics.",
    )
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
    horizon_rows: list[pd.DataFrame] = []
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
        # A rerun replaces only the selected order/frequency/method cells and
        # preserves completed results from other invocations.
        previous_comparison = previous_comparison.loc[
            ~(
                previous_comparison["order_label"].isin(selected_order_labels)
                & previous_comparison["frequency"].isin(frequencies)
                & previous_comparison["method"].isin(methods)
            )
        ]

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
                # Persist completed combinations immediately. This is useful
                # because a full all-account recursive grid can be long-running.
                pd.concat([previous_comparison, *horizon_rows], ignore_index=True).to_csv(
                    comparison_path,
                    index=False,
                )

    if horizon_rows:
        comparison = pd.concat([previous_comparison, *horizon_rows], ignore_index=True)
        comparison.to_csv(
            comparison_path,
            index=False,
        )


if __name__ == "__main__":
    main()
