"""Shared fixed-cost-plus-residual forecasting runner.

The residual models reuse the enriched master, calendar features, leakage-safe
history features, direct/recursive forecast conventions, and model factories
from the existing implementations.  The deterministic baseline is currently
loan repayment only:

    absolute outflow = scheduled loan repayment + predicted outflow residual
    absolute inflow = predicted inflow residual
    net flow = absolute inflow - absolute outflow

Loan schedules are masked by the contract's loan date at each training origin.
During recursive forecasting only the final absolute predictions are appended
to the flow histories.  Test-period actual values are never used as inputs.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fixed_cost_schedule import (
    SCHEDULE_COLUMN,
    fixed_cost_baseline,
    fixed_cost_net_baseline,
    fixed_cost_policy,
)
from enriched_forecast_common import (
    EnrichedForecastData,
    create_enriched_forecast_data,
    enriched_master_path,
    get_available_enriched_account_ids,
    make_history_by_account,
    make_origin_features,
    static_by_account,
    test_actual_lookup,
)
from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    add_derived_flow_columns,
    calculate_metrics,
    make_pipeline,
    save_coefficients,
    save_forecast_plot,
    target_date_columns,
    train_normalization_scale,
)


TREE_GRID_KEYS = {
    "xgboost": (
        "max_depth",
        "n_estimators",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    ),
    "lightgbm": (
        "max_depth",
        "num_leaves",
        "n_estimators",
        "learning_rate",
        "min_child_samples",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "min_split_gain",
    ),
    "random_forest": (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ),
}

TREE_DEFAULT_GRIDS: dict[str, dict[str, list[float | int]]] = {
    "xgboost": {
        "max_depth": [2, 4, 6],
        "n_estimators": [200, 500],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1],
        "subsample": [0.9],
        "colsample_bytree": [0.9],
        "reg_lambda": [1.0],
        "reg_alpha": [0.0],
        "gamma": [0.0],
    },
    "lightgbm": {
        "max_depth": [4],
        "num_leaves": [4],
        "n_estimators": [500],
        "learning_rate": [0.1],
        "min_child_samples": [50],
        "min_child_weight": [0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.0, 1.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "min_split_gain": [0.0, 0.1, 0.5],
    },
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [8, 16],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
        "max_features": [0.7, 1.0],
    },
}


def _make_model(
    model_name: str,
    data: EnrichedForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
):
    if model_name == "linear_regression":
        return make_pipeline(data.numeric_columns, data.categorical_columns)
    if model_name == "xgboost":
        from xgboost_forecast import make_xgb_pipeline

        return make_xgb_pipeline(data, params, n_jobs, random_state)
    if model_name == "lightgbm":
        from lightgbm_forecast import make_lgbm_pipeline

        return make_lgbm_pipeline(data, params, n_jobs, random_state, device_type)
    if model_name == "random_forest":
        from random_forest_forecast_enriched import make_rf_pipeline

        return make_rf_pipeline(data, params, n_jobs, random_state)
    raise ValueError(f"Unknown residual model: {model_name}")


def _training_mask(data: EnrichedForecastData, horizon_index: int) -> pd.Series:
    """Return a leakage-safe mask shared by both residual targets."""
    features = data.feature_tables[horizon_index]
    mask = features[data.feature_columns].notna().all(axis=1)
    for value_column in FLOW_COLUMNS.values():
        _, target_dates = target_date_columns(
            data.df,
            horizon=horizon_index + 1,
            value_column=value_column,
        )
        target_values, _ = target_date_columns(
            data.df,
            horizon=horizon_index + 1,
            value_column=value_column,
        )
        mask &= target_values[horizon_index].notna()
        mask &= target_dates[horizon_index] < data.test_start
    return mask


def _residual_target(
    data: EnrichedForecastData,
    horizon_index: int,
    flow_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Return target-period residual values and dates for one flow."""
    value_column = FLOW_COLUMNS[flow_name]
    values, dates = target_date_columns(
        data.df,
        horizon=horizon_index + 1,
        value_column=value_column,
    )
    values = values[horizon_index].astype(float)
    dates = dates[horizon_index]
    known = data.feature_tables[horizon_index]
    baseline = fixed_cost_baseline(known, flow_name)
    return values - baseline, dates


def _net_residual_scale(data: EnrichedForecastData) -> pd.Series:
    """Compute account-level training scale for residual net flow."""
    inflow, dates = target_date_columns(data.df, 1, FLOW_COLUMNS["inflow"])
    outflow, _ = target_date_columns(data.df, 1, FLOW_COLUMNS["outflow"])
    schedule = data.feature_tables[0][SCHEDULE_COLUMN].astype(float)
    residual = inflow[0].astype(float) - outflow[0].astype(float) + schedule
    mask = dates[0] < data.test_start
    scale = residual.loc[mask].abs().groupby(data.df.loc[mask, "account_id"]).mean()
    return scale.where(scale > 0).clip(lower=1.0)


def residual_normalization_scales(data: EnrichedForecastData) -> dict[str, pd.Series]:
    """Return pre-test account scales for residual metric normalization."""
    inflow_residual, inflow_dates = _residual_target(data, 0, "inflow")
    outflow_residual, outflow_dates = _residual_target(data, 0, "outflow")
    inflow_mask = inflow_dates < data.test_start
    outflow_mask = outflow_dates < data.test_start
    inflow_scale = (
        inflow_residual.loc[inflow_mask]
        .abs()
        .groupby(data.df.loc[inflow_mask, "account_id"])
        .mean()
    )
    outflow_scale = (
        outflow_residual.loc[outflow_mask]
        .abs()
        .groupby(data.df.loc[outflow_mask, "account_id"])
        .mean()
    )
    return {
        "inflow": inflow_scale.where(inflow_scale > 0).clip(lower=1.0),
        "outflow": outflow_scale.where(outflow_scale > 0).clip(lower=1.0),
        "net_flow": _net_residual_scale(data),
    }


def fit_recursive_model(
    model_name: str,
    data: EnrichedForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one residual model per flow for one-step recursive forecasting."""
    mask = _training_mask(data, 0)
    X_train = data.feature_tables[0].loc[mask, data.feature_columns]
    if X_train.empty:
        raise ValueError(f"No training rows for residual {model_name}")

    models: dict[str, Any] = {}
    rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        y_train, _ = _residual_target(data, 0, flow_name)
        model = _make_model(
            model_name, data, params, n_jobs, random_state, device_type
        )
        model.fit(X_train, y_train.loc[mask].astype(float))
        models[flow_name] = model
        rows.append(
            {
                "target": f"{flow_name}_residual",
                "n_train_rows": int(len(X_train)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_models": 1,
                "known_features": ",".join(data.known_feature_names),
                "fixed_cost_baseline": "loan.payments -> scheduled_loan_repayment for outflow only",
            }
        )
    return models, pd.DataFrame(rows)


def fit_direct_model(
    model_name: str,
    data: EnrichedForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
) -> tuple[dict[str, list[Any]], pd.DataFrame]:
    """Fit one residual model per flow and future horizon."""
    models: dict[str, list[Any]] = {}
    rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        flow_models: list[Any] = []
        counts: list[int] = []
        for horizon_index in range(data.horizon):
            mask = _training_mask(data, horizon_index)
            X_train = data.feature_tables[horizon_index].loc[
                mask, data.feature_columns
            ]
            y_train, _ = _residual_target(data, horizon_index, flow_name)
            if X_train.empty:
                raise ValueError(
                    f"No training rows for residual {flow_name}, "
                    f"horizon={horizon_index + 1}"
                )
            model = _make_model(
                model_name, data, params, n_jobs, random_state, device_type
            )
            model.fit(X_train, y_train.loc[mask].astype(float))
            flow_models.append(model)
            counts.append(int(len(X_train)))
        models[flow_name] = flow_models
        rows.append(
            {
                "target": f"{flow_name}_residual",
                "n_train_rows_min": int(min(counts)),
                "n_train_rows_max": int(max(counts)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_models": int(data.horizon),
                "known_features": ",".join(data.known_feature_names),
                "fixed_cost_baseline": "loan.payments -> scheduled_loan_repayment for outflow only",
            }
        )
    return models, pd.DataFrame(rows)


def _residual_record(
    account_id: int,
    target_date: pd.Timestamp,
    origin_period: pd.Timestamp,
    step: int,
    actual: dict[str, float],
    fixed_inflow: float,
    fixed_outflow: float,
    raw_inflow_residual: float,
    raw_outflow_residual: float,
) -> dict[str, object]:
    raw_inflow = fixed_inflow + raw_inflow_residual
    raw_outflow = fixed_outflow + raw_outflow_residual
    prediction_inflow = max(raw_inflow, 0.0)
    prediction_outflow = max(raw_outflow, 0.0)
    actual_inflow_residual = actual["inflow"] - fixed_inflow
    actual_outflow_residual = actual["outflow"] - fixed_outflow
    return {
        "account_id": int(account_id),
        "period_start": target_date,
        "origin_period": origin_period,
        "horizon_step": step,
        "fixed_cost_inflow": fixed_inflow,
        "fixed_cost_outflow": fixed_outflow,
        "fixed_cost_net_flow": fixed_inflow - fixed_outflow,
        "actual_inflow": actual["inflow"],
        "actual_outflow": actual["outflow"],
        "actual_inflow_residual": actual_inflow_residual,
        "actual_outflow_residual": actual_outflow_residual,
        "prediction_inflow_raw": raw_inflow,
        "prediction_outflow_raw": raw_outflow,
        "prediction_inflow": prediction_inflow,
        "prediction_outflow": prediction_outflow,
        "prediction_inflow_residual_raw": raw_inflow_residual,
        "prediction_outflow_residual_raw": raw_outflow_residual,
        "prediction_inflow_residual": prediction_inflow - fixed_inflow,
        "prediction_outflow_residual": prediction_outflow - fixed_outflow,
    }


def add_residual_flow_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and residual net-flow columns to forecast records."""
    result = predictions.copy()
    result["actual_net_flow"] = result["actual_inflow"] - result["actual_outflow"]
    result["prediction_net_flow_raw"] = (
        result["prediction_inflow_raw"] - result["prediction_outflow_raw"]
    )
    result["prediction_net_flow"] = (
        result["prediction_inflow"] - result["prediction_outflow"]
    )
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
    # Compatibility with the original outflow readers.
    result["prediction_raw"] = result["prediction_outflow_raw"]
    result["prediction"] = result["prediction_outflow"]
    return result


def recursive_forecast(
    data: EnrichedForecastData,
    models: dict[str, Any],
) -> pd.DataFrame:
    """Forecast sequentially and append only final predictions to history."""
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    account_ids = sorted(histories)
    records: list[dict[str, object]] = []
    step_delta = FREQUENCY_CONFIG[data.frequency]["step"]

    for step, target_date in enumerate(data.test_dates, start=1):
        known = data.test_known_features[target_date]
        X_origin = make_origin_features(
            histories,
            static,
            data.frequency,
            target_date,
            known,
            data.include_history_features,
        )[data.feature_columns]
        raw_inflow_residual = np.asarray(
            models["inflow"].predict(X_origin), dtype=float
        )
        raw_outflow_residual = np.asarray(
            models["outflow"].predict(X_origin), dtype=float
        )
        fixed_inflow = fixed_cost_baseline(known, "inflow").to_numpy(dtype=float)
        fixed_outflow = fixed_cost_baseline(known, "outflow").to_numpy(dtype=float)

        for index, account_id in enumerate(account_ids):
            account_id = int(account_id)
            record = _residual_record(
                account_id=account_id,
                target_date=target_date,
                origin_period=target_date - step_delta,
                step=step,
                actual=actual[(account_id, target_date)],
                fixed_inflow=float(fixed_inflow[index]),
                fixed_outflow=float(fixed_outflow[index]),
                raw_inflow_residual=float(raw_inflow_residual[index]),
                raw_outflow_residual=float(raw_outflow_residual[index]),
            )
            records.append(record)
            # Future lag/rolling features use the final absolute operational
            # prediction, never the residual alone and never test actuals.
            histories[account_id]["inflow"].append(
                float(record["prediction_inflow"])
            )
            histories[account_id]["outflow"].append(
                float(record["prediction_outflow"])
            )

    return add_residual_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )


def direct_forecast(
    data: EnrichedForecastData,
    models: dict[str, list[Any]],
) -> pd.DataFrame:
    """Forecast all future periods from one origin with horizon models."""
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    account_ids = sorted(histories)
    records: list[dict[str, object]] = []
    origin_period = data.test_dates[0] - FREQUENCY_CONFIG[data.frequency]["step"]

    for step, target_date in enumerate(data.test_dates, start=1):
        known = data.test_known_features[target_date]
        X_origin = make_origin_features(
            histories,
            static,
            data.frequency,
            target_date,
            known,
            data.include_history_features,
        )[data.feature_columns]
        raw_inflow_residual = np.asarray(
            models["inflow"][step - 1].predict(X_origin), dtype=float
        )
        raw_outflow_residual = np.asarray(
            models["outflow"][step - 1].predict(X_origin), dtype=float
        )
        fixed_inflow = fixed_cost_baseline(known, "inflow").to_numpy(dtype=float)
        fixed_outflow = fixed_cost_baseline(known, "outflow").to_numpy(dtype=float)

        for index, account_id in enumerate(account_ids):
            account_id = int(account_id)
            records.append(
                _residual_record(
                    account_id=account_id,
                    target_date=target_date,
                    origin_period=origin_period,
                    step=step,
                    actual=actual[(account_id, target_date)],
                    fixed_inflow=float(fixed_inflow[index]),
                    fixed_outflow=float(fixed_outflow[index]),
                    raw_inflow_residual=float(raw_inflow_residual[index]),
                    raw_outflow_residual=float(raw_outflow_residual[index]),
                )
            )

    return add_residual_flow_columns(pd.DataFrame(records))


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


def _metric_weights(
    target: str,
    under_weight: float = 2.0,
    over_weight: float = 1.0,
) -> tuple[float, float]:
    """Use the same asymmetric evaluation convention as the existing models."""
    if target in {"inflow", "net_flow"}:
        return over_weight, under_weight
    return under_weight, over_weight


def _absolute_scales(data: EnrichedForecastData) -> dict[str, pd.Series]:
    scale_df = data.df.copy()
    scale_df["net_flow"] = scale_df["inflow_amount"] - scale_df["outflow_amount"]
    return {
        "inflow": train_normalization_scale(data.df, data.test_start, "inflow_amount"),
        "outflow": train_normalization_scale(data.df, data.test_start, "outflow_amount"),
        "net_flow": train_normalization_scale(scale_df, data.test_start, "net_flow"),
    }


def _write_metrics(
    predictions: pd.DataFrame,
    data: EnrichedForecastData,
    run_dir: Path,
    grid_id: str | None,
    under_weight: float,
    over_weight: float,
) -> None:
    absolute_scales = _absolute_scales(data)
    residual_scales = residual_normalization_scales(data)
    all_rows: list[pd.DataFrame] = []
    for is_residual, scales in ((False, absolute_scales), (True, residual_scales)):
        metric_rows: list[pd.DataFrame] = []
        for target, actual_col, prediction_col, raw_col, scale_key in _metric_specs(is_residual):
            target_under_weight, target_over_weight = _metric_weights(
                target, under_weight, over_weight
            )
            metric_rows.append(
                calculate_metrics(
                    predictions,
                    scales[scale_key],
                    actual_column=actual_col,
                    prediction_column=prediction_col,
                    raw_prediction_column=raw_col,
                    under_weight=target_under_weight,
                    over_weight=target_over_weight,
                ).assign(
                    target=target,
                    evaluation_target="residual" if is_residual else "absolute",
                    **({"grid_id": grid_id} if grid_id is not None else {}),
                )
            )
        metrics = pd.concat(metric_rows, ignore_index=True)
        metrics.to_csv(
            run_dir / ("residual_metrics.csv" if is_residual else "metrics.csv"),
            index=False,
        )

        horizon_columns = [
            "actual_inflow",
            "actual_outflow",
            "prediction_inflow",
            "prediction_inflow_raw",
            "prediction_outflow",
            "prediction_outflow_raw",
            "fixed_cost_inflow",
            "fixed_cost_outflow",
            "fixed_cost_net_flow",
            "actual_inflow_residual",
            "actual_outflow_residual",
            "prediction_inflow_residual",
            "prediction_inflow_residual_raw",
            "prediction_outflow_residual",
            "prediction_outflow_residual_raw",
        ]
        horizon_predictions = predictions.groupby("account_id", as_index=False)[
            horizon_columns
        ].sum()
        horizon_predictions = add_residual_flow_columns(horizon_predictions)
        horizon_predictions["horizon_step"] = "horizon_total"
        horizon_rows: list[pd.DataFrame] = []
        for target, actual_col, prediction_col, raw_col, scale_key in _metric_specs(is_residual):
            target_under_weight, target_over_weight = _metric_weights(
                target, under_weight, over_weight
            )
            horizon_rows.append(
                calculate_metrics(
                    horizon_predictions,
                    scales[scale_key],
                    actual_column=actual_col,
                    prediction_column=prediction_col,
                    raw_prediction_column=raw_col,
                    under_weight=target_under_weight,
                    over_weight=target_over_weight,
                    scale_multiplier=data.horizon,
                ).assign(
                    target=target,
                    evaluation_target="residual" if is_residual else "absolute",
                    horizon_periods=data.horizon,
                    **({"grid_id": grid_id} if grid_id is not None else {}),
                )
            )
        pd.concat(horizon_rows, ignore_index=True).loc[
            lambda frame: frame["horizon_step"].astype(str).eq("horizon_total")
        ].to_csv(
            run_dir
            / ("residual_horizon_metrics.csv" if is_residual else "horizon_metrics.csv"),
            index=False,
        )


def save_residual_run_outputs(
    data: EnrichedForecastData,
    predictions: pd.DataFrame,
    training_info: pd.DataFrame,
    output_dir: Path,
    method: str,
    model_name: str,
    params: dict[str, float | int],
    grid_id: str | None = None,
    under_weight: float = 2.0,
    over_weight: float = 1.0,
) -> tuple[Path, pd.DataFrame]:
    """Save absolute and residual predictions, metrics, and model metadata."""
    run_dir = output_dir / f"{data.frequency}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = predictions.copy()
    output.insert(0, "frequency", data.frequency)
    output.insert(1, "method", method)
    if grid_id is not None:
        output.insert(2, "grid_id", grid_id)
    output.to_csv(run_dir / "predictions.csv", index=False)
    output.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_metrics(output, data, run_dir, grid_id, under_weight, over_weight)
    training_info.assign(
        frequency=data.frequency,
        method=method,
        model=model_name,
        **({"grid_id": grid_id} if grid_id is not None else {}),
    ).to_csv(run_dir / "training_info.csv", index=False)
    save_forecast_plot(output, data.frequency, method, run_dir / "aggregate_test_forecast.png")
    metadata = {
        "model": model_name,
        "frequency": data.frequency,
        "method": method,
        "params": params,
        "master_file": str(enriched_master_path(data.frequency)),
        "fixed_cost_policy": fixed_cost_policy(),
        "residual_targets": {
            "inflow": "inflow_amount - 0",
            "outflow": "outflow_amount - scheduled_loan_repayment",
            "net_flow": "net_flow - fixed_cost_net_flow",
        },
        "known_features": data.known_feature_names,
        "include_history_features": data.include_history_features,
        "recursive_uses_predictions_in_history": method == "recursive",
        "direct_has_one_model_per_flow_and_horizon": method == "direct",
        "n_accounts": len(data.account_ids),
        "test_start": str(data.test_start.date()),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir, pd.read_csv(run_dir / "horizon_metrics.csv")


def _grid_id(model_name: str, index: int, params: dict[str, float | int]) -> str:
    def fmt(value: float | int) -> str:
        return str(value).replace("-", "m").replace(".", "p")

    if model_name == "xgboost":
        short = (
            f"d{fmt(params['max_depth'])}_e{fmt(params['n_estimators'])}"
            f"_lr{fmt(params['learning_rate'])}_mcw{fmt(params['min_child_weight'])}"
            f"_ss{fmt(params['subsample'])}_cs{fmt(params['colsample_bytree'])}"
            f"_l2{fmt(params['reg_lambda'])}_l1{fmt(params['reg_alpha'])}"
            f"_g{fmt(params['gamma'])}"
        )
    elif model_name == "lightgbm":
        short = (
            f"d{fmt(params['max_depth'])}_leaves{fmt(params['num_leaves'])}"
            f"_e{fmt(params['n_estimators'])}_lr{fmt(params['learning_rate'])}"
            f"_mcs{fmt(params['min_child_samples'])}_mcw{fmt(params['min_child_weight'])}"
            f"_ss{fmt(params['subsample'])}_cs{fmt(params['colsample_bytree'])}"
            f"_l2{fmt(params['reg_lambda'])}_l1{fmt(params['reg_alpha'])}"
            f"_gain{fmt(params['min_split_gain'])}"
        )
    else:
        short = (
            f"e{fmt(params['n_estimators'])}_d{fmt(params['max_depth'])}"
            f"_mss{fmt(params['min_samples_split'])}_msl{fmt(params['min_samples_leaf'])}"
            f"_mf{fmt(params['max_features'])}"
        )
    return f"grid_{index:03d}_{short}"


def _load_grid(model_name: str, args: argparse.Namespace) -> list[dict[str, float | int]]:
    keys = TREE_GRID_KEYS[model_name]
    defaults = TREE_DEFAULT_GRIDS[model_name]
    if args.grid_file is not None:
        payload = json.loads(args.grid_file.read_text(encoding="utf-8"))
        values = {
            key: payload[key]
            for key in keys
            if key in payload and isinstance(payload[key], list) and payload[key]
        }
        if set(values) != set(keys):
            raise ValueError(f"Grid JSON needs non-empty lists for {list(keys)}")
    else:
        values = {key: getattr(args, f"grid_{key}") for key in keys}
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*(values[key] for key in keys))
    ]


def _add_grid_arguments(parser: argparse.ArgumentParser, model_name: str) -> None:
    parser.add_argument("--grid-file", type=Path, default=None)
    defaults = TREE_DEFAULT_GRIDS[model_name]
    for key in TREE_GRID_KEYS[model_name]:
        values = defaults[key]
        value_type = float if any(isinstance(value, float) for value in values) else int
        parser.add_argument(
            f"--grid-{key.replace('_', '-')}",
            nargs="+",
            type=value_type,
            default=values,
        )


def _save_model_artifacts(
    model_name: str,
    models: dict[str, Any],
    run_dir: Path,
    method: str,
) -> None:
    joblib.dump(models, run_dir / "model.joblib")
    if model_name == "linear_regression":
        for target, target_models in models.items():
            if isinstance(target_models, list):
                for step, model in enumerate(target_models, start=1):
                    save_coefficients(
                        model,
                        run_dir / f"{target}_residual_coefficients_h{step}.csv",
                        1,
                    )
            else:
                save_coefficients(
                    target_models,
                    run_dir / f"{target}_residual_coefficients.csv",
                    1,
                )
    else:
        from random_forest_forecast_enriched import save_feature_importance as rf_save

        if model_name == "xgboost":
            from xgboost_forecast import save_feature_importance as save_importance
        elif model_name == "lightgbm":
            from lightgbm_forecast import save_feature_importance as save_importance
        else:
            save_importance = rf_save
        save_importance(models, run_dir / "feature_importance.csv")


def run_one(
    model_name: str,
    frequency: str,
    method: str,
    output_dir: Path,
    account_ids: list[int] | None,
    params: dict[str, float | int],
    args: argparse.Namespace,
    grid_id: str | None,
    data: EnrichedForecastData | None = None,
) -> pd.DataFrame:
    if data is None:
        data = create_enriched_forecast_data(
            frequency,
            account_ids,
            include_history_features=not args.exclude_lag_rolling,
        )
    if method == "recursive":
        models, training_info = fit_recursive_model(
            model_name,
            data,
            params,
            args.n_jobs,
            args.random_state,
            args.device_type,
        )
        predictions = recursive_forecast(data, models)
    else:
        models, training_info = fit_direct_model(
            model_name,
            data,
            params,
            args.n_jobs,
            args.random_state,
            args.device_type,
        )
        predictions = direct_forecast(data, models)

    run_dir, horizon_metrics = save_residual_run_outputs(
        data,
        predictions,
        training_info,
        output_dir,
        method,
        model_name,
        params,
        grid_id,
        under_weight=args.under_weight,
        over_weight=args.over_weight,
    )
    _save_model_artifacts(model_name, models, run_dir, method)
    print(
        f"completed residual {model_name} {frequency}/{method}"
        f"{f' {grid_id}' if grid_id else ''}: {len(data.account_ids):,} accounts"
    )
    return horizon_metrics


def main(model_name: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Fixed-loan-cost residual forecasts using {model_name}."
    )
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / f"{model_name}_residual",
    )
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument("--device-type", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--exclude-lag-rolling", action="store_true")
    parser.add_argument("--limit-combinations", type=int, default=None)
    if model_name != "linear_regression":
        _add_grid_arguments(parser, model_name)
    args = parser.parse_args()

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    accounts_by_frequency: dict[str, list[int] | None] = {}
    for frequency in frequencies:
        if args.account_ids is not None:
            accounts_by_frequency[frequency] = args.account_ids
        elif args.max_accounts is not None:
            accounts_by_frequency[frequency] = get_available_enriched_account_ids(
                frequency, args.max_accounts
            )
        else:
            accounts_by_frequency[frequency] = None

    if model_name == "lightgbm" and args.device_type != "cpu":
        from lightgbm_forecast import validate_device_type

        validate_device_type(args.device_type)

    if model_name == "linear_regression":
        param_grid = [{}]
        grid_ids = [None]
    else:
        param_grid = _load_grid(model_name, args)
        if args.limit_combinations is not None:
            if args.limit_combinations < 1:
                raise ValueError("--limit-combinations must be positive")
            param_grid = param_grid[: args.limit_combinations]
        grid_ids = [
            _grid_id(model_name, index, params)
            for index, params in enumerate(param_grid, start=1)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "horizon_comparison_metrics.csv"
    comparison_rows: list[pd.DataFrame] = []
    data_by_frequency: dict[str, EnrichedForecastData] = {}

    for params, grid_id in zip(param_grid, grid_ids):
        if grid_id is not None:
            (args.output_dir / grid_id).mkdir(parents=True, exist_ok=True)
            (args.output_dir / grid_id / "params.json").write_text(
                json.dumps(params, indent=2), encoding="utf-8"
            )
        for frequency in frequencies:
            if frequency not in data_by_frequency:
                data_by_frequency[frequency] = create_enriched_forecast_data(
                    frequency,
                    accounts_by_frequency[frequency],
                    include_history_features=not args.exclude_lag_rolling,
                )
            base_dir = args.output_dir if grid_id is None else args.output_dir / grid_id
            for method in methods:
                run_dir = base_dir / f"{frequency}_{method}"
                cached = run_dir / "horizon_metrics.csv"
                if cached.exists():
                    try:
                        cached_frame = pd.read_csv(cached)
                        if set(cached_frame.get("target", [])) == {
                            "inflow",
                            "outflow",
                            "net_flow",
                        }:
                            comparison_rows.append(cached_frame)
                            print(f"skipping completed residual {frequency}/{method}{f' {grid_id}' if grid_id else ''}")
                            continue
                    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                        pass
                comparison_rows.append(
                    run_one(
                        model_name,
                        frequency,
                        method,
                        base_dir,
                        accounts_by_frequency[frequency],
                        params,
                        args,
                        grid_id,
                        data_by_frequency[frequency],
                    )
                )

    if comparison_rows:
        pd.concat(comparison_rows, ignore_index=True).to_csv(
            comparison_path, index=False
        )
        print(f"saved comparison: {comparison_path}")
