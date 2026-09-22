"""Shared enriched-master runner for XGBoost and LightGBM.

The two public entry points are ``xgboost_forecast_enriched.py`` and
``lightgbm_forecast_enriched.py``.  This module keeps their forecasting logic
identical while letting each backend retain its own hyperparameter grid.
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

from enriched_forecast_common import (
    EnrichedForecastData,
    create_enriched_forecast_data,
    enriched_master_path,
    get_available_enriched_account_ids,
    make_history_by_account,
    make_origin_features,
    save_enriched_run_outputs,
    static_by_account,
    test_actual_lookup,
)
from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    add_derived_flow_columns,
    save_forecast_plot,
    target_date_columns,
)


XGB_GRID_KEYS = (
    "max_depth", "n_estimators", "learning_rate", "min_child_weight",
    "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma",
)
XGB_DEFAULT_GRID: dict[str, list[float | int]] = {
    "max_depth": [2, 4, 6], "n_estimators": [200, 500],
    "learning_rate": [0.03, 0.1], "min_child_weight": [1],
    "subsample": [0.9], "colsample_bytree": [0.9],
    "reg_lambda": [1.0], "reg_alpha": [0.0], "gamma": [0.0],
}

LGBM_GRID_KEYS = (
    "max_depth", "num_leaves", "n_estimators", "learning_rate",
    "min_child_samples", "min_child_weight", "subsample",
    "colsample_bytree", "reg_lambda", "reg_alpha", "min_split_gain",
)
LGBM_DEFAULT_GRID: dict[str, list[float | int]] = {
    "max_depth": [4], "num_leaves": [4], "n_estimators": [500],
    "learning_rate": [0.1], "min_child_samples": [50],
    "min_child_weight": [0.1], "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0], "reg_lambda": [0.0, 1.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0], "min_split_gain": [0.0, 0.1, 0.5],
}


def _backend(model_name: str):
    if model_name == "xgboost":
        from xgboost_forecast import make_xgb_pipeline, save_feature_importance
        return (
            make_xgb_pipeline,
            save_feature_importance,
            XGB_GRID_KEYS,
            XGB_DEFAULT_GRID,
        )
    if model_name == "lightgbm":
        from lightgbm_forecast import make_lgbm_pipeline, save_feature_importance
        return (
            make_lgbm_pipeline,
            save_feature_importance,
            LGBM_GRID_KEYS,
            LGBM_DEFAULT_GRID,
        )
    raise ValueError(f"Unknown tree model: {model_name}")


def _training_mask(data: EnrichedForecastData, horizon_index: int) -> pd.Series:
    """Use features and targets whose target period is strictly pre-test."""
    feature_table = data.feature_tables[horizon_index]
    mask = feature_table[data.feature_columns].notna().all(axis=1)
    for value_column in FLOW_COLUMNS.values():
        values, dates = target_date_columns(
            data.df, horizon=horizon_index + 1, value_column=value_column
        )
        mask &= values[horizon_index].notna()
        mask &= dates[horizon_index] < data.test_start
    return mask


def _make_model(
    model_name: str,
    data: EnrichedForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
):
    factory, _, _, _ = _backend(model_name)
    if model_name == "xgboost":
        return factory(data, params, n_jobs, random_state)
    return factory(data, params, n_jobs, random_state, device_type)


def fit_recursive_model(
    model_name: str,
    data: EnrichedForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
):
    mask = _training_mask(data, 0)
    X_train = data.feature_tables[0].loc[mask, data.feature_columns]
    if X_train.empty:
        raise ValueError(f"No training rows are available for enriched {model_name}")

    models: dict[str, Any] = {}
    rows: list[dict[str, object]] = []
    for flow_name, value_column in FLOW_COLUMNS.items():
        target_values, _ = target_date_columns(data.df, 1, value_column)
        y_train = target_values[0].loc[mask].astype(float)
        model = _make_model(model_name, data, params, n_jobs, random_state, device_type)
        model.fit(X_train, y_train)
        models[flow_name] = model
        rows.append({
            "target": flow_name, "n_train_rows": len(X_train),
            "n_features_raw": len(data.feature_columns), "n_models": 1,
            "n_outputs": 1, "known_features": ",".join(data.known_feature_names),
        })
    return models, pd.DataFrame(rows)


def fit_direct_model(
    model_name: str,
    data: EnrichedForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
):
    models: dict[str, list[Any]] = {}
    rows: list[dict[str, object]] = []
    for flow_name, value_column in FLOW_COLUMNS.items():
        target_values, _ = target_date_columns(data.df, data.horizon, value_column)
        flow_models: list[Any] = []
        counts: list[int] = []
        for horizon_index in range(data.horizon):
            mask = _training_mask(data, horizon_index)
            X_train = data.feature_tables[horizon_index].loc[
                mask, data.feature_columns
            ]
            y_train = target_values[horizon_index].loc[mask].astype(float)
            if X_train.empty:
                raise ValueError(
                    f"No training rows for {flow_name}, horizon={horizon_index + 1}"
                )
            model = _make_model(model_name, data, params, n_jobs, random_state, device_type)
            model.fit(X_train, y_train)
            flow_models.append(model)
            counts.append(len(X_train))
        models[flow_name] = flow_models
        rows.append({
            "target": flow_name, "n_train_rows_min": min(counts),
            "n_train_rows_max": max(counts), "n_features_raw": len(data.feature_columns),
            "n_models": data.horizon, "n_outputs": data.horizon,
            "known_features": ",".join(data.known_feature_names),
        })
    return models, pd.DataFrame(rows)


def _known_values(known: pd.DataFrame, account_id: int, names: list[str]) -> dict[str, object]:
    return {name: float(known.loc[account_id, name]) for name in names}


def recursive_forecast(data: EnrichedForecastData, models: dict[str, Any]) -> pd.DataFrame:
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    records: list[dict[str, object]] = []
    for step, target_date in enumerate(data.test_dates, start=1):
        known = data.test_known_features[target_date]
        X_origin = make_origin_features(
            histories, static, data.frequency, target_date, known,
            data.include_history_features,
        )[data.feature_columns]
        raw_inflow = np.asarray(models["inflow"].predict(X_origin), dtype=float)
        raw_outflow = np.asarray(models["outflow"].predict(X_origin), dtype=float)
        inflow = np.clip(raw_inflow, 0.0, None)
        outflow = np.clip(raw_outflow, 0.0, None)
        for index, account_id in enumerate(sorted(histories)):
            account_id = int(account_id)
            record: dict[str, object] = {
                "account_id": account_id, "period_start": target_date,
                "origin_period": target_date - FREQUENCY_CONFIG[data.frequency]["step"],
                "horizon_step": step,
                "actual_inflow": actual[(account_id, target_date)]["inflow"],
                "actual_outflow": actual[(account_id, target_date)]["outflow"],
                "prediction_inflow_raw": float(raw_inflow[index]),
                "prediction_inflow": float(inflow[index]),
                "prediction_outflow_raw": float(raw_outflow[index]),
                "prediction_outflow": float(outflow[index]),
            }
            record.update(_known_values(known, account_id, data.known_feature_names))
            records.append(record)
            histories[account_id]["inflow"].append(float(inflow[index]))
            histories[account_id]["outflow"].append(float(outflow[index]))
    return add_derived_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )


def direct_forecast(data: EnrichedForecastData, models: dict[str, list[Any]]) -> pd.DataFrame:
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    account_ids = sorted(histories)
    records: list[dict[str, object]] = []
    origin_period = data.test_dates[0] - FREQUENCY_CONFIG[data.frequency]["step"]
    for step, target_date in enumerate(data.test_dates, start=1):
        known = data.test_known_features[target_date]
        X_origin = make_origin_features(
            histories, static, data.frequency, target_date, known,
            data.include_history_features,
        )[data.feature_columns]
        raw_inflow = np.asarray(models["inflow"][step - 1].predict(X_origin), dtype=float)
        raw_outflow = np.asarray(models["outflow"][step - 1].predict(X_origin), dtype=float)
        for index, account_id in enumerate(account_ids):
            account_id = int(account_id)
            record: dict[str, object] = {
                "account_id": account_id, "period_start": target_date,
                "origin_period": origin_period, "horizon_step": step,
                "actual_inflow": actual[(account_id, target_date)]["inflow"],
                "actual_outflow": actual[(account_id, target_date)]["outflow"],
                "prediction_inflow_raw": float(raw_inflow[index]),
                "prediction_inflow": float(max(raw_inflow[index], 0.0)),
                "prediction_outflow_raw": float(raw_outflow[index]),
                "prediction_outflow": float(max(raw_outflow[index], 0.0)),
            }
            record.update(_known_values(known, account_id, data.known_feature_names))
            records.append(record)
    return add_derived_flow_columns(pd.DataFrame(records))


def _format_value(value: float | int) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def make_grid_id(model_name: str, index: int, params: dict[str, float | int]) -> str:
    if model_name == "xgboost":
        short = (
            f"d{_format_value(params['max_depth'])}_e{_format_value(params['n_estimators'])}"
            f"_lr{_format_value(params['learning_rate'])}_mcw{_format_value(params['min_child_weight'])}"
            f"_ss{_format_value(params['subsample'])}_cs{_format_value(params['colsample_bytree'])}"
            f"_l2{_format_value(params['reg_lambda'])}_l1{_format_value(params['reg_alpha'])}"
            f"_g{_format_value(params['gamma'])}"
        )
    else:
        short = (
            f"d{_format_value(params['max_depth'])}_leaves{_format_value(params['num_leaves'])}"
            f"_e{_format_value(params['n_estimators'])}_lr{_format_value(params['learning_rate'])}"
            f"_mcs{_format_value(params['min_child_samples'])}_mcw{_format_value(params['min_child_weight'])}"
            f"_ss{_format_value(params['subsample'])}_cs{_format_value(params['colsample_bytree'])}"
            f"_l2{_format_value(params['reg_lambda'])}_l1{_format_value(params['reg_alpha'])}"
            f"_gain{_format_value(params['min_split_gain'])}"
        )
    return f"grid_{index:03d}_{short}"


def load_grid(model_name: str, args: argparse.Namespace) -> list[dict[str, float | int]]:
    _, _, keys, defaults = _backend(model_name)
    if args.grid_file is not None:
        payload = json.loads(args.grid_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Grid JSON must contain an object of parameter lists")
        values = {}
        for key in keys:
            if key not in payload or not isinstance(payload[key], list) or not payload[key]:
                raise ValueError(f"Grid JSON needs a non-empty list for {key}")
            values[key] = payload[key]
    else:
        values = {key: getattr(args, f"grid_{key}") for key in keys}
    return [dict(zip(keys, combination)) for combination in itertools.product(*(values[key] for key in keys))]


def _add_grid_arguments(parser: argparse.ArgumentParser, model_name: str) -> None:
    _, _, keys, defaults = _backend(model_name)
    parser.add_argument("--grid-file", type=Path, default=None)
    for key in keys:
        cli_key = key.replace("_", "-")
        values = defaults[key]
        value_type = float if any(isinstance(value, float) for value in values) else int
        parser.add_argument(f"--grid-{cli_key}", nargs="+", type=value_type, default=values)


def run_one(
    model_name: str,
    frequency: str,
    method: str,
    output_dir: Path,
    account_ids: list[int] | None,
    params: dict[str, float | int],
    grid_id: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    data = create_enriched_forecast_data(
        frequency, account_ids, include_history_features=not args.exclude_lag_rolling
    )
    if method == "recursive":
        models, training_info = fit_recursive_model(
            model_name, data, params, args.n_jobs, args.random_state, args.device_type
        )
        predictions = recursive_forecast(data, models)
    else:
        models, training_info = fit_direct_model(
            model_name, data, params, args.n_jobs, args.random_state, args.device_type
        )
        predictions = direct_forecast(data, models)

    run_dir, horizon_metrics = save_enriched_run_outputs(
        data, predictions, training_info, output_dir / grid_id, method,
        under_weight=args.under_weight, over_weight=args.over_weight, grid_id=grid_id,
    )
    horizon_metrics = horizon_metrics.assign(grid_id=grid_id, **params)
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)
    _, save_importance, _, _ = _backend(model_name)
    save_importance(models, run_dir / "feature_importance.csv")
    training_info.assign(grid_id=grid_id, **params).to_csv(run_dir / "training_info.csv", index=False)
    joblib.dump(models, run_dir / "model.joblib")
    metadata = {
        "model": model_name, "frequency": frequency, "method": method,
        "grid_id": grid_id, "params": params,
        "master_file": str(enriched_master_path(frequency)),
        "known_features": data.known_feature_names,
        "loan_schedule_source": "loan.payments with loan.date as-of masking",
        "include_history_features": not args.exclude_lag_rolling,
        "recursive_uses_predictions_in_history": method == "recursive",
        "direct_has_one_model_per_flow_and_horizon": method == "direct",
        "n_accounts": len(data.account_ids), "device_type": args.device_type,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"completed {model_name} {grid_id} {frequency}/{method}: {len(data.account_ids):,} accounts")
    return horizon_metrics


def main(model_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Enriched {model_name} forecast")
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / f"{model_name}_enriched")
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument("--exclude-lag-rolling", action="store_true")
    parser.add_argument("--device-type", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--limit-combinations", type=int, default=None)
    _add_grid_arguments(parser, model_name)
    args = parser.parse_args()

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    accounts_by_frequency = {}
    for frequency in frequencies:
        if args.account_ids is not None:
            accounts_by_frequency[frequency] = args.account_ids
        elif args.max_accounts is not None:
            accounts_by_frequency[frequency] = get_available_enriched_account_ids(frequency, args.max_accounts)
        else:
            accounts_by_frequency[frequency] = None

    if model_name == "lightgbm" and args.device_type != "cpu":
        from lightgbm_forecast import validate_device_type
        validate_device_type(args.device_type)

    param_grid = load_grid(model_name, args)
    if args.limit_combinations is not None:
        if args.limit_combinations < 1:
            raise ValueError("--limit-combinations must be positive")
        param_grid = param_grid[:args.limit_combinations]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "horizon_comparison_metrics.csv"
    rows: list[pd.DataFrame] = []
    for index, params in enumerate(param_grid, start=1):
        grid_id = make_grid_id(model_name, index, params)
        (args.output_dir / grid_id).mkdir(parents=True, exist_ok=True)
        (args.output_dir / grid_id / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        for frequency in frequencies:
            for method in methods:
                run_dir = args.output_dir / grid_id / f"{frequency}_{method}"
                cached = run_dir / "horizon_metrics.csv"
                if cached.exists():
                    try:
                        cached_frame = pd.read_csv(cached)
                        if len(cached_frame) == 3 and set(cached_frame["target"]) == {"inflow", "outflow", "net_flow"}:
                            rows.append(cached_frame)
                            print(f"skipping completed {grid_id} {frequency}/{method}")
                            continue
                    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                        pass
                rows.append(run_one(model_name, frequency, method, args.output_dir, accounts_by_frequency[frequency], params, grid_id, args))
                pd.concat(rows, ignore_index=True).to_csv(comparison_path, index=False)

    if not rows:
        raise ValueError("No grid combinations were generated")
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(comparison_path, index=False)
    _, _, keys, _ = _backend(model_name)
    horizon_total = comparison.loc[comparison["horizon_step"].astype(str).eq("horizon_total")]
    metric_names = ["mae", "rmse", "wape", "asymmetric_mae", "asymmetric_rmse", "nmae_train_mean", "nrmse_train_mean"]
    best: list[dict[str, object]] = []
    for (frequency, method, target), group in horizon_total.groupby(["frequency", "method", "target"], sort=True):
        for metric in metric_names:
            winner = group.loc[group[metric].idxmin()]
            best.append({"frequency": frequency, "method": method, "target": target, "metric": metric, "best_value": float(winner[metric]), "grid_id": winner["grid_id"], **{key: winner[key] for key in keys}})
    pd.DataFrame(best).to_csv(args.output_dir / "best_by_metric.csv", index=False)
    print(f"saved {model_name} comparison metrics: {len(comparison):,} rows")
