"""Leakage-safe XGBoost forecasts for the Berka master tables.

The data preparation and evaluation conventions match
``linear_regression_forecast.py``:

* monthly + recursive one-step forecast
* monthly + direct horizon forecast
* weekly + recursive one-step forecast
* weekly + direct horizon forecast
* inflow and outflow are trained separately; net flow is derived afterward
* recursive forecasts append predictions, never test-period actual values

Unlike Linear Regression, XGBoost does not require feature scaling. Numeric
features are passed through unchanged and categorical features are one-hot
encoded. XGBoost direct forecasts use one model per flow and horizon step.

The default grid is intentionally modest. Every requested hyperparameter can
be changed from the command line or through a JSON grid file.

Example smoke run:

    .\\ts\\Scripts\\python.exe xgboost_forecast.py `
        --frequency monthly --method direct --max-accounts 2 `
        --max-depths 2 --n-estimators 50 --learning-rates 0.1 `
        --output-dir outputs/xgboost_smoke

Example full grid run for all accounts and four configurations:

    .\\ts\\Scripts\\python.exe xgboost_forecast.py `
        --frequency both --method both --output-dir outputs/xgboost
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    ForecastData,
    STATIC_CATEGORICAL,
    STATIC_NUMERIC,
    add_derived_flow_columns,
    calculate_metrics,
    create_forecast_data,
    make_history_by_account,
    make_origin_features,
    make_training_mask,
    save_forecast_plot,
    static_by_account,
    target_date_columns,
    test_actual_lookup,
    train_normalization_scale,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "xgboost"

GRID_KEYS = (
    "max_depth",
    "n_estimators",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
)

# 3 * 2 * 2 = 12 core combinations. The remaining parameters are exposed
# and can be expanded with CLI arguments or --grid-file when desired.
DEFAULT_PARAM_GRID: dict[str, list[float | int]] = {
    "max_depth": [2, 4, 6],
    "n_estimators": [200, 500],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
    "reg_lambda": [1.0],
    "reg_alpha": [0.0],
    "gamma": [0.0],
}


def make_xgb_pipeline(
    data: ForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
) -> Pipeline:
    """Build one single-target XGBoost pipeline without numeric scaling."""
    preprocessor = ColumnTransformer(
        transformers=[
            # Tree splits do not need StandardScaler.
            ("numeric", "passthrough", data.numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                data.categorical_columns,
            ),
        ],
        remainder="drop",
    )
    regressor = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=0,
        **params,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def fit_recursive_model(
    data: ForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit one one-step XGBoost model for each flow."""
    target_values: dict[str, list[pd.Series]] = {}
    target_dates: dict[str, list[pd.Series]] = {}
    for flow_name, value_column in FLOW_COLUMNS.items():
        target_values[flow_name], target_dates[flow_name] = target_date_columns(
            data.df,
            horizon=1,
            value_column=value_column,
        )

    mask = make_training_mask(
        data.feature_table,
        data.feature_columns,
        [series for values in target_values.values() for series in values],
        [series for dates in target_dates.values() for series in dates],
        data.test_start,
    )
    X_train = data.feature_table.loc[mask, data.feature_columns]
    if X_train.empty:
        raise ValueError("No training rows are available for XGBoost")

    models: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        y_train = target_values[flow_name][0].loc[mask].astype(float)
        model = make_xgb_pipeline(data, params, n_jobs, random_state)
        model.fit(X_train, y_train)
        models[flow_name] = model
        rows.append(
            {
                "target": flow_name,
                "n_train_rows": int(len(X_train)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_models": 1,
                "n_outputs": 1,
            }
        )
    return models, pd.DataFrame(rows)


def fit_direct_model(
    data: ForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
) -> tuple[dict[str, list[Pipeline]], pd.DataFrame]:
    """Fit one XGBoost model per flow and future horizon step."""
    target_values: dict[str, list[pd.Series]] = {}
    target_dates: dict[str, list[pd.Series]] = {}
    for flow_name, value_column in FLOW_COLUMNS.items():
        target_values[flow_name], target_dates[flow_name] = target_date_columns(
            data.df,
            horizon=data.horizon,
            value_column=value_column,
        )

    mask = make_training_mask(
        data.feature_table,
        data.feature_columns,
        [series for values in target_values.values() for series in values],
        [series for dates in target_dates.values() for series in dates],
        data.test_start,
    )
    X_train = data.feature_table.loc[mask, data.feature_columns]
    if X_train.empty:
        raise ValueError("No training rows are available for XGBoost")

    models: dict[str, list[Pipeline]] = {}
    rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        flow_models: list[Pipeline] = []
        for horizon_index in range(data.horizon):
            y_train = target_values[flow_name][horizon_index].loc[mask].astype(float)
            model = make_xgb_pipeline(data, params, n_jobs, random_state)
            model.fit(X_train, y_train)
            flow_models.append(model)
        models[flow_name] = flow_models
        rows.append(
            {
                "target": flow_name,
                "n_train_rows": int(len(X_train)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_models": int(data.horizon),
                "n_outputs": int(data.horizon),
            }
        )
    return models, pd.DataFrame(rows)


def recursive_forecast(
    data: ForecastData,
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    """Forecast sequentially, appending only predictions to each history."""
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    records: list[dict[str, object]] = []

    for step, target_date in enumerate(data.test_dates, start=1):
        X_origin = make_origin_features(
            histories,
            static,
            data.frequency,
            data.include_history_features,
        )
        X_origin = X_origin[data.feature_columns]
        raw_inflow = np.asarray(models["inflow"].predict(X_origin), dtype=float)
        raw_outflow = np.asarray(models["outflow"].predict(X_origin), dtype=float)
        inflow_predictions = np.clip(raw_inflow, 0.0, None)
        outflow_predictions = np.clip(raw_outflow, 0.0, None)

        for index, account_id in enumerate(sorted(histories)):
            account_id = int(account_id)
            predicted_inflow = float(inflow_predictions[index])
            predicted_outflow = float(outflow_predictions[index])
            records.append(
                {
                    "account_id": account_id,
                    "period_start": target_date,
                    "origin_period": (
                        data.test_dates[0]
                        - FREQUENCY_CONFIG[data.frequency]["step"]
                        + (step - 1) * FREQUENCY_CONFIG[data.frequency]["step"]
                    ),
                    "horizon_step": step,
                    "actual_inflow": actual[(account_id, target_date)]["inflow"],
                    "actual_outflow": actual[(account_id, target_date)]["outflow"],
                    "prediction_inflow_raw": float(raw_inflow[index]),
                    "prediction_inflow": predicted_inflow,
                    "prediction_outflow_raw": float(raw_outflow[index]),
                    "prediction_outflow": predicted_outflow,
                }
            )
            histories[account_id]["inflow"].append(predicted_inflow)
            histories[account_id]["outflow"].append(predicted_outflow)

    return add_derived_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )


def direct_forecast(
    data: ForecastData,
    models: dict[str, list[Pipeline]],
) -> pd.DataFrame:
    """Forecast all future periods from one origin with horizon-specific models."""
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    X_origin = make_origin_features(
        histories,
        static,
        data.frequency,
        data.include_history_features,
    )
    X_origin = X_origin[data.feature_columns]
    account_ids = sorted(histories)

    raw_inflow_matrix = np.column_stack(
        [model.predict(X_origin) for model in models["inflow"]]
    ).astype(float)
    raw_outflow_matrix = np.column_stack(
        [model.predict(X_origin) for model in models["outflow"]]
    ).astype(float)
    inflow_matrix = np.clip(raw_inflow_matrix, 0.0, None)
    outflow_matrix = np.clip(raw_outflow_matrix, 0.0, None)
    origin_period = data.test_dates[0] - FREQUENCY_CONFIG[data.frequency]["step"]

    records: list[dict[str, object]] = []
    for account_index, account_id in enumerate(account_ids):
        for step, target_date in enumerate(data.test_dates, start=1):
            records.append(
                {
                    "account_id": int(account_id),
                    "period_start": target_date,
                    "origin_period": origin_period,
                    "horizon_step": step,
                    "actual_inflow": actual[(int(account_id), target_date)]["inflow"],
                    "actual_outflow": actual[(int(account_id), target_date)]["outflow"],
                    "prediction_inflow_raw": float(raw_inflow_matrix[account_index, step - 1]),
                    "prediction_inflow": float(inflow_matrix[account_index, step - 1]),
                    "prediction_outflow_raw": float(raw_outflow_matrix[account_index, step - 1]),
                    "prediction_outflow": float(outflow_matrix[account_index, step - 1]),
                }
            )
    return add_derived_flow_columns(pd.DataFrame(records))


def save_feature_importance(
    models: dict[str, Pipeline] | dict[str, list[Pipeline]],
    output_path: Path,
) -> None:
    """Save gain-based feature importance for every trained model."""
    rows: list[dict[str, object]] = []
    for target, target_models in models.items():
        if isinstance(target_models, list):
            model_items = enumerate(target_models, start=1)
        else:
            model_items = [(1, target_models)]

        for horizon_step, model in model_items:
            preprocessor = model.named_steps["preprocess"]
            regressor = model.named_steps["regressor"]
            names = preprocessor.get_feature_names_out()
            gains = np.asarray(regressor.feature_importances_, dtype=float)
            if len(names) != len(gains):
                raise ValueError("XGBoost feature importance length does not match feature names")
            for feature, gain in zip(names, gains):
                rows.append(
                    {
                        "target": target,
                        "horizon_step": horizon_step,
                        "feature": feature,
                        "importance_gain": float(gain),
                    }
                )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["target", "horizon_step", "importance_gain"],
        ascending=[True, True, False],
    )
    result["rank"] = result.groupby(["target", "horizon_step"]).cumcount() + 1
    result.to_csv(output_path, index=False)


def add_param_columns(frame: pd.DataFrame, params: dict[str, float | int]) -> pd.DataFrame:
    result = frame.copy()
    for key in GRID_KEYS:
        result[key] = params[key]
    return result


def format_value(value: float | int) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def make_grid_id(index: int, params: dict[str, float | int]) -> str:
    short = (
        f"d{format_value(params['max_depth'])}"
        f"_e{format_value(params['n_estimators'])}"
        f"_lr{format_value(params['learning_rate'])}"
        f"_mcw{format_value(params['min_child_weight'])}"
        f"_ss{format_value(params['subsample'])}"
        f"_cs{format_value(params['colsample_bytree'])}"
        f"_l2{format_value(params['reg_lambda'])}"
        f"_l1{format_value(params['reg_alpha'])}"
        f"_g{format_value(params['gamma'])}"
    )
    return f"grid_{index:03d}_{short}"


def run_one(
    frequency: str,
    method: str,
    output_dir: Path,
    account_ids: list[int] | None,
    params: dict[str, float | int],
    grid_id: str,
    under_weight: float,
    over_weight: float,
    n_jobs: int,
    random_state: int,
    include_history_features: bool,
) -> pd.DataFrame:
    data = create_forecast_data(
        frequency,
        account_ids,
        include_history_features=include_history_features,
    )
    if method == "recursive":
        models, training_info = fit_recursive_model(
            data,
            params,
            n_jobs,
            random_state,
        )
        predictions = recursive_forecast(data, models)
    else:
        models, training_info = fit_direct_model(
            data,
            params,
            n_jobs,
            random_state,
        )
        predictions = direct_forecast(data, models)

    run_dir = output_dir / grid_id / f"{frequency}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.insert(0, "frequency", frequency)
    predictions.insert(1, "method", method)
    predictions.insert(2, "grid_id", grid_id)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    scale_df = data.df.copy()
    scale_df["net_flow_for_scale"] = scale_df["inflow_amount"] - scale_df["outflow_amount"]
    metric_specs = [
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

    period_metric_rows: list[pd.DataFrame] = []
    for (
        target,
        actual_column,
        prediction_column,
        raw_column,
        scale_column,
        target_under_weight,
        target_over_weight,
    ) in metric_specs:
        scale = train_normalization_scale(
            df=scale_df,
            test_start=data.test_start,
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
                grid_id=grid_id,
                **params,
            )
        )
    pd.concat(period_metric_rows, ignore_index=True).to_csv(
        run_dir / "metrics.csv", index=False
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
    ) in metric_specs:
        scale = train_normalization_scale(
            df=scale_df,
            test_start=data.test_start,
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
                scale_multiplier=data.horizon,
            ).assign(
                target=target,
                frequency=frequency,
                method=method,
                grid_id=grid_id,
                horizon_periods=data.horizon,
                **params,
            )
        )
    horizon_metrics = pd.concat(horizon_metric_rows, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[
        horizon_metrics["horizon_step"].eq("horizon_total")
    ]
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)

    training_info = training_info.assign(
        frequency=frequency,
        method=method,
        grid_id=grid_id,
        **params,
    )
    training_info.to_csv(run_dir / "training_info.csv", index=False)
    save_feature_importance(models, run_dir / "feature_importance.csv")
    save_forecast_plot(
        predictions,
        frequency,
        method,
        run_dir / "aggregate_test_forecast.png",
    )
    joblib.dump(models, run_dir / "model.joblib")

    metadata = {
        "frequency": frequency,
        "method": method,
        "grid_id": grid_id,
        "horizon": data.horizon,
        "test_start": str(data.test_start.date()),
        "test_dates": [str(date.date()) for date in data.test_dates],
        "n_accounts": len(data.account_ids),
        "target_models": list(FLOW_TARGETS),
        "objective": "reg:squarederror",
        "evaluation_metrics": [
            "mae",
            "rmse",
            "wape",
            "asymmetric_mae",
            "asymmetric_rmse",
            "nmae_train_mean",
            "nrmse_train_mean",
        ],
        "params": params,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "include_history_features": include_history_features,
        "recursive_uses_predictions_in_history": method == "recursive",
        "direct_has_one_model_per_flow_and_horizon": method == "direct",
        "under_weight": under_weight,
        "over_weight": over_weight,
        "normalization_scale": "target-specific pre-test mean absolute flow per account",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        f"completed {grid_id} {frequency}/{method}: "
        f"{len(data.account_ids):,} accounts, horizon={data.horizon}"
    )
    return horizon_metrics


def load_param_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    if args.grid_file is not None:
        payload = json.loads(args.grid_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Grid JSON must contain an object of parameter lists")
        values: dict[str, list[float | int]] = {}
        for key in GRID_KEYS:
            if key not in payload:
                raise ValueError(f"Grid JSON is missing parameter: {key}")
            if not isinstance(payload[key], list) or not payload[key]:
                raise ValueError(f"Grid JSON parameter must be a non-empty list: {key}")
            values[key] = payload[key]
    else:
        values = {
            "max_depth": args.max_depths,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rates,
            "min_child_weight": args.min_child_weights,
            "subsample": args.subsamples,
            "colsample_bytree": args.colsample_bytree_values,
            "reg_lambda": args.reg_lambdas,
            "reg_alpha": args.reg_alphas,
            "gamma": args.gammas,
        }

    combinations = [
        dict(zip(GRID_KEYS, combination))
        for combination in itertools.product(*(values[key] for key in GRID_KEYS))
    ]
    return combinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument(
        "--exclude-lag-rolling",
        action="store_true",
        help="Use only static features; exclude all lag and rolling features.",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=None,
        help="JSON file containing a list for every requested hyperparameter.",
    )

    parser.add_argument("--max-depths", nargs="+", type=int, default=DEFAULT_PARAM_GRID["max_depth"])
    parser.add_argument("--n-estimators", nargs="+", type=int, default=DEFAULT_PARAM_GRID["n_estimators"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=DEFAULT_PARAM_GRID["learning_rate"])
    parser.add_argument("--min-child-weights", nargs="+", type=float, default=DEFAULT_PARAM_GRID["min_child_weight"])
    parser.add_argument("--subsamples", nargs="+", type=float, default=DEFAULT_PARAM_GRID["subsample"])
    parser.add_argument("--colsample-bytree-values", nargs="+", type=float, default=DEFAULT_PARAM_GRID["colsample_bytree"])
    parser.add_argument("--reg-lambdas", nargs="+", type=float, default=DEFAULT_PARAM_GRID["reg_lambda"])
    parser.add_argument("--reg-alphas", nargs="+", type=float, default=DEFAULT_PARAM_GRID["reg_alpha"])
    parser.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_PARAM_GRID["gamma"])
    parser.add_argument(
        "--limit-combinations",
        type=int,
        default=None,
        help="Run only the first N grid combinations; useful for smoke tests.",
    )
    args = parser.parse_args()

    param_grid = load_param_grid(args)
    if args.limit_combinations is not None:
        if args.limit_combinations <= 0:
            raise ValueError("--limit-combinations must be positive")
        param_grid = param_grid[: args.limit_combinations]
    if not param_grid:
        raise ValueError("No hyperparameter combinations were generated")

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    account_ids = args.account_ids
    if account_ids is None and args.max_accounts is not None:
        # Use the same deterministic first-N behavior as the LR script. With
        # no limit, create_forecast_data loads every account in the parquet.
        from linear_regression_forecast import get_available_account_ids

        account_ids_by_frequency = {
            frequency: get_available_account_ids(frequency, args.max_accounts)
            for frequency in frequencies
        }
    else:
        account_ids_by_frequency = {frequency: account_ids for frequency in frequencies}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"running {len(param_grid)} hyperparameter combinations, "
        f"frequencies={frequencies}, methods={methods}"
    )

    horizon_rows: list[pd.DataFrame] = []
    for index, params in enumerate(param_grid, start=1):
        grid_id = make_grid_id(index, params)
        (args.output_dir / grid_id).mkdir(parents=True, exist_ok=True)
        (args.output_dir / grid_id / "params.json").write_text(
            json.dumps(params, indent=2),
            encoding="utf-8",
        )
        for frequency in frequencies:
            for method in methods:
                metrics = run_one(
                    frequency=frequency,
                    method=method,
                    output_dir=args.output_dir,
                    account_ids=account_ids_by_frequency[frequency],
                    params=params,
                    grid_id=grid_id,
                    under_weight=args.under_weight,
                    over_weight=args.over_weight,
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                    include_history_features=not args.exclude_lag_rolling,
                )
                horizon_rows.append(metrics)

    comparison = pd.concat(horizon_rows, ignore_index=True)
    comparison.to_csv(args.output_dir / "horizon_comparison_metrics.csv", index=False)

    horizon_total = comparison.loc[
        comparison["horizon_step"].eq("horizon_total")
    ].copy()
    metric_names = [
        "mae",
        "rmse",
        "wape",
        "asymmetric_mae",
        "asymmetric_rmse",
        "nmae_train_mean",
        "nrmse_train_mean",
    ]
    best_rows: list[dict[str, Any]] = []
    for (frequency, method, target), group in horizon_total.groupby(
        ["frequency", "method", "target"], sort=True
    ):
        for metric in metric_names:
            winner = group.loc[group[metric].idxmin()]
            best_rows.append(
                {
                    "frequency": frequency,
                    "method": method,
                    "target": target,
                    "metric": metric,
                    "best_value": float(winner[metric]),
                    "grid_id": winner["grid_id"],
                    **{key: winner[key] for key in GRID_KEYS},
                }
            )
    pd.DataFrame(best_rows).to_csv(
        args.output_dir / "best_by_metric.csv",
        index=False,
    )
    print(f"saved comparison metrics: {len(comparison):,} rows")


if __name__ == "__main__":
    main()
