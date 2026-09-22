"""Leakage-safe LightGBM forecasts for the Berka master tables.

The data preparation and evaluation conventions match
linear_regression_forecast.py and xgboost_forecast.py:

* monthly + recursive one-step forecast
* monthly + direct horizon forecast
* weekly + recursive one-step forecast
* weekly + direct horizon forecast
* inflow and outflow are trained separately; net flow is derived afterward
* recursive forecasts append predictions, never test-period actual values

Like XGBoost, LightGBM does not require feature scaling. Numeric features are
passed through unchanged and categorical features are one-hot encoded. The
model device is explicit: use --device-type cuda in a CUDA-enabled Colab
build, or --device-type cpu for a portable CPU run.

Example smoke run:

    !python lightgbm_forecast.py \
        --frequency monthly --method direct --max-accounts 2 \
        --max-depths 2 --num-leaves 4 --n-estimators 50 \
        --learning-rates 0.1 --device-type cpu \
        --output-dir outputs/lightgbm_smoke

Example full run for all accounts and four configurations:

    !python lightgbm_forecast.py \
        --frequency both --method both --device-type cuda \
        --output-dir outputs/lightgbm
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
from lightgbm import LGBMRegressor

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
    get_available_account_ids,
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
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "lightgbm"

GRID_KEYS = (
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
)

EVALUATION_TARGETS = ("inflow", "outflow", "net_flow")

# Keep the best previously selected core settings fixed and sweep the five
# requested LightGBM regularization and sampling parameters. This creates
# 3 * 3 * 3 * 3 * 3 = 243 combinations. All parameters remain configurable
# through CLI arguments or --grid-file.
DEFAULT_PARAM_GRID: dict[str, list[float | int]] = {
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
}

def make_lgbm_pipeline(
    data: ForecastData,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
    device_type: str,
) -> Pipeline:
    """Build one single-target LightGBM pipeline without numeric scaling."""
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
    regressor = LGBMRegressor(
        objective="regression",
        metric="rmse",
        device_type=device_type,
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=-1,
        importance_type="gain",
        # Make subsample effective whenever a grid uses a value below 1.0.
        subsample_freq=1,
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
    device_type: str,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit one one-step LightGBM model for each flow."""
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
        raise ValueError("No training rows are available for LightGBM")

    models: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        y_train = target_values[flow_name][0].loc[mask].astype(float)
        model = make_lgbm_pipeline(data, params, n_jobs, random_state, device_type)
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
    device_type: str,
) -> tuple[dict[str, list[Pipeline]], pd.DataFrame]:
    """Fit one LightGBM model per flow and future horizon step."""
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
        raise ValueError("No training rows are available for LightGBM")

    models: dict[str, list[Pipeline]] = {}
    rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        flow_models: list[Pipeline] = []
        for horizon_index in range(data.horizon):
            y_train = target_values[flow_name][horizon_index].loc[mask].astype(float)
            model = make_lgbm_pipeline(
                data,
                params,
                n_jobs,
                random_state,
                device_type,
            )
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
                raise ValueError("LightGBM feature importance length does not match feature names")
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
        f"_leaves{format_value(params['num_leaves'])}"
        f"_e{format_value(params['n_estimators'])}"
        f"_lr{format_value(params['learning_rate'])}"
        f"_mcs{format_value(params['min_child_samples'])}"
        f"_mcw{format_value(params['min_child_weight'])}"
        f"_ss{format_value(params['subsample'])}"
        f"_cs{format_value(params['colsample_bytree'])}"
        f"_l2{format_value(params['reg_lambda'])}"
        f"_l1{format_value(params['reg_alpha'])}"
        f"_gain{format_value(params['min_split_gain'])}"
    )
    return f"grid_{index:03d}_{short}"


def load_completed_horizon_metrics(
    frame: pd.DataFrame | None,
    grid_id: str,
    frequency: str,
    method: str,
    params: dict[str, float | int],
    expected_account_count: int | None = None,
) -> pd.DataFrame | None:
    """Return cached horizon metrics when a run is complete and compatible."""
    if frame is None or frame.empty:
        return None

    required_columns = {
        "horizon_step",
        "target",
        "frequency",
        "method",
        "grid_id",
        "horizon_periods",
        "mae",
        "rmse",
        "wape",
        "asymmetric_mae",
        "asymmetric_rmse",
        "nmae_train_mean",
        "nrmse_train_mean",
    }
    if not required_columns.issubset(frame.columns):
        return None

    cached = frame.loc[
        frame["horizon_step"].astype(str).eq("horizon_total")
        & frame["frequency"].astype(str).eq(frequency)
        & frame["method"].astype(str).eq(method)
        & frame["grid_id"].astype(str).eq(grid_id)
    ].copy()
    if len(cached) != len(EVALUATION_TARGETS):
        return None
    if set(cached["target"].astype(str)) != set(EVALUATION_TARGETS):
        return None

    expected_horizon = FREQUENCY_CONFIG[frequency]["horizon"]
    if not np.isclose(cached["horizon_periods"].astype(float), expected_horizon).all():
        return None
    if expected_account_count is not None:
        if "n_observations" not in cached.columns:
            return None
        if not np.isclose(
            cached["n_observations"].astype(float), expected_account_count
        ).all():
            return None
    for key, value in params.items():
        if key not in cached.columns:
            return None
        if not np.isclose(cached[key].astype(float), float(value)).all():
            return None

    metric_columns = [
        "mae",
        "rmse",
        "wape",
        "asymmetric_mae",
        "asymmetric_rmse",
        "nmae_train_mean",
        "nrmse_train_mean",
    ]
    if cached[metric_columns].isna().any().any():
        return None
    return cached.reset_index(drop=True)


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
    device_type: str,
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
            device_type,
        )
        predictions = recursive_forecast(data, models)
    else:
        models, training_info = fit_direct_model(
            data,
            params,
            n_jobs,
            random_state,
            device_type,
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
        "model": "LightGBM",
        "objective": "regression",
        "device_type": device_type,
        "importance_type": "gain",
        "subsample_freq": 1,
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
            "num_leaves": args.num_leaves,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rates,
            "min_child_samples": args.min_child_samples,
            "min_child_weight": args.min_child_weights,
            "subsample": args.subsamples,
            "colsample_bytree": args.colsample_bytree_values,
            "reg_lambda": args.reg_lambdas,
            "reg_alpha": args.reg_alphas,
            "min_split_gain": args.min_split_gains,
        }

    combinations = [
        dict(zip(GRID_KEYS, combination))
        for combination in itertools.product(*(values[key] for key in GRID_KEYS))
    ]
    invalid = [
        combination
        for combination in combinations
        if combination["max_depth"] > 0
        and combination["num_leaves"] > 2 ** combination["max_depth"]
    ]
    if invalid:
        example = invalid[0]
        raise ValueError(
            "For LightGBM, num_leaves must be <= 2**max_depth when max_depth "
            f"is positive; invalid combination: {example}"
        )
    return combinations


def validate_device_type(device_type: str) -> None:
    """Fail early if the requested non-CPU LightGBM build is unavailable."""
    if device_type == "cpu":
        return
    probe_x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    probe_y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    try:
        LGBMRegressor(
            objective="regression",
            device_type=device_type,
            n_estimators=1,
            num_leaves=2,
            max_depth=2,
            min_child_samples=1,
            min_child_weight=0.0,
            verbosity=-1,
        ).fit(probe_x, probe_y)
    except Exception as exc:  # noqa: BLE001 - preserve a clear setup error
        raise RuntimeError(
            f"LightGBM device_type={device_type!r} is unavailable in this "
            "environment. Install a GPU-enabled LightGBM build or use "
            "--device-type cpu."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device-type",
        choices=["cpu", "gpu", "cuda"],
        default="cpu",
        help="LightGBM device; use cuda for a CUDA-enabled Colab build.",
    )
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
    parser.add_argument("--num-leaves", nargs="+", type=int, default=DEFAULT_PARAM_GRID["num_leaves"])
    parser.add_argument("--n-estimators", nargs="+", type=int, default=DEFAULT_PARAM_GRID["n_estimators"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=DEFAULT_PARAM_GRID["learning_rate"])
    parser.add_argument("--min-child-samples", nargs="+", type=int, default=DEFAULT_PARAM_GRID["min_child_samples"])
    parser.add_argument("--min-child-weights", nargs="+", type=float, default=DEFAULT_PARAM_GRID["min_child_weight"])
    parser.add_argument("--subsamples", nargs="+", type=float, default=DEFAULT_PARAM_GRID["subsample"])
    parser.add_argument("--colsample-bytree-values", nargs="+", type=float, default=DEFAULT_PARAM_GRID["colsample_bytree"])
    parser.add_argument("--reg-lambdas", nargs="+", type=float, default=DEFAULT_PARAM_GRID["reg_lambda"])
    parser.add_argument("--reg-alphas", nargs="+", type=float, default=DEFAULT_PARAM_GRID["reg_alpha"])
    parser.add_argument("--min-split-gains", nargs="+", type=float, default=DEFAULT_PARAM_GRID["min_split_gain"])
    parser.add_argument(
        "--limit-combinations",
        type=int,
        default=None,
        help="Run only the first N grid combinations; useful for smoke tests.",
    )
    args = parser.parse_args()

    validate_device_type(args.device_type)
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
        account_ids_by_frequency = {
            frequency: get_available_account_ids(frequency, args.max_accounts)
            for frequency in frequencies
        }
    else:
        account_ids_by_frequency = {frequency: account_ids for frequency in frequencies}

    expected_account_counts = {
        frequency: (
            len(account_ids_by_frequency[frequency])
            if account_ids_by_frequency[frequency] is not None
            else len(get_available_account_ids(frequency))
        )
        for frequency in frequencies
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "horizon_comparison_metrics.csv"
    existing_comparison: pd.DataFrame | None = None
    if comparison_path.exists():
        try:
            existing_comparison = pd.read_csv(comparison_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            existing_comparison = None

    print(
        f"running {len(param_grid)} hyperparameter combinations, "
        f"frequencies={frequencies}, methods={methods}, resume=True"
    )

    horizon_rows: list[pd.DataFrame] = []
    n_skipped = 0
    n_run = 0
    for index, params in enumerate(param_grid, start=1):
        grid_id = make_grid_id(index, params)
        (args.output_dir / grid_id).mkdir(parents=True, exist_ok=True)
        (args.output_dir / grid_id / "params.json").write_text(
            json.dumps(params, indent=2),
            encoding="utf-8",
        )
        for frequency in frequencies:
            for method in methods:
                run_dir = args.output_dir / grid_id / f"{frequency}_{method}"
                cached = None
                horizon_path = run_dir / "horizon_metrics.csv"
                if horizon_path.exists():
                    try:
                        cached = load_completed_horizon_metrics(
                            pd.read_csv(horizon_path),
                            grid_id,
                            frequency,
                            method,
                            params,
                            expected_account_counts[frequency],
                        )
                    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                        cached = None
                if cached is None:
                    cached = load_completed_horizon_metrics(
                        existing_comparison,
                        grid_id,
                        frequency,
                        method,
                        params,
                        expected_account_counts[frequency],
                    )
                if cached is not None:
                    print(f"skipping completed {grid_id} {frequency}/{method}")
                    horizon_rows.append(cached)
                    n_skipped += 1
                    continue

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
                    device_type=args.device_type,
                    include_history_features=not args.exclude_lag_rolling,
                )
                horizon_rows.append(metrics)
                n_run += 1
                # Persist progress after every completed run so an
                # interruption can resume without losing the comparison data.
                pd.concat(horizon_rows, ignore_index=True).to_csv(
                    comparison_path,
                    index=False,
                )

    comparison = pd.concat(horizon_rows, ignore_index=True)
    comparison.to_csv(comparison_path, index=False)

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
    print(
        f"saved comparison metrics: {len(comparison):,} rows; "
        f"ran={n_run}, skipped={n_skipped}"
    )


if __name__ == "__main__":
    main()
