"""Leakage-safe Linear Regression forecasts for the Berka master tables.

Four configurations are supported:

* monthly + recursive one-step forecast
* monthly + direct multi-output forecast
* weekly + recursive one-step forecast
* weekly + direct multi-output forecast

The model predicts period-level ``inflow_amount`` and ``outflow_amount``
separately, then derives net flow as inflow minus outflow.  Existing lag,
rolling, and bagged columns in the parquet files are deliberately not read.
Features are rebuilt from the observed/predicted flow histories so that
test-time recursive forecasts do not consume future actual values.

Run from the project root, for example:

    .\\ts\\Scripts\\python.exe linear_regression_forecast.py --frequency monthly
    .\\ts\\Scripts\\python.exe linear_regression_forecast.py --frequency both
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
MASTER_DIR = ROOT / "Berka Dataset" / "Berka Dataset" / "master_tables"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "linear_regression"

# Only these columns are read.  In particular, no existing *_lag, *_rolling,
# or *_bagged_smooth columns are loaded from the master tables.
RAW_COLUMNS = [
    "account_id",
    "period_start",
    "inflow_amount",
    "outflow_amount",
    "birth_year",
    "gender",
    "account_frequency",
]
FLOW_TARGETS = ("inflow", "outflow")
FLOW_COLUMNS = {
    "inflow": "inflow_amount",
    "outflow": "outflow_amount",
}
STATIC_NUMERIC = ["birth_year"]
STATIC_CATEGORICAL = ["gender", "account_frequency"]

FREQUENCY_CONFIG = {
    "monthly": {
        "suffix": "m",
        "step": pd.DateOffset(months=1),
        "horizon": 6,
        # Existing master-table lag/window profile.
        "lags": [0, 1, 2, 3, 7, 28],
        "windows": [3, 6, 12],
    },
    "weekly": {
        "suffix": "w",
        "step": pd.Timedelta(weeks=1),
        # 1998-06-28 through 1998-12-27 is 27 weekly test periods.
        "horizon": 27,
        # Existing master-table lag/window profile.
        "lags": [0, 1, 2, 3, 7, 28],
        "windows": [4, 8, 13],
    },
}


@dataclass
class ForecastData:
    frequency: str
    horizon: int
    include_history_features: bool
    df: pd.DataFrame
    feature_table: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    test_start: pd.Timestamp
    test_dates: list[pd.Timestamp]
    account_ids: list[int]


def read_master(frequency: str, account_ids: list[int] | None = None) -> pd.DataFrame:
    config = FREQUENCY_CONFIG[frequency]
    path = MASTER_DIR / f"berka_master_{config['suffix']}.parquet"
    df = pd.read_parquet(path, columns=RAW_COLUMNS)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["account_id"] = df["account_id"].astype(int)

    if account_ids:
        requested = set(int(value) for value in account_ids)
        available = set(df["account_id"].unique())
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown account_id(s): {missing}")
        df = df.loc[df["account_id"].isin(requested)].copy()

    return df.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def get_available_account_ids(frequency: str, max_accounts: int | None = None) -> list[int]:
    """Get deterministic account IDs from parquet without loading other columns."""
    config = FREQUENCY_CONFIG[frequency]
    path = MASTER_DIR / f"berka_master_{config['suffix']}.parquet"
    account_ids = (
        pd.read_parquet(path, columns=["account_id"])["account_id"]
        .drop_duplicates()
        .sort_values()
        .astype(int)
        .tolist()
    )
    if max_accounts is not None:
        account_ids = account_ids[:max_accounts]
    return account_ids


def get_test_start(df: pd.DataFrame) -> pd.Timestamp:
    if "test_start" in df.columns:
        values = pd.to_datetime(df["test_start"]).dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Expected one test_start, found {values}")
        return pd.Timestamp(values[0])
    # RAW_COLUMNS intentionally excludes test_start.  The master tables have
    # the same fixed split, so derive it from the known split date by selecting
    # the last six calendar months/weekly periods in main().
    raise ValueError("test_start must be supplied separately")


def derive_test_start(frequency: str, df: pd.DataFrame) -> pd.Timestamp:
    # These are the split boundaries already recorded in the master tables.
    # We derive them without loading any precomputed modeling features.
    if frequency == "monthly":
        return pd.Timestamp("1998-07-01")
    return pd.Timestamp("1998-06-28")


def rolling_series(grouped: pd.core.groupby.SeriesGroupBy, window: int, operation: str) -> pd.Series:
    result = getattr(grouped.rolling(window=window, min_periods=window), operation)()
    return result.reset_index(level=0, drop=True)


def build_feature_table(
    df: pd.DataFrame,
    frequency: str,
    include_history_features: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    config = FREQUENCY_CONFIG[frequency]
    features = pd.DataFrame(index=df.index)
    numeric_columns: list[str] = []

    # Current period is lag_0.  At forecast origin t, it is known and is
    # allowed in the feature vector used to predict t+1.  Both flow
    # histories are modeled so net cash flow can be calculated without using
    # actual future inflows or outflows.
    if include_history_features:
        for flow_name, value_column in FLOW_COLUMNS.items():
            grouped = df.groupby("account_id", sort=False)[value_column]
            for lag in config["lags"]:
                name = f"{flow_name}_lag_{lag}"
                features[name] = grouped.shift(lag)
                numeric_columns.append(name)

            for window in config["windows"]:
                for operation in ("mean", "median", "std"):
                    name = f"{flow_name}_rolling_{operation}_{window}"
                    features[name] = rolling_series(grouped, window, operation)
                    numeric_columns.append(name)

    # These variables are account-level/static in the master table and are
    # known at forecast time.  We do not use loan_status or repayment fields,
    # which can encode information that becomes known only later.
    for column in STATIC_NUMERIC:
        features[column] = pd.to_numeric(df[column], errors="coerce")
    for column in STATIC_CATEGORICAL:
        features[column] = df[column].astype("string").fillna("missing")

    categorical_columns = STATIC_CATEGORICAL.copy()
    feature_columns = numeric_columns + STATIC_NUMERIC + categorical_columns
    return features, feature_columns, numeric_columns + STATIC_NUMERIC, categorical_columns


def target_date_columns(
    df: pd.DataFrame,
    horizon: int,
    value_column: str,
) -> tuple[list[pd.Series], list[pd.Series]]:
    grouped_value = df.groupby("account_id", sort=False)[value_column]
    grouped_date = df.groupby("account_id", sort=False)["period_start"]
    values = [grouped_value.shift(-step) for step in range(1, horizon + 1)]
    dates = [grouped_date.shift(-step) for step in range(1, horizon + 1)]
    return values, dates


def make_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression(n_jobs=-1)),
        ]
    )


def make_training_mask(
    features: pd.DataFrame,
    feature_columns: list[str],
    target_values: list[pd.Series],
    target_dates: list[pd.Series],
    test_start: pd.Timestamp,
) -> pd.Series:
    mask = features[feature_columns].notna().all(axis=1)
    for values, dates in zip(target_values, target_dates):
        mask &= values.notna()
        mask &= dates < test_start
    return mask


def static_by_account(df: pd.DataFrame) -> pd.DataFrame:
    static = (
        df.groupby("account_id", sort=True)[STATIC_NUMERIC + STATIC_CATEGORICAL]
        .first()
        .copy()
    )
    static["birth_year"] = pd.to_numeric(static["birth_year"], errors="coerce")
    for column in STATIC_CATEGORICAL:
        static[column] = static[column].astype("string").fillna("missing")
    return static


def make_history_by_account(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
) -> dict[int, dict[str, list[float]]]:
    pretest = df.loc[df["period_start"] < test_start]
    return {
        int(account_id): {
            flow_name: group[value_column].astype(float).tolist()
            for flow_name, value_column in FLOW_COLUMNS.items()
        }
        for account_id, group in pretest.groupby("account_id", sort=True)
    }


def make_origin_features(
    histories: dict[int, dict[str, list[float]]],
    static: pd.DataFrame,
    frequency: str,
    include_history_features: bool = True,
) -> pd.DataFrame:
    config = FREQUENCY_CONFIG[frequency]
    rows: list[dict[str, object]] = []
    for account_id in sorted(histories):
        row: dict[str, object] = {"account_id": account_id}
        if include_history_features:
            for flow_name in FLOW_COLUMNS:
                history = np.asarray(histories[account_id][flow_name], dtype=float)
                for lag in config["lags"]:
                    row[f"{flow_name}_lag_{lag}"] = history[-1 - lag] if len(history) > lag else np.nan
                for window in config["windows"]:
                    values = history[-window:]
                    if len(values) < window:
                        for operation in ("mean", "median", "std"):
                            row[f"{flow_name}_rolling_{operation}_{window}"] = np.nan
                    else:
                        row[f"{flow_name}_rolling_mean_{window}"] = float(np.mean(values))
                        row[f"{flow_name}_rolling_median_{window}"] = float(np.median(values))
                        row[f"{flow_name}_rolling_std_{window}"] = float(np.std(values, ddof=1))
        for column in STATIC_NUMERIC + STATIC_CATEGORICAL:
            row[column] = static.loc[account_id, column]
        rows.append(row)
    return pd.DataFrame(rows).set_index("account_id")


def test_actual_lookup(
    df: pd.DataFrame,
    test_dates: list[pd.Timestamp],
) -> dict[tuple[int, pd.Timestamp], dict[str, float]]:
    test = df.loc[
        df["period_start"].isin(test_dates),
        ["account_id", "period_start", *FLOW_COLUMNS.values()],
    ]
    return {
        (int(row.account_id), pd.Timestamp(row.period_start)): {
            flow_name: float(getattr(row, value_column))
            for flow_name, value_column in FLOW_COLUMNS.items()
        }
        for row in test.itertuples(index=False)
    }


def fit_recursive_model(data: ForecastData) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    target_values = {}
    target_dates = {}
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
    models: dict[str, Pipeline] = {}
    training_rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        y_train = target_values[flow_name][0].loc[mask].astype(float)
        model = make_pipeline(data.numeric_columns, data.categorical_columns)
        model.fit(X_train, y_train)
        models[flow_name] = model
        training_rows.append(
            {
                "target": flow_name,
                "n_train_rows": int(len(X_train)),
                "n_features_raw": int(len(data.feature_columns)),
            }
        )
    training_info = pd.DataFrame(
        training_rows
    )
    return models, training_info


def fit_direct_model(data: ForecastData) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    target_values = {}
    target_dates = {}
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
    models: dict[str, Pipeline] = {}
    training_rows: list[dict[str, object]] = []
    for flow_name in FLOW_TARGETS:
        y_train = np.column_stack(
            [values.loc[mask].astype(float).to_numpy() for values in target_values[flow_name]]
        )
        model = make_pipeline(data.numeric_columns, data.categorical_columns)
        model.fit(X_train, y_train)
        models[flow_name] = model
        training_rows.append(
            {
                "target": flow_name,
                "n_train_rows": int(len(X_train)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_outputs": int(data.horizon),
            }
        )
    return models, pd.DataFrame(training_rows)


def recursive_forecast(data: ForecastData, models: dict[str, Pipeline]) -> pd.DataFrame:
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
        # 음수 예측값 제거
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
                    "origin_period": data.test_dates[0] - FREQUENCY_CONFIG[data.frequency]["step"] + (step - 1) * FREQUENCY_CONFIG[data.frequency]["step"],
                    "horizon_step": step,
                    "actual_inflow": actual[(account_id, target_date)]["inflow"],
                    "actual_outflow": actual[(account_id, target_date)]["outflow"],
                    "prediction_inflow_raw": float(raw_inflow[index]),
                    "prediction_inflow": predicted_inflow,
                    "prediction_outflow_raw": float(raw_outflow[index]),
                    "prediction_outflow": predicted_outflow,
                }
            )
            # This is the critical recursive step: only predictions, not
            # actual test flows, are added to the history buffers.
            histories[account_id]["inflow"].append(predicted_inflow)
            histories[account_id]["outflow"].append(predicted_outflow)

    result = pd.DataFrame(records)
    return add_derived_flow_columns(result)


def direct_forecast(data: ForecastData, models: dict[str, Pipeline]) -> pd.DataFrame:
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
    raw_inflow_matrix = np.asarray(models["inflow"].predict(X_origin), dtype=float)
    raw_outflow_matrix = np.asarray(models["outflow"].predict(X_origin), dtype=float)
    inflow_matrix = np.clip(raw_inflow_matrix, 0.0, None)
    outflow_matrix = np.clip(raw_outflow_matrix, 0.0, None)
    records: list[dict[str, object]] = []
    account_ids = sorted(histories)
    origin_period = data.test_dates[0] - FREQUENCY_CONFIG[data.frequency]["step"]

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
    result = pd.DataFrame(records)
    return add_derived_flow_columns(result)


def add_derived_flow_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add net-flow columns and legacy outflow aliases to a forecast table."""
    predictions = predictions.copy()
    predictions["actual_net_flow"] = predictions["actual_inflow"] - predictions["actual_outflow"]
    predictions["prediction_net_flow_raw"] = (
        predictions["prediction_inflow_raw"] - predictions["prediction_outflow_raw"]
    )
    predictions["prediction_net_flow"] = (
        predictions["prediction_inflow"] - predictions["prediction_outflow"]
    )
    # Keep the old outflow column names for compatibility with existing
    # outflow-only experiment scripts and downstream readers.
    predictions["prediction_raw"] = predictions["prediction_outflow_raw"]
    predictions["prediction"] = predictions["prediction_outflow"]
    return predictions


def train_normalization_scale(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    value_column: str = "outflow_amount",
) -> pd.Series:
    """Return one positive scale per account from pre-test observations.

    Normalizing by each account's historical mean makes monthly and weekly
    errors dimensionless while preserving account-level differences in size.
    The lower bound only protects accounts with an all-zero history.
    """
    pretest = df.loc[df["period_start"] < test_start]
    scale = pretest.groupby("account_id")[value_column].mean().abs()
    # A zero-history account has no meaningful scale for a normalized error.
    # Keep it as NaN so normalized metrics can exclude it and report the count.
    return scale.where(scale > 0).clip(lower=1.0)


def calculate_metrics(
    predictions: pd.DataFrame,
    normalization_scale: pd.Series,
    actual_column: str = "actual_outflow",
    prediction_column: str = "prediction",
    raw_prediction_column: str | None = "prediction_raw",
    under_weight: float = 2.0,
    over_weight: float = 1.0,
    scale_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Calculate raw, normalized, and asymmetric forecast metrics.

    ``error = prediction - actual``.  Therefore a negative error is an
    underprediction.  The asymmetric metrics are evaluation metrics only:
    LinearRegression itself remains the ordinary least-squares baseline.
    """
    if under_weight <= 0 or over_weight <= 0:
        raise ValueError("under_weight and over_weight must be positive")

    rows: list[dict[str, object]] = []
    groups: list[tuple[object, pd.DataFrame]] = [("all", predictions)]
    if "horizon_step" in predictions.columns:
        groups.extend(predictions.groupby("horizon_step", sort=True))
    for horizon_step, group in groups:
        actual = group[actual_column].to_numpy(dtype=float)
        predicted = group[prediction_column].to_numpy(dtype=float)
        error = predicted - actual
        group_scale = group["account_id"].map(normalization_scale).to_numpy(dtype=float)
        group_scale = group_scale * scale_multiplier
        valid_scale = np.isfinite(group_scale) & (group_scale > 0)
        denominator = float(np.abs(actual).sum())
        under_error = np.maximum(-error, 0.0)
        over_error = np.maximum(error, 0.0)
        asym_weight = np.where(error < 0, under_weight, over_weight)
        normalized_error = error[valid_scale] / group_scale[valid_scale]
        rows.append(
            {
                "horizon_step": horizon_step,
                "n_observations": int(len(group)),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
                "bias": float(error.mean()),
                "underprediction_rate": float((error < 0).mean()),
                "underprediction_mae": float(under_error.mean()),
                "overprediction_mae": float(over_error.mean()),
                "asymmetric_mae": float(np.mean(asym_weight * np.abs(error))),
                "asymmetric_rmse": float(np.sqrt(np.mean(asym_weight * error**2))),
                "n_normalized_observations": int(valid_scale.sum()),
                "zero_scale_rate": float((~valid_scale).mean()),
                "nmae_train_mean": float(np.mean(np.abs(normalized_error))) if valid_scale.any() else np.nan,
                "nrmse_train_mean": float(np.sqrt(np.mean(normalized_error**2))) if valid_scale.any() else np.nan,
                "negative_raw_prediction_rate": (
                    float((group[raw_prediction_column] < 0).mean())
                    if raw_prediction_column is not None
                    else np.nan
                ),
                "under_weight": under_weight,
                "over_weight": over_weight,
            }
        )
    return pd.DataFrame(rows)


def save_forecast_plot(predictions: pd.DataFrame, frequency: str, method: str, output_path: Path) -> None:
    aggregate = (
        predictions.groupby("period_start", as_index=False)[
            [
                "actual_inflow",
                "prediction_inflow",
                "actual_outflow",
                "prediction_outflow",
                "actual_net_flow",
                "prediction_net_flow",
            ]
        ]
        .sum()
        .sort_values("period_start")
    )
    series = [
        ("Inflow", "actual_inflow", "prediction_inflow", "#2ca02c"),
        ("Outflow", "actual_outflow", "prediction_outflow", "#d62728"),
        ("Net flow", "actual_net_flow", "prediction_net_flow", "#9467bd"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, (title, actual_column, prediction_column, color) in zip(axes, series):
        ax.plot(aggregate["period_start"], aggregate[actual_column], label=f"Actual {title.lower()}", linewidth=2, color="#1f77b4")
        ax.plot(aggregate["period_start"], aggregate[prediction_column], label=f"Predicted {title.lower()}", linewidth=2, color=color)
        ax.set_title(title)
        ax.set_ylabel("Amount")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"Test-period aggregate flow — {frequency} / {method}", fontsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_coefficients(model: Pipeline, output_path: Path, horizon: int) -> None:
    preprocess = model.named_steps["preprocess"]
    regressor = model.named_steps["regressor"]
    names = preprocess.get_feature_names_out()
    coefficients = np.asarray(regressor.coef_)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    result = pd.DataFrame({"feature": names})
    for index in range(coefficients.shape[0]):
        result[f"coefficient_h{index + 1}"] = coefficients[index]
    result["mean_abs_coefficient"] = np.abs(coefficients).mean(axis=0)
    result.sort_values("mean_abs_coefficient", ascending=False).to_csv(output_path, index=False)


def create_forecast_data(
    frequency: str,
    account_ids: list[int] | None,
    include_history_features: bool = True,
) -> ForecastData:
    config = FREQUENCY_CONFIG[frequency]
    df = read_master(frequency, account_ids)
    test_start = derive_test_start(frequency, df)
    all_dates = sorted(pd.Timestamp(value) for value in df["period_start"].unique())
    test_dates = [date for date in all_dates if date >= test_start][: config["horizon"]]
    if len(test_dates) != config["horizon"]:
        raise ValueError(f"Expected {config['horizon']} test dates, found {len(test_dates)}")
    feature_table, feature_columns, numeric_columns, categorical_columns = build_feature_table(
        df,
        frequency,
        include_history_features=include_history_features,
    )
    return ForecastData(
        frequency=frequency,
        horizon=config["horizon"],
        include_history_features=include_history_features,
        df=df,
        feature_table=feature_table,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        test_start=test_start,
        test_dates=test_dates,
        account_ids=sorted(df["account_id"].unique().tolist()),
    )


def run_one(
    frequency: str,
    method: str,
    output_dir: Path,
    account_ids: list[int] | None,
    under_weight: float,
    over_weight: float,
    include_history_features: bool = True,
) -> pd.DataFrame:
    data = create_forecast_data(
        frequency,
        account_ids,
        include_history_features=include_history_features,
    )
    if method == "recursive":
        models, training_info = fit_recursive_model(data)
        predictions = recursive_forecast(data, models)
    else:
        models, training_info = fit_direct_model(data)
        predictions = direct_forecast(data, models)

    run_dir = output_dir / f"{frequency}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.insert(0, "frequency", frequency)
    predictions.insert(1, "method", method)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    # Keep a typed, compact version for downstream analysis and plotting.
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
    for target, actual_column, prediction_column, raw_column, scale_column, target_under_weight, target_over_weight in metric_specs:
        scale = train_normalization_scale(df=scale_df, test_start=data.test_start, value_column=scale_column)
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
            )
        )
    pd.concat(period_metric_rows, ignore_index=True).to_csv(
        run_dir / "metrics.csv", index=False
    )

    # Aggregate all forecast periods to one comparable horizon total per
    # account.  This is the primary monthly-vs-weekly comparison output.
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
    for target, actual_column, prediction_column, raw_column, scale_column, target_under_weight, target_over_weight in metric_specs:
        scale = train_normalization_scale(df=scale_df, test_start=data.test_start, value_column=scale_column)
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
                horizon_periods=data.horizon,
                under_weight=target_under_weight,
                over_weight=target_over_weight,
            )
        )
    horizon_metrics = pd.concat(horizon_metric_rows, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[horizon_metrics["horizon_step"].eq("horizon_total")]
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)
    training_info.assign(frequency=frequency, method=method).to_csv(
        run_dir / "training_info.csv", index=False
    )
    save_forecast_plot(predictions, frequency, method, run_dir / "aggregate_test_forecast.png")
    for target, model in models.items():
        save_coefficients(
            model,
            run_dir / f"{target}_coefficients.csv",
            data.horizon if method == "direct" else 1,
        )
    joblib.dump(models, run_dir / "model.joblib")

    metadata = {
        "frequency": frequency,
        "method": method,
        "horizon": data.horizon,
        "test_start": str(data.test_start.date()),
        "test_dates": [str(date.date()) for date in data.test_dates],
        "n_accounts": len(data.account_ids),
        "raw_columns_read": RAW_COLUMNS,
        "feature_columns": data.feature_columns,
        "target_models": list(FLOW_TARGETS),
        "recursive_uses_predictions_in_history": method == "recursive",
        "direct_predicts_all_horizon_steps_at_once": method == "direct",
        "under_weight": under_weight,
        "over_weight": over_weight,
        "normalization_scale": "target-specific pre-test mean absolute flow per account",
        "include_history_features": include_history_features,
        "excluded_history_feature_families": []
        if include_history_features
        else ["lag", "rolling_mean", "rolling_median", "rolling_std"],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"completed {frequency}/{method}: {len(data.account_ids):,} accounts, horizon={data.horizon}")
    return horizon_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument(
        "--exclude-lag-rolling",
        action="store_true",
        help="Retrain using only static features; exclude all lag and rolling features.",
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

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    horizon_rows: list[pd.DataFrame] = []
    for frequency in frequencies:
        account_ids = args.account_ids
        if account_ids is None and args.max_accounts is not None:
            account_ids = get_available_account_ids(frequency, args.max_accounts)
        # With no --account-ids and no --max-accounts, account_ids remains
        # None and read_master loads every account in the parquet file.
        for method in methods:
            horizon_rows.append(
                run_one(
                    frequency,
                    method,
                    args.output_dir,
                    account_ids,
                    under_weight=args.under_weight,
                    over_weight=args.over_weight,
                    include_history_features=not args.exclude_lag_rolling,
                )
            )
    if horizon_rows:
        pd.concat(horizon_rows, ignore_index=True).to_csv(
            args.output_dir / "horizon_comparison_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
