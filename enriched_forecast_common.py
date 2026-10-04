"""Shared leakage-safe data preparation for enriched Berka forecasts.

This module is used by the enriched LR, XGBoost, LightGBM, and DeepAR
scripts.  It keeps the following distinction explicit:

* the enriched parquet contains the canonical contractual schedule;
* training features use only loan contracts known by the origin period;
* test-time calendar and loan features are precomputed once and looked up;
* history-dependent lag/rolling features are still rebuilt from the evolving
  observed/predicted histories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_cost_schedule import (
    first_due_period as _first_due_period,
    parse_berka_date as _parse_berka_date,
)
from linear_regression_forecast import (
    FLOW_COLUMNS,
    FLOW_TARGETS,
    FREQUENCY_CONFIG,
    STATIC_CATEGORICAL,
    STATIC_NUMERIC,
    add_derived_flow_columns,
    calculate_metrics,
    derive_test_start,
    rolling_series,
    save_forecast_plot,
    target_date_columns,
    train_normalization_scale,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Berka Dataset" / "Berka Dataset"
MASTER_DIR = DATA_DIR / "master_tables"
LOAN_PATH = DATA_DIR / "loan.csv"

MONTHLY_KNOWN_FEATURES = [
    "holiday_count",
    "has_holiday",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "scheduled_loan_repayment",
]
WEEKLY_KNOWN_FEATURES = [
    "holiday_count",
    "has_holiday",
    "is_week_before_holiday",
    "is_week_after_holiday",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "week_sin",
    "week_cos",
    "has_month_start",
    "has_month_end",
    "scheduled_loan_repayment",
]


@dataclass
class EnrichedForecastData:
    frequency: str
    horizon: int
    include_history_features: bool
    df: pd.DataFrame
    feature_tables: list[pd.DataFrame]
    feature_table: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    known_feature_names: list[str]
    test_start: pd.Timestamp
    test_dates: list[pd.Timestamp]
    account_ids: list[int]
    test_known_features: dict[pd.Timestamp, pd.DataFrame]


def enriched_master_path(frequency: str) -> Path:
    suffix = "m" if frequency == "monthly" else "w"
    return MASTER_DIR / f"berka_master_{suffix}_enriched.parquet"


def known_feature_names(frequency: str) -> list[str]:
    return (
        MONTHLY_KNOWN_FEATURES.copy()
        if frequency == "monthly"
        else WEEKLY_KNOWN_FEATURES.copy()
    )


def read_enriched_master(
    frequency: str,
    account_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Read an enriched master and optionally restrict accounts."""
    path = enriched_master_path(frequency)
    if not path.exists():
        raise FileNotFoundError(
            f"Enriched master not found: {path}. "
            "Run enrich_master_features.py first."
        )
    df = pd.read_parquet(path)
    df["period_start"] = pd.to_datetime(df["period_start"]).dt.normalize()
    df["account_id"] = df["account_id"].astype(int)
    required = [
        "account_id",
        "period_start",
        "inflow_amount",
        "outflow_amount",
        *STATIC_NUMERIC,
        *STATIC_CATEGORICAL,
        *known_feature_names(frequency),
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Enriched master is missing columns: {missing}")
    if account_ids:
        requested = set(int(value) for value in account_ids)
        available = set(df["account_id"].unique())
        missing_accounts = sorted(requested - available)
        if missing_accounts:
            raise ValueError(f"Unknown account_id(s): {missing_accounts}")
        df = df.loc[df["account_id"].isin(requested)].copy()
    return df.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def get_available_enriched_account_ids(
    frequency: str,
    max_accounts: int | None = None,
) -> list[int]:
    path = enriched_master_path(frequency)
    account_ids = (
        pd.read_parquet(path, columns=["account_id"])["account_id"]
        .drop_duplicates()
        .sort_values()
        .astype(int)
        .tolist()
    )
    return account_ids if max_accounts is None else account_ids[:max_accounts]


def load_loan_events(frequency: str) -> pd.DataFrame:
    """Return one row per contractual due date with its loan start date.

    Keeping ``loan_date`` on each event lets training features mask out a
    repayment whose loan contract was not known at the origin.
    """
    loans = pd.read_csv(
        LOAN_PATH,
        sep=";",
        usecols=["account_id", "date", "payments", "duration"],
    )
    loans["account_id"] = loans["account_id"].astype(int)
    loans["loan_date"] = _parse_berka_date(loans.pop("date"))
    loans["payments"] = pd.to_numeric(loans["payments"], errors="coerce")
    loans["duration"] = pd.to_numeric(loans["duration"], errors="coerce")
    loans = loans.dropna(subset=["loan_date", "payments", "duration"])
    loans["duration"] = loans["duration"].astype(int)

    records: list[dict[str, object]] = []
    for loan in loans.itertuples(index=False):
        first_period = _first_due_period(loan.loan_date)
        for offset in range(int(loan.duration)):
            due_period = first_period + offset
            due_date = due_period.to_timestamp() + pd.Timedelta(days=11)
            if frequency == "monthly":
                period_start = due_period.to_timestamp()
            else:
                days_since_sunday = (due_date.dayofweek + 1) % 7
                period_start = (
                    due_date - pd.Timedelta(days=days_since_sunday)
                ).normalize()
            records.append(
                {
                    "account_id": int(loan.account_id),
                    "period_start": period_start,
                    "loan_date": pd.Timestamp(loan.loan_date),
                    "scheduled_amount": float(loan.payments),
                }
            )
    return pd.DataFrame(
        records,
        columns=["account_id", "period_start", "loan_date", "scheduled_amount"],
    )


def _period_end(period_start: pd.Series, frequency: str) -> pd.Series:
    if frequency == "monthly":
        return period_start + pd.offsets.MonthEnd(0)
    return period_start + pd.Timedelta(days=6)


def _history_feature_table(
    df: pd.DataFrame,
    frequency: str,
    include_history_features: bool,
) -> pd.DataFrame:
    config = FREQUENCY_CONFIG[frequency]
    features = pd.DataFrame(index=df.index)
    if not include_history_features:
        return features

    for flow_name, value_column in FLOW_COLUMNS.items():
        grouped = df.groupby("account_id", sort=False)[value_column]
        for lag in config["lags"]:
            features[f"{flow_name}_lag_{lag}"] = grouped.shift(lag)
        for window in config["windows"]:
            for operation in ("mean", "median", "std"):
                name = f"{flow_name}_rolling_{operation}_{window}"
                features[name] = rolling_series(grouped, window, operation)
    return features


def _calendar_lookup(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    names = known_feature_names(frequency)
    names_without_schedule = [name for name in names if name != "scheduled_loan_repayment"]
    lookup = (
        df[["period_start", *names_without_schedule]]
        .drop_duplicates("period_start")
        .set_index("period_start")
        .sort_index()
    )
    return lookup


def _known_schedule_for_rows(
    df: pd.DataFrame,
    target_dates: pd.Series,
    frequency: str,
    loan_events: pd.DataFrame,
) -> pd.Series:
    """Compute target-period scheduled amounts known at each row's origin."""
    result = pd.Series(0.0, index=df.index, dtype=float)
    if loan_events.empty:
        return result

    rows = pd.DataFrame(
        {
            "_row_index": df.index,
            "account_id": df["account_id"].to_numpy(),
            "target_period": pd.to_datetime(target_dates).to_numpy(),
            "origin_end": _period_end(df["period_start"], frequency).to_numpy(),
        },
        index=df.index,
    )
    rows = rows.dropna(subset=["target_period"])
    if rows.empty:
        return result

    events = loan_events.rename(columns={"period_start": "target_period"})
    matched = rows.merge(
        events,
        on=["account_id", "target_period"],
        how="inner",
        validate="many_to_many",
    )
    matched = matched.loc[matched["loan_date"] <= matched["origin_end"]]
    if matched.empty:
        return result
    values = matched.groupby("_row_index", sort=False)["scheduled_amount"].sum()
    result.loc[values.index] = values.astype(float)
    return result


def _test_schedule_lookup(
    account_ids: list[int],
    test_dates: list[pd.Timestamp],
    test_origin_end: pd.Timestamp,
    loan_events: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Series]:
    """Precompute future schedule lookup once for a fixed test origin."""
    result: dict[pd.Timestamp, pd.Series] = {}
    for target_date in test_dates:
        rows = pd.DataFrame(
            {"account_id": account_ids, "period_start": target_date}
        )
        events = loan_events.loc[
            (loan_events["period_start"] == target_date)
            & (loan_events["loan_date"] <= test_origin_end)
        ]
        values = events.groupby("account_id")["scheduled_amount"].sum()
        result[target_date] = (
            pd.Series(account_ids, index=account_ids, dtype="int64")
            .map(values)
            .fillna(0.0)
            .astype(float)
        )
    return result


def _feature_columns(
    frequency: str,
    include_history_features: bool,
) -> tuple[list[str], list[str], list[str]]:
    config = FREQUENCY_CONFIG[frequency]
    history_columns: list[str] = []
    if include_history_features:
        for flow_name in FLOW_COLUMNS:
            history_columns.extend(
                [f"{flow_name}_lag_{lag}" for lag in config["lags"]]
            )
            for window in config["windows"]:
                history_columns.extend(
                    [
                        f"{flow_name}_rolling_{operation}_{window}"
                        for operation in ("mean", "median", "std")
                    ]
                )
    known = known_feature_names(frequency)
    numeric = history_columns + known + STATIC_NUMERIC
    categorical = STATIC_CATEGORICAL.copy()
    return history_columns + known + STATIC_NUMERIC + categorical, numeric, categorical


def _build_feature_table_for_horizon(
    df: pd.DataFrame,
    frequency: str,
    horizon_step: int,
    include_history_features: bool,
    calendar: pd.DataFrame,
    loan_events: pd.DataFrame,
) -> pd.DataFrame:
    features = _history_feature_table(df, frequency, include_history_features)
    target_dates = df.groupby("account_id", sort=False)["period_start"].shift(-horizon_step)
    calendar_values = calendar.reindex(pd.DatetimeIndex(target_dates))
    calendar_values.index = df.index
    for column in calendar.columns:
        features[column] = calendar_values[column].to_numpy()
    features["scheduled_loan_repayment"] = _known_schedule_for_rows(
        df, target_dates, frequency, loan_events
    ).to_numpy()
    for column in STATIC_NUMERIC:
        features[column] = pd.to_numeric(df[column], errors="coerce")
    for column in STATIC_CATEGORICAL:
        features[column] = df[column].astype("string").fillna("missing")
    return features


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


def _history_origin_features(
    histories: dict[int, dict[str, list[float]]],
    frequency: str,
    include_history_features: bool,
) -> pd.DataFrame:
    config = FREQUENCY_CONFIG[frequency]
    rows: list[dict[str, object]] = []
    for account_id in sorted(histories):
        row: dict[str, object] = {"account_id": int(account_id)}
        if include_history_features:
            for flow_name in FLOW_COLUMNS:
                history = np.asarray(histories[account_id][flow_name], dtype=float)
                for lag in config["lags"]:
                    row[f"{flow_name}_lag_{lag}"] = (
                        history[-1 - lag] if len(history) > lag else np.nan
                    )
                for window in config["windows"]:
                    values = history[-window:]
                    for operation in ("mean", "median", "std"):
                        name = f"{flow_name}_rolling_{operation}_{window}"
                        if len(values) < window:
                            row[name] = np.nan
                        elif operation == "mean":
                            row[name] = float(np.mean(values))
                        elif operation == "median":
                            row[name] = float(np.median(values))
                        else:
                            row[name] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).set_index("account_id")


def make_origin_features(
    histories: dict[int, dict[str, list[float]]],
    static: pd.DataFrame,
    frequency: str,
    target_date: pd.Timestamp,
    known_features: pd.DataFrame,
    include_history_features: bool = True,
) -> pd.DataFrame:
    """Combine evolving history features with precomputed target covariates."""
    result = _history_origin_features(
        histories, frequency, include_history_features
    )
    known = known_features.reindex(result.index)
    result = result.join(known[known_feature_names(frequency)])
    result = result.join(static.loc[result.index, STATIC_NUMERIC + STATIC_CATEGORICAL])
    return result


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


def create_enriched_forecast_data(
    frequency: str,
    account_ids: list[int] | None = None,
    include_history_features: bool = True,
) -> EnrichedForecastData:
    df = read_enriched_master(frequency, account_ids)
    test_start = derive_test_start(frequency, df)
    config = FREQUENCY_CONFIG[frequency]
    all_dates = sorted(pd.Timestamp(value) for value in df["period_start"].unique())
    test_dates = [date for date in all_dates if date >= test_start][: config["horizon"]]
    if len(test_dates) != config["horizon"]:
        raise ValueError(
            f"Expected {config['horizon']} test dates, found {len(test_dates)}"
        )

    calendar = _calendar_lookup(df, frequency)
    loan_events = load_loan_events(frequency)
    feature_tables = [
        _build_feature_table_for_horizon(
            df,
            frequency,
            horizon_step,
            include_history_features,
            calendar,
            loan_events,
        )
        for horizon_step in range(1, config["horizon"] + 1)
    ]
    feature_columns, numeric_columns, categorical_columns = _feature_columns(
        frequency, include_history_features
    )

    account_id_values = sorted(int(value) for value in df["account_id"].unique())
    test_origin_end = test_start - pd.Timedelta(days=1)
    schedule_lookup = _test_schedule_lookup(
        account_id_values,
        test_dates,
        test_origin_end,
        loan_events,
    )
    test_known_features: dict[pd.Timestamp, pd.DataFrame] = {}
    for target_date in test_dates:
        known = calendar.loc[[target_date]].copy()
        known = known.loc[known.index.repeat(len(account_id_values))].copy()
        known.index = account_id_values
        known["scheduled_loan_repayment"] = schedule_lookup[target_date].to_numpy()
        test_known_features[target_date] = known[known_feature_names(frequency)]

    return EnrichedForecastData(
        frequency=frequency,
        horizon=config["horizon"],
        include_history_features=include_history_features,
        df=df,
        feature_tables=feature_tables,
        feature_table=feature_tables[0],
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        known_feature_names=known_feature_names(frequency),
        test_start=test_start,
        test_dates=test_dates,
        account_ids=account_id_values,
        test_known_features=test_known_features,
    )


def save_enriched_run_outputs(
    data: EnrichedForecastData,
    predictions: pd.DataFrame,
    training_info: pd.DataFrame,
    output_dir: Path,
    method: str,
    under_weight: float = 2.0,
    over_weight: float = 1.0,
    grid_id: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Save predictions and the same metric files used by prior models."""
    run_dir = output_dir / f"{data.frequency}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    predictions.insert(0, "frequency", data.frequency)
    predictions.insert(1, "method", method)
    if grid_id is not None:
        predictions.insert(2, "grid_id", grid_id)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    scale_df = data.df.copy()
    scale_df["net_flow_for_scale"] = (
        scale_df["inflow_amount"] - scale_df["outflow_amount"]
    )
    metric_specs = [
        ("inflow", "actual_inflow", "prediction_inflow", "prediction_inflow_raw", "inflow_amount", over_weight, under_weight),
        ("outflow", "actual_outflow", "prediction_outflow", "prediction_outflow_raw", "outflow_amount", under_weight, over_weight),
        ("net_flow", "actual_net_flow", "prediction_net_flow", "prediction_net_flow_raw", "net_flow_for_scale", over_weight, under_weight),
    ]
    period_rows: list[pd.DataFrame] = []
    for target, actual_col, prediction_col, raw_col, scale_col, target_under, target_over in metric_specs:
        scale = train_normalization_scale(scale_df, data.test_start, scale_col)
        period_rows.append(
            calculate_metrics(
                predictions,
                scale,
                actual_column=actual_col,
                prediction_column=prediction_col,
                raw_prediction_column=raw_col,
                under_weight=target_under,
                over_weight=target_over,
            ).assign(
                target=target,
                frequency=data.frequency,
                method=method,
                **({"grid_id": grid_id} if grid_id is not None else {}),
            )
        )
    pd.concat(period_rows, ignore_index=True).to_csv(run_dir / "metrics.csv", index=False)

    horizon_predictions = predictions.groupby("account_id", as_index=False)[
        [
            "actual_inflow",
            "actual_outflow",
            "prediction_inflow",
            "prediction_inflow_raw",
            "prediction_outflow",
            "prediction_outflow_raw",
        ]
    ].sum()
    horizon_predictions = add_derived_flow_columns(horizon_predictions)
    horizon_predictions["horizon_step"] = "horizon_total"
    horizon_rows: list[pd.DataFrame] = []
    for target, actual_col, prediction_col, raw_col, scale_col, target_under, target_over in metric_specs:
        scale = train_normalization_scale(scale_df, data.test_start, scale_col)
        horizon_rows.append(
            calculate_metrics(
                horizon_predictions,
                scale,
                actual_column=actual_col,
                prediction_column=prediction_col,
                raw_prediction_column=raw_col,
                under_weight=target_under,
                over_weight=target_over,
                scale_multiplier=data.horizon,
            ).assign(
                target=target,
                frequency=data.frequency,
                method=method,
                horizon_periods=data.horizon,
                under_weight=target_under,
                over_weight=target_over,
                **({"grid_id": grid_id} if grid_id is not None else {}),
            )
        )
    horizon_metrics = pd.concat(horizon_rows, ignore_index=True)
    horizon_metrics = horizon_metrics.loc[
        horizon_metrics["horizon_step"].eq("horizon_total")
    ]
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)
    training_info.assign(frequency=data.frequency, method=method).to_csv(
        run_dir / "training_info.csv", index=False
    )
    save_forecast_plot(
        predictions,
        data.frequency,
        method,
        run_dir / "aggregate_test_forecast.png",
    )
    return run_dir, horizon_metrics
