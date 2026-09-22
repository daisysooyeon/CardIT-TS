"""Compare existing-master and frequency-aware lag/window configurations.

The comparison uses only pre-test rolling validation windows.  The final test
period is not used to choose a lag/window configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import linear_regression_forecast as lr


CONFIG_PROFILES = {
    "master_existing": {
        "monthly": {"lags": [0, 1, 2, 3, 7, 28], "windows": [3, 6, 12]},
        "weekly": {"lags": [0, 1, 2, 3, 7, 28], "windows": [4, 8, 13]},
    },
    "frequency_aware": {
        "monthly": {"lags": [0, 1, 2, 3, 6, 12], "windows": [3, 6, 12]},
        "weekly": {"lags": [0, 1, 2, 4, 8, 13, 26, 52], "windows": [4, 8, 13, 26, 52]},
    },
}


def validation_starts(data: lr.ForecastData) -> list[pd.Timestamp]:
    all_dates = sorted(pd.Timestamp(value) for value in data.df["period_start"].unique())
    pretest_dates = [date for date in all_dates if date < lr.derive_test_start(data.frequency, data.df)]
    # Two non-overlapping validation windows immediately before the final test.
    return [pretest_dates[-2 * data.horizon], pretest_dates[-data.horizon]]


def with_validation_window(data: lr.ForecastData, start: pd.Timestamp) -> lr.ForecastData:
    all_dates = sorted(pd.Timestamp(value) for value in data.df["period_start"].unique())
    dates = [date for date in all_dates if date >= start][: data.horizon]
    return replace(data, test_start=start, test_dates=dates)


def horizon_metrics(
    predictions: pd.DataFrame,
    data: lr.ForecastData,
    under_weight: float,
    over_weight: float,
) -> pd.DataFrame:
    scale = lr.train_normalization_scale(data.df, data.test_start)
    aggregate = predictions.groupby("account_id", as_index=False)[
        ["actual_outflow", "prediction", "prediction_raw"]
    ].sum()
    aggregate["horizon_step"] = "horizon_total"
    return lr.calculate_metrics(
        aggregate,
        scale,
        under_weight=under_weight,
        over_weight=over_weight,
        scale_multiplier=data.horizon,
    ).loc[lambda frame: frame["horizon_step"].eq("horizon_total")]


def run_profile(
    frequency: str,
    profile_name: str,
    account_ids: list[int] | None,
    under_weight: float,
    over_weight: float,
) -> pd.DataFrame:
    original_config = lr.FREQUENCY_CONFIG[frequency].copy()
    lr.FREQUENCY_CONFIG[frequency] = {
        **original_config,
        **CONFIG_PROFILES[profile_name][frequency],
    }
    try:
        data = lr.create_forecast_data(frequency, account_ids)
        rows: list[dict[str, object]] = []
        for validation_index, start in enumerate(validation_starts(data), start=1):
            validation_data = with_validation_window(data, start)
            for method in ("recursive", "direct"):
                if method == "recursive":
                    model, training_info = lr.fit_recursive_model(validation_data)
                    predictions = lr.recursive_forecast(validation_data, model)
                else:
                    model, training_info = lr.fit_direct_model(validation_data)
                    predictions = lr.direct_forecast(validation_data, model)
                metrics = horizon_metrics(predictions, validation_data, under_weight, over_weight)
                row = metrics.iloc[0].to_dict()
                row.update(
                    {
                        "frequency": frequency,
                        "profile": profile_name,
                        "method": method,
                        "validation_window": validation_index,
                        "validation_start": str(start.date()),
                        "horizon_periods": data.horizon,
                        "n_train_rows": int(training_info.iloc[0]["n_train_rows"]),
                        "lags": ",".join(map(str, lr.FREQUENCY_CONFIG[frequency]["lags"])),
                        "windows": ",".join(map(str, lr.FREQUENCY_CONFIG[frequency]["windows"])),
                        "under_weight": under_weight,
                        "over_weight": over_weight,
                    }
                )
                rows.append(row)
                print(
                    f"completed {frequency}/{profile_name}/{method}/validation_{validation_index}"
                )
        return pd.DataFrame(rows)
    finally:
        lr.FREQUENCY_CONFIG[frequency] = original_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/linear_regression/lag_config_experiment"))
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    args = parser.parse_args()

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    all_rows: list[pd.DataFrame] = []
    for frequency in frequencies:
        account_ids = args.account_ids
        if account_ids is None and args.max_accounts is not None:
            account_ids = lr.get_available_account_ids(frequency, args.max_accounts)
        for profile_name in CONFIG_PROFILES:
            all_rows.append(
                run_profile(
                    frequency,
                    profile_name,
                    account_ids,
                    args.under_weight,
                    args.over_weight,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(args.output_dir / "validation_results.csv", index=False)
    summary = (
        results.groupby(["frequency", "profile", "method"], as_index=False)
        .agg(
            validation_windows=("validation_window", "nunique"),
            mean_nrmse=("nrmse_train_mean", "mean"),
            mean_nmae=("nmae_train_mean", "mean"),
            mean_asymmetric_rmse=("asymmetric_rmse", "mean"),
            mean_wape=("wape", "mean"),
            mean_underprediction_rate=("underprediction_rate", "mean"),
        )
        .sort_values(["frequency", "method", "mean_nrmse"])
    )
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    print("\\nValidation summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
