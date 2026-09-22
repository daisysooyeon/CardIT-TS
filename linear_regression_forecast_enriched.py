"""Linear Regression forecasts using the enriched Berka master tables.

This is a separate version of the original LR script.  It uses calendar,
holiday, month-boundary, and contractual loan-repayment covariates from
``*_enriched.parquet``.  Recursive history features remain prediction-driven;
known future covariates are looked up from a schedule prepared at the test
origin.

Direct models are horizon-specific in this version so a target month's/week's
known covariates are aligned with the corresponding target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

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
    make_pipeline,
    save_coefficients,
    target_date_columns,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "linear_regression_enriched"


def _training_mask(
    data: EnrichedForecastData,
    horizon_index: int,
) -> pd.Series:
    """Return a leakage-safe mask for one target horizon."""
    feature_table = data.feature_tables[horizon_index]
    target_values: list[pd.Series] = []
    target_dates: list[pd.Series] = []
    for value_column in FLOW_COLUMNS.values():
        values, dates = target_date_columns(
            data.df,
            horizon=horizon_index + 1,
            value_column=value_column,
        )
        target_values.append(values[0])
        target_dates.append(dates[0])

    mask = feature_table[data.feature_columns].notna().all(axis=1)
    for values, dates in zip(target_values, target_dates):
        mask &= values.notna()
        mask &= dates < data.test_start
    return mask


def fit_recursive_model(
    data: EnrichedForecastData,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    mask = _training_mask(data, 0)
    X_train = data.feature_tables[0].loc[mask, data.feature_columns]
    if X_train.empty:
        raise ValueError("No training rows are available for enriched LR")

    models: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []
    for flow_name, value_column in FLOW_COLUMNS.items():
        target_values, _ = target_date_columns(
            data.df, horizon=1, value_column=value_column
        )
        y_train = target_values[0].loc[mask].astype(float)
        model = make_pipeline(data.numeric_columns, data.categorical_columns)
        model.fit(X_train, y_train)
        models[flow_name] = model
        rows.append(
            {
                "target": flow_name,
                "n_train_rows": int(len(X_train)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_models": 1,
                "known_features": ",".join(data.known_feature_names),
            }
        )
    return models, pd.DataFrame(rows)


def fit_direct_model(
    data: EnrichedForecastData,
) -> tuple[dict[str, list[Pipeline]], pd.DataFrame]:
    """Fit one LR model per flow and target horizon."""
    models: dict[str, list[Pipeline]] = {}
    rows: list[dict[str, object]] = []
    for flow_name, value_column in FLOW_COLUMNS.items():
        target_values, _ = target_date_columns(
            data.df, horizon=data.horizon, value_column=value_column
        )
        flow_models: list[Pipeline] = []
        train_counts: list[int] = []
        for horizon_index in range(data.horizon):
            mask = _training_mask(data, horizon_index)
            X_train = data.feature_tables[horizon_index].loc[
                mask, data.feature_columns
            ]
            if X_train.empty:
                raise ValueError(
                    f"No training rows for {flow_name}, horizon={horizon_index + 1}"
                )
            y_train = target_values[horizon_index].loc[mask].astype(float)
            model = make_pipeline(data.numeric_columns, data.categorical_columns)
            model.fit(X_train, y_train)
            flow_models.append(model)
            train_counts.append(int(len(X_train)))
        models[flow_name] = flow_models
        rows.append(
            {
                "target": flow_name,
                "n_train_rows_min": int(min(train_counts)),
                "n_train_rows_max": int(max(train_counts)),
                "n_features_raw": int(len(data.feature_columns)),
                "n_models": int(data.horizon),
                "known_features": ",".join(data.known_feature_names),
            }
        )
    return models, pd.DataFrame(rows)


def _known_record_values(
    known: pd.DataFrame,
    account_id: int,
    feature_names: list[str],
) -> dict[str, object]:
    return {
        name: float(known.loc[account_id, name])
        for name in feature_names
    }


def recursive_forecast(
    data: EnrichedForecastData,
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    records: list[dict[str, object]] = []

    for step, target_date in enumerate(data.test_dates, start=1):
        known = data.test_known_features[target_date]
        X_origin = make_origin_features(
            histories,
            static,
            data.frequency,
            target_date,
            known,
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
            record: dict[str, object] = {
                "account_id": account_id,
                "period_start": target_date,
                "origin_period": target_date
                - FREQUENCY_CONFIG[data.frequency]["step"],
                "horizon_step": step,
                "actual_inflow": actual[(account_id, target_date)]["inflow"],
                "actual_outflow": actual[(account_id, target_date)]["outflow"],
                "prediction_inflow_raw": float(raw_inflow[index]),
                "prediction_inflow": predicted_inflow,
                "prediction_outflow_raw": float(raw_outflow[index]),
                "prediction_outflow": predicted_outflow,
            }
            record.update(_known_record_values(known, account_id, data.known_feature_names))
            records.append(record)
            histories[account_id]["inflow"].append(predicted_inflow)
            histories[account_id]["outflow"].append(predicted_outflow)

    return add_derived_flow_columns(
        pd.DataFrame(records).sort_values(["account_id", "period_start"])
    )


def direct_forecast(
    data: EnrichedForecastData,
    models: dict[str, list[Pipeline]],
) -> pd.DataFrame:
    """Forecast each future period from one origin with known covariates."""
    histories = make_history_by_account(data.df, data.test_start)
    static = static_by_account(data.df)
    actual = test_actual_lookup(data.df, data.test_dates)
    account_ids = sorted(histories)
    records: list[dict[str, object]] = []

    for step, target_date in enumerate(data.test_dates, start=1):
        known = data.test_known_features[target_date]
        X_origin = make_origin_features(
            histories,
            static,
            data.frequency,
            target_date,
            known,
            data.include_history_features,
        )
        X_origin = X_origin[data.feature_columns]
        raw_inflow = np.asarray(
            models["inflow"][step - 1].predict(X_origin), dtype=float
        )
        raw_outflow = np.asarray(
            models["outflow"][step - 1].predict(X_origin), dtype=float
        )
        inflow_predictions = np.clip(raw_inflow, 0.0, None)
        outflow_predictions = np.clip(raw_outflow, 0.0, None)

        for index, account_id in enumerate(account_ids):
            account_id = int(account_id)
            record: dict[str, object] = {
                "account_id": account_id,
                "period_start": target_date,
                "origin_period": data.test_dates[0]
                - FREQUENCY_CONFIG[data.frequency]["step"],
                "horizon_step": step,
                "actual_inflow": actual[(account_id, target_date)]["inflow"],
                "actual_outflow": actual[(account_id, target_date)]["outflow"],
                "prediction_inflow_raw": float(raw_inflow[index]),
                "prediction_inflow": float(inflow_predictions[index]),
                "prediction_outflow_raw": float(raw_outflow[index]),
                "prediction_outflow": float(outflow_predictions[index]),
            }
            record.update(_known_record_values(known, account_id, data.known_feature_names))
            records.append(record)

    return add_derived_flow_columns(pd.DataFrame(records))


def _save_models(
    models: dict[str, Pipeline] | dict[str, list[Pipeline]],
    run_dir: Path,
    method: str,
) -> None:
    joblib.dump(models, run_dir / "model.joblib")
    for target, target_models in models.items():
        if isinstance(target_models, list):
            for horizon_step, model in enumerate(target_models, start=1):
                save_coefficients(
                    model,
                    run_dir / f"{target}_coefficients_h{horizon_step}.csv",
                    1,
                )
        else:
            save_coefficients(
                target_models,
                run_dir / f"{target}_coefficients.csv",
                1,
            )


def run_one(
    frequency: str,
    method: str,
    output_dir: Path,
    account_ids: list[int] | None,
    under_weight: float,
    over_weight: float,
    include_history_features: bool,
) -> pd.DataFrame:
    data = create_enriched_forecast_data(
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

    run_dir, horizon_metrics = save_enriched_run_outputs(
        data,
        predictions,
        training_info,
        output_dir,
        method,
        under_weight=under_weight,
        over_weight=over_weight,
    )
    _save_models(models, run_dir, method)
    metadata = {
        "frequency": frequency,
        "method": method,
        "master_file": str(enriched_master_path(frequency)),
        "known_features": data.known_feature_names,
        "known_features_are_target_aligned": True,
        "loan_schedule_amount_source": "loan.payments",
        "loan_schedule_as_of_mask": "loan.date <= origin period end",
        "recursive_history_uses_predictions": method == "recursive",
        "include_history_features": include_history_features,
        "n_accounts": len(data.account_ids),
        "test_start": str(data.test_start.date()),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"completed enriched LR {frequency}/{method}: "
        f"{len(data.account_ids):,} accounts, horizon={data.horizon}"
    )
    return horizon_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--method", choices=["recursive", "direct", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--exclude-lag-rolling", action="store_true")
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    args = parser.parse_args()

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    horizon_rows: list[pd.DataFrame] = []
    for frequency in frequencies:
        account_ids = args.account_ids
        if account_ids is None and args.max_accounts is not None:
            account_ids = get_available_enriched_account_ids(
                frequency, args.max_accounts
            )
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
