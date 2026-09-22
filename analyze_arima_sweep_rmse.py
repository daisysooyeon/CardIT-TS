"""Analyze original ARIMA order-sweep results by horizon and period RMSE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    ("inflow", "actual_inflow", "prediction_inflow"),
    ("outflow", "actual_outflow", "prediction_outflow"),
    ("net_flow", "actual_net_flow", "prediction_net_flow"),
]


def read_completed_runs(sweep_dir: Path) -> pd.DataFrame:
    metrics_path = sweep_dir / "horizon_comparison_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    required = {"order_label", "frequency", "method", "target", "rmse"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Missing columns in {metrics_path}: {missing}")
    return metrics.loc[metrics["horizon_step"].eq("horizon_total")].copy()


def save_horizon_best(metrics: pd.DataFrame, output_dir: Path) -> None:
    keys = ["target", "frequency", "method", "order_label"]
    parameter_columns = [
        column
        for column in ("order", "order_label")
        if column in metrics.columns
    ]
    columns = [
        "target",
        "frequency",
        "method",
        "order_label",
        "order",
        "rmse",
        "mae",
        "wape",
        "asymmetric_rmse",
        "nrmse_train_mean",
    ]
    columns = [column for column in columns if column in metrics.columns]

    rows: list[pd.DataFrame] = []
    for group_columns, filename in [
        (["target"], "best_horizon_rmse_all_available.csv"),
        (["frequency", "target"], "best_horizon_rmse_by_frequency.csv"),
        (["frequency", "method", "target"], "best_horizon_rmse_by_frequency_method.csv"),
    ]:
        indices = metrics.groupby(group_columns)["rmse"].idxmin()
        best = metrics.loc[indices, columns].sort_values(group_columns)
        best.to_csv(output_dir / filename, index=False)
        rows.append(best.assign(selection_level="/".join(group_columns)))

    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "best_horizon_rmse_summary.csv", index=False
    )


def read_period_rmse(sweep_dir: Path, completed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    run_columns = ["order_label", "frequency", "method"]
    runs = completed[run_columns].drop_duplicates().sort_values(run_columns)
    for run in runs.itertuples(index=False):
        prediction_path = (
            sweep_dir
            / str(run.order_label)
            / f"{run.frequency}_{run.method}"
            / "predictions.parquet"
        )
        if not prediction_path.exists():
            prediction_path = prediction_path.with_suffix(".csv")
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing predictions for completed run: {prediction_path}")
        frame = pd.read_parquet(prediction_path) if prediction_path.suffix == ".parquet" else pd.read_csv(prediction_path)
        frame["period_start"] = pd.to_datetime(frame["period_start"])
        for target, actual_column, prediction_column in TARGETS:
            work = frame[["period_start", actual_column, prediction_column]].copy()
            error = work[prediction_column].astype(float) - work[actual_column].astype(float)
            work["squared_error"] = error**2
            grouped = work.groupby("period_start", sort=True)
            for period_start, group in grouped:
                rows.append(
                    {
                        "order_label": run.order_label,
                        "frequency": run.frequency,
                        "method": run.method,
                        "target": target,
                        "period_start": period_start,
                        "n_observations": int(len(group)),
                        "rmse": float(np.sqrt(group["squared_error"].mean())),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["frequency", "target", "period_start", "rmse", "method", "order_label"]
    )


def save_period_best(period_rmse: pd.DataFrame, output_dir: Path) -> None:
    period_rmse.to_csv(output_dir / "period_rmse_all_available.csv", index=False)
    group_columns = ["frequency", "target", "period_start"]
    indices = period_rmse.groupby(group_columns)["rmse"].idxmin()
    winners = period_rmse.loc[indices].sort_values(group_columns).copy()
    winners.to_csv(output_dir / "period_rmse_winners.csv", index=False)
    summary = (
        winners.groupby(["frequency", "target", "method", "order_label"], as_index=False)
        .agg(
            periods_won=("period_start", "size"),
            mean_winner_rmse=("rmse", "mean"),
        )
        .sort_values(["frequency", "target", "periods_won", "mean_winner_rmse"], ascending=[True, True, False, True])
    )
    summary.to_csv(output_dir / "period_rmse_winner_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/arima_order_grid"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/arima_order_grid/analysis_rmse"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = read_completed_runs(args.sweep_dir)
    save_horizon_best(completed, args.output_dir)
    period_rmse = read_period_rmse(args.sweep_dir, completed)
    save_period_best(period_rmse, args.output_dir)

    print(f"completed horizon rows: {len(completed):,}")
    print(f"completed order count: {completed['order_label'].nunique():,}")
    print("completed runs by frequency/method:")
    print(completed.groupby(["frequency", "method"]).size().to_string())
    print("\nbest horizon RMSE by frequency and target:")
    print(pd.read_csv(args.output_dir / "best_horizon_rmse_by_frequency.csv").to_string(index=False))
    print("\nperiod-level winner summary:")
    print(pd.read_csv(args.output_dir / "period_rmse_winner_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
