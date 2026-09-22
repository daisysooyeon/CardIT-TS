"""Plot sampled account forecasts and compare linear-regression runs.

The account-level plots use each run's predictions.parquet (or predictions.csv
as a fallback), while the metric comparison uses the combined
horizon_comparison_metrics.csv file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("monthly", "recursive"),
    ("monthly", "direct"),
    ("weekly", "recursive"),
    ("weekly", "direct"),
]

TARGETS = [
    ("inflow", "actual_inflow", "prediction_inflow"),
    ("outflow", "actual_outflow", "prediction_outflow"),
    ("net_flow", "actual_net_flow", "prediction_net_flow"),
]

# The first value defines how a best result is selected.  The second value
# describes whether the metric is a performance metric or a coverage/diagnostic
# field.  Bias and underprediction_rate do not have a simple lower-is-better
# interpretation, so they are treated as closest to zero and 0.5 respectively.
METRIC_RULES = {
    "mae": ("min", "performance"),
    "rmse": ("min", "performance"),
    "wape": ("min", "performance"),
    "bias": ("closest_to_zero", "performance"),
    "underprediction_rate": ("closest_to_half", "diagnostic"),
    "underprediction_mae": ("min", "performance"),
    "overprediction_mae": ("min", "performance"),
    "asymmetric_mae": ("min", "performance"),
    "asymmetric_rmse": ("min", "performance"),
    "nmae_train_mean": ("min", "performance"),
    "nrmse_train_mean": ("min", "performance"),
    "negative_raw_prediction_rate": ("min", "diagnostic"),
    "zero_scale_rate": ("min", "diagnostic"),
    "n_observations": ("max", "coverage"),
    "n_normalized_observations": ("max", "coverage"),
}


def read_predictions(run_dir: Path) -> pd.DataFrame:
    """Read one run's prediction table, preferring typed Parquet output."""
    parquet_path = run_dir / "predictions.parquet"
    csv_path = run_dir / "predictions.csv"
    if parquet_path.exists():
        predictions = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        predictions = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Neither predictions.parquet nor predictions.csv exists in {run_dir}"
        )

    predictions["period_start"] = pd.to_datetime(predictions["period_start"])
    required = {
        "account_id",
        "period_start",
        "actual_inflow",
        "prediction_inflow",
        "actual_outflow",
        "prediction_outflow",
        "actual_net_flow",
        "prediction_net_flow",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing columns in {run_dir}: {missing}")
    return predictions.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def load_all_predictions(results_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    predictions_by_run: dict[tuple[str, str], pd.DataFrame] = {}
    for frequency, method in RUNS:
        run_key = (frequency, method)
        run_dir = results_dir / f"{frequency}_{method}"
        predictions = read_predictions(run_dir)
        predictions["frequency"] = frequency
        predictions["method"] = method
        predictions_by_run[run_key] = predictions
    return predictions_by_run


def sample_common_accounts(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
    n_accounts: int,
    random_state: int,
) -> list[int]:
    account_sets = [
        set(predictions["account_id"].astype(int).unique())
        for predictions in predictions_by_run.values()
    ]
    common_accounts = sorted(set.intersection(*account_sets))
    if len(common_accounts) < n_accounts:
        raise ValueError(
            f"Only {len(common_accounts)} common accounts are available; "
            f"cannot sample {n_accounts}."
        )

    rng = np.random.default_rng(random_state)
    sampled = rng.choice(common_accounts, size=n_accounts, replace=False)
    return sorted(int(account_id) for account_id in sampled)


def save_sampled_predictions(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
    account_ids: list[int],
    output_dir: Path,
) -> None:
    selected = []
    for predictions in predictions_by_run.values():
        selected.append(
            predictions.loc[predictions["account_id"].isin(account_ids)].copy()
        )
    sampled_predictions = pd.concat(selected, ignore_index=True)
    sampled_predictions.to_csv(
        output_dir / "sampled_account_predictions.csv",
        index=False,
    )
    sampled_predictions.to_parquet(
        output_dir / "sampled_account_predictions.parquet",
        index=False,
    )


def save_account_plots(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
    account_ids: list[int],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "account_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for account_id in account_ids:
        figure, axes = plt.subplots(
            nrows=len(RUNS),
            ncols=len(TARGETS),
            figsize=(18, 14),
            squeeze=False,
        )
        for row_index, run_key in enumerate(RUNS):
            frequency, method = run_key
            account_predictions = predictions_by_run[run_key]
            account_predictions = account_predictions.loc[
                account_predictions["account_id"].eq(account_id)
            ].sort_values("period_start")

            for column_index, (target, actual_column, prediction_column) in enumerate(TARGETS):
                axis = axes[row_index][column_index]
                axis.plot(
                    account_predictions["period_start"],
                    account_predictions[actual_column],
                    marker="o",
                    linewidth=1.8,
                    label="Actual",
                    color="#1f77b4",
                )
                axis.plot(
                    account_predictions["period_start"],
                    account_predictions[prediction_column],
                    marker="o",
                    linewidth=1.8,
                    label="Predicted",
                    color="#d62728",
                )
                axis.set_title(f"{frequency} / {method} — {target}")
                axis.set_ylabel("Amount")
                axis.grid(True, alpha=0.25)
                axis.tick_params(axis="x", rotation=35)
                if row_index == 0 and column_index == 0:
                    axis.legend(frameon=False)

        figure.suptitle(
            f"Account {account_id}: actual vs predicted flows",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(
            plot_dir / f"account_{account_id}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def select_best_value(values: pd.Series, rule: str) -> tuple[float, pd.Series]:
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid = numeric_values.notna()
    if not valid.any():
        return np.nan, valid

    if rule == "min":
        selection_values = numeric_values
        best_selection_value = float(selection_values[valid].min())
    elif rule == "max":
        selection_values = numeric_values
        best_selection_value = float(selection_values[valid].max())
    elif rule == "closest_to_zero":
        selection_values = numeric_values.abs()
        best_selection_value = float(selection_values[valid].min())
    elif rule == "closest_to_half":
        selection_values = (numeric_values - 0.5).abs()
        best_selection_value = float(selection_values[valid].min())
    else:
        raise ValueError(f"Unknown metric selection rule: {rule}")

    is_best = valid & np.isclose(
        selection_values,
        best_selection_value,
        rtol=1e-10,
        atol=1e-12,
    )
    return best_selection_value, is_best


def compare_horizon_metrics(
    results_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    metrics_path = results_dir / "horizon_comparison_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    required = {"target", "frequency", "method"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Missing columns in {metrics_path}: {missing}")

    rows: list[dict[str, object]] = []
    for target in sorted(metrics["target"].dropna().unique()):
        target_metrics = metrics.loc[metrics["target"].eq(target)].copy()
        target_metrics["combination"] = (
            target_metrics["frequency"].astype(str)
            + "_"
            + target_metrics["method"].astype(str)
        )

        for metric, (rule, metric_family) in METRIC_RULES.items():
            if metric not in target_metrics.columns:
                continue
            best_selection_value, is_best = select_best_value(
                target_metrics[metric],
                rule,
            )
            winners = target_metrics.loc[is_best].copy()
            if winners.empty:
                continue

            winner_names = sorted(winners["combination"].astype(str).tolist())
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "selection_rule": rule,
                    "metric_family": metric_family,
                    "best_value": float(winners.iloc[0][metric]),
                    "best_selection_value": best_selection_value,
                    "best_combination": "; ".join(winner_names),
                    "n_tied_winners": len(winner_names),
                    "comparison_note": (
                        "Raw-scale metric; compare frequencies cautiously."
                        if metric in {"mae", "rmse"}
                        else ""
                    ),
                }
            )

    best_metrics = pd.DataFrame(rows).sort_values(
        ["target", "metric"]
    ).reset_index(drop=True)
    best_metrics.to_csv(
        output_dir / "best_by_metric.csv",
        index=False,
    )
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sampled account forecasts and compare LR run metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/linear_regression"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/linear_regression/analysis"),
    )
    parser.add_argument("--n-accounts", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.n_accounts <= 0:
        raise ValueError("--n-accounts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_run = load_all_predictions(args.results_dir)
    account_ids = sample_common_accounts(
        predictions_by_run,
        n_accounts=args.n_accounts,
        random_state=args.random_state,
    )
    pd.DataFrame({"account_id": account_ids}).to_csv(
        args.output_dir / "sampled_account_ids.csv",
        index=False,
    )
    save_sampled_predictions(
        predictions_by_run,
        account_ids,
        args.output_dir,
    )
    save_account_plots(
        predictions_by_run,
        account_ids,
        args.output_dir,
    )
    best_metrics = compare_horizon_metrics(
        args.results_dir,
        args.output_dir,
    )

    print(f"sampled account_ids: {account_ids}")
    print(f"saved plots: {args.output_dir / 'account_plots'}")
    print(f"saved best metrics: {args.output_dir / 'best_by_metric.csv'}")
    print(best_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
