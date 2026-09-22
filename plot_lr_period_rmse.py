"""Plot original (non-enriched) LR forecasts using period-level RMSE winners.

The period RMSE is calculated across accounts separately for each forecast
period, target, frequency, and method.  Monthly and weekly periods are not
mixed when selecting winners because they represent different time units.

Outputs:
    outputs/linear_regression/analysis_period_rmse/
    - period_rmse.csv
    - period_winners.csv
    - winner_summary.csv
    - period_rmse_by_version.png
    - account_plots/account_<id>.png
    - best_period_account_plots/<frequency>/account_<id>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def read_predictions(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "predictions.parquet"
    csv_path = run_dir / "predictions.csv"
    if parquet_path.exists():
        result = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        result = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No predictions file found in {run_dir}")

    result["account_id"] = result["account_id"].astype(int)
    result["period_start"] = pd.to_datetime(result["period_start"])
    required = {"account_id", "period_start"}
    required.update(column for _, actual, prediction in TARGETS for column in (actual, prediction))
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"Missing columns in {run_dir}: {missing}")
    return result.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def load_predictions(results_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for frequency, method in RUNS:
        predictions[(frequency, method)] = read_predictions(
            results_dir / f"{frequency}_{method}"
        )
    return predictions


def calculate_period_rmse(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (frequency, method), frame in predictions_by_run.items():
        for target, actual_column, prediction_column in TARGETS:
            work = frame[["period_start", actual_column, prediction_column]].copy()
            work["squared_error"] = (
                work[prediction_column].astype(float)
                - work[actual_column].astype(float)
            ) ** 2
            grouped = work.groupby("period_start", sort=True)
            for period_start, group in grouped:
                rows.append(
                    {
                        "frequency": frequency,
                        "method": method,
                        "target": target,
                        "period_start": period_start,
                        "n_observations": int(len(group)),
                        "rmse": float(np.sqrt(group["squared_error"].mean())),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["frequency", "target", "period_start", "method"]
    )


def select_period_winners(period_rmse: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (frequency, target, period_start), group in period_rmse.groupby(
        ["frequency", "target", "period_start"], sort=True
    ):
        by_method = group.set_index("method")["rmse"]
        recursive_rmse = float(by_method.get("recursive", np.nan))
        direct_rmse = float(by_method.get("direct", np.nan))
        valid = by_method.dropna()
        if valid.empty:
            continue
        best_method = str(valid.idxmin())
        rows.append(
            {
                "frequency": frequency,
                "target": target,
                "period_start": period_start,
                "recursive_rmse": recursive_rmse,
                "direct_rmse": direct_rmse,
                "best_method": best_method,
                "best_rmse": float(valid.min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["frequency", "target", "period_start"]
    )


def sample_accounts(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path,
    n_accounts: int,
    random_state: int,
) -> list[int]:
    common = set.intersection(
        *[
            set(frame["account_id"].unique())
            for frame in predictions_by_run.values()
        ]
    )
    if len(common) < n_accounts:
        raise ValueError(f"Only {len(common)} common accounts are available")

    existing_ids = output_dir.parent / "analysis" / "sampled_account_ids.csv"
    if existing_ids.exists():
        previous = pd.read_csv(existing_ids)["account_id"].astype(int).tolist()
        if len(previous) >= n_accounts and set(previous[:n_accounts]).issubset(common):
            selected = sorted(previous[:n_accounts])
        else:
            selected = []
    else:
        selected = []

    if not selected:
        rng = np.random.default_rng(random_state)
        selected = sorted(
            int(value) for value in rng.choice(sorted(common), n_accounts, replace=False)
        )
    pd.DataFrame({"account_id": selected}).to_csv(
        output_dir / "sampled_account_ids.csv", index=False
    )
    return selected


def save_period_rmse_plot(period_rmse: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    for row_index, frequency in enumerate(("monthly", "weekly")):
        for column_index, (target, _, _) in enumerate(TARGETS):
            axis = axes[row_index][column_index]
            subset = period_rmse.loc[
                period_rmse["frequency"].eq(frequency)
                & period_rmse["target"].eq(target)
            ]
            for method, color in (("recursive", "#d62728"), ("direct", "#2ca02c")):
                line = subset.loc[subset["method"].eq(method)]
                axis.plot(
                    line["period_start"],
                    line["rmse"],
                    marker="o",
                    linewidth=1.8,
                    label=method,
                    color=color,
                )
            axis.set_title(f"{frequency} / {target}")
            axis.set_ylabel("Period RMSE")
            axis.grid(True, alpha=0.25)
            axis.tick_params(axis="x", rotation=35)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    figure.suptitle("Original LR: period-level RMSE by forecast version", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_version_account_plots(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
    account_ids: list[int],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "account_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for account_id in account_ids:
        figure, axes = plt.subplots(4, 3, figsize=(18, 14), squeeze=False)
        for row_index, (frequency, method) in enumerate(RUNS):
            frame = predictions_by_run[(frequency, method)]
            frame = frame.loc[frame["account_id"].eq(account_id)]
            for column_index, (target, actual_column, prediction_column) in enumerate(TARGETS):
                axis = axes[row_index][column_index]
                axis.plot(frame["period_start"], frame[actual_column], marker="o", label="Actual", color="#1f77b4")
                axis.plot(frame["period_start"], frame[prediction_column], marker="o", label="Predicted", color="#d62728")
                axis.set_title(f"{frequency} / {method} / {target}")
                axis.grid(True, alpha=0.25)
                axis.tick_params(axis="x", rotation=35)
                if row_index == 0 and column_index == 0:
                    axis.legend(frameon=False)
        figure.suptitle(f"Original LR: account {account_id}, all four versions", fontsize=16)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(plot_dir / f"account_{account_id}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def save_best_period_plots(
    predictions_by_run: dict[tuple[str, str], pd.DataFrame],
    winners: pd.DataFrame,
    account_ids: list[int],
    output_dir: Path,
) -> None:
    root = output_dir / "best_period_account_plots"
    for frequency in ("monthly", "weekly"):
        frequency_dir = root / frequency
        frequency_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(
            [
                frame.assign(method=method)
                for (run_frequency, method), frame in predictions_by_run.items()
                if run_frequency == frequency
            ],
            ignore_index=True,
        )
        for account_id in account_ids:
            figure, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
            account_frame = combined.loc[combined["account_id"].eq(account_id)]
            for column_index, (target, actual_column, prediction_column) in enumerate(TARGETS):
                selection = winners.loc[
                    winners["frequency"].eq(frequency) & winners["target"].eq(target)
                ][["period_start", "best_method"]]
                target_frame = account_frame[["period_start", "method", actual_column, prediction_column]].merge(
                    selection, on="period_start", how="inner"
                )
                target_frame = target_frame.loc[target_frame["method"].eq(target_frame["best_method"])]
                target_frame = target_frame.sort_values("period_start")
                axis = axes[0][column_index]
                axis.plot(target_frame["period_start"], target_frame[actual_column], marker="o", label="Actual", color="#1f77b4")
                axis.plot(target_frame["period_start"], target_frame[prediction_column], marker="o", label="Best period prediction", color="#ff7f0e")
                axis.set_title(f"{frequency} / {target}")
                axis.grid(True, alpha=0.25)
                axis.tick_params(axis="x", rotation=35)
                if column_index == 0:
                    axis.legend(frameon=False)
            figure.suptitle(
                f"Original LR: account {account_id}, period-RMSE winner ({frequency})",
                fontsize=15,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            figure.savefig(frequency_dir / f"account_{account_id}.png", dpi=150, bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/linear_regression"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/linear_regression/analysis_period_rmse"))
    parser.add_argument("--n-accounts", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.n_accounts <= 0:
        raise ValueError("--n-accounts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_run = load_predictions(args.results_dir)
    period_rmse = calculate_period_rmse(predictions_by_run)
    winners = select_period_winners(period_rmse)
    account_ids = sample_accounts(predictions_by_run, args.output_dir, args.n_accounts, args.random_state)

    period_rmse.to_csv(args.output_dir / "period_rmse.csv", index=False)
    winners.to_csv(args.output_dir / "period_winners.csv", index=False)
    summary = (
        winners.groupby(["frequency", "target", "best_method"], as_index=False)
        .agg(periods_won=("period_start", "size"), mean_best_rmse=("best_rmse", "mean"))
    )
    summary.to_csv(args.output_dir / "winner_summary.csv", index=False)
    save_period_rmse_plot(period_rmse, args.output_dir / "period_rmse_by_version.png")
    save_version_account_plots(predictions_by_run, account_ids, args.output_dir)
    save_best_period_plots(predictions_by_run, winners, account_ids, args.output_dir)

    print(f"saved output: {args.output_dir}")
    print(f"sampled account_ids: {account_ids}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
