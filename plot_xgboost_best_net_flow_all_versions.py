"""Plot the best non-enriched XGBoost grid for all four forecast versions.

For each criterion, one best grid is selected independently for each of:
monthly_recursive, monthly_direct, weekly_recursive, and weekly_direct.
The same ten account IDs used by the LR plots are then shown against actuals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT / "outputs" / "xgboost_regularization_sweep"
ACCOUNT_IDS_PATH = (
    ROOT
    / "outputs"
    / "linear_regression"
    / "analysis_period_rmse"
    / "sampled_account_ids.csv"
)
OUTPUT_ROOT = SWEEP_ROOT / "analysis_best_net_flow_rmse_all_versions"

VERSION_ORDER = [
    ("monthly", "recursive"),
    ("monthly", "direct"),
    ("weekly", "recursive"),
    ("weekly", "direct"),
]
TARGETS = {
    "inflow": ("actual_inflow", "prediction_inflow"),
    "outflow": ("actual_outflow", "prediction_outflow"),
    "net_flow": ("actual_net_flow", "prediction_net_flow"),
}


def read_predictions(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "predictions.parquet"
    csv_path = run_dir / "predictions.csv"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No predictions file found in {run_dir}")

    required = {"account_id", "period_start"}
    required.update(column for columns in TARGETS.values() for column in columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{run_dir} is missing columns: {missing}")

    frame["account_id"] = frame["account_id"].astype(int)
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    return frame.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def read_sample_account_ids() -> list[int]:
    frame = pd.read_csv(ACCOUNT_IDS_PATH)
    return sorted(frame["account_id"].astype(int).tolist())


def grid_dirs() -> list[Path]:
    return sorted(
        path
        for path in SWEEP_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("grid_")
    )


def load_horizon_candidates() -> pd.DataFrame:
    path = SWEEP_ROOT / "horizon_comparison_metrics.csv"
    frame = pd.read_csv(path)
    return frame.loc[
        frame["target"].eq("net_flow")
        & frame["horizon_step"].astype(str).eq("horizon_total")
    ].copy()


def load_period_candidates() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for grid_dir in grid_dirs():
        for frequency, method in VERSION_ORDER:
            metrics_path = grid_dir / f"{frequency}_{method}" / "metrics.csv"
            metrics = pd.read_csv(metrics_path)
            metrics = metrics.loc[
                metrics["target"].eq("net_flow")
                & ~metrics["horizon_step"].astype(str).eq("all")
            ].copy()
            metrics["grid_id"] = grid_dir.name
            metrics["frequency"] = frequency
            metrics["method"] = method
            rows.append(metrics)

    period_metrics = pd.concat(rows, ignore_index=True)
    grouped = (
        period_metrics.groupby(["frequency", "method", "grid_id"], as_index=False)
        .agg(
            mean_period_rmse=("rmse", "mean"),
            n_periods=("rmse", "size"),
        )
    )
    return grouped


def select_best_runs() -> pd.DataFrame:
    horizon = load_horizon_candidates()
    period = load_period_candidates()
    selected_rows: list[dict[str, object]] = []

    for criterion, candidates, score_column in [
        ("horizon_rmse_best", horizon, "rmse"),
        ("period_mean_rmse_best", period, "mean_period_rmse"),
    ]:
        for frequency, method in VERSION_ORDER:
            group = candidates.loc[
                candidates["frequency"].eq(frequency)
                & candidates["method"].eq(method)
            ].copy()
            group = group.sort_values([score_column, "grid_id"], kind="stable")
            best = group.iloc[0]
            tied = group.loc[
                (group[score_column] - float(best[score_column])).abs() <= 1e-6,
                "grid_id",
            ].tolist()
            selected_rows.append(
                {
                    "criterion": criterion,
                    "frequency": frequency,
                    "method": method,
                    "grid_id": str(best["grid_id"]),
                    "tie_grid_ids": "|".join(tied),
                    "horizon_rmse": float(
                        best.get("rmse", float("nan"))
                        if criterion == "horizon_rmse_best"
                        else horizon.loc[
                            horizon["grid_id"].eq(best["grid_id"])
                            & horizon["frequency"].eq(frequency)
                            & horizon["method"].eq(method),
                            "rmse",
                        ].iloc[0]
                    ),
                    "mean_period_rmse": float(
                        best["mean_period_rmse"]
                        if criterion == "period_mean_rmse_best"
                        else period.loc[
                            period["grid_id"].eq(best["grid_id"])
                            & period["frequency"].eq(frequency)
                            & period["method"].eq(method),
                            "mean_period_rmse",
                        ].iloc[0]
                    ),
                }
            )

    selected = pd.DataFrame(selected_rows)
    parameter_columns = [
        "max_depth",
        "n_estimators",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    ]
    params = pd.read_csv(SWEEP_ROOT / "horizon_comparison_metrics.csv")
    params = params.drop_duplicates("grid_id")["grid_id"].to_frame().merge(
        pd.read_csv(SWEEP_ROOT / "horizon_comparison_metrics.csv")[
            ["grid_id", *parameter_columns]
        ].drop_duplicates("grid_id"),
        on="grid_id",
        how="left",
    )
    return selected.merge(params, on="grid_id", how="left")


def plot_account(
    account_id: int,
    selected: pd.DataFrame,
    predictions: dict[tuple[str, str, str], pd.DataFrame],
    criterion: str,
    output_path: Path,
) -> None:
    chosen = selected.loc[selected["criterion"].eq(criterion)].set_index(
        ["frequency", "method"]
    )
    fig, axes = plt.subplots(4, 3, figsize=(19, 17), squeeze=False)
    handles = None
    labels = None

    for row_index, (frequency, method) in enumerate(VERSION_ORDER):
        selected_run = chosen.loc[(frequency, method)]
        key = (frequency, method, str(selected_run["grid_id"]))
        frame = predictions[key]
        account_frame = frame.loc[frame["account_id"].eq(account_id)].sort_values(
            "period_start"
        )
        if account_frame.empty:
            raise ValueError(f"account_id={account_id} is missing from {key}")

        for column_index, (target, columns) in enumerate(TARGETS.items()):
            actual_column, prediction_column = columns
            axis = axes[row_index, column_index]
            axis.plot(
                account_frame["period_start"],
                account_frame[actual_column],
                color="#222222",
                linewidth=2.0,
                linestyle="--",
                label="Actual",
                zorder=5,
            )
            axis.plot(
                account_frame["period_start"],
                account_frame[prediction_column],
                color="#2f6db0",
                linewidth=1.7,
                label="XGBoost prediction",
            )
            axis.axhline(0.0, color="#999999", linewidth=0.7)
            axis.set_title(
                f"{frequency}_{method} · {target}\n"
                f"{selected_run['grid_id']}"
            )
            axis.set_ylabel("Amount")
            axis.grid(True, alpha=0.25)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
            axis.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m" if frequency == "monthly" else "%m-%d")
            )
            axis.tick_params(axis="x", rotation=35)
            if handles is None:
                handles, labels = axis.get_legend_handles_labels()

    fig.suptitle(
        f"XGBoost non-enriched: actual vs prediction · account_id={account_id}\n"
        f"Selection criterion: {criterion}",
        fontsize=15,
    )
    if handles is not None and labels is not None:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2)
    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.92))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = select_best_runs()
    selected.to_csv(OUTPUT_ROOT / "selected_runs.csv", index=False)

    account_ids = read_sample_account_ids()
    pd.DataFrame({"account_id": account_ids}).to_csv(
        OUTPUT_ROOT / "sampled_account_ids.csv", index=False
    )

    predictions: dict[tuple[str, str, str], pd.DataFrame] = {}
    for grid_id in selected["grid_id"].unique():
        grid_dir = SWEEP_ROOT / str(grid_id)
        for frequency, method in VERSION_ORDER:
            key = (frequency, method, str(grid_id))
            predictions[key] = read_predictions(grid_dir / f"{frequency}_{method}")

    for criterion in selected["criterion"].unique():
        criterion_dir = OUTPUT_ROOT / criterion / "account_plots"
        criterion_dir.mkdir(parents=True, exist_ok=True)
        for account_id in account_ids:
            plot_account(
                account_id,
                selected,
                predictions,
                criterion,
                criterion_dir / f"account_{account_id}.png",
            )

    print(f"Saved plots for {len(account_ids)} accounts")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
