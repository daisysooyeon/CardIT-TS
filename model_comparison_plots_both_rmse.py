"""Compare LR, ARIMA, XGBoost, and LightGBM on common account plots.

For every model and frequency, this script selects two runs independently:

* ``horizon_rmse_best``: lowest horizon-total net-flow RMSE;
* ``period_mean_rmse_best``: lowest mean of the individual test-period
  net-flow RMSE values.

It then creates the same account-level comparison layout used previously:
actual net flow plus one line per model.  The ten account IDs are reused from
the LR period-RMSE analysis so the plots are directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ACCOUNT_IDS_PATH = (
    ROOT
    / "outputs"
    / "linear_regression"
    / "analysis_period_rmse"
    / "sampled_account_ids.csv"
)
OUTPUT_ROOT = ROOT / "outputs" / "model_comparison_both_rmse"

MODEL_ORDER = ["LR", "ARIMA", "XGBoost", "LightGBM"]
FREQUENCIES = ["monthly", "weekly"]
MODEL_COLORS = {
    "LR": "#1f77b4",
    "ARIMA": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "LightGBM": "#9467bd",
}
MODEL_SOURCES = {
    "LR": {
        "root": ROOT / "outputs" / "linear_regression",
        "metrics": ROOT / "outputs" / "linear_regression" / "horizon_comparison_metrics.csv",
        "run_key": None,
    },
    "ARIMA": {
        "root": ROOT / "outputs" / "arima_order_grid",
        "metrics": ROOT / "outputs" / "arima_order_grid" / "horizon_comparison_metrics.csv",
        "run_key": "order_label",
    },
    "XGBoost": {
        "root": ROOT / "outputs" / "xgboost_regularization_sweep",
        "metrics": ROOT / "outputs" / "xgboost_regularization_sweep" / "horizon_comparison_metrics.csv",
        "run_key": "grid_id",
    },
    "LightGBM": {
        "root": ROOT / "outputs" / "lightgbm_regularization_sampling_sweep",
        "metrics": ROOT / "outputs" / "lightgbm_regularization_sampling_sweep" / "horizon_comparison_metrics.csv",
        "run_key": "grid_id",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model comparisons for the common sample or selected account IDs."
    )
    parser.add_argument(
        "--account-ids",
        nargs="+",
        type=int,
        default=None,
        help="Specific account IDs to plot. Defaults to the ten LR sample accounts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory in which comparison plots are saved.",
    )
    return parser.parse_args()


def read_sample_account_ids() -> list[int]:
    accounts = pd.read_csv(ACCOUNT_IDS_PATH)
    return sorted(accounts["account_id"].astype(int).tolist())


def run_dirs(model: str) -> list[tuple[str, Path]]:
    """Return run identifiers and directories for one model."""
    source = MODEL_SOURCES[model]
    root = source["root"]
    run_key = source["run_key"]
    if run_key is None:
        return [("base", root)]

    if model == "ARIMA":
        candidates = sorted(root.glob("order_*"))
    else:
        candidates = sorted(root.glob("grid_*"))
    return [(path.name, path) for path in candidates if path.is_dir()]


def version_metrics_path(model: str, run_dir: Path, frequency: str, method: str) -> Path:
    if model == "LR":
        return run_dir / f"{frequency}_{method}" / "metrics.csv"
    return run_dir / f"{frequency}_{method}" / "metrics.csv"


def version_prediction_path(
    model: str, run_dir: Path, frequency: str, method: str
) -> Path:
    run_dir = run_dir / f"{frequency}_{method}"
    parquet_path = run_dir / "predictions.parquet"
    csv_path = run_dir / "predictions.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"No predictions file found in {run_dir}")


def add_run_identity(
    frame: pd.DataFrame,
    model: str,
    run_id: str,
    frequency: str,
    method: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["model"] = model
    frame["run_id"] = run_id
    frame["frequency"] = frequency
    frame["method"] = method
    return frame


def load_horizon_candidates() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        source = MODEL_SOURCES[model]
        metrics = pd.read_csv(source["metrics"])
        metrics = metrics.loc[
            metrics["target"].eq("net_flow")
            & metrics["horizon_step"].astype(str).eq("horizon_total")
        ].copy()
        if metrics.empty:
            raise ValueError(f"No horizon net-flow rows found for {model}")
        for _, row in metrics.iterrows():
            run_id = "base" if source["run_key"] is None else str(row[source["run_key"]])
            row = row.to_frame().T
            row["model"] = model
            row["run_id"] = run_id
            rows.append(row)
    return pd.concat(rows, ignore_index=True)


def load_period_candidates() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        for run_id, run_dir in run_dirs(model):
            for frequency in FREQUENCIES:
                for method in ("recursive", "direct"):
                    path = version_metrics_path(model, run_dir, frequency, method)
                    if not path.exists():
                        continue
                    metrics = pd.read_csv(path)
                    metrics = metrics.loc[
                        metrics["target"].eq("net_flow")
                        & ~metrics["horizon_step"].astype(str).eq("all")
                    ].copy()
                    if metrics.empty:
                        continue
                    metrics["model"] = model
                    metrics["run_id"] = run_id
                    metrics["frequency"] = frequency
                    metrics["method"] = method
                    rows.append(metrics)

    if not rows:
        raise ValueError("No period-level net-flow metrics found")
    period_rows = pd.concat(rows, ignore_index=True)
    return (
        period_rows.groupby(
            ["model", "run_id", "frequency", "method"], as_index=False
        )
        .agg(
            period_mean_rmse=("rmse", "mean"),
            n_periods=("rmse", "size"),
        )
    )


def select_runs() -> pd.DataFrame:
    horizon = load_horizon_candidates()
    period = load_period_candidates()
    rows: list[dict[str, object]] = []

    for criterion, candidates, score_column in [
        ("horizon_rmse_best", horizon, "rmse"),
        ("period_mean_rmse_best", period, "period_mean_rmse"),
    ]:
        for model in MODEL_ORDER:
            for frequency in FREQUENCIES:
                group = candidates.loc[
                    candidates["model"].eq(model)
                    & candidates["frequency"].eq(frequency)
                ].copy()
                if group.empty:
                    raise ValueError(f"No {criterion} candidates for {model}/{frequency}")
                group = group.sort_values([score_column, "run_id"], kind="stable")
                best = group.iloc[0]
                run_id = str(best["run_id"])

                if criterion == "horizon_rmse_best":
                    horizon_rmse = float(best["rmse"])
                    period_match = period.loc[
                        period["model"].eq(model)
                        & period["run_id"].eq(run_id)
                        & period["frequency"].eq(frequency)
                        & period["method"].eq(best["method"]),
                        "period_mean_rmse",
                    ]
                    period_mean_rmse = float(period_match.iloc[0])
                else:
                    period_mean_rmse = float(best["period_mean_rmse"])
                    horizon_match = horizon.loc[
                        horizon["model"].eq(model)
                        & horizon["run_id"].eq(run_id)
                        & horizon["frequency"].eq(frequency)
                        & horizon["method"].eq(best["method"]),
                        "rmse",
                    ]
                    horizon_rmse = float(horizon_match.iloc[0])

                rows.append(
                    {
                        "criterion": criterion,
                        "model": model,
                        "frequency": frequency,
                        "method": str(best["method"]),
                        "run_id": run_id,
                        "horizon_rmse": horizon_rmse,
                        "period_mean_rmse": period_mean_rmse,
                    }
                )

    return pd.DataFrame(rows)


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "account_id",
        "period_start",
        "actual_net_flow",
        "prediction_net_flow",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["account_id"] = frame["account_id"].astype(int)
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    return frame.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def load_selected_predictions(
    selected: pd.DataFrame,
) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    loaded: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    run_dir_lookup = {
        (model, run_id): run_dir
        for model in MODEL_ORDER
        for run_id, run_dir in run_dirs(model)
    }
    for _, row in selected.iterrows():
        key = (
            str(row["criterion"]),
            str(row["model"]),
            str(row["frequency"]),
            str(row["run_id"]),
        )
        run_dir = run_dir_lookup[(str(row["model"]), str(row["run_id"]))]
        path = version_prediction_path(
            str(row["model"]), run_dir, str(row["frequency"]), str(row["method"])
        )
        loaded[key] = read_predictions(path)
    return loaded


def plot_comparison(
    selected: pd.DataFrame,
    loaded: dict[tuple[str, str, str, str], pd.DataFrame],
    criterion: str,
    frequency: str,
    account_ids: list[int],
    output_path: Path,
) -> None:
    chosen = selected.loc[
        selected["criterion"].eq(criterion) & selected["frequency"].eq(frequency)
    ].set_index("model")
    n_columns = 2
    n_rows = (len(account_ids) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18, max(7.5, 4.5 * n_rows)),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, account_id in zip(axes_flat, account_ids):
        actual_plotted = False
        for model in MODEL_ORDER:
            row = chosen.loc[model]
            key = (criterion, model, frequency, str(row["run_id"]))
            frame = loaded[key]
            account_frame = frame.loc[frame["account_id"].eq(account_id)].sort_values(
                "period_start"
            )
            if account_frame.empty:
                continue
            if not actual_plotted:
                axis.plot(
                    account_frame["period_start"],
                    account_frame["actual_net_flow"],
                    color="#222222",
                    linewidth=2.0,
                    linestyle="--",
                    label="Actual",
                    zorder=5,
                )
                actual_plotted = True
            axis.plot(
                account_frame["period_start"],
                account_frame["prediction_net_flow"],
                color=MODEL_COLORS[model],
                linewidth=1.5,
                label=model,
            )
        axis.set_title(f"account_id = {account_id}")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m" if frequency == "monthly" else "%m-%d")
        )
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel("Amount")

    for axis in axes_flat[len(account_ids) :]:
        axis.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
    )
    basis_label = (
        "horizon-level net-flow RMSE"
        if criterion == "horizon_rmse_best"
        else "mean individual-period net-flow RMSE"
    )
    fig.suptitle(
        f"{frequency.title()} net_flow: actual vs model forecasts\n"
        f"Each model uses its best completed run by {basis_label}",
        y=0.995,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_single_account_comparison(
    selected: pd.DataFrame,
    loaded: dict[tuple[str, str, str, str], pd.DataFrame],
    criterion: str,
    frequency: str,
    account_id: int,
    output_path: Path,
) -> None:
    """Plot one account with actual plus all four model forecasts."""
    chosen = selected.loc[
        selected["criterion"].eq(criterion) & selected["frequency"].eq(frequency)
    ].set_index("model")
    fig, axis = plt.subplots(figsize=(13, 6.5))
    actual_plotted = False

    for model in MODEL_ORDER:
        row = chosen.loc[model]
        key = (criterion, model, frequency, str(row["run_id"]))
        frame = loaded[key]
        account_frame = frame.loc[frame["account_id"].eq(account_id)].sort_values(
            "period_start"
        )
        if account_frame.empty:
            continue
        if not actual_plotted:
            axis.plot(
                account_frame["period_start"],
                account_frame["actual_net_flow"],
                color="#222222",
                linewidth=2.2,
                linestyle="--",
                label="Actual",
                zorder=5,
            )
            actual_plotted = True
        axis.plot(
            account_frame["period_start"],
            account_frame["prediction_net_flow"],
            color=MODEL_COLORS[model],
            linewidth=1.8,
            label=model,
        )

    if not actual_plotted:
        raise ValueError(f"account_id={account_id} is missing from {frequency} predictions")

    basis_label = (
        "horizon-level net-flow RMSE"
        if criterion == "horizon_rmse_best"
        else "mean individual-period net-flow RMSE"
    )
    axis.set_title(
        f"{frequency.title()} net_flow: account_id={account_id}\n"
        f"Each model uses its best completed run by {basis_label}"
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_ylabel("Amount")
    axis.grid(True, alpha=0.25)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m" if frequency == "monthly" else "%m-%d")
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    account_ids = (
        sorted(set(args.account_ids))
        if args.account_ids is not None
        else read_sample_account_ids()
    )
    if not account_ids:
        raise ValueError("At least one account ID is required")
    selected = select_runs()
    selected.to_csv(output_root / "selected_best_runs.csv", index=False)
    pd.DataFrame({"account_id": account_ids}).to_csv(
        output_root / "sampled_account_ids.csv", index=False
    )
    loaded = load_selected_predictions(selected)

    for criterion in selected["criterion"].unique():
        criterion_dir = output_root / criterion
        criterion_dir.mkdir(parents=True, exist_ok=True)
        for frequency in FREQUENCIES:
            plot_comparison(
                selected,
                loaded,
                criterion,
                frequency,
                account_ids,
                criterion_dir / f"{frequency}_net_flow_accounts.png",
            )
            account_plot_dir = criterion_dir / frequency / "account_plots"
            account_plot_dir.mkdir(parents=True, exist_ok=True)
            for account_id in account_ids:
                plot_single_account_comparison(
                    selected,
                    loaded,
                    criterion,
                    frequency,
                    account_id,
                    account_plot_dir / f"account_{account_id}.png",
                )

    print(f"Saved comparison plots to {output_root}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
