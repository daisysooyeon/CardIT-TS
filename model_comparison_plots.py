"""Compare the best completed model forecasts for the Berka master tables.

For each model and frequency, the run with the lowest selected net-flow RMSE
is selected. The selected run is then plotted for the same reproducible sample
of accounts across inflow, outflow, and net flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

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

TARGET_COLUMNS = {
    "inflow": ("actual_inflow", "prediction_inflow"),
    "outflow": ("actual_outflow", "prediction_outflow"),
    "net_flow": ("actual_net_flow", "prediction_net_flow"),
}

MODEL_COLORS = {
    "LR": "#1f77b4",
    "ARIMA": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "LightGBM": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparable LR, ARIMA, XGBoost, and LightGBM forecasts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "model_comparison",
        help="Directory for plots and selection tables.",
    )
    parser.add_argument(
        "--n-accounts",
        type=int,
        default=10,
        help="Number of common accounts to sample per frequency.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used to sample accounts reproducibly.",
    )
    parser.add_argument(
        "--selection-basis",
        choices=["horizon", "period"],
        default="horizon",
        help=(
            "Select runs by horizon-total net-flow RMSE or by individual-period "
            "net-flow RMSE."
        ),
    )
    return parser.parse_args()


def make_selection_row(
    model: str,
    source: dict[str, object],
    best: pd.Series,
    selection_basis: str,
) -> dict[str, object]:
    row = {
        "model": model,
        "frequency": str(best["frequency"]),
        "method": str(best["method"]),
        "rmse": float(best["rmse"]),
        "mae": float(best["mae"]),
        "wape": float(best["wape"]),
        "selection_basis": selection_basis,
    }
    run_key = source["run_key"]
    if run_key is not None:
        row[str(run_key)] = str(best[str(run_key)])

    for column in best.index:
        if column not in row and column in {
            "order", "order_label", "grid_id", "max_depth", "num_leaves",
            "n_estimators", "learning_rate", "min_child_weight",
            "min_child_samples", "subsample", "colsample_bytree",
            "reg_lambda", "reg_alpha", "gamma", "min_split_gain",
        }:
            row[column] = best[column]
    return row


def load_best_horizon_runs() -> pd.DataFrame:
    """Select one completed run per model/frequency by horizon-total RMSE."""
    rows: list[dict[str, object]] = []
    for model, source in MODEL_SOURCES.items():
        metrics_path = source["metrics"]
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found for {model}: {metrics_path}")

        metrics = pd.read_csv(metrics_path)
        required = {"target", "frequency", "method", "rmse"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")

        candidates = metrics.loc[
            metrics["target"].eq("net_flow") & metrics["horizon_step"].eq("horizon_total")
        ].copy()
        if candidates.empty:
            raise ValueError(f"No horizon-level net_flow rows found in {metrics_path}")

        for frequency in ("monthly", "weekly"):
            frequency_candidates = candidates.loc[
                candidates["frequency"].eq(frequency)
            ].copy()
            if frequency_candidates.empty:
                raise ValueError(f"No {frequency} net_flow rows found for {model}")

            tie_columns = [column for column in ("method", "order_label", "grid_id") if column in frequency_candidates]
            frequency_candidates = frequency_candidates.sort_values(
                ["rmse", *tie_columns], kind="stable"
            )
            best = frequency_candidates.iloc[0]
            rows.append(make_selection_row(model, source, best, "horizon"))

    return pd.DataFrame(rows)


def load_best_period_runs() -> pd.DataFrame:
    """Select one completed run per model/frequency by period-level RMSE."""
    rows: list[dict[str, object]] = []
    for model, source in MODEL_SOURCES.items():
        metrics_paths = sorted(source["root"].rglob("metrics.csv"))
        if not metrics_paths:
            raise FileNotFoundError(
                f"No run-level metrics.csv files found for {model}: {source['root']}"
            )

        candidates: list[pd.DataFrame] = []
        for metrics_path in metrics_paths:
            try:
                metrics = pd.read_csv(metrics_path)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                raise ValueError(f"Could not read {metrics_path}") from exc
            required = {"target", "frequency", "method", "horizon_step", "rmse"}
            missing = required - set(metrics.columns)
            if missing:
                raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")
            period_rows = metrics.loc[
                metrics["target"].eq("net_flow")
                & metrics["horizon_step"].astype(str).eq("all")
            ].copy()
            if not period_rows.empty:
                candidates.append(period_rows)

        if not candidates:
            raise ValueError(f"No period-level net_flow rows found for {model}")
        all_candidates = pd.concat(candidates, ignore_index=True)

        for frequency in ("monthly", "weekly"):
            frequency_candidates = all_candidates.loc[
                all_candidates["frequency"].eq(frequency)
            ].copy()
            if frequency_candidates.empty:
                raise ValueError(f"No {frequency} period-level rows found for {model}")
            tie_columns = [
                column
                for column in ("method", "order_label", "grid_id")
                if column in frequency_candidates
            ]
            best = frequency_candidates.sort_values(
                ["rmse", *tie_columns], kind="stable"
            ).iloc[0]
            rows.append(make_selection_row(model, source, best, "period"))

    return pd.DataFrame(rows)


def load_best_runs(selection_basis: str) -> pd.DataFrame:
    if selection_basis == "period":
        return load_best_period_runs()
    return load_best_horizon_runs()


def selection_description(selected: pd.DataFrame) -> str:
    basis = str(selected["selection_basis"].iloc[0])
    if basis == "period":
        return "individual-period net-flow RMSE"
    return "horizon-total net-flow RMSE"


def prediction_path(model: str, frequency: str, method: str, run: pd.Series) -> Path:
    source = MODEL_SOURCES[model]
    root = source["root"]
    if source["run_key"] is None:
        run_dir = root / f"{frequency}_{method}"
    else:
        run_dir = root / str(run[source["run_key"]]) / f"{frequency}_{method}"
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found for {model}: {path}")
    return path


def load_selected_predictions(selected: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    for _, run in selected.iterrows():
        model = str(run["model"])
        frequency = str(run["frequency"])
        path = prediction_path(model, frequency, str(run["method"]), run)
        frame = pd.read_csv(path)
        required = {"account_id", "period_start", *sum(TARGET_COLUMNS.values(), ())}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["period_start"] = pd.to_datetime(frame["period_start"])
        frame["account_id"] = frame["account_id"].astype(int)
        frame = frame.sort_values(["account_id", "period_start"]).reset_index(drop=True)
        loaded[(model, frequency)] = frame
    return loaded


def choose_accounts(
    loaded: dict[tuple[str, str], pd.DataFrame],
    frequency: str,
    n_accounts: int,
    random_state: int,
) -> list[int]:
    account_sets = [
        set(frame["account_id"].unique())
        for (model, frame_frequency), frame in loaded.items()
        if frame_frequency == frequency
    ]
    if not account_sets:
        raise ValueError(f"No loaded predictions for {frequency}")
    common_accounts = sorted(set.intersection(*account_sets))
    if not common_accounts:
        raise ValueError(f"No common accounts across models for {frequency}")
    sample_size = min(n_accounts, len(common_accounts))
    rng = np.random.default_rng(random_state + (0 if frequency == "monthly" else 1))
    return sorted(rng.choice(common_accounts, size=sample_size, replace=False).tolist())


def compact_label(run: pd.Series) -> str:
    model = str(run["model"])
    label = f"{model} · {run['frequency']}_{run['method']}"
    if model == "ARIMA":
        label += f" · {run['order_label']}"
    elif model in {"XGBoost", "LightGBM"}:
        label += f" · {run['grid_id']}"
    return label


def plot_account_comparison(
    loaded: dict[tuple[str, str], pd.DataFrame],
    selected: pd.DataFrame,
    frequency: str,
    target: str,
    account_ids: list[int],
    selection_label: str,
    output_path: Path,
) -> None:
    actual_column, prediction_column = TARGET_COLUMNS[target]
    runs = selected.loc[selected["frequency"].eq(frequency)].set_index("model")
    n_columns = 2
    n_rows = int(np.ceil(len(account_ids) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18, max(7.5, 4.5 * n_rows)),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, account_id in zip(axes_flat, account_ids):
        actual_plotted = False
        for model in MODEL_SOURCES:
            frame = loaded[(model, frequency)]
            account_frame = frame.loc[frame["account_id"].eq(account_id)].sort_values("period_start")
            if account_frame.empty:
                continue
            if not actual_plotted:
                axis.plot(
                    account_frame["period_start"],
                    account_frame[actual_column],
                    color="#222222",
                    linewidth=2.0,
                    linestyle="--",
                    label="Actual",
                    zorder=5,
                )
                actual_plotted = True
            axis.plot(
                account_frame["period_start"],
                account_frame[prediction_column],
                color=MODEL_COLORS[model],
                linewidth=1.5,
                label=model,
            )
        axis.set_title(f"account_id = {account_id}")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
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
    fig.suptitle(
        f"{frequency.title()} {target}: actual vs model forecasts\n"
        f"Each model uses its best completed run by {selection_label}",
        y=0.995,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_net_flow(
    loaded: dict[tuple[str, str], pd.DataFrame],
    selected: pd.DataFrame,
    frequency: str,
    selection_label: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(15, 6))
    plotted_actual = False
    for model in MODEL_SOURCES:
        frame = loaded[(model, frequency)]
        aggregate = (
            frame.groupby("period_start", as_index=False)[
                ["actual_net_flow", "prediction_net_flow"]
            ]
            .sum()
            .sort_values("period_start")
        )
        if not plotted_actual:
            axis.plot(
                aggregate["period_start"],
                aggregate["actual_net_flow"],
                color="#222222",
                linewidth=2.5,
                linestyle="--",
                label="Actual",
            )
            plotted_actual = True
        axis.plot(
            aggregate["period_start"],
            aggregate["prediction_net_flow"],
            color=MODEL_COLORS[model],
            linewidth=1.8,
            label=model,
        )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_title(
        f"{frequency.title()} aggregate net flow: actual vs model forecasts\n"
        f"Best completed run per model selected by {selection_label}"
    )
    axis.set_xlabel("Period")
    axis.set_ylabel("Total net flow")
    axis.grid(True, alpha=0.25)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_accounts < 1:
        raise ValueError("--n-accounts must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_best_runs(args.selection_basis)
    selection_label = selection_description(selected)
    loaded = load_selected_predictions(selected)

    account_rows: list[dict[str, object]] = []
    for frequency in ("monthly", "weekly"):
        account_ids = choose_accounts(
            loaded,
            frequency,
            args.n_accounts,
            args.random_state,
        )
        account_rows.extend(
            {"frequency": frequency, "account_id": account_id}
            for account_id in account_ids
        )
        for target in TARGET_COLUMNS:
            plot_account_comparison(
                loaded,
                selected,
                frequency,
                target,
                account_ids,
                selection_label,
                args.output_dir / f"{frequency}_{target}_accounts.png",
            )
        plot_aggregate_net_flow(
            loaded,
            selected,
            frequency,
            selection_label,
            args.output_dir / f"{frequency}_aggregate_net_flow.png",
        )

    selected.to_csv(args.output_dir / "selected_best_runs.csv", index=False)
    pd.DataFrame(account_rows).to_csv(
        args.output_dir / "sample_account_ids.csv", index=False
    )

    print(f"Saved model comparison outputs to {args.output_dir}")
    print("Selected runs by frequency:")
    print(
        selected[["model", "frequency", "method", "rmse", "mae", "wape"]]
        .sort_values(["frequency", "rmse"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
