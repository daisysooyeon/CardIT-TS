"""Plot non-enriched XGBoost best runs for the LR sample accounts.

The script compares two representative runs from the 27-grid
``xgboost_regularization_sweep`` output:

* Horizon-total RMSE best: monthly direct, grid_016.
* Mean period-level RMSE best: weekly direct, grid_010.

Grid 010 is representative of a tie with grids 011 and 012 for the
period-level criterion.  Each account receives one figure with three target
columns (inflow, outflow, net flow) and two rows (the two selected runs).
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
OUTPUT_ROOT = SWEEP_ROOT / "analysis_best_net_flow_rmse"

TARGETS = {
    "inflow": ("actual_inflow", "prediction_inflow"),
    "outflow": ("actual_outflow", "prediction_outflow"),
    "net_flow": ("actual_net_flow", "prediction_net_flow"),
}

RUN_SPECS = [
    {
        "criterion": "horizon_rmse_best",
        "label": "Horizon RMSE best",
        "frequency": "monthly",
        "method": "direct",
        "grid_prefix": "grid_016_",
    },
    {
        "criterion": "period_mean_rmse_best",
        "label": "Mean period RMSE best",
        "frequency": "weekly",
        "method": "direct",
        "grid_prefix": "grid_010_",
    },
]


def find_grid_dir(prefix: str) -> Path:
    matches = sorted(path for path in SWEEP_ROOT.glob(f"{prefix}*") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one grid directory for {prefix!r}, found: {matches}"
        )
    return matches[0]


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
    if not ACCOUNT_IDS_PATH.exists():
        raise FileNotFoundError(f"LR sample account file not found: {ACCOUNT_IDS_PATH}")
    frame = pd.read_csv(ACCOUNT_IDS_PATH)
    if "account_id" not in frame.columns:
        raise ValueError(f"{ACCOUNT_IDS_PATH} must contain account_id")
    return sorted(frame["account_id"].astype(int).tolist())


def load_selected_runs() -> list[dict[str, object]]:
    horizon_metrics = pd.read_csv(SWEEP_ROOT / "horizon_comparison_metrics.csv")
    selected: list[dict[str, object]] = []

    for spec in RUN_SPECS:
        grid_dir = find_grid_dir(str(spec["grid_prefix"]))
        grid_id = grid_dir.name
        run_dir = grid_dir / f"{spec['frequency']}_{spec['method']}"
        predictions = read_predictions(run_dir)

        horizon_row = horizon_metrics.loc[
            horizon_metrics["grid_id"].eq(grid_id)
            & horizon_metrics["frequency"].eq(spec["frequency"])
            & horizon_metrics["method"].eq(spec["method"])
            & horizon_metrics["target"].eq("net_flow")
            & horizon_metrics["horizon_step"].astype(str).eq("horizon_total")
        ]
        if horizon_row.empty:
            raise ValueError(f"No horizon net-flow RMSE found for {grid_id}")

        period_metrics = pd.read_csv(run_dir / "metrics.csv")
        period_metrics = period_metrics.loc[
            period_metrics["target"].eq("net_flow")
            & ~period_metrics["horizon_step"].astype(str).eq("all")
        ].copy()
        if period_metrics.empty:
            raise ValueError(f"No period net-flow metrics found in {run_dir}")

        selected.append(
            {
                **spec,
                "grid_id": grid_id,
                "grid_dir": grid_dir,
                "run_dir": run_dir,
                "predictions": predictions,
                "horizon_rmse": float(horizon_row.iloc[0]["rmse"]),
                "mean_period_rmse": float(period_metrics["rmse"].mean()),
                "periods": int(len(period_metrics)),
                "max_depth": int(horizon_row.iloc[0]["max_depth"]),
                "n_estimators": int(horizon_row.iloc[0]["n_estimators"]),
                "learning_rate": float(horizon_row.iloc[0]["learning_rate"]),
                "min_child_weight": float(horizon_row.iloc[0]["min_child_weight"]),
                "subsample": float(horizon_row.iloc[0]["subsample"]),
                "colsample_bytree": float(horizon_row.iloc[0]["colsample_bytree"]),
                "reg_lambda": float(horizon_row.iloc[0]["reg_lambda"]),
                "reg_alpha": float(horizon_row.iloc[0]["reg_alpha"]),
                "gamma": float(horizon_row.iloc[0]["gamma"]),
            }
        )

    return selected


def plot_account(
    account_id: int,
    selected: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 9), squeeze=False)
    handles = None
    labels = None

    for row_index, spec in enumerate(selected):
        frame = spec["predictions"]
        account_frame = frame.loc[frame["account_id"].eq(account_id)].sort_values(
            "period_start"
        )
        if account_frame.empty:
            raise ValueError(f"account_id={account_id} is missing from {spec['run_dir']}")

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
            axis.set_title(f"{spec['frequency']}_{spec['method']} · {target}")
            axis.set_ylabel("Amount")
            axis.grid(True, alpha=0.25)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
            axis.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m" if spec["frequency"] == "monthly" else "%m-%d")
            )
            axis.tick_params(axis="x", rotation=35)
            if column_index == 0:
                axis.text(
                    -0.28,
                    0.5,
                    str(spec["label"]),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )
            if handles is None:
                handles, labels = axis.get_legend_handles_labels()

    fig.suptitle(
        f"XGBoost non-enriched: actual vs prediction · account_id={account_id}\n"
        "Horizon RMSE best vs mean period RMSE best",
        fontsize=15,
    )
    if handles is not None and labels is not None:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2)
    fig.tight_layout(rect=(0.03, 0.02, 1.0, 0.90))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_selected_summary(selected: list[dict[str, object]], output_dir: Path) -> None:
    columns = [
        "criterion",
        "label",
        "frequency",
        "method",
        "grid_id",
        "horizon_rmse",
        "mean_period_rmse",
        "periods",
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
    pd.DataFrame([{column: row[column] for column in columns} for row in selected]).to_csv(
        output_dir / "selected_runs.csv", index=False
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = load_selected_runs()
    account_ids = read_sample_account_ids()

    pd.DataFrame({"account_id": account_ids}).to_csv(
        OUTPUT_ROOT / "sampled_account_ids.csv", index=False
    )
    save_selected_summary(selected, OUTPUT_ROOT)

    account_dir = OUTPUT_ROOT / "account_plots"
    account_dir.mkdir(parents=True, exist_ok=True)
    for account_id in account_ids:
        plot_account(account_id, selected, account_dir / f"account_{account_id}.png")

    print(f"Saved {len(account_ids)} account plots to {account_dir}")
    print(pd.read_csv(OUTPUT_ROOT / "selected_runs.csv").to_string(index=False))


if __name__ == "__main__":
    main()
