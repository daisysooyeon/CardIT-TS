"""Compare the completed enriched Random Forest run with prior weekly models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "model_comparison_with_rf"
ACCOUNT_IDS_PATH = (
    ROOT
    / "outputs"
    / "linear_regression"
    / "analysis_period_rmse"
    / "sampled_account_ids.csv"
)

MODEL_ORDER = ["LR", "ARIMA", "XGBoost", "LightGBM", "Random Forest"]
COLORS = {
    "LR": "#1f77b4",
    "ARIMA": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "LightGBM": "#9467bd",
    "Random Forest": "#d62728",
}


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["account_id"] = frame["account_id"].astype(int)
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    if "actual_net_flow" not in frame.columns:
        frame["actual_net_flow"] = frame["actual_inflow"] - frame["actual_outflow"]
    if "prediction_net_flow" not in frame.columns:
        frame["prediction_net_flow"] = (
            frame["prediction_inflow"] - frame["prediction_outflow"]
        )
    return frame.sort_values(["account_id", "period_start"])


def model_paths() -> dict[str, Path]:
    selected_path = (
        ROOT
        / "outputs"
        / "model_comparison_both_rmse"
        / "selected_best_runs.csv"
    )
    selected = pd.read_csv(selected_path)
    selected = selected.loc[
        selected["criterion"].eq("period_mean_rmse_best")
        & selected["frequency"].eq("weekly")
        & selected["method"].eq("direct")
    ].set_index("model")

    paths = {
        "LR": ROOT / "outputs" / "linear_regression" / "weekly_direct" / "predictions.csv",
        "ARIMA": ROOT / "outputs" / "arima_order_grid",
        "XGBoost": ROOT / "outputs" / "xgboost_regularization_sweep",
        "LightGBM": ROOT / "outputs" / "lightgbm_regularization_sampling_sweep",
    }
    for model in ["ARIMA", "XGBoost", "LightGBM"]:
        row = selected.loc[model]
        paths[model] = (
            paths[model]
            / str(row["run_id"])
            / "weekly_direct"
            / "predictions.csv"
        )

    paths["Random Forest"] = (
        ROOT
        / "outputs"
        / "random_forest_enriched_period_rmse_sweep"
        / "full"
        / "grid_003_e200_d8_mss2_msl5_mf0p7"
        / "weekly_direct"
        / "predictions.csv"
    )
    return paths


def main() -> None:
    account_ids = pd.read_csv(ACCOUNT_IDS_PATH)["account_id"].astype(int).tolist()
    paths = model_paths()
    frames = {model: read_predictions(path) for model, path in paths.items()}

    available = set(frames["Random Forest"]["account_id"])
    account_ids = [account_id for account_id in account_ids if account_id in available]

    n_columns = 2
    n_rows = (len(account_ids) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(19, max(7.5, 4.7 * n_rows)),
        squeeze=False,
    )

    for axis, account_id in zip(axes.ravel(), account_ids):
        actual_plotted = False
        for model in MODEL_ORDER:
            frame = frames[model]
            account_frame = frame.loc[frame["account_id"].eq(account_id)]
            if account_frame.empty:
                continue
            if not actual_plotted:
                axis.plot(
                    account_frame["period_start"],
                    account_frame["actual_net_flow"],
                    color="#222222",
                    linewidth=2.0,
                    linestyle="--",
                    marker="o",
                    markersize=3,
                    label="Actual",
                    zorder=5,
                )
                actual_plotted = True
            axis.plot(
                account_frame["period_start"],
                account_frame["prediction_net_flow"],
                color=COLORS[model],
                linewidth=1.35,
                marker="o",
                markersize=2.5,
                label=model,
            )
        axis.set_title(f"account_id = {account_id}")
        axis.set_ylabel("Net flow")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.tick_params(axis="x", rotation=35)

    for axis in axes.ravel()[len(account_ids) :]:
        axis.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "Weekly net_flow: actual vs model forecasts\n"
        "Period-mean RMSE best runs + completed enriched Random Forest grid_003",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "weekly_period_rmse_with_random_forest.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
