"""Plot the completed enriched Random Forest weekly/direct run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT
    / "outputs"
    / "random_forest_enriched_period_rmse_sweep"
    / "full"
    / "grid_003_e200_d8_mss2_msl5_mf0p7"
    / "weekly_direct"
)
ACCOUNT_IDS_PATH = (
    ROOT
    / "outputs"
    / "linear_regression"
    / "analysis_period_rmse"
    / "sampled_account_ids.csv"
)
OUTPUT_PATH = RUN_DIR / "random_forest_weekly_direct_net_flow_accounts.png"


def main() -> None:
    predictions = pd.read_csv(RUN_DIR / "predictions.csv")
    predictions["account_id"] = predictions["account_id"].astype(int)
    predictions["period_start"] = pd.to_datetime(predictions["period_start"])

    account_ids = (
        pd.read_csv(ACCOUNT_IDS_PATH)["account_id"]
        .astype(int)
        .tolist()
    )
    available = set(predictions["account_id"])
    account_ids = [account_id for account_id in account_ids if account_id in available]

    columns = 2
    rows = (len(account_ids) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, max(7.5, 4.5 * rows)),
        squeeze=False,
    )

    for axis, account_id in zip(axes.ravel(), account_ids):
        frame = predictions.loc[
            predictions["account_id"].eq(account_id)
        ].sort_values("period_start")

        axis.plot(
            frame["period_start"],
            frame["actual_net_flow"],
            color="#222222",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=3,
            label="Actual",
            zorder=5,
        )
        axis.plot(
            frame["period_start"],
            frame["prediction_net_flow"],
            color="#d62728",
            linewidth=1.6,
            marker="o",
            markersize=3,
            label="Random Forest",
        )
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(f"account_id = {account_id}")
        axis.set_ylabel("Net flow")
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
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Weekly net_flow: actual vs Random Forest forecasts\n"
        "grid_003, weekly/direct, enriched features",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
