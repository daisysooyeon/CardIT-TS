"""Plot account-level transaction amount time series from Berka master tables.

The master tables already contain the period-level aggregates used here:
``total_transaction_amount``, ``inflow_amount`` and ``outflow_amount``.

Examples
--------
python plot_account_timeseries.py
python plot_account_timeseries.py --account-ids 1 212 3521 --frequency monthly
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
MASTER_DIR = ROOT / "Berka Dataset" / "Berka Dataset" / "master_tables"
OUTPUT_DIR = ROOT / "eda_outputs" / "account_timeseries"

VALUE_COLUMNS = [
    "account_id",
    "period_start",
    "total_transaction_amount",
    "inflow_amount",
    "outflow_amount",
    "net_flow",
    "has_transaction",
]


def read_master(frequency: str) -> pd.DataFrame:
    suffix = "m" if frequency == "monthly" else "w"
    path = MASTER_DIR / f"berka_master_{suffix}.parquet"
    df = pd.read_parquet(path, columns=VALUE_COLUMNS)
    df["period_start"] = pd.to_datetime(df["period_start"])
    return df.sort_values(["account_id", "period_start"]).reset_index(drop=True)


def amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("account_id", as_index=False)
        .agg(
            total_transaction_amount=("total_transaction_amount", "sum"),
            inflow_amount=("inflow_amount", "sum"),
            outflow_amount=("outflow_amount", "sum"),
            active_periods=("has_transaction", "sum"),
            periods=("period_start", "size"),
        )
    )
    summary["activity_rate"] = summary["active_periods"] / summary["periods"]
    return summary.sort_values("total_transaction_amount", ascending=False)


def choose_accounts(summary: pd.DataFrame, account_ids: list[int] | None, n: int) -> list[int]:
    if account_ids:
        available = set(summary["account_id"].astype(int))
        missing = sorted(set(account_ids) - available)
        if missing:
            raise ValueError(f"Unknown account_id(s): {missing}")
        return account_ids
    return summary.head(n)["account_id"].astype(int).tolist()


def money_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.grid(True, alpha=0.25)


def plot_account(df: pd.DataFrame, account_id: int, frequency: str, output_path: Path) -> None:
    account = df.loc[df["account_id"].eq(account_id)].sort_values("period_start")
    if account.empty:
        raise ValueError(f"No observations for account_id={account_id}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1]},
        constrained_layout=True,
    )

    # Positive amounts make total = inflow + outflow visually obvious.
    axes[0].plot(
        account["period_start"], account["total_transaction_amount"],
        label="Total transaction amount", color="#1f77b4", linewidth=1.8,
    )
    axes[0].plot(
        account["period_start"], account["inflow_amount"],
        label="Inflow", color="#2ca02c", linewidth=1.2,
    )
    axes[0].plot(
        account["period_start"], account["outflow_amount"],
        label="Outflow", color="#d62728", linewidth=1.2,
    )
    axes[0].set_ylabel("Amount")
    axes[0].set_title("Transaction amount (positive magnitude)")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    money_axis(axes[0])

    # Directional view is useful for later investable-amount modeling.
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].plot(
        account["period_start"], account["inflow_amount"],
        label="Inflow (+)", color="#2ca02c", linewidth=1.2,
    )
    axes[1].plot(
        account["period_start"], -account["outflow_amount"],
        label="Outflow (-)", color="#d62728", linewidth=1.2,
    )
    axes[1].plot(
        account["period_start"], account["net_flow"],
        label="Net flow", color="#9467bd", linewidth=1.6,
    )
    axes[1].set_ylabel("Cash flow")
    axes[1].set_title("Directional cash flow")
    axes[1].legend(loc="upper left", ncol=3, frameon=False)
    money_axis(axes[1])

    locator = mdates.MonthLocator(interval=3 if frequency == "monthly" else 6)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.suptitle(f"Account {account_id} — {frequency}", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(df: pd.DataFrame, account_ids: list[int], frequency: str, output_path: Path) -> None:
    ncols = 3
    nrows = (len(account_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, account_id in zip(axes, account_ids):
        account = df.loc[df["account_id"].eq(account_id)].sort_values("period_start")
        ax.plot(account["period_start"], account["total_transaction_amount"], label="Total", color="#1f77b4", linewidth=1.3)
        ax.plot(account["period_start"], account["inflow_amount"], label="Inflow", color="#2ca02c", linewidth=0.9)
        ax.plot(account["period_start"], account["outflow_amount"], label="Outflow", color="#d62728", linewidth=0.9)
        ax.set_title(f"Account {account_id}")
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12 if frequency == "monthly" else 24))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes[len(account_ids):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Top accounts by total transaction amount — {frequency}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["monthly", "weekly", "both"], default="both")
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for frequency in frequencies:
        df = read_master(frequency)
        summary = amount_summary(df)
        summary.to_csv(args.output_dir / f"account_summary_{frequency}.csv", index=False)
        account_ids = choose_accounts(summary, args.account_ids, args.top_n)
        plot_overview(df, account_ids, frequency, args.output_dir / f"overview_{frequency}.png")
        for account_id in account_ids:
            plot_account(df, account_id, frequency, args.output_dir / f"account_{account_id}_{frequency}.png")
        print(f"{frequency}: plotted account_ids={account_ids}")


if __name__ == "__main__":
    main()
