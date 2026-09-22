"""EDA for the relationship between account_frequency and transaction activity.

The analysis is performed at account level, not row level.  For each account it
calculates total transactions, active periods, activity rate, and transactions
per active period, then compares their distributions across account_frequency
categories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal


ROOT = Path(__file__).resolve().parent
DEFAULT_MASTER = ROOT / "Berka Dataset" / "Berka Dataset" / "master_tables" / "berka_master_m_enriched.parquet"
DEFAULT_OUTPUT = ROOT / "outputs" / "eda" / "account_frequency_transaction_frequency"

METRICS = [
    ("total_transactions", "Total transaction count"),
    ("active_periods", "Periods with at least one transaction"),
    ("activity_rate", "Active-period rate"),
    ("transactions_per_active_period", "Transactions per active period"),
]


def build_account_summary(master_path: Path) -> pd.DataFrame:
    columns = [
        "account_id",
        "period_start",
        "account_frequency",
        "transaction_count",
        "has_transaction",
    ]
    df = pd.read_parquet(master_path, columns=columns)
    df["account_id"] = df["account_id"].astype(int)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["transaction_count"] = pd.to_numeric(df["transaction_count"], errors="coerce").fillna(0.0)
    df["has_transaction"] = df["has_transaction"].astype(bool)

    frequency_counts = df.groupby("account_id")["account_frequency"].nunique(dropna=False)
    if (frequency_counts > 1).any():
        raise ValueError("account_frequency is not constant within at least one account")

    summary = (
        df.groupby("account_id", sort=True)
        .agg(
            account_frequency=("account_frequency", "first"),
            total_transactions=("transaction_count", "sum"),
            active_periods=("has_transaction", "sum"),
            observed_periods=("period_start", "nunique"),
            mean_transactions_per_period=("transaction_count", "mean"),
        )
        .reset_index()
    )
    summary["activity_rate"] = summary["active_periods"] / summary["observed_periods"]
    summary["transactions_per_active_period"] = (
        summary["total_transactions"]
        / summary["active_periods"].replace(0, np.nan)
    )
    return summary


def save_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        summary.groupby("account_frequency", dropna=False)
        .agg(
            accounts=("account_id", "nunique"),
            total_transactions_mean=("total_transactions", "mean"),
            total_transactions_median=("total_transactions", "median"),
            total_transactions_std=("total_transactions", "std"),
            active_periods_mean=("active_periods", "mean"),
            active_periods_median=("active_periods", "median"),
            activity_rate_mean=("activity_rate", "mean"),
            activity_rate_median=("activity_rate", "median"),
            transactions_per_active_period_mean=("transactions_per_active_period", "mean"),
            transactions_per_active_period_median=("transactions_per_active_period", "median"),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "account_frequency_summary.csv", index=False)

    test_rows: list[dict[str, object]] = []
    for metric, _ in METRICS:
        groups = [
            values[metric].dropna().to_numpy()
            for _, values in summary.groupby("account_frequency", dropna=False)
        ]
        if len(groups) >= 2 and all(len(values) > 0 for values in groups):
            statistic, p_value = kruskal(*groups)
            test_rows.append(
                {
                    "metric": metric,
                    "kruskal_statistic": float(statistic),
                    "p_value": float(p_value),
                    "interpretation": "distribution differs across account_frequency groups" if p_value < 0.05 else "no evidence of distribution difference at 5% level",
                }
            )
    pd.DataFrame(test_rows).to_csv(output_dir / "account_frequency_kruskal_tests.csv", index=False)


def category_labels(summary: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    categories = [str(value) for value in summary["account_frequency"].dropna().unique()]
    labels = {
        "POPLATEK MESICNE": "monthly",
        "POPLATEK TYDNE": "weekly",
        "POPLATEK PO OBRATU": "after_transaction",
    }
    return categories, labels


def save_boxplots(summary: pd.DataFrame, output_path: Path) -> None:
    categories, labels = category_labels(summary)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for axis, (metric, title) in zip(axes.flat, METRICS):
        values = [summary.loc[summary["account_frequency"].eq(category), metric].dropna() for category in categories]
        axis.boxplot(
            values,
            tick_labels=[labels.get(category, category) for category in categories],
            showfliers=False,
        )
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Transaction activity by account service frequency", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_histograms(summary: pd.DataFrame, output_path: Path) -> None:
    categories, labels = category_labels(summary)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), squeeze=False)
    for axis, metric, title in [
        (axes[0][0], "total_transactions", "Total transaction count distribution"),
        (axes[0][1], "transactions_per_active_period", "Transactions per active period distribution"),
    ]:
        all_values = summary[metric].dropna().to_numpy()
        bins = np.histogram_bin_edges(all_values, bins=30)
        for category in categories:
            values = summary.loc[summary["account_frequency"].eq(category), metric].dropna()
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=labels.get(category, category),
            )
        axis.set_title(title)
        axis.set_xlabel(metric)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Distribution of transaction activity by account frequency", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_account_summary(args.master_path)
    summary.to_csv(args.output_dir / "account_level_transaction_frequency.csv", index=False)
    save_summary(summary, args.output_dir)
    save_boxplots(summary, args.output_dir / "transaction_frequency_boxplots.png")
    save_histograms(summary, args.output_dir / "transaction_frequency_histograms.png")

    print(f"saved account-level summary: {args.output_dir / 'account_level_transaction_frequency.csv'}")
    print(f"saved boxplot: {args.output_dir / 'transaction_frequency_boxplots.png'}")
    print(f"saved histogram: {args.output_dir / 'transaction_frequency_histograms.png'}")
    print(pd.read_csv(args.output_dir / "account_frequency_summary.csv").to_string(index=False))
    print(pd.read_csv(args.output_dir / "account_frequency_kruskal_tests.csv").to_string(index=False))


if __name__ == "__main__":
    main()
