"""EDA plots for demographic/account features versus net flow.

The plots use one row per account.  This avoids giving accounts with more
observed periods more weight than other accounts and makes gender and
account-frequency comparisons interpretable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
DEFAULT_MASTER = ROOT / "Berka Dataset" / "Berka Dataset" / "master_tables" / "berka_master_m_enriched.parquet"
DEFAULT_OUTPUT = ROOT / "outputs" / "eda" / "demographics_net_flow"


def build_account_summary(master_path: Path, pretest_only: bool) -> pd.DataFrame:
    columns = [
        "account_id",
        "period_start",
        "net_flow",
        "transaction_count",
        "birth_year",
        "gender",
        "account_frequency",
    ]
    df = pd.read_parquet(master_path, columns=columns)
    df["account_id"] = df["account_id"].astype(int)
    df["period_start"] = pd.to_datetime(df["period_start"])
    if pretest_only:
        df = df.loc[df["period_start"] < pd.Timestamp("1998-07-01")].copy()
    df["net_flow"] = pd.to_numeric(df["net_flow"], errors="coerce")
    df["transaction_count"] = pd.to_numeric(df["transaction_count"], errors="coerce").fillna(0.0)

    summary = (
        df.groupby("account_id", sort=True)
        .agg(
            birth_year=("birth_year", "first"),
            gender=("gender", "first"),
            account_frequency=("account_frequency", "first"),
            mean_net_flow=("net_flow", "mean"),
            median_net_flow=("net_flow", "median"),
            total_net_flow=("net_flow", "sum"),
            mean_transaction_count=("transaction_count", "mean"),
            total_transaction_count=("transaction_count", "sum"),
            observed_periods=("period_start", "nunique"),
        )
        .reset_index()
    )
    summary["birth_year"] = pd.to_numeric(summary["birth_year"], errors="coerce")
    return summary


def czech_label(value: object) -> str:
    labels = {
        "M": "Male",
        "F": "Female",
        "POPLATEK MESICNE": "monthly",
        "POPLATEK TYDNE": "weekly",
        "POPLATEK PO OBRATU": "after_transaction",
    }
    return labels.get(str(value), str(value))


def save_statistics(summary: pd.DataFrame, output_dir: Path) -> None:
    summary.to_csv(output_dir / "account_level_demographic_net_flow.csv", index=False)

    group_rows: list[dict[str, object]] = []
    for group_column in ("gender", "account_frequency"):
        grouped = summary.groupby(group_column, dropna=False)
        for group_value, values in grouped:
            group_rows.append(
                {
                    "group_column": group_column,
                    "group": czech_label(group_value),
                    "raw_group": group_value,
                    "accounts": int(values["account_id"].nunique()),
                    "mean_net_flow_mean": float(values["mean_net_flow"].mean()),
                    "mean_net_flow_median": float(values["mean_net_flow"].median()),
                    "total_transaction_count_mean": float(values["total_transaction_count"].mean()),
                    "total_transaction_count_median": float(values["total_transaction_count"].median()),
                    "mean_transaction_count_mean": float(values["mean_transaction_count"].mean()),
                }
            )
    pd.DataFrame(group_rows).to_csv(output_dir / "demographic_group_summary.csv", index=False)

    valid = summary[["birth_year", "mean_net_flow", "total_transaction_count"]].dropna()
    correlation_rows: list[dict[str, object]] = []
    for target in ("mean_net_flow", "total_transaction_count"):
        pearson_stat, pearson_p = pearsonr(valid["birth_year"], valid[target])
        spearman_stat, spearman_p = spearmanr(valid["birth_year"], valid[target])
        correlation_rows.append(
            {
                "feature": "birth_year",
                "target": target,
                "pearson_r": float(pearson_stat),
                "pearson_p_value": float(pearson_p),
                "spearman_rho": float(spearman_stat),
                "spearman_p_value": float(spearman_p),
            }
        )
    pd.DataFrame(correlation_rows).to_csv(output_dir / "birth_year_correlations.csv", index=False)

    tests: list[dict[str, object]] = []
    for group_column in ("gender", "account_frequency"):
        for target in ("mean_net_flow", "total_transaction_count"):
            groups = [
                values[target].dropna().to_numpy()
                for _, values in summary.groupby(group_column, dropna=False)
            ]
            statistic, p_value = kruskal(*groups)
            tests.append(
                {
                    "group_column": group_column,
                    "target": target,
                    "kruskal_statistic": float(statistic),
                    "p_value": float(p_value),
                    "interpretation": "distribution differs across groups" if p_value < 0.05 else "no evidence of distribution difference at 5% level",
                }
            )
    pd.DataFrame(tests).to_csv(output_dir / "demographic_kruskal_tests.csv", index=False)


def save_birth_year_plot(summary: pd.DataFrame, output_path: Path) -> None:
    work = summary[["birth_year", "mean_net_flow"]].dropna().sort_values("birth_year")
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.scatter(work["birth_year"], work["mean_net_flow"], alpha=0.38, s=18, color="#1f77b4")
    if len(work) >= 2:
        coefficients = np.polyfit(work["birth_year"], work["mean_net_flow"], deg=1)
        x_values = np.linspace(work["birth_year"].min(), work["birth_year"].max(), 100)
        axis.plot(x_values, np.polyval(coefficients, x_values), color="#d62728", linewidth=2, label="linear trend")
        axis.legend(frameon=False)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("Birth year vs account-level mean net flow")
    axis.set_xlabel("Birth year (higher = younger)")
    axis.set_ylabel("Mean net flow across observed periods")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_gender_plot(summary: pd.DataFrame, output_path: Path) -> None:
    categories = [value for value in ("F", "M") if value in set(summary["gender"].dropna())]
    labels = [czech_label(value) for value in categories]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    for axis, target, title in [
        (axes[0][0], "total_transaction_count", "Total transaction count by gender"),
        (axes[0][1], "mean_net_flow", "Mean net flow by gender"),
    ]:
        values = [summary.loc[summary["gender"].eq(category), target].dropna() for category in categories]
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(title)
        axis.set_ylabel(target)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Gender and account-level transaction/net-flow patterns", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_frequency_plot(summary: pd.DataFrame, output_path: Path) -> None:
    category_order = [
        "POPLATEK MESICNE",
        "POPLATEK TYDNE",
        "POPLATEK PO OBRATU",
    ]
    categories = [value for value in category_order if value in set(summary["account_frequency"].dropna())]
    labels = [czech_label(value) for value in categories]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    for axis, target, title in [
        (axes[0][0], "mean_net_flow", "Mean net flow by account frequency"),
        (axes[0][1], "total_transaction_count", "Total transaction count by account frequency"),
    ]:
        values = [summary.loc[summary["account_frequency"].eq(category), target].dropna() for category in categories]
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(title)
        axis.set_ylabel(target)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Account frequency and account-level net-flow/transaction patterns", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretest-only", action="store_true", help="Use only periods before 1998-07-01")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_account_summary(args.master_path, args.pretest_only)
    save_statistics(summary, args.output_dir)
    save_birth_year_plot(summary, args.output_dir / "birth_year_vs_net_flow.png")
    save_gender_plot(summary, args.output_dir / "gender_transaction_net_flow.png")
    save_frequency_plot(summary, args.output_dir / "account_frequency_net_flow.png")
    print(f"saved EDA outputs to {args.output_dir}")
    print(pd.read_csv(args.output_dir / "demographic_group_summary.csv").to_string(index=False))
    print(pd.read_csv(args.output_dir / "birth_year_correlations.csv").to_string(index=False))
    print(pd.read_csv(args.output_dir / "demographic_kruskal_tests.csv").to_string(index=False))


if __name__ == "__main__":
    main()
