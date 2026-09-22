"""Analyze LightGBM feature importance for net-flow-selected best runs.

The LightGBM models directly predict inflow and outflow.  Since net flow is
derived as inflow minus outflow, this script reports a transparent proxy:
the mean of the normalized gain importance from the inflow and outflow
models.  Lag and rolling features are excluded from the ranking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT / "outputs" / "lightgbm_regularization_sampling_sweep"
OUTPUT_ROOT = SWEEP_ROOT / "analysis_net_flow_feature_importance_no_lag_rolling"
VERSION_ORDER = [
    ("monthly", "recursive"),
    ("monthly", "direct"),
    ("weekly", "recursive"),
    ("weekly", "direct"),
]


def load_horizon_candidates() -> pd.DataFrame:
    frame = pd.read_csv(SWEEP_ROOT / "horizon_comparison_metrics.csv")
    return frame.loc[
        frame["target"].eq("net_flow")
        & frame["horizon_step"].astype(str).eq("horizon_total")
    ].copy()


def load_period_candidates() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for grid_dir in sorted(SWEEP_ROOT.glob("grid_*")):
        for frequency, method in VERSION_ORDER:
            path = grid_dir / f"{frequency}_{method}" / "metrics.csv"
            frame = pd.read_csv(path)
            frame = frame.loc[
                frame["target"].eq("net_flow")
                & ~frame["horizon_step"].astype(str).eq("all")
            ].copy()
            frame["grid_id"] = grid_dir.name
            frame["frequency"] = frequency
            frame["method"] = method
            rows.append(frame)

    period = pd.concat(rows, ignore_index=True)
    return (
        period.groupby(["frequency", "method", "grid_id"], as_index=False)
        .agg(mean_period_rmse=("rmse", "mean"), n_periods=("rmse", "size"))
    )


def select_runs() -> pd.DataFrame:
    horizon = load_horizon_candidates()
    period = load_period_candidates()
    selected: list[dict[str, object]] = []

    for criterion, candidates, score_column in [
        ("horizon_rmse_best", horizon, "rmse"),
        ("period_mean_rmse_best", period, "mean_period_rmse"),
    ]:
        for frequency, method in VERSION_ORDER:
            group = candidates.loc[
                candidates["frequency"].eq(frequency)
                & candidates["method"].eq(method)
            ].sort_values([score_column, "grid_id"], kind="stable")
            best = group.iloc[0]
            grid_id = str(best["grid_id"])
            horizon_row = horizon.loc[
                horizon["frequency"].eq(frequency)
                & horizon["method"].eq(method)
                & horizon["grid_id"].eq(grid_id)
            ].iloc[0]
            period_row = period.loc[
                period["frequency"].eq(frequency)
                & period["method"].eq(method)
                & period["grid_id"].eq(grid_id)
            ].iloc[0]
            selected.append(
                {
                    "criterion": criterion,
                    "frequency": frequency,
                    "method": method,
                    "grid_id": grid_id,
                    "horizon_rmse": float(horizon_row["rmse"]),
                    "mean_period_rmse": float(period_row["mean_period_rmse"]),
                    "n_periods": int(period_row["n_periods"]),
                }
            )

    return pd.DataFrame(selected)


def clean_feature_name(value: str) -> str:
    return value.removeprefix("numeric__").removeprefix("categorical__")


def is_excluded_feature(value: str) -> bool:
    name = value.lower()
    return "lag" in name or "rolling" in name


def load_net_flow_proxy(grid_id: str, frequency: str, method: str) -> pd.DataFrame:
    path = SWEEP_ROOT / grid_id / f"{frequency}_{method}" / "feature_importance.csv"
    frame = pd.read_csv(path)
    frame["importance_gain"] = frame["importance_gain"].astype(float)
    frame["gain_normalized"] = frame["importance_gain"] / frame.groupby(
        ["target", "horizon_step"]
    )["importance_gain"].transform("sum")
    frame["feature_clean"] = frame["feature"].map(clean_feature_name)

    filtered = frame.loc[~frame["feature_clean"].map(is_excluded_feature)].copy()
    summary = (
        filtered.groupby(["feature_clean", "target"], as_index=False)["gain_normalized"]
        .mean()
        .pivot(index="feature_clean", columns="target", values="gain_normalized")
        .fillna(0.0)
        .reset_index()
    )
    for target in ("inflow", "outflow"):
        if target not in summary.columns:
            summary[target] = 0.0
    summary["net_flow_proxy"] = (summary["inflow"] + summary["outflow"]) / 2.0
    summary["inflow_pct"] = summary["inflow"] * 100.0
    summary["outflow_pct"] = summary["outflow"] * 100.0
    summary["net_flow_proxy_pct"] = summary["net_flow_proxy"] * 100.0
    retained_total = float(summary["net_flow_proxy"].sum())
    summary["retained_importance_pct"] = (
        summary["net_flow_proxy"] / retained_total * 100.0
        if retained_total > 0
        else 0.0
    )
    return summary.sort_values(
        ["net_flow_proxy", "feature_clean"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = select_runs()
    selected.to_csv(OUTPUT_ROOT / "selected_runs.csv", index=False)

    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for _, run in selected.iterrows():
        importance = load_net_flow_proxy(
            str(run["grid_id"]), str(run["frequency"]), str(run["method"])
        )
        top = importance.head(6).copy()
        top.insert(0, "rank", range(1, len(top) + 1))
        for column in (
            "criterion",
            "frequency",
            "method",
            "grid_id",
            "horizon_rmse",
            "mean_period_rmse",
        ):
            top[column] = run[column]
        detail_rows.append(top)

        row: dict[str, object] = {
            "criterion": run["criterion"],
            "frequency": run["frequency"],
            "method": run["method"],
            "grid_id": run["grid_id"],
            "horizon_rmse": run["horizon_rmse"],
            "mean_period_rmse": run["mean_period_rmse"],
        }
        for rank, item in top.iterrows():
            position = int(item["rank"])
            row[f"top{position}_feature"] = item["feature_clean"]
            row[f"top{position}_importance_pct"] = item["retained_importance_pct"]
        summary_rows.append(row)

    pd.concat(detail_rows, ignore_index=True).to_csv(
        OUTPUT_ROOT / "top6_feature_importance_detail.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_ROOT / "top6_feature_importance_table.csv", index=False
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
