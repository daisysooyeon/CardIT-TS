"""Aggregate Random Forest MDI importance for the selected weekly runs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT / "outputs" / "random_forest_enriched_period_rmse_sweep" / "full"
OUTPUT_ROOT = ROOT / "outputs" / "model_analysis"

GRID_IDS = {
    # Selected under the current primary criterion: weekly period-mean RMSE.
    "period_mean_rmse_best": "grid_023_e500_d8_mss10_msl5_mf0p7",
    # Reference selected under the aggregate 27-week horizon RMSE criterion.
    "horizon_rmse_best": "grid_003_e200_d8_mss2_msl5_mf0p7",
}


def score_summary(run_dir: Path) -> tuple[float, float]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    step = pd.to_numeric(metrics["horizon_step"], errors="coerce")
    period_rows = metrics.loc[step.notna() & metrics["target"].eq("net_flow")]
    period_mean = float(period_rows["rmse"].mean())

    horizon = pd.read_csv(run_dir / "horizon_metrics.csv")
    horizon_rows = horizon.loc[
        horizon["target"].eq("net_flow")
        & horizon["horizon_step"].astype(str).eq("horizon_total")
    ]
    if horizon_rows.empty:
        raise ValueError(f"No horizon_total net_flow row found in {run_dir}")
    horizon_rmse = float(horizon_rows["rmse"].iloc[0])
    return period_mean, horizon_rmse


def aggregate(grid_id: str) -> pd.DataFrame:
    run_dir = SWEEP_ROOT / grid_id / "weekly_direct"
    importance_path = run_dir / "feature_importance.csv"
    if not importance_path.exists():
        raise FileNotFoundError(importance_path)

    frame = pd.read_csv(importance_path)
    required = {"target", "horizon_step", "feature", "importance_mdi"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {importance_path}: {sorted(missing)}")

    # MDI is a share within each fitted target/horizon model.  Average the
    # normalized shares across the 27 direct weekly horizon models.
    totals = frame.groupby(["target", "horizon_step"])["importance_mdi"].transform(
        "sum"
    )
    frame["importance_share"] = frame["importance_mdi"].div(
        totals.where(totals.ne(0), 1.0)
    )
    mean_by_flow = (
        frame.groupby(["target", "feature"], as_index=False)["importance_share"]
        .mean()
        .rename(columns={"importance_share": "mean_importance"})
    )

    pivot = (
        mean_by_flow.pivot(index="feature", columns="target", values="mean_importance")
        .fillna(0.0)
        .reindex(columns=["inflow", "outflow"], fill_value=0.0)
        .reset_index()
    )
    pivot["combined_flow_importance"] = pivot[["inflow", "outflow"]].mean(axis=1)
    pivot["rank_combined"] = (
        pivot["combined_flow_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    pivot["rank_inflow"] = pivot["inflow"].rank(method="first", ascending=False).astype(int)
    pivot["rank_outflow"] = pivot["outflow"].rank(method="first", ascending=False).astype(int)

    period_mean, horizon_rmse = score_summary(run_dir)
    pivot.insert(0, "grid_id", grid_id)
    pivot.insert(1, "selected_run", str(run_dir.relative_to(ROOT)))
    pivot.insert(2, "period_mean_net_flow_rmse", period_mean)
    pivot.insert(3, "horizon_net_flow_rmse", horizon_rmse)
    return pivot.sort_values("combined_flow_importance", ascending=False).reset_index(
        drop=True
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for criterion, grid_id in GRID_IDS.items():
        summary = aggregate(grid_id)
        output_name = (
            f"random_forest_enriched_weekly_direct_{criterion}_feature_importance.csv"
        )
        summary.to_csv(OUTPUT_ROOT / output_name, index=False)

        no_history = summary.loc[
            ~summary["feature"].str.contains("lag|rolling|smooth", case=False, regex=True)
        ].copy()
        no_history.to_csv(
            OUTPUT_ROOT
            / f"random_forest_enriched_weekly_direct_{criterion}_feature_importance_no_lag_rolling.csv",
            index=False,
        )

        print(f"\n--- {criterion}: {grid_id}")
        print(
            summary.head(10)[
                ["feature", "inflow", "outflow", "combined_flow_importance"]
            ].to_string(index=False)
        )
        print("\nTop 10 excluding lag/rolling/smoothing:")
        print(
            no_history.head(10)[
                ["feature", "inflow", "outflow", "combined_flow_importance"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
