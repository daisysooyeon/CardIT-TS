"""Aggregate enriched XGBoost gain importance across weekly direct horizons."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT / "outputs" / "xgboost_enriched_sampling_sweep"
OUTPUT_ROOT = ROOT / "outputs" / "model_analysis"

GRID_IDS = {
    "period_mean_rmse_best": (
        "grid_013_d6_e500_lr0p03_mcw3_ss0p9_cs0p7_l21_l10_g0"
    ),
    "horizon_rmse_best": (
        "grid_019_d6_e500_lr0p03_mcw5_ss0p7_cs0p7_l21_l10_g0"
    ),
}


def aggregate(grid_id: str) -> pd.DataFrame:
    path = SWEEP_ROOT / grid_id / "weekly_direct" / "feature_importance.csv"
    frame = pd.read_csv(path)

    mean_by_flow = (
        frame.groupby(["target", "feature"], as_index=False)["importance_gain"]
        .mean()
        .rename(columns={"importance_gain": "mean_gain"})
    )

    pivot = (
        mean_by_flow.pivot(index="feature", columns="target", values="mean_gain")
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
    pivot["rank_inflow"] = (
        pivot["inflow"].rank(method="first", ascending=False).astype(int)
    )
    pivot["rank_outflow"] = (
        pivot["outflow"].rank(method="first", ascending=False).astype(int)
    )
    return pivot.sort_values("combined_flow_importance", ascending=False).reset_index(
        drop=True
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for criterion, grid_id in GRID_IDS.items():
        summary = aggregate(grid_id)
        summary.insert(0, "criterion", criterion)
        summary.insert(1, "grid_id", grid_id)
        summary.to_csv(
            OUTPUT_ROOT / f"xgboost_enriched_weekly_direct_{criterion}_feature_importance.csv",
            index=False,
        )
        no_history = summary.loc[
            ~summary["feature"].str.contains("lag|rolling", case=False, regex=True)
        ].copy()
        no_history.to_csv(
            OUTPUT_ROOT
            / f"xgboost_enriched_weekly_direct_{criterion}_feature_importance_no_lag_rolling.csv",
            index=False,
        )
        print(f"\n--- {criterion}: {grid_id}")
        print(
            summary.head(10)[
                ["feature", "inflow", "outflow", "combined_flow_importance"]
            ].to_string(index=False)
        )
        print("\nTop 10 excluding lag/rolling:")
        print(
            no_history.head(10)[
                ["feature", "inflow", "outflow", "combined_flow_importance"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
