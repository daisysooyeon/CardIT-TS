from pathlib import Path

import pandas as pd


BASE = Path("outputs/lightgbm_regularization_sampling_sweep")
ENRICHED = Path("outputs/lightgbm_enriched_regularization_sweep")


def period_mean_rmse(path: Path, target: str = "net_flow") -> float:
    frame = pd.read_csv(path)
    step = pd.to_numeric(frame["horizon_step"], errors="coerce")
    selected = frame.loc[step.notna() & frame["target"].eq(target)]
    return float(selected["rmse"].mean())


def collect_period_results(root: Path, source: str) -> pd.DataFrame:
    rows = []
    for path in root.glob("grid_*/weekly_*/metrics.csv"):
        rows.append(
            {
                "source": source,
                "grid_id": path.parent.parent.name,
                "method": path.parent.name.replace("weekly_", ""),
                "period_mean_rmse": period_mean_rmse(path),
            }
        )
    return pd.DataFrame(rows)


def collect_horizon_results(root: Path, source: str) -> pd.DataFrame:
    path = root / "horizon_comparison_metrics.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["frequency"].eq("weekly") & frame["target"].eq("net_flow")
    ].copy()
    frame.insert(0, "source", source)
    return frame[["source", "grid_id", "method", "rmse"]]


def main() -> None:
    period = pd.concat(
        [
            collect_period_results(BASE, "non_enriched"),
            collect_period_results(ENRICHED, "enriched"),
        ],
        ignore_index=True,
    )
    horizon = pd.concat(
        [
            collect_horizon_results(BASE, "non_enriched"),
            collect_horizon_results(ENRICHED, "enriched"),
        ],
        ignore_index=True,
    )

    period_best = (
        period.sort_values("period_mean_rmse")
        .groupby(["source", "method"], as_index=False)
        .first()
    )
    horizon_best = (
        horizon.sort_values("rmse")
        .groupby(["source", "method"], as_index=False)
        .first()
    )

    comparison = period_best.merge(
        horizon_best,
        on=["source", "method", "grid_id"],
        how="outer",
        suffixes=("_period", "_horizon"),
    )
    comparison.to_csv(
        "outputs/model_analysis/lightgbm_enriched_vs_non_enriched_net_flow.csv",
        index=False,
    )

    print("PERIOD-MEAN RMSE BEST")
    print(period_best.to_string(index=False))
    print("\nHORIZON-TOTAL RMSE BEST")
    print(horizon_best.to_string(index=False))
    print("\nENRICHED COMPLETED RUNS")
    print(period.loc[period["source"].eq("enriched")].sort_values("period_mean_rmse").to_string(index=False))
    print(horizon.loc[horizon["source"].eq("enriched")].sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
