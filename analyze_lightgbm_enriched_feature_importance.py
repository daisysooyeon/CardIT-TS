from pathlib import Path

import pandas as pd


ROOT = Path("outputs/lightgbm_enriched_regularization_sweep")
OUTPUT = Path("outputs/model_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)


def period_mean_net_rmse(run_dir: Path) -> float:
    frame = pd.read_csv(run_dir / "metrics.csv")
    step = pd.to_numeric(frame["horizon_step"], errors="coerce")
    selected = frame.loc[step.notna() & frame["target"].eq("net_flow")]
    return float(selected["rmse"].mean())


def select_best_run(method: str = "weekly_direct") -> tuple[Path, float]:
    candidates = []
    for run_dir in ROOT.glob(f"grid_*/{method}"):
        metrics_path = run_dir / "metrics.csv"
        importance_path = run_dir / "feature_importance.csv"
        if metrics_path.exists() and importance_path.exists():
            candidates.append((period_mean_net_rmse(run_dir), run_dir))
    if not candidates:
        raise FileNotFoundError(f"No completed enriched LightGBM runs found for {method}")
    return min(candidates, key=lambda item: item[0])[1], min(candidates, key=lambda item: item[0])[0]


def aggregate_importance(importance_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(importance_path)
    # Raw gain is not comparable across target/horizon models. Normalize each
    # model first, then average normalized shares across models.
    group_cols = ["target", "horizon_step"]
    totals = frame.groupby(group_cols)["importance_gain"].transform("sum")
    frame["importance_share"] = frame["importance_gain"].div(totals.where(totals.ne(0), 1.0))
    target = (
        frame.groupby(["target", "feature"], as_index=False)["importance_share"]
        .mean()
        .rename(columns={"importance_share": "mean_importance_share"})
    )
    combined = (
        target.groupby("feature", as_index=False)["mean_importance_share"]
        .mean()
        .rename(columns={"mean_importance_share": "combined_flow_importance"})
    )
    combined["combined_rank"] = (
        combined["combined_flow_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result = target.merge(combined, on="feature", how="left")
    result["target_rank"] = result.groupby("target")["mean_importance_share"].rank(
        method="first", ascending=False
    ).astype(int)
    return result.sort_values(["combined_rank", "target", "target_rank"])


def main() -> None:
    run_dir, score = select_best_run("weekly_direct")
    result = aggregate_importance(run_dir / "feature_importance.csv")
    result.insert(0, "selected_run", str(run_dir))
    result.insert(1, "period_mean_net_flow_rmse", score)
    result.to_csv(
        OUTPUT / "lightgbm_enriched_weekly_direct_period_rmse_feature_importance.csv",
        index=False,
    )

    no_history = result.loc[
        ~result["feature"].str.contains("lag|rolling|bagged_smooth", case=False, regex=True)
    ].copy()
    no_history.to_csv(
        OUTPUT / "lightgbm_enriched_weekly_direct_period_rmse_feature_importance_no_lag_rolling.csv",
        index=False,
    )

    print(f"selected_run={run_dir}")
    print(f"period_mean_net_flow_rmse={score:.6f}")
    print("\nCOMBINED TOP 10")
    print(result.drop_duplicates("feature").head(10).to_string(index=False))
    print("\nINFLOW TOP 10")
    print(result.loc[result["target"].eq("inflow")].sort_values("target_rank").head(10).to_string(index=False))
    print("\nOUTFLOW TOP 10")
    print(result.loc[result["target"].eq("outflow")].sort_values("target_rank").head(10).to_string(index=False))
    print("\nCOMBINED TOP 10 WITHOUT LAG/ROLLING")
    print(no_history.drop_duplicates("feature").sort_values("combined_rank").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
