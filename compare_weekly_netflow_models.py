from pathlib import Path

import pandas as pd


RUN_ROOTS = {
    "LR enriched": Path("outputs/linear_regression_enriched"),
    "ARIMA": Path("outputs/arima_order_grid"),
    "XGBoost enriched": Path("outputs/xgboost_enriched_sampling_sweep"),
    "LightGBM enriched": Path("outputs/lightgbm_enriched_regularization_sweep"),
    "Random Forest enriched": Path(
        "outputs/random_forest_enriched_period_rmse_sweep/full"
    ),
}


def run_paths(root: Path):
    for metrics in root.glob("**/weekly_*/metrics.csv"):
        horizon = metrics.with_name("horizon_metrics.csv")
        if horizon.exists():
            yield metrics, horizon


def period_mean_rmse(metrics_path: Path) -> float:
    frame = pd.read_csv(metrics_path)
    step = pd.to_numeric(frame["horizon_step"], errors="coerce")
    selected = frame.loc[step.notna() & frame["target"].eq("net_flow")]
    return float(selected["rmse"].mean())


def horizon_rmse(horizon_path: Path) -> float:
    frame = pd.read_csv(horizon_path)
    selected = frame.loc[
        frame["target"].eq("net_flow")
        & frame["horizon_step"].astype(str).eq("horizon_total")
    ]
    return float(selected.iloc[0]["rmse"])


def collect() -> pd.DataFrame:
    rows = []
    for model, root in RUN_ROOTS.items():
        for metrics_path, horizon_path in run_paths(root):
            rows.append(
                {
                    "model": model,
                    "method": metrics_path.parent.name.replace("weekly_", ""),
                    "run_dir": str(metrics_path.parent),
                    "period_mean_rmse": period_mean_rmse(metrics_path),
                    "horizon_rmse": horizon_rmse(horizon_path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    runs = collect()
    period_best = (
        runs.sort_values("period_mean_rmse")
        .groupby("model", as_index=False)
        .first()
        .assign(selection_basis="period_mean_rmse")
    )
    horizon_best = (
        runs.sort_values("horizon_rmse")
        .groupby("model", as_index=False)
        .first()
        .assign(selection_basis="horizon_rmse")
    )
    result = pd.concat([period_best, horizon_best], ignore_index=True)
    result.to_csv(
        "outputs/model_analysis/weekly_netflow_model_comparison_with_rf.csv",
        index=False,
    )
    print("ALL COMPLETED RUNS")
    print(runs.sort_values(["model", "method"]).to_string(index=False))
    print("\nBEST BY PERIOD-MEAN RMSE")
    print(period_best.to_string(index=False))
    print("\nBEST BY HORIZON RMSE")
    print(horizon_best.to_string(index=False))


if __name__ == "__main__":
    main()
