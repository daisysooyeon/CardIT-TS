from pathlib import Path

import pandas as pd


ROOTS = {
    "non_enriched": Path("outputs/lightgbm_regularization_sampling_sweep"),
    "enriched": Path("outputs/lightgbm_enriched_regularization_sweep"),
}
TARGETS = ["inflow", "outflow", "net_flow"]


def collect_period(root: Path, source: str) -> list[dict[str, object]]:
    rows = []
    for path in root.glob("grid_*/weekly_*/metrics.csv"):
        frame = pd.read_csv(path)
        step = pd.to_numeric(frame["horizon_step"], errors="coerce")
        frame = frame.loc[step.notna()]
        for target in TARGETS:
            rows.append(
                {
                    "source": source,
                    "grid_id": path.parent.parent.name,
                    "method": path.parent.name.replace("weekly_", ""),
                    "target": target,
                    "period_mean_rmse": frame.loc[frame["target"].eq(target), "rmse"].mean(),
                }
            )
    return rows


def collect_horizon(root: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "horizon_comparison_metrics.csv")
    return frame.loc[
        frame["frequency"].eq("weekly") & frame["target"].isin(TARGETS),
        ["grid_id", "method", "target", "rmse"],
    ].assign(source=source)


def main() -> None:
    period = pd.DataFrame(
        [row for source, root in ROOTS.items() for row in collect_period(root, source)]
    )
    horizon = pd.concat(
        [collect_horizon(root, source) for source, root in ROOTS.items()],
        ignore_index=True,
    )
    period_best = (
        period.sort_values("period_mean_rmse")
        .groupby(["source", "method", "target"], as_index=False)
        .first()
    )
    horizon_best = (
        horizon.sort_values("rmse")
        .groupby(["source", "method", "target"], as_index=False)
        .first()
    )
    result = period_best.merge(
        horizon_best,
        on=["source", "method", "target"],
        how="outer",
        suffixes=("_period", "_horizon"),
    )
    result.to_csv(
        "outputs/model_analysis/lightgbm_enriched_vs_non_enriched_all_targets.csv",
        index=False,
    )
    print("PERIOD-MEAN RMSE BEST")
    print(period_best.to_string(index=False))
    print("\nHORIZON-TOTAL RMSE BEST")
    print(horizon_best.to_string(index=False))

    for name, frame, metric in [
        ("period_mean_rmse", period_best, "period_mean_rmse"),
        ("horizon_rmse", horizon_best, "rmse"),
    ]:
        pivot = frame.pivot_table(
            index=["method", "target"],
            columns="source",
            values=metric,
            aggfunc="first",
        ).reset_index()
        pivot["enriched_minus_non_enriched"] = (
            pivot["enriched"] - pivot["non_enriched"]
        )
        pivot["change_pct"] = (
            pivot["enriched_minus_non_enriched"] / pivot["non_enriched"] * 100
        )
        print(f"\n{name}: enriched minus non-enriched")
        print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
