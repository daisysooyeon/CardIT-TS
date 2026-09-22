"""Analyze the selected LR and ARIMA models for interpretation.

The selected LR model is monthly direct.  Because net flow is derived as
inflow minus outflow, this script subtracts the two LR coefficient tables to
produce raw net-flow coefficients.

The selected ARIMA model is monthly direct ARIMA(0, 0, 0).  Plain ARIMA has
no external feature columns, so this script reports fitted constants,
innovation variance (sigma2), and AIC instead of feature importance.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_LR_DIR = ROOT / "outputs" / "linear_regression" / "monthly_direct"
DEFAULT_ARIMA_DIR = (
    ROOT
    / "outputs"
    / "arima_order_grid"
    / "order_p0_d0_q0"
    / "monthly_direct"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "model_analysis"


def coefficient_columns(columns: list[str]) -> list[str]:
    """Return coefficient_hN columns in numerical horizon order."""
    pattern = re.compile(r"^coefficient_h(\d+)$")
    return sorted(
        (column for column in columns if pattern.match(column)),
        key=lambda column: int(pattern.match(column).group(1)),
    )


def analyze_lr(lr_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create raw net-flow coefficients for LR monthly direct."""
    inflow = pd.read_csv(lr_dir / "inflow_coefficients.csv")
    outflow = pd.read_csv(lr_dir / "outflow_coefficients.csv")
    if set(inflow["feature"]) != set(outflow["feature"]):
        raise ValueError("LR inflow/outflow coefficient features do not align")

    horizons = coefficient_columns(inflow.columns.tolist())
    outflow_horizons = coefficient_columns(outflow.columns.tolist())
    if not horizons or horizons != outflow_horizons:
        raise ValueError("LR coefficient horizon columns do not align")

    aligned = inflow[["feature", *horizons]].merge(
        outflow[["feature", *outflow_horizons]],
        on="feature",
        how="inner",
        suffixes=("_inflow", "_outflow"),
        validate="one_to_one",
    )
    result = aligned[["feature"]].copy()
    for column in horizons:
        result[f"net_{column}"] = (
            aligned[f"{column}_inflow"].astype(float)
            - aligned[f"{column}_outflow"].astype(float)
        )
    net_columns = [f"net_{column}" for column in horizons]
    result["mean_abs_net_coefficient"] = result[net_columns].abs().mean(axis=1)
    result["mean_net_coefficient"] = result[net_columns].mean(axis=1)
    result = result.sort_values(
        "mean_abs_net_coefficient", ascending=False
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.to_csv(
        output_dir / "lr_monthly_direct_net_coefficients.csv", index=False
    )
    return result


def analyze_arima(arima_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create account-level parameter and fitted-level summaries for ARIMA."""
    artifacts = joblib.load(arima_dir / "model.joblib")
    rows: list[dict[str, object]] = []
    for (account_id, target), artifact in sorted(artifacts.items()):
        params = np.asarray(artifact["params"], dtype=float)
        order = tuple(int(value) for value in artifact["order"])
        row: dict[str, object] = {
            "account_id": int(account_id),
            "target": target,
            "order": str(order),
            "aic": float(artifact["aic"]),
            "n_params": int(len(params)),
        }
        for index, value in enumerate(params):
            row[f"param_{index}"] = float(value)
        if order == (0, 0, 0) and len(params) >= 2:
            row["const"] = float(params[0])
            row["sigma2"] = float(params[1])
        rows.append(row)

    parameters = pd.DataFrame(rows).sort_values(["account_id", "target"])
    parameters.to_csv(
        output_dir / "arima_monthly_direct_p000_parameters.csv", index=False
    )

    summary = (
        parameters.groupby("target", as_index=False)
        .agg(
            n_fitted=("account_id", "size"),
            const_mean=("const", "mean"),
            const_median=("const", "median"),
            const_std=("const", "std"),
            sigma2_mean=("sigma2", "mean"),
            aic_mean=("aic", "mean"),
            aic_median=("aic", "median"),
        )
    )
    summary.to_csv(
        output_dir / "arima_monthly_direct_p000_parameter_summary.csv",
        index=False,
    )

    levels = parameters.pivot(index="account_id", columns="target", values="const")
    levels = levels.rename(
        columns={"inflow": "inflow_const", "outflow": "outflow_const"}
    ).reset_index()
    if {"inflow_const", "outflow_const"}.issubset(levels.columns):
        levels["net_const"] = levels["inflow_const"] - levels["outflow_const"]
    levels.to_csv(
        output_dir / "arima_monthly_direct_p000_fitted_levels.csv", index=False
    )
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-dir", type=Path, default=DEFAULT_LR_DIR)
    parser.add_argument("--arima-dir", type=Path, default=DEFAULT_ARIMA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lr = analyze_lr(args.lr_dir, args.output_dir)
    arima = analyze_arima(args.arima_dir, args.output_dir)
    print(
        f"saved LR net coefficients: {len(lr):,} features; "
        f"ARIMA fitted artifacts: {len(arima):,} account-flow rows"
    )


if __name__ == "__main__":
    main()
