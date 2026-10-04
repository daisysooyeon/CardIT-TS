"""Pruned enriched Random Forest grid search.

The ordinary enriched tree runner evaluates every parameter combination on the
full account population.  That is unnecessarily expensive for Random Forest,
especially for weekly direct forecasting because one model is fitted for each
forecast horizon.  This script uses a two-stage search:

1. evaluate the complete candidate grid on a deterministic account subset;
2. keep the best candidates by period-level net-flow RMSE for each
   frequency/method pair;
3. re-run only those candidates on all accounts and select the final model
   using the same period-level RMSE criterion.

The screening stage is only a pruning device.  All reported final metrics are
computed from the full-account stage.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from enriched_forecast_common import (
    EnrichedForecastData,
    create_enriched_forecast_data,
    get_available_enriched_account_ids,
)
from enriched_tree_forecast_common import (
    RF_GRID_KEYS,
    run_one,
    make_grid_id,
)


DEFAULT_GRID: dict[str, list[float | int]] = {
    "n_estimators": [200, 500],
    "max_depth": [8, 16],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "max_features": [0.7, 1.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pruned enriched Random Forest grid search"
    )
    parser.add_argument(
        "--frequency", choices=["monthly", "weekly", "both"], default="weekly"
    )
    parser.add_argument(
        "--method", choices=["recursive", "direct", "both"], default="both"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "random_forest_enriched_period_rmse_sweep",
    )
    parser.add_argument("--grid-file", type=Path, default=None)
    parser.add_argument(
        "--selected-candidates",
        type=Path,
        default=None,
        help="CSV of explicit grid/frequency/method candidates to evaluate directly",
    )
    parser.add_argument("--screening-max-accounts", type=int, default=100)
    parser.add_argument(
        "--full-max-accounts",
        type=int,
        default=None,
        help="Optional cap for the final phase; omit to use every account",
    )
    parser.add_argument(
        "--top-k-per-version",
        type=int,
        default=2,
        help="Number of screening candidates retained for each frequency/method pair",
    )
    parser.add_argument(
        "--screening-n-estimators",
        type=int,
        default=50,
        help="Temporary tree count used only for fast screening",
    )
    parser.add_argument(
        "--screening-only",
        action="store_true",
        help="Run or reuse screening and stop before full-account fitting",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument("--exclude-lag-rolling", action="store_true")
    return parser.parse_args()


def load_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    if args.grid_file is not None:
        payload = json.loads(args.grid_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Grid JSON must contain an object of parameter lists")
        values = payload
    else:
        values = DEFAULT_GRID

    missing = [key for key in RF_GRID_KEYS if key not in values]
    if missing:
        raise ValueError(f"Grid is missing Random Forest keys: {missing}")
    for key in RF_GRID_KEYS:
        if not isinstance(values[key], list) or not values[key]:
            raise ValueError(f"Grid needs a non-empty list for {key}")
    return [
        dict(zip(RF_GRID_KEYS, combination))
        for combination in itertools.product(*(values[key] for key in RF_GRID_KEYS))
    ]


def load_selected_candidates(
    path: Path,
    frequencies: list[str],
    methods: list[str],
) -> tuple[list[tuple[str, dict[str, float | int]]], set[tuple[str, str, str]]]:
    """Load explicit candidates without expanding a Cartesian grid."""
    selected = pd.read_csv(path)
    required = {"grid_id", "frequency", "method", *RF_GRID_KEYS}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected-candidates CSV is missing columns: {missing}")

    selected = selected.loc[
        selected["frequency"].isin(frequencies)
        & selected["method"].isin(methods)
    ].copy()
    if selected.empty:
        raise ValueError("No selected candidates match frequency/method arguments")

    integer_keys = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
    }
    grids: list[tuple[str, dict[str, float | int]]] = []
    allowed_pairs: set[tuple[str, str, str]] = set()
    seen_grid_ids: set[str] = set()
    for row in selected.itertuples(index=False):
        grid_id = str(row.grid_id)
        params: dict[str, float | int] = {}
        for key in RF_GRID_KEYS:
            value = getattr(row, key)
            params[key] = int(value) if key in integer_keys else float(value)
        if grid_id not in seen_grid_ids:
            grids.append((grid_id, params))
            seen_grid_ids.add(grid_id)
        allowed_pairs.add((grid_id, str(row.frequency), str(row.method)))
    return grids, allowed_pairs


def _metric_rows(
    output_dir: Path,
    grid_id: str,
    frequency: str,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = output_dir / grid_id / f"{frequency}_{method}"
    horizon_path = run_dir / "horizon_metrics.csv"
    metrics_path = run_dir / "metrics.csv"
    if not horizon_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Incomplete cached run: {run_dir}")

    horizon = pd.read_csv(horizon_path)
    metrics = pd.read_csv(metrics_path)
    numeric_step = pd.to_numeric(metrics["horizon_step"], errors="coerce")
    period = metrics.loc[numeric_step.notna()].copy()
    period_mean = (
        period.groupby("target", as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "period_mean_rmse"})
    )
    period_mean["grid_id"] = grid_id
    period_mean["frequency"] = frequency
    period_mean["method"] = method
    params_path = output_dir / grid_id / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        for key, value in params.items():
            period_mean[key] = value
    return horizon, period_mean


def _valid_cache(output_dir: Path, grid_id: str, frequency: str, method: str) -> bool:
    run_dir = output_dir / grid_id / f"{frequency}_{method}"
    return (run_dir / "horizon_metrics.csv").exists() and (
        run_dir / "metrics.csv"
    ).exists()


def _run_phase(
    *,
    phase_name: str,
    phase_dir: Path,
    grids: list[tuple[str, dict[str, float | int]]],
    frequencies: list[str],
    methods: list[str],
    account_ids_by_frequency: dict[str, list[int] | None],
    args: argparse.Namespace,
    allowed_pairs: set[tuple[str, str, str]] | None = None,
    data_by_frequency: dict[str, EnrichedForecastData] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    horizon_rows: list[pd.DataFrame] = []
    period_rows: list[pd.DataFrame] = []
    summary_path = phase_dir / "period_rmse_comparison.csv"
    horizon_path = phase_dir / "horizon_comparison_metrics.csv"
    progress_path = phase_dir / "progress.csv"
    total_pairs = sum(
        1
        for grid_id, _ in grids
        for frequency in frequencies
        for method in methods
        if allowed_pairs is None or (grid_id, frequency, method) in allowed_pairs
    )
    completed_pairs = 0

    for grid_id, params in grids:
        for frequency in frequencies:
            for method in methods:
                if allowed_pairs is not None and (
                    grid_id, frequency, method
                ) not in allowed_pairs:
                    continue
                print(
                    f"[{phase_name}] starting {completed_pairs + 1}/{total_pairs}: "
                    f"{grid_id} {frequency}/{method}",
                    flush=True,
                )
                if _valid_cache(phase_dir, grid_id, frequency, method):
                    print(
                        f"[{phase_name}] using cached {grid_id} "
                        f"{frequency}/{method}",
                        flush=True,
                    )
                else:
                    fit_params = dict(params)
                    if phase_name == "screening":
                        fit_params["n_estimators"] = args.screening_n_estimators
                    run_one(
                        "random_forest",
                        frequency,
                        method,
                        phase_dir,
                        account_ids_by_frequency[frequency],
                        fit_params,
                        grid_id,
                        args,
                        data=(
                            data_by_frequency[frequency]
                            if data_by_frequency is not None
                            else None
                        ),
                    )
                horizon, period = _metric_rows(phase_dir, grid_id, frequency, method)
                for key, value in params.items():
                    period[key] = value
                horizon_rows.append(horizon)
                period_rows.append(period)
                pd.concat(period_rows, ignore_index=True).to_csv(
                    summary_path, index=False
                )
                pd.concat(horizon_rows, ignore_index=True).to_csv(
                    horizon_path, index=False
                )
                completed_pairs += 1
                net_flow_period = period.loc[
                    period["target"].eq("net_flow"), "period_mean_rmse"
                ]
                progress_row = pd.DataFrame(
                    [
                        {
                            "phase": phase_name,
                            "completed": completed_pairs,
                            "total": total_pairs,
                            "grid_id": grid_id,
                            "frequency": frequency,
                            "method": method,
                            "net_flow_period_mean_rmse": (
                                float(net_flow_period.iloc[0])
                                if not net_flow_period.empty
                                else np.nan
                            ),
                        }
                    ]
                )
                progress_mode = "w" if completed_pairs == 1 else "a"
                progress_row.to_csv(
                    progress_path,
                    index=False,
                    mode=progress_mode,
                    header=progress_mode == "w",
                )
                print(
                    f"[{phase_name}] completed {completed_pairs}/{total_pairs}: "
                    f"{grid_id} {frequency}/{method}, "
                    f"net_flow period-mean RMSE="
                    f"{float(net_flow_period.iloc[0]):,.2f}"
                    if not net_flow_period.empty
                    else (
                        f"[{phase_name}] completed {completed_pairs}/{total_pairs}: "
                        f"{grid_id} {frequency}/{method}"
                    ),
                    flush=True,
                )

    return (
        pd.concat(horizon_rows, ignore_index=True),
        pd.concat(period_rows, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    if args.screening_max_accounts < 1:
        raise ValueError("--screening-max-accounts must be positive")
    if args.full_max_accounts is not None and args.full_max_accounts < 1:
        raise ValueError("--full-max-accounts must be positive when provided")
    if args.top_k_per_version < 1:
        raise ValueError("--top-k-per-version must be positive")
    if args.screening_n_estimators < 1:
        raise ValueError("--screening-n-estimators must be positive")

    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    grids = load_grid(args)
    indexed_grids = [
        (make_grid_id("random_forest", index, params), params)
        for index, params in enumerate(grids, start=1)
    ]

    screening_accounts = {
        frequency: get_available_enriched_account_ids(
            frequency, args.screening_max_accounts
        )
        for frequency in frequencies
    }
    full_accounts = {
        frequency: get_available_enriched_account_ids(
            frequency, args.full_max_accounts
        )
        if args.full_max_accounts is not None
        else None
        for frequency in frequencies
    }

    # Keep args compatible with enriched_tree_forecast_common.run_one.
    args.device_type = "cpu"
    screening_dir = args.output_dir / "screening"
    full_dir = args.output_dir / "full"

    if args.selected_candidates is not None:
        selected_grids, allowed_pairs = load_selected_candidates(
            args.selected_candidates, frequencies, methods
        )
        print(
            f"evaluating {len(selected_grids)} explicit grids on full accounts"
        )
        horizon, period = _run_phase(
            phase_name="full",
            phase_dir=full_dir,
            grids=selected_grids,
            frequencies=frequencies,
            methods=methods,
            account_ids_by_frequency=full_accounts,
            args=args,
            allowed_pairs=allowed_pairs,
        )
        period.to_csv(
            args.output_dir / "full_period_rmse_comparison.csv", index=False
        )
        horizon.to_csv(
            args.output_dir / "full_horizon_comparison_metrics.csv", index=False
        )
        best_period = (
            period.sort_values("period_mean_rmse")
            .groupby(["frequency", "method", "target"], as_index=False)
            .first()
        )
        best_horizon = (
            horizon.sort_values("rmse")
            .groupby(["frequency", "method", "target"], as_index=False)
            .first()
        )
        best_period.to_csv(
            args.output_dir / "best_by_period_rmse.csv", index=False
        )
        best_horizon.to_csv(
            args.output_dir / "best_by_horizon_rmse.csv", index=False
        )
        print("saved explicit full-account candidate results")
        print(
            best_period.loc[best_period["target"].eq("net_flow")].to_string(
                index=False
            )
        )
        return

    screening_data = {
        frequency: create_enriched_forecast_data(
            frequency,
            screening_accounts[frequency],
            include_history_features=not args.exclude_lag_rolling,
        )
        for frequency in frequencies
    }

    screening_horizon, screening_period = _run_phase(
        phase_name="screening",
        phase_dir=screening_dir,
        grids=indexed_grids,
        frequencies=frequencies,
        methods=methods,
        account_ids_by_frequency=screening_accounts,
        args=args,
        data_by_frequency=screening_data,
    )
    net_flow = screening_period.loc[screening_period["target"].eq("net_flow")].copy()
    retained_ids: set[str] = set()
    retained_rows: list[pd.DataFrame] = []
    for (frequency, method), group in net_flow.groupby(
        ["frequency", "method"], sort=True
    ):
        # Keep at least one candidate for every n_estimators value.  The
        # fast screening stage may intentionally use a common reduced tree
        # count, so otherwise different n_estimators candidates can tie and
        # the first one in grid order would win arbitrarily.
        diverse = (
            group.sort_values("period_mean_rmse")
            .groupby("n_estimators", group_keys=False)
            .head(1)
        )
        selected = diverse.nsmallest(
            max(args.top_k_per_version, diverse["n_estimators"].nunique()),
            "period_mean_rmse",
        )
        retained_rows.append(selected.assign(selection_frequency=frequency, selection_method=method))
        retained_ids.update(selected["grid_id"].tolist())

    retained = pd.concat(retained_rows, ignore_index=True)
    retained.to_csv(args.output_dir / "screening_selected_candidates.csv", index=False)
    selected_grids = [
        item for item in indexed_grids if item[0] in retained_ids
    ]
    print(
        f"screening retained {len(selected_grids)} unique grids from {len(indexed_grids)}"
    )
    if not selected_grids:
        raise ValueError("Screening did not retain any grid candidates")

    screening_best_period = (
        screening_period.sort_values("period_mean_rmse")
        .groupby(["frequency", "method", "target"], as_index=False)
        .first()
    )
    screening_best_horizon = (
        screening_horizon.sort_values("rmse")
        .groupby(["frequency", "method", "target"], as_index=False)
        .first()
    )
    screening_best_period.to_csv(
        args.output_dir / "screening_best_by_period_rmse.csv", index=False
    )
    screening_best_horizon.to_csv(
        args.output_dir / "screening_best_by_horizon_rmse.csv", index=False
    )
    if args.screening_only:
        print("screening-only requested; skipping full-account fitting")
        print(
            screening_best_period.loc[
                screening_best_period["target"].eq("net_flow")
            ].to_string(index=False)
        )
        return

    horizon, period = _run_phase(
        phase_name="full",
        phase_dir=full_dir,
        grids=selected_grids,
        frequencies=frequencies,
        methods=methods,
        account_ids_by_frequency=full_accounts,
        args=args,
        allowed_pairs={
            (str(row.grid_id), str(row.selection_frequency), str(row.selection_method))
            for row in retained.itertuples()
        },
    )

    period.to_csv(args.output_dir / "full_period_rmse_comparison.csv", index=False)
    horizon.to_csv(args.output_dir / "full_horizon_comparison_metrics.csv", index=False)

    best_period = (
        period.sort_values("period_mean_rmse")
        .groupby(["frequency", "method", "target"], as_index=False)
        .first()
    )
    best_horizon = (
        horizon.sort_values("rmse")
        .groupby(["frequency", "method", "target"], as_index=False)
        .first()
    )
    best_period.to_csv(args.output_dir / "best_by_period_rmse.csv", index=False)
    best_horizon.to_csv(args.output_dir / "best_by_horizon_rmse.csv", index=False)

    print("\nBest full-account models by period mean RMSE:")
    print(
        best_period.loc[best_period["target"].eq("net_flow")].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
