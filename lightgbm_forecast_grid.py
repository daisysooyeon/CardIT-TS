"""Run a selected range of the LightGBM hyperparameter grid.

This runner reuses the implementation in ``lightgbm_forecast.py`` and keeps
the same output layout. Grid numbering is one-based, so the default execution
starts at grid_100 and continues through the last generated combination.

Example (local PowerShell):

    .\ts\Scripts\python.exe lightgbm_forecast_grid.py `
        --start-grid 100 `
        --frequency both `
        --method both `
        --device-type cpu `
        --output-dir outputs/lightgbm_regularization_sampling_sweep `
        --n-jobs -1

Example (Google Colab):

    !python lightgbm_forecast_grid.py \\
        --start-grid 100 \\
        --frequency both \\
        --method both \\
        --device-type cuda \\
        --output-dir outputs/lightgbm_regularization_sampling_sweep \\
        --n-jobs -1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lightgbm_forecast import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAM_GRID,
    EVALUATION_TARGETS,
    GRID_KEYS,
    FREQUENCY_CONFIG,
    get_available_account_ids,
    load_completed_horizon_metrics,
    load_param_grid,
    make_grid_id,
    run_one,
    validate_device_type,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-grid",
        type=int,
        default=100,
        help="One-based first grid number to run. Default: 100.",
    )
    parser.add_argument(
        "--end-grid",
        type=int,
        default=None,
        help="One-based last grid number to run, inclusive. Default: last grid.",
    )
    parser.add_argument(
        "--frequency",
        choices=["monthly", "weekly", "both"],
        default="both",
    )
    parser.add_argument(
        "--method",
        choices=["recursive", "direct", "both"],
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--account-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-accounts", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device-type",
        choices=["cpu", "gpu", "cuda"],
        default="cpu",
        help="LightGBM device; use cuda only with a CUDA-enabled build.",
    )
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument(
        "--exclude-lag-rolling",
        action="store_true",
        help="Use only static features; exclude all lag and rolling features.",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=None,
        help="JSON file containing a list for every requested hyperparameter.",
    )
    parser.add_argument(
        "--max-depths",
        nargs="+",
        type=int,
        default=DEFAULT_PARAM_GRID["max_depth"],
    )
    parser.add_argument(
        "--num-leaves",
        nargs="+",
        type=int,
        default=DEFAULT_PARAM_GRID["num_leaves"],
    )
    parser.add_argument(
        "--n-estimators",
        nargs="+",
        type=int,
        default=DEFAULT_PARAM_GRID["n_estimators"],
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["learning_rate"],
    )
    parser.add_argument(
        "--min-child-samples",
        nargs="+",
        type=int,
        default=DEFAULT_PARAM_GRID["min_child_samples"],
    )
    parser.add_argument(
        "--min-child-weights",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["min_child_weight"],
    )
    parser.add_argument(
        "--subsamples",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["subsample"],
    )
    parser.add_argument(
        "--colsample-bytree-values",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["colsample_bytree"],
    )
    parser.add_argument(
        "--reg-lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["reg_lambda"],
    )
    parser.add_argument(
        "--reg-alphas",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["reg_alpha"],
    )
    parser.add_argument(
        "--min-split-gains",
        nargs="+",
        type=float,
        default=DEFAULT_PARAM_GRID["min_split_gain"],
    )
    return parser


def validate_grid_range(start_grid: int, end_grid: int | None, total: int) -> tuple[int, int]:
    if start_grid < 1:
        raise ValueError("--start-grid must be at least 1")
    resolved_end = total if end_grid is None else end_grid
    if resolved_end < start_grid:
        raise ValueError("--end-grid must be greater than or equal to --start-grid")
    if start_grid > total:
        raise ValueError(f"--start-grid={start_grid} exceeds the {total} generated grids")
    if resolved_end > total:
        raise ValueError(f"--end-grid={resolved_end} exceeds the {total} generated grids")
    return start_grid, resolved_end


def selected_grid_ids(start_grid: int, end_grid: int, param_grid: list[dict[str, float | int]]) -> set[str]:
    return {
        make_grid_id(index, params)
        for index, params in enumerate(param_grid[start_grid - 1 : end_grid], start=start_grid)
    }


def remove_selected_rows(
    frame: pd.DataFrame | None,
    grid_ids: set[str],
    frequencies: list[str],
    methods: list[str],
) -> pd.DataFrame:
    """Keep cached rows outside this runner's selected grid/frequency/method range."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {"grid_id", "frequency", "method"}
    if not required.issubset(frame.columns):
        return frame.copy()
    selected_mask = (
        frame["grid_id"].astype(str).isin(grid_ids)
        & frame["frequency"].astype(str).isin(frequencies)
        & frame["method"].astype(str).isin(methods)
    )
    return frame.loc[~selected_mask].copy()


def calculate_best_by_metric(comparison: pd.DataFrame) -> pd.DataFrame:
    """Recreate the full-grid best-by-metric summary from the comparison rows."""
    horizon_total = comparison.loc[
        comparison["horizon_step"].astype(str).eq("horizon_total")
    ].copy()
    metric_names = [
        "mae",
        "rmse",
        "wape",
        "asymmetric_mae",
        "asymmetric_rmse",
        "nmae_train_mean",
        "nrmse_train_mean",
    ]
    rows: list[dict[str, Any]] = []
    for (frequency, method, target), group in horizon_total.groupby(
        ["frequency", "method", "target"], sort=True
    ):
        for metric in metric_names:
            winner = group.loc[group[metric].idxmin()]
            rows.append(
                {
                    "frequency": frequency,
                    "method": method,
                    "target": target,
                    "metric": metric,
                    "best_value": float(winner[metric]),
                    "grid_id": winner["grid_id"],
                    **{key: winner[key] for key in GRID_KEYS},
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    validate_device_type(args.device_type)

    all_params = load_param_grid(args)
    start_grid, end_grid = validate_grid_range(
        args.start_grid,
        args.end_grid,
        len(all_params),
    )
    params_to_run = all_params[start_grid - 1 : end_grid]
    frequencies = ["monthly", "weekly"] if args.frequency == "both" else [args.frequency]
    methods = ["recursive", "direct"] if args.method == "both" else [args.method]
    grid_ids = selected_grid_ids(start_grid, end_grid, all_params)

    account_ids = args.account_ids
    if account_ids is None and args.max_accounts is not None:
        account_ids_by_frequency = {
            frequency: get_available_account_ids(frequency, args.max_accounts)
            for frequency in frequencies
        }
    else:
        account_ids_by_frequency = {frequency: account_ids for frequency in frequencies}

    expected_account_counts = {
        frequency: (
            len(account_ids_by_frequency[frequency])
            if account_ids_by_frequency[frequency] is not None
            else len(get_available_account_ids(frequency))
        )
        for frequency in frequencies
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "horizon_comparison_metrics.csv"
    existing_comparison: pd.DataFrame | None = None
    if comparison_path.exists():
        try:
            existing_comparison = pd.read_csv(comparison_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            existing_comparison = None

    preserved_rows = remove_selected_rows(
        existing_comparison,
        grid_ids,
        frequencies,
        methods,
    )
    horizon_rows: list[pd.DataFrame] = []
    n_skipped = 0
    n_run = 0

    print(
        f"running grids {start_grid}-{end_grid} of {len(all_params)}, "
        f"frequencies={frequencies}, methods={methods}, resume=True"
    )

    for index, params in enumerate(params_to_run, start=start_grid):
        grid_id = make_grid_id(index, params)
        grid_dir = args.output_dir / grid_id
        grid_dir.mkdir(parents=True, exist_ok=True)
        (grid_dir / "params.json").write_text(
            json.dumps(params, indent=2),
            encoding="utf-8",
        )

        for frequency in frequencies:
            for method in methods:
                run_dir = grid_dir / f"{frequency}_{method}"
                cached = None
                horizon_path = run_dir / "horizon_metrics.csv"
                if horizon_path.exists():
                    try:
                        cached = load_completed_horizon_metrics(
                            pd.read_csv(horizon_path),
                            grid_id,
                            frequency,
                            method,
                            params,
                            expected_account_counts[frequency],
                        )
                    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                        cached = None
                if cached is None:
                    cached = load_completed_horizon_metrics(
                        existing_comparison,
                        grid_id,
                        frequency,
                        method,
                        params,
                        expected_account_counts[frequency],
                    )

                if cached is not None:
                    print(f"skipping completed {grid_id} {frequency}/{method}")
                    horizon_rows.append(cached)
                    n_skipped += 1
                    continue

                metrics = run_one(
                    frequency=frequency,
                    method=method,
                    output_dir=args.output_dir,
                    account_ids=account_ids_by_frequency[frequency],
                    params=params,
                    grid_id=grid_id,
                    under_weight=args.under_weight,
                    over_weight=args.over_weight,
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                    device_type=args.device_type,
                    include_history_features=not args.exclude_lag_rolling,
                )
                horizon_rows.append(metrics)
                n_run += 1
                pd.concat(
                    [preserved_rows, *horizon_rows],
                    ignore_index=True,
                ).to_csv(comparison_path, index=False)

    if horizon_rows:
        comparison = pd.concat(
            [preserved_rows, *horizon_rows],
            ignore_index=True,
        )
    elif not preserved_rows.empty:
        comparison = preserved_rows
    else:
        raise ValueError("No completed grid results were found")

    comparison = comparison.sort_values(
        ["grid_id", "frequency", "method", "target"],
        kind="stable",
    ).reset_index(drop=True)
    comparison.to_csv(comparison_path, index=False)
    calculate_best_by_metric(comparison).to_csv(
        args.output_dir / "best_by_metric.csv",
        index=False,
    )
    print(
        f"saved comparison metrics: {len(comparison):,} rows; "
        f"ran={n_run}, skipped={n_skipped}"
    )


if __name__ == "__main__":
    main()
