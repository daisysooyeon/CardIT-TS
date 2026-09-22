"""XGBoost forecasts using the enriched Berka master tables.

This entry point preserves the original XGBoost grid/forecast conventions but
adds the leakage-safe holiday, calendar, month-boundary, and scheduled-loan
features from ``*_enriched.parquet``.
"""

from enriched_tree_forecast_common import main


if __name__ == "__main__":
    main("xgboost")
