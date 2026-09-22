"""LightGBM forecasts using the enriched Berka master tables.

Use ``--device-type cuda`` only with a CUDA-enabled LightGBM build.  The
features and four monthly/weekly recursive/direct configurations are shared
with the enriched XGBoost runner.
"""

from enriched_tree_forecast_common import main


if __name__ == "__main__":
    main("lightgbm")
