"""Random Forest forecasts using the enriched Berka master tables.

This entry point reuses the leakage-safe enriched tree runner used by the
enriched XGBoost and LightGBM implementations.  It trains separate inflow
and outflow regressors, derives net flow afterward, and supports monthly or
weekly recursive/direct forecasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from enriched_tree_forecast_common import main


def make_rf_pipeline(
    data: Any,
    params: dict[str, float | int],
    n_jobs: int,
    random_state: int,
) -> Pipeline:
    """Build one single-target enriched Random Forest pipeline.

    Random Forest uses tree splits, so numeric scaling is unnecessary.  The
    categorical columns are one-hot encoded to match the other tree models.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", data.numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                data.categorical_columns,
            ),
        ],
        remainder="drop",
    )
    regressor = RandomForestRegressor(
        criterion="squared_error",
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=random_state,
        **params,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def save_feature_importance(
    models: dict[str, Pipeline] | dict[str, list[Pipeline]],
    output_path: Path,
) -> None:
    """Save Random Forest impurity-based feature importance.

    ``feature_importances_`` is mean decrease in impurity (MDI), not gain as
    in XGBoost/LightGBM.  The raw importance values for each model sum to one.
    """
    rows: list[dict[str, object]] = []
    for target, target_models in models.items():
        model_items = (
            enumerate(target_models, start=1)
            if isinstance(target_models, list)
            else [(1, target_models)]
        )
        for horizon_step, model in model_items:
            preprocessor = model.named_steps["preprocess"]
            regressor = model.named_steps["regressor"]
            names = preprocessor.get_feature_names_out()
            importance = np.asarray(regressor.feature_importances_, dtype=float)
            if len(names) != len(importance):
                raise ValueError(
                    "Random Forest feature importance length does not match feature names"
                )
            for feature, value in zip(names, importance):
                rows.append(
                    {
                        "target": target,
                        "horizon_step": horizon_step,
                        "feature": feature,
                        "importance_mdi": float(value),
                    }
                )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["target", "horizon_step", "importance_mdi"],
        ascending=[True, True, False],
    )
    result["rank"] = result.groupby(["target", "horizon_step"]).cumcount() + 1
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    main("random_forest")
