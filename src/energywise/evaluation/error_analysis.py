"""Error analysis for the GBM model's backtest predictions."""

import pandas as pd

from energywise.evaluation.backtest import rolling_origin_backtest
from energywise.models.gbm_model import (
    build_feature_matrix,
    gbm_forecast_fn,
    prepare_features,
)


STEP = 168  # weekly-step, matching run_gbm_backtest.py


def run_error_analysis() -> pd.DataFrame:
    """Run the GBM model through the weekly-step backtest for all 4
    buildings and return a dataframe of residuals per prediction.
    """
    hourly = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
    hourly = prepare_features(hourly)
    features = build_feature_matrix(hourly)

    result = rolling_origin_backtest(
        hourly,
        gbm_forecast_fn(features),
        step=STEP,
    )

    preds = result.predictions.copy()
    preds["residual"] = preds["actual"] - preds["forecast"]
    return preds