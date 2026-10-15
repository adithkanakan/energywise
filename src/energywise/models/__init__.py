"""EnergyWise — forecasting models: the seasonal-naive baseline, the SARIMAX
forecaster, the Random Forest forecaster and the gradient-boosted tree (GBM)
forecaster."""

from energywise.models.baseline import (
    DEFAULT_HORIZON,
    DEFAULT_SEASON,
    seasonal_naive_forecast,
)
from energywise.models.gbm_model import (
    DEFAULT_GBM_PARAMS,
    compare_models_vs_baseline,
    format_gbm_comparison,
    gbm_forecast_fn,
    run_gbm_backtest,
    train_and_forecast_gbm,
)
from energywise.models.random_forest_model import (
    FEATURE_COLUMNS as RF_FEATURE_COLUMNS,
    build_feature_matrix,
    compare_random_forest_vs_baseline,
    format_rf_comparison,
    prepare_features,
    random_forest_forecast_fn,
    run_random_forest_backtest,
    train_and_forecast_random_forest,
)
from energywise.models.sarimax_model import (
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    fit_and_forecast_sarimax,
    sarimax_forecast_fn,
)

__all__ = [
    "seasonal_naive_forecast",
    "DEFAULT_HORIZON",
    "DEFAULT_SEASON",
    "fit_and_forecast_sarimax",
    "sarimax_forecast_fn",
    "DEFAULT_ORDER",
    "DEFAULT_SEASONAL_ORDER",
    "build_feature_matrix",
    "prepare_features",
    "train_and_forecast_random_forest",
    "random_forest_forecast_fn",
    "run_random_forest_backtest",
    "compare_random_forest_vs_baseline",
    "format_rf_comparison",
    "RF_FEATURE_COLUMNS",
    "train_and_forecast_gbm",
    "gbm_forecast_fn",
    "run_gbm_backtest",
    "compare_models_vs_baseline",
    "format_gbm_comparison",
    "DEFAULT_GBM_PARAMS",
]
