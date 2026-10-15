"""Gradient-boosted tree forecaster for the EnergyWise hourly energy series.

This module adds a LightGBM gradient-boosted tree forecaster that predicts
``energy_kwh`` from the *exact same* leakage-safe feature set as the Random
Forest (:mod:`energywise.models.random_forest_model`).  Feature assembly and
NaN-dropping are delegated to
:func:`~energywise.models.random_forest_model.build_feature_matrix` so the
two tree models are guaranteed to see identical inputs -- any comparison
between them is a comparison of the *learner*, not of the features.

Two entry points matter most:

* :func:`train_and_forecast_gbm` -- fits a
  :class:`lightgbm.LGBMRegressor` on the training rows and returns predictions
  for the test rows.  **All imputation is fit inside this function on
  ``train_df`` only** -- the test set's statistics are never used, so there is
  no leakage.  (GBMs are invariant to monotone feature scaling, so no scaler is
  applied; the median imputer is kept so the function is robust to the rare
  missing value that slips past :func:`build_feature_matrix`.)
* :func:`gbm_forecast_fn` -- wraps one building's aligned feature frame in the
  ``forecast_fn(history, horizon)`` closure the
  :func:`energywise.evaluation.backtest.rolling_origin_backtest` harness expects,
  exactly like the Random Forest and SARIMAX adapters.

Backtesting
-----------
:func:`run_gbm_backtest` runs that harness once per building and stitches the
folds back together, and :func:`compare_models_vs_baseline` prints the naive,
Random Forest and GBM metrics (``mae`` / ``wape`` / ``rmse`` / ``peak_mae``)
side by side for a direct, per-building three-way comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from energywise.models.random_forest_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    prepare_features,
    run_random_forest_backtest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "DEFAULT_GBM_PARAMS",
    "train_and_forecast_gbm",
    "gbm_forecast_fn",
    "run_gbm_backtest",
    "compare_models_vs_baseline",
    "format_gbm_comparison",
    "build_feature_matrix",
    "prepare_features",
    "INPUT_PATH",
]

#: Repository root: .../energywise (this file is src/energywise/models/gbm_model.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default input: the hourly energy + weather table from Phase 2.
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"

#: Default LightGBM hyper-parameters.  These are **reasonable starting values,
#: not tuned**: a modest number of small trees with LightGBM's histogram splits
#: keeps a full daily-step rolling-origin backtest (~1,577 refits per building)
#: tractable.  ``verbose=-1`` silences LightGBM's per-fit chatter so the
#: backtest log stays readable.  Revisit them (and consider a small grid search)
#: before treating any number as final.
DEFAULT_GBM_PARAMS: dict = {
    "n_estimators": 60,
    "learning_rate": 0.1,
    "max_depth": -1,
    "num_leaves": 15,
    "min_child_samples": 3,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Fit / forecast
# ---------------------------------------------------------------------------


def train_and_forecast_gbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    **gbm_params,
) -> np.ndarray:
    """Fit a LightGBM gradient-boosted model on ``train_df`` and predict ``test_df``.

    Imputation is wrapped in a :class:`~sklearn.pipeline.Pipeline` that is fit
    **only on the training rows**, so nothing about the test window leaks into
    the preprocessing.  Gradient-boosted trees are invariant to monotone feature
    scaling, so (unlike the Random Forest wrapper) no scaler is applied; missing
    feature values in the test rows are filled with the training medians.

    Parameters
    ----------
    train_df:
        Training rows with the feature columns and the target.  Already free of
        NaNs in the target/features if produced by
        :func:`~energywise.models.random_forest_model.build_feature_matrix`
        (though the imputer makes the function robust either way).
    test_df:
        Rows to predict for.  Only its **features** are read; its target, if
        present, is ignored.
    feature_columns:
        Feature order to use.  Defaults to
        :data:`~energywise.models.random_forest_model.FEATURE_COLUMNS`, i.e. the
        exact same feature set the Random Forest uses.
    target_column:
        Name of the target column.  Defaults to
        :data:`~energywise.models.random_forest_model.TARGET_COLUMN`.
    **gbm_params:
        Overrides forwarded to :class:`lightgbm.LGBMRegressor` (e.g.
        ``n_estimators``, ``learning_rate``, ``num_leaves``, ``random_state``,
        ``n_jobs``).  Missing keys fall back to :data:`DEFAULT_GBM_PARAMS`.

    Returns
    -------
    numpy.ndarray
        A float array of length ``len(test_df)``.  When the training set is
        empty (there is nothing to learn from) an all-NaN array is returned
        rather than raising, mirroring the Random Forest and SARIMAX wrappers'
        robustness.
    """
    n_test = len(test_df)

    if len(train_df) == 0:
        logger.warning("GBM got an empty training set; returning NaNs")
        return np.full(n_test, np.nan, dtype=float)

    params = {**DEFAULT_GBM_PARAMS, **gbm_params}
    feature_columns = list(feature_columns or FEATURE_COLUMNS)

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy(dtype=float)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(**params)),
        ]
    )
    # Fitting (imputer medians, boosted trees) touches train_df only.
    pipeline.fit(X_train, y_train)

    if n_test == 0:
        return np.empty(0, dtype=float)

    X_test = test_df[feature_columns].to_numpy(dtype=float)
    return np.asarray(pipeline.predict(X_test), dtype=float).ravel()


def gbm_forecast_fn(
    feature_frame: pd.DataFrame,
    *,
    log_every: int = 200,
    **gbm_params,
):
    """Return a ``forecast_fn(history, horizon)`` closure for the harness.

    ``feature_frame`` must be **one building's** feature-rich frame, aligned
    row-for-row with the grid the backtest harness builds (use
    :func:`~energywise.models.random_forest_model.prepare_features` and then
    filter to the building).  Because the closure captures the features, the
    harness's plain ``forecast_fn(history, horizon)`` signature is all that is
    needed.

    ``history`` is the building's values strictly before the fold's cutoff; its
    length *is* the cutoff position, so the closure slices the training window
    as ``frame[:cutoff]`` and the test window as ``frame[cutoff:cutoff+horizon]``
    -- guaranteeing no test row is ever used to fit the model.
    """
    frame = feature_frame.reset_index(drop=True)
    progress = {"folds": 0}

    def _forecast(history: np.ndarray, horizon: int) -> np.ndarray:
        cutoff = int(np.asarray(history).shape[0])

        train_block = frame.iloc[:cutoff].copy()
        # Use exactly the history the harness scored against, so the training
        # target can never drift from the fold's own values.
        train_block[TARGET_COLUMN] = np.asarray(history, dtype=float)

        test_block = frame.iloc[cutoff : cutoff + horizon].copy()
        if len(test_block) < horizon:  # defensive: pad short tails with NaN rows
            pad = pd.DataFrame(
                np.nan,
                index=range(horizon - len(test_block)),
                columns=test_block.columns,
            )
            test_block = pd.concat([test_block, pad], ignore_index=True)

        train_matrix = build_feature_matrix(train_block, log=False)
        predictions = train_and_forecast_gbm(train_matrix, test_block, **gbm_params)

        progress["folds"] += 1
        if log_every and progress["folds"] % log_every == 0:
            logger.info(
                "gbm fold %d: n_train=%d, horizon=%d",
                progress["folds"],
                len(train_matrix),
                horizon,
            )
        return predictions

    return _forecast


# ---------------------------------------------------------------------------
# Backtest integration (same harness as SARIMAX / Random Forest / seasonal-naive)
# ---------------------------------------------------------------------------


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "building", "timestamp_ist", "actual", "forecast", "fold", "is_peak",
        ]
    )


def _empty_folds() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "building", "fold", "train_start", "train_end", "test_start",
            "test_end", "n_train", "n_test", "scored", "skipped_reason",
        ]
    )


def run_gbm_backtest(
    source: pd.DataFrame | str | Path = INPUT_PATH,
    buildings: str | list[str] | tuple[str, ...] | None = None,
    horizon: int = 24,
    initial_train_size: int | None = None,
    step: int = 24,
    season: int = 168,
    **gbm_params,
):
    """Run the rolling-origin harness with the LightGBM forecaster.

    The GBM needs exogenous features, which the harness's
    ``forecast_fn(history, horizon)`` signature does not carry.  We therefore
    run the *unmodified* harness once per building with a per-building feature
    closure (:func:`gbm_forecast_fn`) and stitch the folds back into a single
    :class:`~energywise.evaluation.backtest.BacktestResult`.  Everything else --
    the expanding window, the strict ``train_end < test_start`` invariant,
    per-building isolation, pairwise NaN exclusion -- is inherited from the
    harness exactly as for the Random Forest.

    Parameters
    ----------
    source:
        Either the raw hourly frame or a path to the Parquet file.
    buildings:
        Building filter.  ``None`` (default) backtests every building present.
    horizon:
        Forecast length per fold, in hours (default 24 == daily step).
    initial_train_size:
        Warm-up history for the first fold, in hours.  Defaults to the harness
        default (four weeks).
    step:
        Origin advance between folds, in hours (default 24 == daily step).
    season:
        Seasonal lag for the bundled baseline (recorded on the result).
    **gbm_params:
        Forwarded to :func:`train_and_forecast_gbm`.

    Returns
    -------
    BacktestResult
        A single result covering every requested building.
    """
    from energywise.evaluation.backtest import (
        DEFAULT_INITIAL_TRAIN_SIZE,
        BacktestResult,
        rolling_origin_backtest,
    )

    if initial_train_size is None:
        initial_train_size = DEFAULT_INITIAL_TRAIN_SIZE

    hourly = source if isinstance(source, pd.DataFrame) else pd.read_parquet(Path(source))

    if buildings is None:
        selected = sorted(hourly["building"].unique())
    elif isinstance(buildings, str):
        selected = [buildings]
    else:
        selected = list(buildings)

    prepared = prepare_features(hourly)

    preds_list: list[pd.DataFrame] = []
    folds_list: list[pd.DataFrame] = []
    candidate_hours = skipped_hours = excluded_nan_hours = 0

    for building in selected:
        frame = (
            prepared[prepared["building"] == building]
            .sort_values("timestamp_ist", kind="mergesort")
            .reset_index(drop=True)
        )
        forecast_fn = gbm_forecast_fn(frame, **gbm_params)
        result = rolling_origin_backtest(
            hourly,
            forecast_fn=forecast_fn,
            horizon=horizon,
            initial_train_size=initial_train_size,
            step=step,
            season=season,
            buildings=building,
        )
        preds_list.append(result.predictions)
        folds_list.append(result.folds)
        candidate_hours += result.candidate_hours
        skipped_hours += result.skipped_hours
        excluded_nan_hours += result.excluded_nan_hours
        logger.info(
            "gbm backtest: building=%s rows=%d folds=%d",
            building,
            len(result.predictions),
            len(result.folds),
        )

    predictions = (
        pd.concat(preds_list, ignore_index=True) if preds_list else _empty_predictions()
    )
    folds = pd.concat(folds_list, ignore_index=True) if folds_list else _empty_folds()

    return BacktestResult(
        predictions=predictions,
        folds=folds,
        horizon=horizon,
        season=season,
        initial_train_size=initial_train_size,
        step=step,
        candidate_hours=candidate_hours,
        skipped_hours=skipped_hours,
        excluded_nan_hours=excluded_nan_hours,
    )


def compare_models_vs_baseline(
    source: pd.DataFrame | str | Path = INPUT_PATH,
    buildings: str | list[str] | tuple[str, ...] | None = None,
    horizon: int = 24,
    initial_train_size: int | None = None,
    step: int = 24,
    season: int = 168,
    **gbm_params,
) -> pd.DataFrame:
    """Backtest naive, Random Forest and GBM side by side through one harness.

    All three models run through the *same* rolling-origin harness with
    identical horizon / warm-up / step, so the per-building ``mae`` / ``wape`` /
    ``rmse`` / ``peak_mae`` numbers are directly comparable.  The Random Forest
    uses its own module defaults; the GBM uses :data:`DEFAULT_GBM_PARAMS` plus
    any ``**gbm_params`` overrides.

    Returns
    -------
    pandas.DataFrame
        One row per building (plus ``ALL``) with columns ``building`` and, for
        each of the four metrics, ``naive_*`` / ``rf_*`` / ``gbm_*``.
    """
    from energywise.evaluation.backtest import (
        DEFAULT_INITIAL_TRAIN_SIZE,
        rolling_origin_backtest,
        summarise,
    )

    if initial_train_size is None:
        initial_train_size = DEFAULT_INITIAL_TRAIN_SIZE

    hourly = source if isinstance(source, pd.DataFrame) else pd.read_parquet(Path(source))

    baseline_result = rolling_origin_backtest(
        hourly,
        horizon=horizon,
        initial_train_size=initial_train_size,
        step=step,
        season=season,
        buildings=buildings,
    )
    rf_result = run_random_forest_backtest(
        source,
        buildings=buildings,
        horizon=horizon,
        initial_train_size=initial_train_size,
        step=step,
        season=season,
    )
    gbm_result = run_gbm_backtest(
        source,
        buildings=buildings,
        horizon=horizon,
        initial_train_size=initial_train_size,
        step=step,
        season=season,
        **gbm_params,
    )

    baseline_summary = summarise(baseline_result).set_index("building")
    rf_summary = summarise(rf_result).set_index("building")
    gbm_summary = summarise(gbm_result).set_index("building")

    metric_names = ["mae", "wape", "rmse", "peak_mae"]
    rows: list[dict] = []
    for building in baseline_summary.index:
        row: dict = {"building": building}
        for metric in metric_names:
            row[f"naive_{metric}"] = baseline_summary.loc[building, metric]
            row[f"rf_{metric}"] = (
                rf_summary.loc[building, metric] if building in rf_summary.index else np.nan
            )
            row[f"gbm_{metric}"] = (
                gbm_summary.loc[building, metric] if building in gbm_summary.index else np.nan
            )
        rows.append(row)

    columns = ["building"]
    for metric in metric_names:
        columns += [f"naive_{metric}", f"rf_{metric}", f"gbm_{metric}"]
    return pd.DataFrame(rows, columns=columns)


def format_gbm_comparison(table: pd.DataFrame) -> str:
    """Render the naive / Random Forest / GBM comparison as aligned plain text."""
    display = table.copy()
    value_cols = [
        col
        for col in display.columns
        if col.endswith(("_mae", "_rmse")) and col != "building"
    ]
    for col in value_cols:
        display[col] = display[col].map(
            lambda v: f"{v:,.3f}" if pd.notna(v) else "nan"
        )
    for col in (c for c in display.columns if c.endswith("_wape")):
        display[col] = display[col].map(
            lambda v: f"{v * 100:,.2f}%" if pd.notna(v) else "nan"
        )
    return display.to_string(index=False)


def main() -> None:
    """CLI: full daily-step backtest of naive vs RF vs GBM."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    table = compare_models_vs_baseline()
    print("Naive vs Random Forest vs GBM (daily-step rolling-origin backtest)")
    print("=" * 92)
    print(format_gbm_comparison(table))


if __name__ == "__main__":
    main()