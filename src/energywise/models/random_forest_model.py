"""Random Forest forecaster for the EnergyWise hourly energy series.

This module adds a tree-ensemble forecaster that predicts ``energy_kwh`` from
the *already-engineered* leakage-safe features
(:mod:`energywise.features.time_features`,
:mod:`energywise.features.lag_features`) plus the weather columns:

* ``hour_sin`` / ``hour_cos`` / ``dow_sin`` / ``dow_cos`` -- cyclical calendar
* ``is_weekend`` / ``is_holiday``                          -- calendar flags
* ``energy_lag_1h`` / ``energy_lag_24h`` / ``energy_lag_168h``
* ``energy_roll_24h_mean`` / ``energy_roll_24h_min`` / ``energy_roll_24h_max``
* ``temperature_c`` / ``humidity_pct`` / ``precipitation_mm``

Two entry points matter most:

* :func:`build_feature_matrix` -- assembles the model input matrix and, in one
  pass, drops every row whose **target or any feature** is NaN, logging how
  many rows were dropped per building and in total (a missing hour is missing,
  never fabricated).
* :func:`train_and_forecast_random_forest` -- fits a
  :class:`~sklearn.ensemble.RandomForestRegressor` on the training rows and
  returns predictions for the test rows.  **All imputation and scaling is fit
  inside this function on ``train_df`` only** -- the test set's statistics are
  never used, so there is no leakage.

Backtesting
-----------
:func:`random_forest_forecast_fn` wraps one building's aligned feature frame in
the ``forecast_fn(history, horizon)`` closure the
:func:`energywise.evaluation.backtest.rolling_origin_backtest` harness expects,
exactly like the SARIMAX adapter.  :func:`run_random_forest_backtest` runs that
harness once per building and stitches the folds back together, and
:func:`compare_random_forest_vs_baseline` prints the Random Forest metrics
(``mae`` / ``wape`` / ``peak_mae`` / ``rmse``) next to the seasonal-naive
baseline for a direct, per-building comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energywise.features.lag_features import add_lag_features
from energywise.features.time_features import add_time_features

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "build_feature_matrix",
    "prepare_features",
    "train_and_forecast_random_forest",
    "random_forest_forecast_fn",
    "run_random_forest_backtest",
    "compare_random_forest_vs_baseline",
    "format_rf_comparison",
    "INPUT_PATH",
]

#: Repository root: .../energywise (this file is src/energywise/models/random_forest_model.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default input: the hourly energy + weather table from Phase 2.
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"

#: The target the model learns to predict.
TARGET_COLUMN = "energy_kwh"

#: The model's input columns, in a fixed, documented order.  These are the
#: already-engineered features assembled by :func:`build_feature_matrix`.
FEATURE_COLUMNS: list[str] = [
    # --- cyclical calendar (time_features) ---
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # --- calendar flags (time_features) ---
    "is_weekend",
    "is_holiday",
    # --- lagged energy (lag_features) ---
    "energy_lag_1h",
    "energy_lag_24h",
    "energy_lag_168h",
    # --- trailing rolling energy (lag_features) ---
    "energy_roll_24h_mean",
    "energy_roll_24h_min",
    "energy_roll_24h_max",
    # --- weather ---
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
]

#: Columns the raw hourly frame must provide so the grid can be rebuilt.
GRID_VALUE_COLUMNS = (
    "energy_kwh",
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
)

#: Columns the backtest harness needs in the raw frame.
REQUIRED_BACKTEST_COLUMNS = ("building", "timestamp_ist", TARGET_COLUMN)

#: Default Random Forest hyper-parameters.  These are **reasonable starting
#: values, not tuned**: ``max_features="sqrt"`` and a bootstrap subsample
#: (``max_samples``) keep each tree cheap so a full daily-step rolling-origin
#: backtest (~1,577 refits per building) stays tractable without giving up
#: meaningful accuracy.  Revisit them (and consider a small grid search) before
#: treating any number as final.
DEFAULT_RF_PARAMS: dict = {
    "n_estimators": 30,
    "max_depth": None,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "max_samples": 0.4,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Assemble the model input matrix and drop rows with any missing value.

    The frame must already contain the engineered features
    (:data:`FEATURE_COLUMNS`) and the target (:data:`TARGET_COLUMN`); the
    wrapper :func:`prepare_features` produces such a frame from the raw hourly
    table.

    A row is kept only when ``energy_kwh`` **and every** feature column is
    present.  Dropping is done on the target and features together so a model
    is never trained on a partially missing input vector.

    Parameters
    ----------
    df:
        Frame containing ``building``, :data:`TARGET_COLUMN` and every column
        in :data:`FEATURE_COLUMNS`.
    log:
        When ``True`` (default) log how many rows were dropped, **per building**
        and in total.  Set to ``False`` inside a tight backtest loop to keep the
        log quiet.

    Returns
    -------
    pandas.DataFrame
        A new frame with the columns ``building``, the target and the features,
        containing only the complete rows, with a fresh positional index.
    """
    required = ["building", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Input frame is missing required column(s) for the random forest "
            f"feature matrix: {missing}. Build them with add_time_features / "
            "add_lag_features (or call prepare_features) first."
        )

    subset = df[required].copy()
    complete = subset[TARGET_COLUMN].notna() & subset[FEATURE_COLUMNS].notna().all(axis=1)

    if log:
        total_rows = len(subset)
        total_dropped = int((~complete).sum())
        # Per-building breakdown, so a run makes it obvious *which* building's
        # history is losing hours to missing values.
        for building, group in subset.groupby("building", sort=True):
            n_before = len(group)
            n_after = int(complete.loc[group.index].sum())
            logger.info(
                "build_feature_matrix: building=%s kept=%d dropped=%d of %d "
                "(%.1f%% dropped)",
                building,
                n_after,
                n_before - n_after,
                n_before,
                100.0 * (n_before - n_after) / n_before if n_before else 0.0,
            )
        logger.info(
            "build_feature_matrix: TOTAL kept=%d dropped=%d of %d (%.1f%% dropped)",
            total_rows - total_dropped,
            total_dropped,
            total_rows,
            100.0 * total_dropped / total_rows if total_rows else 0.0,
        )

    return subset.loc[complete].reset_index(drop=True)


def prepare_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw hourly frame into a gap-free, feature-rich frame.

    Each building is reindexed onto its own complete ``1h`` grid (missing hours
    become NaN) *before* the lag/rolling features are computed, so
    ``energy_lag_1h`` really is one hour back and the row positions line up
    exactly with the grid the backtest harness builds internally.  The calendar
    and lag features are then added with the existing leakage-safe modules.

    Parameters
    ----------
    hourly:
        Raw hourly frame with ``building``, ``timestamp_ist`` and
        ``energy_kwh`` (weather columns are carried through when present).

    Returns
    -------
    pandas.DataFrame
        A gap-free frame per building carrying the raw value columns plus every
        column in :data:`FEATURE_COLUMNS`.
    """
    missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in hourly.columns]
    if missing:
        raise ValueError(f"Input frame is missing required column(s): {missing}")

    value_cols = [c for c in GRID_VALUE_COLUMNS if c in hourly.columns]

    pieces: list[pd.DataFrame] = []
    for building, group in hourly.groupby("building", sort=False):
        group = group.sort_values("timestamp_ist", kind="mergesort")
        ts = pd.DatetimeIndex(group["timestamp_ist"])
        if len(ts) == 0:
            continue
        full_index = pd.date_range(ts.min(), ts.max(), freq="1h")
        block = group.set_index("timestamp_ist")[value_cols].reindex(full_index)
        block.index.name = "timestamp_ist"
        block = block.reset_index()
        block["building"] = building
        pieces.append(block)

    grid = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else hourly[[*REQUIRED_BACKTEST_COLUMNS, *value_cols]].copy()
    )

    grid = add_time_features(grid)
    grid = add_lag_features(grid)
    return grid


# ---------------------------------------------------------------------------
# Fit / forecast
# ---------------------------------------------------------------------------


def train_and_forecast_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    **rf_params,
) -> np.ndarray:
    """Fit a Random Forest on ``train_df`` and predict ``test_df``.

    Imputation and scaling are wrapped in a
    :class:`~sklearn.pipeline.Pipeline` that is fit **only on the training
    rows**, so nothing about the test window leaks into the preprocessing.
    Missing feature values in the test rows are filled with the training
    medians.

    Parameters
    ----------
    train_df:
        Training rows with the feature columns and the target.  Already free of
        NaNs in the target/features if produced by :func:`build_feature_matrix`
        (though the imputer makes the function robust either way).
    test_df:
        Rows to predict for.  Only its **features** are read; its target, if
        present, is ignored.
    feature_columns:
        Feature order to use.  Defaults to :data:`FEATURE_COLUMNS`.
    target_column:
        Name of the target column.  Defaults to :data:`TARGET_COLUMN`.
    **rf_params:
        Overrides forwarded to
        :class:`~sklearn.ensemble.RandomForestRegressor` (e.g.
        ``n_estimators``, ``max_depth``, ``random_state``, ``n_jobs``).  Missing
        keys fall back to :data:`DEFAULT_RF_PARAMS`.

    Returns
    -------
    numpy.ndarray
        A float array of length ``len(test_df)``.  When the training set is
        empty (there is nothing to learn from) an all-NaN array is returned
        rather than raising, mirroring the SARIMAX wrapper's robustness.
    """
    n_test = len(test_df)

    if len(train_df) == 0:
        logger.warning("Random Forest got an empty training set; returning NaNs")
        return np.full(n_test, np.nan, dtype=float)

    params = {**DEFAULT_RF_PARAMS, **rf_params}
    feature_columns = list(feature_columns or FEATURE_COLUMNS)

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy(dtype=float)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(**params)),
        ]
    )
    # Fitting (imputer medians, scaler mean/scale, forest) touches train_df only.
    pipeline.fit(X_train, y_train)

    if n_test == 0:
        return np.empty(0, dtype=float)

    X_test = test_df[feature_columns].to_numpy(dtype=float)
    return np.asarray(pipeline.predict(X_test), dtype=float).ravel()


def random_forest_forecast_fn(
    feature_frame: pd.DataFrame,
    *,
    log_every: int = 200,
    **rf_params,
):
    """Return a ``forecast_fn(history, horizon)`` closure for the harness.

    ``feature_frame`` must be **one building's** feature-rich frame, aligned
    row-for-row with the grid the backtest harness builds (use
    :func:`prepare_features` and then filter to the building).  Because the
    closure captures the features, the harness's plain
    ``forecast_fn(history, horizon)`` signature is all that is needed.

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
        predictions = train_and_forecast_random_forest(
            train_matrix, test_block, **rf_params
        )

        progress["folds"] += 1
        if log_every and progress["folds"] % log_every == 0:
            logger.info(
                "random forest fold %d: n_train=%d, horizon=%d",
                progress["folds"],
                len(train_matrix),
                horizon,
            )
        return predictions

    return _forecast


# ---------------------------------------------------------------------------
# Backtest integration (same harness as SARIMAX / seasonal-naive)
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


def run_random_forest_backtest(
    source: pd.DataFrame | str | Path = INPUT_PATH,
    buildings: str | list[str] | tuple[str, ...] | None = None,
    horizon: int = 24,
    initial_train_size: int | None = None,
    step: int = 24,
    season: int = 168,
    **rf_params,
):
    """Run the rolling-origin harness with the Random Forest forecaster.

    The Random Forest needs exogenous features, which the harness's
    ``forecast_fn(history, horizon)`` signature does not carry.  We therefore
    run the *unmodified* harness once per building with a per-building feature
    closure (:func:`random_forest_forecast_fn`) and stitch the folds back into a
    single :class:`~energywise.evaluation.backtest.BacktestResult`.  Everything
    else -- the expanding window, the strict ``train_end < test_start``
    invariant, per-building isolation, pairwise NaN exclusion -- is inherited
    from the harness exactly as for SARIMAX.

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
    **rf_params:
        Forwarded to :func:`train_and_forecast_random_forest`.

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
        forecast_fn = random_forest_forecast_fn(frame, **rf_params)
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
            "random forest backtest: building=%s rows=%d folds=%d",
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


def compare_random_forest_vs_baseline(
    source: pd.DataFrame | str | Path = INPUT_PATH,
    buildings: str | list[str] | tuple[str, ...] | None = None,
    horizon: int = 24,
    initial_train_size: int | None = None,
    step: int = 24,
    season: int = 168,
    **rf_params,
) -> pd.DataFrame:
    """Backtest the Random Forest and the seasonal-naive baseline side by side.

    Both models run through the *same* rolling-origin harness with identical
    horizon / warm-up / step, so the per-building ``mae`` / ``wape`` / ``rmse``
    / ``peak_mae`` numbers are directly comparable.

    Returns
    -------
    pandas.DataFrame
        One row per building (plus ``ALL``) with columns
        ``building``, ``naive_*`` and ``rf_*`` for each of the four metrics.
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
        **rf_params,
    )

    baseline_summary = summarise(baseline_result).set_index("building")
    rf_summary = summarise(rf_result).set_index("building")

    metric_names = ["mae", "wape", "rmse", "peak_mae"]
    rows: list[dict] = []
    for building in baseline_summary.index:
        row: dict = {"building": building}
        for metric in metric_names:
            row[f"naive_{metric}"] = baseline_summary.loc[building, metric]
            row[f"rf_{metric}"] = (
                rf_summary.loc[building, metric] if building in rf_summary.index else np.nan
            )
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            "building",
            "naive_mae", "rf_mae",
            "naive_wape", "rf_wape",
            "naive_rmse", "rf_rmse",
            "naive_peak_mae", "rf_peak_mae",
        ],
    )


def format_rf_comparison(table: pd.DataFrame) -> str:
    """Render the Random Forest vs baseline comparison as aligned plain text."""
    display = table.copy()
    for col in ("naive_mae", "rf_mae", "naive_rmse", "rf_rmse", "naive_peak_mae", "rf_peak_mae"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: f"{v:,.3f}" if pd.notna(v) else "nan"
            )
    for col in ("naive_wape", "rf_wape"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: f"{v * 100:,.2f}%" if pd.notna(v) else "nan"
            )
    return display.to_string(index=False)


def main() -> None:
    """CLI: full daily-step backtest of RF vs the seasonal-naive baseline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    table = compare_random_forest_vs_baseline()
    print("Random Forest vs seasonal-naive baseline (daily-step rolling-origin backtest)")
    print("=" * 92)
    print(format_rf_comparison(table))


if __name__ == "__main__":
    main()