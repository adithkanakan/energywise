"""Unit tests for the LightGBM gradient-boosted tree (GBM) forecaster.

Mirrors ``tests/test_random_forest_model.py``.  The headline test fits the model
on a small synthetic dataset whose target is an **obvious** linear function of
two features and asserts the predictions are *meaningfully better than random*
-- both against a no-skill baseline and by a clear margin (a high correlation
with the truth).

Supporting tests cover the shared NaN-dropping contract (via the reused
:func:`~energywise.models.random_forest_model.build_feature_matrix`), the
no-leakage guarantee of the in-function preprocessing, and an end-to-end run
through the rolling-origin harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energywise.evaluation.backtest import rolling_origin_backtest, summarise
from energywise.evaluation.metrics import mae
from energywise.models.gbm_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    gbm_forecast_fn,
    prepare_features,
    run_gbm_backtest,
    train_and_forecast_gbm,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers (identical construction to the Random Forest test)
# ---------------------------------------------------------------------------

#: Two features the synthetic target depends on, with a loud, learnable signal.
_DRIVER_A = "temperature_c"
_DRIVER_B = "energy_lag_1h"


def _synthetic_matrix(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A frame with an obvious feature-target relationship plus small noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "building": "academic",
            _DRIVER_A: rng.normal(25.0, 5.0, size=n),
            _DRIVER_B: rng.uniform(10.0, 50.0, size=n),
        }
    )
    # Every other feature is random noise -- the model must find the two signal
    # columns and mostly ignore the rest.
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = rng.normal(0.0, 1.0, size=n)

    # Clear, strong relationship: energy = 3 * temp - 2 * lag + offset + noise.
    df[TARGET_COLUMN] = (
        3.0 * df[_DRIVER_A] - 2.0 * df[_DRIVER_B] + 100.0 + rng.normal(0.0, 1.0, n)
    )
    return df


# ---------------------------------------------------------------------------
# 1. The headline test: learns an obvious relationship, beats random
# ---------------------------------------------------------------------------


def test_gbm_predicts_meaningfully_better_than_random():
    """On an obvious relationship the GBM must beat a no-skill baseline by a lot."""
    df = _synthetic_matrix(n=800, seed=42)
    split = 600
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    # Use many small, un-regularised trees here: the aim is to confirm the model
    # *can* learn an obvious relationship, so we lift the regularisation the
    # real-data defaults apply (see DEFAULT_GBM_PARAMS).
    predictions = train_and_forecast_gbm(
        train_df,
        test_df,
        n_estimators=400,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=1,
        subsample=1.0,
        colsample_bytree=1.0,
    )

    assert predictions.shape == (len(test_df),)
    assert np.isfinite(predictions).all()

    actual = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    # (a) Beat a no-skill baseline: predicting the training mean. The target's
    #     noise std is ~1.0, so the GBM must be far closer than that.
    train_mean = float(train_df[TARGET_COLUMN].mean())
    gbm_mae = mae(actual, predictions)
    random_baseline_mae = mae(actual, np.full_like(actual, train_mean))

    assert gbm_mae < random_baseline_mae, (
        f"GBM MAE ({gbm_mae:.3f}) should beat the no-skill baseline "
        f"({random_baseline_mae:.3f})"
    )

    # (b) "Meaningfully better": the GBM should explain the relationship to
    #     within a small multiple of the target's own noise (std ~ 1.0).
    assert gbm_mae < 3.0, f"GBM MAE too high on an obvious relationship: {gbm_mae:.3f}"

    # (c) A high correlation between prediction and truth is a strong,
    #     scale-free confirmation the model learned the signal, not the noise.
    corr = float(np.corrcoef(actual, predictions)[0, 1])
    assert corr > 0.95, f"GBM predictions barely correlate with truth: r={corr:.3f}"


def test_gbm_beats_shuffled_target_baseline():
    """Predictions track the true target, not a shuffled copy of it."""
    df = _synthetic_matrix(n=500, seed=7)
    train_df = df.iloc[:380].reset_index(drop=True)
    test_df = df.iloc[380:].reset_index(drop=True)

    predictions = train_and_forecast_gbm(
        train_df,
        test_df,
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=1,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    actual = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    rng = np.random.default_rng(123)
    shuffled = rng.permutation(actual)

    learned_mae = mae(actual, predictions)
    shuffled_mae = mae(actual, shuffled)
    assert learned_mae < shuffled_mae, (
        f"learned MAE ({learned_mae:.3f}) must beat a shuffled-target baseline "
        f"({shuffled_mae:.3f})"
    )


# ---------------------------------------------------------------------------
# 2. Reuses build_feature_matrix -- NaN-dropping contract is inherited
# ---------------------------------------------------------------------------


def test_gbm_reuses_build_feature_matrix_and_drops_nan_rows():
    df = _synthetic_matrix(n=100, seed=1)
    df.loc[0, TARGET_COLUMN] = np.nan                 # missing target
    df.loc[1, FEATURE_COLUMNS[0]] = np.nan            # missing feature
    df.loc[2, FEATURE_COLUMNS[-1]] = np.nan           # another missing feature

    # The GBM module re-exports the very same build_feature_matrix, so the
    # feature set and NaN-dropping logic are guaranteed identical to the RF.
    from energywise.models import gbm_model, random_forest_model

    assert gbm_model.build_feature_matrix is random_forest_model.build_feature_matrix
    assert gbm_model.FEATURE_COLUMNS == random_forest_model.FEATURE_COLUMNS

    matrix = build_feature_matrix(df, log=False)
    assert len(matrix) == len(df) - 3
    assert matrix[TARGET_COLUMN].notna().all()
    assert matrix[FEATURE_COLUMNS].notna().all(axis=1).all()


def test_prepare_features_is_shared_with_random_forest():
    from energywise.models import gbm_model, random_forest_model

    assert gbm_model.prepare_features is random_forest_model.prepare_features


# ---------------------------------------------------------------------------
# 3. No leakage: preprocessing is fit on the training rows only
# ---------------------------------------------------------------------------


def test_gbm_preprocessing_uses_train_statistics_only(monkeypatch):
    """The imputer statistics must come from train_df, never test_df."""
    from energywise.models import gbm_model as gbm

    captured: dict = {}
    real_pipeline = gbm.Pipeline

    def spy_pipeline(steps):
        pipe = real_pipeline(steps)
        captured["pipe"] = pipe
        return pipe

    monkeypatch.setattr(gbm, "Pipeline", spy_pipeline)

    # Train has feature value 0.0 in every row; test has a wildly different 1e6.
    train_df = pd.DataFrame(
        {
            "building": "academic",
            **{col: [0.0, 0.0, 0.0] for col in FEATURE_COLUMNS},
            TARGET_COLUMN: [10.0, 12.0, 14.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "building": "academic",
            **{col: [1e6] * 2 for col in FEATURE_COLUMNS},
        }
    )

    train_and_forecast_gbm(train_df, test_df, n_estimators=5)

    imputer = captured["pipe"].named_steps["imputer"]
    # Fit on the training rows (all zeros), not the 1e6 test rows.
    assert np.allclose(imputer.statistics_, 0.0), "imputer fitted on non-train data"


def test_gbm_empty_training_set_returns_nan_without_raising(caplog):
    train_df = _synthetic_matrix(n=20, seed=0).iloc[:0]
    test_df = _synthetic_matrix(n=5, seed=1)
    with caplog.at_level("WARNING"):
        preds = train_and_forecast_gbm(train_df, test_df)
    assert preds.shape == (5,)
    assert np.isnan(preds).all()


# ---------------------------------------------------------------------------
# 4. End-to-end harness integration
# ---------------------------------------------------------------------------


def _hourly_frame(
    building: str, n: int = 24 * 14, start: str = "2024-01-01 00:00:00"
) -> pd.DataFrame:
    """An hourly frame with a learnable daily cycle and weather columns."""
    ts = pd.date_range(pd.Timestamp(start, tz="Asia/Kolkata"), periods=n, freq="1h")
    t = np.arange(n, dtype=float)
    energy = 100.0 + 20.0 * np.sin(2.0 * np.pi * t / 24.0)
    return pd.DataFrame(
        {
            "building": building,
            "timestamp_ist": ts,
            "energy_kwh": energy,
            "temperature_c": 25.0 + 3.0 * np.cos(2.0 * np.pi * t / 24.0),
            "humidity_pct": 60.0,
            "precipitation_mm": 0.0,
        }
    )


def test_gbm_forecast_fn_matches_harness_signature():
    frame = _hourly_frame("academic")
    prepared = (
        prepare_features(frame).sort_values("timestamp_ist").reset_index(drop=True)
    )
    fn = gbm_forecast_fn(prepared, n_estimators=20)
    history = frame["energy_kwh"].to_numpy(dtype=float)[: 24 * 8]
    forecast = fn(history, 24)
    assert forecast.shape == (24,)


def test_run_gbm_backtest_end_to_end():
    frame = pd.concat(
        [_hourly_frame("academic"), _hourly_frame("library")], ignore_index=True
    )
    result = run_gbm_backtest(
        frame,
        horizon=24,
        initial_train_size=24 * 7,
        step=24,
        n_estimators=40,
    )

    assert set(result.predictions["building"].unique()) == {"academic", "library"}
    # No test hour may fall inside its fold's training window.
    scored = result.folds[result.folds["scored"]]
    assert len(scored) > 0
    assert (scored["train_end"] < scored["test_start"]).all()

    summary = summarise(result)
    academic = summary[summary["building"] == "academic"]
    assert len(academic) == 1
    # A clean daily cycle should be forecast well.
    assert np.isfinite(academic["mae"].iloc[0])
    assert academic["mae"].iloc[0] < 15.0


def test_plain_harness_accepts_gbm_forecast_fn():
    """The closure plugs straight into ``rolling_origin_backtest`` like SARIMAX."""
    frame = _hourly_frame("academic")
    prepared = (
        prepare_features(frame).sort_values("timestamp_ist").reset_index(drop=True)
    )
    forecast_fn = gbm_forecast_fn(prepared, n_estimators=40)

    result = rolling_origin_backtest(
        frame,
        forecast_fn=forecast_fn,
        horizon=24,
        initial_train_size=24 * 7,
        step=24,
        buildings="academic",
    )
    assert len(result.predictions) > 0
    assert result.predictions["forecast"].notna().any()