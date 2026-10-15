"""Unit tests for the Random Forest forecaster and its harness integration.

The headline test fits the model on a small synthetic dataset whose target is
an **obvious** linear function of two features and asserts the predictions are
*meaningfully better than random* -- both against a no-skill baseline and by a
clear margin (a high correlation with the truth).

Supporting tests cover the NaN-dropping contract of :func:`build_feature_matrix`
(including the per-building log), the no-leakage guarantee of the in-function
preprocessing, and an end-to-end run through the rolling-origin harness.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from energywise.evaluation.backtest import rolling_origin_backtest, summarise
from energywise.evaluation.metrics import mae
from energywise.models.random_forest_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    prepare_features,
    random_forest_forecast_fn,
    run_random_forest_backtest,
    train_and_forecast_random_forest,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
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


def test_random_forest_predicts_meaningfully_better_than_random():
    """On an obvious relationship the RF must beat a no-skill baseline by a lot."""
    df = _synthetic_matrix(n=800, seed=42)
    split = 600
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    # Use full-capacity trees here: the aim is to confirm the model *can* learn
    # an obvious relationship, so we lift the regularisation the real-data
    # defaults apply (see DEFAULT_RF_PARAMS).
    predictions = train_and_forecast_random_forest(
        train_df,
        test_df,
        n_estimators=200,
        max_features=None,
        max_samples=1.0,
        min_samples_leaf=1,
    )

    assert predictions.shape == (len(test_df),)
    assert np.isfinite(predictions).all()

    actual = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    # (a) Beat a no-skill baseline: predicting the training mean. The target's
    #     noise std is ~1.0, so the RF must be far closer than that.
    train_mean = float(train_df[TARGET_COLUMN].mean())
    rf_mae = mae(actual, predictions)
    random_baseline_mae = mae(actual, np.full_like(actual, train_mean))

    assert rf_mae < random_baseline_mae, (
        f"RF MAE ({rf_mae:.3f}) should beat the no-skill baseline "
        f"({random_baseline_mae:.3f})"
    )

    # (b) "Meaningfully better": the RF should explain the relationship to
    #     within a small multiple of the target's own noise (std ~ 1.0).
    assert rf_mae < 3.0, f"RF MAE too high on an obvious relationship: {rf_mae:.3f}"

    # (c) A high correlation between prediction and truth is a strong,
    #     scale-free confirmation the model learned the signal, not the noise.
    corr = float(np.corrcoef(actual, predictions)[0, 1])
    assert corr > 0.95, f"RF predictions barely correlate with truth: r={corr:.3f}"


def test_random_forest_beats_shuffled_target_baseline():
    """Predictions track the true target, not a shuffled copy of it."""
    df = _synthetic_matrix(n=500, seed=7)
    train_df = df.iloc[:380].reset_index(drop=True)
    test_df = df.iloc[380:].reset_index(drop=True)

    predictions = train_and_forecast_random_forest(
        train_df,
        test_df,
        n_estimators=150,
        max_features=None,
        max_samples=1.0,
        min_samples_leaf=1,
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
# 2. build_feature_matrix drops NaN rows (target or any feature)
# ---------------------------------------------------------------------------


def test_build_feature_matrix_drops_rows_with_missing_target_or_features():
    df = _synthetic_matrix(n=100, seed=1)
    df.loc[0, TARGET_COLUMN] = np.nan                 # missing target
    df.loc[1, FEATURE_COLUMNS[0]] = np.nan            # missing feature
    df.loc[2, FEATURE_COLUMNS[-1]] = np.nan           # another missing feature

    matrix = build_feature_matrix(df, log=False)

    assert len(matrix) == len(df) - 3
    assert matrix[TARGET_COLUMN].notna().all()
    assert matrix[FEATURE_COLUMNS].notna().all(axis=1).all()
    # A fresh positional index is returned.
    assert list(matrix.index) == list(range(len(matrix)))


def test_build_feature_matrix_logs_per_building_and_total(caplog):
    df = _synthetic_matrix(n=60, seed=2)
    df.loc[5, TARGET_COLUMN] = np.nan
    another = _synthetic_matrix(n=40, seed=3)
    another["building"] = "library"
    another.loc[3, FEATURE_COLUMNS[1]] = np.nan
    frame = pd.concat([df, another], ignore_index=True)

    with caplog.at_level(logging.INFO):
        build_feature_matrix(frame, log=True)

    text = "\n".join(record.message for record in caplog.records)
    assert "building=academic" in text
    assert "building=library" in text
    assert "TOTAL" in text
    # 1 academic row + 1 library row dropped out of 100 total.
    assert "dropped=1" in text
    assert "of 100" in text


def test_build_feature_matrix_requires_all_columns():
    df = _synthetic_matrix(n=10, seed=0).drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing required column"):
        build_feature_matrix(df, log=False)


# ---------------------------------------------------------------------------
# 3. No leakage: preprocessing is fit on the training rows only
# ---------------------------------------------------------------------------


def test_preprocessing_uses_train_statistics_only(monkeypatch):
    """Imputer/scaler statistics must come from train_df, never test_df."""
    from energywise.models import random_forest_model as rf

    captured: dict = {}
    real_pipeline = rf.Pipeline

    def spy_pipeline(steps):
        pipe = real_pipeline(steps)
        captured["pipe"] = pipe
        return pipe

    monkeypatch.setattr(rf, "Pipeline", spy_pipeline)

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

    train_and_forecast_random_forest(train_df, test_df, n_estimators=5)

    imputer = captured["pipe"].named_steps["imputer"]
    scaler = captured["pipe"].named_steps["scaler"]
    # Both were fit on the training rows (all zeros), not the 1e6 test rows.
    assert np.allclose(imputer.statistics_, 0.0)
    assert np.allclose(scaler.mean_, 0.0), "scaler fitted on non-train data"


def test_empty_training_set_returns_nan_without_raising(caplog):
    train_df = _synthetic_matrix(n=20, seed=0).iloc[:0]
    test_df = _synthetic_matrix(n=5, seed=1)
    with caplog.at_level("WARNING"):
        preds = train_and_forecast_random_forest(train_df, test_df)
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


def test_random_forest_forecast_fn_matches_harness_signature():
    frame = _hourly_frame("academic")
    prepared = (
        prepare_features(frame).sort_values("timestamp_ist").reset_index(drop=True)
    )
    fn = random_forest_forecast_fn(prepared, n_estimators=20)
    history = frame["energy_kwh"].to_numpy(dtype=float)[: 24 * 8]
    forecast = fn(history, 24)
    assert forecast.shape == (24,)


def test_run_random_forest_backtest_end_to_end():
    frame = pd.concat(
        [_hourly_frame("academic"), _hourly_frame("library")], ignore_index=True
    )
    result = run_random_forest_backtest(
        frame,
        horizon=24,
        initial_train_size=24 * 7,
        step=24,
        n_estimators=30,
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


def test_plain_harness_accepts_random_forest_forecast_fn():
    """The closure plugs straight into ``rolling_origin_backtest`` like SARIMAX."""
    frame = _hourly_frame("academic")
    prepared = (
        prepare_features(frame).sort_values("timestamp_ist").reset_index(drop=True)
    )
    forecast_fn = random_forest_forecast_fn(prepared, n_estimators=30)

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