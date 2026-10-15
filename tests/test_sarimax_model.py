"""Unit tests for the SARIMAX forecaster and its integration with the harness.

Two behaviours matter most:

1. **It learns the daily cycle.**  On a clean synthetic series with a strong
   24-hour pattern (plus a slow trend), a SARIMAX forecast must be *closer* to
   the true continuation than a naive "repeat the last value" forecast.
2. **A failed fit never crashes the run.**  When the underlying ``SARIMAX``
   constructor/fit raises (the messy-data case, e.g. non-convergence), the
   wrapper logs a warning and returns an **all-NaN** forecast instead of
   propagating the exception.

Supporting tests cover optional exogenous regressors and the harness's building
filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energywise.evaluation.backtest import rolling_origin_backtest
from energywise.evaluation.metrics import mae
from energywise.models import sarimax_model
from energywise.models.sarimax_model import fit_and_forecast_sarimax

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

DAILY_PERIOD = 24


def _synthetic_seasonal_series(days: int = 20, seed: int = 0) -> np.ndarray:
    """A daily-cycle series (period 24) with a slow upward trend and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(days * DAILY_PERIOD, dtype=float)
    level = 100.0 + 0.05 * t  # slow trend
    daily = 25.0 * np.sin(2.0 * np.pi * t / DAILY_PERIOD)
    noise = rng.normal(0.0, 1.5, size=t.shape[0])
    return level + daily + noise


def _naive_repeat_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """The naive comparator: repeat the most recent observed value."""
    return np.full(horizon, history[-1], dtype=float)


# ---------------------------------------------------------------------------
# 1. The forecast beats a naive repeat
# ---------------------------------------------------------------------------


def test_sarimax_beats_naive_repeat_on_seasonal_series():
    """SARIMAX must track the daily cycle better than repeating the last value."""
    series = _synthetic_seasonal_series(days=20)
    horizon = 24
    train = series[:-horizon]
    truth = series[-horizon:]

    sarimax_fc = fit_and_forecast_sarimax(train, horizon)
    naive_fc = _naive_repeat_forecast(train, horizon)

    assert sarimax_fc.shape == (horizon,)
    assert np.isfinite(sarimax_fc).all(), "SARIMAX should fit this clean series"

    sarimax_mae = mae(truth, sarimax_fc)
    naive_mae = mae(truth, naive_fc)

    # Sanity: the daily amplitude makes repeating one value genuinely poor.
    assert naive_mae > sarimax_mae, (
        f"SARIMAX ({sarimax_mae:.3f}) should beat naive repeat "
        f"({naive_mae:.3f}) on a seasonal series"
    )


# ---------------------------------------------------------------------------
# 2. Convergence failure -> all-NaN, no raise
# ---------------------------------------------------------------------------


def test_fit_failure_returns_all_nan_without_raising(monkeypatch, caplog):
    """A raising SARIMAX is swallowed: warn and return NaNs of the right length."""

    class _ExplodingSARIMAX:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated failure to converge")

    monkeypatch.setattr(sarimax_model, "SARIMAX", _ExplodingSARIMAX)

    horizon = 24
    history = _synthetic_seasonal_series(days=10)

    with caplog.at_level("WARNING"):
        forecast = fit_and_forecast_sarimax(history, horizon)

    assert forecast.shape == (horizon,)
    assert np.isnan(forecast).all(), "a failed fit must yield an all-NaN forecast"
    assert any("SARIMAX" in record.message for record in caplog.records)


def test_harness_survives_all_nan_sarimax_folds(monkeypatch):
    """End-to-end: an always-failing SARIMAX yields NaN forecasts, not a crash."""
    history = _synthetic_seasonal_series(days=8)
    timestamps = pd.date_range(
        pd.Timestamp("2024-01-01 00:00:00", tz="Asia/Kolkata"),
        periods=len(history),
        freq="1h",
    )
    frame = pd.DataFrame(
        {"building": "academic", "timestamp_ist": timestamps, "energy_kwh": history}
    )

    def failing_fn(train: np.ndarray, h: int) -> np.ndarray:
        return fit_and_forecast_sarimax(train, h)

    class _ExplodingSARIMAX:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("nope")

    monkeypatch.setattr(sarimax_model, "SARIMAX", _ExplodingSARIMAX)

    # Must not raise; every forecast is NaN so nothing is scored.
    result = rolling_origin_backtest(
        frame,
        forecast_fn=failing_fn,
        horizon=24,
        initial_train_size=24 * 4,
        step=24,
        buildings="academic",
    )
    assert len(result.predictions) > 0
    assert result.predictions["forecast"].isna().all()


# ---------------------------------------------------------------------------
# Optional exogenous regressors
# ---------------------------------------------------------------------------


def test_exogenous_regressors_are_optional_and_accepted():
    """The model runs with no exog, and also accepts aligned exog train/future."""
    series = _synthetic_seasonal_series(days=15)
    horizon = 24
    train = series[:-horizon]

    # No exog -- works.
    without_exog = fit_and_forecast_sarimax(train, horizon)
    assert np.isfinite(without_exog).all()

    # With exog -- also works when train/future are aligned.
    rng = np.random.default_rng(1)
    exog_train = rng.normal(size=(len(train), 2))
    exog_future = rng.normal(size=(horizon, 2))
    with_exog = fit_and_forecast_sarimax(
        train, horizon, exog_train=exog_train, exog_future=exog_future
    )
    assert np.isfinite(with_exog).all()


def test_supplying_only_one_exog_argument_raises():
    """Exog train and future are a matched pair; half of it is an error."""
    series = _synthetic_seasonal_series(days=10)
    with pytest.raises(ValueError, match="exog_train and exog_future"):
        fit_and_forecast_sarimax(series, 24, exog_train=np.ones((len(series), 1)))


def test_invalid_horizon_raises():
    series = _synthetic_seasonal_series(days=5)
    with pytest.raises(ValueError):
        fit_and_forecast_sarimax(series, 0)


# ---------------------------------------------------------------------------
# Harness building filter
# ---------------------------------------------------------------------------


def test_building_filter_restricts_the_backtest():
    """``buildings=`` backtests only the requested building."""
    n = 24 * 8
    ts = pd.date_range(
        pd.Timestamp("2024-01-01 00:00:00", tz="Asia/Kolkata"), periods=n, freq="1h"
    )
    a = pd.DataFrame(
        {
            "building": "academic",
            "timestamp_ist": ts,
            "energy_kwh": np.arange(n, dtype=float),
        }
    )
    b = pd.DataFrame(
        {
            "building": "library",
            "timestamp_ist": ts,
            "energy_kwh": np.arange(n, dtype=float) + 5.0,
        }
    )
    frame = pd.concat([a, b], ignore_index=True)

    result = rolling_origin_backtest(
        frame, horizon=24, initial_train_size=24 * 4, step=24, buildings="academic"
    )
    assert set(result.predictions["building"].unique()) == {"academic"}
    assert set(result.folds["building"].unique()) == {"academic"}

    # A list/tuple of names also works.
    result2 = rolling_origin_backtest(
        frame,
        horizon=24,
        initial_train_size=24 * 4,
        step=24,
        buildings=["academic", "library"],
    )
    assert set(result2.predictions["building"].unique()) == {"academic", "library"}