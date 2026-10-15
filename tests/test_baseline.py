"""Unit tests for the seasonal-naive baseline forecaster.

The two essential behaviours are:

1. **It copies the seasonal hour.**  With a perfectly periodic series the
   forecast is exact, so a backtest of it must yield (near-)zero error.
2. **It never fabricates a value.**  When neither the seasonal source nor the
   24-hour fallback is available (NaN or out of range), the forecast is NaN --
   never 0 and never a carried-forward value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energywise.evaluation.backtest import rolling_origin_backtest
from energywise.evaluation.metrics import mae, rmse, wape
from energywise.models.baseline import seasonal_naive_forecast


# ---------------------------------------------------------------------------
# Perfectly repeating series -> near-zero error
# ---------------------------------------------------------------------------


def _perfect_weekly_series(weeks: int = 12) -> np.ndarray:
    """A series that repeats exactly every 168 hours (one week)."""
    one_week = np.array(
        [100.0 + 10.0 * hour + (5.0 if hour % 2 == 0 else -5.0) for hour in range(168)]
    )
    return np.tile(one_week, weeks)


def test_seasonal_naive_is_exact_on_periodic_series():
    """Forecasting a perfectly repeating series reproduces the next week exactly."""
    history = _perfect_weekly_series(weeks=12)
    horizon = 24
    forecast = seasonal_naive_forecast(history, horizon=horizon, season=168)

    # The true continuation is just the next 24 values of the same pattern.
    truth = np.tile(
        history[:168], len(history) // 168 + 2
    )[len(history) : len(history) + horizon]

    np.testing.assert_allclose(forecast, truth)
    assert mae(truth, forecast) == pytest.approx(0.0, abs=1e-9)
    assert rmse(truth, forecast) == pytest.approx(0.0, abs=1e-9)
    assert wape(truth, forecast) == pytest.approx(0.0, abs=1e-9)


def test_backtest_of_periodic_series_has_near_zero_error():
    """End-to-end: a periodic building through the harness scores ~0 error."""
    values = _perfect_weekly_series(weeks=12)
    n = len(values)
    timestamps = pd.date_range(
        pd.Timestamp("2024-01-01 00:00:00", tz="Asia/Kolkata"),
        periods=n,
        freq="1h",
    )
    frame = pd.DataFrame(
        {
            "building": "A",
            "timestamp_ist": timestamps,
            "energy_kwh": values,
        }
    )

    result = rolling_origin_backtest(
        frame, horizon=24, initial_train_size=168 * 4, step=24
    )
    assert len(result.predictions) > 0

    a = result.predictions["actual"]
    f = result.predictions["forecast"]
    assert mae(a, f) == pytest.approx(0.0, abs=1e-9)
    assert wape(a, f) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# NaN handling -- never fabricate
# ---------------------------------------------------------------------------


def test_missing_seasonal_hour_falls_back_to_same_hour_yesterday():
    """A NaN at N-168 falls back to N-24 when that hour is present."""
    history = np.arange(200, dtype=float)
    history[-168] = np.nan  # the seasonal source for the first forecast hour

    forecast = seasonal_naive_forecast(history, horizon=1, season=168)
    # Fallback = value 24 hours before the target = history[-24].
    assert forecast[0] == pytest.approx(history[-24])


def test_nan_when_both_sources_missing():
    """If both the seasonal and the 24h source are NaN, the forecast is NaN."""
    history = np.arange(200, dtype=float)
    history[-168] = np.nan
    history[-24] = np.nan

    forecast = seasonal_naive_forecast(history, horizon=1, season=168)
    assert np.isnan(forecast[0])


def test_never_returns_zero_for_unavailable_source():
    """A short history yields NaN where no source exists, never 0."""
    # With a 3-hour history and a single-step forecast neither the 168h source
    # nor the 24h fallback exists, so the forecast must be NaN (not 0).
    forecast = seasonal_naive_forecast([5.0, 5.0, 5.0], horizon=1, season=168)
    assert np.isnan(forecast).all()

    # Over a full day the earliest hours have no source at all -> NaN, but the
    # final few hours *do* find their 24h fallback inside the history.
    day = seasonal_naive_forecast([5.0, 5.0, 5.0], horizon=24, season=168)
    assert np.isnan(day[:21]).all()  # no source at all -> NaN
    assert np.allclose(day[21:], 5.0)  # fallback to history[0:3]
    # No value is ever fabricated as 0.
    assert not (day == 0.0).any()


def test_output_length_and_positional_alignment():
    """Position h forecasts hour len(history)+h."""
    history = np.arange(300, dtype=float)
    forecast = seasonal_naive_forecast(history, horizon=24, season=168)
    assert forecast.shape == (24,)
    # Each forecast hour equals its 168-hours-earlier value.
    np.testing.assert_allclose(forecast, history[300 - 168 : 300 - 168 + 24])


def test_invalid_arguments_raise():
    history = np.arange(200, dtype=float)
    with pytest.raises(ValueError):
        seasonal_naive_forecast(history, horizon=0)
    with pytest.raises(ValueError):
        seasonal_naive_forecast(history, horizon=24, season=0)