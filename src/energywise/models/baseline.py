"""Seasonal-naive baseline forecaster for the EnergyWise hourly series.

The seasonal-naive ("persistence with a seasonal lag") benchmark predicts an
hour using the value observed one *season* earlier.  With the default
``season=168`` that means *the same hour on the same weekday one week ago* --
a strong, leakage-free baseline for campus energy demand, which is dominated by
daily and weekly cycles.

Prediction rule
---------------
For a target hour ``N`` (the hour being forecast):

1. **Primary source:** the value at ``N - season`` (default 168 h earlier).
2. **Fallback:** if that hour is NaN (or lies before the start of the series),
   use ``N - 24`` (the same hour one day earlier).
3. **Giving up:** if the fallback is also NaN (or unavailable), the forecast is
   NaN.  A missing hour is *never* fabricated as 0 or a carried-forward value.

The rule is applied independently per building: callers pass one building's
history at a time, so no building's history ever leaks into another's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["seasonal_naive_forecast", "DEFAULT_HORIZON", "DEFAULT_SEASON"]

#: Default number of hours to forecast beyond the end of the history.
DEFAULT_HORIZON = 24

#: Default seasonal lag, in hours (one week == same hour, same weekday).
DEFAULT_SEASON = 168

#: Fallback lag, in hours, used when the seasonal source hour is missing.
FALLBACK_SEASON = 24


def _as_float_array(series: object) -> np.ndarray:
    """Coerce *series* (Series/list/ndarray) to a 1-D float64 array."""
    if isinstance(series, (pd.Series, pd.Index)):
        return series.to_numpy(dtype=float).ravel()
    return np.asarray(series, dtype=float).ravel()


def seasonal_naive_forecast(
    series: object,
    horizon: int = DEFAULT_HORIZON,
    season: int = DEFAULT_SEASON,
) -> np.ndarray:
    """Forecast the next ``horizon`` hours of one building's series.

    Parameters
    ----------
    series:
        The *history* for a single building, oldest first.  Missing hours are
        represented as NaN and are honoured (they are never treated as 0).
        Accepts a pandas Series, a list or a NumPy array.
    horizon:
        How many future hours to predict (default 24).  Must be a positive
        integer.
    season:
        The seasonal lag in hours (default 168 == one week).  Must be a
        positive integer.

    Returns
    -------
    numpy.ndarray
        A float array of length ``horizon``.  Position ``h`` (0-based) is the
        forecast for the hour ``len(series) + h``.  Entries are NaN when
        neither the seasonal source nor its 24-hour fallback is available.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")
    if season <= 0:
        raise ValueError(f"season must be a positive integer, got {season}")

    history = _as_float_array(series)
    n = history.shape[0]
    forecasts = np.full(horizon, np.nan, dtype=float)

    for h in range(horizon):
        target = n + h  # 0-based position of the hour being forecast

        # 1. Seasonal source, one ``season`` hours earlier.
        primary = target - season
        if 0 <= primary < n and not np.isnan(history[primary]):
            forecasts[h] = history[primary]
            continue

        # 2. Fallback, one day (24 h) earlier.
        fallback = target - FALLBACK_SEASON
        if 0 <= fallback < n and not np.isnan(history[fallback]):
            forecasts[h] = history[fallback]
            continue

        # 3. Neither source is available -- return NaN rather than fabricate.
        forecasts[h] = np.nan

    return forecasts


def main() -> None:
    """CLI demo: print a one-week-ahead baseline for the default dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    input_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "processed"
        / "hourly_energy_weather.parquet"
    )
    hourly = pd.read_parquet(input_path)
    for building, group in hourly.groupby("building", sort=True):
        history = group.sort_values("timestamp_ist")["energy_kwh"]
        fc = seasonal_naive_forecast(history, horizon=DEFAULT_HORIZON)
        print(f"{building}: next {DEFAULT_HORIZON}h -> {np.round(fc, 3)}")


if __name__ == "__main__":
    main()