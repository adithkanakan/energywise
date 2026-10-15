"""SARIMAX forecaster for the EnergyWise hourly energy series.

This module wraps :class:`statsmodels.tsa.statespace.sarimax.SARIMAX` in a
single per-building entry point, :func:`fit_and_forecast_sarimax`, that fits on
one building's training window and returns a fixed-length horizon forecast.

Because the campus series are hourly and dominated by a daily cycle, the
default model is a small SARIMAX with a **24-hour seasonal period**::

    order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)

.. note::

   ``(1, 0, 1) x (1, 0, 1, 24)`` are **reasonable starting values, not tuned**.
   They are deliberately small so that a full rolling-origin backtest stays
   tractable.  The orders should be revisited (ACF/PACF, information criteria,
   or a light grid search) before any numbers are treated as a final result.

Optional exogenous regressors
-----------------------------
``exog_train`` / ``exog_future`` allow weather or calendar features to be
passed in later.  They are entirely optional: when both are ``None`` the model
is a plain (S)ARIMA on the endogenous series alone.  If *either* is supplied
both must be, because a model fitted with exogenous regressors requires the
future regressors to produce a forecast.

Robustness
----------
Real hourly data are messy.  A SARIMAX fit can fail to converge or can raise
for a particular window.  Rather than crash a long backtest,
:func:`fit_and_forecast_sarimax` catches any such failure, logs a warning and
returns an **all-NaN** forecast of the requested length.  The evaluation
harness already drops NaN forecast hours pairwise, so a single bad fold simply
drops out of the metrics instead of taking the whole run down.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

__all__ = [
    "fit_and_forecast_sarimax",
    "sarimax_forecast_fn",
    "DEFAULT_ORDER",
    "DEFAULT_SEASONAL_ORDER",
]

#: Non-seasonal (p, d, q) order.  A modest starting point, not tuned.
DEFAULT_ORDER: tuple[int, int, int] = (1, 0, 1)

#: Seasonal (P, D, Q, s) order -- one daily cycle of hourly data (s = 24).
#: A modest starting point, not tuned.
DEFAULT_SEASONAL_ORDER: tuple[int, int, int, int] = (1, 0, 1, 24)


def _as_float_array(values: object) -> np.ndarray:
    """Coerce *values* (Series/list/ndarray) to a 1-D float64 array."""
    if isinstance(values, (pd.Series, pd.Index)):
        return values.to_numpy(dtype=float).ravel()
    return np.asarray(values, dtype=float).ravel()


def _as_2d_float_array(values: object) -> np.ndarray:
    """Coerce exogenous *values* to a 2-D float array (rows == observations)."""
    if isinstance(values, pd.DataFrame):
        return values.to_numpy(dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def fit_and_forecast_sarimax(
    train_series: object,
    horizon: int,
    exog_train: object | None = None,
    exog_future: object | None = None,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
) -> np.ndarray:
    """Fit a SARIMAX model on one building's window and forecast ``horizon``.

    Parameters
    ----------
    train_series:
        The *history* for a single building, oldest first.  Accepts a pandas
        Series (its index is ignored), a list or a NumPy array.  Missing hours
        are represented as NaN; ``SARIMAX`` handles missing endog values via the
        Kalman filter.
    horizon:
        How many future hours to predict.  Must be a positive integer.
    exog_train:
        Optional exogenous regressors aligned to ``train_series`` -- one row per
        training hour (a 2-D array or a DataFrame).  A 1-D array is treated as
        a single regressor.  Pass ``None`` (default) for no exogenous input.
    exog_future:
        Optional exogenous regressors aligned to the forecast window -- one row
        per forecast hour.  Required whenever ``exog_train`` is supplied.
    order:
        Non-seasonal ``(p, d, q)`` order.  Defaults to ``(1, 0, 1)`` (a
        reasonable starting value, not tuned).
    seasonal_order:
        Seasonal ``(P, D, Q, s)`` order.  Defaults to ``(1, 0, 1, 24)`` -- one
        daily cycle of hourly data (a reasonable starting value, not tuned).

    Returns
    -------
    numpy.ndarray
        A float array of length ``horizon``.  On a fit/forecast failure (for
        example a non-converging window) an **all-NaN** array of the same length
        is returned and a warning is logged, rather than raising.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    history = _as_float_array(train_series)

    exog_train_arr: np.ndarray | None = None
    exog_future_arr: np.ndarray | None = None
    if exog_train is not None or exog_future is not None:
        if exog_train is None or exog_future is None:
            raise ValueError(
                "exog_train and exog_future must be provided together; a model "
                "fitted with exogenous regressors needs future regressors too"
            )
        exog_train_arr = _as_2d_float_array(exog_train)
        exog_future_arr = _as_2d_float_array(exog_future)
        if exog_train_arr.shape[0] != history.shape[0]:
            raise ValueError(
                "exog_train must have one row per training observation, got "
                f"{exog_train_arr.shape[0]} rows for {history.shape[0]} hours"
            )
        if exog_future_arr.shape[0] != horizon:
            raise ValueError(
                "exog_future must have one row per forecast hour, got "
                f"{exog_future_arr.shape[0]} rows for horizon {horizon}"
            )

    try:
        model = SARIMAX(
            history,
            exog=exog_train_arr,
            order=order,
            seasonal_order=seasonal_order,
            # Relaxing the stationarity/invertibility constraints keeps fitting
            # fast and avoids some spurious "failed to converge" failures on
            # short or noisy training windows.  It is a deliberate start value.
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=horizon, exog=exog_future_arr)
        forecast = np.asarray(forecast, dtype=float).ravel()
        if forecast.shape[0] != horizon:
            raise ValueError(
                f"SARIMAX returned {forecast.shape[0]} values, expected {horizon}"
            )
        return forecast
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module doc.
        logger.warning(
            "SARIMAX fit/forecast failed (n_train=%d, horizon=%d, order=%s, "
            "seasonal_order=%s): %s -- returning all-NaN forecast",
            history.shape[0],
            horizon,
            order,
            seasonal_order,
            exc,
        )
        return np.full(horizon, np.nan, dtype=float)


def sarimax_forecast_fn(**kwargs):
    """Return a ``forecast_fn(history, horizon)`` closure for the backtest harness.

    Any keyword arguments are forwarded to :func:`fit_and_forecast_sarimax`, so
    this adapter can be passed straight to
    :func:`energywise.evaluation.backtest.rolling_origin_backtest` and run in
    exactly the same way as the seasonal-naive baseline.
    """

    def _forecast(history: np.ndarray, horizon: int) -> np.ndarray:
        return fit_and_forecast_sarimax(history, horizon, **kwargs)

    return _forecast