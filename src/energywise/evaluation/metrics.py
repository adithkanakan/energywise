"""Forecast error metrics for the EnergyWise evaluation harness.

Four point-forecast metrics are provided:

* :func:`mae`  -- mean absolute error, ``mean(|actual - forecast|)``.
* :func:`rmse` -- root mean squared error, ``sqrt(mean((actual - forecast) ** 2))``.
* :func:`wape` -- weighted absolute percentage error,
  ``sum(|actual - forecast|) / sum(|actual|)``.  WAPE is a scale-free
  percentage (multiply by 100 for a percentage) and is well suited to energy
  demand, which regularly crosses zero.
* :func:`peak_period_mae` -- mean absolute error restricted to a building's
  *peak* hours (the top 25% of hours by ``energy_kwh``), so accuracy on the
  highest-demand hours is reported on its own.  Pairwise NaN exclusion applies
  to this metric too.

Correctness rules
-----------------
* **Pairwise NaN exclusion.**  ``actual`` and ``forecast`` are compared
  element-by-element and any position where *either* value is NaN is dropped
  from **all** the metrics -- for both the numerator and the denominator of
  WAPE.  Two series of unequal length are compared positionally after a length
  check.
* **No zero-filling.**  A missing hour is missing, never 0.  NaNs are excluded,
  not replaced.
* **Undefined, not fabricated.**  A metric over zero valid pairs (or, for WAPE,
  a denominator of zero) is returned as NaN rather than ``0.0`` or ``inf``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "mae",
    "rmse",
    "wape",
    "peak_period_mae",
    "peak_hour_mask",
    "paired_residuals",
]


def _as_float_array(values: object) -> np.ndarray:
    """Coerce *values* (Series, list, ndarray, ...) to a 1-D float array."""
    if isinstance(values, (pd.Series, pd.Index)):
        return values.to_numpy(dtype=float).ravel()
    return np.asarray(values, dtype=float).ravel()


def _as_bool_array(values: object) -> np.ndarray:
    """Coerce *values* (Series, list, ndarray, scalar) to a 1-D bool array."""
    if isinstance(values, (pd.Series, pd.Index)):
        return values.to_numpy(dtype=bool).ravel()
    return np.asarray(values, dtype=bool).ravel()


def paired_residuals(
    actual: object, forecast: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairwise-complete ``(actual, forecast)`` arrays.

    Positions where *either* series is NaN are removed.  The two inputs must
    have the same length (they are compared positionally).
    """
    a = _as_float_array(actual)
    f = _as_float_array(forecast)
    if a.shape[0] != f.shape[0]:
        raise ValueError(
            f"actual and forecast must have equal length, got {a.shape[0]} "
            f"and {f.shape[0]}"
        )
    mask = ~(np.isnan(a) | np.isnan(f))
    return a[mask], f[mask]


def mae(actual: object, forecast: object) -> float:
    """Mean absolute error with pairwise NaN exclusion."""
    a, f = paired_residuals(actual, forecast)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - f)))


def rmse(actual: object, forecast: object) -> float:
    """Root mean squared error with pairwise NaN exclusion."""
    a, f = paired_residuals(actual, forecast)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - f) ** 2)))


def wape(actual: object, forecast: object) -> float:
    """Weighted absolute percentage error with pairwise NaN exclusion.

    ``wape = sum(|actual - forecast|) / sum(|actual|)`` computed over the
    pairwise-complete positions only.  Returns NaN when there are no valid
    pairs or when ``sum(|actual|)`` is zero (the ratio is undefined).
    """
    a, f = paired_residuals(actual, forecast)
    if a.size == 0:
        return float("nan")
    denominator = float(np.sum(np.abs(a)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(np.abs(a - f)) / denominator)


def peak_hour_mask(
    values: object,
    quantile: float = 0.75,
) -> np.ndarray:
    """Mark the *peak* hours of a single building's series.

    Peak hours are the top ``1 - quantile`` fraction of hours by
    ``energy_kwh`` -- with the default ``quantile=0.75`` that is the **top
    25%**.  The cutoff is the ``quantile``-th percentile of *this building's
    own* distribution (``numpy.nanquantile``); every hour whose value is at or
    above the cutoff is marked peak.  Ties exactly at the cutoff can therefore
    push the peak fraction marginally above 25%.

    Which distribution?  (design choice)
    ------------------------------------
    The threshold is intended to be computed **once from the full building
    series** -- the entire history -- and *not* re-derived inside each
    backtest fold.  This keeps "peak" a stable, building-level concept: the
    same hours stay peak in every fold, rather than becoming a moving target
    that shifts as the training window grows.  Callers should pass the
    building's full ``energy_kwh`` array and then use the resulting mask to
    flag forecast hours.

    NaN hours (missing readings) are never peak hours.  If every hour is NaN,
    or the input is empty, the mask is all ``False``.

    Parameters
    ----------
    values:
        One building's ``energy_kwh`` values, in chronological order.
    quantile:
        The percentile, in ``[0, 1]``, at or above which an hour is peak.
        Defaults to ``0.75`` (top 25%).

    Returns
    -------
    numpy.ndarray
        A boolean array the same length as ``values``.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be within [0, 1], got {quantile}")

    arr = _as_float_array(values)
    mask = np.zeros(arr.shape[0], dtype=bool)
    if arr.shape[0] == 0:
        return mask

    valid = ~np.isnan(arr)
    if not valid.any():
        return mask

    threshold = float(np.nanquantile(arr, quantile))
    return valid & (arr >= threshold)


def peak_period_mae(
    actual: object,
    forecast: object,
    is_peak: object,
) -> float:
    """Mean absolute error restricted to a building's peak-period hours.

    ``is_peak`` marks which positions belong to the building's peak set -- see
    :func:`peak_hour_mask`, which should be computed **once from the full
    series** and then matched to the forecast hours.  Only positions that are
    peak **and** have both an actual and a forecast present contribute; every
    other position is dropped, exactly as the other metrics drop pairwise
    NaNs.  A peak position whose actual or forecast is NaN is therefore
    *excluded* rather than filled with 0.

    Parameters
    ----------
    actual, forecast:
        Equal-length series compared positionally.
    is_peak:
        A boolean mask the same length as ``actual``/``forecast``.

    Returns
    -------
    float
        The MAE over the qualifying peak hours, or NaN when there are none.
    """
    a = _as_float_array(actual)
    f = _as_float_array(forecast)
    peak = _as_bool_array(is_peak)

    if a.shape[0] != f.shape[0]:
        raise ValueError(
            f"actual and forecast must have equal length, got {a.shape[0]} "
            f"and {f.shape[0]}"
        )
    if peak.shape[0] != a.shape[0]:
        raise ValueError(
            "is_peak must be the same length as actual/forecast, got "
            f"{peak.shape[0]} and {a.shape[0]}"
        )

    selected = peak & ~(np.isnan(a) | np.isnan(f))
    if not selected.any():
        return float("nan")
    return float(np.mean(np.abs(a[selected] - f[selected])))
