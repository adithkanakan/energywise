"""Unit tests for the forecast error metrics.

The defining rule is **pairwise NaN exclusion**: a position where either the
actual or the forecast is NaN is dropped from every metric -- numerator and
denominator alike -- and never filled with 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energywise.evaluation.metrics import (
    mae,
    paired_residuals,
    peak_hour_mask,
    peak_period_mae,
    rmse,
    wape,
)


def test_mae_rmse_wape_known_values():
    actual = [10.0, 20.0, 30.0, 40.0]
    forecast = [12.0, 18.0, 33.0, 37.0]
    # errors: 2, 2, 3, 3  -> mae = 2.5
    assert mae(actual, forecast) == pytest.approx(2.5)
    # rmse = sqrt((4+4+9+9)/4) = sqrt(6.5)
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(6.5))
    # wape = (2+2+3+3) / (10+20+30+40) = 10/100
    assert wape(actual, forecast) == pytest.approx(0.10)


def test_nan_excluded_pairwise_not_filled_with_zero():
    """NaN positions are dropped from *both* the numerator and denominator."""
    actual = [10.0, np.nan, 30.0, np.nan]
    forecast = [12.0, 5.0, np.nan, 1.0]
    # Only the first pair is complete.
    assert mae(actual, forecast) == pytest.approx(2.0)
    assert rmse(actual, forecast) == pytest.approx(2.0)
    assert wape(actual, forecast) == pytest.approx(2.0 / 10.0)

    # If NaNs had been treated as 0 the WAPE denominator would have been 40.
    assert wape(actual, forecast) != pytest.approx(2.0 / 40.0)


def test_zero_filled_would_differ():
    """Explicitly demonstrates the zero-fill anti-pattern is not used."""
    actual = [100.0, 100.0]
    forecast = [110.0, np.nan]
    # Correct (pairwise exclusion): only the first pair counts -> 10/100.
    assert wape(actual, forecast) == pytest.approx(0.10)
    # Zero-filling the missing forecast would have added |100 - 0| = 100 to the
    # numerator and 100 to the denominator -> 110/200, a completely different,
    # wrong answer.
    assert wape(actual, forecast) != pytest.approx(110.0 / 200.0)


def test_all_nan_returns_nan_not_zero():
    actual = [np.nan, np.nan]
    forecast = [1.0, 2.0]
    assert np.isnan(mae(actual, forecast))
    assert np.isnan(rmse(actual, forecast))
    assert np.isnan(wape(actual, forecast))


def test_zero_denominator_wape_is_nan():
    """WAPE is undefined (NaN) when sum(|actual|) is 0 -- not inf."""
    actual = [0.0, 0.0]
    forecast = [1.0, 2.0]
    assert np.isnan(wape(actual, forecast))
    # MAE/RMSE remain well-defined.
    assert mae(actual, forecast) == pytest.approx(1.5)


def test_paired_residuals_drops_union_of_nans():
    actual = pd.Series([1.0, np.nan, 3.0, 4.0])
    forecast = pd.Series([1.0, 2.0, np.nan, 4.0])
    a, f = paired_residuals(actual, forecast)
    np.testing.assert_allclose(a, [1.0, 4.0])
    np.testing.assert_allclose(f, [1.0, 4.0])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        mae([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Peak-period MAE
# ---------------------------------------------------------------------------


def test_peak_period_mae_ignores_errors_in_non_peak_hours():
    """A huge error in a non-peak hour must not move peak_period_mae.

    The synthetic series has clearly-known peak and non-peak hours (values
    0..99): the top 25% (>= 75) are peak, everything below is non-peak.  We
    inject a massive error only into a *non-peak* hour and check that
    peak_period_mae is unchanged, while the plain MAE is not.
    """
    actual = [float(i) for i in range(100)]
    forecast = [float(i) for i in range(100)]  # perfect forecast to start

    is_peak = peak_hour_mask(actual, quantile=0.75)
    # Sanity: exactly the top 25 hours are peak (75..99).
    assert is_peak.sum() == 25
    assert is_peak[:75].sum() == 0
    assert is_peak[75:].sum() == 25

    baseline_peak_mae = peak_period_mae(actual, forecast, is_peak)
    assert baseline_peak_mae == pytest.approx(0.0)

    # Inject a giant error into a non-peak hour (hour 3, value 3 -> < 75).
    broken = forecast.copy()
    broken[3] = 10_000.0
    assert not is_peak[3]

    # Peak MAE is completely unaffected by the non-peak error.
    assert peak_period_mae(actual, broken, is_peak) == pytest.approx(0.0)
    # ...whereas the overall MAE is dragged up by it, proving the mask filters.
    assert mae(actual, broken) > 1.0


def test_peak_period_mae_reflects_errors_in_peak_hours():
    """An error inside a peak hour *does* move peak_period_mae."""
    actual = [float(i) for i in range(100)]
    forecast = [float(i) for i in range(100)]
    is_peak = peak_hour_mask(actual, quantile=0.75)

    broken = forecast.copy()
    broken[80] = broken[80] + 10.0  # hour 80 is peak (>= 75)
    assert is_peak[80]

    # One peak hour with error 10 -> peak MAE = 10 / 25.
    assert peak_period_mae(actual, broken, is_peak) == pytest.approx(10.0 / 25.0)


def test_peak_period_mae_excludes_nan_peak_pairs():
    """A NaN actual or forecast in a peak hour is excluded, not zero-filled."""
    actual = [90.0, 95.0, np.nan, 80.0]
    forecast = [90.0, np.nan, 100.0, 70.0]
    is_peak = np.array([True, True, True, True])

    # Only the first (err 0) and last (err 10) pairs are complete.
    assert peak_period_mae(actual, forecast, is_peak) == pytest.approx((0.0 + 10.0) / 2)


def test_peak_period_mae_nan_when_no_peak_pairs():
    """No qualifying peak hours -> NaN, not 0."""
    actual = [10.0, 20.0]
    forecast = [11.0, 21.0]
    is_peak = np.array([False, False])
    assert np.isnan(peak_period_mae(actual, forecast, is_peak))


def test_peak_period_mae_mask_length_mismatch_raises():
    with pytest.raises(ValueError, match="is_peak"):
        peak_period_mae([1.0, 2.0], [1.0, 2.0], [True])
