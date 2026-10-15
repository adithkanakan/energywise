"""Unit tests for the per-building lag / trailing-rolling feature module.

Two behaviours are required:

1. **Exact NaNs, never silent zeros.**  A lag or rolling statistic is NaN
   exactly where the source hour is NaN or where there is not yet enough
   history -- it is never quietly filled with 0 or a wrong value.
2. **Strictly backward-looking.**  A rolling value at row ``N`` depends only
   on rows before ``N``: changing ``energy_kwh`` at row ``N`` or any later row
   must not move it.

Supporting tests confirm the per-building isolation (one building's history
never leaks into another's).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energywise.features.lag_features import (
    LAG_COLUMNS,
    ROLLING_WINDOW,
    add_lag_features,
)

IST_TZ = "Asia/Kolkata"
N_HOURS = 200  # comfortably more than the 168h warm-up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(building: str, values: list[float], start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    """One building's hourly frame with the given ``energy_kwh`` values."""
    ts = pd.date_range(pd.Timestamp(start, tz=IST_TZ), periods=len(values), freq="1h")
    return pd.DataFrame(
        {
            "building": building,
            "timestamp_ist": ts,
            "energy_kwh": values,
        }
    )


# ---------------------------------------------------------------------------
# (1) NaNs are exact -- never 0, never wrong
# ---------------------------------------------------------------------------


def test_lags_and_rolling_are_nan_exactly_where_expected():
    """One known missing hour -> precise NaN pattern for every feature."""
    values = [1.0] * N_HOURS
    missing_at = 100
    values[missing_at] = np.nan

    out = add_lag_features(_series("A", values))

    rows = pd.RangeIndex(N_HOURS)

    # --- lag 1h: NaN only at the first row and where the source (i-1) is NaN.
    expected_lag1_nan = {0, missing_at + 1}
    assert set(out.index[out["energy_lag_1h"].isna()]) == expected_lag1_nan
    # Everywhere else it is the *previous* hour's value (1.0), not 0.0.
    valid_lag1 = out.loc[rows.difference(list(expected_lag1_nan)), "energy_lag_1h"]
    assert np.allclose(valid_lag1, 1.0)

    # --- lag 24h: NaN for the first 24 rows and where source (i-24) is NaN.
    expected_lag24_nan = set(range(0, 24)) | {missing_at + 24}
    assert set(out.index[out["energy_lag_24h"].isna()]) == expected_lag24_nan

    # --- lag 168h: NaN for the first 168 rows (warm-up), value thereafter.
    expected_lag168_nan = set(range(0, 168))
    assert set(out.index[out["energy_lag_168h"].isna()]) == expected_lag168_nan
    assert out.loc[168, "energy_lag_168h"] == pytest.approx(1.0)

    # --- rolling 24h: NaN while warming up (first 24 rows) and for any row
    # whose backward-looking 24-hour window contains the missing hour (101..124).
    expected_roll_nan = set(range(0, ROLLING_WINDOW)) | set(
        range(missing_at + 1, missing_at + ROLLING_WINDOW + 1)
    )
    for col in ("energy_roll_24h_mean", "energy_roll_24h_min", "energy_roll_24h_max"):
        assert set(out.index[out[col].isna()]) == expected_roll_nan, col

    # The rolling values that are valid equal 1.0 (they are not 0).
    valid_roll = out.loc[rows.difference(list(expected_roll_nan)), "energy_roll_24h_mean"]
    assert np.allclose(valid_roll, 1.0)
    assert (valid_roll != 0).all()


def test_missing_hour_is_not_treated_as_zero():
    """A missing hour must never make a lag or rolling stat equal 0.

    With a series of genuine 5.0 readings and one NaN, no produced feature may
    be 0.0 -- that would betray a silent ``fillna(0)``.
    """
    values = [5.0] * N_HOURS
    values[120] = np.nan

    out = add_lag_features(_series("A", values))

    for col in LAG_COLUMNS:
        non_nan = out[col].dropna()
        assert not (non_nan == 0.0).any(), f"{col} contains a spurious 0.0"


def test_rolling_nan_if_any_window_hour_missing():
    """A gap anywhere in the trailing window makes the whole stat NaN."""
    values = [2.0] * N_HOURS
    gap_at = 50
    values[gap_at] = np.nan

    out = add_lag_features(_series("A", values))

    # The gap is inside the window for rows gap_at+1 .. gap_at+24.
    for i in range(gap_at + 1, gap_at + ROLLING_WINDOW + 1):
        assert np.isnan(out.loc[i, "energy_roll_24h_mean"])
    # Just outside the window it is valid again.
    assert out.loc[gap_at + ROLLING_WINDOW + 1, "energy_roll_24h_mean"] == pytest.approx(2.0)
    assert out.loc[gap_at, "energy_roll_24h_mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# (2) Strictly backward-looking -- no leakage from the current/future rows
# ---------------------------------------------------------------------------


def test_rolling_mean_ignores_current_and_future_rows():
    """Changing ``energy_kwh`` at row N or later cannot move row N's features."""
    values = [float(i) for i in range(N_HOURS)]
    base = add_lag_features(_series("A", values))

    # Pick a row with a fully-populated history.
    n = 150
    assert not np.isnan(base.loc[n, "energy_roll_24h_mean"])

    # Vandalise the current row and everything after it.
    tampered_values = values.copy()
    for i in range(n, N_HOURS):
        tampered_values[i] = 999999.0
    tampered = add_lag_features(_series("A", tampered_values))

    # Rows up to and including n must be byte-for-byte identical.
    pd.testing.assert_frame_equal(
        base.loc[:n, LAG_COLUMNS],
        tampered.loc[:n, LAG_COLUMNS],
    )

    # Sanity: a *later* row's rolling mean DOES change (proving the mutation
    # took effect and the test isn't vacuous).
    assert base.loc[n + 1, "energy_roll_24h_mean"] != tampered.loc[
        n + 1, "energy_roll_24h_mean"
    ]


def test_rolling_window_matches_manual_prior_24_mean():
    """The rolling mean equals the mean of exactly the previous 24 hours."""
    values = [float(i) for i in range(N_HOURS)]
    out = add_lag_features(_series("A", values))

    n = 100
    manual = np.mean(values[n - ROLLING_WINDOW : n])  # indices n-24 .. n-1
    assert out.loc[n, "energy_roll_24h_mean"] == pytest.approx(manual)
    # min/max over the same window.
    assert out.loc[n, "energy_roll_24h_min"] == pytest.approx(min(values[n - 24 : n]))
    assert out.loc[n, "energy_roll_24h_max"] == pytest.approx(max(values[n - 24 : n]))


# ---------------------------------------------------------------------------
# Per-building isolation
# ---------------------------------------------------------------------------


def test_features_are_computed_per_building():
    """One building's history never bleeds into another's."""
    a = _series("A", [1.0] * 30)
    b = _series("B", [10.0] * 30, start="2024-01-05 00:00:00")
    frame = pd.concat([a, b], ignore_index=True)

    out = add_lag_features(frame)
    out_a = out[out["building"] == "A"].reset_index(drop=True)
    out_b = out[out["building"] == "B"].reset_index(drop=True)

    # Each building's first row has no lag-1 history (NaN) despite the other
    # building's data being adjacent in the concatenated frame.
    assert np.isnan(out_a.loc[0, "energy_lag_1h"])
    assert np.isnan(out_b.loc[0, "energy_lag_1h"])

    # And each building's lags reflect its own constant, not the other's.
    assert out_a.loc[5, "energy_lag_1h"] == pytest.approx(1.0)
    assert out_b.loc[5, "energy_lag_1h"] == pytest.approx(10.0)


def test_row_order_is_preserved():
    """The output keeps the input's original row order and adds all columns."""
    a = _series("A", [1.0] * 5, start="2024-01-01 00:00:00")
    b = _series("B", [2.0] * 5, start="2024-01-02 00:00:00")
    # Interleave so input order is not sorted by building.
    frame = pd.concat([a, b], ignore_index=True).sort_values(
        "timestamp_ist", kind="mergesort"
    ).reset_index(drop=True)
    before = frame[["building", "timestamp_ist"]].copy()

    out = add_lag_features(frame)

    pd.testing.assert_frame_equal(out[["building", "timestamp_ist"]], before)
    for col in LAG_COLUMNS:
        assert col in out.columns


def test_input_not_mutated_and_validation():
    """Default call leaves the input untouched; missing columns raise."""
    frame = _series("A", [1.0] * 5)
    before = list(frame.columns)

    add_lag_features(frame)

    assert list(frame.columns) == before
    assert "energy_lag_1h" not in frame.columns

    with pytest.raises(ValueError, match="energy_kwh"):
        add_lag_features(pd.DataFrame({"building": ["A"], "timestamp_ist": [pd.Timestamp("2024-01-01")]}))