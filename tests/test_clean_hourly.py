"""Unit tests for the hourly energy cleaning module.

Covers the three required behaviours:

* 60 minutes at 1000 W gives ``1.0`` kWh,
* an hour with only 30 minutes present gives ``NaN``,
* an hour of all zeros gives ``0.0`` (not ``NaN``).

The tests exercise the pure per-building aggregation helper directly with
synthetic minute-resolution data, so they run fast and need no external files.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from energywise.ingestion.clean_hourly import (
    MIN_MINUTES_PRESENT,
    OUTPUT_COLUMNS,
    _hourly_for_building,
    _prepare,
)

IST_TZ = "Asia/Kolkata"


def _minutes_frame(powers: list[float | None], start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    """Build a one-building raw-like frame with one reading per minute.

    ``powers`` may contain ``None`` to represent an NA power reading.
    """
    start_ts = pd.Timestamp(start, tz=IST_TZ)
    timestamps = pd.date_range(start_ts, periods=len(powers), freq="1min", tz=IST_TZ)
    return pd.DataFrame(
        {
            "building": "test",
            "timestamp_utc": timestamps.tz_convert("UTC"),
            "timestamp_ist": timestamps,
            "power": powers,
        }
    )


def _hourly(powers: list[float | None], start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    """Prepare + aggregate a synthetic frame down to hourly rows."""
    prepared = _prepare(_minutes_frame(powers, start))
    return _hourly_for_building(prepared)


# ---------------------------------------------------------------------------
# Required behaviours
# ---------------------------------------------------------------------------


def test_full_hour_at_1000w_is_1_kwh():
    """60 minutes at 1000 W -> 1.0 kWh, 60 minutes present, not missing."""
    hourly = _hourly([1000.0] * 60)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["energy_kwh"] == pytest.approx(1.0)
    assert row["minutes_present"] == 60
    assert bool(row["is_missing"]) is False


def test_partial_hour_30_minutes_is_nan():
    """An hour with only 30 minutes present -> NaN energy and is_missing=True."""
    assert MIN_MINUTES_PRESENT == 45  # guard against threshold drift

    hourly = _hourly([1000.0] * 30)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["minutes_present"] == 30
    assert math.isnan(row["energy_kwh"])
    assert bool(row["is_missing"]) is True


def test_all_zeros_hour_is_zero_not_nan():
    """An hour of 60 zero-watt readings -> 0.0 kWh (a genuine zero, not missing)."""
    hourly = _hourly([0.0] * 60)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["minutes_present"] == 60
    assert bool(row["is_missing"]) is False
    assert row["energy_kwh"] == pytest.approx(0.0)
    assert not math.isnan(row["energy_kwh"])


# ---------------------------------------------------------------------------
# Supporting behaviours
# ---------------------------------------------------------------------------


def test_na_power_is_missing_not_zero():
    """NA power readings reduce minutes_present and are excluded from the mean.

    A full hour where 30 readings are 2000 W and 30 are NA has 30 minutes
    present -> below the threshold -> NaN (and definitely not treated as 0 W).
    """
    powers = [2000.0] * 30 + [None] * 30
    hourly = _hourly(powers)

    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["minutes_present"] == 30
    assert math.isnan(row["energy_kwh"])
    assert bool(row["is_missing"]) is True


def test_na_reduces_minutes_but_mean_uses_present_only():
    """With 50 present minutes (>=45), energy = mean of present / 1000."""
    # 50 readings at 600 W and 10 NA -> 50 minutes present, mean = 600 W.
    powers = [600.0] * 50 + [None] * 10
    hourly = _hourly(powers)

    row = hourly.iloc[0]
    assert row["minutes_present"] == 50
    assert bool(row["is_missing"]) is False
    assert row["energy_kwh"] == pytest.approx(0.6)


def test_threshold_boundary_45_minutes_is_valid():
    """Exactly MIN_MINUTES_PRESENT minutes present is valid (not missing)."""
    hourly = _hourly([1000.0] * MIN_MINUTES_PRESENT)

    row = hourly.iloc[0]
    assert row["minutes_present"] == MIN_MINUTES_PRESENT
    assert bool(row["is_missing"]) is False
    assert row["energy_kwh"] == pytest.approx(1.0)


def test_threshold_boundary_44_minutes_is_missing():
    """One minute below the threshold is missing."""
    hourly = _hourly([1000.0] * (MIN_MINUTES_PRESENT - 1))

    row = hourly.iloc[0]
    assert row["minutes_present"] == MIN_MINUTES_PRESENT - 1
    assert bool(row["is_missing"]) is True
    assert math.isnan(row["energy_kwh"])


def test_gap_hours_between_readings_are_explicit_nan_rows():
    """A missing hour between observations appears as an explicit NaN row."""
    # Hour 0 fully observed, hour 1 absent, hour 2 fully observed.
    first = _minutes_frame([1000.0] * 60, start="2024-01-01 00:00:00")
    third = _minutes_frame([1000.0] * 60, start="2024-01-01 02:00:00")
    prepared = _prepare(pd.concat([first, third], ignore_index=True))

    hourly = _hourly_for_building(prepared)

    assert len(hourly) == 3  # 00:00, 01:00, 02:00 -- index is complete
    middle = hourly.iloc[1]
    assert middle["timestamp_ist"].hour == 1
    assert middle["minutes_present"] == 0
    assert bool(middle["is_missing"]) is True
    assert math.isnan(middle["energy_kwh"])


def test_output_columns_and_schema():
    """The output has exactly the required columns, in order."""
    hourly = _hourly([1000.0] * 60)
    assert list(hourly.columns) == OUTPUT_COLUMNS
    assert list(hourly.columns) == [
        "building",
        "timestamp_ist",
        "energy_kwh",
        "minutes_present",
        "is_missing",
    ]


def test_dropped_columns_absent():
    """Voltage/current/frequency/power_factor never reach the hourly output."""
    frame = _minutes_frame([1000.0] * 60)
    for col in ("voltage", "current", "frequency", "power_factor"):
        frame[col] = 999.0

    prepared = _prepare(frame)
    hourly = _hourly_for_building(prepared)

    for col in ("voltage", "current", "frequency", "power_factor"):
        assert col not in prepared.columns
        assert col not in hourly.columns