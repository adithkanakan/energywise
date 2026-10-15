"""Unit tests for the calendar / cyclical time-feature module.

The headline requirement is the **cyclical wrap-around**: hour ``0`` and hour
``23`` must land close together in ``(hour_sin, hour_cos)`` space, unlike the
raw ``hour`` column which puts them 23 units apart.  Supporting tests confirm
the other derived features and the leakage-safety guarantee (nothing depends on
``energy_kwh``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from energywise.features.time_features import (
    FEATURE_COLUMNS,
    IST_TZ,
    add_time_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(timestamps: list[str]) -> pd.DataFrame:
    """Build a minimal frame whose only meaningful column is ``timestamp_ist``."""
    return pd.DataFrame(
        {
            "building": "test",
            "timestamp_ist": pd.to_datetime(timestamps).tz_localize(IST_TZ),
            "energy_kwh": 1.0,
            "minutes_present": 60,
            "is_missing": False,
        }
    )


# ---------------------------------------------------------------------------
# Headline: cyclical wrap-around
# ---------------------------------------------------------------------------


def test_hour_0_and_hour_23_are_close_in_cyclical_space():
    """hour 0 and hour 23 must be near-neighbours on the unit circle.

    Euclidean distance between ``(hour_sin, hour_cos)`` at the two hours should
    be small (one hour apart), and vastly smaller than the raw ``|0 - 23|`` gap.
    """
    frame = _frame(["2024-01-01 00:00:00", "2024-01-01 23:00:00"])
    out = add_time_features(frame)

    h0 = out.iloc[0]
    h23 = out.iloc[1]

    # Sanity: the raw hour values are the extremes of the range.
    assert h0["hour"] == 0
    assert h23["hour"] == 23

    distance = math.hypot(
        h0["hour_sin"] - h23["hour_sin"],
        h0["hour_cos"] - h23["hour_cos"],
    )

    # One step on a 24-point unit circle = 2*sin(pi/24) ~= 0.2611.
    # A threshold well below 1.0 proves they are treated as adjacent rather
    # than 23 units apart.
    assert distance < 0.3, f"hour 0 and 23 too far apart: {distance}"
    assert distance == pytest.approx(2 * math.sin(math.pi / 24), abs=1e-12)

    # And confirm every *other* hour is at least as far from hour 0.
    for h in range(1, 23):
        other = add_time_features(_frame([f"2024-01-01 {h:02d}:00:00"])).iloc[0]
        other_distance = math.hypot(
            h0["hour_sin"] - other["hour_sin"],
            h0["hour_cos"] - other["hour_cos"],
        )
        assert other_distance >= distance - 1e-12, (
            f"hour {h} unexpectedly closer to hour 0 than hour 23"
        )


def test_adjacent_hours_are_closer_than_distant_hours():
    """Cyclical distance grows monotonically up to the antipode (hour 12)."""
    hours = sorted({0, 1, 6, 12, 18, 23})
    out = add_time_features(_frame([f"2024-01-01 {h:02d}:00:00" for h in hours]))

    def dist(a: int, b: int) -> float:
        ra = out[out["hour"] == a].iloc[0]
        rb = out[out["hour"] == b].iloc[0]
        return math.hypot(ra["hour_sin"] - rb["hour_sin"], ra["hour_cos"] - rb["hour_cos"])

    assert dist(0, 1) < dist(0, 6) < dist(0, 12)
    # Wrap-around: hour 23 is exactly as close to 0 as hour 0 is to 1.
    assert dist(0, 23) == pytest.approx(dist(0, 1), abs=1e-12)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def test_calendar_features_values():
    """hour/day_of_week/month/is_weekend match the timestamp.

    2024-01-05 is a Friday (day_of_week=4); 2024-01-06 is a Saturday.
    """
    frame = _frame(["2024-01-05 13:00:00", "2024-01-06 07:00:00"])
    out = add_time_features(frame)

    fri, sat = out.iloc[0], out.iloc[1]

    assert fri["hour"] == 13
    assert fri["day_of_week"] == 4
    assert fri["month"] == 1
    assert bool(fri["is_weekend"]) is False

    assert sat["hour"] == 7
    assert sat["day_of_week"] == 5
    assert bool(sat["is_weekend"]) is True


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2013-01-26", True),   # Republic Day
        ("2015-08-15", True),   # Independence Day
        ("2016-10-02", True),   # Gandhi Jayanti
        ("2017-12-25", True),   # Christmas Day
        ("2014-07-09", False),  # ordinary weekday
    ],
)
def test_is_holiday(date: str, expected: bool):
    """Known holidays are flagged; an ordinary day is not."""
    out = add_time_features(_frame([f"{date} 09:00:00"]))
    assert bool(out.iloc[0]["is_holiday"]) is expected


def test_dow_cyclical_wraps_sunday_to_monday():
    """Sunday (6) and Monday (0) are adjacent in ``(dow_sin, dow_cos)`` space."""
    # 2024-01-07 is a Sunday, 2024-01-08 is a Monday.
    out = add_time_features(_frame(["2024-01-07 00:00:00", "2024-01-08 00:00:00"]))
    sun, mon = out.iloc[0], out.iloc[1]

    assert sun["day_of_week"] == 6
    assert mon["day_of_week"] == 0

    distance = math.hypot(sun["dow_sin"] - mon["dow_sin"], sun["dow_cos"] - mon["dow_cos"])
    # One step on a 7-point circle = 2*sin(pi/7) ~= 0.868.
    assert distance == pytest.approx(2 * math.sin(math.pi / 7), abs=1e-12)


# ---------------------------------------------------------------------------
# Schema / integrity
# ---------------------------------------------------------------------------


def test_all_feature_columns_present():
    """Every advertised feature column is produced."""
    out = add_time_features(_frame(["2024-01-01 00:00:00"]))
    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"missing feature column: {col}"


def test_cyclical_encodings_lie_on_unit_circle():
    """sin^2 + cos^2 == 1 for every hour of the day."""
    out = add_time_features(_frame([f"2024-01-01 {h:02d}:00:00" for h in range(24)]))

    hour_norm = out["hour_sin"] ** 2 + out["hour_cos"] ** 2
    dow_norm = out["dow_sin"] ** 2 + out["dow_cos"] ** 2
    assert np.allclose(hour_norm, 1.0)
    assert np.allclose(dow_norm, 1.0)


def test_input_is_not_mutated_by_default():
    """The default call returns a copy and leaves the input frame untouched."""
    frame = _frame(["2024-01-01 00:00:00"])
    before = list(frame.columns)

    add_time_features(frame)

    assert list(frame.columns) == before
    assert "hour_sin" not in frame.columns


def test_features_are_leakage_safe():
    """Changing ``energy_kwh`` must not change any time feature.

    This is the core guarantee: time features depend only on ``timestamp_ist``.
    """
    frame = _frame(["2024-03-15 18:00:00", "2024-03-16 05:00:00"])
    baseline = add_time_features(frame)

    tampered = frame.copy()
    tampered["energy_kwh"] = [999999.0, -12345.0]
    mutated = add_time_features(tampered)

    pd.testing.assert_frame_equal(baseline[FEATURE_COLUMNS], mutated[FEATURE_COLUMNS])


def test_missing_timestamp_raises():
    """A frame without ``timestamp_ist`` is rejected with a clear error."""
    with pytest.raises(ValueError, match="timestamp_ist"):
        add_time_features(pd.DataFrame({"energy_kwh": [1.0]}))