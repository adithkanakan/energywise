"""Unit tests for the historical weather ingestion module.

The download step is not exercised (it needs the network), but the date
chunking and the row-preserving left join are.  The key guarantee under test
is that :func:`join_energy_weather` never duplicates or drops energy rows.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from energywise.ingestion.fetch_weather import (
    MAX_CHUNK_DAYS,
    WEATHER_JOIN_COLUMNS,
    _date_chunks,
    join_energy_weather,
)

IST_TZ = "Asia/Kolkata"


def _energy_frame(timestamps: list[str]) -> pd.DataFrame:
    """Minimal energy frame matching the cleaned hourly schema."""
    return pd.DataFrame(
        {
            "building": ["academic"] * len(timestamps),
            "timestamp_ist": pd.to_datetime(timestamps, utc=True).tz_convert(IST_TZ),
            "energy_kwh": [1.0] * len(timestamps),
            "minutes_present": [60] * len(timestamps),
            "is_missing": [False] * len(timestamps),
        }
    )


def _weather_frame(timestamps: list[str]) -> pd.DataFrame:
    """Minimal weather frame matching the processed hourly schema."""
    return pd.DataFrame(
        {
            "timestamp_ist": pd.to_datetime(timestamps, utc=True).tz_convert(IST_TZ),
            "temperature_c": [30.0] * len(timestamps),
            "humidity_pct": [60.0] * len(timestamps),
            "precipitation_mm": [0.0] * len(timestamps),
        }
    )


# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------


def test_date_chunks_cover_range_without_overlap_or_gaps():
    """Chunks are contiguous and cover the whole range exactly once."""
    chunks = _date_chunks(date(2013, 8, 10), date(2017, 12, 31))
    assert len(chunks) == 5  # 4 full years + a tail

    # First chunk starts at the requested start, last ends at the requested end.
    assert chunks[0][0] == date(2013, 8, 10)
    assert chunks[-1][1] == date(2017, 12, 31)

    # Each chunk is contiguous with the next (no gap, no overlap).
    for (_, end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == end + timedelta(days=1)


def test_date_chunks_respects_max_days():
    """No chunk exceeds ``max_days``."""
    for start, end in _date_chunks(date(2013, 8, 10), date(2017, 12, 31), MAX_CHUNK_DAYS):
        assert (end - start).days + 1 <= MAX_CHUNK_DAYS


# ---------------------------------------------------------------------------
# Join row-count preservation
# ---------------------------------------------------------------------------


def test_join_preserves_row_count(tmp_path):
    """The left join keeps exactly the energy row count (no dup/drop)."""
    energy = _energy_frame(
        ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
    )
    weather = _weather_frame(
        ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
    )

    energy_path = tmp_path / "hourly_energy.parquet"
    weather_path = tmp_path / "weather_hourly.parquet"
    output_path = tmp_path / "hourly_energy_weather.parquet"
    energy.to_parquet(energy_path, index=False)
    weather.to_parquet(weather_path, index=False)

    joined = join_energy_weather(energy_path, weather_path, output_path)

    assert len(joined) == len(energy)
    assert joined[WEATHER_JOIN_COLUMNS].notna().all().all()
    assert output_path.exists()
    assert len(pd.read_parquet(output_path)) == len(energy)


def test_join_preserves_row_count_with_missing_weather(tmp_path):
    """Energy hours without weather are kept, with NaN weather (never dropped)."""
    energy = _energy_frame(
        ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
    )
    # Only two of the three energy hours have weather.
    weather = _weather_frame(["2024-01-01 00:00:00", "2024-01-01 01:00:00"])

    energy_path = tmp_path / "hourly_energy.parquet"
    weather_path = tmp_path / "weather_hourly.parquet"
    output_path = tmp_path / "hourly_energy_weather.parquet"
    energy.to_parquet(energy_path, index=False)
    weather.to_parquet(weather_path, index=False)

    joined = join_energy_weather(energy_path, weather_path, output_path)

    assert len(joined) == len(energy)  # row count preserved
    assert joined[WEATHER_JOIN_COLUMNS].isna().any(axis=1).sum() == 1


def test_join_rejects_duplicate_weather_hours(tmp_path):
    """A duplicated weather hour would break the row-preserving join."""
    energy = _energy_frame(["2024-01-01 00:00:00"])
    weather = _weather_frame(["2024-01-01 00:00:00", "2024-01-01 00:00:00"])

    energy_path = tmp_path / "hourly_energy.parquet"
    weather_path = tmp_path / "weather_hourly.parquet"
    energy.to_parquet(energy_path, index=False)
    weather.to_parquet(weather_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        join_energy_weather(energy_path, weather_path, tmp_path / "out.parquet")