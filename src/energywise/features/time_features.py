"""Calendar / cyclical time features for the EnergyWise hourly table.

Takes the hourly energy+weather frame produced upstream
(``data/processed/hourly_energy_weather.parquet``) and adds a set of
**leakage-safe** time features:

* ``hour``        -- hour of day in IST, ``0``-``23``
* ``day_of_week`` -- Monday=``0`` ... Sunday=``6``
* ``month``       -- calendar month, ``1``-``12``
* ``is_weekend``  -- ``True`` for Sat/Sun
* ``is_holiday``  -- ``True`` on an Indian public holiday
* ``hour_sin`` / ``hour_cos`` -- cyclical encoding of ``hour``
* ``dow_sin`` / ``dow_cos``   -- cyclical encoding of ``day_of_week``

**Leakage-safety.** Every feature here is a deterministic function of
``timestamp_ist`` alone.  Nothing is derived from ``energy_kwh``,
``minutes_present``, ``is_missing`` or any measured value, so an hour's
features are fully known in advance and the model cannot peek at the target.

**Why cyclical encodings?**  A plain ``hour`` column tells a model that hour
``23`` and hour ``0`` are 23 units apart when in reality they are adjacent.
Projecting onto a circle with a sine/cosine pair makes those two hours
neighbouring points, so the model can learn smooth daily/weekly cycles.

Output
------
:func:`add_time_features` returns a new :class:`pandas.DataFrame` (the input is
not mutated) with the columns above appended, unless they already exist (in
which case they are overwritten deterministically).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

#: Repository root: .../energywise (this file is src/energywise/features/time_features.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default input: the hourly energy + weather table from Phase 2.
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"

#: Reporting / binning timezone (the dataset convention).
IST_TZ = "Asia/Kolkata"

#: Names of the feature columns this module adds.
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

#: Hours in a day / days in a week, used for the cyclical transforms.
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7


# ---------------------------------------------------------------------------
# Indian public holidays
# ---------------------------------------------------------------------------
#
# A deliberately small, fixed list of well-known Indian public holidays for the
# 2013-2017 window covered by the dataset.  It contains:
#
#   * the three gazetted national holidays (fixed dates every year):
#     Republic Day, Independence Day, Gandhi Jayanti;
#   * New Year's Day and Christmas Day;
#   * a handful of major movable festivals (Holi, Good Friday, Dussehra,
#     Diwali) with their commonly-cited dates for each year.
#
# TODO: This list is NOT exhaustive -- review it for completeness (regional
#       holidays, Eid, Raksha Bandhan, Mahavir Jayanti, Guru Nanak Jayanti,
#       Muharram, etc.) before treating ``is_holiday`` as production-grade.
#       Movable festival dates in particular should be verified against an
#       authoritative source (e.g. the Govt. of India holiday calendar).
#
# Dates are IST calendar dates written as ISO ``YYYY-MM-DD`` strings; the set is
# compared against ``timestamp_ist.dt.date`` (see :func:`add_time_features`).
_HOLIDAY_STRINGS: tuple[str, ...] = (
    # --- 2013 ---
    "2013-01-01",  # New Year's Day
    "2013-01-26",  # Republic Day
    "2013-03-27",  # Holi
    "2013-03-29",  # Good Friday
    "2013-08-15",  # Independence Day
    "2013-10-02",  # Gandhi Jayanti
    "2013-10-13",  # Dussehra (Vijayadashami)
    "2013-11-03",  # Diwali (Deepavali)
    "2013-12-25",  # Christmas Day
    # --- 2014 ---
    "2014-01-01",  # New Year's Day
    "2014-01-26",  # Republic Day
    "2014-03-17",  # Holi
    "2014-04-18",  # Good Friday
    "2014-08-15",  # Independence Day
    "2014-10-02",  # Gandhi Jayanti
    "2014-10-03",  # Dussehra (Vijayadashami)
    "2014-10-23",  # Diwali (Deepavali)
    "2014-12-25",  # Christmas Day
    # --- 2015 ---
    "2015-01-01",  # New Year's Day
    "2015-01-26",  # Republic Day
    "2015-03-06",  # Holi
    "2015-04-03",  # Good Friday
    "2015-08-15",  # Independence Day
    "2015-10-02",  # Gandhi Jayanti
    "2015-10-22",  # Dussehra (Vijayadashami)
    "2015-11-11",  # Diwali (Deepavali)
    "2015-12-25",  # Christmas Day
    # --- 2016 ---
    "2016-01-01",  # New Year's Day
    "2016-01-26",  # Republic Day
    "2016-03-24",  # Holi
    "2016-03-25",  # Good Friday
    "2016-08-15",  # Independence Day
    "2016-10-02",  # Gandhi Jayanti
    "2016-10-11",  # Dussehra (Vijayadashami)
    "2016-10-30",  # Diwali (Deepavali)
    "2016-12-25",  # Christmas Day
    # --- 2017 ---
    "2017-01-01",  # New Year's Day
    "2017-01-26",  # Republic Day
    "2017-03-13",  # Holi
    "2017-04-14",  # Good Friday
    "2017-08-15",  # Independence Day
    "2017-10-02",  # Gandhi Jayanti
    "2017-09-30",  # Dussehra (Vijayadashami)
    "2017-10-19",  # Diwali (Deepavali)
    "2017-12-25",  # Christmas Day
)

#: Public holiday dates as :class:`datetime.date` objects for fast membership tests.
INDIAN_HOLIDAYS = frozenset(pd.Timestamp(s).date() for s in _HOLIDAY_STRINGS)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _ensure_ist(df: pd.DataFrame) -> pd.Series:
    """Return ``timestamp_ist`` as a tz-aware IST :class:`~pandas.Series`.

    Accepts either a tz-aware timestamp (converted to IST) or a naive one
    (assumed to already be IST and localised), and raises if the column is
    missing.
    """
    if "timestamp_ist" not in df.columns:
        raise ValueError("Input frame is missing the required 'timestamp_ist' column")

    ts = pd.to_datetime(df["timestamp_ist"])
    if ts.dt.tz is None:
        # Naive timestamps are assumed to already be in the reporting timezone.
        ts = ts.dt.tz_localize(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    return ts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Add leakage-safe calendar and cyclical time features.

    All features are derived exclusively from ``timestamp_ist``; no measured
    column (``energy_kwh``, weather, ...) is read, so the columns are known in
    advance for any future hour and cannot leak the target.

    Parameters
    ----------
    df:
        Hourly frame containing at least a ``timestamp_ist`` column.
    inplace:
        When ``True`` the features are written onto ``df`` and ``df`` is
        returned; otherwise (default) a copy is returned and the input is left
        untouched.

    Returns
    -------
    pandas.DataFrame
        The frame with the columns in :data:`FEATURE_COLUMNS` added/overwritten.
    """
    out = df if inplace else df.copy()
    ts = _ensure_ist(out)

    out["hour"] = ts.dt.hour.astype("int64")                # 0-23
    out["day_of_week"] = ts.dt.dayofweek.astype("int64")    # Mon=0 ... Sun=6
    out["month"] = ts.dt.month.astype("int64")              # 1-12
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(bool)
    out["is_holiday"] = ts.dt.date.isin(INDIAN_HOLIDAYS).astype(bool)

    # Cyclical encodings: hour of day and day of week.
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / HOURS_PER_DAY)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / HOURS_PER_DAY)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["day_of_week"] / DAYS_PER_WEEK)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["day_of_week"] / DAYS_PER_WEEK)

    return out


def build_time_features(
    input_path: Path | str = INPUT_PATH,
    output_path: Path | str | None = None,
    write_parquet: bool = False,
) -> pd.DataFrame:
    """Load the hourly frame, add time features, and optionally persist it.

    Parameters
    ----------
    input_path:
        Hourly energy+weather Parquet file (read-only).
    output_path:
        Destination Parquet path.  Only used when ``write_parquet`` is ``True``.
    write_parquet:
        When ``True`` the enriched frame is written to ``output_path``.

    Returns
    -------
    pandas.DataFrame
        The hourly frame with the :data:`FEATURE_COLUMNS` added.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Hourly frame not found: {input_path}. Build it with the Phase 2 "
            "ingestion steps first."
        )

    hourly = pd.read_parquet(input_path)
    enriched = add_time_features(hourly)

    logger.info(
        "Added time features to %d row(s); %d holiday hour(s) flagged",
        len(enriched),
        int(enriched["is_holiday"].sum()),
    )

    if write_parquet:
        if output_path is None:
            raise ValueError("output_path is required when write_parquet=True")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)
        logger.info("Wrote time-featured frame -> %s", output_path)

    return enriched


def main() -> None:
    """CLI entry point: enrich the default hourly frame and print a preview."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    enriched = build_time_features()
    print(enriched[["timestamp_ist", *FEATURE_COLUMNS]].head().to_string(index=False))


if __name__ == "__main__":
    main()