"""Lag and trailing rolling energy features for the EnergyWise hourly table.

Adds, for each building independently, a small set of **leakage-safe**
history features derived from the target ``energy_kwh``:

* ``energy_lag_1h``        -- energy one hour earlier
* ``energy_lag_24h``       -- energy 24 hours (1 day) earlier
* ``energy_lag_168h``      -- energy 168 hours (7 days) earlier
* ``energy_roll_24h_mean`` -- mean over the trailing 24 hours
* ``energy_roll_24h_min``  -- min  over the trailing 24 hours
* ``energy_roll_24h_max``  -- max  over the trailing 24 hours

Correctness rules
-----------------
* **Per building.**  Every feature is computed within a single ``building``;
  one building's history is never mixed into another's.  The first rows of each
  building therefore have NaNs (there is no neighbour to borrow from).
* **Strictly backward.**  The rolling window at hour ``N`` covers hours
  ``N-24 ... N-1`` only -- the current hour and anything in the future are
  excluded, so the features never leak the target.
* **No silent zero-filling.**  A lag is NaN wherever the *source* hour it looks
  back to is NaN (a missing hour is missing, not zero).  A rolling statistic is
  NaN if *any* hour in its window is NaN (``min_periods`` equals the window
  size), so a gap inside the window is never quietly skipped.
* **Warm-up.**  Values are NaN until enough history exists -- e.g. the first
  168 hours of a building's series cannot have a 7-day lag.

The function is a pure transform: :func:`add_lag_features` returns a new frame
(the input is not mutated) with the six columns above appended, preserving the
input's original row order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

#: Repository root: .../energywise (this file is src/energywise/features/lag_features.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default input: the hourly energy + weather table from Phase 2.
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"

#: Reporting / binning timezone (the dataset convention).
IST_TZ = "Asia/Kolkata"

#: Lag offsets, in hours, applied to ``energy_kwh``.
LAG_HOURS: tuple[int, ...] = (1, 24, 168)

#: Size, in hours, of the trailing rolling window.
ROLLING_WINDOW = 24

#: Names of the feature columns this module adds.
LAG_COLUMNS = [
    "energy_lag_1h",
    "energy_lag_24h",
    "energy_lag_168h",
    "energy_roll_24h_mean",
    "energy_roll_24h_min",
    "energy_roll_24h_max",
]

#: Columns the input frame must provide.
REQUIRED_COLUMNS = ("building", "timestamp_ist", "energy_kwh")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _compute_sorted_features(sorted_work: pd.DataFrame) -> pd.DataFrame:
    """Compute lag/rolling features on a frame already sorted per building.

    ``sorted_work`` must be sorted by ``["building", "timestamp_ist"]`` so that
    ``shift`` and ``rolling`` walk each building's series chronologically.  The
    returned frame is indexed like ``sorted_work``.
    """
    grouped = sorted_work.groupby("building", sort=False)["energy_kwh"]

    # The rolling window looks back strictly: shift by one first, then roll.
    prior = grouped.shift(1)
    prior_grouped = prior.groupby(sorted_work["building"], sort=False)
    rolling = prior_grouped.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)

    return pd.DataFrame(
        {
            "energy_lag_1h": grouped.shift(1),
            "energy_lag_24h": grouped.shift(24),
            "energy_lag_168h": grouped.shift(168),
            # reset_index(level=0) drops the grouping key, keeping row alignment.
            "energy_roll_24h_mean": rolling.mean().reset_index(level=0, drop=True),
            "energy_roll_24h_min": rolling.min().reset_index(level=0, drop=True),
            "energy_roll_24h_max": rolling.max().reset_index(level=0, drop=True),
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_lag_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Add per-building lag and trailing rolling energy features.

    Parameters
    ----------
    df:
        Hourly frame containing ``building``, ``timestamp_ist`` and
        ``energy_kwh``.
    inplace:
        When ``True`` the features are written onto ``df`` and ``df`` is
        returned; otherwise (default) a copy is returned and the input is left
        untouched.

    Returns
    -------
    pandas.DataFrame
        The frame with the columns in :data:`LAG_COLUMNS` added, in the input's
        original row order.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing required column(s): {missing}")

    # Work on a positional-indexed copy so we can sort internally for correct
    # chronological shifts, then restore the caller's original row order.
    work = df.reset_index(drop=True)[["building", "timestamp_ist", "energy_kwh"]].copy()
    sort_order = work.sort_values(
        ["building", "timestamp_ist"], kind="mergesort"
    ).index
    sorted_work = work.loc[sort_order]

    features = _compute_sorted_features(sorted_work).sort_index()

    out = df if inplace else df.copy()
    for col in LAG_COLUMNS:
        # Assign positionally to avoid any index-alignment surprises.
        out[col] = features[col].to_numpy()

    return out


def build_lag_features(
    input_path: Path | str = INPUT_PATH,
    output_path: Path | str | None = None,
    write_parquet: bool = False,
) -> pd.DataFrame:
    """Load the hourly frame, add lag features, and optionally persist it.

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
        The hourly frame with the :data:`LAG_COLUMNS` added.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Hourly frame not found: {input_path}. Build it with the Phase 2 "
            "ingestion steps first."
        )

    hourly = pd.read_parquet(input_path)
    enriched = add_lag_features(hourly)

    logger.info(
        "Added lag features to %d row(s) across %d building(s)",
        len(enriched),
        enriched["building"].nunique() if "building" in enriched else 0,
    )

    if write_parquet:
        if output_path is None:
            raise ValueError("output_path is required when write_parquet=True")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)
        logger.info("Wrote lag-featured frame -> %s", output_path)

    return enriched


def main() -> None:
    """CLI entry point: enrich the default hourly frame and print a preview."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    enriched = build_lag_features()
    print(
        enriched[["building", "timestamp_ist", *LAG_COLUMNS]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()