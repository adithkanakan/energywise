"""Hourly energy cleaning for the EnergyWise building meter data.

Reads the combined raw-layer Parquet file (``data/raw/combined_raw.parquet``,
**never modified**) and writes a cleaned hourly energy table to
``data/processed/hourly_energy.parquet``.

Pipeline (per the build plan):

1. Convert timestamps from UTC to ``Asia/Kolkata`` and keep the original UTC
   timestamp as a separate column.
2. Treat rows with ``NA`` power as missing (never as zero), while keeping
   genuine zero-watt readings as zero.
3. Per building, resample to hourly bins in IST (labelled by hour start) and
   compute ``energy_kwh`` = mean(power in watts over the minutes present in
   that hour) / 1000.  One hour of mean kW equals one kWh.
4. Record ``minutes_present`` for every hour; if ``minutes_present`` is below
   :data:`MIN_MINUTES_PRESENT` (45), set ``energy_kwh`` to NaN and
   ``is_missing`` to ``True``.
5. Emit explicit NaN rows for hours without readings so the hourly index is
   complete, but only between each building's first and last observed
   timestamp (the observed span).
6. Never interpolate or fill anything.

Output columns: ``building``, ``timestamp_ist``, ``energy_kwh``,
``minutes_present``, ``is_missing``.  Voltage, current, frequency and
power_factor are dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

#: Repository root: .../energywise (this file is src/energywise/ingestion/clean_hourly.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "combined_raw.parquet"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy.parquet"

#: Reporting / binning timezone (the dataset convention).
IST_TZ = "Asia/Kolkata"

#: Minimum number of minutes that must be present in an hour for its mean power
#: (and therefore its energy) to be considered valid.  Below this the hour is
#: treated as missing.
MIN_MINUTES_PRESENT = 45

#: Watts-per-kilowatt conversion.
WATTS_PER_KW = 1000.0

#: Columns carried into the output.
OUTPUT_COLUMNS = ["building", "timestamp_ist", "energy_kwh", "minutes_present", "is_missing"]

#: Raw measurement columns that are intentionally dropped.
DROPPED_COLUMNS = ["voltage", "current", "frequency", "power_factor"]


def _configure_logging(level: int = logging.INFO) -> None:
    """Ensure the root logger has a console handler at an INFO level."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    if logger.getEffectiveLevel() > level:
        logger.setLevel(level)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Localise to IST, keep a UTC column, and make ``NA`` power missing.

    * ``timestamp_utc`` is preserved as a separate column.
    * ``timestamp_ist`` is derived from the UTC timestamp (UTC+05:30).
    * Rows with ``NA`` power are dropped so they contribute neither to the
      mean power nor to ``minutes_present``.  Genuine zero-watt readings are
      kept as ``0.0``.
    """
    out = df.copy()

    # (1) UTC kept as a separate column; IST derived for binning/reporting.
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    else:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out["timestamp_ist"] = out["timestamp_utc"].dt.tz_convert(IST_TZ)

    if "power" not in out.columns:
        raise ValueError("Raw frame is missing the required 'power' column")

    # (2) NA power is missing, not zero.  Genuine zeros survive.
    out = out.dropna(subset=["power"])

    return out[["building", "timestamp_utc", "timestamp_ist", "power"]]


def _hourly_for_building(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one building's rows to complete hourly IST bins.

    The returned frame has one row per hour between the building's first and
    last observed timestamp (inclusive), with ``energy_kwh``,
    ``minutes_present`` and ``is_missing`` columns.
    """
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    indexed = df.set_index("timestamp_ist").sort_index()

    # (3) hourly bins labelled by hour start; mean over minutes present.
    grouped = indexed["power"].resample("1h")
    mean_power = grouped.mean()
    minutes_present = grouped.count().astype("int64")

    energy_kwh = mean_power / WATTS_PER_KW

    # (4) hours with too few minutes present are missing -> NaN energy.
    is_missing = minutes_present < MIN_MINUTES_PRESENT
    energy_kwh = energy_kwh.mask(is_missing)

    hourly = pd.DataFrame(
        {
            "energy_kwh": energy_kwh,
            "minutes_present": minutes_present,
            "is_missing": is_missing,
        }
    )

    # (5) complete the hourly index between the first and last observed hour.
    first_hour = indexed.index.min().floor("h")
    last_hour = indexed.index.max().floor("h")
    # ``first_hour`` / ``last_hour`` are already tz-aware (IST); passing a
    # ``tz`` argument alongside tz-aware bounds would raise.
    full_index = pd.date_range(first_hour, last_hour, freq="1h")
    hourly = hourly.reindex(full_index)

    # Reindexed (never-observed) hours: 0 minutes present, missing, NaN energy.
    hourly["minutes_present"] = hourly["minutes_present"].fillna(0).astype("int64")
    hourly["is_missing"] = hourly["is_missing"].fillna(True).astype(bool)
    # energy_kwh stays NaN for the inserted rows (never filled/interpolated).

    hourly = hourly.reset_index().rename(columns={"index": "timestamp_ist"})
    hourly["timestamp_ist"] = hourly["timestamp_ist"].dt.tz_convert(IST_TZ)
    hourly["building"] = df["building"].iloc[0]

    return hourly[OUTPUT_COLUMNS]


def _log_stats(building: str, hourly: pd.DataFrame) -> None:
    """Log row count, missing count/percentage and energy min/max/mean."""
    n_rows = len(hourly)
    n_missing = int(hourly["is_missing"].sum())
    pct_missing = (n_missing / n_rows * 100) if n_rows else 0.0

    valid = hourly.loc[~hourly["is_missing"], "energy_kwh"]
    if len(valid):
        e_min, e_max, e_mean = float(valid.min()), float(valid.max()), float(valid.mean())
    else:
        e_min = e_max = e_mean = float("nan")

    logger.info(
        "%-11s rows=%6d | missing=%6d (%.2f%%) | energy_kwh min=%.4f max=%.4f mean=%.4f",
        building,
        n_rows,
        n_missing,
        pct_missing,
        e_min,
        e_max,
        e_mean,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_hourly(
    raw_path: Path | str = RAW_INPUT_PATH,
    output_path: Path | str = PROCESSED_OUTPUT_PATH,
    write_parquet: bool = True,
) -> pd.DataFrame:
    """Read the raw layer and produce the cleaned hourly energy table.

    Parameters
    ----------
    raw_path:
        Combined raw-layer Parquet file (read-only).
    output_path:
        Destination for the cleaned hourly Parquet file.
    write_parquet:
        When ``True`` (default) the hourly frame is written to ``output_path``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``building``, ``timestamp_ist``, ``energy_kwh``,
        ``minutes_present``, ``is_missing``.
    """
    _configure_logging()

    raw_path = Path(raw_path)
    output_path = Path(output_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw parquet not found: {raw_path}. Run the ingestion step first "
            "(energywise.ingestion.load_raw)."
        )

    raw = pd.read_parquet(raw_path)
    prepared = _prepare(raw)

    frames = []
    for building in sorted(prepared["building"].unique()):
        bdf = prepared[prepared["building"] == building]
        hourly = _hourly_for_building(bdf)
        _log_stats(building, hourly)
        frames.append(hourly)

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )

    logger.info("Cleaned hourly layer: %d rows across %d building(s)", len(result), len(frames))

    if write_parquet:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info("Wrote cleaned hourly energy -> %s", output_path)

    return result


def main() -> None:
    """CLI entry point."""
    _configure_logging()
    clean_hourly()


if __name__ == "__main__":
    main()