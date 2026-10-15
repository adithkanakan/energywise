"""Raw-layer ingestion for the EnergyWise building meter data.

Reads the four building mains CSVs from ``data/raw/energy_dataset/`` and
combines them into a single raw-layer Parquet file at
``data/raw/combined_raw.parquet``.

Each source CSV has the columns::

    timestamp      Unix epoch seconds (UTC)
    power          power in watts (per the dataset README)
    current        current (A)
    voltage        voltage (V)
    frequency      frequency (Hz)
    power_factor   dimensionless

For every file we:

* convert ``timestamp`` into timezone-aware datetime columns (UTC + IST),
* tag each row with a ``building`` name derived from the filename,
* concatenate all four frames into one raw-layer DataFrame.

The original CSVs are never modified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    """Ensure the root logger has a console handler and an INFO level.

    ``logging.basicConfig`` is a no-op if the root logger already has
    handlers, so this is safe to call repeatedly (e.g. every time
    :func:`load_raw` is invoked, including via ``import``).
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    # Make sure our own logger doesn't fall below the desired level, even if
    # the root logger was previously configured at WARNING or higher.
    if logger.getEffectiveLevel() > level:
        logger.setLevel(level)


# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

#: Repository root: .../energywise  (this file is src/energywise/ingestion/load_raw.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "energy_dataset"
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "combined_raw.parquet"

#: Source filename -> human-readable building tag.
BUILDING_FILES: dict[str, str] = {
    "acad_build_mains.csv": "academic",
    "lecture_build_mains.csv": "lecture",
    "library_build_mains.csv": "library",
    "facilities_build_mains.csv": "facilities",
}

#: Local timezone for the IST datetime column.
IST_TZ = "Asia/Kolkata"

#: Expected measurement columns (order as given in the dataset).
MEASUREMENT_COLUMNS = ["power", "current", "voltage", "frequency", "power_factor"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _read_single_file(path: Path, building: str) -> pd.DataFrame:
    """Load one building CSV and return a normalised frame.

    Adds:

    * ``building``       - building tag derived from the filename,
    * ``timestamp_utc``  - timezone-aware UTC datetime,
    * ``timestamp_ist``  - timezone-aware ``Asia/Kolkata`` datetime.

    The original ``timestamp`` (epoch seconds) column is retained.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing building CSV: {path}")

    df = pd.read_csv(path)

    missing = {"timestamp", *MEASUREMENT_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")

    # Timestamp -> timezone-aware datetimes (epoch seconds are UTC).
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert(IST_TZ)

    # Tag every row with its building.
    df.insert(0, "building", building)

    return df


def _log_file_stats(name: str, df: pd.DataFrame) -> None:
    """Log row count, date range and NA fraction of ``current`` for one file."""
    row_count = len(df)
    if row_count:
        start = df["timestamp_utc"].min()
        end = df["timestamp_utc"].max()
        date_range = f"{start} -> {end} (UTC)"
    else:
        date_range = "n/a (empty file)"

    current_na_fraction = float(df["current"].isna().mean()) if row_count else float("nan")

    logger.info(
        "%-28s rows=%d | range=%s | current NA fraction=%.4f",
        name,
        row_count,
        date_range,
        current_na_fraction,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw(
    raw_dir: Path | str = RAW_DATA_DIR,
    output_path: Path | str = RAW_OUTPUT_PATH,
    write_parquet: bool = True,
) -> pd.DataFrame:
    """Load, tag and concatenate the four building CSVs into one raw frame.

    Parameters
    ----------
    raw_dir:
        Directory containing the building CSVs.
    output_path:
        Destination for the combined Parquet file.
    write_parquet:
        When ``True`` (default) the combined frame is written to ``output_path``.

    Returns
    -------
    pandas.DataFrame
        The concatenated raw-layer DataFrame with a reset integer index.
    """
    _configure_logging()

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    frames: list[pd.DataFrame] = []
    for filename, building in BUILDING_FILES.items():
        path = raw_dir / filename
        df = _read_single_file(path, building)
        _log_file_stats(filename, df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Combined raw layer: %d rows across %d buildings (%s)",
        len(combined),
        combined["building"].nunique(),
        ", ".join(sorted(combined["building"].unique())),
    )

    if write_parquet:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        logger.info("Wrote combined raw layer -> %s", output_path)

    return combined


def main() -> None:
    """CLI entry point: configure logging and run the raw ingestion."""
    _configure_logging()
    load_raw()


if __name__ == "__main__":
    main()
